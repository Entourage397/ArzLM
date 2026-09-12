"""Sample remote STEM text and train/evaluate a 32k STEM-mixture BPE."""

from __future__ import annotations

import json
from pathlib import Path

from arzlm.data.catalog import DOMAINS, MIXTURE_TARGET, PES2O_S2ORC_TRAIN_SHARDS
from arzlm.data.network import ByteCounter
from arzlm.data.sources import (
    iter_fineweb_edu,
    iter_finemath_4plus,
    iter_pes2o_s2orc,
    iter_stack_edu_language,
)
from arzlm.paths import TOKENIZER_DIR
from arzlm.tokenizer.compare import (
    evaluate_tokenizer,
    train_stem_candidate,
    weighted_chars_per_token,
    write_comparison,
)
from arzlm.tokenizer.train import VOCAB_SIZE

# Languages used to train the candidate BPE (held-out code eval stays Python).
_CODE_TRAIN_LANGS = ("Python", "C", "Cpp", "Java", "JavaScript", "SQL")


def take_chars(docs, max_chars: int, *, on_progress=None) -> tuple[str, int, int]:
    chunks: list[str] = []
    n = 0
    n_docs = 0
    for doc in docs:
        if not doc.text.strip():
            continue
        chunks.append(doc.text)
        n += len(doc.text)
        n_docs += 1
        if on_progress is not None and (n_docs == 1 or n_docs % 25 == 0):
            on_progress(n, n_docs)
        if n >= max_chars:
            break
    return "\n\n".join(chunks), n, n_docs


def _code_train_stream(counter: ByteCounter):
    for lang in _CODE_TRAIN_LANGS:
        yield from iter_stack_edu_language(lang, counter=counter, fetch_workers=8)


def collect_samples(held_chars: int, train_chars_by_domain: dict[str, int]) -> tuple[dict[str, str], dict[str, str], dict]:
    """Held-out and train text are consecutive prefixes of the same streams (no overlap)."""
    code_counter = ByteCounter("code")
    sci_counter = ByteCounter("science")
    science_shards = list(PES2O_S2ORC_TRAIN_SHARDS[:5])
    streams = {
        "general": iter_fineweb_edu(),
        "math": iter_finemath_4plus(),
        "science": iter_pes2o_s2orc(
            counter=sci_counter,
            shard_indices=science_shards,
            max_docs_per_shard=400,
        ),
        "code": _code_train_stream(code_counter),
    }
    held: dict[str, str] = {}
    train: dict[str, str] = {}
    meta: dict = {"held": {}, "train": {}}
    for domain in DOMAINS:
        h_text, n_h, n_hd = take_chars(streams[domain], held_chars)
        held[domain] = h_text
        meta["held"][domain] = {"chars": n_h, "docs": n_hd}
        print(f"held-out {domain}: {n_h:,} chars, {n_hd} docs", flush=True)
        t_text, n_t, n_td = take_chars(streams[domain], train_chars_by_domain[domain])
        train[domain] = t_text
        meta["train"][domain] = {"chars": n_t, "docs": n_td}
        print(f"train-sample {domain}: {n_t:,} chars, {n_td} docs", flush=True)
    meta["code_network"] = code_counter.snapshot()
    meta["science_network"] = sci_counter.snapshot()
    print("code network", meta["code_network"], "science network", meta["science_network"], flush=True)
    return held, train, meta


def mixed_iterator(train: dict[str, str], train_chars: int):
    """Yield strings approximating 55/20/15/10 until train_chars.

    Domain blobs are sliced to the budget so a 400k/2M/8M sweep is not
    accidentally identical (a 4M general blob would otherwise overflow a 2M mix).
    """
    budgets = {d: int(MIXTURE_TARGET[d] * train_chars) for d in DOMAINS}
    for domain, budget in budgets.items():
        text = train[domain]
        if not text or budget <= 0:
            continue
        acc = 0
        while acc < budget:
            need = budget - acc
            start = acc % len(text)
            piece = text[start : start + need]
            if len(piece) < need:
                piece += text[: need - len(piece)]
            yield piece
            acc += len(piece)


def run_study(
    *,
    current_dir: Path = TOKENIZER_DIR,
    candidate_dir: Path = Path("tokenizer/stem-v1"),
    out_path: Path = Path("docs/data/tokenizer-stem-comparison.json"),
    held_chars: int = 80_000,
    train_sizes: tuple[int, ...] = (400_000, 2_000_000, 8_000_000),
) -> dict:
    unique_train = {d: int(MIXTURE_TARGET[d] * max(train_sizes)) for d in DOMAINS}
    held, train, sample_meta = collect_samples(held_chars, unique_train)
    current = evaluate_tokenizer(current_dir, held)
    payload: dict = {
        "target_vocab": VOCAB_SIZE,
        "current": current,
        "candidates": [],
        "held_chars_per_domain": {k: len(v) for k, v in held.items()},
        "train_chars_per_domain": {k: len(v) for k, v in train.items()},
        "sample_meta": sample_meta,
        "code_train_languages": list(_CODE_TRAIN_LANGS),
    }
    best = None
    for n_chars in train_sizes:
        dest = candidate_dir if n_chars == train_sizes[-1] else Path(str(candidate_dir) + f"-{n_chars}")
        print(f"training STEM BPE on ~{n_chars:,} mixed chars -> {dest}", flush=True)
        train_stem_candidate(mixed_iterator(train, n_chars), dest)
        cand = evaluate_tokenizer(dest, held)
        row = {
            "train_chars_target": n_chars,
            "dir": str(dest),
            "report": cand,
            "weighted_chars_per_token": weighted_chars_per_token(cand, MIXTURE_TARGET),
        }
        payload["candidates"].append(row)
        best = row
    payload["current_weighted_chars_per_token"] = weighted_chars_per_token(current, MIXTURE_TARGET)
    if best is not None:
        payload["candidate_weighted_chars_per_token"] = best["weighted_chars_per_token"]
        # Keep FineWeb tokenizer unless STEM mix compresses the weighted mix
        # without collapsing general English (general chars/token drop < 8%).
        cur_g = current["domains"].get("general", {}).get("chars_per_token")
        cand_g = best["report"]["domains"].get("general", {}).get("chars_per_token")
        keep = True
        reason = "keep_fineweb_tokenizer"
        if best["weighted_chars_per_token"] > payload["current_weighted_chars_per_token"] * 1.02:
            if cur_g and cand_g and cand_g >= cur_g * 0.92:
                keep = False
                reason = "select_stem_tokenizer"
        payload["decision"] = {
            "keep_current": keep,
            "reason": reason,
            "selected_dir": str(current_dir if keep else Path(best["dir"])),
        }
    write_comparison(out_path, payload)
    print(
        json.dumps(
            {k: payload[k] for k in ("current_weighted_chars_per_token", "candidate_weighted_chars_per_token", "decision") if k in payload},
            indent=2,
        )
    )
    return payload


if __name__ == "__main__":
    run_study()
