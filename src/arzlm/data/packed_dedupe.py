"""Drop exact packed-document copies that survived source-hash dedup.

Source dedup is SHA-256 of sanitized *text*. Cosmopedia (and similar) can
emit distinct strings that tokenize to the same uint16 span. Those later
copies never entered the sqlite content table as duplicates, but they are
duplicate *model input*. This rewrite keeps the first complete EOS-delimited
span and drops later exact token-span copies. It does not loosen validation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from arzlm.data.catalog import DOMAINS
from arzlm.data.corpus_validate import audit_packed_split_hashes
from arzlm.data.packed import DTYPE
from arzlm.data.shards import DEFAULT_SHARD_TOKENS, ShardAccumulator, shard_token_sum

MARKER_NAME = "packed-exact-dedupe.json"


def rewrite_split_drop_exact_duplicates(
    corpus_dir: Path,
    files: list[str],
    *,
    eos_id: int,
    domain: str,
    split: str,
    seq_length: int,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    extra_drop: set[bytes] | None = None,
) -> dict[str, Any]:
    """Rewrite one split if stitched packed-document hashes collide.

    `extra_drop` removes those complete-document hashes even when the split
    itself is unique (used to strip train copies of val documents).
    """
    corpus_dir = Path(corpus_dir)
    extra_drop = set(extra_drop or ())
    audit = audit_packed_split_hashes(
        corpus_dir, files, eos_id=eos_id, domain=domain, split=split
    )
    if int(audit["n_collisions"]) == 0 and not extra_drop:
        return {
            "rewritten": False,
            "domain": domain,
            "split": split,
            "n_collisions": 0,
            "n_docs_in": audit["n_docs"],
            "n_unique": audit["n_unique"],
            "dropped_docs": 0,
            "dropped_tokens": 0,
        }

    live_dir = corpus_dir / split / domain
    tmp_dir = corpus_dir / split / f".{domain}.exact-dedupe-tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    acc = ShardAccumulator(
        tmp_dir,
        domain=domain,
        split=split,
        shard_tokens=int(shard_tokens),
        seq_length=int(seq_length),
        prefix="shard",
        start_index=0,
    )
    seen: set[bytes] = set()
    dropped_docs = 0
    dropped_tokens = 0
    kept_docs = 0
    carry = np.zeros((0,), dtype=DTYPE)
    eos_arr = np.array([int(eos_id)], dtype=DTYPE)
    try:
        for rel in files:
            mmap = np.memmap(corpus_dir / rel, dtype=DTYPE, mode="r")
            try:
                shard = np.asarray(mmap, dtype=DTYPE)
                data = np.concatenate([carry, shard]) if carry.size else shard
            finally:
                del mmap
            eos_idx = np.flatnonzero(data == int(eos_id))
            start = 0
            for end in eos_idx.tolist():
                if end > start:
                    chunk = np.asarray(data[start:end], dtype=DTYPE)
                    digest = hashlib.sha256(chunk.tobytes()).digest()
                    n_tokens = int(chunk.size)
                    if digest in seen or digest in extra_drop:
                        dropped_docs += 1
                        dropped_tokens += n_tokens + 1
                    else:
                        seen.add(digest)
                        kept_docs += 1
                        acc.append(np.concatenate([chunk, eos_arr]), n_docs=1)
                start = int(end) + 1
            carry = np.array(data[start:], dtype=DTYPE, copy=True)
        if carry.size:
            acc.append(np.asarray(carry, dtype=DTYPE), n_docs=0)
        acc.flush(final=True)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    records = list(acc.records)
    if not records:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"{domain} {split} exact-dedupe produced no shards")

    bak = live_dir.with_name(f"{live_dir.name}.pre-exact-dedupe")
    if bak.exists():
        shutil.rmtree(bak)
    live_dir.rename(bak)
    try:
        tmp_dir.rename(live_dir)
    except Exception:
        bak.rename(live_dir)
        raise
    shutil.rmtree(bak, ignore_errors=True)

    new_files = [f"{split}/{domain}/{r['file']}" for r in records]
    return {
        "rewritten": True,
        "domain": domain,
        "split": split,
        "n_collisions": int(audit["n_collisions"]),
        "n_docs_in": audit["n_docs"],
        "n_unique": audit["n_unique"],
        "dropped_docs": dropped_docs,
        "dropped_tokens": dropped_tokens,
        "kept_docs": kept_docs,
        "n_tokens_out": shard_token_sum(records),
        "files": new_files,
        "shards": records,
    }


def repair_corpus_exact_duplicates(
    corpus_dir: str | Path,
    *,
    eos_id: int = 2,
    seq_length: int | None = None,
    shard_tokens: int | None = None,
) -> dict[str, Any]:
    """Drop exact packed duplicates in every domain split; refresh meta/manifest."""
    corpus_dir = Path(corpus_dir)
    meta_path = corpus_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seq_length = int(seq_length or meta.get("seq_length") or 2048)
    shard_tokens = int(shard_tokens or meta.get("shard_tokens_target") or DEFAULT_SHARD_TOKENS)
    domains = meta.get("domains") or {}
    splits_out: list[dict[str, Any]] = []
    for domain in DOMAINS:
        rec = domains.get(domain) or {}
        for split in ("train", "val"):
            split_rec = rec.get(split) or {}
            files = list(split_rec.get("files") or [])
            if not files:
                continue
            result = rewrite_split_drop_exact_duplicates(
                corpus_dir,
                files,
                eos_id=int(eos_id),
                domain=domain,
                split=split,
                seq_length=seq_length,
                shard_tokens=shard_tokens,
            )
            splits_out.append(result)
            if not result["rewritten"]:
                continue
            split_rec["files"] = result["files"]
            split_rec["shards"] = result["shards"]
            split_rec["n_tokens"] = result["n_tokens_out"]
            rec[split] = split_rec
            state_path = corpus_dir / "state" / f"{domain}.json"
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state[f"tokens_{split}"] = result["n_tokens_out"]
                state[f"{split}_shards"] = result["shards"]
                state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        train_files = list((rec.get("train") or {}).get("files") or [])
        val_files = list((rec.get("val") or {}).get("files") or [])
        if train_files and val_files:
            train_audit = audit_packed_split_hashes(
                corpus_dir, train_files, eos_id=int(eos_id), domain=domain, split="train"
            )
            val_audit = audit_packed_split_hashes(
                corpus_dir, val_files, eos_id=int(eos_id), domain=domain, split="val"
            )
            leak = set(train_audit["hashes"]).intersection(val_audit["hashes"])
            if leak:
                result = rewrite_split_drop_exact_duplicates(
                    corpus_dir,
                    train_files,
                    eos_id=int(eos_id),
                    domain=domain,
                    split="train",
                    seq_length=seq_length,
                    shard_tokens=shard_tokens,
                    extra_drop=leak,
                )
                result["reason"] = "train_val_leak"
                splits_out.append(result)
                if result["rewritten"]:
                    split_rec = rec.get("train") or {}
                    split_rec["files"] = result["files"]
                    split_rec["shards"] = result["shards"]
                    split_rec["n_tokens"] = result["n_tokens_out"]
                    rec["train"] = split_rec
                    state_path = corpus_dir / "state" / f"{domain}.json"
                    if state_path.is_file():
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                        state["tokens_train"] = result["n_tokens_out"]
                        state["train_shards"] = result["shards"]
                        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        domains[domain] = rec
    train_total = sum(int((domains[d].get("train") or {}).get("n_tokens") or 0) for d in DOMAINS)
    val_total = sum(int((domains[d].get("val") or {}).get("n_tokens") or 0) for d in DOMAINS)
    meta["domains"] = domains
    meta["train_tokens_total"] = train_total
    meta["val_tokens_total"] = val_total
    if train_total:
        meta["mixture_actual"] = {
            d: int((domains[d].get("train") or {}).get("n_tokens") or 0) / train_total for d in DOMAINS
        }
    meta["packed_exact_dedupe"] = {
        "dropped_docs": sum(int(s.get("dropped_docs") or 0) for s in splits_out),
        "dropped_tokens": sum(int(s.get("dropped_tokens") or 0) for s in splits_out),
        "rewritten_splits": [s for s in splits_out if s.get("rewritten")],
    }
    payload = json.dumps(meta, indent=2) + "\n"
    meta_path.write_text(payload, encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(payload, encoding="utf-8")
    report = {
        "train_tokens_total": train_total,
        "val_tokens_total": val_total,
        "mixture_actual": meta.get("mixture_actual"),
        "splits": splits_out,
        "dropped_docs": meta["packed_exact_dedupe"]["dropped_docs"],
        "dropped_tokens": meta["packed_exact_dedupe"]["dropped_tokens"],
    }
    marker = corpus_dir / "state" / MARKER_NAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
