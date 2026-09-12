"""Phase-06 corpus integrity checks. Training must not start if these fail."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from arzlm.data.catalog import (
    DOMAINS,
    TRAIN_TOKEN_QUOTA,
    VAL_TOKEN_QUOTA,
)
from arzlm.data.packed import DTYPE, FORMAT_NAME
from arzlm.data.security.audit import KNOWN_UNSAFE_FINEWEB_PATH
from arzlm.data.security.gate import evaluate_file, load_default_denylist
from arzlm.data.security.policy import RemoteFileMeta
from arzlm.paths import SECURITY_DIR, TOKENIZER_DIR

# Bump when validation semantics change so a blocked phase retries after deploy.
VALIDATION_POLICY_VERSION = 3

# Source-candidate redundancy is a diagnostic: docs_duplicate / docs_inspected
# counts packer-stream exact-hash rejects, not packed-output duplication.
SOURCE_REDUNDANCY_WARN_RATIO = 0.25
SOURCE_REDUNDANCY_PATHOLOGICAL_RATIO = 0.85
# Packed uniqueness hashes EOS-delimited *complete* documents after stitching
# across shard cuts (shards are block-aligned and may split mid-document).
# Collisions shorter than one typical sentence are markdown fences / headings
# (observed: 6-token empty ```python``` in general). Real paper copies are
# hundreds of tokens; fail those.
MIN_PACKED_DOC_TOKENS_FOR_DUP_FAIL = 32

DENIED_SHARD_MARKERS = (
    KNOWN_UNSAFE_FINEWEB_PATH,
    "000_00041.parquet",
    "CC-MAIN-2024-38/000_00041",
)


class CorpusValidationError(RuntimeError):
    """Corpus is not safe to train on."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_text_blobs(root: Path) -> list[str]:
    blobs: list[str] = []
    if not root.is_dir():
        return blobs
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".txt", ".yaml", ".yml"}:
            continue
        try:
            blobs.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return blobs


def _assert_denied_shard_absent(corpus_dir: Path) -> None:
    for blob in _walk_text_blobs(corpus_dir):
        for marker in DENIED_SHARD_MARKERS:
            if marker in blob:
                raise CorpusValidationError(
                    f"denied FineWeb-Edu shard referenced in corpus artifacts: {marker}"
                )
    # Physical filename must never appear under the packed corpus.
    for path in corpus_dir.rglob("*"):
        name = str(path).replace("\\", "/")
        if "000_00041" in name:
            raise CorpusValidationError(f"denied shard path present on disk: {path}")


def _assert_denylist_gate_active() -> dict[str, Any]:
    denylist_path = SECURITY_DIR / "denylist.json"
    if not denylist_path.is_file():
        raise CorpusValidationError(f"missing denylist {denylist_path}")
    denylist = load_default_denylist()
    if not denylist:
        raise CorpusValidationError("denylist is empty")
    from arzlm.data.catalog import FINEWEB_EDU

    meta = RemoteFileMeta(
        dataset_id=FINEWEB_EDU.dataset_id,
        revision=FINEWEB_EDU.revision,
        path=KNOWN_UNSAFE_FINEWEB_PATH,
        sha256="87753428a95294f19aac1f579d145c264ab55484d9fd205a413698821aaf1dc2",
        size=802227208,
        security_status="unsafe",
        av_status="unsafe",
    )
    verdict = evaluate_file(meta, denylist=denylist)
    if verdict.kind != "denied":
        raise CorpusValidationError(f"denylist gate failed for known-unsafe shard: {verdict}")
    return {"denylist": str(denylist_path), "denied_kind": verdict.kind, "entries": len(denylist)}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CorpusValidationError(f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorpusValidationError(f"{path} is not a JSON object")
    return payload


def _check_shard(path: Path, record: dict[str, Any], *, vocab_size: int) -> dict[str, Any]:
    if not path.is_file():
        raise CorpusValidationError(f"manifest points at missing shard {path}")
    n_tokens = int(record.get("n_tokens") or 0)
    size = path.stat().st_size
    expected = n_tokens * int(np.dtype(DTYPE).itemsize)
    if n_tokens <= 0:
        raise CorpusValidationError(f"shard {path} has n_tokens={n_tokens}")
    if size != expected:
        raise CorpusValidationError(f"shard {path} size {size} != {expected} for {n_tokens} uint16 tokens")
    if path.name.endswith(".partial.bin") or bool(record.get("partial")):
        raise CorpusValidationError(f"partial shard treated as final: {path}")
    digest = record.get("sha256")
    if digest:
        actual = _sha256_file(path)
        if actual != digest:
            raise CorpusValidationError(f"shard hash mismatch {path}: {actual} != {digest}")
    sidecar = Path(str(path) + ".json")
    if not sidecar.is_file():
        raise CorpusValidationError(f"missing sidecar {sidecar}")
    mmap = np.memmap(path, dtype=DTYPE, mode="r")
    try:
        sample_n = min(int(mmap.size), 8192)
        if sample_n == 0:
            raise CorpusValidationError(f"unreadable empty mmap {path}")
        sample = np.asarray(mmap[:sample_n])
        max_id = int(sample.max()) if sample.size else -1
        if max_id >= vocab_size:
            raise CorpusValidationError(f"token id {max_id} >= vocab {vocab_size} in {path}")
    finally:
        del mmap
    fmt = record.get("format")
    if fmt not in (None, FORMAT_NAME, "arzlm-packed-v1"):
        raise CorpusValidationError(f"unexpected packed format {fmt} in {path}")
    return {"file": str(path), "n_tokens": n_tokens, "max_sampled_id": max_id}


def hash_complete_spans(data: np.ndarray, *, eos_id: int) -> tuple[list[tuple[bytes, int]], np.ndarray]:
    """Hash EOS-terminated spans; return leftover suffix that has no EOS yet."""
    data = np.asarray(data, dtype=DTYPE)
    eos_idx = np.flatnonzero(data == int(eos_id))
    spans: list[tuple[bytes, int]] = []
    start = 0
    for end in eos_idx.tolist():
        if end > start:
            n = int(end - start)
            spans.append((hashlib.sha256(data[start:end].tobytes()).digest(), n))
        start = int(end) + 1
    leftover = np.array(data[start:], dtype=DTYPE, copy=True)
    return spans, leftover


def iter_packed_document_hashes(path: Path, *, eos_id: int) -> tuple[list[tuple[bytes, int]], int]:
    """Complete EOS-delimited (hash, n_tokens) spans in one shard, plus leftover size.

    Trailing no-EOS bytes are *not* treated as documents. Use
    `audit_packed_split_hashes` to stitch leftovers into the next shard.
    """
    mmap = np.memmap(path, dtype=DTYPE, mode="r")
    try:
        spans, leftover = hash_complete_spans(np.asarray(mmap), eos_id=eos_id)
        return spans, int(leftover.size)
    finally:
        del mmap


def source_redundancy_note(domain: str, inspected: int, dups: int) -> dict[str, Any]:
    """Interpret packer-stream duplicate counters as input redundancy."""
    ratio = (dups / inspected) if inspected else 0.0
    note = {
        "docs_inspected": inspected,
        "docs_duplicate": dups,
        "source_candidate_duplication_ratio": ratio,
        "meaning": (
            "docs_duplicate/docs_inspected is the fraction of source candidates "
            "rejected by exact content/key dedup before packing. Rejected rows "
            "do not enter output shards."
        ),
        "severity": "info",
    }
    if inspected >= 1000 and ratio >= SOURCE_REDUNDANCY_PATHOLOGICAL_RATIO:
        note["severity"] = "warning"
        note["warning"] = (
            f"{domain} source candidate duplication ratio {dups}/{inspected} "
            f"({ratio:.4%}) is pathological; packed-output uniqueness is still required"
        )
    elif inspected >= 1000 and ratio >= SOURCE_REDUNDANCY_WARN_RATIO:
        note["severity"] = "warning"
        note["warning"] = (
            f"{domain} source candidate duplication ratio {dups}/{inspected} "
            f"({ratio:.4%}); deduper rejected these before packing"
        )
    return note


def audit_packed_split_hashes(
    corpus_dir: Path,
    files: list[str],
    *,
    eos_id: int,
    domain: str,
    split: str,
) -> dict[str, Any]:
    """Exact-hash uniqueness of complete packed documents in one domain split.

    Shards are block-aligned and may cut mid-document. Leftover tokens from
    shard N are prepended to shard N+1 before hashing. Incomplete bytes after
    the last shard are alignment padding, not a document.
    """
    seen: dict[bytes, tuple[str, int]] = {}
    collisions = 0
    short_collisions = 0
    n_docs = 0
    n_stitched = 0
    first_collision: str | None = None
    first_short: str | None = None
    carry = np.zeros((0,), dtype=DTYPE)
    for rel in files:
        path = corpus_dir / rel
        mmap = np.memmap(path, dtype=DTYPE, mode="r")
        try:
            shard = np.asarray(mmap, dtype=DTYPE)
            if carry.size:
                data = np.concatenate([carry, shard])
                n_stitched += 1
            else:
                data = shard
        finally:
            del mmap
        spans, carry = hash_complete_spans(data, eos_id=eos_id)
        for digest, n_tokens in spans:
            n_docs += 1
            prev = seen.get(digest)
            if prev is not None:
                collisions += 1
                prev_rel, prev_n = prev
                msg = (
                    f"{domain} {split} packed document hash repeats in {rel} "
                    f"(also {prev_rel}, {n_tokens} tokens)"
                )
                if min(int(n_tokens), int(prev_n)) >= MIN_PACKED_DOC_TOKENS_FOR_DUP_FAIL:
                    if first_collision is None:
                        first_collision = msg
                else:
                    short_collisions += 1
                    if first_short is None:
                        first_short = msg
            else:
                seen[digest] = (rel, int(n_tokens))
    return {
        "n_docs": n_docs,
        "n_unique": len(seen),
        "n_collisions": collisions,
        "n_short_collisions": short_collisions,
        "n_stitched_shards": n_stitched,
        "n_unclosed_tokens": int(carry.size),
        "hashes": seen,
        "first_collision": first_collision,
        "first_short_collision": first_short,
    }


def _validate_tokenizer(tokenizer_dir: Path, *, expected_sha: str | None) -> dict[str, Any]:
    tok_json = tokenizer_dir / "tokenizer.json"
    cfg = tokenizer_dir / "tokenizer_config.json"
    if not tok_json.is_file() or not cfg.is_file():
        raise CorpusValidationError(f"tokenizer incomplete under {tokenizer_dir}")
    digest = _sha256_file(tok_json)
    manifest = tokenizer_dir / "training-manifest.json"
    vocab = None
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        vocab = int(meta.get("vocab_size") or 0)
        man_sha = meta.get("sha256")
        if man_sha and man_sha != digest:
            raise CorpusValidationError("tokenizer.json hash does not match training-manifest.json")
    if expected_sha and expected_sha != digest:
        raise CorpusValidationError("packed corpus tokenizer hash does not match tokenizer.json")
    from litgpt.tokenizer import Tokenizer

    lit = Tokenizer(tokenizer_dir)
    actual_vocab = int(lit.vocab_size)
    if actual_vocab != 32000:
        raise CorpusValidationError(f"tokenizer vocab {actual_vocab} != 32000")
    if vocab not in (None, 32000):
        raise CorpusValidationError(f"training-manifest vocab {vocab} != 32000")
    special = tokenizer_dir / "special-tokens.json"
    special_payload = json.loads(special.read_text(encoding="utf-8")) if special.is_file() else {}
    return {
        "dir": str(tokenizer_dir),
        "sha256": digest,
        "vocab_size": actual_vocab,
        "eos_id": int(lit.eos_id) if lit.eos_id is not None else None,
        "special_tokens": special_payload,
    }


def validate_stem_corpus(
    corpus_dir: str | Path,
    *,
    tokenizer_dir: str | Path | None = None,
    train_quota: dict[str, int] | None = None,
    val_quota: dict[str, int] | None = None,
    vocab_size: int = 32000,
    min_fraction: float = 0.999,
    load_tokenizer: bool = True,
) -> dict[str, Any]:
    """Raise CorpusValidationError unless the packed STEM corpus is train-ready."""
    corpus_dir = Path(corpus_dir)
    train_quota = dict(train_quota or TRAIN_TOKEN_QUOTA)
    val_quota = dict(val_quota or VAL_TOKEN_QUOTA)
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else TOKENIZER_DIR

    if not corpus_dir.is_dir():
        raise CorpusValidationError(f"corpus dir missing: {corpus_dir}")

    _assert_denied_shard_absent(corpus_dir)
    gate = _assert_denylist_gate_active()

    leftover_partials = [
        p
        for p in corpus_dir.rglob("*.partial.bin")
        if "train" in p.parts or "val" in p.parts
    ]
    # state/ partials are allowed only while packing; phase 06 requires complete domains.
    state_partials = list((corpus_dir / "state").glob("*.partial.bin")) if (corpus_dir / "state").is_dir() else []
    if leftover_partials:
        raise CorpusValidationError(f"partial bins under train/val: {leftover_partials}")
    if state_partials:
        raise CorpusValidationError(f"unflushed partial bins remain after packing: {state_partials}")

    meta = _load_json(corpus_dir / "meta.json")
    domains = meta.get("domains") or {}
    train_total = int(meta.get("train_tokens_total") or 0)
    expected_total = sum(int(v) for v in train_quota.values())

    tok_info: dict[str, Any] | None = None
    if load_tokenizer:
        tok_info = _validate_tokenizer(tokenizer_dir, expected_sha=meta.get("tokenizer_sha256"))
    eos_id = 2
    if tok_info and tok_info.get("eos_id") is not None:
        eos_id = int(tok_info["eos_id"])
    elif meta.get("eos_id") is not None:
        eos_id = int(meta["eos_id"])

    domain_reports: dict[str, Any] = {}
    warnings: list[str] = []
    for domain in DOMAINS:
        rec = domains.get(domain)
        if not isinstance(rec, dict):
            # Fall back to per-domain state files written during packing.
            state = _load_json(corpus_dir / "state" / f"{domain}.json")
            if state.get("status") != "complete":
                raise CorpusValidationError(f"{domain} status is {state.get('status')!r}, not complete")
            rec = {
                "train": {
                    "n_tokens": state.get("tokens_train"),
                    "n_documents": state.get("docs_accepted_train"),
                    "shards": state.get("train_shards") or [],
                    "files": [f"train/{domain}/{s['file']}" for s in (state.get("train_shards") or [])],
                },
                "val": {
                    "n_tokens": state.get("tokens_val"),
                    "n_documents": state.get("docs_accepted_val"),
                    "shards": state.get("val_shards") or [],
                    "files": [f"val/{domain}/{s['file']}" for s in (state.get("val_shards") or [])],
                },
                "docs_duplicate": state.get("docs_duplicate"),
                "docs_inspected": state.get("docs_inspected"),
            }
        train = rec.get("train") or {}
        val = rec.get("val") or {}
        train_tokens = int(train.get("n_tokens") or 0)
        val_tokens = int(val.get("n_tokens") or 0)
        if train_tokens < int(train_quota[domain] * min_fraction):
            raise CorpusValidationError(
                f"{domain} train tokens {train_tokens} < quota {train_quota[domain]}"
            )
        if val_tokens < int(val_quota[domain] * 0.99):
            raise CorpusValidationError(
                f"{domain} val tokens {val_tokens} < quota {val_quota[domain]}"
            )
        inspected = int(rec.get("docs_inspected") or 0)
        dups = int(rec.get("docs_duplicate") or 0)
        redundancy = source_redundancy_note(domain, inspected, dups)
        if redundancy.get("warning"):
            warnings.append(str(redundancy["warning"]))

        shard_reports = []
        for split, split_rec in (("train", train), ("val", val)):
            files = list(split_rec.get("files") or [])
            shards = list(split_rec.get("shards") or [])
            if not files or not shards:
                raise CorpusValidationError(f"{domain} {split} has no shards")
            if len(files) != len(shards):
                raise CorpusValidationError(f"{domain} {split} file/shard count mismatch")
            for rel, shard in zip(files, shards, strict=True):
                path = corpus_dir / rel
                shard_reports.append(_check_shard(path, shard, vocab_size=vocab_size))

        train_files = list(train.get("files") or [])
        val_files = list(val.get("files") or [])
        train_audit = audit_packed_split_hashes(
            corpus_dir, train_files, eos_id=eos_id, domain=domain, split="train"
        )
        val_audit = audit_packed_split_hashes(
            corpus_dir, val_files, eos_id=eos_id, domain=domain, split="val"
        )
        if train_audit["first_collision"]:
            raise CorpusValidationError(train_audit["first_collision"])
        if val_audit["first_collision"]:
            raise CorpusValidationError(val_audit["first_collision"])
        if train_audit["n_short_collisions"]:
            warnings.append(
                f"{domain} train packed boilerplate collisions: "
                f"{train_audit['n_short_collisions']} exact token-span repeats "
                f"shorter than {MIN_PACKED_DOC_TOKENS_FOR_DUP_FAIL} tokens"
                + (f" (e.g. {train_audit['first_short_collision']})" if train_audit.get("first_short_collision") else "")
            )
        if val_audit["n_short_collisions"]:
            warnings.append(
                f"{domain} val packed boilerplate collisions: "
                f"{val_audit['n_short_collisions']} exact token-span repeats "
                f"shorter than {MIN_PACKED_DOC_TOKENS_FOR_DUP_FAIL} tokens"
            )
        substantial_leak = 0
        short_leak = 0
        for digest, (_rel, n_tok) in train_audit["hashes"].items():
            other = val_audit["hashes"].get(digest)
            if other is None:
                continue
            _vrel, val_n = other
            if min(int(n_tok), int(val_n)) >= MIN_PACKED_DOC_TOKENS_FOR_DUP_FAIL:
                substantial_leak += 1
            else:
                short_leak += 1
        leak_n = substantial_leak + short_leak
        if leak_n:
            raise CorpusValidationError(
                f"{domain} train/val packed-document hash overlap "
                f"({substantial_leak} documents >={MIN_PACKED_DOC_TOKENS_FOR_DUP_FAIL} tokens, "
                f"{short_leak} shorter). Remove the train copies."
            )
        domain_reports[domain] = {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "n_train_shards": len(train.get("shards") or []),
            "n_val_shards": len(val.get("shards") or []),
            "docs_inspected": inspected,
            "docs_duplicate": dups,
            "source_redundancy": {k: v for k, v in redundancy.items()},
            "packed_train_docs": train_audit["n_docs"],
            "packed_train_unique_docs": train_audit["n_unique"],
            "packed_val_docs": val_audit["n_docs"],
            "packed_val_unique_docs": val_audit["n_unique"],
            "packed_train_short_collisions": train_audit["n_short_collisions"],
            "packed_val_short_collisions": val_audit["n_short_collisions"],
            "packed_train_val_overlap": leak_n,
            "packed_train_val_short_overlap": short_leak,
            "sampled_shards": shard_reports[:4],
        }

    if train_total < int(expected_total * min_fraction):
        raise CorpusValidationError(f"train_tokens_total {train_total} < {expected_total}")

    report = {
        "passed": True,
        "policy_version": VALIDATION_POLICY_VERSION,
        "min_packed_doc_tokens_for_dup_fail": MIN_PACKED_DOC_TOKENS_FOR_DUP_FAIL,
        "corpus_dir": str(corpus_dir),
        "train_tokens_total": train_total,
        "denied_shard": KNOWN_UNSAFE_FINEWEB_PATH,
        "denied_shard_absent": True,
        "gate": gate,
        "tokenizer": tok_info,
        "warnings": warnings,
        "domains": domain_reports,
        "parquet_native_ids": "global-row-index in iter_locked_text_table; covered by tests",
    }
    return report
