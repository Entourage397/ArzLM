"""Build ArzLM-STEM-1B-v1: stream, tokenize once, pack uint16 shards, discard text."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm

from arzlm.data.catalog import (
    CORPUS_NAME,
    CORPUS_NAME_6B,
    DOMAINS,
    MIXTURE_TARGET,
    STEM_FORMAT,
    TRAIN_TOKEN_QUOTA,
    TRAIN_TOKEN_QUOTA_6B,
    VAL_TOKEN_QUOTA,
    VAL_TOKEN_QUOTA_6B,
    code_language_weights_from_paths,
    code_token_quotas,
    MATH_DECISION,
    SOURCES,
)
from arzlm.data.hashing import DedupIndex, content_sha256, source_key
from arzlm.data.network import ByteCounter, dir_nbytes, prune_project_hf_dataset_cache
from arzlm.data.packed import FORMAT_NAME, pack_documents
from arzlm.data.prepare import IsolatedEncoder, _encode_doc
from arzlm.data.textutil import as_token_ids
from arzlm.data.shards import (
    DEFAULT_SHARD_TOKENS,
    ShardAccumulator,
    load_shard_records,
    partial_bin_path,
    read_partial,
    sha256_file,
    shard_token_sum,
    unlink_partial,
    write_partial,
)
from arzlm.data.sources import (
    SourceDoc,
    iter_finemath_4plus,
    iter_general_locked,
    iter_science_locked,
    iter_stack_edu_language,
    synthetic_stem_documents,
)
from arzlm.paths import HF_CACHE_DIR, PREPARED_DIR, TOKENIZER_DIR, ensure_project_dirs
from arzlm.training.report import git_commit, utc_now

MAX_DOC_BYTES = 8 * 1024 * 1024
VAL_BUCKET_MOD = 64  # ~1.56% of docs are val candidates until the val quota fills

STEM_PRESETS: dict[str, dict[str, dict[str, int]]] = {
    "tiny": {
        "train": {"general": 2_048, "math": 1_024, "code": 1_024, "science": 1_024},
        "val": {"general": 256, "math": 256, "code": 256, "science": 256},
    },
    "sanity-10m": {
        "train": {"general": 5_500_000, "math": 2_000_000, "code": 1_500_000, "science": 1_000_000},
        "val": {"general": 64_000, "math": 64_000, "code": 64_000, "science": 64_000},
    },
    "1b": {
        "train": dict(TRAIN_TOKEN_QUOTA),
        "val": dict(VAL_TOKEN_QUOTA),
    },
    "6b": {
        "train": dict(TRAIN_TOKEN_QUOTA_6B),
        "val": dict(VAL_TOKEN_QUOTA_6B),
    },
}


@dataclass
class DomainStats:
    domain: str
    docs_inspected: int = 0
    docs_accepted_train: int = 0
    docs_accepted_val: int = 0
    docs_duplicate: int = 0
    docs_skipped: int = 0
    docs_unencodable: int = 0
    tokens_train: int = 0
    tokens_val: int = 0
    stream_index: int = 0
    started_s: float = 0.0
    elapsed_s: float = 0.0
    network_bytes: int = 0
    lang_train_tokens: dict[str, int] = field(default_factory=dict)
    lang_skip: dict[str, int] = field(default_factory=dict)
    score3_lang_skip: dict[str, int] = field(default_factory=dict)


def _val_bucket(digest: str) -> bool:
    return int(digest[:16], 16) % VAL_BUCKET_MOD == 0


def _tokenizer_sha(tokenizer_dir: Path) -> str:
    path = Path(tokenizer_dir) / "tokenizer.json"
    return sha256_file(path)


def _offer_split(
    digest: str,
    stats: DomainStats,
    train_quota: int,
    val_quota: int,
) -> str | None:
    """Return 'train', 'val', or None if this domain is complete for this doc."""
    train_full = stats.tokens_train >= train_quota
    val_full = stats.tokens_val >= val_quota
    if train_full and val_full:
        return None
    if not val_full and _val_bucket(digest):
        return "val"
    if not train_full:
        return "train"
    if not val_full:
        return None  # keep scanning for val-bucket docs
    return None


def pack_document_stream(
    docs: Iterator[SourceDoc],
    *,
    domain: str,
    out_dir: Path,
    tokenizer_dir: Path,
    train_quota: int,
    val_quota: int,
    seq_length: int,
    encoder: IsolatedEncoder | None,
    tokenizer: Tokenizer,
    dedup: DedupIndex,
    shard_tokens: int,
    resume: dict[str, Any] | None,
    max_docs: int | None = None,
    language_train_quotas: dict[str, int] | None = None,
    checkpoint_path: Path | None = None,
    live: dict[str, Any] | None = None,
    lang_skip: dict[str, int] | None = None,
) -> tuple[DomainStats, list[dict[str, Any]], list[dict[str, Any]]]:
    eos_id = tokenizer.eos_id
    if eos_id is None:
        raise RuntimeError("tokenizer has no EOS")
    stats = DomainStats(domain=domain)
    if resume:
        stats.docs_inspected = int(resume.get("docs_inspected", 0))
        stats.docs_accepted_train = int(resume.get("docs_accepted_train", 0))
        stats.docs_accepted_val = int(resume.get("docs_accepted_val", 0))
        stats.docs_duplicate = int(resume.get("docs_duplicate", 0))
        stats.docs_skipped = int(resume.get("docs_skipped", 0))
        stats.docs_unencodable = int(resume.get("docs_unencodable", 0))
        stats.stream_index = int(resume.get("stream_index", 0))

    train_dir = out_dir / "train" / domain
    val_dir = out_dir / "val" / domain
    state_dir = out_dir / "state"
    disk_train = load_shard_records(train_dir)
    disk_val = load_shard_records(val_dir)
    resume_train = disk_train or list((resume or {}).get("train_shards") or [])
    resume_val = disk_val or list((resume or {}).get("val_shards") or [])
    resume = resume or {}
    if disk_train and resume.get("train_shards"):
        json_train = list(resume.get("train_shards") or [])
        if len(disk_train) >= len(json_train):
            resume_train = disk_train
        elif shard_token_sum(json_train) > shard_token_sum(disk_train):
            resume_train = json_train
    if disk_val and resume.get("val_shards"):
        json_val = list(resume.get("val_shards") or [])
        if len(disk_val) >= len(json_val):
            resume_val = disk_val
        elif shard_token_sum(json_val) > shard_token_sum(disk_val):
            resume_val = json_val
    train_next = max(
        [0]
        + [int((resume or {}).get("train_next_shard", 0))]
        + [int(r.get("creation_order", -1)) + 1 for r in resume_train],
    )
    val_next = max(
        [0]
        + [int((resume or {}).get("val_next_shard", 0))]
        + [int(r.get("creation_order", -1)) + 1 for r in resume_val],
    )
    train_acc = ShardAccumulator(
        train_dir,
        domain=domain,
        split="train",
        shard_tokens=shard_tokens,
        seq_length=seq_length,
        start_index=train_next,
    )
    train_acc.records = list(resume_train)
    val_acc = ShardAccumulator(
        val_dir,
        domain=domain,
        split="val",
        shard_tokens=max(shard_tokens, val_quota + seq_length + 1),
        seq_length=seq_length,
        start_index=val_next,
    )
    val_acc.records = list(resume_val)
    train_partial = partial_bin_path(state_dir, domain, "train")
    val_partial = partial_bin_path(state_dir, domain, "val")
    t_part, t_docs = read_partial(train_partial)
    v_part, v_docs = read_partial(val_partial)
    train_acc.import_partial(t_part, t_docs)
    val_acc.import_partial(v_part, v_docs)
    # Physical tokens only. JSON counters can include an unflushed buffer lost in a crash.
    stats.tokens_train = shard_token_sum(train_acc.records) + int(train_acc.buffered)
    stats.tokens_val = shard_token_sum(val_acc.records) + int(val_acc.buffered)
    lang_train_tokens: dict[str, int] = dict((resume or {}).get("lang_train_tokens") or {})
    if lang_skip is None:
        lang_skip = dict((resume or {}).get("code_lang_skip") or {})
    if live is None:
        live = {}
    if "start_source" not in live:
        live["start_source"] = int((resume or {}).get("start_source") or 0)
    live["tokens_val"] = stats.tokens_val
    live["tokens_train"] = stats.tokens_train
    live["lang_train_tokens"] = lang_train_tokens
    live["lang_skip"] = lang_skip

    t0 = time.perf_counter()
    stats.started_s = t0
    target_bar = train_quota + val_quota
    progress = tqdm(total=target_bar, initial=stats.tokens_train + stats.tokens_val, desc=f"pack:{domain}", unit="tok")
    try:
        for doc in docs:
            stats.stream_index += 1
            stats.docs_inspected += 1
            if max_docs is not None and stats.docs_inspected > max_docs:
                break
            if not isinstance(doc, SourceDoc):
                stats.docs_skipped += 1
                continue
            text = doc.text
            if not isinstance(text, str) or not text.strip():
                stats.docs_skipped += 1
                continue
            if len(text) > MAX_DOC_BYTES:
                stats.docs_skipped += 1
                continue
            digest = content_sha256(text)
            hit = dedup.lookup_content(digest)
            if hit is not None:
                stats.docs_duplicate += 1
                dedup.record_dup(digest, domain, hit[0], "content")
                continue
            keys: list[str] = []
            for raw_key in doc.source_keys:
                if raw_key.startswith("url:"):
                    key = source_key("url", raw_key[4:])
                else:
                    key = raw_key
                if key:
                    keys.append(key)
                    khit = dedup.lookup_key(key)
                    if khit is not None:
                        stats.docs_duplicate += 1
                        dedup.record_dup(digest, domain, khit[0], "key")
                        break
            else:
                keys = keys  # no key dup
                split = _offer_split(digest, stats, train_quota, val_quota)
                if split is None:
                    if stats.tokens_train >= train_quota and stats.tokens_val >= val_quota:
                        break
                    stats.docs_skipped += 1
                    continue
                lang = doc.language
                if (
                    split == "train"
                    and language_train_quotas
                    and lang is not None
                    and lang in language_train_quotas
                    and int(lang_train_tokens.get(lang, 0)) >= language_train_quotas[lang]
                    and not live.get("allow_lang_overflow")
                ):
                    stats.docs_skipped += 1
                    continue
                if encoder is not None:
                    ids = as_token_ids(encoder.encode(text))
                    if ids is None:
                        stats.docs_unencodable += 1
                        continue
                else:
                    ids = as_token_ids(_encode_doc(tokenizer, text))
                    if ids is None:
                        stats.docs_unencodable += 1
                        continue
                if not ids:
                    stats.docs_skipped += 1
                    continue
                packed = pack_documents([ids], eos_id=int(eos_id))
                n = int(packed.size)
                acc = train_acc if split == "train" else val_acc
                acc.append(packed, n_docs=1)
                dedup.add(digest, domain, doc.native_id, split, keys)
                if split == "train":
                    stats.docs_accepted_train += 1
                    stats.tokens_train += n
                    if lang is not None:
                        lang_train_tokens[lang] = int(lang_train_tokens.get(lang, 0)) + n
                else:
                    stats.docs_accepted_val += 1
                    stats.tokens_val += n
                live["tokens_val"] = stats.tokens_val
                live["tokens_train"] = stats.tokens_train
                live["lang_train_tokens"] = lang_train_tokens
                live["lang_skip"] = lang_skip
                progress.update(n)
                if checkpoint_path is not None and (stats.docs_accepted_train + stats.docs_accepted_val) % 32 == 0:
                    t_tok, t_n = train_acc.export_partial()
                    v_tok, v_n = val_acc.export_partial()
                    write_partial(train_partial, t_tok, n_documents=t_n)
                    write_partial(val_partial, v_tok, n_documents=v_n)
                    _write_state(
                        checkpoint_path,
                        _snapshot_pack(
                            stats,
                            train_acc,
                            val_acc,
                            lang_train_tokens=lang_train_tokens,
                            lang_skip=lang_skip,
                            status="in_progress",
                            start_source=int(live.get("start_source") or 0),
                            score3_lang_skip=dict(live.get("score3_lang_skip") or {}),
                        ),
                    )
                if stats.tokens_train >= train_quota and stats.tokens_val >= val_quota:
                    break
    finally:
        progress.close()
        train_acc.flush(final=True)
        val_acc.flush(final=True)
        unlink_partial(train_partial)
        unlink_partial(val_partial)
        stats.tokens_train = shard_token_sum(train_acc.records)
        stats.tokens_val = shard_token_sum(val_acc.records)
        stats.elapsed_s = time.perf_counter() - t0
        stats.lang_train_tokens = dict(lang_train_tokens)
        stats.lang_skip = dict(lang_skip)
        stats.score3_lang_skip = dict(live.get("score3_lang_skip") or {})
        live["tokens_val"] = stats.tokens_val
        live["tokens_train"] = stats.tokens_train
        live["lang_train_tokens"] = lang_train_tokens
        live["lang_skip"] = lang_skip
        dedup.flush()
        if checkpoint_path is not None:
            _write_state(
                checkpoint_path,
                _snapshot_pack(
                    stats,
                    train_acc,
                    val_acc,
                    lang_train_tokens=lang_train_tokens,
                    lang_skip=lang_skip,
                    status="in_progress",
                    start_source=int(live.get("start_source") or 0),
                    score3_lang_skip=dict(live.get("score3_lang_skip") or {}),
                ),
            )

    slack = seq_length + 1
    if stats.tokens_train < train_quota - slack:
        raise RuntimeError(f"{domain}: packed {stats.tokens_train:,} train tokens; needed {train_quota:,}")
    if stats.tokens_val < val_quota - slack:
        raise RuntimeError(f"{domain}: packed {stats.tokens_val:,} val tokens; needed {val_quota:,}")
    return stats, train_acc.records, val_acc.records


def general_resume_cursor(resume: dict[str, Any] | None, *, lock=None) -> tuple[int, int]:
    """Return (start_source, skip) for general-domain resume.

    Combined-stream skip used to re-read FineWeb and then eat Cosmopedia
    documents. If FineWeb unique yield is already packed (high dup ratio),
    jump to lock supplements instead of reconstructing 7M+ SourceDocs.
    """
    if not resume:
        return 0, 0
    start = int(resume.get("start_source") or 0)
    skip = int(resume.get("stream_index") or 0)
    if start >= 1:
        return start, int(resume.get("source_stream_index") or 0)
    inspected = int(resume.get("docs_inspected") or 0)
    dups = int(resume.get("docs_duplicate") or 0)
    tokens = int(resume.get("tokens_train") or 0)
    if tokens > 0 and inspected >= 1_000 and dups / inspected >= 0.8:
        if lock is None:
            try:
                from arzlm.data.security.lockfile import load_source_lock

                lock = load_source_lock()
            except Exception:
                return 0, skip
        if len(lock.sources_for_domain("general")) > 1:
            return 1, 0
    return 0, skip


def _domain_iterator(
    domain: str,
    *,
    source: str,
    skip: int,
    counters: dict[str, ByteCounter],
    start_source: int = 0,
) -> Iterator[SourceDoc]:
    if source == "synthetic":
        return synthetic_stem_documents(domain, n_docs=200_000, seed=42 + DOMAINS.index(domain))
    if domain == "general":
        return iter_general_locked(skip=skip, start_source=start_source)
    if domain == "math":
        return iter_finemath_4plus(skip=skip)
    if domain == "science":
        return iter_science_locked(counter=counters["science"], skip=skip)
    raise ValueError(f"use pack_code_domain for code, not {domain}")


def pack_code_domain(
    *,
    out_dir: Path,
    tokenizer_dir: Path,
    train_quota: int,
    val_quota: int,
    seq_length: int,
    encoder: IsolatedEncoder | None,
    tokenizer: Tokenizer,
    dedup: DedupIndex,
    shard_tokens: int,
    resume: dict[str, Any] | None,
    source: str,
    counters: dict[str, ByteCounter],
    checkpoint_path: Path | None = None,
) -> tuple[DomainStats, list[dict[str, Any]], list[dict[str, Any]]]:
    if source == "synthetic":
        docs = synthetic_stem_documents("code", n_docs=200_000, seed=99)
        return pack_document_stream(
            docs,
            domain="code",
            out_dir=out_dir,
            tokenizer_dir=tokenizer_dir,
            train_quota=train_quota,
            val_quota=val_quota,
            seq_length=seq_length,
            encoder=encoder,
            tokenizer=tokenizer,
            dedup=dedup,
            shard_tokens=shard_tokens,
            resume=resume,
            checkpoint_path=checkpoint_path,
        )
    from arzlm.data.security.lockfile import load_source_lock

    lock = load_source_lock()
    code_src = lock.sources.get("code")
    if code_src is None or not code_src.files:
        raise RuntimeError("source-lock has no Hub-safe Stack-Edu files")
    lang_quotas = code_token_quotas(
        train_quota,
        code_language_weights_from_paths(list(code_src.files)),
    )
    live: dict[str, Any] = {
        "tokens_val": int((resume or {}).get("tokens_val", 0)),
        "tokens_train": int((resume or {}).get("tokens_train", 0)),
        "lang_train_tokens": dict((resume or {}).get("lang_train_tokens") or {}),
        "allow_lang_overflow": False,
    }
    lang_rows: dict[str, int] = dict((resume or {}).get("code_lang_skip") or {})
    score3_rows: dict[str, int] = dict((resume or {}).get("score3_lang_skip") or {})
    live["score3_lang_skip"] = score3_rows
    fetch_workers = int(os.environ.get("ARZLM_SWH_WORKERS", "8"))

    def _emit(
        lang: str,
        q: int,
        *,
        skip_map: dict[str, int],
        min_score: float,
        max_score: float | None,
        stop_at_lang_quota: bool,
    ) -> Iterator[SourceDoc]:
        skip = int(skip_map.get(lang, 0))
        row_counter = [skip]
        for doc in iter_stack_edu_language(
            lang,
            counter=counters["code"],
            skip=skip,
            row_counter=row_counter,
            fetch_workers=fetch_workers,
            min_score=min_score,
            max_score=max_score,
        ):
            skip_map[lang] = int(row_counter[0])
            live["lang_skip"] = lang_rows
            live["score3_lang_skip"] = score3_rows
            yield doc
            if int(live.get("tokens_train") or 0) >= train_quota and int(live.get("tokens_val", 0)) >= val_quota:
                return
            if (
                stop_at_lang_quota
                and int(live["lang_train_tokens"].get(lang, 0)) >= q
                and int(live.get("tokens_val", 0)) >= val_quota
            ):
                return

    def mixed() -> Iterator[SourceDoc]:
        # Pass 1: locked Stack-Edu, permissive, int_score >= 4 (existing shards).
        for lang, q in lang_quotas.items():
            already = int(live["lang_train_tokens"].get(lang, 0))
            if already >= q and int(live.get("tokens_val", 0)) >= val_quota:
                continue
            yield from _emit(
                lang,
                q,
                skip_map=lang_rows,
                min_score=4.0,
                max_score=None,
                stop_at_lang_quota=True,
            )
        # Pass 2: unused official educational band int_score==3 on under-quota langs.
        # Stack-Edu's Hub card filters at threshold 3; the first pass used 4.
        under = [
            (lang, q)
            for lang, q in lang_quotas.items()
            if not (
                int(live["lang_train_tokens"].get(lang, 0)) >= q
                and int(live.get("tokens_val", 0)) >= val_quota
            )
        ]
        under.sort(
            key=lambda item: item[1] - int(live["lang_train_tokens"].get(item[0], 0)),
            reverse=True,
        )
        for lang, q in under:
            yield from _emit(
                lang,
                q,
                skip_map=score3_rows,
                min_score=3.0,
                max_score=4.0,
                stop_at_lang_quota=True,
            )
        # Pass 3: remaining global code quota from the same unused band of capped langs.
        live["allow_lang_overflow"] = True
        for lang, q in lang_quotas.items():
            if int(live.get("tokens_train") or 0) >= train_quota and int(live.get("tokens_val", 0)) >= val_quota:
                return
            if int(live["lang_train_tokens"].get(lang, 0)) < q:
                continue
            yield from _emit(
                lang,
                q,
                skip_map=score3_rows,
                min_score=3.0,
                max_score=4.0,
                stop_at_lang_quota=False,
            )

    return pack_document_stream(
        mixed(),
        domain="code",
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        train_quota=train_quota,
        val_quota=val_quota,
        seq_length=seq_length,
        encoder=encoder,
        tokenizer=tokenizer,
        dedup=dedup,
        shard_tokens=shard_tokens,
        resume=resume,
        language_train_quotas=lang_quotas,
        checkpoint_path=checkpoint_path,
        live=live,
        lang_skip=lang_rows,
    )


def _snapshot_pack(
    stats: DomainStats,
    train_acc: ShardAccumulator,
    val_acc: ShardAccumulator,
    *,
    lang_train_tokens: dict[str, int] | None = None,
    lang_skip: dict[str, int] | None = None,
    status: str = "in_progress",
    start_source: int = 0,
    score3_lang_skip: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "domain": stats.domain,
        "docs_inspected": stats.docs_inspected,
        "docs_accepted_train": stats.docs_accepted_train,
        "docs_accepted_val": stats.docs_accepted_val,
        "docs_duplicate": stats.docs_duplicate,
        "docs_skipped": stats.docs_skipped,
        "docs_unencodable": stats.docs_unencodable,
        "tokens_train": stats.tokens_train,
        "tokens_val": stats.tokens_val,
        "stream_index": stats.stream_index,
        "elapsed_s": stats.elapsed_s,
        "train_shards": list(train_acc.records),
        "val_shards": list(val_acc.records),
        "train_next_shard": train_acc.next_index,
        "val_next_shard": val_acc.next_index,
        "lang_train_tokens": lang_train_tokens or {},
        "code_lang_skip": lang_skip or {},
        "score3_lang_skip": score3_lang_skip or {},
        "start_source": int(start_source),
    }


def _maybe_require_remote_audit(source: str) -> None:
    if source != "remote":
        return
    if os.environ.get("ARZLM_SKIP_PACK_AUDIT") == "1":
        return
    from arzlm.data.security.audit import require_remote_security_audit

    require_remote_security_audit()


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    from arzlm.cloud.runtime import maybe_commit_pack

    maybe_commit_pack({"path": str(path), "status": payload.get("status")})


def pack_one_stem_domain(
    domain: str,
    *,
    preset: str,
    source: str,
    out_dir: Path,
    tokenizer_dir: Path,
    seq_length: int,
    shard_tokens: int,
    encoder: IsolatedEncoder | None,
    tokenizer: Tokenizer,
    dedup: DedupIndex,
    counters: dict[str, ByteCounter],
) -> dict[str, Any]:
    """Pack a single STEM domain. Safe to call from parallel Modal workers
    when each worker uses its own DedupIndex sqlite path."""
    spec = STEM_PRESETS[preset]
    state_path = out_dir / "state" / f"{domain}.json"
    resume = None
    if state_path.is_file():
        resume = json.loads(state_path.read_text(encoding="utf-8"))
        if resume.get("status") == "complete":
            return resume
    train_q = spec["train"][domain]
    val_q = spec["val"][domain]
    cache_domain0 = dir_nbytes(HF_CACHE_DIR)
    live: dict[str, Any] | None = None
    if domain == "code":
        stats, train_shards, val_shards = pack_code_domain(
            out_dir=out_dir,
            tokenizer_dir=tokenizer_dir,
            train_quota=train_q,
            val_quota=val_q,
            seq_length=seq_length,
            encoder=encoder,
            tokenizer=tokenizer,
            dedup=dedup,
            shard_tokens=shard_tokens,
            resume=resume,
            source=source,
            counters=counters,
            checkpoint_path=state_path,
        )
    else:
        skip = 0 if resume is None else int(resume.get("stream_index", 0))
        start_source = 0
        if domain == "general" and source == "remote":
            start_source, skip = general_resume_cursor(resume)
            live = {"start_source": start_source}
        docs = _domain_iterator(
            domain,
            source=source,
            skip=skip,
            counters=counters,
            start_source=start_source,
        )
        enc = encoder
        stats, train_shards, val_shards = pack_document_stream(
            docs,
            domain=domain,
            out_dir=out_dir,
            tokenizer_dir=tokenizer_dir,
            train_quota=train_q,
            val_quota=val_q,
            seq_length=seq_length,
            encoder=enc,
            tokenizer=tokenizer,
            dedup=dedup,
            shard_tokens=shard_tokens,
            resume=resume,
            checkpoint_path=state_path,
            live=live,
        )
    snap = counters[domain].snapshot()
    cache_delta = max(0, dir_nbytes(HF_CACHE_DIR) - cache_domain0)
    stats.network_bytes = int(snap["bytes"]) + cache_delta
    payload = {
        "status": "complete",
        "domain": domain,
        "docs_inspected": stats.docs_inspected,
        "docs_accepted_train": stats.docs_accepted_train,
        "docs_accepted_val": stats.docs_accepted_val,
        "docs_duplicate": stats.docs_duplicate,
        "docs_skipped": stats.docs_skipped,
        "docs_unencodable": stats.docs_unencodable,
        "tokens_train": stats.tokens_train,
        "tokens_val": stats.tokens_val,
        "stream_index": stats.stream_index,
        "elapsed_s": stats.elapsed_s,
        "network_bytes": stats.network_bytes,
        "network_requests": snap["requests"],
        "hf_cache_delta_bytes": cache_delta,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "train_next_shard": len(train_shards),
        "val_next_shard": len(val_shards),
        "lang_train_tokens": dict(stats.lang_train_tokens),
        "code_lang_skip": dict(stats.lang_skip),
        "score3_lang_skip": dict(stats.score3_lang_skip),
        "start_source": int((live or {}).get("start_source") or 0),
    }
    _write_state(state_path, payload)
    return payload


def pack_stem_domain_standalone(
    domain: str,
    *,
    preset: str,
    source: str = "remote",
    out_dir: str | Path,
    tokenizer_dir: str | Path,
    seq_length: int | None = None,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    isolated_encode: bool = True,
) -> dict[str, Any]:
    """Worker entry: own encoder + per-domain sqlite so Modal can pack in parallel."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}")
    _maybe_require_remote_audit(source)
    out_dir = Path(out_dir)
    tokenizer_dir = Path(tokenizer_dir)
    if seq_length is None:
        seq_length = 2048 if preset == "6b" else 1024
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    encoder = IsolatedEncoder(tokenizer_dir) if isolated_encode else None
    tokenizer = Tokenizer(tokenizer_dir)
    dedup = DedupIndex(out_dir / "state" / f"dedup-{domain}.sqlite")
    counters = {d: ByteCounter(d) for d in DOMAINS}
    try:
        return pack_one_stem_domain(
            domain,
            preset=preset,
            source=source,
            out_dir=out_dir,
            tokenizer_dir=tokenizer_dir,
            seq_length=seq_length,
            shard_tokens=shard_tokens,
            encoder=encoder,
            tokenizer=tokenizer,
            dedup=dedup,
            counters=counters,
        )
    finally:
        if encoder is not None:
            encoder.close()
        dedup.flush()
        dedup.close()


def build_stem_corpus(
    preset: str = "1b",
    *,
    source: str = "remote",
    out_dir: str | Path | None = None,
    tokenizer_dir: str | Path | None = None,
    seed: int = 42,
    seq_length: int | None = None,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    isolated_encode: bool | None = None,
) -> Path:
    if preset not in STEM_PRESETS:
        raise ValueError(f"unknown STEM preset {preset!r}")
    if source not in {"remote", "synthetic"}:
        raise ValueError("source must be remote|synthetic")
    _maybe_require_remote_audit(source)
    ensure_project_dirs()
    spec = STEM_PRESETS[preset]
    if seq_length is None:
        seq_length = 2048 if preset == "6b" else 1024
    default_name = "arzlm-stem-6b-v1" if preset == "6b" else "arzlm-stem-1b-v1"
    out_dir = Path(out_dir) if out_dir else PREPARED_DIR / default_name
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else TOKENIZER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (tokenizer_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(f"tokenizer not found at {tokenizer_dir}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os_isolated = True if isolated_encode is None else isolated_encode
    encoder = IsolatedEncoder(tokenizer_dir) if os_isolated else None
    tokenizer = Tokenizer(tokenizer_dir)
    dedup = DedupIndex(out_dir / "state" / "dedup.sqlite")
    from arzlm.data.security.provenance import UpstreamLedger, reset_ledger, set_ledger

    ledger = UpstreamLedger(out_dir / "state" / "upstream-files.jsonl") if source == "remote" else None
    ledger_token = set_ledger(ledger) if ledger is not None else None
    counters = {d: ByteCounter(d) for d in DOMAINS}
    cache_before = dir_nbytes(HF_CACHE_DIR)
    t_all = time.perf_counter()
    domain_meta: dict[str, Any] = {}
    try:
        for domain in DOMAINS:
            domain_meta[domain] = pack_one_stem_domain(
                domain,
                preset=preset,
                source=source,
                out_dir=out_dir,
                tokenizer_dir=tokenizer_dir,
                seq_length=seq_length,
                shard_tokens=shard_tokens,
                encoder=encoder,
                tokenizer=tokenizer,
                dedup=dedup,
                counters=counters,
            )
    finally:
        if encoder is not None:
            encoder.close()
        dedup.flush()
        if ledger_token is not None:
            reset_ledger(ledger_token)

    dups = dedup.dup_counts()
    dedup.close()
    return finalize_stem_corpus(
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        tokenizer=tokenizer,
        preset=preset,
        source=source,
        seed=seed,
        seq_length=seq_length,
        shard_tokens=shard_tokens,
        domain_meta=domain_meta,
        dups=dups,
        cache_before=cache_before,
        t_all=t_all,
        ledger=ledger,
    )


def finalize_stem_corpus(
    *,
    out_dir: Path,
    tokenizer_dir: Path,
    tokenizer: Tokenizer,
    preset: str,
    source: str,
    seed: int,
    seq_length: int,
    shard_tokens: int,
    domain_meta: dict[str, Any] | None = None,
    dups: dict[str, Any] | None = None,
    cache_before: int = 0,
    t_all: float | None = None,
    ledger: Any = None,
) -> Path:
    out_dir = Path(out_dir)
    if domain_meta is None:
        domain_meta = {}
        for domain in DOMAINS:
            state_path = out_dir / "state" / f"{domain}.json"
            if not state_path.is_file():
                raise FileNotFoundError(f"missing domain state {state_path}")
            rec = json.loads(state_path.read_text(encoding="utf-8"))
            if rec.get("status") != "complete":
                raise RuntimeError(f"{domain} packing is not complete")
            domain_meta[domain] = rec
    if t_all is None:
        t_all = time.perf_counter()
    cache_after = dir_nbytes(HF_CACHE_DIR)
    packed_bytes = dir_nbytes(out_dir)
    peak_tmp = max(cache_before, cache_after, packed_bytes, cache_after + packed_bytes)
    pruned = {"bytes_removed": 0, "files_removed": 0}
    if source == "remote":
        prune_ids = [SOURCES[d].dataset_id for d in DOMAINS]
        try:
            from arzlm.data.security.lockfile import load_source_lock

            lock = load_source_lock()
            prune_ids.extend(src.dataset_id for src in lock.all_locked_sources())
        except Exception:
            pass
        pruned = prune_project_hf_dataset_cache(HF_CACHE_DIR, prune_ids)
    cache_after_prune = dir_nbytes(HF_CACHE_DIR)
    train_total = sum(int(domain_meta[d]["tokens_train"]) for d in DOMAINS)
    val_total = sum(int(domain_meta[d]["tokens_val"]) for d in DOMAINS)
    actual_mix = {d: domain_meta[d]["tokens_train"] / train_total for d in DOMAINS}
    locked_by_domain: dict[str, list[dict[str, Any]]] = {}
    if source == "remote":
        try:
            from arzlm.data.security.lockfile import load_source_lock

            lock = load_source_lock()
            for d in DOMAINS:
                locked_by_domain[d] = [
                    {
                        "dataset_id": src.dataset_id,
                        "revision": src.revision,
                        "n_files": len(src.files),
                        "prefix": src.prefix,
                    }
                    for src in lock.sources_for_domain(d)
                ]
        except Exception:
            locked_by_domain = {}

    def _rel(files: list[dict[str, Any]], domain: str, split: str) -> list[str]:
        return [f"{split}/{domain}/{r['file']}" for r in files]

    domains_out = {}
    for d in DOMAINS:
        rec = domain_meta[d]
        domains_out[d] = {
            "train": {
                "files": _rel(rec["train_shards"], d, "train"),
                "n_tokens": rec["tokens_train"],
                "n_documents": rec["docs_accepted_train"],
                "shards": rec["train_shards"],
            },
            "val": {
                "files": _rel(rec["val_shards"], d, "val"),
                "n_tokens": rec["tokens_val"],
                "n_documents": rec["docs_accepted_val"],
                "shards": rec["val_shards"],
            },
            "source": {
                "dataset_id": "synthetic" if source == "synthetic" else SOURCES[d].dataset_id,
                "config": None if source == "synthetic" else SOURCES[d].config,
                "revision": None if source == "synthetic" else SOURCES[d].revision,
                "license": "synthetic" if source == "synthetic" else SOURCES[d].license,
                "locked": locked_by_domain.get(d) or None,
            },
            "docs_inspected": rec["docs_inspected"],
            "docs_duplicate": rec["docs_duplicate"],
            "docs_skipped": rec["docs_skipped"],
            "docs_unencodable": rec["docs_unencodable"],
            "elapsed_s": rec["elapsed_s"],
            "network_bytes": rec["network_bytes"],
        }

    corpus_name = CORPUS_NAME_6B if preset == "6b" else CORPUS_NAME
    manifest = {
        "corpus": corpus_name,
        "format": STEM_FORMAT,
        "packed_format": FORMAT_NAME,
        "preset": preset,
        "source_mode": source,
        "git_commit": git_commit(),
        "created_at": utc_now(),
        "seed": seed,
        "seq_length": seq_length,
        "dtype": "uint16",
        "endian": "little",
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "tokenizer_sha256": _tokenizer_sha(tokenizer_dir),
        "eos_id": int(tokenizer.eos_id),
        "mixture_target": dict(MIXTURE_TARGET),
        "mixture_actual": actual_mix,
        "math_decision": MATH_DECISION,
        "train_tokens_total": train_total,
        "val_tokens_total": val_total,
        "domains": domains_out,
        "duplicate_counts": dups or {},
        "hf_cache_bytes_before": cache_before,
        "hf_cache_bytes_after": cache_after,
        "hf_cache_delta_bytes": cache_after - cache_before,
        "hf_cache_bytes_after_prune": cache_after_prune,
        "hf_cache_pruned": pruned,
        "packed_bytes": packed_bytes,
        "peak_temporary_storage_bytes": peak_tmp,
        "elapsed_s": time.perf_counter() - t_all,
        "shard_tokens_target": shard_tokens,
        "upstream_files": list(ledger.records) if ledger is not None else [],
    }
    (out_dir / "meta.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"corpus": corpus_name, "train_tokens_total": train_total, "mixture_actual": actual_mix}, indent=2))
    return out_dir
