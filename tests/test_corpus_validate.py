"""Phase-06 corpus validation and IsolatedEncoder wiring for code packing."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from arzlm.data.catalog import DOMAINS
from arzlm.data.corpus_validate import CorpusValidationError, validate_stem_corpus
from arzlm.data.packed import DTYPE
from arzlm.data.security.audit import KNOWN_UNSAFE_FINEWEB_PATH
from arzlm.data.shards import write_shard_atomic
from arzlm.data.stem_builder import pack_code_domain


def _shard(dir_path: Path, name: str, n: int, token: int = 7, eos_id: int = 2) -> dict:
    tokens = np.full(n, token, dtype=DTYPE)
    if n > 0:
        tokens[-1] = eos_id
    return write_shard_atomic(
        dir_path / name,
        tokens,
        meta={"creation_order": 0, "n_documents": 1},
    )


def _mini_corpus(tmp_path: Path, *, extra_meta: dict | None = None) -> Path:
    corpus = tmp_path / "corpus"
    domains_out = {}
    train_total = 0
    for i, domain in enumerate(DOMAINS):
        train_dir = corpus / "train" / domain
        val_dir = corpus / "val" / domain
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        train_n = 1000 + i
        val_n = 200 + i
        train_rec = _shard(train_dir, "shard-00000.bin", train_n)
        val_rec = _shard(val_dir, "shard-00000.bin", val_n)
        domains_out[domain] = {
            "train": {
                "files": [f"train/{domain}/shard-00000.bin"],
                "n_tokens": train_n,
                "n_documents": 1,
                "shards": [train_rec],
            },
            "val": {
                "files": [f"val/{domain}/shard-00000.bin"],
                "n_tokens": val_n,
                "n_documents": 1,
                "shards": [val_rec],
            },
            "docs_inspected": 10,
            "docs_duplicate": 0,
        }
        train_total += train_n
        (corpus / "state").mkdir(exist_ok=True)
        (corpus / "state" / f"{domain}.json").write_text(
            json.dumps({"status": "complete", "domain": domain, "tokens_train": train_n, "tokens_val": val_n})
            + "\n",
            encoding="utf-8",
        )
    meta = {
        "train_tokens_total": train_total,
        "domains": domains_out,
        "tokenizer_sha256": None,
    }
    if extra_meta:
        meta.update(extra_meta)
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return corpus


def test_validate_mini_corpus_passes(tmp_path: Path) -> None:
    corpus = _mini_corpus(tmp_path)
    quotas = {d: 1000 + i for i, d in enumerate(DOMAINS)}
    val_q = {d: 200 + i for i, d in enumerate(DOMAINS)}
    report = validate_stem_corpus(
        corpus,
        train_quota=quotas,
        val_quota=val_q,
        load_tokenizer=False,
    )
    assert report["passed"] is True
    assert report["denied_shard_absent"] is True
    assert report["denied_shard"] == KNOWN_UNSAFE_FINEWEB_PATH


def test_validate_rejects_denied_shard_reference(tmp_path: Path) -> None:
    corpus = _mini_corpus(tmp_path)
    (corpus / "state" / "upstream-files.jsonl").write_text(
        json.dumps({"path": KNOWN_UNSAFE_FINEWEB_PATH}) + "\n",
        encoding="utf-8",
    )
    quotas = {d: 1000 + i for i, d in enumerate(DOMAINS)}
    val_q = {d: 200 + i for i, d in enumerate(DOMAINS)}
    with pytest.raises(CorpusValidationError, match="denied FineWeb-Edu"):
        validate_stem_corpus(corpus, train_quota=quotas, val_quota=val_q, load_tokenizer=False)


def test_validate_rejects_partial_as_final(tmp_path: Path) -> None:
    corpus = _mini_corpus(tmp_path)
    (corpus / "train" / "code" / "shard.partial.bin").write_bytes(b"\x00\x00")
    quotas = {d: 1000 + i for i, d in enumerate(DOMAINS)}
    val_q = {d: 200 + i for i, d in enumerate(DOMAINS)}
    with pytest.raises(CorpusValidationError, match="partial"):
        validate_stem_corpus(corpus, train_quota=quotas, val_quota=val_q, load_tokenizer=False)


def test_validate_rejects_underfilled_domain(tmp_path: Path) -> None:
    corpus = _mini_corpus(tmp_path)
    quotas = {d: 10_000 for d in DOMAINS}
    val_q = {d: 1 for d in DOMAINS}
    with pytest.raises(CorpusValidationError, match="train tokens"):
        validate_stem_corpus(corpus, train_quota=quotas, val_quota=val_q, load_tokenizer=False)


def test_validate_rejects_token_id_outside_vocab(tmp_path: Path) -> None:
    corpus = _mini_corpus(tmp_path)
    bad = np.array([32000], dtype=DTYPE)
    rec = write_shard_atomic(
        corpus / "train" / "math" / "shard-00000.bin",
        bad,
        meta={"creation_order": 0, "n_documents": 1},
    )
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    meta["domains"]["math"]["train"]["n_tokens"] = int(rec["n_tokens"])
    meta["domains"]["math"]["train"]["shards"] = [rec]
    meta["domains"]["math"]["train"]["files"] = ["train/math/shard-00000.bin"]
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    quotas = {d: 1 for d in DOMAINS}
    val_q = {d: 1 for d in DOMAINS}
    with pytest.raises(CorpusValidationError, match="token id"):
        validate_stem_corpus(
            corpus,
            train_quota=quotas,
            val_quota=val_q,
            vocab_size=32000,
            min_fraction=0.0,
            load_tokenizer=False,
        )


def test_pack_code_domain_passes_isolated_encoder() -> None:
    src = inspect.getsource(pack_code_domain)
    assert "encoder=encoder" in src
    assert "encoder=None" not in src


def test_pack_code_domain_resumes_score3_without_resetting_lang_skip() -> None:
    src = inspect.getsource(pack_code_domain)
    assert 'resume or {}).get("code_lang_skip")' in src
    assert 'resume or {}).get("score3_lang_skip")' in src
    assert "min_score=4.0" in src
    assert "max_score=4.0" in src


def test_source_redundancy_is_warning_not_failure(tmp_path: Path) -> None:
    from arzlm.data.corpus_validate import source_redundancy_note

    note = source_redundancy_note("science", 3_154_665, 867_237)
    assert note["severity"] == "warning"
    assert abs(note["source_candidate_duplication_ratio"] - 867237 / 3154665) < 1e-12
    corpus = _mini_corpus(tmp_path)
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    meta["domains"]["science"]["docs_inspected"] = 3_154_665
    meta["domains"]["science"]["docs_duplicate"] = 867_237
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    quotas = {d: 1000 + i for i, d in enumerate(DOMAINS)}
    val_q = {d: 200 + i for i, d in enumerate(DOMAINS)}
    report = validate_stem_corpus(corpus, train_quota=quotas, val_quota=val_q, load_tokenizer=False)
    assert report["passed"] is True
    from arzlm.data.corpus_validate import VALIDATION_POLICY_VERSION

    assert report["policy_version"] == VALIDATION_POLICY_VERSION
    assert any("source candidate duplication ratio" in w for w in report["warnings"])
    assert report["domains"]["science"]["packed_train_unique_docs"] >= 1


def test_validate_rejects_duplicate_packed_documents(tmp_path: Path) -> None:
    from arzlm.data.packed import pack_documents

    corpus = _mini_corpus(tmp_path)
    long_doc = list(range(10, 50))
    packed = pack_documents([long_doc, long_doc], eos_id=2)
    rec = write_shard_atomic(
        corpus / "train" / "science" / "shard-00000.bin",
        packed,
        meta={"creation_order": 0, "n_documents": 2},
    )
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    meta["domains"]["science"]["train"]["n_tokens"] = int(rec["n_tokens"])
    meta["domains"]["science"]["train"]["shards"] = [rec]
    meta["train_tokens_total"] = sum(int(meta["domains"][d]["train"]["n_tokens"]) for d in DOMAINS)
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    quotas = {d: 1 for d in DOMAINS}
    val_q = {d: 1 for d in DOMAINS}
    with pytest.raises(CorpusValidationError, match="packed document hash repeats"):
        validate_stem_corpus(
            corpus,
            train_quota=quotas,
            val_quota=val_q,
            min_fraction=0.0,
            load_tokenizer=False,
        )


def test_short_boilerplate_packed_collision_warns_but_passes(tmp_path: Path) -> None:
    from arzlm.data.packed import pack_documents

    corpus = _mini_corpus(tmp_path)
    packed = pack_documents([[7, 8, 9], [7, 8, 9]], eos_id=2)
    rec = write_shard_atomic(
        corpus / "train" / "science" / "shard-00000.bin",
        packed,
        meta={"creation_order": 0, "n_documents": 2},
    )
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    meta["domains"]["science"]["train"]["n_tokens"] = int(rec["n_tokens"])
    meta["domains"]["science"]["train"]["shards"] = [rec]
    meta["train_tokens_total"] = sum(int(meta["domains"][d]["train"]["n_tokens"]) for d in DOMAINS)
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    quotas = {d: 1 for d in DOMAINS}
    val_q = {d: 1 for d in DOMAINS}
    report = validate_stem_corpus(
        corpus,
        train_quota=quotas,
        val_quota=val_q,
        min_fraction=0.0,
        load_tokenizer=False,
    )
    assert report["passed"] is True
    assert report["domains"]["science"]["packed_train_short_collisions"] >= 1
    assert any("boilerplate collisions" in w for w in report["warnings"])


def test_cross_shard_split_is_not_a_packed_collision(tmp_path: Path) -> None:
    from arzlm.data.corpus_validate import audit_packed_split_hashes
    from arzlm.data.packed import pack_documents

    corpus = tmp_path / "c"
    train = corpus / "train" / "general"
    train.mkdir(parents=True)
    doc = list(range(10, 90))
    packed = pack_documents([doc], eos_id=2)
    assert packed.size > 40
    write_shard_atomic(train / "shard-00000.bin", packed[:40], meta={"creation_order": 0})
    write_shard_atomic(train / "shard-00001.bin", packed[40:], meta={"creation_order": 1})
    audit = audit_packed_split_hashes(
        corpus,
        ["train/general/shard-00000.bin", "train/general/shard-00001.bin"],
        eos_id=2,
        domain="general",
        split="train",
    )
    assert audit["n_docs"] == 1
    assert audit["n_collisions"] == 0
    assert audit["n_stitched_shards"] == 1
    assert audit["first_collision"] is None


def test_stitched_duplicate_long_document_still_fails(tmp_path: Path) -> None:
    from arzlm.data.packed import pack_documents

    corpus = _mini_corpus(tmp_path)
    long_doc = list(range(10, 90))
    packed = pack_documents([long_doc], eos_id=2)
    train_dir = corpus / "train" / "science"
    rec0 = write_shard_atomic(train_dir / "shard-00000.bin", packed[:30], meta={"creation_order": 0})
    rec1 = write_shard_atomic(train_dir / "shard-00001.bin", packed[30:], meta={"creation_order": 1})
    rec2 = write_shard_atomic(train_dir / "shard-00002.bin", packed, meta={"creation_order": 2})
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    n_tokens = int(rec0["n_tokens"]) + int(rec1["n_tokens"]) + int(rec2["n_tokens"])
    meta["domains"]["science"]["train"]["n_tokens"] = n_tokens
    meta["domains"]["science"]["train"]["shards"] = [rec0, rec1, rec2]
    meta["domains"]["science"]["train"]["files"] = [
        "train/science/shard-00000.bin",
        "train/science/shard-00001.bin",
        "train/science/shard-00002.bin",
    ]
    meta["train_tokens_total"] = sum(int(meta["domains"][d]["train"]["n_tokens"]) for d in DOMAINS)
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    quotas = {d: 1 for d in DOMAINS}
    val_q = {d: 1 for d in DOMAINS}
    with pytest.raises(CorpusValidationError, match="packed document hash repeats"):
        validate_stem_corpus(
            corpus,
            train_quota=quotas,
            val_quota=val_q,
            min_fraction=0.0,
            load_tokenizer=False,
        )


def test_validate_rejects_train_val_packed_leak(tmp_path: Path) -> None:
    from arzlm.data.packed import pack_documents

    corpus = _mini_corpus(tmp_path)
    long_doc = list(range(10, 50))
    packed = pack_documents([long_doc], eos_id=2)
    train_rec = write_shard_atomic(
        corpus / "train" / "science" / "shard-00000.bin",
        packed,
        meta={"creation_order": 0, "n_documents": 1},
    )
    val_rec = write_shard_atomic(
        corpus / "val" / "science" / "shard-00000.bin",
        packed,
        meta={"creation_order": 0, "n_documents": 1},
    )
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    meta["domains"]["science"]["train"]["n_tokens"] = int(train_rec["n_tokens"])
    meta["domains"]["science"]["train"]["shards"] = [train_rec]
    meta["domains"]["science"]["val"]["n_tokens"] = int(val_rec["n_tokens"])
    meta["domains"]["science"]["val"]["shards"] = [val_rec]
    meta["train_tokens_total"] = sum(int(meta["domains"][d]["train"]["n_tokens"]) for d in DOMAINS)
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    quotas = {d: 1 for d in DOMAINS}
    val_q = {d: 1 for d in DOMAINS}
    with pytest.raises(CorpusValidationError, match="train/val packed-document hash overlap"):
        validate_stem_corpus(
            corpus,
            train_quota=quotas,
            val_quota=val_q,
            min_fraction=0.0,
            load_tokenizer=False,
        )


def test_validate_rejects_short_train_val_packed_leak(tmp_path: Path) -> None:
    from arzlm.data.packed import pack_documents

    corpus = _mini_corpus(tmp_path)
    packed = pack_documents([[7, 8, 9]], eos_id=2)
    train_rec = write_shard_atomic(
        corpus / "train" / "science" / "shard-00000.bin",
        packed,
        meta={"creation_order": 0, "n_documents": 1},
    )
    val_rec = write_shard_atomic(
        corpus / "val" / "science" / "shard-00000.bin",
        packed,
        meta={"creation_order": 0, "n_documents": 1},
    )
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    meta["domains"]["science"]["train"]["n_tokens"] = int(train_rec["n_tokens"])
    meta["domains"]["science"]["train"]["shards"] = [train_rec]
    meta["domains"]["science"]["val"]["n_tokens"] = int(val_rec["n_tokens"])
    meta["domains"]["science"]["val"]["shards"] = [val_rec]
    meta["train_tokens_total"] = sum(int(meta["domains"][d]["train"]["n_tokens"]) for d in DOMAINS)
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    quotas = {d: 1 for d in DOMAINS}
    val_q = {d: 1 for d in DOMAINS}
    with pytest.raises(CorpusValidationError, match="train/val packed-document hash overlap"):
        validate_stem_corpus(
            corpus,
            train_quota=quotas,
            val_quota=val_q,
            min_fraction=0.0,
            load_tokenizer=False,
        )


def test_validate_rejects_stale_shard_checksum(tmp_path: Path) -> None:
    corpus = _mini_corpus(tmp_path)
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    meta["domains"]["math"]["train"]["shards"][0]["sha256"] = "00" * 32
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    quotas = {d: 1 for d in DOMAINS}
    val_q = {d: 1 for d in DOMAINS}
    with pytest.raises(CorpusValidationError, match="shard hash mismatch"):
        validate_stem_corpus(
            corpus,
            train_quota=quotas,
            val_quota=val_q,
            min_fraction=0.0,
            load_tokenizer=False,
        )


def test_rewrite_drops_exact_packed_duplicates_then_validate_passes(tmp_path: Path) -> None:
    from arzlm.data.packed import pack_documents
    from arzlm.data.packed_dedupe import rewrite_split_drop_exact_duplicates

    corpus = _mini_corpus(tmp_path)
    long_doc = list(range(10, 90))
    packed = pack_documents([long_doc, long_doc], eos_id=2)
    rec = write_shard_atomic(
        corpus / "train" / "science" / "shard-00000.bin",
        packed,
        meta={"creation_order": 0, "n_documents": 2},
    )
    files = ["train/science/shard-00000.bin"]
    result = rewrite_split_drop_exact_duplicates(
        corpus,
        files,
        eos_id=2,
        domain="science",
        split="train",
        seq_length=15,
        shard_tokens=8_388_608,
    )
    assert result["rewritten"] is True
    assert result["dropped_docs"] == 1
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    meta["domains"]["science"]["train"]["n_tokens"] = int(result["n_tokens_out"])
    meta["domains"]["science"]["train"]["shards"] = result["shards"]
    meta["domains"]["science"]["train"]["files"] = result["files"]
    meta["train_tokens_total"] = sum(int(meta["domains"][d]["train"]["n_tokens"]) for d in DOMAINS)
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    quotas = {d: 1 for d in DOMAINS}
    val_q = {d: 1 for d in DOMAINS}
    report = validate_stem_corpus(
        corpus,
        train_quota=quotas,
        val_quota=val_q,
        min_fraction=0.0,
        load_tokenizer=False,
    )
    assert report["passed"] is True
    assert report["domains"]["science"]["packed_train_short_collisions"] == 0


def test_rewrite_is_noop_when_packed_docs_are_unique(tmp_path: Path) -> None:
    from arzlm.data.packed_dedupe import rewrite_split_drop_exact_duplicates

    corpus = _mini_corpus(tmp_path)
    result = rewrite_split_drop_exact_duplicates(
        corpus,
        ["train/math/shard-00000.bin"],
        eos_id=2,
        domain="math",
        split="train",
        seq_length=2048,
    )
    assert result["rewritten"] is False
    assert result["dropped_docs"] == 0


def test_rewrite_drops_train_copy_of_val_document(tmp_path: Path) -> None:
    from arzlm.data.packed import pack_documents
    from arzlm.data.packed_dedupe import rewrite_split_drop_exact_duplicates
    from arzlm.data.corpus_validate import hash_complete_spans

    corpus = _mini_corpus(tmp_path)
    shared = list(range(10, 90))
    other = list(range(100, 180))
    packed = pack_documents([shared, other], eos_id=2)
    write_shard_atomic(
        corpus / "train" / "science" / "shard-00000.bin",
        packed,
        meta={"creation_order": 0, "n_documents": 2},
    )
    spans, _ = hash_complete_spans(packed, eos_id=2)
    shared_digest = spans[0][0]
    result = rewrite_split_drop_exact_duplicates(
        corpus,
        ["train/science/shard-00000.bin"],
        eos_id=2,
        domain="science",
        split="train",
        seq_length=15,
        extra_drop={shared_digest},
    )
    assert result["rewritten"] is True
    assert result["dropped_docs"] == 1
    leftover, _ = hash_complete_spans(
        __import__("numpy").fromfile(corpus / "train" / "science" / "shard-00000.bin", dtype="uint16"),
        eos_id=2,
    )
    assert all(h != shared_digest for h, _n in leftover)


def test_packer_resume_restores_stream_counters() -> None:
    from arzlm.data.stem_builder import pack_document_stream, pack_one_stem_domain

    stream_src = inspect.getsource(pack_document_stream)
    one_src = inspect.getsource(pack_one_stem_domain)
    assert 'stats.docs_duplicate = int(resume.get("docs_duplicate"' in stream_src
    assert 'stats.docs_inspected = int(resume.get("docs_inspected"' in stream_src
    assert 'skip = 0 if resume is None else int(resume.get("stream_index"' in one_src
    inspected, dups, acc_tr, acc_va, skipped, unenc = 3_154_665, 867_237, 2_279_924, 7_504, 0, 0
    assert inspected == acc_tr + acc_va + dups + skipped + unenc
