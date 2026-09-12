"""Tests for ArzLM-STEM corpus construction, dedup, mixture, and local epochs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from litgpt.tokenizer import Tokenizer

from arzlm.data.catalog import (
    FINEWEB_EDU,
    FINEMATH_4PLUS,
    FORBIDDEN_DATASETS,
    MATH_DECISION,
    MIXTURE_TARGET,
    NEMOTRON_CC_MATH,
    PES2O,
    STACK_EDU,
    assert_dataset_allowed,
    code_token_quotas,
)
from arzlm.data.hashing import DedupIndex, content_sha256
from arzlm.data.mixture import (
    CyclingDataset,
    OffsetSampler,
    epoch_assignments,
    load_stem_train_dataset,
)
from arzlm.data.packed import ConcatPackedDataset, PackedTokenDataset, pack_documents, read_meta
from arzlm.data.shards import sha256_file, write_shard_atomic, load_shard_records, write_partial, read_partial
from arzlm.data.sources import SourceDoc, synthetic_stem_documents
from arzlm.data.stem_builder import build_stem_corpus, pack_document_stream
from arzlm.data.textutil import as_token_ids
from arzlm.tokenizer.compare import evaluate_tokenizer, train_stem_candidate
from arzlm.tokenizer.train import VOCAB_SIZE
from tests.helpers import write_mini_tokenizer


def test_fineweb_parquet_prefix_is_10bt_only() -> None:
    from arzlm.data.sources import FINEWEB_PARQUET_PREFIX

    assert FINEWEB_PARQUET_PREFIX == "sample/10BT/"
    assert "100BT" not in FINEWEB_PARQUET_PREFIX
    assert "350BT" not in FINEWEB_PARQUET_PREFIX
    for name in FORBIDDEN_DATASETS:
        with pytest.raises(ValueError, match="FineWeb"):
            assert_dataset_allowed(name)
    assert_dataset_allowed(FINEWEB_EDU.dataset_id)


def test_source_revisions_are_pinned() -> None:
    assert len(FINEWEB_EDU.revision) == 40
    assert len(FINEMATH_4PLUS.revision) == 40
    assert len(STACK_EDU.revision) == 40
    assert len(PES2O.revision) == 40
    assert FINEWEB_EDU.config == "sample-10BT"
    assert FINEMATH_4PLUS.config == "finemath-4plus"
    assert PES2O.config == "v2"
    assert NEMOTRON_CC_MATH.pretraining_permitted is False
    assert MATH_DECISION["chosen"] == "HuggingFaceTB/finemath"


def test_truncated_gzip_jsonl_does_not_raise() -> None:
    import gzip
    import io
    import json

    from arzlm.data.sources import _iter_gzip_json_lines

    rows = [json.dumps({"text": f"doc-{i}", "source": "s2orc"}) + "\n" for i in range(8)]
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb") as gz:
        gz.write("".join(rows).encode("utf-8"))
    payload = raw.getvalue()
    truncated = io.BytesIO(payload[: max(20, len(payload) // 2)])
    lines = list(_iter_gzip_json_lines(truncated))
    assert all(isinstance(line, bytes) for line in lines)
    assert len(lines) < 8


def test_pes2o_s2orc_shards_are_full_text_files() -> None:
    from arzlm.data.catalog import PES2O_S2AG_TRAIN_SHARDS, PES2O_S2ORC_TRAIN_SHARDS
    from arzlm.data.sources import is_s2orc_source

    assert PES2O_S2AG_TRAIN_SHARDS == tuple(range(0, 10))
    assert PES2O_S2ORC_TRAIN_SHARDS == tuple(range(10, 20))
    assert is_s2orc_source("s2orc")
    assert is_s2orc_source("s2orc/train")
    assert not is_s2orc_source("s2ag/train")
    assert not is_s2orc_source("s2ag")


def test_code_language_mix_sums_to_one() -> None:
    from arzlm.data.catalog import CODE_LANGUAGE_WEIGHTS, TRAIN_TOKEN_QUOTA, TRAIN_TOKEN_TOTAL, TOKENS_PER_OPT_STEP, EXPECTED_OPT_STEPS, TOKEN_EXPOSURES

    assert abs(sum(CODE_LANGUAGE_WEIGHTS.values()) - 1.0) < 1e-9
    q = code_token_quotas(TRAIN_TOKEN_QUOTA["code"])
    assert sum(q.values()) == TRAIN_TOKEN_QUOTA["code"]
    assert CODE_LANGUAGE_WEIGHTS["Markdown"] <= 0.02
    assert TRAIN_TOKEN_TOTAL == 1_000_005_632
    assert TRAIN_TOKEN_TOTAL % TOKENS_PER_OPT_STEP == 0
    assert all(n % TOKENS_PER_OPT_STEP == 0 for n in TRAIN_TOKEN_QUOTA.values())
    assert TOKEN_EXPOSURES == 2_000_011_264
    assert EXPECTED_OPT_STEPS == 244_142


def test_take_chars_stops_at_budget_and_reports_progress() -> None:
    from types import SimpleNamespace

    from arzlm.tokenizer.stem_study import take_chars

    docs = [SimpleNamespace(text=f"token{i:02d}-xxxx") for i in range(20)]
    seen: list[tuple[int, int]] = []
    text, n, n_docs = take_chars(docs, 30, on_progress=lambda c, d: seen.append((c, d)))
    assert n >= 30
    assert n_docs >= 2
    assert n <= len(text)
    assert seen and seen[0][1] == 1


def test_keep_hub_parquet_row_allows_stack_edu_blob_ids() -> None:
    from arzlm.data.sources import keep_hub_parquet_row

    assert keep_hub_parquet_row({"blob_id": "abc", "license_type": "permissive"}) is True
    assert keep_hub_parquet_row({"text": "hello"}) is True
    assert keep_hub_parquet_row({"text": "  "}) is False
    assert keep_hub_parquet_row({"text": ""}) is False


def test_stack_edu_score3_fill_band_keeps_license_gate() -> None:
    from arzlm.data.sources import stack_edu_row_eligible
    from arzlm.data.stem_builder import pack_code_domain

    blob = {"blob_id": "a" * 40, "license_type": "permissive", "int_score": 3}
    assert stack_edu_row_eligible(blob) is False
    assert stack_edu_row_eligible(blob, min_score=3.0, max_score=4.0) is True
    assert stack_edu_row_eligible({**blob, "int_score": 4}, min_score=3.0, max_score=4.0) is False
    assert stack_edu_row_eligible({**blob, "int_score": 4}) is True
    assert (
        stack_edu_row_eligible(
            {"blob_id": "b" * 40, "license_type": "no_license", "int_score": 3},
            min_score=3.0,
            max_score=4.0,
        )
        is False
    )
    assert (
        stack_edu_row_eligible(
            {"blob_id": "c" * 40, "detected_licenses": ["MIT"], "int_score": 3},
            min_score=3.0,
            max_score=4.0,
        )
        is True
    )
    src = __import__("inspect").getsource(pack_code_domain)
    assert "min_score=3.0" in src
    assert "score3_lang_skip" in src
    assert "allow_lang_overflow" in src


def test_uint16_safety() -> None:
    packed = pack_documents([[0, 1, 32767, 65535]], eos_id=2)
    assert packed.dtype == np.uint16
    with pytest.raises(ValueError, match="uint16"):
        pack_documents([[65536]], eos_id=2)


def test_exact_content_hash_dedup(tmp_path: Path) -> None:
    idx = DedupIndex(tmp_path / "dedup.sqlite")
    a = content_sha256("Hello world")
    b = content_sha256("Hello world")
    assert a == b
    assert idx.lookup_content(a) is None
    idx.add(a, "general", "doc-a", "train", ["url:http://example.com/x"])
    idx.flush()
    hit = idx.lookup_content(b)
    assert hit is not None and hit[0] == "general"
    idx.record_dup(b, "math", "general", "content")
    counts = idx.dup_counts()
    assert counts["content:general->math"] == 1
    idx.close()


def test_text_table_column_prefers_text() -> None:
    from arzlm.data.sources import text_table_column

    assert text_table_column(["document", "text", "prompt"]) == "text"
    assert text_table_column(["article"]) == "article"
    assert text_table_column(["prompt", "id"]) == "prompt"


def test_locked_domain_skip_does_not_consume_supplements(monkeypatch) -> None:
    from arzlm.data.catalog import FINEWEB_EDU
    from arzlm.data.security.lockfile import LockedSource, SourceLock
    from arzlm.data.security.policy import LockFileEntry
    from arzlm.data.sources import _iter_locked_domain
    from arzlm.data.stem_builder import general_resume_cursor

    primary = [
        SourceDoc("general", FINEWEB_EDU.dataset_id, f"p{i}", f"primary document {i} " * 8)
        for i in range(5)
    ]
    extra = [
        SourceDoc("general", "HuggingFaceTB/smollm-corpus", f"s{i}", f"supplement document {i} " * 8)
        for i in range(3)
    ]

    def fake_table(src, skip: int = 0):
        assert skip == 0
        assert src.dataset_id == "HuggingFaceTB/smollm-corpus"
        yield from extra

    monkeypatch.setattr("arzlm.data.sources.iter_locked_text_table", fake_table)
    lock = SourceLock(
        schema_version=1,
        corpus="t",
        created_at="x",
        policy={},
        sources={
            "general": LockedSource(
                domain="general",
                dataset_id=FINEWEB_EDU.dataset_id,
                config="sample-10BT",
                revision=FINEWEB_EDU.revision,
                prefix="sample/10BT/",
                files={"sample/10BT/013_00000.parquet": LockFileEntry(path="sample/10BT/013_00000.parquet")},
            )
        },
        supplements=[
            LockedSource(
                domain="general",
                dataset_id="HuggingFaceTB/smollm-corpus",
                config=None,
                revision="abc",
                prefix="cosmopedia-v2/",
                files={"cosmopedia-v2/train-00000-of-00104.parquet": LockFileEntry(path="cosmopedia-v2/train-00000-of-00104.parquet")},
            )
        ],
    )
    docs = list(
        _iter_locked_domain(
            lock,
            "general",
            skip=10,
            primary_factory=lambda: iter(primary),
        )
    )
    assert [d.native_id for d in docs] == ["s0", "s1", "s2"]

    jumped = list(
        _iter_locked_domain(
            lock,
            "general",
            skip=0,
            start_source=1,
            primary_factory=lambda: iter(primary),
        )
    )
    assert [d.native_id for d in jumped] == ["s0", "s1", "s2"]

    start, skip = general_resume_cursor(
        {
            "tokens_train": 194_445_575,
            "docs_inspected": 7_707_881,
            "docs_duplicate": 7_520_775,
            "stream_index": 7_707_881,
        },
        lock=lock,
    )
    assert start == 1
    assert skip == 0

    start, skip = general_resume_cursor({"start_source": 1, "source_stream_index": 0, "stream_index": 99})
    assert start == 1
    assert skip == 0

    start, skip = general_resume_cursor(
        {
            "tokens_train": 1_000,
            "docs_inspected": 2_000,
            "docs_duplicate": 10,
            "stream_index": 2_000,
        },
        lock=lock,
    )
    assert start == 0
    assert skip == 2_000


def test_parquet_native_ids_are_global_not_batch_local(tmp_path: Path, monkeypatch) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from arzlm.data.security.lockfile import LockedSource
    from arzlm.data.security.policy import LockFileEntry
    from arzlm.data.sources import iter_locked_text_table

    rows = pa.table({"text": [f"cosmopedia unique document {i} " * 4 for i in range(300)]})
    parquet_path = tmp_path / "train-00000-of-00104.parquet"
    pq.write_table(rows, parquet_path)
    monkeypatch.setattr("arzlm.data.sources.gated_hub_download", lambda *a, **k: parquet_path)
    src = LockedSource(
        domain="general",
        dataset_id="HuggingFaceTB/smollm-corpus",
        config=None,
        revision="abc",
        prefix="cosmopedia-v2/",
        files={
            "cosmopedia-v2/train-00000-of-00104.parquet": LockFileEntry(
                path="cosmopedia-v2/train-00000-of-00104.parquet"
            )
        },
    )
    docs = list(iter_locked_text_table(src))
    ids = [d.native_id for d in docs]
    assert len(ids) == 300
    assert ids[0].endswith(":0")
    assert ids[256].endswith(":256")
    assert len(set(ids)) == 300


def test_as_token_ids_rejects_non_integers() -> None:
    assert as_token_ids([0, 1, 65535]) == [0, 1, 65535]
    assert as_token_ids([int]) is None
    assert as_token_ids([[1, 2, 3]]) is None
    assert as_token_ids([65536]) is None
    assert as_token_ids(None) is None


def test_packer_skips_non_integer_token_ids(tmp_path: Path) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    tokenizer = Tokenizer(tok_dir)
    dedup = DedupIndex(tmp_path / "d.sqlite")

    class Encoder:
        n = 0

        def encode(self, text: str):
            self.n += 1
            if "BADTYPE" in text:
                return [int]
            return tokenizer.encode(text, bos=False, eos=False).tolist()

    docs = [
        SourceDoc("general", "synthetic", "bad", "BADTYPE document should not crash the packer."),
        SourceDoc("general", "synthetic", "ok1", "photosynthesis converts light energy into chemical energy."),
        SourceDoc("general", "synthetic", "ok2", "force equals mass times acceleration in newtonian mechanics."),
        SourceDoc("general", "synthetic", "ok3", "the water cycle includes evaporation condensation and precipitation."),
        SourceDoc("general", "synthetic", "ok4", "prime numbers are integers greater than one with no positive divisors."),
    ]
    stats, train_shards, val_shards = pack_document_stream(
        iter(docs),
        domain="general",
        out_dir=tmp_path / "out-bad",
        tokenizer_dir=tok_dir,
        train_quota=40,
        val_quota=9,
        seq_length=8,
        encoder=Encoder(),
        tokenizer=tokenizer,
        dedup=dedup,
        shard_tokens=10_000,
        resume=None,
    )
    assert stats.docs_unencodable >= 1
    assert stats.docs_accepted_train + stats.docs_accepted_val >= 1
    assert train_shards or val_shards


def test_malformed_documents_are_skipped(tmp_path: Path) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    tokenizer = Tokenizer(tok_dir)
    dedup = DedupIndex(tmp_path / "d.sqlite")
    docs = [
        SourceDoc("general", "synthetic", "empty", "   "),
        SourceDoc("general", "synthetic", "ok1", "photosynthesis converts light energy into chemical energy."),
        SourceDoc("general", "synthetic", "ok2", "force equals mass times acceleration in newtonian mechanics."),
        SourceDoc("general", "synthetic", "ok3", "the water cycle includes evaporation condensation and precipitation."),
        SourceDoc("general", "synthetic", "ok4", "prime numbers are integers greater than one with no positive divisors."),
        SourceDoc("general", "synthetic", "none-text", ""),
    ]
    stats, train_shards, val_shards = pack_document_stream(
        iter(docs),
        domain="general",
        out_dir=tmp_path / "out",
        tokenizer_dir=tok_dir,
        train_quota=40,
        val_quota=9,
        seq_length=8,
        encoder=None,
        tokenizer=tokenizer,
        dedup=dedup,
        shard_tokens=4096,
        resume=None,
    )
    assert stats.docs_skipped >= 1
    assert stats.docs_accepted_train >= 1
    assert train_shards
    dedup.close()


def test_eos_boundaries_and_train_val_isolation(tmp_path: Path) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    out = build_stem_corpus(
        "tiny",
        source="synthetic",
        out_dir=tmp_path / "stem",
        tokenizer_dir=tok_dir,
        seq_length=32,
        shard_tokens=2048,
        isolated_encode=True,
    )
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "arzlm-stem-v1"
    tokenizer = Tokenizer(tok_dir)
    eos = int(meta["eos_id"])
    assert eos == tokenizer.eos_id
    for domain, spec in meta["domains"].items():
        for split in ("train", "val"):
            found_eos = False
            for rel in spec[split]["files"]:
                data = np.memmap(out / rel, dtype=np.uint16, mode="r")
                if (np.asarray(data) == eos).any():
                    found_eos = True
                    break
            assert found_eos, f"{domain} {split} has no EOS={eos}"
            assert spec[split]["n_tokens"] > 0
            assert spec[split]["n_documents"] > 0
    for domain in MIXTURE_TARGET:
        assert domain in meta["mixture_actual"]


def test_atomic_shards_and_hashes(tmp_path: Path) -> None:
    tokens = np.arange(100, dtype=np.uint16)
    path = tmp_path / "shard-00000.bin"
    rec = write_shard_atomic(
        path,
        tokens,
        meta={"domain": "math", "split": "train", "creation_order": 0, "n_documents": 4},
    )
    assert path.is_file()
    sidecar = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    assert rec["sha256"] == sha256_file(path)
    assert rec["n_tokens"] == 100
    assert rec["n_documents"] == 4
    assert sidecar["n_documents"] == 4


def test_interrupted_preparation_resume(tmp_path: Path) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    tokenizer = Tokenizer(tok_dir)
    dedup = DedupIndex(tmp_path / "d.sqlite")
    docs = list(synthetic_stem_documents("math", n_docs=400, seed=1))
    stats1, train1, val1 = pack_document_stream(
        iter(docs[:80]),
        domain="math",
        out_dir=tmp_path / "out",
        tokenizer_dir=tok_dir,
        train_quota=80,
        val_quota=9,
        seq_length=8,
        encoder=None,
        tokenizer=tokenizer,
        dedup=dedup,
        shard_tokens=512,
        resume=None,
    )
    resume = {
        "docs_inspected": stats1.docs_inspected,
        "docs_accepted_train": stats1.docs_accepted_train,
        "docs_accepted_val": stats1.docs_accepted_val,
        "docs_duplicate": stats1.docs_duplicate,
        "docs_skipped": stats1.docs_skipped,
        "docs_unencodable": stats1.docs_unencodable,
        "tokens_train": stats1.tokens_train,
        "tokens_val": stats1.tokens_val,
        "stream_index": 0,
        "train_shards": train1,
        "val_shards": val1,
        "train_next_shard": len(train1),
        "val_next_shard": len(val1),
    }
    stats2, train2, val2 = pack_document_stream(
        iter(docs[80:]),
        domain="math",
        out_dir=tmp_path / "out",
        tokenizer_dir=tok_dir,
        train_quota=200,
        val_quota=18,
        seq_length=8,
        encoder=None,
        tokenizer=tokenizer,
        dedup=dedup,
        shard_tokens=512,
        resume=resume,
    )
    assert stats2.tokens_train >= stats1.tokens_train
    assert len(train2) >= len(train1)
    dedup.close()


def test_deterministic_mixture_and_epoch_change() -> None:
    lengths = {"general": 55, "math": 20, "code": 15, "science": 10}
    a = epoch_assignments(seed=42, epoch=0, lengths=lengths, weights=MIXTURE_TARGET)
    b = epoch_assignments(seed=42, epoch=0, lengths=lengths, weights=MIXTURE_TARGET)
    c = epoch_assignments(seed=42, epoch=1, lengths=lengths, weights=MIXTURE_TARGET)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    from arzlm.data.mixture import DOMAIN_IDS as IDS

    counts = {d: int((a[:, 0] == IDS[d]).sum()) for d in lengths}
    total = sum(counts.values())
    for d, w in MIXTURE_TARGET.items():
        assert abs(counts[d] / total - w) < 0.08


def test_weight_override_changes_mix() -> None:
    lengths = {"general": 50, "math": 50, "code": 50, "science": 50}
    heavy_math = {"general": 0.10, "math": 0.70, "code": 0.10, "science": 0.10}
    a = epoch_assignments(seed=0, epoch=0, lengths=lengths, weights=heavy_math)
    from arzlm.data.mixture import DOMAIN_IDS

    math_frac = float((a[:, 0] == DOMAIN_IDS["math"]).mean())
    assert math_frac > 0.5


def test_concat_shards_and_cycling_resume(tmp_path: Path) -> None:
    seq = 8
    block = seq + 1
    a = np.arange(block * 4, dtype=np.uint16)
    b = np.arange(100, 100 + block * 4, dtype=np.uint16)
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    a.tofile(p1)
    b.tofile(p2)
    ds = ConcatPackedDataset([p1, p2], seq_length=seq)
    assert len(ds) == 8
    inner = PackedTokenDataset(p1, seq_length=seq)
    n_items = 6
    cyc = CyclingDataset(inner, seed=42, n_items=n_items)
    first = [cyc[i].tolist() for i in range(3)]
    sampler = OffsetSampler(3, 6)
    rest = [cyc[i].tolist() for i in sampler]
    assert rest[0] == cyc[3].tolist()
    assert first[0] != rest[0] or len(inner) > 1


def test_no_remote_on_repeated_local_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    out = build_stem_corpus(
        "tiny",
        source="synthetic",
        out_dir=tmp_path / "stem",
        tokenizer_dir=tok_dir,
        seq_length=32,
        shard_tokens=2048,
        isolated_encode=True,
    )

    def boom(*_a, **_k):
        raise RuntimeError("remote access not allowed on local epoch")

    monkeypatch.setattr("datasets.load_dataset", boom)
    ds = load_stem_train_dataset(out, seq_length=32, seed=42, n_items=64)
    _ = ds[0]
    _ = ds[len(ds) // 2 if len(ds) > 2 else 0]
    # Second epoch indices
    if len(ds) > 8:
        _ = ds[8]


def test_tokenizer_math_code_unicode(tmp_path: Path) -> None:
    texts = [
        r"Solve $\frac{a}{b}$ and \sum_{i=1}^{n} i.",
        "def foo(x):\n    return x + 1\n",
        "The Hamiltonian of the electron is H = p^2/2m.",
        "Photosynthesis converts light energy into chemical energy.",
    ] * 80
    train_stem_candidate(texts, tmp_path / "tok", vocab_size=800)
    tok_dir = tmp_path / "tok"
    report = evaluate_tokenizer(tok_dir, {"math": texts[0] * 20, "code": texts[1] * 20, "science": texts[2] * 20, "general": texts[3] * 20})
    assert report["unicode_roundtrip"] is True
    assert report["math_fragmentation"][r"\frac{a}{b}"] >= 1
    assert report["code_fragmentation"]["    def foo(x):\n        return x + 1\n"] >= 1
    lit = Tokenizer(tok_dir)
    sample = "naïve café 日本語 ☃"
    ids = lit.encode(sample, bos=False, eos=False)
    decoded = lit.decode(ids)
    assert "caf" in decoded or "naïve" in decoded or len(ids) > 0


def test_target_vocab_constant() -> None:
    assert VOCAB_SIZE == 32000


def test_mixed_iterator_slices_to_budget() -> None:
    from arzlm.tokenizer.stem_study import mixed_iterator

    train = {d: "x" * 50_000 for d in MIXTURE_TARGET}
    got = "".join(mixed_iterator(train, 1000))
    assert len(got) == 1000


def test_stem_dataset_identity_in_manifest(tmp_path: Path) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    out = build_stem_corpus(
        "tiny",
        source="synthetic",
        out_dir=tmp_path / "stem",
        tokenizer_dir=tok_dir,
        seq_length=32,
        shard_tokens=2048,
        isolated_encode=True,
    )
    meta = read_meta(out)
    assert meta["math_decision"]["rejected"].startswith("nvidia/")
    for domain in MIXTURE_TARGET:
        assert meta["domains"][domain]["source"]["dataset_id"] == "synthetic"


def test_train_val_native_ids_are_disjoint(tmp_path: Path) -> None:
    import sqlite3

    tok_dir = write_mini_tokenizer(tmp_path)
    out = build_stem_corpus(
        "tiny",
        source="synthetic",
        out_dir=tmp_path / "stem",
        tokenizer_dir=tok_dir,
        seq_length=32,
        shard_tokens=2048,
        isolated_encode=False,
    )
    conn = sqlite3.connect(str(out / "state" / "dedup.sqlite"))
    rows = conn.execute("SELECT native_id, split FROM content").fetchall()
    conn.close()
    train_ids = {r[0] for r in rows if r[1] == "train"}
    val_ids = {r[0] for r in rows if r[1] == "val"}
    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)


def test_crash_resume_uses_disk_shards_not_inflated_json(tmp_path: Path) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    tokenizer = Tokenizer(tok_dir)
    dedup = DedupIndex(tmp_path / "d.sqlite")
    docs = list(synthetic_stem_documents("math", n_docs=500, seed=3))
    stats1, train1, _val1 = pack_document_stream(
        iter(docs[:120]),
        domain="math",
        out_dir=tmp_path / "out",
        tokenizer_dir=tok_dir,
        train_quota=80,
        val_quota=9,
        seq_length=8,
        encoder=None,
        tokenizer=tokenizer,
        dedup=dedup,
        shard_tokens=256,
        resume=None,
    )
    assert train1
    disk = load_shard_records(tmp_path / "out" / "train" / "math")
    assert disk
    lying_resume = {
        "docs_inspected": stats1.docs_inspected,
        "docs_accepted_train": stats1.docs_accepted_train,
        "docs_accepted_val": stats1.docs_accepted_val,
        "docs_duplicate": stats1.docs_duplicate,
        "docs_skipped": stats1.docs_skipped,
        "docs_unencodable": stats1.docs_unencodable,
        "tokens_train": 10_000_000,
        "tokens_val": 0,
        "stream_index": 0,
        "train_shards": [],
        "val_shards": [],
        "train_next_shard": 0,
        "val_next_shard": 0,
    }
    stats2, train2, _val2 = pack_document_stream(
        iter(docs[120:]),
        domain="math",
        out_dir=tmp_path / "out",
        tokenizer_dir=tok_dir,
        train_quota=220,
        val_quota=18,
        seq_length=8,
        encoder=None,
        tokenizer=tokenizer,
        dedup=dedup,
        shard_tokens=256,
        resume=lying_resume,
    )
    assert stats2.tokens_train >= 220
    assert len(train2) >= len(train1)
    first = (tmp_path / "out" / "train" / "math" / train1[0]["file"]).read_bytes()
    assert first
    dedup.close()


def test_partial_buffer_resume(tmp_path: Path) -> None:
    tokens = np.arange(40, dtype=np.uint16)
    path = tmp_path / "math.train.partial.bin"
    write_partial(path, tokens, n_documents=2)
    got, n_docs = read_partial(path)
    assert got.tolist() == list(range(40))
    assert n_docs == 2


def test_language_quota_skips_full_lang(tmp_path: Path) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    tokenizer = Tokenizer(tok_dir)
    dedup = DedupIndex(tmp_path / "d.sqlite")
    py = [
        SourceDoc("code", "synthetic", f"py{i}", f"def foo():\n    return {i}\n" + "alpha " * 8, language="Python")
        for i in range(30)
    ]
    c_docs = [
        SourceDoc("code", "synthetic", f"c{i}", f"int main() {{ return {i}; }}\n" + "beta " * 8, language="C")
        for i in range(30)
    ]
    stats, train_shards, _val = pack_document_stream(
        iter(py + c_docs),
        domain="code",
        out_dir=tmp_path / "out",
        tokenizer_dir=tok_dir,
        train_quota=80,
        val_quota=9,
        seq_length=8,
        encoder=None,
        tokenizer=tokenizer,
        dedup=dedup,
        shard_tokens=2048,
        resume=None,
        language_train_quotas={"Python": 20, "C": 80},
    )
    assert stats.lang_train_tokens.get("C", 0) > 0
    assert stats.docs_skipped >= 1
    assert train_shards
    dedup.close()


def test_yaml_mixture_override(tmp_path: Path) -> None:
    from arzlm.paths import REPO_ROOT
    from arzlm.training.config import load_experiment_config

    model = (REPO_ROOT / "configs" / "model-arzlm-100m.yaml").as_posix()
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        f"""
model_config: {model}
out_dir: {(tmp_path / 'out').as_posix()}
data_dir: {(tmp_path / 'data').as_posix()}
tokenizer_dir: {(tmp_path / 'tok').as_posix()}
precision: 32-true
compile: false
seed: 42
devices: 1
logger_name: csv
resume: false
num_workers: 0
mixture:
  general: 0.10
  math: 0.70
  code: 0.10
  science: 0.10
train:
  save_interval: 50
  log_interval: 1
  global_batch_size: 1
  micro_batch_size: 1
  lr_warmup_steps: 1
  max_tokens: 1000
  max_steps: 1
  max_seq_length: 32
  tie_embeddings: true
  max_norm: 1.0
  min_lr: 6.0e-5
eval:
  interval: 100
  max_iters: 1
  initial_validation: false
  final_validation: false
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 6.0e-4
    weight_decay: 0.1
    betas: [0.9, 0.95]
""",
        encoding="utf-8",
    )
    exp = load_experiment_config(cfg)
    assert exp.mixture is not None
    assert abs(float(exp.mixture["math"]) - 0.70) < 1e-9


def test_stem_tokenizer_vocab_is_32000() -> None:
    from arzlm.paths import REPO_ROOT

    tok_dir = REPO_ROOT / "tokenizer" / "stem-v1"
    if not (tok_dir / "tokenizer.json").is_file():
        pytest.skip("stem-v1 tokenizer not in tree")
    lit = Tokenizer(tok_dir)
    assert int(lit.vocab_size) == 32000
    assert lit.eos_id is not None


def test_stem_mixture_checkpoint_data_position(tmp_path: Path) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    out = build_stem_corpus(
        "tiny",
        source="synthetic",
        out_dir=tmp_path / "stem",
        tokenizer_dir=tok_dir,
        seq_length=32,
        shard_tokens=2048,
        isolated_encode=False,
    )
    from arzlm.training.loop import make_dataloaders

    loader0, _, domains = make_dataloaders(
        out, seq_length=32, batch_size=1, num_workers=0, seed=3, n_items=8, start_index=0
    )
    batches = [batch.clone() for batch in loader0]
    loader1, _, _ = make_dataloaders(
        out, seq_length=32, batch_size=1, num_workers=0, seed=3, n_items=8, start_index=2
    )
    resumed = next(iter(loader1))
    assert torch.equal(resumed, batches[2])
    assert domains
    assert set(domains).issubset(set(MIXTURE_TARGET))
