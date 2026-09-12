"""Source-file security gate: denylist, fail-closed Hub metadata, lock pinning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arzlm.data.catalog import FINEWEB_EDU, FINEMATH_4PLUS, PES2O, STACK_EDU
from arzlm.data.security.audit import (
    SecurityAuditError,
    build_source_lock,
    format_audit_report,
    require_remote_security_audit,
    run_source_security_audit,
)
from arzlm.data.security.gate import (
    SecurityError,
    assert_file_fetch_allowed,
    evaluate_file,
    list_fetchable_parquet,
)
from arzlm.data.security.hub import FakeHubClient
from arzlm.data.security.inert import assert_inert_load_kwargs, load_dataset_inert
from arzlm.data.security.lockfile import dump_source_lock, load_source_lock, write_source_lock, LockedSource, SourceLock
from arzlm.data.security.policy import (
    LockFileEntry,
    RemoteFileMeta,
    classify_remote_file,
    load_denylist,
)
from arzlm.data.security.quarantine import looks_like_user_hf_cache, quarantine_project_file
from arzlm.data.security.scan_local import maybe_scan_local_file
from arzlm.paths import REPO_ROOT, SECURITY_DIR

FLAGGED_PATH = "data/CC-MAIN-2024-38/000_00041.parquet"
FLAGGED_SHA = "87753428a95294f19aac1f579d145c264ab55484d9fd205a413698821aaf1dc2"
FW = FINEWEB_EDU.dataset_id
REV = FINEWEB_EDU.revision


def _meta(**kwargs) -> RemoteFileMeta:
    base = dict(
        dataset_id=FW,
        revision=REV,
        path="sample/10BT/000_00000.parquet",
        size=100,
        sha256="a" * 64,
        security_status="safe",
        av_status="safe",
        av_message="No security issues detected",
        pickle_status="unscanned",
    )
    base.update(kwargs)
    return RemoteFileMeta(**base)


def _denylist(tmp_path: Path | None = None):
    path = SECURITY_DIR / "denylist.json"
    return load_denylist(path)


def test_known_unsafe_file_is_rejected() -> None:
    denylist = _denylist()
    meta = _meta(
        path=FLAGGED_PATH,
        sha256=FLAGGED_SHA,
        security_status="unsafe",
        av_status="unsafe",
        av_message="Hugging Face ClamAV detected 1 infection(s)",
    )
    v = classify_remote_file(meta, denylist=denylist)
    assert v.kind == "denied"
    assert v.denylist is True


def test_known_sha256_denylist_works() -> None:
    denylist = _denylist()
    meta = _meta(
        dataset_id="HuggingFaceFW/fineweb-edu",
        path="relocated/not-the-original-name.parquet",
        sha256=FLAGGED_SHA,
        security_status="safe",
        av_status="safe",
    )
    v = classify_remote_file(meta, denylist=denylist)
    assert v.kind == "denied"
    assert "denylist" in v.reason


def test_explicitly_unsafe_remote_metadata_fails_closed() -> None:
    meta = _meta(
        path="sample/10BT/evil.parquet",
        sha256="b" * 64,
        security_status="unsafe",
        av_status="unsafe",
        av_message="infected",
    )
    v = classify_remote_file(meta, denylist=[])
    assert v.kind == "denied"
    with pytest.raises(SecurityError, match="DENIED"):
        assert_file_fetch_allowed(
            FW,
            REV,
            meta.path,
            denylist=[],
            lock=None,
            client=FakeHubClient([meta]),
        )


def test_unknown_security_state_fails_closed() -> None:
    meta = _meta(security_status=None, av_status=None, av_message=None, path="sample/10BT/unknown.parquet")
    v = classify_remote_file(meta, denylist=[])
    assert v.kind == "unresolved"
    meta2 = _meta(security_status="unscanned", av_status="unscanned", path="sample/10BT/pending.parquet")
    v2 = classify_remote_file(meta2, denylist=[])
    assert v2.kind == "unresolved"


def test_approved_file_proceeds() -> None:
    meta = _meta()
    v = classify_remote_file(meta, denylist=_denylist())
    assert v.kind == "approved"
    got = assert_file_fetch_allowed(FW, REV, meta.path, denylist=_denylist(), client=FakeHubClient([meta]))
    assert got.path == meta.path


def test_queued_with_safe_av_is_approved() -> None:
    meta = _meta(security_status="queued", av_status="safe", path="finemath-4plus/train-00000-of-00064.parquet")
    meta = RemoteFileMeta(
        dataset_id=FINEMATH_4PLUS.dataset_id,
        revision=FINEMATH_4PLUS.revision,
        path="finemath-4plus/train-00000-of-00064.parquet",
        size=1,
        sha256="c" * 64,
        security_status="queued",
        av_status="safe",
        pickle_status="unscanned",
    )
    assert classify_remote_file(meta, denylist=[]).kind == "approved"


def test_revision_pinning_mismatch_aborts() -> None:
    from arzlm.data.security.lockfile import LockedSource, SourceLock

    meta = _meta()
    lock = SourceLock(
        schema_version=1,
        corpus="t",
        created_at="x",
        policy={},
        sources={
            "general": LockedSource(
                domain="general",
                dataset_id=FW,
                config="sample-10BT",
                revision="0" * 40,
                prefix="sample/10BT/",
                files={meta.path: LockFileEntry(path=meta.path, sha256=meta.sha256)},
            )
        },
    )
    client = FakeHubClient([meta], resolved={(FW, REV): REV})
    _, verdict = __import__("arzlm.data.security.gate", fromlist=["resolve_and_evaluate"]).resolve_and_evaluate(
        FW, REV, meta.path, denylist=[], lock=lock, client=client
    )
    assert verdict.kind == "mismatch"


def test_hash_mismatch_aborts() -> None:
    meta = _meta(sha256="d" * 64)
    lock_entry = LockFileEntry(path=meta.path, sha256="e" * 64)
    v = classify_remote_file(meta, denylist=[], lock_entry=lock_entry)
    assert v.kind == "mismatch"
    assert "sha256" in v.reason


def test_unscanned_can_be_reviewed_allow() -> None:
    meta = _meta(security_status="unscanned", av_status="unscanned")
    lock_entry = LockFileEntry(path=meta.path, sha256=meta.sha256, reviewed_allow_unscanned=True)
    v = classify_remote_file(meta, denylist=[], lock_entry=lock_entry)
    assert v.kind == "approved"


def test_full_dump_path_denied_even_if_unscanned() -> None:
    meta = _meta(path="data/CC-MAIN-2024-10/000_00000.parquet", sha256="f" * 64, security_status="safe", av_status="safe")
    v = classify_remote_file(meta, denylist=[])
    assert v.kind == "denied"
    assert "CC-MAIN" in v.reason


def test_pickle_format_denied() -> None:
    meta = _meta(path="sample/10BT/weights.pkl", security_status="safe", av_status="safe")
    v = classify_remote_file(meta, denylist=[])
    assert v.kind == "denied"


def test_no_trust_remote_code() -> None:
    with pytest.raises(SecurityError, match="trust_remote_code"):
        assert_inert_load_kwargs({"trust_remote_code": True})
    out = assert_inert_load_kwargs({"path": "x", "streaming": True})
    assert out["trust_remote_code"] is False
    text = ""
    for p in (REPO_ROOT / "src/arzlm/data").rglob("*.py"):
        text += p.read_text(encoding="utf-8")
    assert "trust_remote_code=True)" not in text
    assert "trust_remote_code = True" not in text
    assert 'kwargs["trust_remote_code"] = True' not in text


def test_no_document_url_fetch_in_fineweb_iterator() -> None:
    text = (REPO_ROOT / "src/arzlm/data/sources.py").read_text(encoding="utf-8")
    assert "keys.append(f\"url:{url}\")" in text
    assert "urlopen(row.get(\"url\")" not in text
    assert "requests.get" not in text


def test_stack_edu_is_data_only() -> None:
    swh = (REPO_ROOT / "src/arzlm/data/swh.py").read_text(encoding="utf-8")
    sources = (REPO_ROOT / "src/arzlm/data/sources.py").read_text(encoding="utf-8")
    assert "Never exec" in swh or "inert training text" in swh
    assert "training text only" in sources
    assert "subprocess" not in sources.split("def iter_stack_edu_language")[1].split("def synthetic_stem")[0]


def test_source_lock_reproducibility(tmp_path: Path) -> None:
    files = [
        _meta(),
        RemoteFileMeta(
            dataset_id=FINEMATH_4PLUS.dataset_id,
            revision=FINEMATH_4PLUS.revision,
            path="finemath-4plus/train-00000-of-00064.parquet",
            size=10,
            sha256="1" * 64,
            security_status="safe",
            av_status="safe",
        ),
        RemoteFileMeta(
            dataset_id=STACK_EDU.dataset_id,
            revision=STACK_EDU.revision,
            path="Python/train-00000-of-00005.parquet",
            size=10,
            sha256="2" * 64,
            security_status="safe",
            av_status="safe",
        ),
        RemoteFileMeta(
            dataset_id=PES2O.dataset_id,
            revision=PES2O.revision,
            path="data/v2/train-00010-of-00020.json.gz",
            size=10,
            sha256="3" * 64,
            security_status="queued",
            av_status="safe",
        ),
    ]
    client = FakeHubClient(files)
    a = dump_source_lock(build_source_lock(client=client, denylist=[]))
    b = dump_source_lock(build_source_lock(client=client, denylist=[]))
    a.pop("created_at")
    b.pop("created_at")
    a["policy"].pop("checked_at_note", None)
    b["policy"].pop("checked_at_note", None)
    assert a["sources"] == b["sources"]
    path = tmp_path / "source-lock.json"
    write_source_lock(build_source_lock(client=client, denylist=[]), path)
    loaded = load_source_lock(path)
    assert loaded.sources["general"].revision == REV
    assert "sample/10BT/000_00000.parquet" in loaded.sources["general"].files


def _passing_client() -> FakeHubClient:
    files = []
    files.append(_meta(path="sample/10BT/000_00000.parquet", sha256="aa" * 32))
    files.append(
        RemoteFileMeta(
            dataset_id=FW,
            revision=REV,
            path=FLAGGED_PATH,
            size=802227208,
            sha256=FLAGGED_SHA,
            security_status="unsafe",
            av_status="unsafe",
            av_message="Hugging Face ClamAV detected 1 infection(s)",
        )
    )
    files.append(
        RemoteFileMeta(
            dataset_id=FINEMATH_4PLUS.dataset_id,
            revision=FINEMATH_4PLUS.revision,
            path="finemath-4plus/train-00000-of-00064.parquet",
            size=10,
            sha256="11" * 32,
            security_status="safe",
            av_status="safe",
        )
    )
    for lang in ("Python", "C", "Cpp", "Java", "JavaScript", "TypeScript", "Rust", "Go", "SQL", "Shell", "CSharp", "PHP", "Ruby", "Swift", "Markdown"):
        files.append(
            RemoteFileMeta(
                dataset_id=STACK_EDU.dataset_id,
                revision=STACK_EDU.revision,
                path=f"{lang}/train-00000-of-00001.parquet",
                size=10,
                sha256=f"{lang[:2]}".encode().hex()[:2] * 32,
                security_status="safe",
                av_status="safe",
            )
        )
    for i in range(10, 20):
        files.append(
            RemoteFileMeta(
                dataset_id=PES2O.dataset_id,
                revision=PES2O.revision,
                path=f"data/v2/train-{i:05d}-of-00020.json.gz",
                size=10,
                sha256=f"{i:02d}" * 32,
                security_status="queued",
                av_status="safe",
            )
        )
    return FakeHubClient(files)


def test_audit_passing_fixture_and_denylist_canary() -> None:
    client = _passing_client()
    lock = build_source_lock(client=client, denylist=_denylist())
    report = run_source_security_audit(client=client, lock=lock, denylist=_denylist())
    text = format_audit_report(report)
    assert "FineWeb-Edu: PASS" in text
    assert "Math: PASS" in text
    assert "Code: PASS" in text
    assert "Science: PASS" in text
    assert FLAGGED_PATH in text
    assert "DENIED" in text
    assert report.passed
    require_remote_security_audit(client=client, lock=lock, denylist=_denylist())


def test_audit_unresolved_blocks_corpus_prep() -> None:
    meta = _meta(security_status="unscanned", av_status="unscanned")
    client = FakeHubClient([meta])
    with pytest.raises(SecurityAuditError):
        require_remote_security_audit(client=client, lock=None, denylist=_denylist())


def test_lock_pinned_safe_when_live_metadata_omitted() -> None:
    from arzlm.data.security.lockfile import LockedSource, SourceLock

    live = _meta(path="Python/train-00000-of-00005.parquet", security_status=None, av_status=None, sha256="ab" * 32)
    live = RemoteFileMeta(
        dataset_id=STACK_EDU.dataset_id,
        revision=STACK_EDU.revision,
        path="Python/train-00000-of-00005.parquet",
        size=10,
        sha256="ab" * 32,
        security_status=None,
        av_status=None,
    )
    lock = SourceLock(
        schema_version=1,
        corpus="t",
        created_at="x",
        policy={},
        sources={
            "code": LockedSource(
                domain="code",
                dataset_id=STACK_EDU.dataset_id,
                config=None,
                revision=STACK_EDU.revision,
                prefix="Python/",
                files={
                    live.path: LockFileEntry(
                        path=live.path,
                        sha256=live.sha256,
                        security_status="safe",
                        av_status="safe",
                    )
                },
            )
        },
    )
    v = evaluate_file(live, denylist=[], lock=lock)
    assert v.kind == "approved"
    assert "lock-pinned" in v.reason


def test_lock_ignores_unlocked_unscanned_siblings() -> None:
    """Approved locked files pass even when Hub still lists unscanned siblings."""
    good = _meta(path="sample/10BT/013_00000.parquet", sha256="ab" * 32)
    pending = _meta(
        path="sample/10BT/000_00000.parquet",
        sha256="cd" * 32,
        security_status="unscanned",
        av_status="unscanned",
    )
    client = FakeHubClient([good, pending])
    lock = build_source_lock(client=client, denylist=_denylist())
    assert "sample/10BT/013_00000.parquet" in lock.sources["general"].files
    assert "sample/10BT/000_00000.parquet" not in lock.sources["general"].files
    report = run_source_security_audit(client=client, lock=lock, denylist=_denylist())
    assert report.domains["general"].passed
    approved = list_fetchable_parquet(
        FW, REV, "sample/10BT/", denylist=_denylist(), lock=lock, client=client
    )
    assert [m.path for m in approved] == [good.path]


def test_list_fetchable_excludes_denied_and_fails_on_unresolved() -> None:
    good = _meta(path="sample/10BT/000_00000.parquet", sha256="ab" * 32)
    pending = _meta(
        path="sample/10BT/001_00000.parquet",
        sha256="cd" * 32,
        security_status="unscanned",
        av_status="unscanned",
    )
    denied = _meta(
        path="sample/10BT/evil.parquet",
        sha256=FLAGGED_SHA,
        security_status="unsafe",
        av_status="unsafe",
    )
    client = FakeHubClient([good, denied])
    approved = list_fetchable_parquet(FW, REV, "sample/10BT/", denylist=_denylist(), lock=None, client=client)
    assert [m.path for m in approved] == [good.path]
    with pytest.raises(SecurityError, match="unresolved"):
        list_fetchable_parquet(
            FW,
            REV,
            "sample/10BT/",
            denylist=[],
            lock=None,
            client=FakeHubClient([good, pending]),
        )


def test_local_av_skipped_by_default(tmp_path: Path) -> None:
    blob = tmp_path / "x.parquet"
    blob.write_bytes(b"PAR1")
    result = maybe_scan_local_file(blob, enabled=False)
    assert result.skipped is True
    assert result.infected is False


def test_quarantine_refuses_user_cache(tmp_path: Path) -> None:
    sneaky = tmp_path / ".cache" / "huggingface" / "hub" / "x.parquet"
    sneaky.parent.mkdir(parents=True)
    sneaky.write_bytes(b"nope")
    assert looks_like_user_hf_cache(sneaky)
    with pytest.raises(PermissionError):
        quarantine_project_file(sneaky, reason="test")


def test_existing_packed_local_training_unaffected(tmp_path: Path) -> None:
    from arzlm.data.packed import PackedTokenDataset, pack_documents
    from tests.helpers import cpu_experiment, write_mini_tokenizer

    packed = pack_documents([[1, 2, 3, 4]], eos_id=2)
    assert packed.dtype.kind == "u"
    write_mini_tokenizer(tmp_path)
    exp = cpu_experiment(tmp_path, seq_length=32, max_steps=1)
    assert exp.data_dir.joinpath("train.bin").is_file()
    ds = PackedTokenDataset(exp.data_dir / "train.bin", seq_length=32)
    assert len(ds) >= 1


def test_cli_data_security_audit_registered() -> None:
    from arzlm.cli import build_parser

    parser = build_parser()
    names: list[str] = []
    for action in parser._actions:
        if getattr(action, "choices", None):
            names.extend(action.choices)
    assert "data-security-audit" in names


def test_eval_split_paths_are_skipped() -> None:
    from arzlm.data.security.audit import is_eval_split_path

    assert is_eval_split_path("document/test-00000-of-00001.parquet")
    assert is_eval_split_path("section/validation-00000-of-00001.parquet")
    assert not is_eval_split_path("document/train-00000-of-00015.parquet")
    assert not is_eval_split_path("sample/10BT/013_00000.parquet")


def test_general_supplement_budget_covers_observed_1b_shortfall() -> None:
    from arzlm.data.security.audit import GENERAL_SUPPLEMENTS

    assert GENERAL_SUPPLEMENTS[0][3] >= 20_000_000_000


def test_science_fallback_and_general_supplement() -> None:
    fw = _meta(path="sample/10BT/013_00000.parquet", sha256="aa" * 32, size=100)
    math = RemoteFileMeta(
        dataset_id=FINEMATH_4PLUS.dataset_id,
        revision=FINEMATH_4PLUS.revision,
        path="finemath-4plus/train-00000-of-00064.parquet",
        size=10,
        sha256="bb" * 32,
        security_status="safe",
        av_status="safe",
    )
    pes2o = RemoteFileMeta(
        dataset_id=PES2O.dataset_id,
        revision=PES2O.revision,
        path="data/v2/train-00010-of-00020.json.gz",
        size=10,
        sha256="cc" * 32,
        security_status="unscanned",
        av_status="unscanned",
    )
    train = RemoteFileMeta(
        dataset_id="ccdv/arxiv-summarization",
        revision="main",
        path="document/train-00000-of-00015.parquet",
        size=100,
        sha256="dd" * 32,
        security_status="safe",
        av_status="safe",
    )
    heldout = RemoteFileMeta(
        dataset_id="ccdv/arxiv-summarization",
        revision="main",
        path="document/test-00000-of-00001.parquet",
        size=100,
        sha256="ee" * 32,
        security_status="safe",
        av_status="safe",
    )
    cosmo = RemoteFileMeta(
        dataset_id="HuggingFaceTB/smollm-corpus",
        revision="main",
        path="cosmopedia-v2/train-00000-of-00104.parquet",
        size=100,
        sha256="ff" * 32,
        security_status="safe",
        av_status="safe",
    )
    py = RemoteFileMeta(
        dataset_id=STACK_EDU.dataset_id,
        revision=STACK_EDU.revision,
        path="Python/train-00000-of-00005.parquet",
        size=10,
        sha256="11" * 32,
        security_status="safe",
        av_status="safe",
    )
    client = FakeHubClient([fw, math, pes2o, train, heldout, cosmo, py])
    lock = build_source_lock(client=client, denylist=_denylist())
    assert lock.sources["science"].dataset_id == "ccdv/arxiv-summarization"
    assert "document/train-00000-of-00015.parquet" in lock.sources["science"].files
    assert "document/test-00000-of-00001.parquet" not in lock.sources["science"].files
    assert lock.supplements
    assert lock.supplements[0].dataset_id == "HuggingFaceTB/smollm-corpus"
    report = run_source_security_audit(client=client, lock=lock, denylist=_denylist())
    assert report.domains["science"].passed
    assert report.domains["general"].passed
    assert report.passed


def test_list_fetchable_empty_locked_language_is_empty() -> None:
    py = RemoteFileMeta(
        dataset_id=STACK_EDU.dataset_id,
        revision=STACK_EDU.revision,
        path="Python/train-00000-of-00005.parquet",
        size=10,
        sha256="11" * 32,
        security_status="safe",
        av_status="safe",
    )
    lock = SourceLock(
        schema_version=1,
        corpus="t",
        created_at="x",
        policy={},
        sources={
            "code": LockedSource(
                domain="code",
                dataset_id=STACK_EDU.dataset_id,
                config=None,
                revision=STACK_EDU.revision,
                prefix="(per-language parquet)",
                files={
                    py.path: LockFileEntry(
                        path=py.path,
                        sha256=py.sha256,
                        security_status="safe",
                        av_status="safe",
                    )
                },
            )
        },
    )
    out = list_fetchable_parquet(
        STACK_EDU.dataset_id,
        STACK_EDU.revision,
        "CSharp/",
        denylist=[],
        lock=lock,
        client=FakeHubClient([py]),
    )
    assert out == []


def test_code_weights_drop_missing_languages() -> None:
    from arzlm.data.catalog import code_language_weights_from_paths

    weights = code_language_weights_from_paths(["Python/a.parquet", "C/b.parquet"])
    assert "CSharp" not in weights
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["Python"] > weights["C"]

