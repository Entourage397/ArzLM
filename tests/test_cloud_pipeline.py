from pathlib import Path
import inspect
import json

import pytest

from arzlm.cloud.pipeline import (
    assert_a100_40gb_identity,
    clear_lease,
    iso_age_s,
    lease_fresh,
    named_lock_is_held,
    trainer_heartbeat_fresh,
    write_lease,
)
from arzlm.cloud.runtime import maybe_commit_pack, set_after_pack_checkpoint, stop_requested, write_status
from arzlm.paths import REPO_ROOT
from arzlm.training.report import utc_now


def test_named_lock_blocks_other_holder_within_ttl() -> None:
    now = 1_000.0
    existing = {"holder": "ta-alive", "t": now - 60}
    assert named_lock_is_held(existing, now=now, ttl_s=1800, holder="ta-other") is True
    assert named_lock_is_held(existing, now=now, ttl_s=1800, holder="ta-alive") is False
    stale = {"holder": "ta-dead", "t": now - 4000}
    assert named_lock_is_held(stale, now=now, ttl_s=1800, holder="ta-other") is False
    assert named_lock_is_held(None, now=now, ttl_s=1800, holder="ta-other") is False


def test_a100_40gb_identity_rejects_h100_and_80gb() -> None:
    assert_a100_40gb_identity("NVIDIA A100-SXM4-40GB", 40.0)
    with pytest.raises(RuntimeError, match="A100 40GB"):
        assert_a100_40gb_identity("NVIDIA A100-SXM4-80GB", 80.0)
    with pytest.raises(RuntimeError, match="A100 40GB"):
        assert_a100_40gb_identity("NVIDIA H100 80GB HBM3", 80.0)
    with pytest.raises(RuntimeError, match="unexpected GPU name"):
        assert_a100_40gb_identity("NVIDIA H100", 40.0)
    with pytest.raises(RuntimeError, match="too small"):
        assert_a100_40gb_identity("NVIDIA A100", 16.0)


def test_lease_and_trainer_heartbeat(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARZLM_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    from arzlm.cloud import pipeline as P

    monkeypatch.setattr(P, "artifacts_root", lambda: Path(tmp_path / "artifacts"))
    write_lease("train")
    assert lease_fresh("train", max_age_s=1800)
    assert trainer_heartbeat_fresh(max_age_s=1800)
    clear_lease("train")
    assert not lease_fresh("train", max_age_s=1800)

    status_dir = P.train_out_dir()
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "status.json").write_text(
        '{"phase": "TRAIN_300M_6B", "heartbeat_at": "%s", "budget_complete": false}\n' % utc_now(),
        encoding="utf-8",
    )
    assert trainer_heartbeat_fresh(max_age_s=1800)
    (status_dir / "status.json").write_text(
        '{"phase": "TRAIN_SEGMENT_EXIT", "heartbeat_at": "%s", "exit_reason": "max_time"}\n' % utc_now(),
        encoding="utf-8",
    )
    assert trainer_heartbeat_fresh(max_age_s=1800) is False


def test_stop_requested_desired_state(tmp_path, monkeypatch) -> None:
    desired = tmp_path / "desired.json"
    monkeypatch.setenv("ARZLM_DESIRED_STATE_PATH", str(desired))
    monkeypatch.setenv("ARZLM_STOP_PATH", str(tmp_path / "STOP_REQUESTED"))
    assert stop_requested(tmp_path) is False
    desired.write_text('{"desired_state": "STOP_REQUESTED"}\n', encoding="utf-8")
    assert stop_requested(tmp_path) is True


def test_pack_commit_hook_throttles(monkeypatch) -> None:
    calls: list[dict] = []
    set_after_pack_checkpoint(lambda payload: calls.append(payload), min_interval_s=60)
    maybe_commit_pack({"a": 1})
    maybe_commit_pack({"b": 2})
    maybe_commit_pack({"c": 3}, force=True)
    assert len(calls) == 2
    assert calls[0]["a"] == 1
    assert calls[1]["c"] == 3
    set_after_pack_checkpoint(None)


def test_write_status_roundtrip(tmp_path, monkeypatch) -> None:
    seen: list[dict] = []
    monkeypatch.setattr("arzlm.cloud.runtime._after_status", lambda payload: seen.append(payload))
    path = write_status(tmp_path, {"training_tokens": 131072, "latest_loss": 5.2, "finite": True})
    assert path.is_file()
    assert seen[0]["training_tokens"] == 131072


def test_modal_app_is_single_a100_40gb() -> None:
    src = (REPO_ROOT / "modal" / "arzlm_300m.py").read_text(encoding="utf-8")
    assert 'gpu="A100-40GB"' in src
    assert "max_containers=1" in src
    assert "H100" not in src
    assert "A100-80GB" not in src
    assert "finally:" in src
    assert "PACK_LOCK_TTL_S = 1800.0" in src
    assert src.count('gpu="A100-40GB"') == 1
    assert "evaluate_probation" in src
    assert "ARZLM_PROBATION_TOKENS" in src
    assert "train-pending" in src
    assert "corpus_validate_blocked" in src
    assert "parameter_count_mismatch" in src
    assert "packed-exact-dedupe" in src or "packed_exact_dedupe" in src
    assert "MARKER_NAME" in src
    from arzlm.cloud import pipeline as P
    from arzlm.cloud.pipeline import decode_category_samples
    import inspect as _inspect

    decode_src = _inspect.getsource(decode_category_samples)
    assert "torch.from_numpy" in decode_src
    assert "data[:256]" in decode_src
    eval_src = _inspect.getsource(P.run_evaluate)
    export_src = _inspect.getsource(P.run_export)
    smoke_src = _inspect.getsource(P.run_final_smoke)
    load_src = _inspect.getsource(P._load_tied_gpt)
    assert "greedy_complete" in eval_src
    assert "_load_tied_gpt" in eval_src
    assert "build_gpt" in load_src
    assert "tie_embeddings=True" in load_src
    assert "strict=True" in load_src
    assert "torch.tensor([tok.encode" not in eval_src
    assert "torch.tensor([tok.encode" not in smoke_src
    assert "optimizer_state_included" in export_src
    assert "LITGPT_INFERENCE" in export_src
    assert "refusing to overwrite source checkpoint" in export_src
    assert "posttrain_alive" in src
    assert "train_already_complete" in src
    assert "token_budget_complete" in src
    assert "VALIDATION_POLICY_VERSION" in src
    assert iso_age_s(None) is None


def test_blocked_validate_persists_last_error_until_policy_changes(tmp_path, monkeypatch) -> None:
    from arzlm.cloud import pipeline as P
    from arzlm.data.corpus_validate import VALIDATION_POLICY_VERSION

    monkeypatch.setattr(P, "pipeline_dir", lambda: tmp_path)
    monkeypatch.setattr(P, "state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(P, "desired_state_path", lambda: tmp_path / "desired.json")
    state = {
        "schema_version": 1,
        "phases": {},
        "desired_state": "RUNNING",
        "last_error": None,
    }
    P.mark_phase(
        state,
        "CORPUS_VALIDATE",
        "blocked",
        error="CorpusValidationError: science packed document hash repeats",
        error_class="CorpusValidationError",
        policy_version=VALIDATION_POLICY_VERSION,
    )
    assert state["last_error"].startswith("CorpusValidationError")
    assert P.phase_is_blocked(state, "CORPUS_VALIDATE", policy_version=VALIDATION_POLICY_VERSION)
    assert P.phase_is_blocked(state, "CORPUS_VALIDATE", policy_version=VALIDATION_POLICY_VERSION + 1) is False
    P.mark_phase(state, "CORPUS_VALIDATE", "complete", train_tokens=1)
    assert state["last_error"] is None


def test_cost_projection_continue_and_abort() -> None:
    from arzlm.cloud.cost import TARGET_TOKENS, classify_probation_failure, project_complete_build

    remaining = TARGET_TOKENS - 16_777_216
    ok = project_complete_build(
        current_metered_usd=32.0,
        remaining_preprocess_usd=0.0,
        remaining_train_tokens=remaining,
        steady_tokens_per_sec=55_000.0,
    )
    assert ok["decision_continue"] is True
    assert ok["projected_usd"] <= 98.0
    slow = project_complete_build(
        current_metered_usd=32.0,
        remaining_preprocess_usd=0.0,
        remaining_train_tokens=remaining,
        steady_tokens_per_sec=30_000.0,
    )
    assert slow["decision_continue"] is False
    assert slow["projected_usd"] > 98.0
    assert classify_probation_failure(
        finite=True,
        checkpoint_ok=True,
        resume_ok=True,
        steady_tokens_per_sec=30_000.0,
        projected=slow,
    ) == "projected_over_budget"
    assert classify_probation_failure(
        finite=False,
        checkpoint_ok=True,
        resume_ok=True,
        steady_tokens_per_sec=55_000.0,
        projected=ok,
    ) == "non_finite_loss"


def test_checkpoint_reload_counts_tied_embeddings() -> None:
    from arzlm.cloud import pipeline as P
    from arzlm.model.config import EXPECTED_TRAINABLE_PARAMETERS_300M

    src = inspect.getsource(P.verify_checkpoint_reloadable)
    assert "tie_embeddings=True" in src
    assert "build_gpt" in src
    assert "count_parameters" in src
    # Untied GPT counts lm_head + wte separately (32000 * 1024).
    assert 336_121_856 - EXPECTED_TRAINABLE_PARAMETERS_300M == 32_000 * 1024


def test_write_packaged_tokenizer_sets_2048_without_rewriting_vocab(tmp_path) -> None:
    from arzlm.cloud.pipeline import _write_packaged_tokenizer

    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer.json").write_text('{"version":"1.0","model":{"type":"BPE","vocab":{}}}\n', encoding="utf-8")
    (src / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"PreTrainedTokenizerFast","model_max_length":1024,"add_bos_token":false}\n',
        encoding="utf-8",
    )
    dest = tmp_path / "dest"
    meta = _write_packaged_tokenizer(src, dest, model_max_length=2048)
    assert meta["model_max_length"] == 2048
    cfg = json.loads((dest / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert cfg["model_max_length"] == 2048
    assert (dest / "tokenizer.json").read_text(encoding="utf-8") == (src / "tokenizer.json").read_text(encoding="utf-8")
    src_cfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert src_cfg["model_max_length"] == 1024


def test_preprocess_estimate_uses_measured_hours(tmp_path, monkeypatch) -> None:
    from arzlm.cloud import pipeline as P
    from arzlm.cloud.cost import PACK_USD_PER_CONTAINER_HOUR

    monkeypatch.setattr(P, "corpus_dir", lambda: tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "math.json").write_text('{"status": "complete", "elapsed_s": 3600, "tokens_train": 1}\n', encoding="utf-8")
    (state / "general.json").write_text('{"status": "complete", "elapsed_s": 7200, "tokens_train": 1}\n', encoding="utf-8")
    (state / "code.json").write_text('{"status": "complete", "elapsed_s": 0, "tokens_train": 1}\n', encoding="utf-8")
    (state / "science.json").write_text('{"status": "complete", "elapsed_s": 0, "tokens_train": 1}\n', encoding="utf-8")
    assert P.estimate_preprocess_usd() == pytest.approx(3.0 * PACK_USD_PER_CONTAINER_HOUR)
    assert P.estimate_remaining_pack_usd() == 0.0
