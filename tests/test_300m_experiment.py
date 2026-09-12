from arzlm.data.catalog import (
    MIXTURE_TARGET,
    TOKEN_EXPOSURES_6B,
    TOKENS_PER_OPT_STEP_300M,
    TRAIN_TOKEN_QUOTA_6B,
    TRAIN_TOKEN_TOTAL_6B,
)
from arzlm.data.security.audit import GENERAL_SUPPLEMENTS, GENERAL_SUPPLEMENTS_6B
from arzlm.data.stem_builder import STEM_PRESETS
from arzlm.paths import CONFIGS_DIR
from arzlm.cloud.pipeline import iso_age_s
from arzlm.training.config import load_experiment_config
from arzlm.training.report import utc_now


def test_6b_quotas_and_mixture() -> None:
    assert TRAIN_TOKEN_TOTAL_6B == sum(TRAIN_TOKEN_QUOTA_6B.values())
    assert TOKEN_EXPOSURES_6B == 45_778 * TOKENS_PER_OPT_STEP_300M
    assert abs(TRAIN_TOKEN_QUOTA_6B["general"] / TRAIN_TOKEN_TOTAL_6B - MIXTURE_TARGET["general"]) < 0.002
    assert abs(TRAIN_TOKEN_QUOTA_6B["math"] / TRAIN_TOKEN_TOTAL_6B - MIXTURE_TARGET["math"]) < 0.002
    assert abs(TRAIN_TOKEN_QUOTA_6B["code"] / TRAIN_TOKEN_TOTAL_6B - MIXTURE_TARGET["code"]) < 0.002
    assert abs(TRAIN_TOKEN_QUOTA_6B["science"] / TRAIN_TOKEN_TOTAL_6B - MIXTURE_TARGET["science"]) < 0.002
    assert "6b" in STEM_PRESETS
    assert STEM_PRESETS["6b"]["train"] == TRAIN_TOKEN_QUOTA_6B


def test_6b_supplements_expand_existing_family() -> None:
    assert GENERAL_SUPPLEMENTS[0][3] >= 20_000_000_000
    prefixes = {row[1] for row in GENERAL_SUPPLEMENTS_6B}
    datasets = {row[0] for row in GENERAL_SUPPLEMENTS_6B}
    assert prefixes == {"cosmopedia-v2/", "fineweb-edu-dedup/"}
    assert datasets == {"HuggingFaceTB/smollm-corpus"}
    assert all(row[3] >= 24_000_000_000 for row in GENERAL_SUPPLEMENTS_6B)


def test_300m_experiment_yaml_loads() -> None:
    exp = load_experiment_config(CONFIGS_DIR / "arzlm-300m-final-6b.yaml")
    assert exp.model.name == "ArzLM-300M"
    assert exp.model.block_size == 2048
    assert exp.train.max_seq_length == 2048
    assert exp.train.global_batch_size == 64
    assert exp.train.micro_batch_size == 8
    assert exp.train.max_tokens == TOKEN_EXPOSURES_6B
    assert exp.train.lr_warmup_steps == 800
    assert exp.train.max_time == 64800.0
    assert exp.optimizer["name"] == "muon_hybrid"
    assert exp.resume == "auto"
    assert exp.compile == "true"
    cloud = exp.raw.get("cloud") or {}
    assert cloud.get("projected_continue_max") == 98.0
    assert cloud.get("hard_usage_limit") == 100.0
    assert cloud.get("probation_tokens") == 16_777_216
    assert int((exp.raw.get("checkpoint") or {}).get("first_checkpoint_tokens") or 0) == 16_777_216


def test_iso_age_parses_utc_now() -> None:
    age = iso_age_s(utc_now())
    assert age is not None
    assert 0 <= age < 5
    assert iso_age_s(None) is None
    assert iso_age_s("not-a-timestamp") is None
