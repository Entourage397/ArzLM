from dataclasses import fields

from litgpt.config import Config

from arzlm.model.config import (
    ARZLM_300M_FIELDS,
    ARZLM_300M_NAME,
    EXPECTED_TRAINABLE_PARAMETERS_300M,
    TARGET_PARAMETERS_300M,
    build_gpt,
    build_model_config,
    count_parameters,
    formula_parameter_count,
    load_model_config,
)
from arzlm.paths import CONFIGS_DIR


def test_300m_fields_are_litgpt_config_keys() -> None:
    known = {f.name for f in fields(Config)}
    assert set(ARZLM_300M_FIELDS).issubset(known)


def test_300m_parameter_count() -> None:
    config = load_model_config(CONFIGS_DIR / "model-arzlm-300m.yaml")
    assert config.name == ARZLM_300M_NAME
    assert config.n_layer == 24
    assert config.n_embd == 1024
    assert config.n_head == 16
    assert config.n_query_groups == 4
    assert config.head_size == 64
    assert config.intermediate_size == 2816
    assert config.block_size == 2048
    assert config.padded_vocab_size == 32000
    counts = count_parameters(config, tie_embeddings=True)
    assert counts["trainable"] == counts["formula"] == formula_parameter_count(config, tie_embeddings=True)
    assert counts["trainable"] == EXPECTED_TRAINABLE_PARAMETERS_300M
    rel = abs(counts["trainable"] - TARGET_PARAMETERS_300M) / TARGET_PARAMETERS_300M
    assert rel < 0.05
    model = build_gpt(config, tie_embeddings=True)
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    attn = model.transformer.h[0].attn
    expected_qkv = (config.n_head + 2 * config.n_query_groups) * config.head_size
    assert attn.qkv.out_features == expected_qkv == 1536
    assert config.rope_n_elem == 64


def test_100m_default_is_unchanged() -> None:
    cfg = build_model_config()
    assert cfg.name == "ArzLM-100M"
    assert cfg.n_layer == 12
    assert cfg.block_size == 1024
