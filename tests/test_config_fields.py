from dataclasses import fields

from litgpt.config import Config

from arzlm.model.config import ARZLM_100M_FIELDS, ARZLM_300M_FIELDS, build_model_config, validate_config_fields


def test_all_architecture_keys_are_litgpt_config_fields() -> None:
    known = {f.name for f in fields(Config)}
    assert set(ARZLM_100M_FIELDS).issubset(known)
    assert set(ARZLM_300M_FIELDS).issubset(known)


def test_unknown_field_is_rejected() -> None:
    try:
        validate_config_fields({"not_a_real_field": 1})
    except ValueError as exc:
        assert "not_a_real_field" in str(exc)
    else:
        raise AssertionError("unknown field was accepted")


def test_built_config_matches_target_shape() -> None:
    cfg = build_model_config()
    assert cfg.n_layer == 12
    assert cfg.n_embd == 768
    assert cfg.n_head == 12
    assert cfg.head_size == 64
    assert cfg.n_query_groups == 4
    assert cfg.intermediate_size == 2048
    assert cfg.mlp_class_name == "LLaMAMLP"
    assert cfg.norm_class_name == "RMSNorm"
    assert cfg.norm_qk is True
    assert cfg.rotary_percentage == 1.0
    assert cfg.bias is False
    assert cfg.attn_bias is False
    assert cfg.parallel_residual is False
    assert cfg.vocab_size == 32000
    assert cfg.padded_vocab_size == 32000
    assert cfg.block_size == 1024
    assert cfg.rope_n_elem == 64
