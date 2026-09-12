from arzlm.model.config import (
    EXPECTED_TRAINABLE_PARAMETERS,
    TARGET_PARAMETERS,
    build_gpt,
    build_model_config,
    count_parameters,
    formula_parameter_count,
)


def test_parameter_count_matches_formula_and_target() -> None:
    config = build_model_config()
    counts = count_parameters(config, tie_embeddings=True)
    assert counts["trainable"] == counts["formula"] == formula_parameter_count(config, tie_embeddings=True)
    assert counts["trainable"] == EXPECTED_TRAINABLE_PARAMETERS
    rel = abs(counts["trainable"] - TARGET_PARAMETERS) / TARGET_PARAMETERS
    assert rel < 0.05
    # Tied embeddings must actually share storage.
    model = build_gpt(config, tie_embeddings=True)
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print(f"trainable={counts['trainable']}")
