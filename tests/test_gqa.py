from litgpt.model import CausalSelfAttention

from arzlm.model.config import build_gpt, build_model_config


def test_gqa_qkv_projection_shape() -> None:
    config = build_model_config()
    expected = (config.n_head + 2 * config.n_query_groups) * config.head_size
    assert expected == (12 + 2 * 4) * 64 == 1280
    model = build_gpt(config, tie_embeddings=True)
    attn = model.transformer.h[0].attn
    assert isinstance(attn, CausalSelfAttention)
    assert attn.qkv.in_features == 768
    assert attn.qkv.out_features == expected
    assert attn.proj.in_features == config.n_head * config.head_size
    assert attn.proj.out_features == config.n_embd
    assert attn.norm_q is not None and attn.norm_k is not None
    assert tuple(attn.norm_q.weight.shape) == (config.head_size,)
    assert tuple(attn.norm_k.weight.shape) == (config.head_size,)
