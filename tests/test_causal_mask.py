import torch

from arzlm.model.config import build_gpt, build_model_config


def test_causal_masking_prefix_invariance() -> None:
    """Tokens after position t must not change logits at positions <= t."""
    torch.manual_seed(1)
    model = build_gpt(build_model_config(), tie_embeddings=True)
    model.eval()
    prefix = torch.tensor([[11, 22, 33, 44]], dtype=torch.long)
    a = torch.cat([prefix, torch.tensor([[55]], dtype=torch.long)], dim=1)
    b = torch.cat([prefix, torch.tensor([[99]], dtype=torch.long)], dim=1)
    with torch.no_grad():
        logits_a = model(a)
        logits_b = model(b)
    # Compare the shared prefix positions 0..3
    torch.testing.assert_close(logits_a[:, :4], logits_b[:, :4], rtol=1e-4, atol=1e-4)
    assert not torch.allclose(logits_a[:, 4], logits_b[:, 4])
