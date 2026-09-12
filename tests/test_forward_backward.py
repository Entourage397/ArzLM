import torch

from arzlm.model.config import build_gpt, build_model_config


def test_forward_and_backward_finite() -> None:
    torch.manual_seed(0)
    model = build_gpt(build_model_config(), tie_embeddings=True)
    model.train()
    x = torch.randint(0, 32000, (2, 8), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (2, 8, 32000)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32000), x[:, 1:].reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
