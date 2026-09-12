"""Prompt encoding for LitGPT's Tokenizer, which already returns a 1-D tensor.

`tok.encode()` returns `torch.int` rank-1. Wrapping that tensor in
`torch.tensor([...])` raises TypeError and was the posttrain crash.
Training uses `add_bos_token: false`; prompts therefore pass `bos=False`.
"""

from __future__ import annotations

from typing import Any

import torch


def encode_prompt_batch(
    tok: Any,
    prompt: str,
    *,
    device: torch.device | None = None,
    bos: bool = False,
    eos: bool = False,
) -> torch.Tensor:
    """Return int64 token ids with shape `[1, seq]` for `GPT.forward`."""
    encoded = tok.encode(prompt, bos=bos, eos=eos, device=device)
    if isinstance(encoded, torch.Tensor):
        ids = encoded.detach()
    else:
        ids = torch.as_tensor(encoded, device=device)
    if ids.ndim == 2:
        if ids.shape[0] != 1:
            raise ValueError(f"encode() returned batch {tuple(ids.shape)}, expected 1")
        ids = ids[0]
    elif ids.ndim == 0:
        ids = ids.view(1)
    elif ids.ndim != 1:
        raise ValueError(f"encode() returned rank-{ids.ndim} tensor {tuple(ids.shape)}")
    ids = ids.to(dtype=torch.long)
    if device is not None:
        ids = ids.to(device)
    return ids.unsqueeze(0).contiguous()


def decode_token_ids(tok: Any, ids: torch.Tensor | list[int]) -> str:
    """Decode a 1-D sequence. LitGPT `decode` requires a tensor with `.ndim`."""
    if isinstance(ids, torch.Tensor):
        tensor = ids.detach().cpu()
        if tensor.ndim == 2:
            if tensor.shape[0] != 1:
                raise ValueError(f"decode batch {tuple(tensor.shape)}, expected 1")
            tensor = tensor[0]
        elif tensor.ndim == 0:
            tensor = tensor.view(1)
    else:
        tensor = torch.as_tensor(ids, dtype=torch.long)
    return tok.decode(tensor)


def greedy_complete(
    model: torch.nn.Module,
    tok: Any,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Deterministic greedy continuation. Stops at EOS when the tokenizer has one."""
    x = encode_prompt_batch(tok, prompt, device=device, bos=False, eos=False)
    prompt_len = int(x.shape[1])
    eos_id = getattr(tok, "eos_id", None)
    last_logits = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            last_logits = model(x)
            if not torch.isfinite(last_logits).all():
                raise RuntimeError("NaN/Inf logits during generation")
            nxt = last_logits[:, -1].argmax(dim=-1, keepdim=True).to(dtype=torch.long)
            x = torch.cat([x, nxt], dim=1)
            if eos_id is not None and int(nxt.item()) == int(eos_id):
                break
    text = decode_token_ids(tok, x[0])
    new_ids = x[0, prompt_len:].detach().cpu().tolist()
    return {
        "prompt": prompt,
        "completion": text,
        "prompt_tokens": prompt_len,
        "new_tokens": new_ids,
        "stopped_at_eos": bool(eos_id is not None and new_ids and int(new_ids[-1]) == int(eos_id)),
    }
