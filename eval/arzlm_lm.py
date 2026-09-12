"""lm-evaluation-harness adapter for ArzLM LitGPT checkpoints.

Register with `--model arzlm` after `--include_path eval`.
Does not modify model weights. Does not use chat templates.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from arzlm.infer import load_inference_checkpoint


def _register():
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model

    @register_model("arzlm")
    class ArzLMLM(LM):
        def __init__(
            self,
            checkpoint: str,
            device: str | None = None,
            dtype: str | None = "bf16",
            batch_size: int | str = 1,
            max_length: int = 2048,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            self.loaded = load_inference_checkpoint(
                checkpoint,
                device=device,
                dtype=None if dtype in {None, "auto"} else str(dtype),
            )
            self._max_length = int(max_length)
            self._batch_size = int(batch_size)
            self._rank = 0
            self._world_size = 1

        @property
        def eot_token_id(self) -> int:
            eos = self.loaded.tokenizer.eos_id
            if eos is None:
                raise RuntimeError("tokenizer has no eos_id")
            return int(eos)

        @property
        def max_length(self) -> int:
            return min(self._max_length, int(self.loaded.config.block_size))

        @property
        def max_gen_toks(self) -> int:
            return 256

        @property
        def batch_size(self) -> int:
            return self._batch_size

        @property
        def device(self):
            return self.loaded.device

        @property
        def rank(self) -> int:
            return self._rank

        @property
        def world_size(self) -> int:
            return self._world_size

        def tok_encode(self, string: str, add_special_tokens: bool = False) -> list[int]:
            # Training and export use add_bos_token=false. Never inject BOS here.
            ids = self.loaded.tokenizer.encode(string, bos=False, eos=False)
            return ids.detach().cpu().tolist()

        def tok_decode(self, tokens: list[int]) -> str:
            tensor = torch.tensor(tokens, dtype=torch.long)
            return self.loaded.tokenizer.decode(tensor)

        def _forward_logits(self, idx: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                logits = self.loaded.model(idx)
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite logits in ArzLMLM")
            return logits

        def _prepare_ll(self, context: str, continuation: str) -> tuple[list[int], list[int], list[int]] | None:
            ctx_ids = self.tok_encode(context) if context else [self.eot_token_id]
            cont_ids = self.tok_encode(continuation)
            if not cont_ids:
                return None
            seq = ctx_ids + cont_ids
            if len(seq) > self.max_length:
                overflow = len(seq) - self.max_length
                if overflow >= len(ctx_ids):
                    seq = seq[-(self.max_length) :]
                    ctx_ids = seq[:1]
                    cont_ids = seq[1:]
                else:
                    ctx_ids = ctx_ids[overflow:]
                    seq = ctx_ids + cont_ids
            return ctx_ids, cont_ids, seq

        def _score_row(
            self,
            logprobs: torch.Tensor,
            greedy: torch.Tensor,
            target: torch.Tensor,
            ctx_len: int,
            cont_len: int,
        ) -> tuple[float, bool]:
            start = ctx_len - 1
            end = start + cont_len
            slice_tgt = target[start:end]
            slice_log = logprobs[start:end]
            slice_greedy = greedy[start:end]
            if slice_tgt.numel() != cont_len:
                raise RuntimeError(
                    f"continuation align error: scored {slice_tgt.numel()} vs {cont_len}"
                )
            gathered = slice_log.gather(-1, slice_tgt.unsqueeze(-1)).squeeze(-1)
            ll = float(gathered.sum().item())
            if not math.isfinite(ll):
                raise RuntimeError("non-finite loglikelihood")
            return ll, bool(torch.equal(slice_greedy, slice_tgt))

        def _loglikelihood_one(self, context: str, continuation: str) -> tuple[float, bool]:
            prepared = self._prepare_ll(context, continuation)
            if prepared is None:
                return 0.0, True
            ctx_ids, cont_ids, seq = prepared
            inp = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
            logits = self._forward_logits(inp)[0, :-1]
            target = inp[0, 1:]
            logprobs = F.log_softmax(logits.float(), dim=-1)
            greedy = logits.argmax(dim=-1)
            return self._score_row(logprobs, greedy, target, len(ctx_ids), len(cont_ids))

        def loglikelihood(self, requests):
            out: list[tuple[float, bool] | None] = [None] * len(requests)
            pending: list[tuple[int, list[int], list[int], list[int]]] = []
            for i, req in enumerate(requests):
                context, continuation = req.args
                prepared = self._prepare_ll(context, continuation)
                if prepared is None:
                    out[i] = (0.0, True)
                    continue
                ctx_ids, cont_ids, seq = prepared
                pending.append((i, ctx_ids, cont_ids, seq))
            bs = max(1, self._batch_size)
            pad_id = int(getattr(self.loaded.tokenizer, "pad_id", None) or 3)
            for start in range(0, len(pending), bs):
                chunk = pending[start : start + bs]
                max_t = max(len(seq) for _, _, _, seq in chunk)
                batch = torch.full((len(chunk), max_t), pad_id, dtype=torch.long, device=self.device)
                for row, (_, _, _, seq) in enumerate(chunk):
                    batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                logits = self._forward_logits(batch)[:, :-1]
                logprobs = F.log_softmax(logits.float(), dim=-1)
                greedy = logits.argmax(dim=-1)
                target = batch[:, 1:]
                for row, (orig_i, ctx_ids, cont_ids, _seq) in enumerate(chunk):
                    out[orig_i] = self._score_row(
                        logprobs[row], greedy[row], target[row], len(ctx_ids), len(cont_ids)
                    )
            return out

        def loglikelihood_rolling(self, requests):
            out = []
            max_len = self.max_length
            for req in requests:
                (string,) = req.args
                ids = self.tok_encode(string)
                if not ids:
                    out.append(0.0)
                    continue
                total = 0.0
                predicted = 0
                while predicted < len(ids):
                    if predicted == 0:
                        take = min(max_len - 1, len(ids))
                        window = [self.eot_token_id] + ids[:take]
                        pred = ids[:take]
                    else:
                        take = min(max_len - 1, len(ids) - predicted)
                        window = ids[predicted - 1 : predicted + take]
                        pred = ids[predicted : predicted + take]
                    inp = torch.tensor(window, dtype=torch.long, device=self.device).unsqueeze(0)
                    logits = self._forward_logits(inp)[0, :-1]
                    target = inp[0, 1:]
                    logprobs = F.log_softmax(logits.float(), dim=-1)
                    scored = target[-len(pred) :]
                    gathered = logprobs[-len(pred) :].gather(-1, scored.unsqueeze(-1)).squeeze(-1)
                    total += float(gathered.sum().item())
                    predicted += take
                out.append(total)
            return out

        def generate_until(self, requests):
            generations = []
            for req in requests:
                context, gen_kwargs = req.args
                until = gen_kwargs.get("until") or []
                if isinstance(until, str):
                    until = [until]
                max_gen = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
                ctx_ids = self.tok_encode(context)
                if len(ctx_ids) >= self.max_length:
                    ctx_ids = ctx_ids[-(self.max_length - 1) :]
                x = torch.tensor(ctx_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                decoded_ctx = self.tok_decode(ctx_ids)
                new_ids: list[int] = []
                with torch.no_grad():
                    for _ in range(max_gen):
                        if x.shape[1] >= self.max_length:
                            break
                        logits = self._forward_logits(x)
                        nxt = int(logits[0, -1].argmax().item())
                        new_ids.append(nxt)
                        x = torch.cat(
                            [x, torch.tensor([[nxt]], device=self.device, dtype=torch.long)],
                            dim=1,
                        )
                        if nxt == self.eot_token_id:
                            break
                        text = self.tok_decode(ctx_ids + new_ids)
                        rest = text[len(decoded_ctx) :]
                        if any(stop and stop in rest for stop in until):
                            break
                text = self.tok_decode(ctx_ids + new_ids)
                rest = text[len(decoded_ctx) :]
                for stop in until:
                    if stop:
                        rest = rest.split(stop)[0]
                generations.append(rest)
            return generations

    return ArzLMLM


ArzLMLM = _register()
