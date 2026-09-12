"""Durable engineering rules for ArzLM-100M.

This is a research-quality training repository, not a demo. Follow these rules
in every change.

## Verify external APIs against installed source

Do not assume LitGPT, Lightning, or PyTorch APIs from memory. Read the
installed package under `.venv` (or the current environment) and confirm every
config field against `litgpt.config.Config`, `litgpt.args.TrainArgs`, and the
current `litgpt.pretrain` / `litgpt.model` implementation before changing
architecture or training behavior.

Unknown LitGPT config keys are a hard error. Missing keys must not be
silently dropped.

## Never silently change architecture

The baseline is declared in `configs/model-arzlm-100m.yaml` and
`src/arzlm/model/config.py`. Changing depth, width, heads, GQA groups, MLP
size, vocab, RoPE, QK-Norm, tying, or residual style requires:

1. An explicit explanation in `docs/decisions.md`
2. A printed, programmatic parameter count
3. Regenerated tests if shapes change

Do not "fix" a parameter count by quietly padding vocab or widening a layer.

## Never start expensive or long training without explicit instruction

Do not launch 50M-token, 100M-token, or multi-hour runs unless the user
explicitly asks. Stage 0–2 (tests, smoke, 1M overfit, 10M benchmark) are the
default ceiling. Configs for 50M/100M exist so the next command is ready, not
so it can be started opportunistically.

## Benchmark changes rather than assuming they improve speed

Do not add FSDP, DeepSpeed, bitsandbytes, gradient checkpointing, extra
FlashAttention packages, or compile flags because they are fashionable.
Measure tokens/sec, step time, and peak VRAM on this machine (RTX 4070 Laptop
8GB) with `arzlm bench` before and after. Keep the faster stable configuration.

## Maintain reproducibility

Pin seeds in experiment YAML. Packed data preparation must take a seed, a
fixed HuggingFace FineWeb-Edu config (`sample-10BT`, never the full corpus),
and an exact token budget. Record tokenizer path, dataset config, and token
counts in `meta.json`. Resume must load model, optimizer, and step counters.

## Run tests after meaningful changes

After model, data, tokenizer, or trainer edits, run:

```
.venv\Scripts\python -m pytest tests -q
```

GPU tests are extra; CPU tests must still cover parameter count, forward,
backward, causal mask, GQA shapes, tokenizer roundtrip, packed EOS
boundaries, checkpoint save/resume, determinism, and finite loss.

## Document experimental results

Write measured numbers in `docs/experiments.md` or the run's
`train_result.json`. Do not claim speedups, loss improvements, or VRAM
savings without an artifact. Candidate ideas (Muon, modded-nanoGPT tricks)
stay in `docs/experiments.md` until someone implements them as an A/B, not
in the baseline trainer.
"""
