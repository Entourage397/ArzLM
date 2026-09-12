# ArzLM

ArzLM-300M-Base is a **303,353,856**-parameter decoder-only language model trained from scratch on **6,000,214,016** tokens.

It is a **base / pretrained** model. It is **not** instruction tuned and it is **not** a chat model.

Weights: [Kymaris/ArzLM-300M-Base](https://huggingface.co/Kymaris/ArzLM-300M-Base)

This repository is source, configs, training recipe, inference code, and the local benchmark harness. It does **not** contain model binaries.

## Model card (facts)

| | |
| --- | --- |
| Name | ArzLM-300M-Base |
| Parameters | 303,353,856 (tied embeddings; counted after tying) |
| Architecture | LitGPT `GPT`, LLaMA-style sequential residual |
| Layers | 24 |
| `d_model` | 1024 |
| Query heads | 16 (`head_dim` 64) |
| KV heads | 4 (GQA) |
| MLP | SwiGLU (`LLaMAMLP`), intermediate size 2816 |
| Norm | RMSNorm + QK-Norm (`norm_qk: true`, `norm_qk_type: default`) |
| Position | RoPE, `rotary_percentage: 1.0`, `rope_base: 10000` |
| Biases | none |
| Vocab | 32,000 (`padding_multiple: 64`, no silent pad to 32256) |
| Context | **2048** |
| Tokenizer | `arzlm-stem-32k-v2`, byte-level BPE, `add_bos_token: false` |
| Precision (train) | BF16 (`bf16-true`) |
| Optimizer | Muon hybrid (Muon on 2D hidden matrices, AdamW on embeddings/norms) |
| Train tokens | 6,000,214,016 (45,778 optimizer steps × 131,072 tokens/step) |
| Hardware | 1× NVIDIA A100-SXM4-40GB on Modal |
| Final val loss | 2.6909 |
| Final val PPL | 14.75 (held-out packed mixture, **not** WikiText) |

Do not load this checkpoint with `AutoModelForCausalLM.from_pretrained`. The Hugging Face folder uses **ArzLM / LitGPT tensor names** (fused `qkv`, QK-Norm). The supported loader is in this repo.

## Limitations (observed)

This is a 300M-scale base model trained on 6B tokens. It is factually unreliable, weak at mathematics, can repeat, can emit template-like Cosmopedia-style tutorials, and can generate incorrect code. Context is 2048 tokens. It is not conversationally aligned. Qualitatively, greedy “The derivative of x^2 is” completed as `1/x^2` (wrong) while “The capital of France is” reached Paris and then repeated.

## Training data

Packed mixture **ArzLM-STEM-6B-v1**, unique packed quotas then sampled to 6.000B exposures at 55 / 20 / 15 / 10:

| Domain | Target mix | Unique packed train tokens | Sources |
| --- | ---: | ---: | --- |
| general | 0.55 | 3,465,019,392 | FineWeb-Edu `sample-10BT` (Hub-safe shard only) + SmolLM-corpus Cosmopedia-v2 |
| math | 0.20 | 1,259,995,136 | FineMath-4+ |
| code | 0.15 | 945,029,120 | Stack-Edu → Software Heritage, permissive licenses, `int_score` ≥ 4 then ≥ 3 fill |
| science | 0.10 | 630,063,104 | peS2o v2 s2orc full text |

Exact file pins: `data/security/source-lock-6b.json`. Nemotron-CC-Math was **not** used.

Domain validation CE after 6B (same packed val shards, not a public benchmark):

| Domain | val loss |
| --- | ---: |
| general | 3.300 |
| math | 1.800 |
| code | 1.472 |
| science | 2.955 |
| stem (math+code+science) | 1.947 |
| overall | 2.691 |

## Training setup

- Config: `configs/arzlm-300m-final-6b.yaml` + `configs/model-arzlm-300m.yaml`
- Effective batch: 64 × 2048 = 131,072 tokens / update (micro-batch 8, accumulation 8)
- Muon lr 0.05, momentum 0.95, Newton–Schulz 5, Jordan scale, wd 0.1
- AdamW lr 6e-4, β=(0.9, 0.95), wd 0.1 on non-Muon params
- Cosine to `min_lr` 6e-5, warmup 800 steps (~104.9M tokens)
- `torch.compile` enabled on the A100; Flash SDPA
- Seed 42
- Wall-clock ~8.75 h at ~62.8k tok/s steady state after compile warmup

## Install

Python 3.12, PyTorch CUDA wheel, then this package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[eval]"
```

Download weights separately (not in git):

```bash
hf download Kymaris/ArzLM-300M-Base --local-dir ./ArzLM-300M-Base
```

## Inference (supported)

```python
from arzlm.infer import generate, load_inference_checkpoint

loaded = load_inference_checkpoint("./ArzLM-300M-Base/litgpt", device="cuda", dtype="bf16")
print(loaded.parameters, loaded.config.block_size)
print(generate(loaded, "The capital of France is", max_new_tokens=32)["completion"])
```

CLI:

```bash
python -m arzlm verify-checkpoint --checkpoint ./ArzLM-300M-Base/litgpt
python -m arzlm generate --checkpoint ./ArzLM-300M-Base/litgpt --max-new-tokens 32
```

`verify-checkpoint` checks parameter count **303,353,856**, context **2048**, `strict=True` load, and refuses optimizer-bearing files.

## Benchmarks

Local harness (EleutherAI `lm-eval==0.4.13` plus this adapter):

```bash
python eval/adapter_selftest.py --checkpoint ./ArzLM-300M-Base/litgpt
python eval/run_suite.py --arzlm-checkpoint ./ArzLM-300M-Base/litgpt --sanity
python eval/run_suite.py --arzlm-checkpoint ./ArzLM-300M-Base/litgpt --out benchmark_results.json
python eval/performance.py --kind arzlm --checkpoint ./ArzLM-300M-Base/litgpt --out perf-arzlm.json
```

Settings are pinned in `eval/settings.yaml` and are **identical** for every model. Chat templates are off. Primary table in `BENCHMARK_REPORT.md` is local-only; published leaderboard scores are not mixed in.

Fair-compute intended baseline: `cerebras/Cerebras-GPT-256M` (~256M params, ~5.12B tokens, 2048 context). As of 2026-09-12 that Hub repo returns **404** for an authenticated download. The suite still *tries* it and records the error rather than substituting another model silently.

Modern heavily-trained comparisons (much larger pretraining budgets): `google/gemma-3-270m` (base, gated Gemma license) and `HuggingFaceTB/SmolLM2-360M` (base).

HumanEval / MBPP execution is optional and must be sandboxed (`unshare -n`, no network). If a safe executor cannot be used, those scores are omitted rather than run on the unrestricted host.

## License

Apache License 2.0 for original code and for the released weights, with dataset attribution in `NOTICE`. See `docs/licensing.md` for why this is not an invented default.

## Historical 100M work

Earlier laptop-scale 100M FineWeb-Edu experiments remain under `docs/baselines/` and `configs/model-arzlm-100m.yaml`. They are not this 300M release.
