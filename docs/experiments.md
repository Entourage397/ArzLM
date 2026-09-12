# Experiments

Baseline is frozen at git tag **`baseline-v0`** (`dc4f53bb`). Do not rewrite `docs/baselines/baseline-v0/`.

## Completed

| Stage | Intent | Command | Result |
| --- | --- | --- | --- |
| 0 tests | Unit + CPU/GPU smoke tests | `.venv\Scripts\python -m pytest tests -q` | **14 passed** in ~43s on baseline-v0; **18 passed** after Muon tests |
| 0 smoke | 5 optimizer steps | `arzlm train --config configs/train-smoke.yaml` | loss 10.60 → 8.48, val 8.17, finite |
| bench | Micro-batch sweep, 5 warmup + 10 measure | `arzlm bench --data-dir data/prepared/tiny --compile false` | **best micro-batch=2**, 19,060 tok/s, 2.59 GB allocated |
| 1 | ~1M-token overfit on FineWeb-Edu packed subset | `arzlm train --config configs/train-overfit-1m.yaml` | train 8.49 → 5.61, val 7.63, 17.4k tok/s including eval, 57.3s, finite |
| 2 | 10M-token FineWeb-Edu run | `arzlm train --config configs/train-bench-10m.yaml` | train ~9.10 → 5.29, **val 6.49 → 5.50 (ppl 660 → 244)**, 18.8k tok/s including eval/ckpt, 533s, 2.80 GB peak allocated |
| WSL | Linux eager vs `torch.compile` BF16, seq=1024, seed 42, micro 1/2/4 | `docs/wsl2.md` | Compile **27.4k tok/s** at micro-4 (**+44%** vs Windows 19.06k). Linux eager only **+4%**. |
| Muon tune | 2M-token AdamW vs Muon LR sweep (Windows eager) | `python scripts/run_optimizer_tune.py` | **Muon lr=0.05** best of 0.01/0.02/0.05 (val 6.306 vs AdamW 6.450). Upper bound, not bracketed. |
| Muon compile | WSL2 hybrid Muon eager vs `torch.compile` | `scripts/wsl/run_muon_compile_bench.sh` | Fastest stable: **model compile, CUDA graphs off, micro-4 (~18.5k bench / ~25k train-step)**. Graphs-on / sequential recompile can SIGSEGV. |
| Muon LR bracket | WSL eager 2M: 0.05 / 0.075 / 0.10 | `arzlm train` on `configs/tune/muon-jordan-lr*-2m.yaml` | **0.05 best at 2M** (6.308). 0.10 worse; **did not run 0.15**. |
| Muon 10M eager | Fresh 10M of top two LRs, micro-2 eager | `configs/tune/muon-10m-lr0.05.yaml` and `lr0.075` | Eager 10M: 0.075 wins (5.209 vs 5.239) after a late crossover. **Not the 50M stack.** |
| Muon 10M compile | Fresh 10M on compile + micro-4 | `configs/tune/muon-10m-lr0.05-compile.yaml` and `lr0.075-compile` | **lr=0.05 wins** (5.153 vs 5.220) at every common checkpoint. |
| Muon WD | 2M wd 0.05/0.10/0.20 at lr=0.075 and at lr=0.05 | `configs/tune/muon-wd*-2m.yaml` | wd=0.20 ~0.02 better at 2M on both LRs; **keep 0.10**. |
| 50M matched | AdamW vs Muon, one 50M FineWeb-Edu pack, WSL compile | `configs/adamw-50m.yaml` then `configs/muon-50m.yaml` | **Muon wins at 50M** (val **4.251** vs AdamW **4.347**, Δ **−0.096**). Curves crossed: AdamW ahead ~11–38M. **Stop. Do not start 250M here.** |

### Benchmark (Windows eager, seq=1024, bf16-true, fused AdamW, flash SDPA, no compile)

Frozen copy: `docs/baselines/baseline-v0/windows-benchmark.json`.

| micro-batch | tok/s | step time | peak allocated | peak reserved |
| --- | --- | --- | --- | --- |
| 1 | 14,288 | 71.7 ms | 1.62 GB | 1.67 GB |
| **2** | **19,060** | **107.5 ms** | **2.59 GB** | **2.66 GB** |
| 4 | 19,005 | 215.5 ms | 4.71 GB | 4.95 GB |
| 8 | 12,181 | 672.5 ms | 8.47 GB | 9.26 GB (memory pressure) |

`torch.compile` on native Windows CUDA raises `TritonMissing`. Do not install unofficial Triton.

### WSL2 / torch.compile (Linux filesystem `~/arzlm`)

JSON: `docs/wsl/results/linux-eager-bf16.json`, `docs/wsl/results/linux-compile-bf16.json`.

Stack: Python 3.12.14, PyTorch **2.11.0+cu128**, Triton **3.6.0**, CUDA runtime **12.8**, driver 616.64, SDPA **flash**, BF16, seed 42, seq 1024, fused AdamW. `gcc` is required; Triton failed with `Failed to find C compiler` until `build-essential` was installed. Windows CUDA drivers were not modified.

| platform | micro | tok/s | step | alloc | reserved | compile s | loss | max abs logit Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Windows eager | 2 | 19,060 | 107.5 ms | 2.59 GB | 2.66 GB | n/a | ~4.29 (bench) | n/a |
| Linux eager | 1 | 16,980 | 60.3 ms | 1.54 GB | 1.59 GB | 0 | 3.683 | n/a |
| Linux eager | 2 | 19,851 | 103.2 ms | 2.63 GB | 2.70 GB | 0 | 3.746 | n/a |
| Linux eager | 4 | 19,722 | 207.7 ms | 4.36 GB | 4.61 GB | 0 | 4.143 | n/a |
| Linux compile | 1 | 21,574 | 47.5 ms | 1.27 GB | 1.32 GB | 19.6 | 3.630 | 0.059 |
| Linux compile | 2 | 25,289 | 81.0 ms | 1.95 GB | 2.31 GB | 27.9 | 4.923 | 0.059 |
| **Linux compile** | **4** | **27,446** | **149.2 ms** | **3.50 GB** | **4.20 GB** | 0.14 (cache) | 3.920 | 0.062 |

Numerical: compiled vs eager logits at init are finite and within `atol=0.1` (max abs ~0.06, typical BF16 compile noise). Init CE 10.69 eager vs ~10.66 compiled.

**Migration:** Linux eager is only ~4% faster than Windows at micro-2 — not enough. Linux **torch.compile** is **+33% (micro-2)** to **+44% (micro-4)** and was stable after gcc. Recommend moving **training** to WSL2 + compile. Keep architecture/data/seed. First compile of a new shape costs ~20–28 s and must be warmed up.

### Muon + torch.compile (WSL2, identical arch, BF16, seq=1024, seed 42)

JSON: `docs/wsl/results/linux-muon-eager-bf16.json`, `linux-muon-compile-bf16.json`, `linux-muon-ckpt-eager.json`, `linux-muon-ckpt-compile.json`.

Pinned stack (not upgraded): Python 3.12.14, PyTorch **2.11.0+cu128**, Triton **3.6.0**, KellerJordan `SingleDeviceMuonWithAuxAdam` as vendored. Optimizer is bound on the **uncompiled** module, then `torch.compile` (Muon disables CUDA graphs).

Bench is **accum=1** (optimizer step every micro-batch), so it understates training tok/s when global batch 8 uses accumulation.

| setup | micro | tok/s | step | alloc | reserved | compile s | loss finite | grads finite | max abs logit Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Muon eager | 2 | 11,747 | 174 ms | 2.47 GB | 2.54 GB | 0 | yes | yes | n/a |
| Muon eager | 4 | 14,665 | 279 ms | 4.63 GB | 5.05 GB | 0 | yes | yes | n/a |
| Muon compile, graphs **on** | 2/4 | — | — | — | — | — | segfault (exit 139) | — | — |
| **Muon compile, graphs off** | 2 | 13,405 | 153 ms | 1.94 GB | 2.29 GB | 20.9 | yes | yes | 0.059 |
| **Muon compile, graphs off** | **4** | **18,521** | **221 ms** | **3.40 GB** | **4.49 GB** | 20.5 | yes | yes | 0.062 |

Dynamo `graph_break` counters were empty. Compiled vs eager logits at init are finite and within `atol=0.1`.

**train_step / optimizer-step compile:** not default. Compiling `_one_step` graph-breaks on `.item()` in `as_float` and previously died (exit 11). Sequential model-compile jobs can also SIGSEGV (exit 139) after earlier Inductor runs. Do not use train-step compile.

**Checkpoint:** eager save/resume with an explicit `Path` to `final/lit_model.pth` is OK (4 steps → 6 steps, finite). Compile save/resume in a **fresh** process is OK (`inductor-no-cudagraphs`, finite). A string resume path is ignored by `find_resume_path` (looks for `step-*` instead).

**Selected for 50M Muon:** WSL2, **model `torch.compile`**, **CUDA graphs off**, **micro-batch 4**, global batch 8 (same 8192 tokens/step as the eager sweeps). Compiled 10M lr=0.05 measured **20.5k tok/s including eval** (~164 ms/iter after warmup → ~25k tok/s excluding eval). AdamW 50M keeps CUDA graphs (its fastest stable stack, ~28k tok/s at the same micro/global batch).

### Optimizer sweep (2M tokens, Windows eager, seed 42, same 10m packed data)

Commit `1658ba2303ad9173cd60c2b9d18d2efb0b076a42`. Configs under `configs/tune/`. Raw curves in `docs/tune/`. FLOP estimate 6ND = 1.200×10^15 for every trial.

Muon: `SingleDeviceMuonWithAuxAdam` on 75,497,472 hidden matrix params; AdamW on 24,596,736 embed/norm params. Cosine+warmup, `min_lr` ratio 0.1, wd 0.1, momentum 0.95, ns_steps 5, Nesterov, Jordan scale.

| trial | muon lr | final val CE | ppl | train loss | tok/s | wall s | alloc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adamw-2m | — | 6.450 | 632 | 6.387 | 18,532 | 107.9 | 2.80 GB |
| muon-0.01 | 0.01 | 6.355 | 575 | 6.298 | 15,751 | 126.9 | 2.86 GB |
| muon-0.02 | 0.02 | 6.332 | 563 | 6.279 | 15,687 | 127.4 | 3.07 GB |
| **muon-0.05** | **0.05** | **6.306** | **548** | **6.257** | 15,932 | 125.5 | 3.26 GB |

Validation CE vs tokens (primary):

| tokens | adamw | muon 0.01 | muon 0.02 | muon 0.05 |
| --- | --- | --- | --- | --- |
| 0.41M | 7.285 | 7.112 | 7.119 | 7.145 |
| 0.82M | 6.832 | 6.732 | 6.746 | 6.733 |
| 1.23M | 6.616 | 6.524 | 6.520 | 6.503 |
| 1.64M | 6.493 | 6.402 | 6.381 | 6.374 |
| 2.00M | 6.450 | 6.355 | 6.332 | **6.306** |

Validation CE vs wall-clock (secondary): at ~105 s AdamW is finished at 6.45; Muon-0.05 is already 6.37 at 101 s (1.64M tokens) and 6.31 at 124 s. Muon is ~15% lower tok/s and still wins both plots.

All four runs were finite. That sweep did **not** bracket the LR (0.05 was the top grid point). Expanded WSL results below supersede the 50M LR choice.

### Expanded 2M LR bracket (WSL eager, seed 42, same 10m data / val / arch / muon extras)

Same recipe as the Windows sweep: `compile: false`, micro 2, global 8, momentum 0.95, Nesterov, ns_steps=5, muon wd 0.1, aux AdamW unchanged. 0.05 was re-run on WSL so 0.075/0.10 share the platform. **0.15 was not run** (0.10 was worse, not best).

| trial | muon lr | final val CE | ppl | train loss | tok/s | wall s | max grad | spikes | qkv max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **muon-0.05** | **0.05** | **6.308** | **549** | **6.238** | 17,107 | 116.8 | 18.9 | 0 | 0.46 |
| muon-0.075 | 0.075 | 6.319 | 555 | 6.261 | 17,028 | 117.4 | 18.8 | 0 | 0.75 |
| muon-0.10 | 0.10 | 6.341 | 567 | 6.310 | 16,864 | 118.5 | 18.8 | 0 | 0.80 |

Validation CE vs tokens (primary):

| tokens | muon 0.05 | muon 0.075 | muon 0.10 |
| ---: | ---: | ---: | ---: |
| 0.41M | 7.171 | **7.144** | 7.164 |
| 0.82M | **6.745** | 6.755 | 6.765 |
| 1.23M | **6.504** | 6.525 | 6.541 |
| 1.64M | **6.367** | 6.383 | 6.415 |
| 2.00M | **6.308** | 6.319 | 6.341 |

Validation CE vs wall-clock (secondary): all three finish in 117–119 s, so the ranking is the same. No NaN/Inf, no loss spikes, no late blow-up. Q/K RMS grows with LR but stays finite; QK-Norm RMS stayed 1.0. Grad norms ~19 are pre-clip (`max_norm=1.0`).

WSL 0.05 (6.308) matches Windows 0.05 (6.306). **Best two for 10M: 0.05 and 0.075.**

### 10M confirmation (fresh runs, not 2M checkpoints)

Eager 10M (micro-2, `compile: false`) was run first, matched to the 2M methodology. Step 3 then required the **compile + micro-4** stack that 50M will use. Both pairs used seed 42, `data/prepared/10m`, the same val set, global batch 8, warmup 50, cosine to 10% of each group's peak, eval every 50 optimizer steps. Eager artifacts were preserved.

Eager 10M (not the 50M stack):

| tokens | muon 0.05 eager | muon 0.075 |
| ---: | ---: | ---: |
| 2.05M | **6.244** | 6.274 |
| 4.10M | **5.821** | 5.845 |
| 6.14M | **5.543** | 5.627 |
| 8.19M | **5.322** | 5.328 |
| **10.00M** | 5.239 | **5.209** |

| trial | final val CE | ppl | train loss | tok/s | wall s | finite | spikes | qkv max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| muon-10m-0.05 eager | 5.239 | 188 | 4.925 | 16,993 | 588 | yes | 0 | 0.58 |
| muon-10m-0.075 eager | 5.209 | 183 | 4.923 | 16,958 | 590 | yes | 0 | 0.67 |

Compiled 10M (50M stack: model compile, CUDA graphs off, micro-4). **lr=0.05 is better at every common checkpoint**; the eager late crossover did not reproduce.

| tokens | muon 0.05 compile | muon 0.075 compile |
| ---: | ---: | ---: |
| 2.05M | **6.217** | 6.241 |
| 4.10M | **5.771** | 5.813 |
| 6.14M | **5.490** | 5.584 |
| 8.19M | **5.253** | 5.341 |
| **10.00M** | **5.153** | 5.220 |

| trial | final val CE | ppl | train loss | tok/s | wall s | alloc | finite | spikes | qkv max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **muon-10m-0.05 compile** | **5.153** | **173** | **4.941** | 20,544 | 487 | 3.64 GB | yes | 0 | 0.52 |
| muon-10m-0.075 compile | 5.220 | 185 | 4.992 | 21,786 | 459 | 3.64 GB | yes | 0 | 0.55 |

Stage-2 AdamW 10M (Windows, same packed 10m) ended at val **5.50**. Both compiled Muon 10M runs beat that. Choose **lr=0.05** for 50M from compiled final val CE and the full curve. No divergence.

### Weight-decay sanity (2M, aux AdamW wd unchanged at 0.1)

First sweep at lr=0.075 (eager, before compiled 10M):

| muon wd | final val CE | train loss | notes |
| ---: | ---: | ---: | --- |
| 0.05 | 6.346 | 6.308 | worse |
| 0.10 | 6.319 | 6.261 | 2M LR-bracket trial |
| 0.20 | **6.298** | **6.239** | +0.021 vs 0.10 |

Repeat at the compiled-10M winner lr=0.05 (eager, matched 2M recipe):

| muon wd | final val CE | train loss | notes |
| ---: | ---: | ---: | --- |
| 0.05 | 6.327 | 6.294 | worse |
| 0.10 | 6.308 | 6.238 | 2M LR-bracket trial |
| 0.20 | **6.282** | **6.227** | +0.026 vs 0.10 |

One compiled 2M point at lr=0.05 wd=0.05 reached val 6.298; sequential compile of the other WD cells SIGSEGV'd (exit 139), so the matched WD grid stayed eager. 0.20 is slightly better at 2M on both LRs, but 2M already reversed vs compiled 10M on LR. **Do not promote wd=0.20.** Keep **0.10**. All completed runs finite, zero spikes.

### Recommended Muon hyperparameters (50M)

- Hybrid KellerJordan `SingleDeviceMuonWithAuxAdam` (pinned, not replaced)
- Muon: **lr=0.05**, momentum=0.95, Nesterov, ns_steps=5, Jordan scale, **wd=0.10**
- Aux AdamW: lr=6e-4, betas=(0.9, 0.95), eps=1e-8, wd=0.1
- WSL2 + model `torch.compile` + **CUDA graphs off** (automatic for `muon_hybrid`)
- micro-batch **4**, global batch **8**, seq 1024, BF16, seed **42**
- Architecture/tokenizer/data unchanged from baseline-v0

### 50M matched AdamW vs Muon (completed)

Artifacts: `docs/experiments/50m-optimizer-comparison/`. Shared pack hashes in `50m-pack-hashes.json`. Do **not** regenerate `data/prepared/50m` between these runs.

Packed once on WSL ext4 (`HuggingFaceFW/fineweb-edu` `sample-10BT`, seed 42, vocab 32k, eos_id=2):

| split | tokens | bytes | sha256 |
| --- | ---: | ---: | --- |
| train.bin | 49,999,500 | 99,999,000 | `1077390786abd0e4be65fa3500fb1ac50a36f16722ab651cf40c64d2ad95b504` |
| val.bin | 249,075 | 498,150 | `eec7b6bef683308f0acb2fbbeea36762aa9ed93fada5f3f110732956c3c435af` |
| tokenizer.json | — | — | `3132f2ffe4b8046661ffed8ffce19dc82ec5acbe90b469db8ef12639638e5ed8` |

46,721 documents. An earlier pack aborted in HuggingFace `tokenizers` NFC (`normalizer.rs:373`). The completed pack is the one hashed above. `IsolatedEncoder` now encodes FineWeb in a subprocess so a future panic skips one document instead of killing the process.

Both trains: git `94461cdf`, 100,094,208 params, seq 1024, micro 4, global 8, BF16, seed 42, `torch.compile` **active**, finite, 49,999,872 tokens, 6103 steps. AdamW CUDA graphs on; Muon graphs off (`inductor-no-cudagraphs`). `compile=true` now aborts if compile is not active (an eager fallback started and was killed after a `MemoryError` during a retry).

| | AdamW | Muon |
| --- | ---: | ---: |
| final val CE | **4.347** | **4.251** |
| val ppl | 77.28 | 70.18 |
| train loss (last) | 4.425 | 4.326 |
| tok/s including eval | 26,688 | 21,668 |
| wall s | 1874 | 2308 |
| peak allocated | 3.79 GB | 3.64 GB |
| compile backend | inductor | inductor-no-cudagraphs |

Primary metric is val CE vs tokens (eval every 200 steps ≈ 1.64M tokens):

| tokens | AdamW val CE | Muon val CE | delta (Muon−AdamW) |
| ---: | ---: | ---: | ---: |
| 1.64M | 6.5271 | 6.4537 | −0.0735 |
| 3.28M | 6.1022 | 5.9166 | −0.1856 |
| 9.83M | 5.3292 | 5.3273 | −0.0018 |
| 11.47M | 5.2160 | 5.2486 | +0.0326 |
| 21.30M | 4.8073 | 4.9891 | +0.1819 |
| 29.49M | 4.5572 | 4.7250 | +0.1678 |
| 39.32M | 4.4026 | 4.3882 | −0.0144 |
| 50.00M | 4.3474 | 4.2511 | **−0.0963** |

Muon leads early, AdamW is clearly ahead from ~11M to ~38M (peak gap ~0.18 at 21M), then Muon recrosses and wins the 50M endpoint by 0.096 CE. Plots: `val_ce_vs_tokens.svg`, `ppl_vs_tokens.svg`, `val_ce_vs_wall.svg`, `train_loss_vs_tokens.svg`.

**Recommendation for a later 250M:** use **Muon** (hybrid KellerJordan, lr 0.05, wd 0.10, compile, graphs off, micro-4) on the 50M endpoint. The mid-run reversal means the ranking is horizon-dependent; do not treat 0.096 as a guarantee at 250M. **Stop here. Do not start 250M in this step.**

## Later A/B candidates (not in baseline)

- **QK-Norm off vs on** (baseline has LitGPT default QK-Norm enabled).
- **RoPE:** partial rotary (`rotary_percentage=0.25`) vs full; RoPE base at 1024 context.
- **Loss masking** on the first token after EOS vs training through EOS.
- **Context 2048** only if VRAM and tok/s still work.
- **Dataset mix:** FineWeb-Edu only vs a small code/math slice.
- **z-loss / logit softcapping:** LitGPT can disable fused SDPA when attention softcapping is on.
- Moonlight RMS-matching (`0.2 * sqrt(max(A,B))` so Muon shares AdamW's LR) vs the Jordan spectral LR used here.

## Compute note

A compute-optimal ~100M model is typically trained on **billions** of tokens (Chinchilla-style). 10M/50M/100M local runs are diagnostics, not the production pretrain.

50M-token matched runs are **done**. See the section above. Do **not** start 250M from these commands.
