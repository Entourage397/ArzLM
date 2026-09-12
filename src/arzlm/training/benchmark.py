"""Short GPU throughput benchmark with warmup and micro-batch sweep."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
from litgpt.pretrain import initialize_weights
from litgpt.utils import chunked_cross_entropy, num_parameters

from arzlm.data.packed import load_split_dataset
from arzlm.model.config import build_gpt, build_model_config
from arzlm.training.compile import dynamo_reset, dynamo_snapshot, maybe_compile, triton_version
from arzlm.training.environment import collect_environment, preferred_sdpa_name
from arzlm.training.optim.factory import build_optimizer
from arzlm.training.report import git_commit, git_describe
from arzlm.training.stability import as_float

DEFAULT_MICROBATCHES = (1, 2, 4, 8)
MUON_BENCH_SPEC = {
    "name": "muon_hybrid",
    "class_path": "arzlm.training.optim.muon.SingleDeviceMuonWithAuxAdam",
    "muon": {"lr": 0.05, "momentum": 0.95, "weight_decay": 0.1},
    "adamw": {"lr": 6e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
}
ADAMW_BENCH_SPEC = {
    "name": "adamw",
    "class_path": "torch.optim.AdamW",
    "init_args": {"lr": 6e-4, "weight_decay": 0.1, "betas": [0.9, 0.95]},
}


def _one_step(fabric: L.Fabric, model, optimizer, batch, seq_length: int, accum: int) -> tuple[float, float | None, bool]:
    input_ids = batch[:, 0:seq_length].contiguous().long()
    targets = batch[:, 1 : seq_length + 1].contiguous().long()
    logits = model(input_ids)
    loss = chunked_cross_entropy(logits, targets)
    fabric.backward(loss / accum)
    total_norm = fabric.clip_gradients(model, optimizer, max_norm=1.0, error_if_nonfinite=False)
    grad_norm = as_float(total_norm)
    finite = bool(torch.isfinite(loss).all().item()) and (grad_norm is None or (grad_norm == grad_norm and abs(grad_norm) != float("inf")))
    optimizer.step()
    optimizer.zero_grad()
    return float(loss.detach().item()), grad_norm, finite


def _numeric_compare(eager: torch.Tensor, compiled: torch.Tensor) -> dict[str, Any]:
    delta = (eager.float() - compiled.float()).abs()
    eager_ok = bool(torch.isfinite(eager).all().item())
    compiled_ok = bool(torch.isfinite(compiled).all().item())
    return {
        "eager_finite": eager_ok,
        "compiled_finite": compiled_ok,
        "max_abs_logit_diff": float(delta.max().item()),
        "mean_abs_logit_diff": float(delta.mean().item()),
        "close_atol_1e-2": bool(torch.allclose(eager.float(), compiled.float(), atol=1e-2, rtol=1e-2)),
        "close_atol_1e-1": bool(torch.allclose(eager.float(), compiled.float(), atol=1e-1, rtol=1e-1)),
    }


def _try_microbatch(
    *,
    micro_batch_size: int,
    seq_length: int,
    data_dir: Path,
    precision: str,
    compile_requested: str,
    compile_scope: str,
    optimizer_name: str,
    warmup_steps: int,
    measure_steps: int,
    seed: int,
    tie_embeddings: bool,
) -> dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    dynamo_reset()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    fabric = L.Fabric(devices=1, strategy="auto", precision=precision)
    fabric.launch()
    fabric.seed_everything(seed)

    config = build_model_config()
    with fabric.init_module(empty_init=False):
        model = build_gpt(config, tie_embeddings=tie_embeddings)
    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)
    model.max_seq_length = seq_length
    n_params = num_parameters(model, requires_grad=True)
    opt_spec = MUON_BENCH_SPEC if optimizer_name == "muon" else ADAMW_BENCH_SPEC

    dataset = load_split_dataset(data_dir, "train", seq_length=seq_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    loader = fabric.setup_dataloaders(loader)
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i + 1 >= warmup_steps + measure_steps:
            break
    if len(batches) < warmup_steps + measure_steps:
        raise RuntimeError(
            f"Need {warmup_steps + measure_steps} batches for benchmark, packed dataset only has {len(batches)}"
        )

    probe = batches[0][:, 0:seq_length].contiguous().long()
    probe_targets = batches[0][:, 1 : seq_length + 1].contiguous().long()
    numerical: dict[str, Any] | None = None
    compile_forward_s = 0.0
    compile_time_s = 0.0
    compile_backend = None
    step_compile_error = None

    if compile_requested in {"true", "auto"} and compile_scope == "model":
        with torch.no_grad():
            eager_logits = model(probe)
        eager_loss = float(chunked_cross_entropy(eager_logits, probe_targets).item())
    else:
        eager_logits = None
        eager_loss = None

    # Optimizer must bind the uncompiled Parameter objects, then the module may be compiled.
    optimizer = build_optimizer(opt_spec, model, fused_cuda=fabric.device.type == "cuda")
    compile_result = maybe_compile(
        model,
        compile_requested if compile_scope == "model" else "false",
        fabric.device,
        disable_cudagraphs=optimizer_name == "muon",
    )
    model = fabric.setup(compile_result.model)
    optimizer = fabric.setup_optimizers(optimizer)
    compile_backend = compile_result.backend
    compiled_step = None
    if compile_requested in {"true", "auto"} and compile_scope == "train_step":
        try:
            compiled_step = torch.compile(_one_step, backend="inductor")
            compile_backend = "inductor:train_step"
        except Exception as exc:
            step_compile_error = f"{type(exc).__name__}: {exc}"
            compiled_step = None

    if compile_result.active and eager_logits is not None:
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        with torch.no_grad():
            compiled_logits = model(probe)
        torch.cuda.synchronize()
        compile_forward_s = time.perf_counter() - t_fwd
        numerical = _numeric_compare(eager_logits, compiled_logits)
        numerical["eager_loss"] = eager_loss
        numerical["compiled_loss"] = float(chunked_cross_entropy(compiled_logits, probe_targets).item())
        del compiled_logits
    if eager_logits is not None:
        del eager_logits

    step_fn = compiled_step or _one_step
    compile_active = compile_result.active or compiled_step is not None
    warmup_used = warmup_steps
    grads_finite = True
    last_grad_norm = None
    if compile_active:
        torch.cuda.synchronize()
        t_compile = time.perf_counter()
        first_loss, last_grad_norm, step_ok = step_fn(fabric, model, optimizer, batches[0], seq_length, accum=1)
        torch.cuda.synchronize()
        compile_time_s = time.perf_counter() - t_compile
        grads_finite = grads_finite and step_ok
        if numerical is not None:
            numerical["first_compiled_train_loss"] = first_loss
        start_warmup = 1
    else:
        start_warmup = 0

    for i in range(start_warmup, warmup_used):
        _, last_grad_norm, step_ok = step_fn(fabric, model, optimizer, batches[i], seq_length, accum=1)
        grads_finite = grads_finite and step_ok
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    last_loss = None
    for i in range(measure_steps):
        last_loss, last_grad_norm, step_ok = step_fn(fabric, model, optimizer, batches[warmup_used + i], seq_length, accum=1)
        grads_finite = grads_finite and step_ok
        if not step_ok:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tokens = measure_steps * micro_batch_size * seq_length
    dynamo = dynamo_snapshot() if compile_active else {}
    result = {
        "ok": True,
        "mode": "compile" if compile_active else "eager",
        "compile_scope": compile_scope,
        "optimizer": optimizer_name,
        "micro_batch_size": micro_batch_size,
        "seq_length": seq_length,
        "gradient_accumulation": 1,
        "warmup_steps": warmup_used,
        "measure_steps": measure_steps,
        "elapsed_s": elapsed,
        "step_time_s": elapsed / measure_steps,
        "tokens": tokens,
        "tokens_per_sec": tokens / elapsed,
        "sequences_per_sec": (measure_steps * micro_batch_size) / elapsed,
        "training_loss": last_loss,
        "loss_finite": last_loss is not None and last_loss == last_loss and abs(last_loss) != float("inf"),
        "grads_finite": grads_finite,
        "last_grad_norm": last_grad_norm,
        "compile_active": compile_active,
        "compile_backend": compile_backend,
        "compile_error": compile_result.error or step_compile_error,
        "compile_time_s": compile_time_s,
        "compile_forward_s": compile_forward_s,
        "numerical": numerical,
        "dynamo": dynamo,
        "parameter_count": n_params,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "sdpa": preferred_sdpa_name(),
        "precision": precision,
        "fused_adamw": optimizer_name == "adamw",
        "triton_version": triton_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    del model, optimizer, loader, batches
    torch.cuda.empty_cache()
    return result


def run_benchmark(
    data_dir: str | Path,
    *,
    out_path: str | Path | None = None,
    seq_length: int = 1024,
    precision: str = "bf16-true",
    compile: str = "auto",
    compile_scope: str = "model",
    optimizer: str = "adamw",
    warmup_steps: int = 5,
    measure_steps: int = 10,
    microbatches: tuple[int, ...] = DEFAULT_MICROBATCHES,
    seed: int = 42,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires CUDA")
    if optimizer not in {"adamw", "muon"}:
        raise ValueError("optimizer must be adamw|muon")
    if compile_scope not in {"model", "train_step"}:
        raise ValueError("compile_scope must be model|train_step")
    env = collect_environment()
    data_dir = Path(data_dir)
    trials: list[dict[str, Any]] = []
    for micro in microbatches:
        torch.cuda.empty_cache()
        try:
            trial = _try_microbatch(
                micro_batch_size=micro,
                seq_length=seq_length,
                data_dir=data_dir,
                precision=precision,
                compile_requested=compile,
                compile_scope=compile_scope,
                optimizer_name=optimizer,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
                seed=seed,
                tie_embeddings=True,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            trial = {
                "ok": False,
                "micro_batch_size": micro,
                "optimizer": optimizer,
                "error": "CUDA out of memory",
            }
        except Exception as exc:
            torch.cuda.empty_cache()
            trial = {
                "ok": False,
                "micro_batch_size": micro,
                "optimizer": optimizer,
                "error": f"{type(exc).__name__}: {exc}",
            }
        trials.append(trial)
        status = "ok" if trial.get("ok") else trial.get("error")
        print(f"micro_batch={micro}: {status}")

    successful = [t for t in trials if t.get("ok")]
    best = max(successful, key=lambda t: t["tokens_per_sec"]) if successful else None
    report = {
        "environment": env,
        "gpu_name": env.get("gpu_name"),
        "pytorch_version": env.get("pytorch"),
        "cuda_version": env.get("cuda_runtime"),
        "triton_version": env.get("triton_version"),
        "bf16_support": env.get("bf16_supported"),
        "git_commit": git_commit(),
        "git_describe": git_describe(),
        "seed": seed,
        "compile_requested": compile,
        "compile_scope": compile_scope,
        "optimizer": optimizer,
        "trials": trials,
        "best": best,
        "model_parameter_count": None if best is None else best["parameter_count"],
        "peak_allocated_vram_bytes": None if best is None else best["peak_allocated_vram_bytes"],
        "peak_reserved_vram_bytes": None if best is None else best["peak_reserved_vram_bytes"],
        "tokens_per_sec": None if best is None else best["tokens_per_sec"],
        "sequences_per_sec": None if best is None else best["sequences_per_sec"],
        "step_time_s": None if best is None else best["step_time_s"],
        "achieved_batch_size": None if best is None else best["micro_batch_size"],
        "gradient_accumulation": 1,
        "context_length": seq_length,
        "training_loss": None if best is None else best["training_loss"],
        "torch_compile_active": None if best is None else best["compile_active"],
        "sdpa_backend": preferred_sdpa_name(),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return report
