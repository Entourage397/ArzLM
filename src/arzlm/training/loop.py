"""Single-GPU pretraining loop built on litgpt.GPT + Lightning Fabric.

Intentionally avoids FSDP, DeepSpeed, bitsandbytes, and gradient checkpointing.
The optimizer, precision, SDPA path, cosine schedule, and weight init are the
same primitives litgpt.pretrain uses on one device.

AdamW remains the baseline optimizer. Muon is an experimental alternative that
must be selected explicitly in YAML (`optimizer.name: muon_hybrid`).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
from litgpt.args import TrainArgs
from litgpt.pretrain import initialize_weights
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    find_resume_path,
    num_parameters,
    save_config,
)
from torch.utils.data import DataLoader

from arzlm.data.catalog import DOMAINS
from arzlm.data.mixture import (
    CyclingDataset,
    OffsetSampler,
    is_stem_meta,
    load_domain_dataset,
    load_stem_train_dataset,
    normalize_weights,
)
from arzlm.data.packed import load_split_dataset, read_meta
from arzlm.model.config import (
    expected_trainable_parameters,
    TARGET_BY_NAME,
    assert_parameter_budget,
    build_gpt,
    format_parameter_report,
    formula_parameter_count,
)
from arzlm.training.compile import dynamo_snapshot, maybe_compile
from arzlm.training.config import ExperimentConfig, resolve_path
from arzlm.training.environment import collect_environment, preferred_sdpa_name
from arzlm.training.optim.factory import build_optimizer, optimizer_record, schedule_param_group_lrs
from arzlm.training.report import (
    append_jsonl,
    estimate_train_flops,
    git_commit,
    git_describe,
    perplexity,
    utc_now,
    write_json,
)
from arzlm.cloud.runtime import (
    on_checkpoint,
    stop_requested,
    write_latest_valid,
    write_status,
)
from arzlm.training.stability import as_float, attention_health


def make_dataloaders(
    data_dir: Path,
    seq_length: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    *,
    n_items: int,
    start_index: int = 0,
    mixture: dict[str, float] | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader] | None]:
    meta = read_meta(data_dir)
    val_by_domain: dict[str, DataLoader] | None = None
    if is_stem_meta(meta):
        train_ds = load_stem_train_dataset(
            data_dir,
            seq_length,
            seed=seed,
            weights=mixture,
            n_items=n_items,
        )
        domain_loaders: dict[str, DataLoader] = {}
        first_val = None
        for name in DOMAINS:
            if name not in meta.get("domains", {}):
                continue
            ds = load_domain_dataset(data_dir, name, "val", seq_length)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )
            domain_loaders[name] = loader
            if first_val is None:
                first_val = loader
        if first_val is None:
            raise RuntimeError(f"STEM corpus {data_dir} has no validation shards")
        val_loader = first_val
        val_by_domain = domain_loaders
    else:
        inner = load_split_dataset(data_dir, "train", seq_length=seq_length)
        train_ds = CyclingDataset(inner, seed=seed, n_items=n_items)
        val_ds = load_split_dataset(data_dir, "val", seq_length=seq_length)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        sampler=OffsetSampler(start_index, n_items),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, val_by_domain


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    max_iters: int,
    seq_length: int | None = None,
) -> torch.Tensor:
    model.eval()
    t = seq_length or getattr(model, "max_seq_length", None)
    if t is None:
        raise ValueError("seq_length is required for validation")
    losses = []
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids = batch[:, 0:t].contiguous().long()
        targets = batch[:, 1 : t + 1].contiguous().long()
        logits = model(input_ids)
        losses.append(chunked_cross_entropy(logits, targets))
    model.train()
    if not losses:
        return torch.tensor(float("nan"), device=fabric.device)
    return torch.stack(losses).mean()


def validate_domains(
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_by_domain: dict[str, DataLoader],
    max_iters: int,
    seq_length: int,
    mixture: dict[str, float] | None,
) -> tuple[float, dict[str, float]]:
    per: dict[str, float] = {}
    for name, loader in val_by_domain.items():
        per[name] = float(validate(fabric, model, loader, max_iters=max_iters, seq_length=seq_length))
    weights = normalize_weights(mixture)
    present = {k: weights[k] for k in per if k in weights}
    z = sum(present.values()) or 1.0
    agg = sum((present[k] / z) * per[k] for k in per)
    stem_keys = [k for k in ("math", "code", "science") if k in per]
    if stem_keys:
        stem_w = {k: weights.get(k, 0.0) for k in stem_keys}
        stem_z = sum(stem_w.values()) or 1.0
        per = dict(per)
        per["stem"] = sum((stem_w[k] / stem_z) * per[k] for k in stem_keys)
    return float(agg), per


def _max_iters(train: TrainArgs, seq_length: int, devices: int, num_nodes: int) -> int:
    tokens_per_iter = train.micro_batch_size * seq_length
    candidates: list[int] = []
    if train.max_tokens is not None:
        candidates.append(max(1, train.max_tokens // tokens_per_iter))
    if train.max_steps is not None:
        accum = train.gradient_accumulation_iters(devices, num_nodes)
        candidates.append(max(1, train.max_steps * accum))
    if not candidates:
        raise ValueError("Set train.max_tokens and/or train.max_steps")
    return min(candidates)


def save_checkpoint(
    fabric: L.Fabric,
    state: dict,
    tokenizer_dir: Path,
    checkpoint_file: Path,
    config,
) -> None:
    dest_dir = checkpoint_file.parent
    tmp_dir = dest_dir.with_name(dest_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / checkpoint_file.name
    fabric.print(f"Saving checkpoint to {checkpoint_file}")
    fabric.save(tmp_file, state)
    if fabric.global_rank == 0:
        save_config(config, tmp_dir)
        if tokenizer_dir.exists():
            copy_config_files(tokenizer_dir, tmp_dir)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        os.replace(tmp_dir, dest_dir)


def _rotate_full_checkpoints(out_dir: Path, *, keep: int = 3) -> None:
    dirs = sorted(
        p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("step-") and not p.name.endswith(".tmp")
    )
    for old in dirs[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def _maybe_milestone(tokens_seen: int, milestones: list[int], done: set[int]) -> int | None:
    for mark in milestones:
        if tokens_seen >= mark and mark not in done:
            return mark
    return None


def _persist_full_checkpoint(
    fabric: L.Fabric,
    state: dict,
    exp: ExperimentConfig,
    tokens_seen: int,
    *,
    keep_full: int,
    milestones: list[int],
    milestone_done: set[int],
) -> Path:
    dest = exp.out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth"
    save_checkpoint(fabric, state, exp.tokenizer_dir, dest, exp.model)
    if fabric.global_rank == 0:
        _rotate_full_checkpoints(exp.out_dir, keep=keep_full)
        write_latest_valid(exp.out_dir, dest.parent, tokens_seen, int(state["step_count"]))
        on_checkpoint(dest)
    mark = _maybe_milestone(tokens_seen, milestones, milestone_done)
    if mark is not None:
        fabric.print(f"Model-only milestone {mark} tokens")
        save_checkpoint(
            fabric,
            {"model": state["model"], "iter_num": state["iter_num"], "step_count": state["step_count"]},
            exp.tokenizer_dir,
            exp.out_dir / "milestones" / f"{mark}" / "lit_model.pth",
            exp.model,
        )
        milestone_done.add(mark)
        if fabric.global_rank == 0:
            on_checkpoint(exp.out_dir / "milestones" / f"{mark}" / "lit_model.pth")
    gpu_runtime_guard(repo_root=exp.out_dir)
    return dest.parent


def query_gpu_runtime() -> dict[str, Any]:
    payload: dict[str, Any] = {"at": utc_now()}
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            parts = [p.strip() for p in smi.stdout.strip().split(",")]
            if len(parts) >= 5:
                payload["gpu_util_pct"] = float(parts[0])
                payload["mem_util_pct"] = float(parts[1])
                payload["mem_used_mb"] = float(parts[2])
                payload["mem_total_mb"] = float(parts[3])
                payload["temp_c"] = float(parts[4])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    if torch.cuda.is_available():
        payload["torch_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        payload["torch_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    return payload


def gpu_runtime_guard(*, max_temp_c: int = 87, min_disk_gb: float = 12.0, repo_root: Path | None = None) -> None:
    """Pause on thermal headroom; abort before disk exhaustion. Never changes power limits."""
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            temp_s, _, _ = smi.stdout.strip().split(",")[0:3]
            temp = int(float(temp_s.strip()))
            while temp >= max_temp_c:
                time.sleep(20)
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                temp = int(float(smi.stdout.strip().split(",")[0]))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    root = repo_root or Path(".")
    usage = shutil.disk_usage(root)
    if usage.free < min_disk_gb * 1024**3:
        raise RuntimeError(f"disk free {usage.free / 1024**3:.1f} GiB is below {min_disk_gb} GiB; aborting")


def run_training(exp: ExperimentConfig, *, dry_meta: bool = False) -> dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    env = collect_environment()
    devices = exp.devices
    train = exp.train
    if train.tie_embeddings is None:
        train.tie_embeddings = True
    seq_length = train.max_seq_length or exp.model.block_size

    logger = choose_logger(
        exp.logger_name,
        exp.out_dir,
        name=f"pretrain-{exp.model.name}",
        resume=bool(exp.resume),
        log_interval=train.log_interval,
    )
    fabric = L.Fabric(
        devices=devices,
        strategy="auto",
        precision=exp.precision,
        loggers=[logger],
    )
    fabric.launch()
    fabric.seed_everything(exp.seed)
    exp.out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = find_resume_path(exp.resume, exp.out_dir)
    metrics_path = exp.out_dir / "metrics.jsonl"
    if metrics_path.exists() and not resume_path:
        metrics_path.unlink()

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = build_gpt(exp.model, tie_embeddings=bool(train.tie_embeddings))
    initialize_weights(fabric, model, n_layer=exp.model.n_layer, n_embd=exp.model.n_embd)
    init_from_raw = exp.raw.get("init_from")
    if init_from_raw and not resume_path:
        init_from = resolve_path(init_from_raw)
        if init_from is None or not init_from.is_file():
            raise FileNotFoundError(f"init_from missing: {init_from_raw}")
        blob = torch.load(init_from, map_location="cpu", weights_only=False)
        weights = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model.load_state_dict(weights, strict=True)
        fabric.print(f"Loaded model weights from {init_from} (optimizer not restored)")
    model.max_seq_length = seq_length
    expected = expected_trainable_parameters(exp.model)
    param_counts = {
        "trainable": num_parameters(model, requires_grad=True),
        "total": num_parameters(model),
        "formula": formula_parameter_count(exp.model, tie_embeddings=bool(train.tie_embeddings)),
        "target": TARGET_BY_NAME.get(exp.model.name, expected),
        "expected": expected,
        "name": exp.model.name,
    }
    fabric.print(format_parameter_report(param_counts))
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.2f}s")
    if int(param_counts["trainable"]) != expected:
        raise RuntimeError(
            f"Refusing to train: trainable={param_counts['trainable']:,} "
            f"expected {expected:,}"
        )
    assert_parameter_budget(
        {**param_counts, "formula": param_counts["formula"]},
        name=exp.model.name,
    )
    if dry_meta:
        return {"parameters": param_counts, "environment": env}

    # Bind Muon/AdamW groups on the uncompiled module so LitGPT names still match.
    optimizer = build_optimizer(exp.optimizer, model, fused_cuda=fabric.device.type == "cuda")
    opt_record = optimizer_record(exp.optimizer, optimizer)
    compile_result = maybe_compile(
        model,
        exp.compile,
        fabric.device,
        disable_cudagraphs=opt_record["kind"] == "muon_hybrid",
    )
    if compile_result.active:
        fabric.print("torch.compile is active")
        fabric.print(f"compile backend={compile_result.backend}")
    elif compile_result.error:
        fabric.print(f"torch.compile not active: {compile_result.error}")
        if str(exp.compile) == "true":
            raise RuntimeError(
                f"compile=true but torch.compile is not active: {compile_result.error}"
            )
    else:
        fabric.print(f"torch.compile disabled (requested={exp.compile})")
    model = fabric.setup(compile_result.model)
    optimizer = fabric.setup_optimizers(optimizer)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    fabric.print(f"optimizer={opt_record['kind']} class={opt_record['class']}")

    if exp.tokenizer_dir.exists():
        Tokenizer(exp.tokenizer_dir)  # validate files exist; copies go with checkpoints

    state = {
        "model": model,
        "optimizer": optimizer,
        "iter_num": 0,
        "step_count": 0,
    }
    if resume_path:
        fabric.print(f"Resuming from {resume_path}")
        fabric.load(resume_path, state)

    accum = train.gradient_accumulation_iters(devices, 1)
    max_iters = _max_iters(train, seq_length, devices, 1)
    n_items = max_iters * train.micro_batch_size
    start_index = int(state["iter_num"]) * train.micro_batch_size
    train_loader, val_loader, val_by_domain = make_dataloaders(
        exp.data_dir,
        seq_length=seq_length,
        batch_size=train.micro_batch_size,
        num_workers=exp.num_workers,
        seed=exp.seed,
        n_items=n_items,
        start_index=start_index,
        mixture=exp.mixture,
    )
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    if val_by_domain:
        val_by_domain = {name: fabric.setup_dataloaders(loader) for name, loader in val_by_domain.items()}
    warmup_iters = train.warmup_iters(devices, 1, max_iters, train_loader)
    train_iterator = iter(train_loader)
    log_iter_interval = train.log_interval * accum
    peak_loss = None
    finite = True
    t_train = time.perf_counter()
    tokens_seen = int(state["iter_num"]) * int(train.micro_batch_size) * seq_length * fabric.world_size
    segment_tokens_start = tokens_seen
    last_loss = float("nan")
    last_val_loss = None
    last_val_by_domain: dict[str, float] | None = None
    lr = 0.0
    commit = git_commit()
    last_grad_norm = None
    max_grad_norm = 0.0
    loss_spike_count = 0
    prev_logged_loss = None
    qk_health_last = None
    ckpt_cfg = dict(exp.raw.get("checkpoint") or {})
    keep_full = int(ckpt_cfg.get("keep_full", 3))
    milestones = [int(x) for x in (ckpt_cfg.get("milestones_tokens") or [])]
    milestone_done: set[int] = set()
    first_checkpoint_tokens = int(ckpt_cfg.get("first_checkpoint_tokens") or 0)
    save_interval_tokens = int(ckpt_cfg.get("save_interval_tokens") or 0)
    last_ckpt_tokens = tokens_seen
    first_ckpt_done = first_checkpoint_tokens > 0 and tokens_seen >= first_checkpoint_tokens
    last_status_t = time.perf_counter()
    status_every = float((exp.raw.get("cloud") or {}).get("status_commit_seconds") or 60)
    exit_reason = "complete"
    tokens_per_update = int(train.micro_batch_size * accum * seq_length * fabric.world_size)
    probation_tokens = int(os.environ.get("ARZLM_PROBATION_TOKENS") or 0)
    compile_warmup_s: float | None = None
    steady_t0: float | None = None
    steady_tokens0: int | None = None

    write_json(
        exp.out_dir / "experiment.json",
        {
            "started_at": utc_now(),
            "git_commit": commit,
            "git_describe": git_describe(),
            "seed": exp.seed,
            "precision": exp.precision,
            "compile": exp.compile,
            "data_dir": str(exp.data_dir),
            "tokenizer_dir": str(exp.tokenizer_dir),
            "out_dir": str(exp.out_dir),
            "model": exp.model.name,
            "train": {
                "max_tokens": train.max_tokens,
                "max_steps": train.max_steps,
                "max_seq_length": seq_length,
                "micro_batch_size": train.micro_batch_size,
                "global_batch_size": train.global_batch_size,
                "lr_warmup_steps": train.lr_warmup_steps,
                "min_lr": train.min_lr,
                "max_norm": train.max_norm,
                "tie_embeddings": train.tie_embeddings,
            },
            "optimizer": opt_record,
            "lr_schedule": {
                "type": "linear_warmup_cosine",
                "warmup_iters": warmup_iters,
                "max_iters": max_iters,
                "min_lr": train.min_lr,
            },
            "raw_config": exp.raw,
            "environment": env,
        },
    )

    fabric.print(
        f"Training: max_iters={max_iters} accum={accum} seq={seq_length} "
        f"micro_batch={train.micro_batch_size} global_batch={train.global_batch_size} "
        f"sdpa={preferred_sdpa_name()} precision={exp.precision} seed={exp.seed} "
        f"git={commit}"
    )

    while state["iter_num"] < max_iters:
        batch = next(train_iterator)
        lr = schedule_param_group_lrs(optimizer, state["iter_num"], warmup_iters, max_iters, train.min_lr)
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        input_ids = batch[:, 0:seq_length].contiguous().long()
        targets = batch[:, 1 : seq_length + 1].contiguous().long()
        is_accumulating = state["iter_num"] % accum != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets)
            fabric.backward(loss / accum)
        if not torch.isfinite(loss).all():
            finite = False
            exit_reason = "non_finite"
            fabric.print(f"Non-finite loss at iter {state['iter_num']}: {loss}")
            break
        last_loss = float(loss.detach().item())
        peak_loss = last_loss if peak_loss is None else min(peak_loss, last_loss)
        tokens_seen += int(train.micro_batch_size * seq_length * fabric.world_size)

        if not is_accumulating:
            total_norm = fabric.clip_gradients(
                model, optimizer, max_norm=train.max_norm, error_if_nonfinite=False
            )
            last_grad_norm = as_float(total_norm)
            if last_grad_norm is not None:
                if last_grad_norm != last_grad_norm or abs(last_grad_norm) == float("inf"):
                    finite = False
                    exit_reason = "non_finite"
                    fabric.print(f"Non-finite grad_norm at iter {state['iter_num']}: {last_grad_norm}")
                    break
                max_grad_norm = max(max_grad_norm, last_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
            if compile_warmup_s is None:
                compile_warmup_s = time.perf_counter() - t_train
            if state["step_count"] >= 32 and steady_t0 is None:
                steady_t0 = time.perf_counter()
                steady_tokens0 = tokens_seen

        if state["iter_num"] % log_iter_interval == 0:
            dt = time.perf_counter() - iter_t0
            wall = time.perf_counter() - t_train
            metrics = {
                "loss": last_loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "learning_rate": lr,
                "iter_time": dt,
                "tokens": tokens_seen,
            }
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']} "
                f"loss {last_loss:.4f} lr {lr:.6g} {dt * 1000:.1f} ms"
            )
            if prev_logged_loss is not None and last_loss > max(8.0, 2.0 * prev_logged_loss):
                loss_spike_count += 1
                fabric.print(f"loss spike at iter {state['iter_num']}: {prev_logged_loss:.4f} -> {last_loss:.4f}")
            prev_logged_loss = last_loss
            append_jsonl(
                metrics_path,
                {
                    "kind": "train",
                    "iter": state["iter_num"],
                    "step": state["step_count"],
                    "tokens": tokens_seen,
                    "wall_s": wall,
                    "train_loss": last_loss,
                    "learning_rate": lr,
                    "iter_time_s": dt,
                    "grad_norm": last_grad_norm,
                },
            )
            if fabric.global_rank == 0 and (time.perf_counter() - last_status_t) >= status_every:
                elapsed_now = time.perf_counter() - t_train
                seg_tokens = max(0, tokens_seen - segment_tokens_start)
                tps = seg_tokens / elapsed_now if elapsed_now > 0 else 0.0
                remaining = max(0, int(train.max_tokens or 0) - tokens_seen)
                hours_left = (remaining / tps / 3600.0) if tps > 0 else None
                latest = exp.out_dir / "latest_valid.json"
                latest_blob = None
                if latest.is_file():
                    try:
                        latest_blob = json.loads(latest.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        latest_blob = None
                gpu_rt = query_gpu_runtime()
                gpu_rt["iteration_time_s"] = dt
                gpu_rt["tokens_per_second"] = tps
                gpu_rt["step"] = int(state["step_count"])
                gpu_rt["tokens"] = int(tokens_seen)
                append_jsonl(exp.out_dir / "gpu_runtime.jsonl", gpu_rt)
                steady_now = None
                if steady_t0 is not None and steady_tokens0 is not None:
                    steady_elapsed = max(1e-6, time.perf_counter() - steady_t0)
                    steady_now = max(0, tokens_seen - steady_tokens0) / steady_elapsed
                write_status(
                    exp.out_dir,
                    {
                        "schema_version": 1,
                        "run_id": exp.raw.get("name") or exp.model.name,
                        "phase": "TRAIN_300M_6B" if exp.model.name == "ArzLM-300M" else "TRAIN",
                        "desired_state": "RUNNING",
                        "heartbeat_at": utc_now(),
                        "model_parameter_count": int(param_counts["trainable"]),
                        "optimizer_step": int(state["step_count"]),
                        "training_tokens": int(tokens_seen),
                        "target_tokens": int(train.max_tokens or 0),
                        "latest_loss": last_loss,
                        "lr": lr,
                        "grad_norm": last_grad_norm,
                        "iteration_time": dt,
                        "tokens_per_second": tps,
                        "steady_tokens_per_sec": steady_now,
                        "compile_warmup_s": compile_warmup_s,
                        "tokens_per_update": tokens_per_update,
                        "micro_batch_size": train.micro_batch_size,
                        "global_batch_size": train.global_batch_size,
                        "gpu_util_pct": gpu_rt.get("gpu_util_pct"),
                        "mem_used_mb": gpu_rt.get("mem_used_mb"),
                        "projected_gpu_hours_remaining": hours_left,
                        "projected_cost_remaining": (
                            None
                            if hours_left is None
                            else hours_left * float((exp.raw.get("cloud") or {}).get("a100_usd_per_hour") or 2.10)
                        ),
                        "finite": finite,
                        "exit_reason": None,
                        "latest_valid_checkpoint": None if latest_blob is None else latest_blob.get("checkpoint_dir"),
                        "checkpoint_token_count": None if latest_blob is None else latest_blob.get("tokens"),
                        "peak_allocated_vram_bytes": (
                            int(torch.cuda.max_memory_allocated()) if fabric.device.type == "cuda" else None
                        ),
                    },
                )
                last_status_t = time.perf_counter()

        if (
            val_loader is not None
            and not is_accumulating
            and exp.eval.interval
            and state["step_count"] > 0
            and state["step_count"] % exp.eval.interval == 0
        ):
            if val_by_domain:
                last_val_loss, last_val_by_domain = validate_domains(
                    fabric, model, val_by_domain, exp.eval.max_iters, seq_length, exp.mixture
                )
            else:
                last_val_loss = float(validate(fabric, model, val_loader, max_iters=exp.eval.max_iters, seq_length=seq_length))
                last_val_by_domain = None
            val_ppl = perplexity(last_val_loss)
            wall = time.perf_counter() - t_train
            qk_health_last = attention_health(model)
            extra = ""
            if last_val_by_domain:
                extra = " " + " ".join(f"{k}={v:.4f}" for k, v in last_val_by_domain.items())
            fabric.print(f"val loss {last_val_loss:.4f} ppl {val_ppl:.3f}{extra}")
            log_vals = {"val_loss": last_val_loss, "val_ppl": val_ppl}
            if last_val_by_domain:
                log_vals.update({f"val_loss_{k}": v for k, v in last_val_by_domain.items()})
            fabric.log_dict(log_vals, step=state["iter_num"])
            append_jsonl(
                metrics_path,
                {
                    "kind": "val",
                    "iter": state["iter_num"],
                    "step": state["step_count"],
                    "tokens": tokens_seen,
                    "wall_s": wall,
                    "train_loss": last_loss,
                    "val_loss": last_val_loss,
                    "val_ppl": val_ppl,
                    "val_by_domain": last_val_by_domain,
                    "learning_rate": lr,
                    "grad_norm": last_grad_norm,
                    "qk_health": qk_health_last,
                },
            )

        if not is_accumulating and state["step_count"] > 0:
            token_due = save_interval_tokens > 0 and (tokens_seen - last_ckpt_tokens) >= save_interval_tokens
            step_due = bool(train.save_interval) and state["step_count"] % train.save_interval == 0
            first_due = (not first_ckpt_done) and first_checkpoint_tokens > 0 and tokens_seen >= first_checkpoint_tokens
            if token_due or step_due or first_due:
                _persist_full_checkpoint(
                    fabric,
                    state,
                    exp,
                    tokens_seen,
                    keep_full=keep_full,
                    milestones=milestones,
                    milestone_done=milestone_done,
                )
                last_ckpt_tokens = tokens_seen
                first_ckpt_done = True

        if not is_accumulating and state["step_count"] > 0:
            if train.max_time and (time.perf_counter() - t_train) >= float(train.max_time):
                exit_reason = "max_time"
                fabric.print(f"Segment max_time={train.max_time}s reached; writing recovery checkpoint")
                break
            if stop_requested(exp.out_dir):
                exit_reason = "stop_requested"
                fabric.print("STOP_REQUESTED observed; writing recovery checkpoint")
                break
            if (
                probation_tokens > 0
                and first_ckpt_done
                and tokens_seen >= probation_tokens
            ):
                exit_reason = "probation"
                fabric.print(
                    f"Probation token cap {probation_tokens} reached at {tokens_seen} tokens; writing recovery checkpoint"
                )
                break

    if exit_reason in {"max_time", "stop_requested", "probation"} and state["step_count"] > 0:
        _persist_full_checkpoint(
            fabric,
            state,
            exp,
            tokens_seen,
            keep_full=keep_full,
            milestones=milestones,
            milestone_done=milestone_done,
        )

    budget_complete = (
        finite
        and exit_reason == "complete"
        and state["iter_num"] >= max_iters
    )

    if exp.eval.final_validation and budget_complete:
        if val_by_domain:
            last_val_loss, last_val_by_domain = validate_domains(
                fabric, model, val_by_domain, exp.eval.max_iters, seq_length, exp.mixture
            )
        else:
            last_val_loss = float(validate(fabric, model, val_loader, max_iters=exp.eval.max_iters, seq_length=seq_length))
            last_val_by_domain = None
        qk_health_last = attention_health(model)
        extra = ""
        if last_val_by_domain:
            extra = " " + " ".join(f"{k}={v:.4f}" for k, v in last_val_by_domain.items())
        fabric.print(f"final val loss {last_val_loss:.4f}{extra}")
        append_jsonl(
            metrics_path,
            {
                "kind": "val_final",
                "iter": state["iter_num"],
                "step": state["step_count"],
                "tokens": tokens_seen,
                "wall_s": time.perf_counter() - t_train,
                "train_loss": last_loss,
                "val_loss": last_val_loss,
                "val_ppl": perplexity(last_val_loss),
                "val_by_domain": last_val_by_domain,
                "learning_rate": lr,
                "grad_norm": last_grad_norm,
                "qk_health": qk_health_last,
            },
        )
    else:
        val_loss = None if last_val_loss is None else torch.tensor(last_val_loss)

    if budget_complete:
        save_checkpoint(fabric, state, exp.tokenizer_dir, exp.out_dir / "final" / "lit_model.pth", exp.model)
        if fabric.global_rank == 0:
            on_checkpoint(exp.out_dir / "final" / "lit_model.pth")
    elapsed = time.perf_counter() - t_train
    seg_tokens = max(0, tokens_seen - segment_tokens_start)
    if steady_t0 is not None and steady_tokens0 is not None:
        steady_elapsed = max(1e-6, time.perf_counter() - steady_t0)
        steady_tokens_per_sec = max(0, tokens_seen - steady_tokens0) / steady_elapsed
    else:
        steady_tokens_per_sec = seg_tokens / elapsed if elapsed > 0 else 0.0
    val_ce = None if last_val_loss is None else last_val_loss
    result = {
        "environment": env,
        "parameters": param_counts,
        "compile_active": compile_result.active,
        "compile_error": compile_result.error,
        "loss": last_loss,
        "best_train_loss": peak_loss,
        "val_loss": val_ce,
        "val_ppl": perplexity(val_ce),
        "val_by_domain": last_val_by_domain,
        "steps": state["step_count"],
        "iters": state["iter_num"],
        "tokens": tokens_seen,
        "elapsed_s": elapsed,
        "tokens_per_sec": seg_tokens / elapsed if elapsed > 0 else 0.0,
        "steady_tokens_per_sec": steady_tokens_per_sec,
        "compile_warmup_s": compile_warmup_s,
        "end_to_end_tokens_per_sec": seg_tokens / elapsed if elapsed > 0 else 0.0,
        "finite": finite,
        "exit_reason": exit_reason,
        "budget_complete": budget_complete,
        "out_dir": str(exp.out_dir),
        "sdpa": preferred_sdpa_name(),
        "precision": exp.precision,
        "micro_batch_size": train.micro_batch_size,
        "global_batch_size": train.global_batch_size,
        "gradient_accumulation": accum,
        "seq_length": seq_length,
        "seed": exp.seed,
        "git_commit": commit,
        "git_describe": git_describe(),
        "optimizer": opt_record,
        "lr_schedule": "linear_warmup_cosine",
        "flop_estimate_6nd": estimate_train_flops(int(param_counts["trainable"]), int(tokens_seen)),
        "metrics_path": str(metrics_path),
        "max_grad_norm": max_grad_norm,
        "last_grad_norm": last_grad_norm,
        "loss_spike_count": loss_spike_count,
        "qk_health": qk_health_last,
        "dynamo": dynamo_snapshot() if compile_result.active else None,
    }
    if fabric.device.type == "cuda":
        result["peak_allocated_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        result["peak_reserved_vram_bytes"] = int(torch.cuda.max_memory_reserved())
    write_json(exp.out_dir / "segment_result.json", result)
    if budget_complete:
        (exp.out_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        write_json(exp.out_dir / "experiment_result.json", result)
    if fabric.global_rank == 0:
        write_status(
            exp.out_dir,
            {
                "schema_version": 1,
                "run_id": exp.raw.get("name") or exp.model.name,
                "phase": "TRAIN_COMPLETE" if budget_complete else "TRAIN_SEGMENT_EXIT",
                "desired_state": "STOPPED" if exit_reason == "stop_requested" else ("COMPLETE" if budget_complete else "RUNNING"),
                "heartbeat_at": utc_now(),
                "model_parameter_count": int(param_counts["trainable"]),
                "optimizer_step": int(state["step_count"]),
                "training_tokens": int(tokens_seen),
                "target_tokens": int(train.max_tokens or 0),
                "latest_loss": last_loss,
                "lr": lr,
                "grad_norm": last_grad_norm,
                "tokens_per_second": result["tokens_per_sec"],
                "steady_tokens_per_sec": result["steady_tokens_per_sec"],
                "compile_warmup_s": compile_warmup_s,
                "finite": finite,
                "exit_reason": exit_reason,
                "budget_complete": budget_complete,
            },
        )
    fabric.print(json.dumps({k: result[k] for k in ("loss", "val_loss", "steps", "tokens", "tokens_per_sec", "finite", "exit_reason")}, indent=2))
    return result
