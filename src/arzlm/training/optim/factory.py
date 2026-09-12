"""Build baseline AdamW or experimental Muon+AdamW from experiment YAML."""

from __future__ import annotations

from typing import Any

import torch
from litgpt.utils import instantiate_torch_optimizer
from litgpt.pretrain import get_lr

from arzlm.training.optim.groups import partition_litgpt_parameters
from arzlm.training.optim.muon import SingleDeviceMuonWithAuxAdam


def optimizer_kind(spec: dict[str, Any]) -> str:
    name = str(spec.get("name") or "").lower()
    class_path = str(spec.get("class_path") or "")
    if name in {"muon", "muon_hybrid"} or "Muon" in class_path:
        return "muon_hybrid"
    return "adamw"


def _as_betas(value: Any) -> tuple[float, float]:
    if value is None:
        return (0.9, 0.95)
    return (float(value[0]), float(value[1]))


def build_optimizer(spec: dict[str, Any], model: torch.nn.Module, *, fused_cuda: bool) -> torch.optim.Optimizer:
    kind = optimizer_kind(spec)
    if kind == "muon_hybrid":
        parts = partition_litgpt_parameters(model)
        muon_cfg = dict(spec.get("muon") or {})
        adam_cfg = dict(spec.get("adamw") or spec.get("init_args") or {})
        muon_group = {
            "params": parts["muon"],
            "lr": float(muon_cfg.get("lr", 0.02)),
            "momentum": float(muon_cfg.get("momentum", 0.95)),
            "weight_decay": float(muon_cfg.get("weight_decay", 0.1)),
            "use_muon": True,
        }
        adam_group = {
            "params": parts["adamw"],
            "lr": float(adam_cfg.get("lr", 6e-4)),
            "betas": _as_betas(adam_cfg.get("betas")),
            "eps": float(adam_cfg.get("eps", 1e-8)),
            "weight_decay": float(adam_cfg.get("weight_decay", 0.1)),
            "use_muon": False,
        }
        optimizer = SingleDeviceMuonWithAuxAdam([muon_group, adam_group])
        optimizer.param_partition = {
            "muon_names": parts["muon_names"],
            "adamw_names": parts["adamw_names"],
            "muon_n_params": parts["muon_n_params"],
            "adamw_n_params": parts["adamw_n_params"],
        }
    else:
        torch_spec = {
            "class_path": spec.get("class_path", "torch.optim.AdamW"),
            "init_args": spec.get("init_args") or {},
        }
        optimizer = instantiate_torch_optimizer(torch_spec, model.parameters(), fused=fused_cuda)
        optimizer.param_partition = None
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def schedule_param_group_lrs(
    optimizer: torch.optim.Optimizer,
    it: int,
    warmup_iters: int,
    max_iters: int,
    min_lr: float,
) -> float:
    """Cosine+warmup per group, preserving relative LRs in a hybrid optimizer.

    For a single AdamW group this matches `litgpt.pretrain.get_lr` with an
    absolute `min_lr`. For hybrid groups, each group's floor is
    `initial_lr * (min_lr / ref_lr)` where `ref_lr` is the first group's peak
    (AdamW peak in our configs).
    """
    groups = optimizer.param_groups
    hybrid = any(g.get("use_muon") for g in groups)
    if not hybrid:
        ref_lr = float(groups[0].get("initial_lr", groups[0]["lr"]))
        lr = get_lr(ref_lr, it, warmup_iters, max_iters, min_lr)
        for group in groups:
            group["lr"] = lr
        return lr
    adam_groups = [g for g in groups if not g.get("use_muon")]
    ref_lr = float(adam_groups[0].get("initial_lr", adam_groups[0]["lr"])) if adam_groups else 6e-4
    min_ratio = (min_lr / ref_lr) if ref_lr else 0.1
    logged = 0.0
    for group in groups:
        peak = float(group.get("initial_lr", group["lr"]))
        group["lr"] = get_lr(peak, it, warmup_iters, max_iters, peak * min_ratio)
        if group.get("use_muon"):
            logged = group["lr"]
    return logged if logged else float(groups[0]["lr"])


def optimizer_record(spec: dict[str, Any], optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    groups = []
    for group in optimizer.param_groups:
        item = {k: v for k, v in group.items() if k != "params"}
        item["n_params"] = int(sum(p.numel() for p in group["params"]))
        item["n_tensors"] = len(group["params"])
        groups.append(item)
    record = {
        "kind": optimizer_kind(spec),
        "class": type(optimizer).__name__,
        "spec": spec,
        "param_groups": groups,
    }
    partition = getattr(optimizer, "param_partition", None)
    if partition:
        record["partition"] = {
            "muon_n_params": partition["muon_n_params"],
            "adamw_n_params": partition["adamw_n_params"],
            "muon_names": partition["muon_names"],
            "adamw_names": partition["adamw_names"],
        }
    return record
