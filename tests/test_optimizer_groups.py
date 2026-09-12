from __future__ import annotations

import torch

from arzlm.model.config import build_gpt, build_model_config
from arzlm.training.optim.factory import build_optimizer, optimizer_kind, schedule_param_group_lrs
from arzlm.training.optim.groups import partition_litgpt_parameters


def test_partition_covers_every_parameter_once() -> None:
    model = build_gpt(build_model_config(), tie_embeddings=True)
    parts = partition_litgpt_parameters(model)
    muon_ids = {id(p) for p in parts["muon"]}
    adam_ids = {id(p) for p in parts["adamw"]}
    assert muon_ids.isdisjoint(adam_ids)
    unique = {id(p) for p in model.parameters()}
    assert muon_ids | adam_ids == unique
    assert all(p.ndim == 2 for p in parts["muon"])
    assert "lm_head.weight" in parts["adamw_names"]
    assert not any("wte" in name for name in parts["muon_names"])
    assert any(".attn.qkv.weight" in n for n in parts["muon_names"])
    assert any(".mlp.fc_1.weight" in n for n in parts["muon_names"])
    assert any("norm_1.weight" in n for n in parts["adamw_names"])
    assert any("norm_q.weight" in n for n in parts["adamw_names"])
    assert parts["muon_n_params"] + parts["adamw_n_params"] == sum(p.numel() for p in model.parameters())


def test_muon_factory_builds_hybrid_groups() -> None:
    model = build_gpt(build_model_config(), tie_embeddings=True)
    spec = {
        "name": "muon_hybrid",
        "muon": {"lr": 0.02, "momentum": 0.95, "weight_decay": 0.1},
        "adamw": {"lr": 6e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
    }
    opt = build_optimizer(spec, model, fused_cuda=False)
    assert optimizer_kind(spec) == "muon_hybrid"
    assert type(opt).__name__ == "SingleDeviceMuonWithAuxAdam"
    muon_groups = [g for g in opt.param_groups if g["use_muon"]]
    adam_groups = [g for g in opt.param_groups if not g["use_muon"]]
    assert len(muon_groups) == 1 and len(adam_groups) == 1
    assert muon_groups[0]["lr"] == 0.02
    assert adam_groups[0]["lr"] == 6e-4
    assert muon_groups[0]["initial_lr"] == 0.02


def test_hybrid_lr_peak_and_floor() -> None:
    model = build_gpt(build_model_config(), tie_embeddings=True)
    spec = {
        "name": "muon_hybrid",
        "muon": {"lr": 0.02, "momentum": 0.95, "weight_decay": 0.1},
        "adamw": {"lr": 6e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
    }
    opt = build_optimizer(spec, model, fused_cuda=False)
    schedule_param_group_lrs(opt, it=100, warmup_iters=100, max_iters=1000, min_lr=6e-5)
    muon_lr = next(g["lr"] for g in opt.param_groups if g["use_muon"])
    adam_lr = next(g["lr"] for g in opt.param_groups if not g["use_muon"])
    assert abs(muon_lr - 0.02) < 1e-12
    assert abs(adam_lr - 6e-4) < 1e-12
    schedule_param_group_lrs(opt, it=1001, warmup_iters=100, max_iters=1000, min_lr=6e-5)
    muon_lr = next(g["lr"] for g in opt.param_groups if g["use_muon"])
    adam_lr = next(g["lr"] for g in opt.param_groups if not g["use_muon"])
    assert abs(muon_lr - 0.002) < 1e-12
    assert abs(adam_lr - 6e-5) < 1e-12


def test_muon_optimizer_state_roundtrips(tmp_path) -> None:
    model = build_gpt(build_model_config(), tie_embeddings=True)
    spec = {
        "name": "muon_hybrid",
        "muon": {"lr": 0.05, "momentum": 0.95, "weight_decay": 0.1, "nesterov": True, "ns_steps": 5},
        "adamw": {"lr": 6e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
    }
    opt = build_optimizer(spec, model, fused_cuda=False)
    path = tmp_path / "muon-opt.pt"
    torch.save({"optimizer": opt.state_dict()}, path)
    opt2 = build_optimizer(spec, model, fused_cuda=False)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    opt2.load_state_dict(blob["optimizer"])
    assert type(opt2).__name__ == "SingleDeviceMuonWithAuxAdam"
    assert len(opt2.param_groups) == len(opt.param_groups)


def test_adamw_factory_still_default() -> None:
    model = build_gpt(build_model_config(), tie_embeddings=True)
    spec = {
        "class_path": "torch.optim.AdamW",
        "init_args": {"lr": 6e-4, "weight_decay": 0.1, "betas": [0.9, 0.95]},
    }
    opt = build_optimizer(spec, model, fused_cuda=False)
    assert optimizer_kind(spec) == "adamw"
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["initial_lr"] == 6e-4
