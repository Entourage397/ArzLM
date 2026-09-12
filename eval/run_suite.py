#!/usr/bin/env python3
"""Local quality + performance benchmark suite.

Primary scores are produced here. Do not mix published leaderboard numbers
into the main table.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = Path(__file__).resolve().parent / "settings.yaml"

MODELS = [
    {
        "key": "arzlm-300m",
        "display": "ArzLM-300M-Base",
        "kind": "arzlm",
        "repo": "Kymaris/ArzLM-300M-Base",
        "params": 303_353_856,
        "train_tokens": 6_000_214_016,
        "train_tokens_note": "6.000B unique packed tokens (this work)",
    },
    {
        "key": "cerebras-256m",
        "display": "Cerebras-GPT-256M",
        "kind": "hf",
        "repo": "cerebras/Cerebras-GPT-256M",
        "params": 256_000_000,
        "train_tokens": 5_120_000_000,
        "train_tokens_note": "~20 tokens/parameter Chinchilla-style (Dey et al. 2023); Hub repo may be unavailable",
    },
    {
        "key": "gemma3-270m",
        "display": "Gemma-3-270M (base)",
        "kind": "hf",
        "repo": "google/gemma-3-270m",
        "params": 268_098_176,
        "train_tokens": None,
        "train_tokens_note": "Gemma 3 pretrained; training-token budget is much larger than 6B (not a fair compute match)",
    },
    {
        "key": "smollm2-360m",
        "display": "SmolLM2-360M (base)",
        "kind": "hf",
        "repo": "HuggingFaceTB/SmolLM2-360M",
        "params": 361_821_568,
        "train_tokens": None,
        "train_tokens_note": "SmolLM2 pretrained on ~4T tokens (not a fair compute match)",
    },
]

PRIMARY_TASKS = [
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "boolq",
    "mmlu",
    "gsm8k",
    "lambada_openai",
    "wikitext",
]

SANITY_TASKS = ["hellaswag", "piqa"]


def _git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _gpu() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
            info["bf16"] = bool(torch.cuda.is_bf16_supported())
            info["cuda"] = torch.version.cuda
    except Exception as exc:
        info["error"] = str(exc)
    return info


def load_settings(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_lm_eval(
    *,
    model_kind: str,
    model_args: str,
    tasks: list[str],
    settings: dict[str, Any],
    output_path: Path,
    limit: int | None,
    include_path: Path | None,
) -> dict[str, Any]:
    from lm_eval import evaluator

    fewshot = settings["fewshot"]
    results_by_task: dict[str, Any] = {}
    t0 = time.perf_counter()
    for task in tasks:
        num_fewshot = fewshot.get(task)
        kwargs = dict(
            model=model_kind,
            model_args=model_args,
            tasks=[task],
            num_fewshot=num_fewshot,
            batch_size=settings.get("batch_size", 1),
            device="cuda:0",
            random_seed=settings["seed"],
            numpy_random_seed=settings["seed"],
            torch_random_seed=settings["seed"],
            fewshot_random_seed=settings["seed"],
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            confirm_run_unsafe_code=False,
            log_samples=False,
            bootstrap_iters=1000,
        )
        if limit is not None:
            kwargs["limit"] = limit
        print(f"=== {model_kind} {model_args} task={task} fewshot={num_fewshot} limit={limit}", flush=True)
        task_t0 = time.perf_counter()
        try:
            raw = evaluator.simple_evaluate(**kwargs)
            elapsed = time.perf_counter() - task_t0
            results_by_task[task] = {
                "ok": True,
                "elapsed_s": elapsed,
                "num_fewshot": num_fewshot,
                "results": raw.get("results", {}),
                "n-shot": raw.get("n-shot", {}),
                "configs": {
                    k: {
                        "num_fewshot": (v or {}).get("num_fewshot") if isinstance(v, dict) else None,
                        "dataset_path": (v or {}).get("dataset_path") if isinstance(v, dict) else None,
                    }
                    for k, v in (raw.get("configs") or {}).items()
                },
            }
        except Exception as exc:
            results_by_task[task] = {
                "ok": False,
                "elapsed_s": time.perf_counter() - task_t0,
                "num_fewshot": num_fewshot,
                "error_class": type(exc).__name__,
                "error": str(exc)[:2000],
            }
            print(f"FAILED {task}: {type(exc).__name__}: {exc}", flush=True)
        output_path.write_text(json.dumps({"tasks": results_by_task}, indent=2) + "\n", encoding="utf-8")
    return {
        "elapsed_s": time.perf_counter() - t0,
        "tasks": results_by_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--arzlm-checkpoint", type=Path, required=True)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", default="arzlm-300m,cerebras-256m,gemma3-270m,smollm2-360m")
    parser.add_argument("--tasks", default=None)
    args = parser.parse_args()

    settings = load_settings(args.settings)
    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else (SANITY_TASKS if args.sanity else PRIMARY_TASKS)
    if args.sanity and args.limit is None:
        args.limit = 8

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "lm_eval": _pkg_version("lm_eval") or _pkg_version("lm-eval"),
        "transformers": _pkg_version("transformers"),
        "torch": _pkg_version("torch"),
        "litgpt": _pkg_version("litgpt"),
        "gpu": _gpu(),
        "settings": settings,
        "git_eval": _git_rev(ROOT),
    }
    payload: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": env,
        "models": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    include = Path(__file__).resolve().parent
    sys.path.insert(0, str(include))
    import lm_eval.models  # noqa: F401  # registers hf and other stock backends
    import arzlm_lm  # noqa: F401  # registers lm-eval --model arzlm

    for spec in MODELS:
        if spec["key"] not in selected:
            continue
        print(f"\n######## {spec['display']} ########", flush=True)
        rec: dict[str, Any] = dict(spec)
        rec["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            if spec["kind"] == "arzlm":
                model_args = (
                    f"checkpoint={args.arzlm_checkpoint},"
                    f"dtype={settings.get('dtype','bf16')},"
                    f"max_length={settings['max_length']}"
                )
                rec["lm_eval"] = run_lm_eval(
                    model_kind="arzlm",
                    model_args=model_args,
                    tasks=tasks,
                    settings=settings,
                    output_path=args.out.with_name(f"{spec['key']}.partial.json"),
                    limit=args.limit,
                    include_path=include,
                )
            else:
                model_args = (
                    f"pretrained={spec['repo']},"
                    f"dtype={settings.get('dtype','bfloat16')},"
                    f"max_length={settings['max_length']},"
                    "add_bos_token=False,"
                    "trust_remote_code=False"
                )
                rec["lm_eval"] = run_lm_eval(
                    model_kind="hf",
                    model_args=model_args,
                    tasks=tasks,
                    settings=settings,
                    output_path=args.out.with_name(f"{spec['key']}.partial.json"),
                    limit=args.limit,
                    include_path=None,
                )
            rec["ok"] = True
        except Exception as exc:
            rec["ok"] = False
            rec["error_class"] = type(exc).__name__
            rec["error"] = str(exc)[:4000]
            print(f"MODEL FAILED {spec['key']}: {exc}", flush=True)
        rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload["models"][spec["key"]] = rec
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
