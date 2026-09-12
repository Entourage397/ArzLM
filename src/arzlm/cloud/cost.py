"""Cost projection for the ArzLM-300M / 6B Modal run.

Metered dollars are compared against the continue ceiling ($98) and the
workspace hard cap ($100). The continue ceiling is the operating limit so a
checkpoint/eval cannot be killed at $100 mid-write.
"""

from __future__ import annotations

from typing import Any

A100_USD_PER_HOUR = 2.10
PROJECTED_CONTINUE_MAX = 98.0
HARD_USAGE_LIMIT = 100.0
EVAL_EXPORT_RESERVE_USD = 2.0
SAFETY_MARGIN_USD = 2.0
DEFAULT_PREPROCESS_USD = 32.0
TARGET_TOKENS = 6_000_214_016
PROBATION_TOKENS = 16_777_216
PROBATION_MAX_SECONDS = 3600.0
PACK_USD_PER_CONTAINER_HOUR = 1.05


def gpu_hours_for_tokens(tokens: int, tokens_per_sec: float) -> float:
    if tokens_per_sec <= 0 or tokens <= 0:
        return float("inf")
    return float(tokens) / float(tokens_per_sec) / 3600.0


def project_complete_build(
    *,
    current_metered_usd: float,
    remaining_preprocess_usd: float,
    remaining_train_tokens: int,
    steady_tokens_per_sec: float,
    a100_usd_per_hour: float = A100_USD_PER_HOUR,
    eval_export_reserve_usd: float = EVAL_EXPORT_RESERVE_USD,
    continue_max: float = PROJECTED_CONTINUE_MAX,
    hard_limit: float = HARD_USAGE_LIMIT,
    safety_margin_usd: float = SAFETY_MARGIN_USD,
) -> dict[str, Any]:
    hours = gpu_hours_for_tokens(remaining_train_tokens, steady_tokens_per_sec)
    gpu_usd = hours * a100_usd_per_hour if hours != float("inf") else float("inf")
    preprocess = max(0.0, float(current_metered_usd) + float(remaining_preprocess_usd))
    reserve = max(0.0, float(eval_export_reserve_usd))
    total = preprocess + gpu_usd + reserve if gpu_usd != float("inf") else float("inf")
    room = hard_limit - safety_margin_usd
    return {
        "current_metered_usd": float(current_metered_usd),
        "remaining_preprocess_usd": float(remaining_preprocess_usd),
        "remaining_train_tokens": int(remaining_train_tokens),
        "steady_tokens_per_sec": float(steady_tokens_per_sec),
        "gpu_hours_remaining": hours,
        "gpu_usd_remaining": gpu_usd,
        "eval_export_reserve_usd": reserve,
        "projected_usd": total,
        "continue_max": float(continue_max),
        "hard_limit": float(hard_limit),
        "within_continue_max": total <= continue_max,
        "within_hard_margin": total <= room,
        "decision_continue": bool(
            total <= continue_max and total <= room and hours != float("inf")
        ),
    }


def classify_probation_failure(
    *,
    finite: bool,
    checkpoint_ok: bool,
    resume_ok: bool,
    steady_tokens_per_sec: float,
    projected: dict[str, Any],
) -> str | None:
    if not finite:
        return "non_finite_loss"
    if not checkpoint_ok:
        return "missing_or_invalid_checkpoint"
    if not resume_ok:
        return "checkpoint_reload_failed"
    if steady_tokens_per_sec <= 0:
        return "no_measured_throughput"
    if not projected.get("decision_continue"):
        return "projected_over_budget"
    return None
