"""Cloud pipeline state machine for ArzLM-300M / 6B.

Modal functions call these helpers. The LitGPT/Fabric trainer remains in
`arzlm.training.loop`. This module only owns durable phase state, paths,
and CPU-side gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arzlm.data.catalog import DOMAINS, MIXTURE_TARGET, TRAIN_TOKEN_QUOTA_6B, VAL_TOKEN_QUOTA_6B
from arzlm.model.config import (
    ARZLM_300M_NAME,
    EXPECTED_TRAINABLE_PARAMETERS_300M,
    assert_parameter_budget,
    count_parameters,
    load_model_config,
)
from arzlm.training.report import git_commit, git_describe, utc_now, write_json

PIPELINE_ID = "arzlm-300m-full-build-v1"
RUN_ID = "arzlm-300m-base-6b"
SCHEMA_VERSION = 1

PHASES = [
    "INIT",
    "SOURCE_LOCK",
    "TOKENIZER_SAMPLE",
    "TOKENIZER_TRAIN",
    "CORPUS_PREP",
    "CORPUS_VALIDATE",
    "MODEL_PREFLIGHT",
    "TRAIN_300M_6B",
    "FINAL_VALIDATE",
    "EVALUATE",
    "EXPORT",
    "FINAL_SMOKE",
    "COMPLETE",
]


def data_root() -> Path:
    return Path(os.environ.get("ARZLM_DATA_ROOT", "/vol/data"))


def artifacts_root() -> Path:
    return Path(os.environ.get("ARZLM_ARTIFACTS_ROOT", "/vol/artifacts"))


def pipeline_dir() -> Path:
    return artifacts_root() / "pipeline" / PIPELINE_ID


def state_path() -> Path:
    return pipeline_dir() / "state.json"


def desired_state_path() -> Path:
    return pipeline_dir() / "desired.json"


def lease_path(kind: str = "train") -> Path:
    return pipeline_dir() / f"lease-{kind}.json"


def named_lock_is_held(existing: Any, *, now: float, ttl_s: float, holder: str) -> bool:
    """True when another worker still owns a Modal Dict lock within ttl_s."""
    if not isinstance(existing, dict):
        return False
    other = existing.get("holder")
    if not other or other == holder:
        return False
    try:
        age = now - float(existing.get("t") or 0)
    except (TypeError, ValueError):
        return False
    return 0 <= age < ttl_s


def assert_a100_40gb_identity(name: str, mem_gb: float) -> None:
    """Refuse H100 / A100-80GB / tiny GPUs. Used by the Modal trainer container."""
    if mem_gb > 50:
        raise RuntimeError(f"Refusing unexpected GPU {name} with {mem_gb:.1f} GB; want A100 40GB")
    if mem_gb < 30:
        raise RuntimeError(f"GPU memory too small: {name} {mem_gb:.1f} GB")
    if "A100" not in str(name).upper():
        raise RuntimeError(f"Refusing unexpected GPU name {name!r}")


def iso_age_s(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()


def write_lease(kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "holder": os.environ.get("MODAL_TASK_ID")
        or os.environ.get("HOSTNAME")
        or str(os.getpid()),
        "heartbeat_at": utc_now(),
        **(extra or {}),
    }
    write_json(lease_path(kind), payload)
    return payload


def clear_lease(kind: str | None = None) -> None:
    path = lease_path(kind or "train")
    if not path.is_file():
        return
    if kind:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rec = {}
        if rec.get("kind") not in {None, kind}:
            return
    try:
        path.unlink()
    except OSError:
        pass


def lease_fresh(kind: str, *, max_age_s: float = 1800) -> bool:
    path = lease_path(kind)
    if not path.is_file():
        return False
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if rec.get("kind") != kind:
        return False
    age = iso_age_s(rec.get("heartbeat_at"))
    return age is not None and 0 <= age < max_age_s


def trainer_heartbeat_fresh(*, max_age_s: float = 1800) -> bool:
    if lease_fresh("train", max_age_s=max_age_s):
        return True
    path = train_out_dir() / "status.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("budget_complete"):
        return False
    phase = str(payload.get("phase") or "")
    if phase in {"TRAIN_SEGMENT_EXIT", "TRAIN_COMPLETE"}:
        return False
    if payload.get("exit_reason") in {"complete", "stop_requested", "non_finite", "probation", "max_time"}:
        return False
    age = iso_age_s(payload.get("heartbeat_at"))
    return age is not None and 0 <= age < max_age_s


def tokenizer_dir() -> Path:
    return data_root() / "tokenizer" / "arzlm-stem-32k-v2"


def tokenizer_sample_dir() -> Path:
    return data_root() / "tokenizer-corpus" / "arzlm-stem-32k-v2"


def corpus_dir() -> Path:
    return data_root() / "prepared" / "arzlm-stem-6b-v1"


def source_lock_path() -> Path:
    return data_root() / "security" / "source-lock.json"


def train_out_dir() -> Path:
    return artifacts_root() / "runs" / RUN_ID


def probation_path() -> Path:
    return pipeline_dir() / "probation.json"


def load_probation() -> dict[str, Any] | None:
    path = probation_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def estimate_preprocess_usd() -> float:
    from arzlm.cloud.cost import DEFAULT_PREPROCESS_USD, PACK_USD_PER_CONTAINER_HOUR

    total_s = 0.0
    state_dir = corpus_dir() / "state"
    if state_dir.is_dir():
        for domain in DOMAINS:
            path = state_dir / f"{domain}.json"
            if not path.is_file():
                continue
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            total_s += float(rec.get("elapsed_s") or 0)
    if total_s <= 0:
        return float(DEFAULT_PREPROCESS_USD)
    return (total_s / 3600.0) * PACK_USD_PER_CONTAINER_HOUR


def estimate_remaining_pack_usd() -> float:
    from arzlm.cloud.cost import PACK_USD_PER_CONTAINER_HOUR
    from arzlm.data.catalog import TRAIN_TOKEN_QUOTA_6B

    hours = 0.0
    state_dir = corpus_dir() / "state"
    for domain in DOMAINS:
        path = state_dir / f"{domain}.json"
        if not path.is_file():
            hours += 6.0
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            hours += 6.0
            continue
        if rec.get("status") == "complete":
            continue
        tokens = float(rec.get("tokens_train") or 0)
        elapsed = float(rec.get("elapsed_s") or 0)
        target = float(TRAIN_TOKEN_QUOTA_6B[domain])
        if tokens <= 0 or elapsed <= 0:
            hours += 6.0
            continue
        remain = max(0.0, target - tokens)
        hours += remain / (tokens / elapsed) / 3600.0
    return hours * PACK_USD_PER_CONTAINER_HOUR


def load_metered_usd() -> float | None:
    path = pipeline_dir() / "metered.json"
    if not path.is_file():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return float(rec["usd"])
    except (KeyError, TypeError, ValueError):
        return None


def verify_checkpoint_reloadable(ckpt_dir: Path | None = None) -> dict[str, Any]:
    import torch
    from litgpt.config import Config

    from arzlm.model.config import build_gpt, count_parameters

    try:
        root = ckpt_dir or latest_checkpoint_dir()
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc)}
    ckpt = root / "lit_model.pth"
    cfg_path = root / "model_config.yaml"
    if not ckpt.is_file() or not cfg_path.is_file():
        return {"ok": False, "error": f"incomplete checkpoint at {root}"}
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob or "optimizer" not in blob:
        return {"ok": False, "error": "checkpoint missing model/optimizer"}
    if not isinstance(blob["optimizer"], dict) or not blob["optimizer"]:
        return {"ok": False, "error": "optimizer state empty"}
    cfg = Config.from_file(cfg_path)
    model = build_gpt(cfg, tie_embeddings=True)
    model.load_state_dict(blob["model"], strict=True)
    n_params = int(count_parameters(cfg, model=model, tie_embeddings=True)["trainable"])
    del model
    return {
        "ok": True,
        "checkpoint_dir": str(root),
        "step_count": blob.get("step_count"),
        "iter_num": blob.get("iter_num"),
        "n_params": n_params,
    }


def _summarize_gpu_runtime() -> dict[str, Any]:
    path = train_out_dir() / "gpu_runtime.jsonl"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                rows.append(rec)
    utils = [float(r["gpu_util_pct"]) for r in rows if r.get("gpu_util_pct") is not None]
    mems = [float(r["mem_used_mb"]) for r in rows if r.get("mem_used_mb") is not None]
    iters = [float(r["iteration_time_s"]) for r in rows if r.get("iteration_time_s") is not None]
    return {
        "samples": len(rows),
        "mean_gpu_util_pct": (sum(utils) / len(utils)) if utils else None,
        "mean_mem_used_mb": (sum(mems) / len(mems)) if mems else None,
        "mean_iteration_time_s": (sum(iters) / len(iters)) if iters else None,
        "last": rows[-1] if rows else None,
    }


def run_evaluate_probation() -> dict[str, Any]:
    from arzlm.cloud.cost import (
        EVAL_EXPORT_RESERVE_USD,
        PROBATION_TOKENS,
        TARGET_TOKENS,
        classify_probation_failure,
        project_complete_build,
    )
    from arzlm.model.config import EXPECTED_TRAINABLE_PARAMETERS_300M

    status_file = train_out_dir() / "status.json"
    segment_file = train_out_dir() / "segment_result.json"
    status: dict[str, Any] = {}
    segment: dict[str, Any] = {}
    if status_file.is_file():
        status = json.loads(status_file.read_text(encoding="utf-8"))
    if segment_file.is_file():
        segment = json.loads(segment_file.read_text(encoding="utf-8"))
    tokens = int(status.get("training_tokens") or segment.get("tokens") or 0)
    finite = bool(status.get("finite", segment.get("finite", False)))
    loss = status.get("latest_loss", segment.get("loss"))
    steady = float(status.get("steady_tokens_per_sec") or segment.get("steady_tokens_per_sec") or 0.0)
    e2e = float(status.get("tokens_per_second") or segment.get("tokens_per_sec") or 0.0)
    warmup_s = status.get("compile_warmup_s", segment.get("compile_warmup_s"))
    reload = verify_checkpoint_reloadable()
    latest = train_out_dir() / "latest_valid.json"
    latest_blob = json.loads(latest.read_text(encoding="utf-8")) if latest.is_file() else {}
    checkpoint_ok = reload.get("ok") is True and int(latest_blob.get("tokens") or 0) >= int(0.9 * PROBATION_TOKENS)
    remaining = max(0, TARGET_TOKENS - tokens)
    pack_usd = estimate_preprocess_usd()
    remain_pack = estimate_remaining_pack_usd()
    gpu_so_far = float(segment.get("elapsed_s") or 0) / 3600.0 * 2.10
    metered = load_metered_usd()
    if metered is None:
        current = pack_usd + gpu_so_far
        remain_pre = remain_pack
    else:
        current = metered
        remain_pre = remain_pack
    gpu_runtime = _summarize_gpu_runtime()
    projected = project_complete_build(
        current_metered_usd=current,
        remaining_preprocess_usd=remain_pre,
        remaining_train_tokens=remaining,
        steady_tokens_per_sec=steady,
        eval_export_reserve_usd=EVAL_EXPORT_RESERVE_USD,
    )
    bottleneck = classify_probation_failure(
        finite=bool(
            finite
            and loss is not None
            and abs(float(loss)) != float("inf")
            and float(loss) == float(loss)
        ),
        checkpoint_ok=checkpoint_ok,
        resume_ok=bool(reload.get("ok")),
        steady_tokens_per_sec=steady,
        projected=projected,
    )
    if reload.get("ok") and int(reload.get("n_params") or 0) != EXPECTED_TRAINABLE_PARAMETERS_300M:
        bottleneck = bottleneck or "parameter_count_mismatch"
    decision = "continue" if bottleneck is None else "abort"
    report = {
        "decision": decision,
        "bottleneck": bottleneck,
        "tokens": tokens,
        "loss": loss,
        "finite": finite,
        "steady_tokens_per_sec": steady,
        "end_to_end_tokens_per_sec": e2e,
        "compile_warmup_s": warmup_s,
        "checkpoint": reload,
        "latest_valid": latest_blob,
        "projected": projected,
        "gpu_seconds": segment.get("elapsed_s"),
        "gpu_usd_so_far": gpu_so_far,
        "pack_usd_estimate": pack_usd,
        "metered_usd_snapshot": metered,
        "gpu_runtime": gpu_runtime,
        "single_trainer_lease_clear": not lease_fresh("train", max_age_s=120),
        "grad_norm": segment.get("last_grad_norm") or status.get("grad_norm"),
        "peak_allocated_vram_bytes": segment.get("peak_allocated_vram_bytes")
        or status.get("peak_allocated_vram_bytes"),
        "evaluated_at": utc_now(),
    }
    write_json(probation_path(), report)
    write_json(train_out_dir() / "probation.json", report)
    return report


def load_state() -> dict[str, Any]:
    path = state_path()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "run_id": RUN_ID,
        "started_at": utc_now(),
        "phase": "INIT",
        "desired_state": "RUNNING",
        "phases": {},
        "last_error": None,
    }


def save_state(state: dict[str, Any]) -> None:
    pipeline_dir().mkdir(parents=True, exist_ok=True)
    write_json(state_path(), state)
    write_json(
        desired_state_path(),
        {"desired_state": state.get("desired_state", "RUNNING"), "updated_at": utc_now()},
    )


def mark_phase(state: dict[str, Any], phase: str, status: str, **extra: Any) -> dict[str, Any]:
    rec = dict(state.setdefault("phases", {}).get(phase) or {})
    rec["status"] = status
    rec["updated_at"] = utc_now()
    rec.update(extra)
    state["phases"][phase] = rec
    state["phase"] = phase
    if "error" in extra:
        state["last_error"] = extra.get("error")
    if status == "complete":
        state["last_error"] = None
    save_state(state)
    return state


def phase_complete(state: dict[str, Any], phase: str) -> bool:
    rec = (state.get("phases") or {}).get(phase) or {}
    return rec.get("status") == "complete"


def phase_is_blocked(state: dict[str, Any], phase: str, *, policy_version: int | None = None) -> bool:
    rec = (state.get("phases") or {}).get(phase) or {}
    if rec.get("status") != "blocked":
        return False
    if policy_version is None:
        return True
    return rec.get("policy_version") == policy_version


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_snapshot() -> dict[str, Any]:
    from arzlm.paths import REPO_ROOT

    files = []
    for rel in (
        "configs/model-arzlm-300m.yaml",
        "configs/arzlm-300m-final-6b.yaml",
        "src/arzlm/model/config.py",
        "src/arzlm/training/loop.py",
        "data/security/denylist.json",
    ):
        path = REPO_ROOT / rel
        if path.is_file():
            files.append({"path": rel, "sha256": sha256_file(path)})
    return {
        "git_commit": git_commit(),
        "git_describe": git_describe(),
        "files": files,
        "created_at": utc_now(),
    }


def source_lock_ready() -> bool:
    path = source_lock_path()
    if not path.is_file():
        return False
    blob = path.read_text(encoding="utf-8")
    if "000_00041" in blob:
        return False
    lock = json.loads(blob)
    if lock.get("corpus") != "ArzLM-STEM-6B-v1":
        return False
    sources = lock.get("sources") or {}
    if any(d not in sources for d in DOMAINS):
        return False
    supplements = lock.get("supplements") or []
    prefixes = {str(s.get("prefix") or "") for s in supplements}
    if not any("cosmopedia-v2" in p for p in prefixes):
        return False
    # fineweb-edu-dedup is listed (234 parquet) but currently 0 Hub-AV-safe
    # files, so it cannot enter the lock under fail-closed policy.
    return True


def run_source_lock() -> dict[str, Any]:
    from arzlm.data.security.audit import refresh_source_lock, require_remote_security_audit
    from arzlm.data.security.lockfile import load_source_lock
    from arzlm.paths import SECURITY_DIR

    dest = source_lock_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.environ["ARZLM_SOURCE_LOCK"] = str(dest)
    denylist_src = SECURITY_DIR / "denylist.json"
    if denylist_src.is_file():
        shutil.copy2(denylist_src, dest.parent / "denylist.json")
    refresh_source_lock(dest, scale="6b")
    os.environ["ARZLM_SOURCE_LOCK"] = str(dest)
    lock = load_source_lock(dest)
    report = require_remote_security_audit(lock=lock)
    blob = dest.read_text(encoding="utf-8")
    if "000_00041" in blob:
        raise RuntimeError("denied FineWeb shard leaked into the 6B source lock")
    summary = {
        "lock_path": str(dest),
        "corpus": lock.corpus,
        "n_sources": len(lock.sources),
        "n_supplements": len(lock.supplements),
        "general_files": len(lock.sources["general"].files) if "general" in lock.sources else 0,
        "supplement_files": sum(len(s.files) for s in lock.supplements),
        "audit_passed": report.passed,
        "sha256": sha256_file(dest),
    }
    return summary


def run_tokenizer_sample(*, train_chars: int = 100_000_000) -> dict[str, Any]:
    from arzlm.data.sources import iter_finemath_4plus, iter_general_locked, iter_science_locked, iter_stack_edu_language
    from arzlm.data.catalog import CODE_LANGUAGE_WEIGHTS
    from arzlm.tokenizer.stem_study import take_chars

    os.environ["ARZLM_SOURCE_LOCK"] = str(source_lock_path())
    sample_root = tokenizer_sample_dir()
    sample_root.mkdir(parents=True, exist_ok=True)
    train_budget = {d: int(train_chars * MIXTURE_TARGET[d]) for d in DOMAINS}
    held_budget = {d: 2_000_000 for d in DOMAINS}
    langs = tuple(CODE_LANGUAGE_WEIGHTS)

    def _code():
        for lang in langs:
            yield from iter_stack_edu_language(lang, fetch_workers=int(os.environ.get("ARZLM_SWH_WORKERS", "8")))

    streams = {
        "general": iter_general_locked(),
        "math": iter_finemath_4plus(),
        "science": iter_science_locked(),
        "code": _code(),
    }
    meta: dict[str, Any] = {"train": {}, "held": {}, "mixture_target": dict(MIXTURE_TARGET)}
    for domain in DOMAINS:
        held_path = sample_root / f"{domain}.held.txt"
        train_path = sample_root / f"{domain}.train.txt"
        if train_path.is_file() and train_path.stat().st_size > 0 and held_path.is_file():
            meta["held"][domain] = {"chars": held_path.stat().st_size, "reused": True}
            meta["train"][domain] = {"chars": train_path.stat().st_size, "reused": True}
            continue
        h_text, n_h, n_hd = take_chars(streams[domain], held_budget[domain])
        t_text, n_t, n_td = take_chars(streams[domain], train_budget[domain])
        held_path.write_text(h_text, encoding="utf-8")
        train_path.write_text(t_text, encoding="utf-8")
        meta["held"][domain] = {"chars": n_h, "docs": n_hd}
        meta["train"][domain] = {"chars": n_t, "docs": n_td}
        if n_t <= 0:
            raise RuntimeError(f"tokenizer sample for {domain} produced 0 chars")
        from arzlm.cloud.runtime import maybe_commit_pack

        maybe_commit_pack({"tokenizer_sample": domain}, force=True)
    write_json(sample_root / "sample-manifest.json", meta)
    return meta


def run_tokenizer_train() -> dict[str, Any]:
    from arzlm.tokenizer.train import VOCAB_SIZE, save_litgpt_tokenizer_files, train_bpe_from_files
    from litgpt.tokenizer import Tokenizer as LitTokenizer

    sample_root = tokenizer_sample_dir()
    files = [sample_root / f"{d}.train.txt" for d in DOMAINS]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"tokenizer sample missing: {missing}")
    dest = tokenizer_dir()
    dest.mkdir(parents=True, exist_ok=True)
    trained = train_bpe_from_files(files, vocab_size=VOCAB_SIZE)
    save_litgpt_tokenizer_files(trained, dest)
    lit = LitTokenizer(dest)
    digest = sha256_file(dest / "tokenizer.json")
    (dest / "tokenizer.sha256").write_text(digest + "\n", encoding="utf-8")
    info = {
        "dir": str(dest),
        "vocab_size": int(lit.vocab_size),
        "eos_id": int(lit.eos_id) if lit.eos_id is not None else None,
        "sha256": digest,
        "identity": "arzlm-stem-32k-v2",
    }
    if info["vocab_size"] != 32000:
        raise RuntimeError(f"tokenizer vocab {info['vocab_size']} != 32000")
    write_json(dest / "tokenizer-meta.json", info)
    return info


def run_corpus_validate() -> dict[str, Any]:
    from arzlm.data.corpus_validate import validate_stem_corpus

    report = validate_stem_corpus(
        corpus_dir(),
        tokenizer_dir=tokenizer_dir(),
        train_quota=TRAIN_TOKEN_QUOTA_6B,
        val_quota=VAL_TOKEN_QUOTA_6B,
        vocab_size=32000,
        min_fraction=0.98,
    )
    samples = decode_category_samples(corpus_dir(), tokenizer_dir())
    report["decoded_samples"] = samples
    write_json(corpus_dir() / "validation-report.json", report)
    return report


def repair_packed_exact_duplicates() -> dict[str, Any]:
    """Idempotent: drop exact packed-document copies, then mark done."""
    from arzlm.data.packed_dedupe import MARKER_NAME, repair_corpus_exact_duplicates

    marker = corpus_dir() / "state" / MARKER_NAME
    if marker.is_file():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        existing["skipped"] = True
        return existing
    return repair_corpus_exact_duplicates(corpus_dir(), eos_id=2)


def decode_category_samples(corpus: Path, tok_dir: Path, n: int = 2) -> dict[str, list[str]]:
    import numpy as np
    import torch
    from litgpt.tokenizer import Tokenizer as LitTokenizer

    tok = LitTokenizer(tok_dir)
    out: dict[str, list[str]] = {}
    for domain in DOMAINS:
        texts: list[str] = []
        train = corpus / "train" / domain
        bins = sorted(train.glob("shard-*.bin")) if train.is_dir() else []
        for path in bins[:1]:
            data = np.memmap(path, dtype=np.uint16, mode="r")
            if data.size < 64:
                continue
            ids = torch.from_numpy(np.asarray(data[:256], dtype=np.int64).copy())
            texts.append(tok.decode(ids)[:800])
            if len(texts) >= n:
                break
        if not texts:
            raise RuntimeError(f"could not decode packed samples for {domain}")
        out[domain] = texts
    return out


def run_model_preflight() -> dict[str, Any]:
    from arzlm.model.config import build_gpt
    from arzlm.paths import REPO_ROOT
    from arzlm.training.config import load_experiment_config
    from arzlm.training.optim.groups import partition_litgpt_parameters

    cfg_path = REPO_ROOT / "configs" / "model-arzlm-300m.yaml"
    config = load_model_config(cfg_path)
    model = build_gpt(config, tie_embeddings=True)
    counts = count_parameters(config, model=model, tie_embeddings=True)
    assert_parameter_budget(counts, name=ARZLM_300M_NAME)
    if counts["trainable"] != EXPECTED_TRAINABLE_PARAMETERS_300M:
        raise RuntimeError("parameter count mismatch")
    if config.block_size != 2048:
        raise RuntimeError(f"block_size {config.block_size} != 2048")
    tok_meta = json.loads((tokenizer_dir() / "tokenizer-meta.json").read_text(encoding="utf-8"))
    if int(tok_meta["vocab_size"]) != int(config.vocab_size):
        raise RuntimeError("tokenizer/model vocab mismatch")
    parts = partition_litgpt_parameters(model)
    exp = load_experiment_config(REPO_ROOT / "configs" / "arzlm-300m-final-6b.yaml")
    meta = json.loads((corpus_dir() / "meta.json").read_text(encoding="utf-8"))
    if int(meta.get("seq_length") or 0) != 2048:
        raise RuntimeError(f"packed seq_length {meta.get('seq_length')} != 2048")
    return {
        "parameters": counts,
        "n_layer": config.n_layer,
        "n_embd": config.n_embd,
        "intermediate_size": config.intermediate_size,
        "n_head": config.n_head,
        "n_query_groups": config.n_query_groups,
        "head_size": config.head_size,
        "block_size": config.block_size,
        "vocab_size": config.vocab_size,
        "muon_n_params": parts["muon_n_params"],
        "adamw_n_params": parts["adamw_n_params"],
        "train_max_tokens": exp.train.max_tokens,
        "micro_batch_size": exp.train.micro_batch_size,
        "global_batch_size": exp.train.global_batch_size,
        "corpus_train_tokens": meta.get("train_tokens_total"),
    }


def latest_checkpoint_dir(out_dir: Path | None = None) -> Path:
    from litgpt.utils import find_resume_path

    root = out_dir or train_out_dir()
    final = root / "final" / "lit_model.pth"
    if final.is_file():
        return final.parent
    found = find_resume_path("auto", root)
    if found is None:
        raise FileNotFoundError(f"no checkpoint under {root}")
    return Path(found).parent


def _checkpoint_model_state(blob: Any) -> dict[str, Any]:
    import torch

    if isinstance(blob, dict) and "model" in blob:
        raw = blob["model"]
    else:
        raw = blob
    if hasattr(raw, "state_dict"):
        raw = raw.state_dict()
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint model payload is {type(raw)!r}, expected state_dict")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"non-tensor in model state: {key}={type(value)!r}")
        out[key] = value.detach().cpu().contiguous()
    return out


def _tokenizer_source(ckpt_dir: Path) -> Path:
    if (ckpt_dir / "tokenizer.json").is_file():
        return ckpt_dir
    return tokenizer_dir()


def _load_tied_gpt(ckpt_dir: Path, *, blob: Any | None = None) -> tuple[Any, Any, Any, dict[str, Any], int]:
    """Instantiate GPT with tied embeddings and load `strict=True` model weights."""
    import torch
    from litgpt.config import Config

    from arzlm.model.config import build_gpt

    ckpt = ckpt_dir / "lit_model.pth"
    if blob is None:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = _checkpoint_model_state(blob)
    cfg = Config.from_file(ckpt_dir / "model_config.yaml")
    if int(cfg.block_size) != 2048:
        raise RuntimeError(f"block_size {cfg.block_size} != 2048")
    model = build_gpt(cfg, tie_embeddings=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    n = int(count_parameters(cfg, model=model, tie_embeddings=True)["trainable"])
    if n != EXPECTED_TRAINABLE_PARAMETERS_300M:
        raise RuntimeError(f"parameter count {n} != {EXPECTED_TRAINABLE_PARAMETERS_300M}")
    return model, cfg, blob, state, n


def _write_packaged_tokenizer(src: Path, dest: Path, *, model_max_length: int = 2048) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "tokenizer.sha256",
        "tokenizer-meta.json",
    ):
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dest / name)
            copied.append(name)
    if "tokenizer.json" not in copied:
        raise FileNotFoundError(f"tokenizer.json missing under {src}")
    cfg_path = dest / "tokenizer_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "add_bos_token": False,
            "add_eos_token": False,
            "clean_up_tokenization_spaces": False,
        }
    cfg["model_max_length"] = int(model_max_length)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    gen_path = dest / "generation_config.json"
    if gen_path.is_file():
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
    else:
        gen = {}
    gen["max_length"] = int(model_max_length)
    gen["max_position_embeddings"] = int(model_max_length)
    gen_path.write_text(json.dumps(gen, indent=2) + "\n", encoding="utf-8")
    special = {
        "bos_token": cfg.get("bos_token"),
        "eos_token": cfg.get("eos_token"),
        "unk_token": cfg.get("unk_token"),
        "pad_token": cfg.get("pad_token"),
    }
    (dest / "special_tokens_map.json").write_text(json.dumps(special, indent=2) + "\n", encoding="utf-8")
    return {"copied": copied, "model_max_length": int(model_max_length)}


def run_evaluate() -> dict[str, Any]:
    from litgpt.tokenizer import Tokenizer as LitTokenizer

    from arzlm.cloud.prompting import greedy_complete

    ckpt_dir = latest_checkpoint_dir()
    ckpt = ckpt_dir / "lit_model.pth"
    tok = LitTokenizer(_tokenizer_source(ckpt_dir))
    model, _cfg, blob, _state, n = _load_tied_gpt(ckpt_dir)
    prompts = [
        "The derivative of x^2 is",
        "def binary_search(arr, target):",
        "Photosynthesis converts",
        "In English, a complete sentence must",
        "The capital of France is",
        "A prime number is",
    ]
    rows = [greedy_complete(model, tok, prompt, max_new_tokens=32) for prompt in prompts]
    step = blob.get("step_count") if isinstance(blob, dict) else None
    report = {
        "generated_at": utc_now(),
        "checkpoint": str(ckpt),
        "step": step,
        "parameters": n,
        "bos": False,
        "eos": False,
        "max_new_tokens": 32,
        "finite": True,
        "generations": rows,
    }
    dest = train_out_dir() / "eval" / "generations.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, report)
    return report


def run_export() -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    from arzlm.cloud.runtime import write_latest_valid

    export_root = artifacts_root() / "export" / "ArzLM-300M-Base"
    export_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = latest_checkpoint_dir()
    ckpt = ckpt_dir / "lit_model.pth"
    source_final = (train_out_dir() / "final").resolve()
    if ckpt_dir.resolve() != source_final and (source_final / "lit_model.pth").is_file():
        ckpt_dir = source_final
        ckpt = ckpt_dir / "lit_model.pth"
    model, cfg, blob, state, n = _load_tied_gpt(ckpt_dir)
    del model
    step = None
    if isinstance(blob, dict) and blob.get("step_count") is not None:
        step = int(blob["step_count"])

    inference = export_root / "LITGPT_INFERENCE"
    if inference.resolve() == ckpt_dir.resolve():
        raise RuntimeError(f"refusing to overwrite source checkpoint {ckpt_dir}")
    if inference.exists():
        shutil.rmtree(inference)
    inference.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_dir / "model_config.yaml", inference / "model_config.yaml")
    weights_only = {
        "model": state,
        "step_count": step,
        "iter_num": blob.get("iter_num") if isinstance(blob, dict) else None,
    }
    torch.save(weights_only, inference / "lit_model.pth")
    tok_src = _tokenizer_source(ckpt_dir)
    tok_meta = _write_packaged_tokenizer(tok_src, inference, model_max_length=2048)

    hf_dir = export_root / "huggingface"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)
    hf_dir.mkdir(parents=True, exist_ok=True)
    _write_packaged_tokenizer(tok_src, hf_dir, model_max_length=2048)
    shutil.copy2(ckpt_dir / "model_config.yaml", hf_dir / "model_config.yaml")
    hf_tensors: dict[str, torch.Tensor] = {}
    seen_ptrs: dict[int, str] = {}
    for key, tensor in state.items():
        ptr = int(tensor.data_ptr())
        if ptr in seen_ptrs:
            hf_tensors[key] = tensor.detach().contiguous().clone()
        else:
            seen_ptrs[ptr] = key
            hf_tensors[key] = tensor.detach().contiguous()
    save_file(hf_tensors, str(hf_dir / "model.safetensors"))
    hf_config = {
        "model_type": "arzlm",
        "architectures": ["ArzLMForCausalLM"],
        "vocab_size": int(cfg.vocab_size),
        "hidden_size": int(cfg.n_embd),
        "intermediate_size": int(cfg.intermediate_size),
        "num_hidden_layers": int(cfg.n_layer),
        "num_attention_heads": int(cfg.n_head),
        "num_key_value_heads": int(cfg.n_query_groups),
        "head_dim": int(cfg.head_size),
        "hidden_act": "silu",
        "rms_norm_eps": float(cfg.norm_eps),
        "rope_theta": float(cfg.rope_base),
        "max_position_embeddings": 2048,
        "tie_word_embeddings": True,
        "qk_norm": True,
        "qk_norm_type": str(cfg.norm_qk_type),
        "torch_dtype": "bfloat16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "unk_token_id": 0,
        "load_with": "LitGPT GPT + model_config.yaml or LITGPT_INFERENCE/lit_model.pth",
        "auto_model_note": "Not a stock transformers LlamaForCausalLM checkpoint (QK-Norm + fused qkv).",
    }
    write_json(hf_dir / "config.json", hf_config)

    tok_only = export_root / "tokenizer"
    if tok_only.exists():
        shutil.rmtree(tok_only)
    _write_packaged_tokenizer(tok_src, tok_only, model_max_length=2048)

    native = export_root / "LITGPT_NATIVE"
    if native.exists():
        shutil.rmtree(native)
    native.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inference / "lit_model.pth", native / "lit_model.pth")
    shutil.copy2(inference / "model_config.yaml", native / "model_config.yaml")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"):
        src = inference / name
        if src.is_file():
            shutil.copy2(src, native / name)

    token_count = 6_000_214_016
    status_path = train_out_dir() / "status.json"
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            token_count = int(status.get("training_tokens") or token_count)
            if step is None and status.get("optimizer_step") is not None:
                step = int(status["optimizer_step"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    write_latest_valid(train_out_dir(), ckpt_dir, token_count, int(step or 0))
    meta = {
        "model": ARZLM_300M_NAME,
        "parameters": n,
        "base_pretraining_context_length": 2048,
        "checkpoint": str(ckpt),
        "source_of_truth": str(ckpt_dir),
        "litgpt_inference": str(inference),
        "huggingface": str(hf_dir),
        "exported_at": utc_now(),
        "tokenizer": "arzlm-stem-32k-v2",
        "tokenizer_model_max_length": 2048,
        "full_checkpoint": False,
        "optimizer_state_included": False,
        "source_checkpoint_had_optimizer": isinstance(blob, dict)
        and any(k in blob for k in ("optimizer", "optim", "optimizer_states")),
        "step_count": step,
        "continued_pretraining": {
            "in_this_build": False,
            "intended_later": [4096, 8192],
        },
        "tokenizer_packaging": tok_meta,
    }
    write_json(export_root / "EXPORT_META.json", meta)
    return {
        "export_dir": str(export_root),
        "litgpt_inference": str(inference),
        "huggingface": str(hf_dir),
        "parameters": n,
        "step": step,
    }


def run_final_smoke() -> dict[str, Any]:
    from litgpt.tokenizer import Tokenizer as LitTokenizer
    from safetensors.torch import load_file

    from arzlm.cloud.prompting import greedy_complete

    export_root = artifacts_root() / "export" / "ArzLM-300M-Base"
    native = export_root / "LITGPT_INFERENCE"
    hf_dir = export_root / "huggingface"
    import torch

    tok = LitTokenizer(native)
    blob_probe = torch.load(native / "lit_model.pth", map_location="cpu", weights_only=False)
    if isinstance(blob_probe, dict) and any(k in blob_probe for k in ("optimizer", "optim", "optimizer_states")):
        raise RuntimeError("inference checkpoint still contains optimizer state")
    model, cfg, _blob, _state, n = _load_tied_gpt(native, blob=blob_probe)
    tok_cfg = json.loads((native / "tokenizer_config.json").read_text(encoding="utf-8"))
    if int(tok_cfg.get("model_max_length") or 0) != 2048:
        raise RuntimeError(f"exported tokenizer_config model_max_length={tok_cfg.get('model_max_length')} != 2048")
    if int(cfg.block_size) != 2048:
        raise RuntimeError(f"exported block_size {cfg.block_size} != 2048")

    st = load_file(str(hf_dir / "model.safetensors"))
    missing = sorted(set(model.state_dict()) - set(st))
    extra = sorted(set(st) - set(model.state_dict()))
    if missing or extra:
        raise RuntimeError(f"safetensors key mismatch missing={missing[:8]} extra={extra[:8]}")
    for key, tensor in model.state_dict().items():
        if tuple(st[key].shape) != tuple(tensor.shape):
            raise RuntimeError(f"safetensors shape mismatch {key}: {tuple(st[key].shape)} vs {tuple(tensor.shape)}")

    prompts = [
        "The capital of France is",
        "The derivative of x^2 is",
        "def binary_search(arr, target):",
        "Photosynthesis converts",
        "A prime number is",
    ]
    rows = [greedy_complete(model, tok, prompt, max_new_tokens=24) for prompt in prompts]
    payload = {
        "passed": True,
        "parameters": n,
        "context_length": 2048,
        "tokenizer_model_max_length": 2048,
        "optimizer_state_included": False,
        "safetensors_keys": len(st),
        "sample": rows[0]["completion"],
        "generations": rows,
        "tested_at": utc_now(),
    }
    write_json(export_root / "FINAL_SMOKE.json", payload)
    return payload


def write_build_complete(extra: dict[str, Any] | None = None) -> Path:
    dest = artifacts_root() / "BUILD_COMPLETE.json"
    payload = {
        "run_id": RUN_ID,
        "pipeline_id": PIPELINE_ID,
        "model_name": ARZLM_300M_NAME,
        "parameter_count": EXPECTED_TRAINABLE_PARAMETERS_300M,
        "context": 2048,
        "base_pretraining_context_length": 2048,
        "tokenizer": "arzlm-stem-32k-v2",
        "completed_at": utc_now(),
        **(extra or {}),
    }
    write_json(dest, payload)
    write_json(pipeline_dir() / "BUILD_COMPLETE.json", payload)
    return dest

