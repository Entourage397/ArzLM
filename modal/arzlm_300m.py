"""Modal orchestration for ArzLM-300M / 6B.

The training loop itself remains LitGPT + Lightning Fabric in
`arzlm.training.loop`. This module owns Images, Volumes, CPU workers,
the A100 segment, and the autonomous supervisor.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

LOCAL_REPO = Path(__file__).resolve().parents[1]

import modal

APP_NAME = "arzlm-300m-6b"
DOMAINS = ("general", "math", "code", "science")

DATA_NAME = "arzlm-300m-6b-data-v1"
ARTIFACTS_NAME = "arzlm-300m-6b-artifacts-v1"
CACHE_NAME = "arzlm-hf-cache-v1"

IGNORE = [
    ".git",
    ".venv",
    ".venv-gcc",
    ".venv-clang-broken",
    "runs",
    "hf-cache",
    "logs",
    "state",
    "data/prepared",
    "__pycache__",
    ".pytest_cache",
    ".cursor",
    "*.pyc",
]

CPU_PACKAGES = [
    "litgpt==0.5.13",
    "lightning==2.6.5",
    "tokenizers==0.23.2",
    "datasets==3.6.0",
    "huggingface-hub==1.3.7",
    "safetensors==0.8.0",
    "pyyaml==6.0.3",
    "tqdm==4.70.0",
    "numpy==2.5.3",
    "tensorboard==2.21.0",
    "jsonargparse==4.41.0",
    "pyarrow==25.0.1",
    "sympy==1.14.0",
]


def _image(torch_pkg: str, index_url: str) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install(torch_pkg, index_url=index_url)
        .pip_install(*CPU_PACKAGES)
        .env(
            {
                "PYTHONPATH": "/opt/arzlm/src",
                "ARZLM_REPO_ROOT": "/opt/arzlm",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            }
        )
        .add_local_dir(str(LOCAL_REPO), "/opt/arzlm", copy=True, ignore=IGNORE)
        .workdir("/opt/arzlm")
    )


cpu_image = _image("torch==2.11.0", "https://download.pytorch.org/whl/cpu")
gpu_image = _image("torch==2.11.0+cu128", "https://download.pytorch.org/whl/cu128").env(
    {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "ARZLM_SWH_WORKERS": "16",
    }
)

app = modal.App(APP_NAME)
DATA = modal.Volume.from_name(DATA_NAME, create_if_missing=True)
ARTIFACTS = modal.Volume.from_name(ARTIFACTS_NAME, create_if_missing=True)
HF_CACHE = modal.Volume.from_name(CACHE_NAME, create_if_missing=True)
LOCKS = modal.Dict.from_name("arzlm-300m-6b-locks-v1", create_if_missing=True)

VOLUMES = {
    "/vol/data": DATA,
    "/vol/artifacts": ARTIFACTS,
    "/vol/hf": HF_CACHE,
}

CPU_KW = dict(
    image=cpu_image,
    volumes=VOLUMES,
    timeout=24 * 60 * 60,
    cpu=8,
    memory=16384,
)


def _cloud_env():
    os.environ["ARZLM_REPO_ROOT"] = "/opt/arzlm"
    os.environ["ARZLM_DATA_ROOT"] = "/vol/data"
    os.environ["ARZLM_ARTIFACTS_ROOT"] = "/vol/artifacts"
    os.environ["ARZLM_SECURITY_DIR"] = "/opt/arzlm/data/security"
    os.environ["HF_HOME"] = "/vol/hf"
    os.environ.setdefault("HF_HUB_CACHE", "/vol/hf/hub")
    os.environ.setdefault("HF_XET_LOG_DIR", "/tmp/xet-logs")
    Path("/tmp/xet-logs").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ARZLM_SWH_WORKERS", "16")
    os.environ.setdefault("ARZLM_SKIP_PACK_AUDIT", "1")
    from arzlm.cloud import pipeline as P

    os.environ["ARZLM_SOURCE_LOCK"] = str(P.source_lock_path())
    os.environ["ARZLM_STATUS_PATH"] = str(P.train_out_dir() / "status.json")
    os.environ["ARZLM_STOP_PATH"] = str(P.pipeline_dir() / "STOP_REQUESTED")
    os.environ["ARZLM_DESIRED_STATE_PATH"] = str(P.desired_state_path())
    return P


PACK_LOCK_TTL_S = 1800.0
PACK_STALE_S = 1800.0
TRAIN_LOCK_TTL_S = 1800.0


def _domain_pack_status(domain: str) -> str:
    from arzlm.cloud import pipeline as P

    path = P.corpus_dir() / "state" / f"{domain}.json"
    if not path.is_file():
        return "missing"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "missing"
    return str(rec.get("status") or "unknown")


def _stale_in_progress(domain: str, max_age_s: float = PACK_STALE_S) -> bool:
    from arzlm.cloud import pipeline as P

    if _domain_pack_status(domain) == "complete":
        return False
    lease = P.corpus_dir() / "state" / f"{domain}.lease.json"
    if lease.is_file():
        try:
            rec = json.loads(lease.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return True
        age = P.iso_age_s(rec.get("heartbeat_at"))
        if age is not None:
            return age >= max_age_s
    path = P.corpus_dir() / "state" / f"{domain}.json"
    if not path.is_file():
        return True
    age = time.time() - path.stat().st_mtime
    return age > max_age_s


def _commit_hf() -> None:
    try:
        HF_CACHE.commit()
    except Exception as exc:
        print(f"HF cache volume commit skipped: {exc}", flush=True)


def _commit_all() -> None:
    DATA.commit()
    ARTIFACTS.commit()
    _commit_hf()


def _commit_artifacts() -> None:
    ARTIFACTS.commit()


def _lock_holder() -> str:
    return os.environ.get("MODAL_TASK_ID") or os.environ.get("HOSTNAME") or f"pid-{os.getpid()}"


def _acquire_lock(key: str, *, ttl_s: float = PACK_LOCK_TTL_S) -> bool:
    from arzlm.cloud.pipeline import named_lock_is_held

    me = _lock_holder()
    now = time.time()
    existing = LOCKS.get(key)
    if named_lock_is_held(existing, now=now, ttl_s=ttl_s, holder=me):
        return False
    LOCKS[key] = {"holder": me, "t": now}
    time.sleep(1.5)
    again = LOCKS.get(key)
    return isinstance(again, dict) and again.get("holder") == me


def _refresh_lock(key: str) -> None:
    LOCKS[key] = {"holder": _lock_holder(), "t": time.time()}


def _release_lock(key: str) -> None:
    me = _lock_holder()
    existing = LOCKS.get(key)
    if isinstance(existing, dict) and existing.get("holder") not in {None, me}:
        return
    try:
        LOCKS.pop(key)
    except Exception:
        pass


@app.function(**CPU_KW)
def source_lock() -> dict:
    from arzlm.training.report import write_json

    P = _cloud_env()
    state = P.load_state()
    P.mark_phase(state, "SOURCE_LOCK", "running")
    summary = P.run_source_lock()
    snap = P.source_snapshot()
    write_json(P.artifacts_root() / "source_snapshot.json", snap)
    P.mark_phase(state, "SOURCE_LOCK", "complete", **summary)
    _commit_all()
    return summary


@app.function(**CPU_KW)
def tokenizer_sample() -> dict:
    P = _cloud_env()
    state = P.load_state()
    P.mark_phase(state, "TOKENIZER_SAMPLE", "running")
    meta = P.run_tokenizer_sample()
    P.mark_phase(state, "TOKENIZER_SAMPLE", "complete")
    _commit_all()
    return {"ok": True, "meta": {k: meta.get(k) for k in ("train", "held")}}


@app.function(**CPU_KW)
def tokenizer_train() -> dict:
    P = _cloud_env()
    state = P.load_state()
    P.mark_phase(state, "TOKENIZER_TRAIN", "running")
    info = P.run_tokenizer_train()
    P.mark_phase(state, "TOKENIZER_TRAIN", "complete", **info)
    _commit_all()
    return info


@app.function(**{**CPU_KW, "cpu": 16, "memory": 32768, "timeout": 24 * 60 * 60})
def pack_domain(domain: str) -> dict:
    from arzlm.cloud.runtime import maybe_commit_pack, set_after_pack_checkpoint
    from arzlm.data.stem_builder import pack_stem_domain_standalone
    from arzlm.training.report import utc_now, write_json

    P = _cloud_env()
    DATA.reload()
    HF_CACHE.reload()
    lock_key = f"pack-{domain}"
    state_path = P.corpus_dir() / "state" / f"{domain}.json"
    lease = P.corpus_dir() / "state" / f"{domain}.lease.json"
    if state_path.is_file():
        rec = json.loads(state_path.read_text(encoding="utf-8"))
        if rec.get("status") == "complete":
            return {"domain": domain, "status": "complete", "tokens_train": rec.get("tokens_train"), "skipped": True}
    if not _acquire_lock(lock_key, ttl_s=PACK_LOCK_TTL_S):
        return {"domain": domain, "skipped": True, "reason": "lock_held"}

    def _commit_pack(_payload: dict) -> None:
        write_json(
            lease,
            {
                "domain": domain,
                "holder": _lock_holder(),
                "heartbeat_at": utc_now(),
            },
        )
        _refresh_lock(lock_key)
        DATA.commit()
        _commit_hf()

    set_after_pack_checkpoint(_commit_pack, min_interval_s=600)
    P.corpus_dir().mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if not state_path.is_file():
        write_json(
            state_path,
            {"status": "in_progress", "domain": domain, "tokens_train": 0, "started_at": utc_now()},
        )
    _commit_pack({})
    try:
        result = pack_stem_domain_standalone(
            domain,
            preset="6b",
            source="remote",
            out_dir=P.corpus_dir(),
            tokenizer_dir=P.tokenizer_dir(),
            seq_length=2048,
            isolated_encode=True,
        )
        maybe_commit_pack({"status": "complete"}, force=True)
        return {"domain": domain, "tokens_train": result.get("tokens_train"), "status": result.get("status")}
    except Exception:
        maybe_commit_pack({"status": "error"}, force=True)
        raise
    finally:
        if lease.is_file():
            try:
                lease.unlink()
            except OSError:
                pass
        _release_lock(lock_key)
        try:
            DATA.commit()
            _commit_hf()
        except Exception as exc:
            print(f"pack volume commit skipped: {exc}", flush=True)


@app.function(**CPU_KW)
def corpus_reduce() -> dict:
    from litgpt.tokenizer import Tokenizer
    from arzlm.data.stem_builder import finalize_stem_corpus

    P = _cloud_env()
    finalize_stem_corpus(
        out_dir=P.corpus_dir(),
        tokenizer_dir=P.tokenizer_dir(),
        tokenizer=Tokenizer(P.tokenizer_dir()),
        preset="6b",
        source="remote",
        seed=42,
        seq_length=2048,
        shard_tokens=8_388_608,
    )
    DATA.commit()
    return {"corpus": str(P.corpus_dir())}


@app.function(**CPU_KW)
def corpus_validate() -> dict:
    P = _cloud_env()
    state = P.load_state()
    P.mark_phase(state, "CORPUS_VALIDATE", "running")
    report = P.run_corpus_validate()
    P.mark_phase(state, "CORPUS_VALIDATE", "complete", train_tokens=report.get("train_tokens_total"))
    _commit_all()
    return report


@app.function(**{**CPU_KW, "memory": 32768})
def model_preflight() -> dict:
    from arzlm.training.report import write_json

    P = _cloud_env()
    state = P.load_state()
    P.mark_phase(state, "MODEL_PREFLIGHT", "running")
    info = P.run_model_preflight()
    write_json(P.artifacts_root() / "preflight.json", info)
    P.mark_phase(state, "MODEL_PREFLIGHT", "complete", parameters=info["parameters"]["trainable"])
    _commit_artifacts()
    return info


def _assert_a100_40gb() -> dict:
    import torch

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    print("nvidia-smi:", smi.stdout.strip(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the trainer container")
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    from arzlm.cloud.pipeline import assert_a100_40gb_identity

    assert_a100_40gb_identity(name, mem_gb)
    return {
        "name": name,
        "mem_gb": mem_gb,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": subprocess.check_output(["python", "--version"], text=True).strip(),
    }


def _stage_packed() -> Path:
    from arzlm.cloud import pipeline as P

    src = P.corpus_dir()
    dst = Path("/tmp/arzlm-stem-6b-v1")
    marker = dst / "meta.json"
    if marker.is_file():
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"staging packed train/val {src} -> {dst}", flush=True)
    for name in ("train", "val"):
        part = src / name
        if part.exists():
            shutil.copytree(part, dst / name)
    meta = src / "meta.json"
    if meta.is_file():
        shutil.copy2(meta, dst / "meta.json")
    for extra in ("validation-report.json", "manifest.json"):
        path = src / extra
        if path.is_file():
            shutil.copy2(path, dst / extra)
    return dst


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    volumes=VOLUMES,
    timeout=24 * 60 * 60,
    cpu=8,
    memory=65536,
    max_containers=1,
)
def train_segment() -> dict:
    from arzlm.cloud.runtime import set_after_checkpoint, set_after_status, set_before_stop_check
    from arzlm.paths import REPO_ROOT
    from arzlm.training.config import load_experiment_config
    from arzlm.training.loop import run_training
    from arzlm.training.report import utc_now, write_json

    P = _cloud_env()
    ARTIFACTS.reload()
    DATA.reload()
    state = P.load_state()
    if str(state.get("desired_state", "RUNNING")).upper() in {"STOP_REQUESTED", "STOPPED", "COMPLETE"}:
        return {"skipped": True, "reason": "stopped"}
    if P.phase_complete(state, "TRAIN_300M_6B") or P.phase_complete(state, "COMPLETE"):
        return {"skipped": True, "reason": "train_already_complete"}
    status_early = P.train_out_dir() / "status.json"
    if status_early.is_file():
        try:
            early = json.loads(status_early.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            early = {}
        if early.get("budget_complete") or early.get("exit_reason") == "complete":
            return {"skipped": True, "reason": "token_budget_complete"}
    probation = P.load_probation()
    if probation and probation.get("decision") == "abort":
        return {"skipped": True, "reason": "probation_abort", "bottleneck": probation.get("bottleneck")}
    if P.trainer_heartbeat_fresh(max_age_s=1200):
        return {"skipped": True, "reason": "trainer_alive"}
    if not _acquire_lock("train-arzlm-300m-base-6b", ttl_s=TRAIN_LOCK_TTL_S):
        return {"skipped": True, "reason": "train_lock_held"}
    P.clear_lease("train-pending")
    try:
        gpu_info = _assert_a100_40gb()
    except Exception:
        _release_lock("train-arzlm-300m-base-6b")
        raise
    P.train_out_dir().mkdir(parents=True, exist_ok=True)
    prev_status = {}
    status_file = P.train_out_dir() / "status.json"
    if status_file.is_file():
        try:
            prev_status = json.loads(status_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev_status = {}
    write_json(P.train_out_dir() / "gpu.json", gpu_info)
    P.write_lease("train", extra={"gpu": gpu_info})
    write_json(
        P.train_out_dir() / "status.json",
        {
            "schema_version": 1,
            "run_id": P.RUN_ID,
            "phase": "TRAIN_300M_6B",
            "desired_state": "RUNNING",
            "heartbeat_at": utc_now(),
            "gpu": gpu_info,
            "optimizer_step": int(prev_status.get("optimizer_step") or 0),
            "training_tokens": int(prev_status.get("training_tokens") or 0),
            "target_tokens": int(prev_status.get("target_tokens") or 6_000_214_016),
            "latest_loss": prev_status.get("latest_loss"),
        },
    )
    P.mark_phase(state, "TRAIN_300M_6B", "running", gpu=gpu_info)
    _commit_artifacts()

    staged = _stage_packed()
    tok_src = P.tokenizer_dir()
    tok_dst = Path("/tmp/arzlm-stem-32k-v2")
    if tok_dst.exists():
        shutil.rmtree(tok_dst)
    shutil.copytree(tok_src, tok_dst)

    exp = load_experiment_config(REPO_ROOT / "configs" / "arzlm-300m-final-6b.yaml")
    exp.out_dir = P.train_out_dir()
    exp.data_dir = staged
    exp.tokenizer_dir = tok_dst
    exp.resume = "auto"
    override = P.pipeline_dir() / "micro_batch.json"
    if override.is_file():
        exp.train.micro_batch_size = int(json.loads(override.read_text())["micro_batch_size"])
    os.environ.pop("ARZLM_PROBATION_TOKENS", None)
    if not probation or probation.get("decision") != "continue":
        cloud_cfg = dict(exp.raw.get("cloud") or {})
        os.environ["ARZLM_PROBATION_TOKENS"] = str(int(cloud_cfg.get("probation_tokens") or 16_777_216))
        exp.train.max_time = float(cloud_cfg.get("probation_max_seconds") or 3600)

    def _after_ckpt(_path: Path) -> None:
        P.write_lease("train")
        _refresh_lock("train-arzlm-300m-base-6b")
        ARTIFACTS.commit()

    def _after_status(_payload: dict) -> None:
        P.write_lease("train")
        _refresh_lock("train-arzlm-300m-base-6b")
        ARTIFACTS.commit()

    def _reload_stop() -> None:
        ARTIFACTS.reload()

    set_after_checkpoint(_after_ckpt)
    set_after_status(_after_status)
    set_before_stop_check(_reload_stop, min_interval_s=30)
    P.train_out_dir().mkdir(parents=True, exist_ok=True)

    try:
        result = run_training(exp)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or ("CUDA" in str(exc) and "memory" in str(exc).lower()):
            current = int(exp.train.micro_batch_size)
            nxt = max(1, current // 2)
            write_json(override, {"micro_batch_size": nxt, "reason": "oom", "from": current})
            P.clear_lease("train")
            _release_lock("train-arzlm-300m-base-6b")
            ARTIFACTS.commit()
            return {"oom": True, "next_micro_batch": nxt, "error": str(exc)[:500]}
        P.clear_lease("train")
        _release_lock("train-arzlm-300m-base-6b")
        ARTIFACTS.commit()
        raise
    finally:
        P.clear_lease("train")
        _release_lock("train-arzlm-300m-base-6b")
        ARTIFACTS.commit()

    if result.get("budget_complete"):
        posttrain.spawn()
    elif result.get("exit_reason") in {"probation", "max_time", "non_finite"} or (
        (probation is None or probation.get("decision") not in {"continue", "abort"})
        and int(result.get("tokens") or 0) >= 10_000_000
    ):
        P.write_lease("probation-pending")
        ARTIFACTS.commit()
        evaluate_probation.spawn()
    return {
        "exit_reason": result.get("exit_reason"),
        "budget_complete": result.get("budget_complete"),
        "tokens": result.get("tokens"),
        "steps": result.get("steps"),
        "loss": result.get("loss"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "steady_tokens_per_sec": result.get("steady_tokens_per_sec"),
        "compile_warmup_s": result.get("compile_warmup_s"),
        "finite": result.get("finite"),
        "gpu": gpu_info,
    }


@app.function(**{**CPU_KW, "memory": 32768, "timeout": 2 * 60 * 60})
def evaluate_probation() -> dict:
    """CPU gate after the 10–20M token probation segment. No GPU."""
    P = _cloud_env()
    ARTIFACTS.reload()
    DATA.reload()
    if P.lease_fresh("probation", max_age_s=1800):
        return {"skipped": True, "reason": "probation_eval_alive"}
    existing = P.load_probation()
    if existing and existing.get("decision") == "continue":
        return {"skipped": True, "reason": "already_decided", "decision": "continue"}
    if (
        existing
        and existing.get("decision") == "abort"
        and existing.get("bottleneck") != "parameter_count_mismatch"
    ):
        return {"skipped": True, "reason": "already_decided", "decision": "abort"}
    if not _acquire_lock("probation-eval", ttl_s=1800.0):
        return {"skipped": True, "reason": "probation_eval_lock"}
    try:
        P.clear_lease("probation-pending")
        P.write_lease("probation")
        _commit_artifacts()
        report = P.run_evaluate_probation()
        state = P.load_state()
        P.mark_phase(
            state,
            "TRAIN_300M_6B",
            "running" if report.get("decision") == "continue" else "paused",
            probation=report.get("decision"),
            bottleneck=report.get("bottleneck"),
            projected_usd=(report.get("projected") or {}).get("projected_usd"),
        )
        _commit_artifacts()
        return report
    finally:
        P.clear_lease("probation")
        _release_lock("probation-eval")
        _commit_artifacts()


@app.function(**{**CPU_KW, "memory": 32768, "timeout": 4 * 60 * 60, "max_containers": 1})
def posttrain() -> dict:
    P = _cloud_env()
    ARTIFACTS.reload()
    DATA.reload()
    state = P.load_state()
    desired = str(state.get("desired_state") or "RUNNING").upper()
    if desired in {"STOP_REQUESTED", "STOPPED"}:
        P.clear_lease("posttrain")
        _commit_artifacts()
        return {"skipped": True, "reason": "stopped"}
    if P.phase_complete(state, "COMPLETE"):
        P.clear_lease("posttrain")
        _commit_artifacts()
        return {"skipped": True, "reason": "already_complete"}
    if not P.phase_complete(state, "TRAIN_300M_6B"):
        P.clear_lease("posttrain")
        _commit_artifacts()
        return {"skipped": True, "reason": "train_incomplete"}
    P.write_lease("posttrain")
    _commit_artifacts()
    try:
        P.mark_phase(state, "FINAL_VALIDATE", "running")
        reload = P.verify_checkpoint_reloadable()
        if not reload.get("ok"):
            raise RuntimeError(f"final checkpoint not reloadable: {reload.get('error')}")
        if int(reload.get("n_params") or 0) != P.EXPECTED_TRAINABLE_PARAMETERS_300M:
            raise RuntimeError(
                f"final checkpoint params {reload.get('n_params')} != {P.EXPECTED_TRAINABLE_PARAMETERS_300M}"
            )
        P.mark_phase(
            state,
            "FINAL_VALIDATE",
            "complete",
            checkpoint_dir=reload.get("checkpoint_dir"),
            n_params=reload.get("n_params"),
            step_count=reload.get("step_count"),
        )
        P.mark_phase(state, "EVALUATE", "running")
        eval_report = P.run_evaluate()
        P.mark_phase(state, "EVALUATE", "complete")
        P.mark_phase(state, "EXPORT", "running")
        export_report = P.run_export()
        P.mark_phase(state, "EXPORT", "complete")
        P.mark_phase(state, "FINAL_SMOKE", "running")
        smoke = P.run_final_smoke()
        P.mark_phase(state, "FINAL_SMOKE", "complete")
        complete = P.write_build_complete(
            {
                "eval": eval_report,
                "export": export_report,
                "smoke": smoke,
            }
        )
        state = P.load_state()
        state["desired_state"] = "COMPLETE"
        P.mark_phase(state, "COMPLETE", "complete", build_complete=str(complete))
        _commit_artifacts()
        return {"build_complete": str(complete)}
    except Exception as exc:
        state = P.load_state()
        P.mark_phase(
            state,
            str(state.get("phase") or "EVALUATE"),
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            error_class=type(exc).__name__,
        )
        _commit_artifacts()
        raise
    finally:
        P.clear_lease("posttrain")
        _commit_artifacts()


@app.function(**{**CPU_KW, "timeout": 24 * 60 * 60, "memory": 8192, "max_containers": 1})
def advance() -> dict:
    """Run the next unfinished CPU phase, then spawn packing/training as needed."""
    P = _cloud_env()
    ARTIFACTS.reload()
    DATA.reload()
    HF_CACHE.reload()
    state = P.load_state()
    if P.lease_fresh("advance", max_age_s=7200):
        return {"idle": True, "reason": "advance_alive", "phase": state.get("phase")}
    P.write_lease("advance")
    _commit_artifacts()

    def _commit_pack(_payload: dict) -> None:
        DATA.commit()
        _commit_hf()
        ARTIFACTS.commit()

    from arzlm.cloud.runtime import set_after_pack_checkpoint

    set_after_pack_checkpoint(_commit_pack, min_interval_s=60)
    try:
        return _advance_inner(P, state)
    finally:
        P.clear_lease("advance")
        _commit_artifacts()


def _spawn_train_once(P, reason: str) -> dict:
    if P.lease_fresh("train-pending", max_age_s=900):
        return {"idle": True, "reason": "train_spawn_pending"}
    if P.trainer_heartbeat_fresh(max_age_s=1800):
        return {"idle": True, "reason": "trainer_alive", "phase": "TRAIN_300M_6B"}
    P.write_lease("train-pending")
    _commit_artifacts()
    train_segment.spawn()
    return {"ran": reason}


def _advance_inner(P, state) -> dict:
    from arzlm.data.stem_builder import finalize_stem_corpus
    from arzlm.training.report import write_json

    desired = str(state.get("desired_state") or "RUNNING").upper()
    if desired in {"STOPPED", "COMPLETE"}:
        return {"idle": True, "desired": desired, "phase": state.get("phase")}
    if desired == "STOP_REQUESTED":
        return {"idle": True, "desired": desired}

    def _done(phase: str) -> bool:
        return P.phase_complete(state, phase)

    if not _done("INIT"):
        P.mark_phase(state, "INIT", "complete", snapshot=P.source_snapshot())
        _commit_artifacts()
        state = P.load_state()

    if not _done("SOURCE_LOCK") or not P.source_lock_ready():
        P.mark_phase(state, "SOURCE_LOCK", "running")
        summary = P.run_source_lock()
        write_json(P.artifacts_root() / "source_snapshot.json", P.source_snapshot())
        P.mark_phase(state, "SOURCE_LOCK", "complete", **summary)
        _commit_all()
        return {"ran": "source_lock", **summary}
    if not _done("TOKENIZER_SAMPLE"):
        P.mark_phase(state, "TOKENIZER_SAMPLE", "running")
        meta = P.run_tokenizer_sample()
        P.mark_phase(state, "TOKENIZER_SAMPLE", "complete")
        _commit_all()
        return {"ran": "tokenizer_sample"}
    if not _done("TOKENIZER_TRAIN"):
        P.mark_phase(state, "TOKENIZER_TRAIN", "running")
        info = P.run_tokenizer_train()
        P.mark_phase(state, "TOKENIZER_TRAIN", "complete", **info)
        _commit_all()
        return {"ran": "tokenizer_train", **info}
    if not _done("CORPUS_PREP"):
        if all(_domain_pack_status(d) == "complete" for d in DOMAINS):
            from litgpt.tokenizer import Tokenizer

            finalize_stem_corpus(
                out_dir=P.corpus_dir(),
                tokenizer_dir=P.tokenizer_dir(),
                tokenizer=Tokenizer(P.tokenizer_dir()),
                preset="6b",
                source="remote",
                seed=42,
                seq_length=2048,
                shard_tokens=8_388_608,
            )
            P.mark_phase(state, "CORPUS_PREP", "complete")
            _commit_all()
            return {"ran": "corpus_reduce"}
        spawned: list[str] = []
        skipped_live: list[str] = []
        for domain in DOMAINS:
            st = _domain_pack_status(domain)
            if st == "complete":
                continue
            if st == "in_progress" and not _stale_in_progress(domain):
                skipped_live.append(domain)
                continue
            pack_domain.spawn(domain)
            spawned.append(domain)
        P.mark_phase(state, "CORPUS_PREP", "running", spawned=spawned, live=skipped_live)
        _commit_all()
        return {"ran": "corpus_prep_spawned", "spawned": spawned, "live": skipped_live}
    if not _done("CORPUS_VALIDATE"):
        from arzlm.data.corpus_validate import CorpusValidationError, VALIDATION_POLICY_VERSION
        from arzlm.data.packed_dedupe import MARKER_NAME

        marker = P.corpus_dir() / "state" / MARKER_NAME
        if marker.is_file() and P.phase_is_blocked(
            state, "CORPUS_VALIDATE", policy_version=VALIDATION_POLICY_VERSION
        ):
            return {
                "idle": True,
                "reason": "corpus_validate_blocked",
                "error": state.get("last_error"),
                "policy_version": VALIDATION_POLICY_VERSION,
            }
        try:
            P.mark_phase(state, "CORPUS_VALIDATE", "running", policy_version=VALIDATION_POLICY_VERSION)
            if not marker.is_file():
                repair = P.repair_packed_exact_duplicates()
                _commit_all()
                print(
                    json.dumps(
                        {
                            "packed_exact_dedupe": {
                                "dropped_docs": repair.get("dropped_docs"),
                                "dropped_tokens": repair.get("dropped_tokens"),
                                "train_tokens_total": repair.get("train_tokens_total"),
                            }
                        }
                    ),
                    flush=True,
                )
            report = P.run_corpus_validate()
            P.mark_phase(
                state,
                "CORPUS_VALIDATE",
                "complete",
                train_tokens=report.get("train_tokens_total"),
                policy_version=VALIDATION_POLICY_VERSION,
                warnings=report.get("warnings") or [],
            )
            _commit_all()
            return {"ran": "corpus_validate", "train_tokens_total": report.get("train_tokens_total")}
        except CorpusValidationError as exc:
            P.mark_phase(
                state,
                "CORPUS_VALIDATE",
                "blocked",
                error=f"{type(exc).__name__}: {exc}",
                error_class=type(exc).__name__,
                policy_version=VALIDATION_POLICY_VERSION,
            )
            _commit_artifacts()
            return {
                "blocked": True,
                "phase": "CORPUS_VALIDATE",
                "error": str(exc),
                "policy_version": VALIDATION_POLICY_VERSION,
            }
        except Exception as exc:
            P.mark_phase(
                state,
                "CORPUS_VALIDATE",
                "running",
                error=f"{type(exc).__name__}: {exc}",
                error_class=type(exc).__name__,
                policy_version=VALIDATION_POLICY_VERSION,
            )
            _commit_artifacts()
            raise
    if not _done("MODEL_PREFLIGHT"):
        P.mark_phase(state, "MODEL_PREFLIGHT", "running")
        info = P.run_model_preflight()
        write_json(P.artifacts_root() / "preflight.json", info)
        P.mark_phase(state, "MODEL_PREFLIGHT", "complete", parameters=info["parameters"]["trainable"])
        _commit_artifacts()
        return {"ran": "model_preflight", "parameters": info["parameters"]["trainable"]}
    if not _done("TRAIN_300M_6B"):
        train_out = P.train_out_dir()
        status_file = train_out / "status.json"
        probation = P.load_probation()
        if probation and probation.get("decision") == "abort":
            if probation.get("bottleneck") != "parameter_count_mismatch":
                return {
                    "idle": True,
                    "reason": "probation_abort",
                    "bottleneck": probation.get("bottleneck"),
                    "projected": (probation.get("projected") or {}).get("projected_usd"),
                }
        if status_file.is_file():
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if status.get("budget_complete") or status.get("desired_state") == "COMPLETE":
                state = P.load_state()
                P.mark_phase(state, "TRAIN_300M_6B", "complete", tokens=status.get("training_tokens"))
                _commit_artifacts()
                posttrain.spawn()
                return {"ran": "train_complete_handoff"}
        if P.trainer_heartbeat_fresh(max_age_s=1800):
            return {"idle": True, "reason": "trainer_alive", "phase": "TRAIN_300M_6B"}
        if P.lease_fresh("probation", max_age_s=1800):
            return {"idle": True, "reason": "probation_eval_alive"}
        if P.lease_fresh("probation-pending", max_age_s=900):
            return {"idle": True, "reason": "probation_eval_pending"}
        if P.lease_fresh("train-pending", max_age_s=900):
            return {"idle": True, "reason": "train_spawn_pending"}
        if probation and probation.get("decision") == "continue":
            return _spawn_train_once(P, "train_segment_spawned")
        status = {}
        if status_file.is_file():
            status = json.loads(status_file.read_text(encoding="utf-8"))
        tokens = int(status.get("training_tokens") or 0)
        exit_reason = status.get("exit_reason")
        if (
            status.get("phase") == "TRAIN_SEGMENT_EXIT"
            or exit_reason in {"probation", "max_time", "non_finite", "stop_requested"}
            or tokens >= 10_000_000
        ):
            P.write_lease("probation-pending")
            _commit_artifacts()
            evaluate_probation.spawn()
            return {"ran": "evaluate_probation_spawned", "tokens": tokens, "exit_reason": exit_reason}
        return _spawn_train_once(P, "train_probation_spawned")
    if not _done("COMPLETE"):
        if P.lease_fresh("posttrain", max_age_s=14400):
            return {"idle": True, "reason": "posttrain_alive", "phase": state.get("phase")}
        P.write_lease("posttrain", extra={"reason": "spawn"})
        _commit_artifacts()
        posttrain.spawn()
        return {"ran": "posttrain_spawned"}
    return {"idle": True, "phase": state.get("phase")}


@app.function(
    image=cpu_image,
    volumes=VOLUMES,
    timeout=15 * 60,
    cpu=1,
    memory=2048,
    schedule=modal.Period(minutes=5),
)
def supervisor() -> dict:
    """Laptop-independent reconciler. Deployed on a 5-minute Period."""
    from arzlm.training.report import utc_now

    P = _cloud_env()
    try:
        DATA.reload()
        ARTIFACTS.reload()
        state = P.load_state()
        desired = str(state.get("desired_state") or "RUNNING").upper()
        if desired in {"STOP_REQUESTED", "STOPPED", "COMPLETE"} or P.phase_complete(state, "COMPLETE"):
            return {
                "supervisor": True,
                "idle": True,
                "desired": desired,
                "phase": state.get("phase"),
                "at": utc_now(),
            }
        call = advance.spawn()
        return {
            "supervisor": True,
            "spawned": "advance",
            "object_id": getattr(call, "object_id", None),
            "at": utc_now(),
        }
    except Exception as exc:
        return {"supervisor": True, "error": str(exc)[:800], "at": utc_now()}


@app.function(**{**CPU_KW, "timeout": 120, "cpu": 1, "memory": 1024})
def request_stop() -> dict:
    from arzlm.training.report import utc_now

    P = _cloud_env()
    state = P.load_state()
    state["desired_state"] = "STOP_REQUESTED"
    P.save_state(state)
    P.pipeline_dir().mkdir(parents=True, exist_ok=True)
    (P.pipeline_dir() / "STOP_REQUESTED").write_text(utc_now() + "\n", encoding="utf-8")
    _commit_artifacts()
    return {"desired_state": "STOP_REQUESTED"}


@app.function(**{**CPU_KW, "timeout": 120, "cpu": 1, "memory": 1024})
def request_resume() -> dict:
    P = _cloud_env()
    state = P.load_state()
    state["desired_state"] = "RUNNING"
    stop = P.pipeline_dir() / "STOP_REQUESTED"
    if stop.is_file():
        stop.unlink()
    P.save_state(state)
    _commit_artifacts()
    advance.spawn()
    return {"desired_state": "RUNNING", "spawned": "advance"}


@app.function(**{**CPU_KW, "timeout": 120, "cpu": 1, "memory": 1024})
def status() -> dict:
    from arzlm.training.report import utc_now

    P = _cloud_env()
    DATA.reload()
    ARTIFACTS.reload()
    state = P.load_state()
    train_status = None
    path = P.train_out_dir() / "status.json"
    if path.is_file():
        train_status = json.loads(path.read_text(encoding="utf-8"))
    return {"pipeline": state, "train": train_status, "at": utc_now()}


@app.local_entrypoint()
def main(action: str = "launch"):
    if action == "stop":
        print(request_stop.remote())
        return
    if action == "resume":
        print(request_resume.remote())
        return
    if action == "status":
        print(json.dumps(status.remote(), indent=2, default=str))
        return
    if action != "launch":
        raise SystemExit(f"unknown action {action!r}")
    print("launching ArzLM-300M/6B cloud pipeline", flush=True)
    call = advance.spawn()
    print({"spawned": "advance", "object_id": getattr(call, "object_id", None)}, flush=True)
    print("Supervisor Period(5m) will continue if this container exits.", flush=True)
    print("Laptop may disconnect.", flush=True)
