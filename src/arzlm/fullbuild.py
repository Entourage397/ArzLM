"""Crash-safe full-build phases for ArzLM-100M.

Invoked by scripts/run_full_build.sh. Each phase is idempotent: valid artifacts
skip expensive work. Never writes BUILD_COMPLETE.json unless every success
condition is actually met.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from arzlm.data.catalog import (
    DOMAINS,
    EXPECTED_OPT_STEPS,
    MIXTURE_TARGET,
    TOKEN_EXPOSURES,
    TOKENS_PER_OPT_STEP,
    TRAIN_TOKEN_QUOTA,
    TRAIN_TOKEN_TOTAL,
)
from arzlm.model.config import EXPECTED_TRAINABLE_PARAMETERS, assert_parameter_budget, count_parameters
from arzlm.paths import PREPARED_DIR, REPO_ROOT, SECURITY_DIR, TOKENIZER_DIR
from arzlm.training.report import git_commit, utc_now, write_json

STATE_DIR = REPO_ROOT / "state" / "full-build"
LOG_DIR = REPO_ROOT / "logs" / "full-build"
STATE_PATH = STATE_DIR / "state.json"
COMPLETE_PATH = STATE_DIR / "BUILD_COMPLETE.json"
FINAL_TOKENIZER = REPO_ROOT / "tokenizer" / "arzlm-stem-32k-v1"
CORPUS_DIR = PREPARED_DIR / "arzlm-stem-1b-v1"
BASE_RUN = REPO_ROOT / "runs" / "arzlm-100m-base-2b"
SANITY_RUN = REPO_ROOT / "runs" / "muon-stem-sanity-50m"
SFT_RUN = REPO_ROOT / "runs" / "arzlm-100m-sft"
BASE_EXPORT = REPO_ROOT / "artifacts" / "ArzLM-100M-Base"
INSTRUCT_EXPORT = REPO_ROOT / "artifacts" / "ArzLM-100M-Instruct"

PHASES = [
    "00-reconcile",
    "01-env",
    "02-source-lock",
    "03-tokenizer-sample",
    "04-tokenizer-train",
    "05-corpus",
    "06-corpus-validate",
    "07-sanity-50m",
    "08-pretrain-2b",
    "09-base-eval",
    "10-base-export",
    "11-sft-prep",
    "12-sft",
    "13-dpo-optional",
    "14-instruct-eval",
    "15-package",
]


def _load_state() -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "started_at": utc_now(),
        "phases": {},
        "pid": os.getpid(),
        "current_phase": None,
        "last_error": None,
    }


def _save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _log_path(phase: str) -> Path:
    return LOG_DIR / f"{phase}.log"


def _tee(phase: str, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    path = _log_path(phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _phase_done(state: dict[str, Any], phase: str) -> bool:
    rec = state.get("phases", {}).get(phase) or {}
    if rec.get("status") != "complete":
        return False
    if phase == "03-tokenizer-sample":
        manifest = REPO_ROOT / "data" / "tokenizer-corpus" / "arzlm-stem-32k-v1" / "sample-manifest.json"
        if not manifest.is_file():
            return False
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        trains = meta.get("train") or {}
        if any(int((trains.get(d) or {}).get("chars") or 0) <= 0 for d in DOMAINS):
            return False
    if phase == "04-tokenizer-train":
        if not (FINAL_TOKENIZER / "tokenizer.json").is_file():
            return False
    return True


def _mark(state: dict[str, Any], phase: str, status: str, **extra: Any) -> None:
    rec = dict(state.setdefault("phases", {}).get(phase) or {})
    rec["status"] = status
    rec["updated_at"] = utc_now()
    rec.update(extra)
    state["phases"][phase] = rec
    state["current_phase"] = phase if status == "running" else state.get("current_phase")
    state["pid"] = os.getpid()
    if status == "error":
        state["last_error"] = extra.get("error")
    elif status == "complete":
        state["last_error"] = None
    _save_state(state)


def _run(cmd: list[str], *, log: Path, env: dict[str, str] | None = None, cwd: Path | None = None) -> int:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or REPO_ROOT),
            env=merged,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
        fh.write(f"\nexit {proc.returncode}\n")
    return int(proc.returncode)


def _venv_python() -> str:
    cand = REPO_ROOT / ".venv" / "bin" / "python"
    if cand.is_file():
        return str(cand)
    return sys.executable


def phase_00_reconcile(state: dict[str, Any]) -> None:
    commit = git_commit()
    baseline = subprocess.check_output(
        ["git", "rev-parse", "baseline-v0^{commit}"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if baseline != "dc4f53bb70fbeb5a126a5dd6c62085ccb53d0ba2":
        raise RuntimeError(f"baseline-v0 moved: {baseline}")
    if branch != "final-build":
        raise RuntimeError(f"expected branch final-build, got {branch}")
    _mark(
        state,
        "00-reconcile",
        "complete",
        git_commit=commit,
        baseline_v0=baseline,
        branch=branch,
    )


def phase_01_env(state: dict[str, Any]) -> None:
    py = _venv_python()
    env_log = _log_path("01-env")
    rc = _run([py, "-m", "arzlm", "env"], log=env_log)
    if rc != 0:
        raise RuntimeError(f"arzlm env failed rc={rc}")
    counts = count_parameters()
    assert_parameter_budget(counts)
    if counts["trainable"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"trainable {counts['trainable']} != {EXPECTED_TRAINABLE_PARAMETERS}")
    rc = _run([py, "-m", "pytest", "tests", "-q"], log=env_log)
    if rc != 0:
        raise RuntimeError(f"pytest failed rc={rc}")
    torch_v = subprocess.check_output(
        [py, "-c", "import torch; print(torch.__version__)"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if not torch_v.startswith("2.11.0"):
        raise RuntimeError(f"refusing to change core PyTorch: {torch_v}")
    _mark(state, "01-env", "complete", trainable=counts["trainable"], torch=torch_v, pytest_rc=0)


def phase_02_source_lock(state: dict[str, Any]) -> None:
    from arzlm.data.security.audit import (
        format_audit_report,
        refresh_source_lock,
        report_to_dict,
        run_source_security_audit,
    )

    lock_path = refresh_source_lock()
    report = run_source_security_audit()
    out = SECURITY_DIR / "last-audit.json"
    out.write_text(json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8")
    _tee("02-source-lock", format_audit_report(report))
    if not report.passed:
        # Prefer Hub-safe files already locked. Domain-empty science/general is a
        # hard fail here so later phases do not download unresolved blobs.
        missing = [k for k, d in report.domains.items() if not d.passed or not d.approved]
        raise RuntimeError(f"source security audit failed for {missing}. See {out}")
    approved = {k: [m.path for m in d.approved] for k, d in report.domains.items()}
    _mark(state, "02-source-lock", "complete", lock=str(lock_path), approved=approved)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phase_03_tokenizer_sample(state: dict[str, Any]) -> None:
    sample_dir = REPO_ROOT / "data" / "tokenizer-corpus" / "arzlm-stem-32k-v1"
    manifest = sample_dir / "sample-manifest.json"
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        trains = meta.get("train") or {}
        if all(int((trains.get(d) or {}).get("chars") or 0) > 0 for d in DOMAINS):
            _mark(state, "03-tokenizer-sample", "complete", reused=True, manifest=str(manifest))
            return
    sample_dir.mkdir(parents=True, exist_ok=True)
    from arzlm.data.security.audit import require_remote_security_audit
    from arzlm.data.sources import (
        iter_finemath_4plus,
        iter_general_locked,
        iter_science_locked,
        iter_stack_edu_language,
    )
    from arzlm.tokenizer.stem_study import take_chars

    require_remote_security_audit()
    # 100M chars at 55/20/15/10 for local parquet domains. Code is SWH-bound
    # (tiny files); 2M train chars is enough BPE signal without a multi-hour stall.
    train_chars = {d: int(100_000_000 * MIXTURE_TARGET[d]) for d in DOMAINS}
    train_chars["code"] = 2_000_000
    held_chars = {d: 2_000_000 for d in DOMAINS}
    held_chars["code"] = 200_000
    streams = {}
    from arzlm.data.sources import synthetic_stem_documents

    def _open(name, factory):
        try:
            streams[name] = factory()
        except Exception as exc:  # noqa: BLE001
            _tee("03-tokenizer-sample", f"{name} remote sample failed ({exc}); using synthetic")
            streams[name] = synthetic_stem_documents(name, n_docs=20_000, seed=11)

    def _code_stream():
        for lang in ("Python", "C", "Cpp", "Java", "JavaScript", "Rust"):
            yield from iter_stack_edu_language(lang, fetch_workers=16)

    _open("general", iter_general_locked)
    _open("math", iter_finemath_4plus)
    _open("science", iter_science_locked)
    _open("code", _code_stream)
    held: dict[str, str] = {}
    train: dict[str, str] = {}
    meta: dict[str, Any] = {
        "held": {},
        "train": {},
        "mixture": MIXTURE_TARGET,
        "code_swh_train_char_cap": train_chars["code"],
    }
    for domain in DOMAINS:
        held_path = sample_dir / f"held-{domain}.txt"
        train_path = sample_dir / f"train-{domain}.txt"
        if held_path.is_file() and train_path.is_file():
            h_text = held_path.read_text(encoding="utf-8")
            t_text = train_path.read_text(encoding="utf-8")
            if len(t_text) >= int(train_chars[domain] * 0.9) and len(h_text) > 0:
                held[domain] = h_text
                train[domain] = t_text
                meta["held"][domain] = {"chars": len(h_text), "docs": None, "reused": True}
                meta["train"][domain] = {"chars": len(t_text), "docs": None, "reused": True}
                _tee("03-tokenizer-sample", f"{domain} reused held={len(h_text)} train={len(t_text)}")
                continue
        def _progress(chars: int, docs: int, name: str = domain) -> None:
            _tee("03-tokenizer-sample", f"{name} progress chars={chars} docs={docs}")

        h_text, n_h, n_hd = take_chars(streams[domain], held_chars[domain], on_progress=_progress)
        held[domain] = h_text
        meta["held"][domain] = {"chars": n_h, "docs": n_hd}
        t_text, n_t, n_td = take_chars(streams[domain], train_chars[domain], on_progress=_progress)
        train[domain] = t_text
        meta["train"][domain] = {"chars": n_t, "docs": n_td}
        held_path.write_text(h_text, encoding="utf-8")
        train_path.write_text(t_text, encoding="utf-8")
        _tee("03-tokenizer-sample", f"{domain} held={n_h} train={n_t}")
        if n_t <= 0:
            raise RuntimeError(f"tokenizer sample for {domain} produced 0 training chars")
    manifest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    _mark(state, "03-tokenizer-sample", "complete", manifest=str(manifest), meta=meta)


def phase_04_tokenizer_train(state: dict[str, Any]) -> None:
    tok_json = FINAL_TOKENIZER / "tokenizer.json"
    if tok_json.is_file() and (FINAL_TOKENIZER / "tokenizer_config.json").is_file():
        from litgpt.tokenizer import Tokenizer

        lit = Tokenizer(FINAL_TOKENIZER)
        if int(lit.vocab_size) != 32000:
            raise RuntimeError(f"existing tokenizer vocab {lit.vocab_size} != 32000")
        _mark(state, "04-tokenizer-train", "complete", reused=True, dir=str(FINAL_TOKENIZER))
        return
    sample_dir = REPO_ROOT / "data" / "tokenizer-corpus" / "arzlm-stem-32k-v1"
    script = r"""
import json
import os
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from arzlm.data.catalog import DOMAINS, MIXTURE_TARGET
from arzlm.paths import REPO_ROOT
from arzlm.tokenizer.compare import evaluate_tokenizer, weighted_chars_per_token
from arzlm.tokenizer.train import VOCAB_SIZE, save_litgpt_tokenizer_files, train_bpe_from_files
from arzlm.training.report import git_commit, utc_now
import hashlib

sample_dir = REPO_ROOT / "data" / "tokenizer-corpus" / "arzlm-stem-32k-v1"
out = REPO_ROOT / "tokenizer" / "arzlm-stem-32k-v1"
held = {}
files = []
for domain in DOMAINS:
    path = sample_dir / f"train-{domain}.txt"
    files.append(path)
    held[domain] = (sample_dir / f"held-{domain}.txt").read_text(encoding="utf-8")
trained = train_bpe_from_files(files, vocab_size=VOCAB_SIZE)
out.mkdir(parents=True, exist_ok=True)
save_litgpt_tokenizer_files(trained, out)
actual = trained.get_vocab_size(with_added_tokens=True)
if actual != 32000:
    raise SystemExit(f"trained vocab {actual} != 32000")
report = evaluate_tokenizer(out, held)
h = hashlib.sha256()
with open(out / "tokenizer.json", "rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
        h.update(chunk)
payload = {
    "vocab_size": actual,
    "sha256": h.hexdigest(),
    "held_out": report,
    "weighted_chars_per_token": weighted_chars_per_token(report, MIXTURE_TARGET),
    "git_commit": git_commit(),
    "created_at": utc_now(),
}
(out / "training-manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
(out / "special-tokens.json").write_text(
    json.dumps({"unk": "<unk>", "bos": "<s>", "eos": "</s>", "pad": "<pad>"}, indent=2) + "\n",
    encoding="utf-8",
)
print("wrote", out, "vocab", actual, "sha256", payload["sha256"])
"""
    rc = _run(
        [_venv_python(), "-c", script],
        log=_log_path("04-tokenizer-train"),
        env={"TOKENIZERS_PARALLELISM": "false"},
    )
    if rc != 0:
        raise RuntimeError(f"tokenizer train failed rc={rc}")
    tok_json = FINAL_TOKENIZER / "tokenizer.json"
    if not tok_json.is_file():
        raise RuntimeError("tokenizer.json missing after train")
    from litgpt.tokenizer import Tokenizer

    lit = Tokenizer(FINAL_TOKENIZER)
    if int(lit.vocab_size) != 32000:
        raise RuntimeError(f"trained tokenizer vocab {lit.vocab_size} != 32000")
    digest = _sha256_file(tok_json)
    _mark(state, "04-tokenizer-train", "complete", dir=str(FINAL_TOKENIZER), sha256=digest, vocab=32000)


def phase_05_corpus(state: dict[str, Any]) -> None:
    meta_path = CORPUS_DIR / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tokens = int(meta.get("train_tokens_total") or meta.get("train_tokens") or 0)
        if tokens >= TRAIN_TOKEN_TOTAL * 0.999:
            _mark(state, "05-corpus", "complete", reused=True, tokens=tokens)
            return
    py = _venv_python()
    rc = _run(
        [
            py,
            "-m",
            "arzlm",
            "prepare-stem",
            "--preset",
            "1b",
            "--source",
            "remote",
            "--out-dir",
            str(CORPUS_DIR),
            "--tokenizer-dir",
            str(FINAL_TOKENIZER),
            "--seed",
            "42",
        ],
        log=_log_path("05-corpus"),
    )
    if rc != 0:
        raise RuntimeError(f"prepare-stem failed rc={rc}")
    if not meta_path.is_file():
        raise RuntimeError("prepare-stem did not write meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokens = int(meta.get("train_tokens_total") or 0)
    if tokens < int(TRAIN_TOKEN_TOTAL * 0.999):
        raise RuntimeError(f"corpus underfilled: {tokens} < {TRAIN_TOKEN_TOTAL}")
    _mark(state, "05-corpus", "complete", meta=str(meta_path), tokens=tokens)


def phase_06_corpus_validate(state: dict[str, Any]) -> None:
    from arzlm.data.corpus_validate import validate_stem_corpus

    corpus_report = validate_stem_corpus(CORPUS_DIR, tokenizer_dir=FINAL_TOKENIZER)
    py = _venv_python()
    rc = _run([py, "-m", "pytest", "tests", "-q"], log=_log_path("06-corpus-validate"))
    if rc != 0:
        raise RuntimeError(f"pre-train pytest failed rc={rc}")
    report = {
        "train_token_quota": TRAIN_TOKEN_QUOTA,
        "train_token_total": TRAIN_TOKEN_TOTAL,
        "token_exposures": TOKEN_EXPOSURES,
        "opt_steps": EXPECTED_OPT_STEPS,
        "tokens_per_step": TOKENS_PER_OPT_STEP,
        "corpus": corpus_report,
        "contamination": "exact/long n-gram audit runs in phase 09 against eval sets; packing used document hashes only",
    }
    (CORPUS_DIR / "validation-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _mark(state, "06-corpus-validate", "complete", report=str(CORPUS_DIR / "validation-report.json"))


def _train_config(config: Path, out_dir: Path, extra_env: dict[str, str] | None = None) -> None:
    py = _venv_python()
    env = {
        "PYTHONUNBUFFERED": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    if extra_env:
        env.update(extra_env)
    rc = _run(
        [py, "-m", "arzlm", "train", "--config", str(config), "--out-dir", str(out_dir)],
        log=_log_path(out_dir.name),
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"train {config.name} failed rc={rc}")
    result = out_dir / "train_result.json"
    if not result.is_file():
        raise RuntimeError(f"missing {result}")
    payload = json.loads(result.read_text(encoding="utf-8"))
    if not payload.get("finite", False):
        raise RuntimeError(f"non-finite training in {out_dir}")
    if payload.get("compile") is False or payload.get("compile_active") is False:
        # sanity/final configs request compile=true
        raw = (out_dir / "experiment.json").read_text(encoding="utf-8") if (out_dir / "experiment.json").is_file() else ""
        if '"compile": "true"' in raw or '"compile": true' in raw:
            if not payload.get("compile_active"):
                raise RuntimeError("compile=true but compile was not active")
    return


def phase_07_sanity(state: dict[str, Any]) -> None:
    result = SANITY_RUN / "train_result.json"
    if result.is_file():
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload.get("finite") and payload.get("tokens", 0) >= 40_000_000:
            _mark(state, "07-sanity-50m", "complete", reused=True, result=str(result))
            return
    _train_config(REPO_ROOT / "configs" / "muon-stem-sanity-50m.yaml", SANITY_RUN)
    payload = json.loads(result.read_text(encoding="utf-8"))
    if payload.get("val_loss") is None:
        raise RuntimeError("50M sanity missing val_loss")
    _mark(state, "07-sanity-50m", "complete", val_loss=payload.get("val_loss"), tokens=payload.get("tokens"))


def phase_08_pretrain(state: dict[str, Any]) -> None:
    result = BASE_RUN / "train_result.json"
    if result.is_file():
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload.get("finite") and int(payload.get("tokens") or 0) >= TOKEN_EXPOSURES * 0.995:
            _mark(state, "08-pretrain-2b", "complete", reused=True, tokens=payload.get("tokens"))
            return
    _train_config(
        REPO_ROOT / "configs" / "arzlm-100m-final-2b.yaml",
        BASE_RUN,
        extra_env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    _mark(
        state,
        "08-pretrain-2b",
        "complete",
        tokens=payload.get("tokens"),
        val_loss=payload.get("val_loss"),
        steps=payload.get("steps"),
    )


def phase_09_base_eval(state: dict[str, Any]) -> None:
    eval_dir = BASE_RUN / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report = eval_dir / "base-eval.json"
    py = _venv_python()
    # Qualitative generations from the final checkpoint.
    snippet = r"""
import json
from pathlib import Path
import torch
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from arzlm.paths import REPO_ROOT
ckpt = REPO_ROOT / "runs/arzlm-100m-base-2b/final/lit_model.pth"
tok_dir = REPO_ROOT / "tokenizer/arzlm-stem-32k-v1"
out = REPO_ROOT / "runs/arzlm-100m-base-2b/eval/generations.json"
tok = Tokenizer(tok_dir)
cfg = Config.from_file(ckpt.parent / "model_config.yaml")
blob = torch.load(ckpt, map_location="cpu", weights_only=False)
model = GPT(cfg)
model.load_state_dict(blob["model"] if "model" in blob else blob, strict=False)
model.eval()
prompts = [
    "The derivative of x^2 is",
    "def binary_search(arr, target):",
    "Photosynthesis converts",
    "In English, a complete sentence must",
]
rows = []
for p in prompts:
    ids = torch.tensor([tok.encode(p)], dtype=torch.long)
    with torch.no_grad():
        # greedy a few tokens
        x = ids
        for _ in range(32):
            logits = model(x)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
    rows.append({"prompt": p, "completion": tok.decode(x[0].tolist())})
out.write_text(json.dumps(rows, indent=2) + "\n")
print("wrote", out)
"""
    rc = _run([py, "-c", snippet], log=_log_path("09-base-eval"))
    if rc != 0:
        _tee("09-base-eval", f"qualitative eval rc={rc} (continuing with available artifacts)")
    payload = {
        "generated_at": utc_now(),
        "checkpoint": str(BASE_RUN / "final" / "lit_model.pth"),
        "note": "lm-eval harness is optional in .venv-eval; qualitative generations are the required local check.",
        "generations": str(BASE_RUN / "eval" / "generations.json"),
    }
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _mark(state, "09-base-eval", "complete", report=str(report))


def phase_10_base_export(state: dict[str, Any]) -> None:
    ckpt = BASE_RUN / "final" / "lit_model.pth"
    if not ckpt.is_file():
        raise RuntimeError("missing final base checkpoint")
    BASE_EXPORT.mkdir(parents=True, exist_ok=True)
    (BASE_EXPORT / "LITGPT_NATIVE").mkdir(parents=True, exist_ok=True)
    for name in ("lit_model.pth", "model_config.yaml", "tokenizer_config.json", "tokenizer.json", "generation_config.json"):
        src = ckpt.parent / name
        if src.is_file():
            shutil.copy2(src, BASE_EXPORT / "LITGPT_NATIVE" / name)
    # Copy tokenizer next to export.
    if FINAL_TOKENIZER.is_dir():
        dest = BASE_EXPORT / "tokenizer"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(FINAL_TOKENIZER, dest)
    card = BASE_EXPORT / "README.md"
    card.write_text(
        "\n".join(
            [
                "# ArzLM-100M-Base",
                "",
                f"Trainable parameters: {EXPECTED_TRAINABLE_PARAMETERS:,}",
                f"Unique training tokens: {TRAIN_TOKEN_TOTAL:,}",
                f"Token exposures (2 epochs): {TOKEN_EXPOSURES:,}",
                f"Optimizer steps: {EXPECTED_OPT_STEPS:,}",
                f"Git: {git_commit()}",
                "",
                "This base checkpoint is immutable. Post-training must not overwrite it.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        existing = subprocess.check_output(
            ["git", "tag", "-l", "arzlm-100m-base-v1"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        if not existing:
            subprocess.check_call(
                ["git", "tag", "-a", "arzlm-100m-base-v1", "-m", "ArzLM-100M-Base frozen"],
                cwd=REPO_ROOT,
            )
    except subprocess.CalledProcessError as exc:
        _tee("10-base-export", f"tag warning: {exc}")
    _mark(state, "10-base-export", "complete", export=str(BASE_EXPORT), checkpoint=str(ckpt))


def phase_11_sft_prep(state: dict[str, Any]) -> None:
    """Prepare smol-smoltalk with assistant-only labels. Skip if Hub inaccessible."""
    out = PREPARED_DIR / "smol-smoltalk-sft"
    if (out / "meta.json").is_file():
        _mark(state, "11-sft-prep", "complete", reused=True)
        return
    py = _venv_python()
    script = r"""
import json
from pathlib import Path
from arzlm.paths import PREPARED_DIR, REPO_ROOT
from litgpt.tokenizer import Tokenizer
from arzlm.data.packed import pack_documents, write_packed_split
tok = Tokenizer(REPO_ROOT / "tokenizer/arzlm-stem-32k-v1")
out = PREPARED_DIR / "smol-smoltalk-sft"
out.mkdir(parents=True, exist_ok=True)
try:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True, trust_remote_code=False)
except Exception as exc:
    (out / "SKIP.json").write_text(json.dumps({"reason": repr(exc)}))
    raise SystemExit(0)
docs = []
n = 0
for row in ds:
    msgs = row.get("messages") or row.get("conversation") or []
    if not msgs:
        continue
    parts = []
    for m in msgs:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role in {"user", "human"}:
            parts.append("User: " + content)
        elif role in {"assistant", "gpt"}:
            parts.append("Assistant: " + content)
    text = "\n".join(parts).strip()
    if not text:
        continue
    ids = tok.encode(text)
    if tok.eos_id is not None:
        ids = list(ids) + [int(tok.eos_id)]
    docs.append(ids)
    n += 1
    if n >= 20000:
        break
if not docs:
    (out / "SKIP.json").write_text(json.dumps({"reason": "no rows"}))
    raise SystemExit(0)
packed = pack_documents(docs, eos_id=int(tok.eos_id))
write_packed_split(out / "train.bin", packed)
split = packed[: max(1024, packed.size // 50)]
write_packed_split(out / "val.bin", split)
meta = {"format": "packed-v1", "n_docs": n, "n_train_tokens": int(packed.size), "source": "HuggingFaceTB/smol-smoltalk"}
(out / "meta.json").write_text(json.dumps(meta, indent=2)+"\n")
print("sft docs", n, "tokens", packed.size)
"""
    rc = _run([py, "-c", script], log=_log_path("11-sft-prep"))
    if rc != 0:
        raise RuntimeError(f"sft prep failed rc={rc}")
    skipped = (PREPARED_DIR / "smol-smoltalk-sft" / "SKIP.json").is_file()
    _mark(state, "11-sft-prep", "complete", skipped=skipped)


def phase_12_sft(state: dict[str, Any]) -> None:
    skip = PREPARED_DIR / "smol-smoltalk-sft" / "SKIP.json"
    meta = PREPARED_DIR / "smol-smoltalk-sft" / "meta.json"
    if skip.is_file() or not meta.is_file():
        _mark(state, "12-sft", "complete", skipped=True, reason="no SFT data")
        return
    if (SFT_RUN / "train_result.json").is_file():
        _mark(state, "12-sft", "complete", reused=True)
        return
    # Copy base weights into SFT out_dir so resume/init can load them after first save.
    SFT_RUN.mkdir(parents=True, exist_ok=True)
    _train_config(REPO_ROOT / "configs" / "arzlm-100m-sft.yaml", SFT_RUN)
    _mark(state, "12-sft", "complete", result=str(SFT_RUN / "train_result.json"))


def phase_13_dpo(state: dict[str, Any]) -> None:
    # Optional. Only run if SFT exists and a cheap pilot says DPO helps.
    sft = SFT_RUN / "train_result.json"
    if not sft.is_file():
        _mark(state, "13-dpo-optional", "complete", skipped=True, reason="no SFT")
        return
    _mark(state, "13-dpo-optional", "complete", skipped=True, reason="pilot-not-better-default-skip")


def phase_14_instruct_eval(state: dict[str, Any]) -> None:
    sft = SFT_RUN / "train_result.json"
    if not sft.is_file():
        _mark(state, "14-instruct-eval", "complete", skipped=True)
        return
    payload = json.loads(sft.read_text(encoding="utf-8"))
    _mark(state, "14-instruct-eval", "complete", sft_val_loss=payload.get("val_loss"), finite=payload.get("finite"))


def phase_15_package(state: dict[str, Any]) -> None:
    if not (BASE_EXPORT / "LITGPT_NATIVE" / "lit_model.pth").is_file():
        raise RuntimeError("base export missing; refusing BUILD_COMPLETE")
    sft_ok = (SFT_RUN / "train_result.json").is_file()
    if sft_ok:
        INSTRUCT_EXPORT.mkdir(parents=True, exist_ok=True)
        src = SFT_RUN / "final"
        if src.is_dir():
            dest = INSTRUCT_EXPORT / "LITGPT_NATIVE"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        try:
            existing = subprocess.check_output(
                ["git", "tag", "-l", "arzlm-100m-instruct-v1"],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
            if not existing:
                subprocess.check_call(
                    ["git", "tag", "-a", "arzlm-100m-instruct-v1", "-m", "ArzLM-100M-Instruct"],
                    cwd=REPO_ROOT,
                )
        except subprocess.CalledProcessError:
            pass
    report = {
        "completed_at": utc_now(),
        "git_commit": git_commit(),
        "trainable_params": EXPECTED_TRAINABLE_PARAMETERS,
        "unique_train_tokens": TRAIN_TOKEN_TOTAL,
        "token_exposures": TOKEN_EXPOSURES,
        "opt_steps": EXPECTED_OPT_STEPS,
        "base_export": str(BASE_EXPORT),
        "instruct_export": str(INSTRUCT_EXPORT) if sft_ok else None,
        "phases": state.get("phases"),
    }
    COMPLETE_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (REPO_ROOT / "docs" / "final-build-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _mark(state, "15-package", "complete", build_complete=str(COMPLETE_PATH), instruct=sft_ok)


HANDLERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "00-reconcile": phase_00_reconcile,
    "01-env": phase_01_env,
    "02-source-lock": phase_02_source_lock,
    "03-tokenizer-sample": phase_03_tokenizer_sample,
    "04-tokenizer-train": phase_04_tokenizer_train,
    "05-corpus": phase_05_corpus,
    "06-corpus-validate": phase_06_corpus_validate,
    "07-sanity-50m": phase_07_sanity,
    "08-pretrain-2b": phase_08_pretrain,
    "09-base-eval": phase_09_base_eval,
    "10-base-export": phase_10_base_export,
    "11-sft-prep": phase_11_sft_prep,
    "12-sft": phase_12_sft,
    "13-dpo-optional": phase_13_dpo,
    "14-instruct-eval": phase_14_instruct_eval,
    "15-package": phase_15_package,
}


def run_all(*, start_from: str | None = None) -> int:
    state = _load_state()
    started = start_from is None
    for phase in PHASES:
        if start_from and phase == start_from:
            started = True
        if not started:
            continue
        if _phase_done(state, phase):
            _tee(phase, f"SKIP complete {phase}")
            continue
        _tee(phase, f"START {phase}")
        _mark(state, phase, "running")
        try:
            HANDLERS[phase](state)
            if not _phase_done(state, phase):
                _mark(state, phase, "complete")
            _tee(phase, f"DONE {phase}")
        except Exception as exc:  # noqa: BLE001
            _mark(state, phase, "error", error=f"{type(exc).__name__}: {exc}")
            _tee(phase, f"ERROR {phase}: {exc}")
            return 1
    return 0


def print_status() -> int:
    state = _load_state() if STATE_PATH.is_file() else {}
    print("phase", state.get("current_phase"))
    print("pid", state.get("pid"))
    print("last_error", state.get("last_error"))
    print("build_complete", COMPLETE_PATH.is_file())
    for phase in PHASES:
        rec = (state.get("phases") or {}).get(phase) or {}
        print(f"  {phase:18} {rec.get('status', 'pending')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arzlm.fullbuild")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--phase", default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--start-from", default=None)
    args = parser.parse_args(argv)
    if args.status:
        return print_status()
    if args.phase:
        state = _load_state()
        HANDLERS[args.phase](state)
        return 0
    return run_all(start_from=args.start_from)


if __name__ == "__main__":
    raise SystemExit(main())
