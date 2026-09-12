"""Optional cloud hooks used by the LitGPT/Fabric training loop.

Local runs no-op unless environment variables are set. Modal registers
commit/status callbacks before calling run_training().
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_after_checkpoint: Callable[[Path], None] | None = None
_after_status: Callable[[dict[str, Any]], None] | None = None
_after_pack: Callable[[dict[str, Any]], None] | None = None
_pack_min_interval_s = 600.0
_last_pack_commit = 0.0
_before_stop: Callable[[], None] | None = None
_stop_min_interval_s = 30.0
_last_stop_reload = 0.0


def set_after_checkpoint(fn: Callable[[Path], None] | None) -> None:
    global _after_checkpoint
    _after_checkpoint = fn


def set_after_status(fn: Callable[[dict[str, Any]], None] | None) -> None:
    global _after_status
    _after_status = fn


def set_after_pack_checkpoint(
    fn: Callable[[dict[str, Any]], None] | None,
    *,
    min_interval_s: float = 600.0,
) -> None:
    global _after_pack, _pack_min_interval_s, _last_pack_commit
    _after_pack = fn
    _pack_min_interval_s = float(min_interval_s)
    _last_pack_commit = 0.0


def maybe_commit_pack(payload: dict[str, Any] | None = None, *, force: bool = False) -> None:
    """Throttle Volume commits during packing. No-op unless Modal registered a hook."""
    global _last_pack_commit
    if _after_pack is None:
        return
    now = time.monotonic()
    if not force and (now - _last_pack_commit) < _pack_min_interval_s:
        return
    _after_pack(payload or {})
    _last_pack_commit = now


def set_before_stop_check(
    fn: Callable[[], None] | None,
    *,
    min_interval_s: float = 30.0,
) -> None:
    global _before_stop, _stop_min_interval_s, _last_stop_reload
    _before_stop = fn
    _stop_min_interval_s = float(min_interval_s)
    _last_stop_reload = 0.0


def status_path(out_dir: Path) -> Path:
    override = os.environ.get("ARZLM_STATUS_PATH")
    return Path(override) if override else out_dir / "status.json"


def stop_path(out_dir: Path) -> Path:
    override = os.environ.get("ARZLM_STOP_PATH")
    return Path(override) if override else out_dir / "STOP_REQUESTED"


def desired_state_path() -> Path | None:
    override = os.environ.get("ARZLM_DESIRED_STATE_PATH")
    return Path(override) if override else None


def stop_requested(out_dir: Path) -> bool:
    global _last_stop_reload
    if _before_stop is not None:
        now = time.monotonic()
        if now - _last_stop_reload >= _stop_min_interval_s:
            _before_stop()
            _last_stop_reload = now
    if stop_path(out_dir).is_file():
        return True
    desired = desired_state_path()
    if desired and desired.is_file():
        try:
            payload = json.loads(desired.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        state = str(payload.get("desired_state") or payload.get("state") or "").upper()
        return state in {"STOP_REQUESTED", "STOPPED"}
    return False


def write_status(out_dir: Path, payload: dict[str, Any]) -> Path:
    path = status_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    if _after_status is not None:
        _after_status(payload)
    return path


def on_checkpoint(path: Path) -> None:
    if _after_checkpoint is not None:
        _after_checkpoint(path)


def write_latest_valid(out_dir: Path, checkpoint_dir: Path, tokens: int, step: int) -> None:
    pointer = out_dir / "latest_valid.json"
    payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "tokens": int(tokens),
        "step": int(step),
    }
    tmp = pointer.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(pointer)
