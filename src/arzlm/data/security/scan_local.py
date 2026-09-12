"""Optional local ClamAV scan of project-owned artifacts. Disabled by default."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalScanResult:
    engine: str
    available: bool
    skipped: bool
    infected: bool
    detail: str


def clamav_executable() -> str | None:
    return shutil.which("clamdscan") or shutil.which("clamscan")


def local_scan_enabled() -> bool:
    raw = os.environ.get("ARZLM_LOCAL_AV", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def maybe_scan_local_file(path: Path, *, enabled: bool | None = None) -> LocalScanResult:
    """Scan one already-downloaded project file. Never executes the artifact.

    Skip if ClamAV is missing or scanning is disabled. A positive detection is
    a hard failure for the caller — it is not treated as a harmless false
    positive.
    """
    if enabled is None:
        enabled = local_scan_enabled()
    if not enabled:
        return LocalScanResult("clamav", available=bool(clamav_executable()), skipped=True, infected=False, detail="disabled")
    exe = clamav_executable()
    if exe is None:
        return LocalScanResult("clamav", available=False, skipped=True, infected=False, detail="clamav not installed")
    path = Path(path)
    proc = subprocess.run(
        [exe, "--no-summary", "--stdout", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # clamscan: 0 clean, 1 infected, 2 error
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 1 or "FOUND" in text:
        return LocalScanResult("clamav", available=True, skipped=False, infected=True, detail=text.strip()[:2000])
    if proc.returncode not in {0, 1}:
        return LocalScanResult(
            "clamav",
            available=True,
            skipped=False,
            infected=False,
            detail=f"scanner error rc={proc.returncode}: {text.strip()[:1000]}",
        )
    return LocalScanResult("clamav", available=True, skipped=False, infected=False, detail="clean")
