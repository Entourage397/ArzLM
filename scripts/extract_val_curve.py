"""Print validation CE vs tokens from a metrics.jsonl file."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("val_loss") is None:
            continue
        print(
            json.dumps(
                {
                    "tokens": row.get("tokens"),
                    "val_loss": row.get("val_loss"),
                    "elapsed_s": row.get("wall_s", row.get("elapsed_s")),
                    "grad_norm": row.get("grad_norm"),
                    "loss": row.get("loss"),
                    "loss_spike_count": row.get("loss_spike_count"),
                    "qk_health": row.get("qk_health"),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
