"""Print aligned validation CE for one or more metrics.jsonl files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_vals(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("val_loss") is None:
            continue
        rows.append(row)
    return rows


def nearest(rows: list[dict], tokens: int) -> dict:
    return min(rows, key=lambda r: abs(int(r.get("tokens") or 0) - tokens))


def main() -> int:
    marks = [2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000]
    paths = sys.argv[1:]
    series = [(Path(p), load_vals(Path(p))) for p in paths]
    print("file n_val final")
    for path, rows in series:
        last = rows[-1] if rows else {}
        print(path.name, len(rows), last.get("val_loss"), last.get("tokens"))
    header = ["tokens"] + [p.name.replace("-metrics.jsonl", "") for p, _ in series]
    print("\t".join(header))
    for mark in marks:
        cols = [str(mark)]
        for _, rows in series:
            row = nearest(rows, mark)
            cols.append(f"{row['val_loss']:.4f}@{row['tokens']}")
        print("\t".join(cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
