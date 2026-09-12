"""Build the 50M AdamW vs Muon comparison table and SVG plots (stdlib only)."""

from __future__ import annotations

import json
import math
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


def load_train(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        loss = row.get("train_loss", row.get("loss"))
        if loss is None or row.get("tokens") is None:
            continue
        row = {**row, "loss": loss}
        rows.append(row)
    return rows


def nearest(rows: list[dict], tokens: int) -> dict:
    return min(rows, key=lambda r: abs(int(r.get("tokens") or 0) - tokens))


def ppl(ce: float | None) -> float | None:
    if ce is None or not math.isfinite(ce):
        return None
    return math.exp(ce)


def _svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    if len(points) < 2:
        return ""
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{d}"/>'


def _chart(
    series: list[tuple[str, str, list[tuple[float, float]]]],
    *,
    width: int = 900,
    height: int = 420,
    xlabel: str,
    ylabel: str,
    title: str,
) -> str:
    left, right, top, bottom = 70, 24, 36, 48
    xs = [x for _, _, pts in series for x, _ in pts]
    ys = [y for _, _, pts in series for _, y in pts]
    if not xs or not ys:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    pad = (ymax - ymin) * 0.08 if ymax != ymin else 0.1
    ymin -= pad
    ymax += pad
    pw = width - left - right
    ph = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * pw

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{width/2:.0f}" y="22" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
        f'<text x="{width/2:.0f}" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="12">{xlabel}</text>',
        f'<text x="16" y="{height/2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="12" '
        f'transform="rotate(-90 16 {height/2:.0f})">{ylabel}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>',
    ]
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        y = sy(yv)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(
            f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{yv:.3f}</text>'
        )
    for name, color, pts in series:
        mapped = [(sx(x), sy(y)) for x, y in pts]
        parts.append(_svg_polyline(mapped, color))
        if mapped:
            x, y = mapped[-1]
            parts.append(
                f'<text x="{x+6:.1f}" y="{y+4:.1f}" font-family="sans-serif" font-size="11" fill="{color}">{name}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def write_report(dest: Path, adamw_metrics: Path, muon_metrics: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    adamw = load_vals(adamw_metrics)
    muon = load_vals(muon_metrics)
    train_a = load_train(adamw_metrics)
    train_m = load_train(muon_metrics)
    tokens_marks = sorted(
        {int(r["tokens"]) for r in adamw} | {int(r["tokens"]) for r in muon}
    )
    table = []
    for mark in tokens_marks:
        a = nearest(adamw, mark)
        m = nearest(muon, mark)
        a_ce = float(a["val_loss"])
        m_ce = float(m["val_loss"])
        table.append(
            {
                "tokens": mark,
                "adamw_val_ce": a_ce,
                "muon_val_ce": m_ce,
                "delta_muon_minus_adamw": m_ce - a_ce,
                "adamw_tokens": int(a["tokens"]),
                "muon_tokens": int(m["tokens"]),
            }
        )

    def series_xy(rows: list[dict], xkey: str, ykey: str) -> list[tuple[float, float]]:
        out = []
        for row in rows:
            if row.get(xkey) is None or row.get(ykey) is None:
                continue
            out.append((float(row[xkey]), float(row[ykey])))
        return out

    plots = {
        "val_ce_vs_tokens.svg": _chart(
            [
                ("AdamW", "#1f77b4", series_xy(adamw, "tokens", "val_loss")),
                ("Muon", "#d62728", series_xy(muon, "tokens", "val_loss")),
            ],
            xlabel="tokens",
            ylabel="val CE",
            title="50M matched: val CE vs tokens",
        ),
        "ppl_vs_tokens.svg": _chart(
            [
                ("AdamW", "#1f77b4", [(float(r["tokens"]), ppl(float(r["val_loss"])) or 0.0) for r in adamw]),
                ("Muon", "#d62728", [(float(r["tokens"]), ppl(float(r["val_loss"])) or 0.0) for r in muon]),
            ],
            xlabel="tokens",
            ylabel="val perplexity",
            title="50M matched: val ppl vs tokens",
        ),
        "val_ce_vs_wall.svg": _chart(
            [
                (
                    "AdamW",
                    "#1f77b4",
                    [(float(r.get("wall_s") or r.get("elapsed_s") or 0), float(r["val_loss"])) for r in adamw],
                ),
                (
                    "Muon",
                    "#d62728",
                    [(float(r.get("wall_s") or r.get("elapsed_s") or 0), float(r["val_loss"])) for r in muon],
                ),
            ],
            xlabel="wall seconds",
            ylabel="val CE",
            title="50M matched: val CE vs wall time",
        ),
        "train_loss_vs_tokens.svg": _chart(
            [
                ("AdamW", "#1f77b4", series_xy(train_a, "tokens", "loss")),
                ("Muon", "#d62728", series_xy(train_m, "tokens", "loss")),
            ],
            xlabel="tokens",
            ylabel="train loss",
            title="50M matched: train loss vs tokens",
        ),
    }
    for name, svg in plots.items():
        (dest / name).write_text(svg, encoding="utf-8")

    final_a = adamw[-1]["val_loss"] if adamw else None
    final_m = muon[-1]["val_loss"] if muon else None
    winner = None
    if final_a is not None and final_m is not None:
        if final_m < final_a - 0.02:
            winner = "muon"
        elif final_a < final_m - 0.02:
            winner = "adamw"
        else:
            winner = "tie_or_too_close"
    report = {
        "primary_metric": "val_ce_vs_tokens",
        "table": table,
        "final_adamw_val_ce": final_a,
        "final_muon_val_ce": final_m,
        "delta_muon_minus_adamw": None if final_a is None or final_m is None else final_m - final_a,
        "winner": winner,
        "note": "Recommend Muon for 250M only if it clearly wins val CE at equal tokens. Do not start 250M here.",
    }
    (dest / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    dest = Path("docs/experiments/50m-optimizer-comparison")
    report = write_report(
        dest,
        dest / "adamw-50m-metrics.jsonl",
        dest / "muon-50m-metrics.jsonl",
    )
    print(json.dumps({k: report[k] for k in ("final_adamw_val_ce", "final_muon_val_ce", "delta_muon_minus_adamw", "winner")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
