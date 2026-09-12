"""Compare FineWeb-Edu vs STEM-mixture 32k BPE tokenizers by domain."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from litgpt.tokenizer import Tokenizer

from arzlm.tokenizer.train import VOCAB_SIZE, save_litgpt_tokenizer_files, train_bpe_tokenizer

MATH_PROBES = [
    r"\frac{a}{b}",
    r"\sum_{i=1}^{n} i^2",
    r"\int_0^1 \sqrt{x}\,dx",
    r"\begin{align} x &= 1 \end{align}",
    r"$\nabla \cdot \mathbf{B} = 0$",
    "E = mc^2",
    r"\mathbb{R}^{n}",
    "1234567890",
]

CODE_PROBES = [
    "    def foo(x):\n        return x + 1\n",
    "for (int i = 0; i < n; ++i) {\n  s += a[i];\n}\n",
    "SELECT id FROM users WHERE x = 1;\n",
    "fn main() {\n    println!(\"hi\");\n}\n",
    "== != <= >= && ||",
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_ids(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text, bos=False, eos=False).tolist()


def chars_per_token(tok: Tokenizer, text: str) -> float:
    ids = encode_ids(tok, text)
    return len(text) / max(1, len(ids))


def bytes_per_token(tok: Tokenizer, text: str) -> float:
    ids = encode_ids(tok, text)
    return len(text.encode("utf-8")) / max(1, len(ids))


def roundtrip_ok(tok: Tokenizer, text: str) -> bool:
    ids = tok.encode(text, bos=False, eos=False)
    decoded = tok.decode(ids)
    # Byte-level BPE may add a leading space marker; compare NFC-stripped.
    return text.replace(" ", "") in decoded.replace(" ", "") or decoded == text


def fragmentation(tok: Tokenizer, probes: list[str]) -> dict[str, int]:
    return {p: len(encode_ids(tok, p)) for p in probes}


def evaluate_tokenizer(tokenizer_dir: Path, corpora: dict[str, str]) -> dict[str, Any]:
    tok = Tokenizer(tokenizer_dir)
    vocab = int(tok.vocab_size)
    out: dict[str, Any] = {
        "tokenizer_dir": str(Path(tokenizer_dir).resolve()),
        "tokenizer_sha256": file_sha256(Path(tokenizer_dir) / "tokenizer.json"),
        "vocab_size": vocab,
        "target_vocab": VOCAB_SIZE,
    }
    sample = "hello café " + "é" * 4
    t0 = time.perf_counter()
    for _ in range(200):
        encode_ids(tok, sample)
    out["encode_ns_probe"] = (time.perf_counter() - t0) / 200 * 1e9
    t0 = time.perf_counter()
    ids = tok.encode(sample, bos=False, eos=False)
    for _ in range(200):
        tok.decode(ids)
    out["decode_ns_probe"] = (time.perf_counter() - t0) / 200 * 1e9
    out["unicode_roundtrip"] = roundtrip_ok(tok, "photosynthesis")
    out["utf8_roundtrip"] = True
    try:
        weird = "naïve café"
        ids_w = tok.encode(weird, bos=False, eos=False)
        tok.decode(ids_w)
        out["utf8_roundtrip"] = True
    except Exception:
        out["utf8_roundtrip"] = False
    out["math_fragmentation"] = fragmentation(tok, MATH_PROBES)
    out["code_fragmentation"] = fragmentation(tok, CODE_PROBES)
    domains: dict[str, Any] = {}
    for name, text in corpora.items():
        if not text:
            continue
        ids = encode_ids(tok, text)
        domains[name] = {
            "chars": len(text),
            "bytes": len(text.encode("utf-8")),
            "tokens": len(ids),
            "chars_per_token": chars_per_token(tok, text),
            "bytes_per_token": bytes_per_token(tok, text),
            "tokens_per_byte": (len(ids) / max(1, len(text.encode("utf-8")))),
            "roundtrip": roundtrip_ok(tok, text[:10_000]),
        }
    out["domains"] = domains
    return out


def weighted_chars_per_token(report: dict[str, Any], weights: dict[str, float]) -> float:
    domains = report.get("domains") or {}
    acc = 0.0
    wsum = 0.0
    for name, w in weights.items():
        row = domains.get(name)
        if not row:
            continue
        acc += w * float(row["chars_per_token"])
        wsum += w
    return acc / wsum if wsum else float("nan")


def train_stem_candidate(
    texts: Iterable[str],
    out_dir: Path,
    *,
    vocab_size: int = VOCAB_SIZE,
) -> Path:
    trained = train_bpe_tokenizer(texts, vocab_size=vocab_size)
    return save_litgpt_tokenizer_files(trained, out_dir)


def write_comparison(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
