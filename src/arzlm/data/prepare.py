"""Prepare FineWeb-Edu (or synthetic) packed shards without downloading the full corpus."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm

from arzlm.data.encode_worker import read_frame, unpack_ids, write_frame
from arzlm.data.catalog import FINEWEB_EDU
from arzlm.data.packed import DTYPE, FORMAT_NAME, pack_documents, write_packed_split
from arzlm.data.textutil import as_token_ids, encode_doc, sanitize_document
from arzlm.paths import HF_CACHE_DIR, PREPARED_DIR, TOKENIZER_DIR, ensure_project_dirs
from arzlm.tokenizer.train import EOS, save_litgpt_tokenizer_files, train_bpe_tokenizer

# HuggingFace `tokenizers` can abort the process on rare FineWeb unicode (Rust
# NFC panic). Encode those documents in a subprocess so a panic skips one doc
# instead of killing the pack. multiprocessing.Pool cannot be used here:
# it segfaults HuggingFace `load_dataset` via filelock fork-safety.
_ENCODE_TIMEOUT_S = 120.0

FINEWEB_DATASET = FINEWEB_EDU.dataset_id
FINEWEB_CONFIG = FINEWEB_EDU.config

DATA_PRESETS: dict[str, dict[str, int]] = {
    # tiny is large enough for a 15-step micro-batch sweep at batch 8, seq 1024
    "tiny": {"train_tokens": 262_144, "val_tokens": 16_384, "tokenizer_chars": 400_000},
    "overfit": {"train_tokens": 65_536, "val_tokens": 8_192, "tokenizer_chars": 8_000_000},
    "10m": {"train_tokens": 10_000_000, "val_tokens": 100_000, "tokenizer_chars": 8_000_000},
    "50m": {"train_tokens": 50_000_000, "val_tokens": 250_000, "tokenizer_chars": 8_000_000},
    "100m": {"train_tokens": 100_000_000, "val_tokens": 500_000, "tokenizer_chars": 8_000_000},
}


@dataclass
class PrepareConfig:
    preset: str
    source: str  # "fineweb-edu" | "synthetic"
    out_dir: Path
    tokenizer_dir: Path
    seed: int = 42
    seq_length: int = 1024
    max_documents: int | None = None


def synthetic_documents(n_docs: int, seed: int) -> Iterator[str]:
    """Deterministic educational-style text for tests and offline smoke runs.

    Random letter-words are mixed in so a 32k BPE can actually fill its vocab
    from a small corpus (purely repetitive lessons saturate too early).
    """
    rng = np.random.default_rng(seed)
    topics = [
        "photosynthesis converts light energy into chemical energy in chloroplasts",
        "newton's second law states that force equals mass times acceleration",
        "the water cycle includes evaporation, condensation, and precipitation",
        "prime numbers are integers greater than one with no positive divisors other than one and themselves",
        "dna stores genetic information as a sequence of nucleotide bases",
        "supply and demand determine prices in a competitive market",
        "the roman republic preceded the roman empire and used elected magistrates",
        "fractions represent parts of a whole and can be added with a common denominator",
    ]
    for i in range(n_docs):
        topic = topics[i % len(topics)]
        extra = int(rng.integers(2, 8))
        sentences = [f"Lesson {i}: {topic.capitalize()}."]
        for j in range(extra):
            sentences.append(
                f"Detail {j + 1} expands on this idea with a worked example and a short definition for students."
            )
        n_noise = int(rng.integers(12, 40))
        noise = []
        for _ in range(n_noise):
            k = int(rng.integers(3, 11))
            noise.append("".join(chr(int(rng.integers(97, 123))) for _ in range(k)))
        sentences.append("Glossary: " + " ".join(noise) + ".")
        yield " ".join(sentences)


def _iter_fineweb_text(max_docs: int | None) -> Iterator[str]:
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from arzlm.data.sources import iter_fineweb_edu

    for i, doc in enumerate(iter_fineweb_edu()):
        if max_docs is not None and i >= max_docs:
            break
        text = doc.text or ""
        if text.strip():
            yield text


def _collect_tokenizer_corpus(texts: Iterator[str], max_chars: int) -> list[str]:
    corpus: list[str] = []
    total = 0
    for text in texts:
        corpus.append(text)
        total += len(text)
        if total >= max_chars:
            break
    if not corpus:
        raise RuntimeError("Tokenizer corpus is empty")
    return corpus


def _sanitize_document(text: str) -> str:
    return sanitize_document(text)


def _encode_doc(tokenizer: Tokenizer, text: str) -> list[int]:
    return encode_doc(tokenizer, text)


class IsolatedEncoder:
    """Encode text in a subprocess so a Rust tokenizer abort is skippable."""

    def __init__(self, tokenizer_dir: Path, timeout_s: float = _ENCODE_TIMEOUT_S) -> None:
        self.tokenizer_dir = str(Path(tokenizer_dir).resolve())
        self.timeout_s = timeout_s
        self.n_worker_restarts = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._start()

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "arzlm.data.encode_worker", self.tokenizer_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def encode(self, text: str) -> list[int] | None:
        """Return token ids, `[]` for empty text, or `None` if the worker died."""
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        stdin = self._proc.stdin
        stdout = self._proc.stdout
        try:
            write_frame(stdin, text.encode("utf-8"))
            box: list[bytes | None] = []
            error: list[BaseException] = []

            def _read() -> None:
                try:
                    box.append(read_frame(stdout))
                except BaseException as exc:  # noqa: BLE001 — worker pipe can fail any way
                    error.append(exc)

            reader = threading.Thread(target=_read, daemon=True)
            reader.start()
            reader.join(self.timeout_s)
            if reader.is_alive():
                raise TimeoutError("encode worker timed out")
            if error:
                raise error[0]
            raw = box[0]
            if raw is None:
                raise RuntimeError("encode worker closed")
            return as_token_ids(unpack_ids(raw))
        except Exception:
            self.n_worker_restarts += 1
            self.close()
            self._start()
            return None

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def prepare_corpus(
    preset: str,
    *,
    source: str = "fineweb-edu",
    out_dir: str | Path | None = None,
    tokenizer_dir: str | Path | None = None,
    seed: int = 42,
    seq_length: int = 1024,
    reuse_tokenizer: bool = True,
) -> Path:
    if preset not in DATA_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {sorted(DATA_PRESETS)}")
    if source not in {"fineweb-edu", "synthetic"}:
        raise ValueError("source must be 'fineweb-edu' or 'synthetic'")
    if source == "fineweb-edu":
        from arzlm.data.security.audit import require_remote_security_audit

        require_remote_security_audit()

    ensure_project_dirs()
    spec = DATA_PRESETS[preset]
    out_dir = Path(out_dir) if out_dir else PREPARED_DIR / preset
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else TOKENIZER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    target_total = spec["train_tokens"] + spec["val_tokens"]
    tokenizer_json = tokenizer_dir / "tokenizer.json"

    def document_stream() -> Iterator[str]:
        if source == "synthetic":
            # Over-generate; packing stops at the token budget.
            return synthetic_documents(n_docs=200_000, seed=seed)
        return _iter_fineweb_text(max_docs=None)

    if tokenizer_json.is_file() and reuse_tokenizer:
        print(f"Reusing tokenizer at {tokenizer_dir}")
    else:
        print(f"Training 32k tokenizer from {source} ({spec['tokenizer_chars']:,} chars)...")
        corpus = _collect_tokenizer_corpus(document_stream(), spec["tokenizer_chars"])
        trained = train_bpe_tokenizer(corpus)
        save_litgpt_tokenizer_files(trained, tokenizer_dir)
        print(f"Wrote tokenizer to {tokenizer_dir}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Start the encode subprocess before load_dataset so we never fork after
    # HuggingFace has spawned its metadata threads.
    encoder: IsolatedEncoder | None = IsolatedEncoder(tokenizer_dir) if source == "fineweb-edu" else None
    tokenizer = Tokenizer(tokenizer_dir)
    eos_id = tokenizer.eos_id
    if eos_id is None:
        raise RuntimeError("Tokenizer has no EOS id")
    litgpt_vocab = tokenizer.vocab_size
    print(f"litgpt.Tokenizer vocab_size={litgpt_vocab} eos_id={eos_id} bos_id={tokenizer.bos_id}")

    packed_chunks: list[np.ndarray] = []
    n_tokens = 0
    n_docs = 0
    n_eos = 0
    n_skipped_unencodable = 0
    n_empty_docs = 0
    progress = tqdm(total=target_total, desc=f"pack:{preset}", unit="tok")
    try:
        for text in document_stream():
            if encoder is not None:
                ids = encoder.encode(text)
                if ids is None:
                    n_skipped_unencodable += 1
                    continue
            else:
                ids = _encode_doc(tokenizer, text)
            if not ids:
                n_empty_docs += 1
                continue
            packed = pack_documents([ids], eos_id=eos_id)
            packed_chunks.append(packed)
            n_tokens += int(packed.size)
            n_docs += 1
            n_eos += 1
            progress.update(int(packed.size))
            if n_tokens >= target_total:
                break
    finally:
        progress.close()
        if encoder is not None:
            encoder.close()

    if n_tokens < target_total:
        raise RuntimeError(f"Only packed {n_tokens:,} tokens; needed {target_total:,}")

    tokens = np.concatenate(packed_chunks)
    tokens = tokens[:target_total]
    val_n = spec["val_tokens"]
    train_tokens = tokens[:-val_n]
    val_tokens = tokens[-val_n:]

    # Drop remainders that cannot form a full (seq+1) window so epoch length is exact.
    block = seq_length + 1
    train_tokens = train_tokens[: len(train_tokens) - (len(train_tokens) % block)]
    val_tokens = val_tokens[: len(val_tokens) - (len(val_tokens) % block)]
    if len(train_tokens) < block or len(val_tokens) < block:
        raise RuntimeError("Packed split too small for seq_length")

    write_packed_split(out_dir / "train.bin", train_tokens)
    write_packed_split(out_dir / "val.bin", val_tokens)

    meta = {
        "format": FORMAT_NAME,
        "dtype": "uint16",
        "endian": "little",
        "preset": preset,
        "source": source,
        "dataset": FINEWEB_DATASET if source == "fineweb-edu" else "synthetic",
        "dataset_config": FINEWEB_CONFIG if source == "fineweb-edu" else None,
        "seed": seed,
        "seq_length": seq_length,
        "eos_id": int(eos_id),
        "bos_id": tokenizer.bos_id,
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "litgpt_vocab_size": int(litgpt_vocab),
        "n_documents": n_docs,
        "n_eos": n_eos,
        "n_skipped_unencodable": n_skipped_unencodable,
        "n_empty_docs": n_empty_docs,
        "encoder_worker_restarts": 0 if encoder is None else encoder.n_worker_restarts,
        "target_train_tokens": spec["train_tokens"],
        "target_val_tokens": spec["val_tokens"],
        "splits": {
            "train": {"files": ["train.bin"], "n_tokens": int(train_tokens.size)},
            "val": {"files": ["val.bin"], "n_tokens": int(val_tokens.size)},
        },
        "numpy_dtype": "uint16",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("preset", "source", "splits", "n_documents")}, indent=2))
    return out_dir
