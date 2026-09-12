"""Train a 32k byte-level BPE tokenizer compatible with litgpt.Tokenizer."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
UNK, BOS, EOS, PAD = SPECIAL_TOKENS
VOCAB_SIZE = 32000
# tokenizers 0.23.2 ByteLevel panics (normalizer.rs OOB / SIGSEGV) on FineWeb
# paragraphs of a few hundred KB when trained as whole files.
_MAX_PIECE = 50_000


def _training_pieces(texts: Iterable[str]) -> Iterable[str]:
    """Yield bounded documents. Whole-file or mid-grapheme slices panic ByteLevel."""
    for text in texts:
        if not text:
            continue
        parts = text.split("\n\n") if "\n\n" in text else [text]
        for part in parts:
            part = part.strip()
            if len(part) < 4:
                continue
            if len(part) <= _MAX_PIECE:
                yield part
                continue
            start = 0
            while start < len(part):
                end = min(start + _MAX_PIECE, len(part))
                if end < len(part):
                    cut = part.rfind("\n", start + 1, end)
                    if cut <= start:
                        cut = part.rfind(" ", start + 1, end)
                    if cut > start:
                        end = cut
                chunk = part[start:end].strip()
                if len(chunk) >= 4:
                    yield chunk
                start = end if end > start else start + _MAX_PIECE


def _new_bytelevel_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def _fit_bpe(tokenizer: Tokenizer, texts: Iterable[str], *, vocab_size: int, min_frequency: int) -> Tokenizer:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pieces = [p for p in _training_pieces(texts) if p]
    if not pieces:
        raise ValueError("no tokenizer training pieces")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(pieces, trainer=trainer, length=len(pieces))
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    actual = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual > vocab_size:
        raise RuntimeError(f"Tokenizer vocab_size={actual}, expected <= {vocab_size}")
    if actual < vocab_size:
        print(f"Warning: trained vocab_size={actual} < target {vocab_size}; model embeddings stay {vocab_size}")
    return tokenizer


def train_bpe_from_files(
    paths: list[str | Path],
    *,
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = 1,
) -> Tokenizer:
    files = [Path(p) for p in paths if Path(p).is_file() and Path(p).stat().st_size > 0]
    if not files:
        raise ValueError("no tokenizer training files")
    texts = (path.read_text(encoding="utf-8", errors="replace") for path in files)
    return _fit_bpe(_new_bytelevel_tokenizer(), texts, vocab_size=vocab_size, min_frequency=min_frequency)


def train_bpe_tokenizer(
    texts: Iterable[str],
    *,
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = 1,
) -> Tokenizer:
    return _fit_bpe(_new_bytelevel_tokenizer(), texts, vocab_size=vocab_size, min_frequency=min_frequency)


def save_litgpt_tokenizer_files(
    tokenizer: Tokenizer,
    out_dir: str | Path,
    *,
    model_max_length: int = 2048,
) -> Path:
    """Write tokenizer.json + tokenizer_config.json for litgpt.Tokenizer.

    `model_max_length` is metadata only (context the files should advertise).
    It does not change vocabulary or token ids. ArzLM-300M base pretraining
    uses 2048; pass 1024 only when packaging a 100M-context tokenizer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    bos_id = tokenizer.token_to_id(BOS)
    eos_id = tokenizer.token_to_id(EOS)
    unk_id = tokenizer.token_to_id(UNK)
    pad_id = tokenizer.token_to_id(PAD)
    if None in (bos_id, eos_id, unk_id, pad_id):
        raise RuntimeError("Special tokens missing from trained tokenizer")

    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": BOS,
        "eos_token": EOS,
        "unk_token": UNK,
        "pad_token": PAD,
        "add_bos_token": False,
        "add_eos_token": False,
        "model_max_length": int(model_max_length),
        "clean_up_tokenization_spaces": False,
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    generation = {"bos_token_id": bos_id, "eos_token_id": eos_id, "pad_token_id": pad_id, "unk_token_id": unk_id}
    (out_dir / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n", encoding="utf-8")
    return out_dir
