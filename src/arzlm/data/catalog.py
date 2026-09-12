"""Pinned STEM corpus sources. Unknown / forbidden dataset IDs are hard errors.

Revisions were resolved from the Hugging Face Hub dataset APIs on 2026-09-08.
Do not load ordinary FineWeb. Do not silently swap unofficial mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Ordinary FineWeb (not FineWeb-Edu). Loading these is a hard error.
FORBIDDEN_DATASETS = frozenset(
    {
        "HuggingFaceFW/fineweb",
        "HuggingFaceFW/fineweb-2",
    }
)

DOMAINS = ("general", "math", "code", "science")

MIXTURE_TARGET = {
    "general": 0.55,
    "math": 0.20,
    "code": 0.15,
    "science": 0.10,
}

# Unique training tokens for ArzLM-STEM-1B-v1. Validation is extra.
# Exact quotas are divisible by the optimizer token batch (8192).
TOKENS_PER_OPT_STEP = 8_192
TRAIN_TOKEN_QUOTA = {
    "general": 550_002_688,  # 67,139 steps
    "math": 199_999_488,  # 24,414 steps
    "code": 150_003_712,  # 18,311 steps
    "science": 99_999_744,  # 12,207 steps
}
TRAIN_TOKEN_TOTAL = sum(TRAIN_TOKEN_QUOTA.values())  # 1,000,005,632
EPOCHS = 2
TOKEN_EXPOSURES = TRAIN_TOKEN_TOTAL * EPOCHS  # 2,000,011,264
EXPECTED_OPT_STEPS = TOKEN_EXPOSURES // TOKENS_PER_OPT_STEP  # 244,142

VAL_TOKEN_QUOTA = {
    "general": 1_000_000,
    "math": 1_000_000,
    "code": 1_000_000,
    "science": 1_000_000,
}

# ArzLM-300M / 6B unique packed train tokens. 131,072 tokens/optimizer step
# (64 sequences * 2048). Quotas are unique packed material, not repeated epochs.
TOKENS_PER_OPT_STEP_300M = 131_072
TRAIN_TOKEN_QUOTA_6B = {
    "general": 3_465_019_392,  # 26,436 steps
    "math": 1_259_995_136,  # 9,613 steps
    "code": 945_029_120,  # 7,210 steps
    "science": 630_063_104,  # 4,807 steps
}
TRAIN_TOKEN_TOTAL_6B = sum(TRAIN_TOKEN_QUOTA_6B.values())  # 6,300,106,752
VAL_TOKEN_QUOTA_6B = {
    "general": 2_097_152,
    "math": 2_097_152,
    "code": 2_097_152,
    "science": 2_097_152,
}
# Consumed training budget (mixture sampling stops at this exposure count).
TOKEN_EXPOSURES_6B = 6_000_214_016  # 45,778 * 131,072
EXPECTED_OPT_STEPS_6B = TOKEN_EXPOSURES_6B // TOKENS_PER_OPT_STEP_300M  # 45,778
EPOCHS_6B = 1

# Within the 150M code allocation. Markdown is capped so it cannot eat CS.
CODE_LANGUAGE_WEIGHTS = {
    "Python": 0.35,
    "C": 0.10,
    "Cpp": 0.10,
    "Java": 0.10,
    "JavaScript": 0.07,
    "TypeScript": 0.03,
    "Rust": 0.04,
    "Go": 0.04,
    "SQL": 0.05,
    "Shell": 0.02,
    "CSharp": 0.03,
    "PHP": 0.02,
    "Ruby": 0.02,
    "Swift": 0.01,
    "Markdown": 0.02,
}

PERMISSIVE_LICENSE_TYPES = frozenset({"permissive"})
PERMISSIVE_SPDX = frozenset(
    {
        "mit",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd-3-clause-clear",
        "bsd-2-clause-freebsd",
        "cc0-1.0",
        "unlicense",
        "isc",
        "0bsd",
        "mit-0",
        "zlib",
        "boost-1.0",
        "artistic-2.0",
        "python-2.0",
        "postgresql",
        "wtfpl",
        "ncsa",
        "ms-pl",
        "postgresql",
    }
)


@dataclass(frozen=True)
class SourceSpec:
    domain: str
    dataset_id: str
    config: str | None
    revision: str
    license: str
    pretraining_permitted: bool
    schema: tuple[str, ...]
    streaming: bool
    content_hosted: str
    notes: str
    extra: dict[str, Any] = field(default_factory=dict)


FINEWEB_EDU = SourceSpec(
    domain="general",
    dataset_id="HuggingFaceFW/fineweb-edu",
    config="sample-10BT",
    revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    license="ODC-By-1.0 (dataset); Common Crawl Terms of Use for crawled pages",
    pretraining_permitted=True,
    schema=(
        "text",
        "id",
        "dump",
        "url",
        "date",
        "file_path",
        "language",
        "language_score",
        "token_count",
        "score",
        "int_score",
    ),
    streaming=True,
    content_hosted="huggingface parquet (sample/10BT)",
    notes="Never substitute HuggingFaceFW/fineweb. Stream only; do not pull the 1.3T dump.",
)

FINEMATH_4PLUS = SourceSpec(
    domain="math",
    dataset_id="HuggingFaceTB/finemath",
    config="finemath-4plus",
    revision="e92b25a616738fe95dc186b64dfb19f9c8525594",
    license="ODC-By-1.0 (dataset); Common Crawl Terms of Use for crawled pages",
    pretraining_permitted=True,
    schema=(
        "url",
        "fetch_time",
        "content_mime_type",
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
        "text",
        "token_count",
        "char_count",
        "metadata",
        "score",
        "int_score",
        "crawl",
        "snapshot_type",
        "language",
        "language_score",
    ),
    streaming=True,
    content_hosted="huggingface parquet; text field is inline",
    notes="Chosen math source for ArzLM-STEM-1B-v1. See MATH_DECISION.",
)

NEMOTRON_CC_MATH = SourceSpec(
    domain="math",
    dataset_id="nvidia/Nemotron-CC-Math-v1",
    config="4plus",
    revision="unknown-gated",
    license=(
        "NVIDIA Data Agreement for Model Training: internal training of Company AI "
        "Solutions only; gated Hub access; Phi-4 cleaning may attach Phi-4 license "
        "to distributed models"
    ),
    pretraining_permitted=False,
    schema=("text",),
    streaming=True,
    content_hosted="huggingface (gated)",
    notes="Quality is competitive (NVIDIA 8B mid-train ablations) but not legally usable here.",
)

STACK_EDU = SourceSpec(
    domain="code",
    dataset_id="HuggingFaceTB/stack-edu",
    config=None,
    revision="eeec5caac5cc3758a18f1d3ba4416837a9ba814c",
    license=(
        "Dataset intended for LM training; file contents are Software Heritage blobs "
        "under original FOSS licenses (The Stack v2 terms). Training must follow "
        "Software Heritage LLM principles. We keep permissive-license files and "
        "record blob_id provenance."
    ),
    pretraining_permitted=True,
    schema=(
        "blob_id",
        "language",
        "repo_name",
        "path",
        "src_encoding",
        "length_bytes",
        "score",
        "int_score",
        "detected_licenses",
        "license_type",
    ),
    streaming=True,
    content_hosted="IDs on Hugging Face; file bytes from Software Heritage S3 content/{blob_id}",
    extra={"languages": tuple(CODE_LANGUAGE_WEIGHTS.keys()), "s3_bucket": "softwareheritage"},
    notes="Do not use an unofficial text mirror. Fetch official SWH objects.",
)

PES2O = SourceSpec(
    domain="science",
    dataset_id="allenai/peS2o",
    config="v2",
    revision="636a503e44a3ca1b58e01fb61eab0825cd574de0",
    license="ODC-By (corpus). Paper copyrights remain with authors.",
    pretraining_permitted=True,
    schema=("added", "created", "id", "source", "text", "version"),
    streaming=True,
    content_hosted="huggingface json.gz under data/v2/; loader default is v2",
    extra={
        "prefer_source": "s2orc",
        "source_values": "s2orc/train (full text), s2ag/train (title/abstract)",
        "skip_source": "s2ag",
        "s2ag_train_shards": list(range(0, 10)),
        "s2orc_train_shards": list(range(10, 20)),
        "v3_present_but_not_in_loader": True,
        "official_latest": "v2",
    },
    notes=(
        "README and peS2o.py DEFAULT_CONFIG_NAME=v2. data/v3/*.zst exists but is not "
        "wired into the datasets script. Hub v2 files train-00000..00009 are ~1.5GB "
        "s2ag/train abstracts; train-00010..00019 are ~7GB s2orc/train full text. "
        "Never load_dataset (pulls ~87GB). Schema has no field-of-study; diversity is "
        "multi-shard sampling of s2orc files, not exact FoS quotas."
    ),
)

# Empirically from Hub file sizes on 2026-09-08 (not documented on the card).
PES2O_S2AG_TRAIN_SHARDS = tuple(range(0, 10))
PES2O_S2ORC_TRAIN_SHARDS = tuple(range(10, 20))

SOURCES = {
    "general": FINEWEB_EDU,
    "math": FINEMATH_4PLUS,
    "code": STACK_EDU,
    "science": PES2O,
}

MATH_DECISION = {
    "chosen": "HuggingFaceTB/finemath",
    "config": "finemath-4plus",
    "rejected": "nvidia/Nemotron-CC-Math-v1",
    "reason": (
        "Nemotron-CC-Math-4plus reports stronger 8B mid-training MATH/GSM8K/code "
        "than FineMath (Karimi Mahabadi et al., 2025, arXiv:2508.15096; Hub card "
        "MATH 44.2 vs FineMath-3+ 34.6 at their 3+ setting, and 4plus is the "
        "high-quality subset). It is gated behind the NVIDIA Data Agreement for "
        "Model Training, which permits only internal training and forbids making "
        "the dataset available to others. Cleaning used Phi-4; distributing a "
        "model trained on it may inherit Phi-4 license terms. FineMath-4+ is "
        "public ODC-By-1.0, streams text directly, is decontaminated vs GSM8k/"
        "MATH/MMLU/ARC, and preserves Markdown+LaTeX. For a reproducible local "
        "corpus we can legally train and later release, FineMath-4+ is the "
        "primary math source. We do not mix both."
    ),
    "sources": [
        "https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1",
        "https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/raw/main/LICENSE.md",
        "https://arxiv.org/abs/2508.15096",
        "https://huggingface.co/datasets/HuggingFaceTB/finemath",
        "https://arxiv.org/abs/2502.02737",
    ],
}

STEM_FORMAT = "arzlm-stem-v1"
CORPUS_NAME = "ArzLM-STEM-1B-v1"
CORPUS_NAME_6B = "ArzLM-STEM-6B-v1"


def assert_dataset_allowed(dataset_id: str) -> None:
    name = (dataset_id or "").strip()
    if not name:
        raise ValueError("dataset_id is empty")
    if name in FORBIDDEN_DATASETS:
        raise ValueError(
            f"Refusing {name!r}. ArzLM general text must be HuggingFaceFW/fineweb-edu "
            f"config sample-10BT, never ordinary FineWeb."
        )
    lower = name.lower()
    if lower.startswith("huggingfacefw/fineweb") and "fineweb-edu" not in lower:
        raise ValueError(f"Refusing FineWeb variant {name!r}; use HuggingFaceFW/fineweb-edu")


def source_by_id(dataset_id: str) -> SourceSpec:
    assert_dataset_allowed(dataset_id)
    for spec in (FINEWEB_EDU, FINEMATH_4PLUS, STACK_EDU, PES2O):
        if spec.dataset_id == dataset_id:
            return spec
    raise ValueError(f"Unknown STEM source {dataset_id!r}")


def code_language_weights_from_paths(paths: list[str] | tuple[str, ...]) -> dict[str, float]:
    present = {p.replace("\\", "/").split("/")[0] for p in paths}
    raw = {lang: w for lang, w in CODE_LANGUAGE_WEIGHTS.items() if lang in present}
    total = float(sum(raw.values()))
    if total <= 0:
        raise ValueError("no Hub-safe Stack-Edu languages available in the source lock")
    return {lang: w / total for lang, w in raw.items()}


def code_token_quotas(total_code_tokens: int, weights: dict[str, float] | None = None) -> dict[str, int]:
    src = dict(weights or CODE_LANGUAGE_WEIGHTS)
    if not src:
        raise ValueError("code language weights are empty")
    raw = {lang: int(round(w * total_code_tokens)) for lang, w in src.items()}
    drift = total_code_tokens - sum(raw.values())
    pivot = "Python" if "Python" in raw else next(iter(raw))
    raw[pivot] += drift
    return raw
