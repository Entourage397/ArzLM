"""ArzLM command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arzlm.model.config import (
    assert_parameter_budget,
    build_model_config,
    count_parameters,
    format_parameter_report,
)
from arzlm.paths import PREPARED_DIR, RUNS_DIR, TOKENIZER_DIR
from arzlm.training.environment import collect_environment


def _cmd_params(args: argparse.Namespace) -> int:
    from arzlm.model.config import load_model_config

    if getattr(args, "model_config", None):
        config = load_model_config(args.model_config)
    else:
        config = build_model_config()
    counts = count_parameters(config, tie_embeddings=True)
    print(format_parameter_report(counts))
    print(json.dumps({**counts, "config_name": config.name, "padded_vocab_size": config.padded_vocab_size}, indent=2))
    assert_parameter_budget(counts)
    return 0


def _cmd_env(_args: argparse.Namespace) -> int:
    print(json.dumps(collect_environment(), indent=2))
    return 0


def _cmd_prepare_stem(args: argparse.Namespace) -> int:
    from arzlm.data.stem_builder import STEM_PRESETS, build_stem_corpus

    if args.list:
        print(json.dumps({k: v for k, v in STEM_PRESETS.items()}, indent=2))
        return 0
    build_stem_corpus(
        args.preset,
        source=args.source,
        out_dir=args.out_dir,
        tokenizer_dir=args.tokenizer_dir,
        seed=args.seed,
    )
    return 0


def _cmd_data_security_audit(args: argparse.Namespace) -> int:
    from arzlm.data.security.audit import (
        format_audit_report,
        refresh_source_lock,
        report_to_dict,
        run_source_security_audit,
    )

    if args.write_lock:
        path = refresh_source_lock()
        print(f"wrote source-lock {path}")
    report = run_source_security_audit()
    print(format_audit_report(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8")
    return 0 if report.passed else 2


def _cmd_tokenizer_compare(args: argparse.Namespace) -> int:
    from arzlm.tokenizer.compare import evaluate_tokenizer, train_stem_candidate, weighted_chars_per_token, write_comparison
    from arzlm.data.catalog import MIXTURE_TARGET
    from arzlm.data.sources import synthetic_stem_documents
    from arzlm.data.catalog import DOMAINS

    corpora: dict[str, str] = {}
    for domain in DOMAINS:
        chunks = []
        n = 0
        for doc in synthetic_stem_documents(domain, n_docs=400, seed=7):
            chunks.append(doc.text)
            n += len(doc.text)
            if n >= args.chars_per_domain:
                break
        corpora[domain] = "\n\n".join(chunks)

    if args.train_candidate:
        mix = []
        for domain, w in MIXTURE_TARGET.items():
            take = int(args.train_chars * w)
            acc = 0
            for doc in synthetic_stem_documents(domain, n_docs=20_000, seed=11):
                mix.append(doc.text)
                acc += len(doc.text)
                if acc >= take:
                    break
        train_stem_candidate(mix, args.candidate_dir)

    current = evaluate_tokenizer(args.current_dir, corpora)
    payload: dict = {"current": current}
    if args.candidate_dir and (args.candidate_dir / "tokenizer.json").is_file():
        cand = evaluate_tokenizer(args.candidate_dir, corpora)
        payload["candidate"] = cand
        payload["weighted_chars_per_token"] = {
            "current": weighted_chars_per_token(current, MIXTURE_TARGET),
            "candidate": weighted_chars_per_token(cand, MIXTURE_TARGET),
        }
    write_comparison(args.out, payload)
    print(json.dumps(payload["weighted_chars_per_token"] if "weighted_chars_per_token" in payload else current["domains"], indent=2))
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    from arzlm.data.prepare import DATA_PRESETS, prepare_corpus

    if args.list:
        print(json.dumps(DATA_PRESETS, indent=2))
        return 0
    prepare_corpus(
        args.preset,
        source=args.source,
        out_dir=args.out_dir,
        tokenizer_dir=args.tokenizer_dir,
        seed=args.seed,
        reuse_tokenizer=not args.retrain_tokenizer,
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from arzlm.training.config import load_experiment_config
    from arzlm.training.loop import run_training

    exp = load_experiment_config(args.config)
    if args.out_dir:
        exp.out_dir = Path(args.out_dir)
    if args.data_dir:
        exp.data_dir = Path(args.data_dir)
    if args.max_steps is not None:
        exp.train.max_steps = args.max_steps
    if args.micro_batch_size is not None:
        exp.train.micro_batch_size = args.micro_batch_size
    if args.compile:
        exp.compile = args.compile
    run_training(exp)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from arzlm.training.benchmark import run_benchmark

    run_benchmark(
        args.data_dir,
        out_path=args.out,
        seq_length=args.seq_length,
        precision=args.precision,
        compile=args.compile,
        compile_scope=args.compile_scope,
        optimizer=args.optimizer,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        microbatches=tuple(int(x) for x in args.microbatches.split(",")),
        seed=args.seed,
    )
    return 0


DEFAULT_GENERATE_PROMPTS = [
    "The capital of France is",
    "A prime number is",
    "The derivative of x^2 is",
    "Photosynthesis converts",
    "def binary_search(arr, target):",
    "In English, a complete sentence must",
]


def _cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    from arzlm.infer import load_inference_checkpoint

    loaded = load_inference_checkpoint(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    print(
        json.dumps(
            {
                "source": loaded.source,
                "checkpoint_dir": str(loaded.checkpoint_dir),
                "parameters": loaded.parameters,
                "context": int(loaded.config.block_size),
                "vocab_size": int(loaded.config.vocab_size),
                "step_count": loaded.step_count,
                "device": str(loaded.device),
                "dtype": str(loaded.dtype).replace("torch.", ""),
                "missing_keys": list(loaded.missing_keys),
                "unexpected_keys": list(loaded.unexpected_keys),
            },
            indent=2,
        )
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from arzlm.infer import generate, load_inference_checkpoint

    loaded = load_inference_checkpoint(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    prompts = list(args.prompt) if args.prompt else list(DEFAULT_GENERATE_PROMPTS)
    rows = [generate(loaded, prompt, max_new_tokens=args.max_new_tokens) for prompt in prompts]
    print(
        json.dumps(
            {
                "parameters": loaded.parameters,
                "context": int(loaded.config.block_size),
                "device": str(loaded.device),
                "dtype": str(loaded.dtype).replace("torch.", ""),
                "max_new_tokens": args.max_new_tokens,
                "generations": rows,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arzlm", description="ArzLM-100M training utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_params = sub.add_parser("params", help="Print exact trainable parameter count")
    p_params.add_argument("--model-config", type=Path, default=None)
    p_params.set_defaults(func=_cmd_params)
    sub.add_parser("env", help="Print detected GPU/PyTorch/CUDA environment").set_defaults(func=_cmd_env)

    p = sub.add_parser("prepare", help="Stream, tokenize, and pack a FineWeb-Edu subset")
    p.add_argument("--preset", choices=["tiny", "overfit", "10m", "50m", "100m"], default="tiny")
    p.add_argument("--source", choices=["fineweb-edu", "synthetic"], default="fineweb-edu")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrain-tokenizer", action="store_true")
    p.add_argument("--list", action="store_true")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("prepare-stem", help="Build the ArzLM-STEM local packed corpus")
    p.add_argument("--preset", choices=["tiny", "sanity-10m", "1b", "6b"], default="tiny")
    p.add_argument("--source", choices=["remote", "synthetic"], default="synthetic")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--list", action="store_true")
    p.set_defaults(func=_cmd_prepare_stem)

    p = sub.add_parser(
        "data-security-audit",
        help="Audit candidate upstream dataset files before corpus preparation (metadata only)",
    )
    p.add_argument("--write-lock", action="store_true", help="Refresh data/security/source-lock.json from Hub metadata")
    p.add_argument("--json-out", type=Path, default=None)
    p.set_defaults(func=_cmd_data_security_audit)

    p = sub.add_parser("tokenizer-compare", help="Evaluate current vs STEM-mixture tokenizer")
    p.add_argument("--current-dir", type=Path, default=TOKENIZER_DIR)
    p.add_argument("--candidate-dir", type=Path, default=Path("tokenizer/stem-v1"))
    p.add_argument("--out", type=Path, default=Path("docs/data/tokenizer-stem-comparison.json"))
    p.add_argument("--chars-per-domain", type=int, default=20_000)
    p.add_argument("--train-chars", type=int, default=400_000)
    p.add_argument("--train-candidate", action="store_true")
    p.set_defaults(func=_cmd_tokenizer_compare)

    p = sub.add_parser("train", help="Run a configured training experiment")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--compile", choices=["auto", "true", "false"], default=None)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("bench", help="RTX 4070 throughput benchmark with micro-batch sweep")
    p.add_argument("--data-dir", type=Path, default=PREPARED_DIR / "tiny")
    p.add_argument("--out", type=Path, default=RUNS_DIR / "benchmark" / "latest.json")
    p.add_argument("--seq-length", type=int, default=1024)
    p.add_argument("--precision", default="bf16-true")
    p.add_argument("--compile", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--compile-scope", choices=["model", "train_step"], default="model")
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=10)
    p.add_argument("--microbatches", default="1,2,4,8")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser(
        "verify-checkpoint",
        help="Strict-load an ArzLM LitGPT inference checkpoint (not AutoModelForCausalLM)",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    p.set_defaults(func=_cmd_verify_checkpoint)

    p = sub.add_parser("generate", help="Greedy completions from an ArzLM inference checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    p.set_defaults(func=_cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
