# Publication manifest (2026-09-12)

## Publish (GitHub `Entourage397/ArzLM`)

Source, configs, tests, docs, Modal orchestration (no secrets), tokenizer *training* code, `data/security` lockfiles, eval harness.

## Publish (Hugging Face `Kymaris/ArzLM-300M-Base`)

Clean inference artifacts only:

* `README.md` (model card)
* `config.json`, `model_config.yaml`, tokenizer JSON, `generation_config.json`, `special_tokens_map.json`
* `model.safetensors` (~642 MiB)
* `litgpt/lit_model.pth` (~579 MiB, keys `model` / `step_count` / `iter_num` only)
* `EXPORT_META.json`, `FINAL_SMOKE.json`

Tokenizer `model_max_length` = 2048. Vocab / token IDs unchanged from training.

## Do not publish

* `runs/**` rolling and `final/` optimizer-bearing checkpoints
* Modal volume packed corpora, HF caches, `.venv*`
* `.env`, tokens, `~/.modal.toml`
* Raw datasets
* GGUF / quantized weights (not part of this release)

## Secrets scan

Working-tree text files + git history: no Hugging Face / GitHub / OpenAI / AWS / PEM private-key matches. No `.env` files. Inference `lit_model.pth` top-level keys are `model`, `step_count`, `iter_num` only.
