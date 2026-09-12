# Local evaluation

Pinned harness: `lm-eval==0.4.13` (see `pyproject.toml` extra `eval`).

ArzLM is **not** Transformers-native. `eval/arzlm_lm.py` registers `--model arzlm`.

Do not apply chat templates. Do not change few-shot counts per model (`eval/settings.yaml`).

`Cerebras-GPT-256M` is the intended fair-compute baseline. If Hugging Face returns 404, leave that column marked unavailable. Do not replace it with a 300B-token Pythia/OPT model and still call the comparison fair.
