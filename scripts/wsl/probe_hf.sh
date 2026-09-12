#!/usr/bin/env bash
set -euo pipefail
cd /home/asus/arzlm
export PYTHONUNBUFFERED=1
export HF_HOME=/home/asus/arzlm/hf-cache
echo "=== pyarrow/datasets ==="
.venv/bin/python -c "import pyarrow, datasets, huggingface_hub; print('pyarrow', pyarrow.__version__); print('datasets', datasets.__version__); print('hub', huggingface_hub.__version__)"
echo "=== tiny stream ==="
.venv/bin/python -c "from datasets import load_dataset; ds=load_dataset('hf-internal-testing/tiny-random-gpt2', split='train', streaming=True); print(next(iter(ds))); print('TINY_OK')"
echo "=== faulthandler fineweb ==="
.venv/bin/python -c "import faulthandler; faulthandler.enable(); from datasets import load_dataset; ds=load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True); it=iter(ds); print('got iterator'); print(next(it)['text'][:60]); print('FW_OK')"
echo ALL_OK
