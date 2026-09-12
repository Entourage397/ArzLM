#!/usr/bin/env bash
set -euo pipefail
cd /home/asus/arzlm
export PYTHONUNBUFFERED=1
export HF_HOME=/home/asus/arzlm/hf-cache
echo "=== cache tree ==="
du -sh hf-cache 2>/dev/null || true
find hf-cache -maxdepth 4 -type d 2>/dev/null | head -80
echo "=== fineweb files ==="
find hf-cache -iname '*fineweb*' 2>/dev/null | head -40
echo "=== faulthandler fineweb ==="
.venv/bin/python -c "import faulthandler, os; faulthandler.enable(); os.environ.setdefault('HF_HOME','/home/asus/arzlm/hf-cache'); from datasets import load_dataset; print('loading'); ds=load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True); print('loaded', ds); it=iter(ds); print('iter'); row=next(it); print('text', (row.get('text') or '')[:80]); print('FW_OK')"
echo FW_DONE
