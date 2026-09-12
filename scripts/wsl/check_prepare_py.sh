#!/usr/bin/env bash
set -euo pipefail
echo "=== IsolatedEncoder in WSL prepare.py? ==="
rg -n "IsolatedEncoder|n_skipped_unencodable|_sanitize_document" /home/asus/arzlm/src/arzlm/data/prepare.py || true
echo "=== pyc mtimes ==="
ls -l /home/asus/arzlm/src/arzlm/data/prepare.py /home/asus/arzlm/src/arzlm/data/__pycache__/prepare*.pyc 2>/dev/null || true
