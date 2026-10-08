#!/bin/zsh
# Phase C: full TruFor inference on all unique images in the manifest (each exactly once).
# Starts or resumes: images already saved (status=ok row + .npz) are skipped, failed ones are retried.
# Survives closing the terminal (nohup) and keeps the Mac awake while it runs (caffeinate -i).
cd "$(dirname "$0")/../.." || exit 1
if pgrep -f "run_trufor.py . outputs/manifest/manifest_unique.csv" > /dev/null; then
  echo "already running (pid $(pgrep -f 'run_trufor.py . outputs/manifest/manifest_unique.csv'))"; exit 1
fi
mkdir -p logs outputs/inference
nohup caffeinate -i ./.venv-trufor/bin/python code/phaseB/run_trufor.py . outputs/manifest/manifest_unique.csv \
  outputs/inference --log logs/inference.log >> logs/inference_console.log 2>&1 < /dev/null &
echo "launched pid $!"
