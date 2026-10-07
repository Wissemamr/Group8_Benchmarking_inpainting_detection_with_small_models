#!/usr/bin/env bash
# Whole AEROBLADE benchmark: index -> resamples -> AEROBLADE over every image -> dataframes/tables.
# Resumable: re-running continues where the compute step stopped. Extra arguments go to
# run_aeroblade.py (e.g. --max-sources 20 for a quick partial run, --shard 0/2 for half the images).
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/build_index.py
[ -f resamples/resample_1.csv ] || python scripts/make_resamples.py
python scripts/run_aeroblade.py "$@"
python scripts/evaluate.py
