#!/usr/bin/env bash
# Download the F5 serving bundle (model, client risk, history + replay data) from the Kaggle
# output of anish2906/aiml01-serving-bundle into ./serving_bundle (gitignored).
# Runs in Git Bash on Windows or in WSL. KAGGLE can point at the kaggle CLI if it is not on PATH.
set -euo pipefail
KAGGLE="${KAGGLE:-kaggle}"
OUT="${1:-serving_bundle}"
TMP="$(mktemp -d)"
"$KAGGLE" kernels output anish2906/aiml01-serving-bundle -p "$TMP" --file-pattern '\.(json|parquet)$'
rm -rf "$OUT"
mv "$TMP/serving_bundle" "$OUT"
rm -rf "$TMP"
# F6 drift reference for the live monitor (optional; it falls back to history.parquet).
"$KAGGLE" kernels output anish2906/aiml01-drift -p "$OUT/data" --file-pattern 'drift_reference\.json$' \
  || echo "no drift reference found; the monitor will build one from history.parquet"
du -sh "$OUT"/models "$OUT"/data
