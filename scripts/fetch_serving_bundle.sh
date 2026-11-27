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
du -sh "$OUT"/models "$OUT"/data
