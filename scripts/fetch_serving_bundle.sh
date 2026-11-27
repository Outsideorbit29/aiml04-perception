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
# F6 drift reference for the live monitor (optional; it falls back to the first live scores).
# The Kaggle job writes it as f6_drift/drift_reference.json.
TMP="$(mktemp -d)"
if "$KAGGLE" kernels output anish2906/aiml01-drift -p "$TMP" \
    --file-pattern 'drift_reference\.json$' && [ -f "$TMP/f6_drift/drift_reference.json" ]; then
  mv "$TMP/f6_drift/drift_reference.json" "$OUT/data/drift_reference.json"
else
  echo "no drift reference found; the monitor will use the first live scores as its baseline"
fi
rm -rf "$TMP"
du -sh "$OUT"/models "$OUT"/data
