"""F2.3: check Feast returns the same features as the builder, offline and online.

Runs after scripts.build_features on Kaggle (needs `pip install feast`):
    python -m scripts.feast_check --features-dir /kaggle/working/f2_features
1. `feast apply` on a temp copy of feature_repo/ pointed at client_features.parquet.
2. Offline: point-in-time `get_historical_features` for a sample of transactions; compare.
3. Online: push a few rows to the push source, read back with `get_online_features`; compare.
Writes feast_check.json next to the features.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--repo", default="feature_repo")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--push-rows", type=int, default=20)
    args = ap.parse_args(argv)

    from feast import FeatureStore  # imported late: feast is installed only for this step

    feats_path = Path(args.features_dir) / "client_features.parquet"
    work = Path(tempfile.mkdtemp()) / "feature_repo"
    shutil.copytree(args.repo, work)
    (work / "data").mkdir(exist_ok=True)
    os.environ["FRAUD_CLIENT_FEATURES"] = str(feats_path.resolve())
    subprocess.run(["feast", "apply"], cwd=work, check=True)
    sys.path.insert(0, str(work))
    from definitions import FEATURES  # noqa: E402

    store = FeatureStore(repo_path=str(work))
    full = pd.read_parquet(feats_path)
    result: dict = {"feast_rows": int(len(full))}

    # Offline point-in-time join. Same-second transactions of one client share a timestamp, so
    # Feast may return a sibling's row there; those rows are reported separately.
    sample = full.sample(min(args.sample, len(full)), random_state=0)
    entity_df = sample[["client_key", "event_timestamp"]].copy()
    entity_df["row"] = np.arange(len(entity_df))
    hist = store.get_historical_features(
        entity_df=entity_df, features=[f"client_features:{n}" for n in FEATURES]
    ).to_df()
    hist = hist.sort_values("row").reset_index(drop=True)
    expected = sample.reset_index(drop=True)
    tie = full.duplicated(["client_key", "event_timestamp"], keep=False)
    tie_rows = set(full.index[tie])
    is_tie = np.array([i in tie_rows for i in sample.index])
    match = np.ones(len(expected), dtype=bool)
    for n in FEATURES:
        a = expected[n].to_numpy(dtype="float64")
        b = hist[n].to_numpy(dtype="float64")
        same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-5, atol=1e-6)
        match &= same
    result["offline_sample"] = int(len(expected))
    result["offline_match_rate"] = float(match.mean())
    result["offline_match_rate_no_ties"] = float(match[~is_tie].mean()) if (~is_tie).any() else None
    result["offline_tie_share"] = float(is_tie.mean())

    # Online: push the latest row of a few cards, read them back.
    latest = full.sort_values("event_timestamp").groupby("client_key").tail(1)
    push = latest.head(args.push_rows).drop(columns=["TransactionID"], errors="ignore")
    store.push("client_features_push", push)
    online = store.get_online_features(
        features=[f"client_features:{n}" for n in FEATURES],
        entity_rows=[{"client_key": k} for k in push["client_key"]],
    ).to_df()
    ok = True
    for n in FEATURES:
        a = push[n].to_numpy(dtype="float64")
        b = online[n].to_numpy(dtype="float64")
        ok &= bool(np.all((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-5, atol=1e-6)))
    result["online_rows"] = int(len(push))
    result["online_match"] = ok

    out = Path(args.features_dir) / "feast_check.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
