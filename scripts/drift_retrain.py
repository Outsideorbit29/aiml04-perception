"""F6: launch the F3 retraining job on Kaggle when a drift decision says so (laptop, Windows).

    python -m scripts.drift_retrain --status serving_bundle/data/monitor_status.json
    python -m scripts.drift_retrain --summary f6_drift/drift_summary.json [--window w08_d175-181]
    ... --execute    # actually run `kaggle kernels push -p kaggle/train` (default: dry run)

Reads the latest decision from the live monitor status file, or one window of the offline Kaggle
drift summary (default: the last weekly window). Set KAGGLE to the CLI path if it is not on PATH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fraud.config import load_config
from services.monitor.trigger import maybe_retrain


def load_decision(status: str | None, summary: str | None, window: str | None) -> dict[str, Any]:
    if status:
        return json.loads(Path(status).read_text())["decision"]
    data = json.loads(Path(summary).read_text())
    windows = [w for w in data["windows"] if w["window"] != "all"]
    chosen = next((w for w in windows if w["window"] == window), None) if window else windows[-1]
    if chosen is None:
        raise KeyError(f"window {window} not in {summary}")
    print(f"window {chosen['window']}", flush=True)
    return chosen["decision"]


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/drift.yaml")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--status", help="monitor_status.json from services/monitor")
    src.add_argument("--summary", help="drift_summary.json from the Kaggle drift job")
    ap.add_argument("--window", help="window name in the summary (default: last)")
    ap.add_argument("--execute", action="store_true", help="really launch the Kaggle job")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    decision = load_decision(args.status, args.summary, args.window)
    print(json.dumps({"retrain": decision["retrain"], "reasons": decision["reasons"]}), flush=True)
    result = maybe_retrain(
        decision, cfg["trigger"], cfg["retrain"]["cooldown_hours"], execute=args.execute
    )
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
