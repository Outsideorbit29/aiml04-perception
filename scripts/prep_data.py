"""F1 data prep: load + join IEEE-CIS, time split, EDA summary and figures.

Runs on Kaggle (see kaggle/data_prep). Locally it can run on the synthetic fixtures:
    python -m scripts.prep_data --config configs/data.yaml --data-dir tests/fixtures --out-dir out
Outputs in out_dir: splits/{train,val,test}.parquet, split_summary.json, eda_summary.json,
figures/*.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud.config import load_config
from fraud.data import add_time_columns, find_data_dir, load_raw, split_summary, time_split
from fraud.eda import eda_summary, save_figures


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--data-dir", help="override config data_dir")
    ap.add_argument("--out-dir", help="override config out_dir")
    ap.add_argument("--smoke", action="store_true", help="read only smoke_rows rows")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = find_data_dir(args.data_dir or cfg["data_dir"])
    out_dir = Path(args.out_dir or cfg["out_dir"])
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    target, time_col = cfg["target"], cfg["time_column"]

    nrows = cfg["smoke_rows"] if args.smoke else None
    df = load_raw(data_dir, nrows=nrows, id_column=cfg["id_column"])
    df = add_time_columns(df, cfg["start_date"], time_col)
    print(f"loaded {len(df):,} rows x {df.shape[1]} columns from {data_dir}", flush=True)

    splits = time_split(df, cfg["split"], time_col)
    for name, part in splits.items():
        part.to_parquet(out_dir / "splits" / f"{name}.parquet", index=False)
    s_summary = split_summary(splits, target, time_col)
    (out_dir / "split_summary.json").write_text(json.dumps(s_summary, indent=2))
    print(json.dumps(s_summary, indent=2), flush=True)

    # EDA on the training split only, so later modelling choices don't peek at val/test.
    e_summary = eda_summary(splits["train"], target, cfg["eda"]["top_n_categories"])
    e_summary["smoke"] = bool(args.smoke)
    (out_dir / "eda_summary.json").write_text(json.dumps(e_summary, indent=2))
    if not args.no_figures:
        save_figures(splits["train"], out_dir / "figures", target, cfg["eda"]["amount_log_bins"])
    return {"split_summary": s_summary, "eda_summary": e_summary}


if __name__ == "__main__":
    main()
