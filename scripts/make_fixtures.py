"""Write the synthetic IEEE-CIS-shaped fixtures used by the tests (tests/fixtures/)."""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud.synthetic import make_synthetic


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="tests/fixtures")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tx, ident = make_synthetic(args.rows, fraud_rate=0.05, seed=args.seed)
    tx.to_csv(out / "train_transaction.csv", index=False)
    ident.to_csv(out / "train_identity.csv", index=False)
    print(f"wrote {len(tx)} transactions, {len(ident)} identity rows to {out}")


if __name__ == "__main__":
    main()
