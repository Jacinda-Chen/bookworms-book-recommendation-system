# scripts/preview_parquet.py
import argparse
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to a .parquet file")
    ap.add_argument("--n", type=int, default=5, help="Rows to preview")
    ap.add_argument("--cols", default="", help="Comma-separated list of columns to show (optional)")
    ap.add_argument("--skip", type=int, default=0, help="Skip first N rows, then show next n")
    ap.add_argument("--sample", action="store_true", help="Random sample n rows instead of head()")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (only used with --sample)")
    args = ap.parse_args()

    df = pd.read_parquet(args.path)

    if args.cols.strip():
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]
        df = df[cols]

    if args.sample:
        view = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    else:
        start = max(args.skip, 0)
        view = df.iloc[start : start + args.n]

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 120)

    print(view.to_string(index=False))


if __name__ == "__main__":
    main()