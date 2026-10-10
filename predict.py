
import argparse
import sys

import pandas as pd

from spam_detector.core import MODEL_PATH, explain, load_bundle, predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="one or more messages to classify")
    ap.add_argument("--file", help="CSV/TXT file; CSV needs a message column, TXT is one message per line")
    ap.add_argument("--column", default="message")
    ap.add_argument("--out", help="write batch results to this CSV")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run: python train_model.py")
    bundle = load_bundle()

    if args.file:
        if args.file.lower().endswith(".csv"):
            texts = pd.read_csv(args.file)[args.column].dropna().astype(str).tolist()
        else:
            texts = [l.strip() for l in open(args.file, encoding="utf-8") if l.strip()]
    else:
        texts = args.text
    if not texts:
        sys.exit("Nothing to classify. Pass text or --file.")

    res = predict(bundle, texts)
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"Wrote {len(res)} results to {args.out} ({(res.prediction == 'spam').sum()} spam)")
        return
    for _, r in res.iterrows():
        print(f"[{r.prediction.upper():4s}] {r.spam_probability:6.1%}  {r.message[:90]}")
        if len(res) == 1:
            for w, d in explain(bundle, r.message):
                print(f"        {'spam' if d > 0 else 'ham ':4s} cue: {w!r} ({d:+.2f})")


if __name__ == "__main__":
    main()
