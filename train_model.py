
from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from spam_detector.core import (DATA_PATH, MODEL_PATH, OUT_DIR, RANDOM_STATE,
                                build_candidates, load_data, spam_probability)


def baseline_pipeline() -> Pipeline:
    """The original project's model (word TF-IDF + LR) kept as a reference point."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def tune_threshold(y_true, p_spam, min_precision: float) -> float:
    """Highest-F1 threshold whose precision is at least min_precision (else best F1)."""
    prec, rec, thr = precision_recall_curve(np.asarray(y_true) == "spam", p_spam)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    ok = prec >= min_precision
    idx = np.argmax(np.where(ok, f1, -1)) if ok.any() else np.argmax(f1)
    return float(thr[idx])


def metrics_at(y_true, p_spam, threshold: float) -> dict:
    pred = np.where(p_spam >= threshold, "spam", "ham")
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_spam": precision_score(y_true, pred, pos_label="spam", zero_division=0),
        "recall_spam": recall_score(y_true, pred, pos_label="spam", zero_division=0),
        "f1_spam": f1_score(y_true, pred, pos_label="spam", zero_division=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-precision", type=float, default=0.98)
    args = ap.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    raw = load_data(args.data, dedupe=False)
    df = load_data(args.data, dedupe=True)
    print(f"Rows: {len(raw)} raw -> {len(df)} after removing {len(raw) - len(df)} duplicate messages")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df.message, df.label, test_size=0.2, random_state=RANDOM_STATE, stratify=df.label)
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=RANDOM_STATE)

    candidates = {"baseline_tfidf_lr": baseline_pipeline(), **build_candidates()}
    rows, oof = [], {}
    for name, pipe in candidates.items():
        p_oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)
        p_oof = p_oof[:, list(np.unique(y_tr)).index("spam")]
        thr = tune_threshold(y_tr, p_oof, args.min_precision)
        oof[name] = (p_oof, thr)
        pipe.fit(X_tr, y_tr)
        p_te = spam_probability(pipe, X_te)
        row = {"model": name, "cv_f1_spam": metrics_at(y_tr, p_oof, 0.5)["f1_spam"],
               "cv_f1_spam_tuned": metrics_at(y_tr, p_oof, thr)["f1_spam"], "threshold": thr,
               "roc_auc": roc_auc_score(y_te == "spam", p_te)}
        row.update({f"test_{k}": v for k, v in metrics_at(y_te, p_te, thr).items()})
        rows.append(row)
        print(f"  {name:22s} CV F1 {row['cv_f1_spam_tuned']:.4f} | test F1 {row['test_f1_spam']:.4f} (thr {thr:.3f})")

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_DIR / "model_comparison.csv", index=False)

   
    best_name = comp[comp.model != "baseline_tfidf_lr"].sort_values("cv_f1_spam_tuned", ascending=False).iloc[0].model
    best, threshold = candidates[best_name], oof[best_name][1]
    p_te = spam_probability(best, X_te)
    pred = np.where(p_te >= threshold, "spam", "ham")

    cm = confusion_matrix(y_te, pred, labels=["ham", "spam"])
    (OUT_DIR / "classification_report.txt").write_text(classification_report(y_te, pred, digits=3))
    errors = pd.DataFrame({"message": X_te.values, "actual": y_te.values, "predicted": pred,
                           "spam_probability": p_te.round(4)})
    errors[errors.actual != errors.predicted].sort_values("spam_probability").to_csv(
        OUT_DIR / "misclassified.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred ham", "Pred spam"])
    ax.set_yticks([0, 1], ["Actual ham", "Actual spam"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_title(f"{best_name}\nthreshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    final = metrics_at(y_te, p_te, threshold)
    metrics = {
        "selected_model": best_name, "threshold": threshold, "min_precision": args.min_precision,
        "rows_raw": len(raw), "rows_clean": len(df), "duplicates_removed": len(raw) - len(df),
        "train_examples": len(X_tr), "test_examples": len(X_te),
        "class_counts": df.label.value_counts().to_dict(),
        "test": final, "roc_auc": float(roc_auc_score(y_te == "spam", p_te)),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "folds": args.folds, "random_state": RANDOM_STATE,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    
    deploy = build_candidates()[best_name] if best_name in build_candidates() else best
    deploy.fit(df.message, df.label)
    joblib.dump({"pipeline": deploy, "threshold": threshold, "model_name": best_name,
                 "metrics": metrics}, MODEL_PATH, compress=3)
    print(f"\nSelected {best_name} | thr {threshold:.3f} | acc {final['accuracy']:.4f} | "
          f"P {final['precision_spam']:.4f} R {final['recall_spam']:.4f} F1 {final['f1_spam']:.4f} "
          f"| FP {cm[0,1]} FN {cm[1,0]} | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
