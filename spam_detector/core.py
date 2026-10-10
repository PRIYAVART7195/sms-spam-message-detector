"""Shared data loading, feature engineering, model definitions and inference helpers.

Used by train_model.py, predict.py and app.py so all three behave identically.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection.csv"
OUT_DIR = ROOT / "outputs"
MODEL_PATH = OUT_DIR / "spam_detector.joblib"

LABEL_MAP = {"legitimate": "ham", "not spam": "ham", "junk": "spam"}
RANDOM_STATE = 42


# --------------------------------------------------------------------------- data
def load_data(path: Path = DATA_PATH, dedupe: bool = True) -> pd.DataFrame:
    """Load and clean the dataset.

    Exact duplicate messages are dropped by default: the raw SMS Spam Collection
    contains ~400 of them, and leaving them in lets identical texts land in both the
    train and test split, which inflates the reported scores.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See data/README_DATA.md.")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if not {"label", "message"}.issubset(df.columns):
        raise ValueError("CSV must contain columns named 'label' and 'message'.")
    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].astype(str).str.lower().str.strip().replace(LABEL_MAP)
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"]) & (df["message"] != "")]
    if dedupe:
        df = df.drop_duplicates(subset="message", keep="first")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------- features
_URL = re.compile(r"(https?://|www\.|\b\w+\.(com|co\.uk|net|org|info|biz)\b)", re.I)
_PHONE = re.compile(r"\b\d{5,}\b")
_MONEY = re.compile(r"[£$€]|\b(?:rs|inr)\.?\s?\d", re.I)
_URGENT = re.compile(r"\b(urgent|now|free|win|won|winner|prize|claim|cash|txt|reply|call|stop)\b", re.I)


def _style_features(texts) -> np.ndarray:
    """Hand-crafted, non-negative, roughly 0-1 scaled cues that spam often carries."""
    rows = []
    for t in texts:
        t = str(t)
        n = max(len(t), 1)
        letters = [c for c in t if c.isalpha()]
        upper = sum(c.isupper() for c in letters) / max(len(letters), 1)
        digits = sum(c.isdigit() for c in t) / n
        rows.append([
            float(bool(_URL.search(t))),
            float(bool(_PHONE.search(t))),
            float(bool(_MONEY.search(t))),
            min(len(_URGENT.findall(t)) / 3, 1.0),
            upper,
            digits,
            min(t.count("!") / 3, 1.0),
            min(n / 300, 1.0),
        ])
    return np.asarray(rows, dtype=float)


def _features(char_ngrams: bool = True) -> FeatureUnion:
    parts = [("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                      strip_accents="unicode"))]
    if char_ngrams:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                              sublinear_tf=True, min_df=2, max_features=60000)))
        parts.append(("style", FunctionTransformer(_style_features, validate=False)))
    return FeatureUnion(parts)


def build_candidates() -> dict[str, Pipeline]:
    """The models compared during training (all share one fit/predict_proba API)."""
    return {
        "naive_bayes": Pipeline([
            ("features", _features(char_ngrams=False)),
            ("model", MultinomialNB(alpha=0.1)),
        ]),
        "logistic_regression": Pipeline([
            ("features", _features()),
            ("model", LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
        ]),
        "linear_svm": Pipeline([
            ("features", _features()),
            ("model", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)),
        ]),
    }


# ---------------------------------------------------------------------- inference
def load_bundle(path: Path = MODEL_PATH) -> dict:
    """Load the saved bundle: {'pipeline', 'threshold', 'model_name', 'metrics'}."""
    return joblib.load(path)


def spam_probability(pipeline, texts) -> np.ndarray:
    classes = list(pipeline.classes_)
    return pipeline.predict_proba(list(texts))[:, classes.index("spam")]


def predict(bundle: dict, texts) -> pd.DataFrame:
    """Classify texts using the tuned decision threshold stored in the bundle."""
    texts = [str(t) for t in texts]
    p = spam_probability(bundle["pipeline"], texts)
    return pd.DataFrame({
        "message": texts,
        "spam_probability": p,
        "prediction": np.where(p >= bundle["threshold"], "spam", "ham"),
    })


def explain(bundle: dict, text: str, top: int = 5) -> list[tuple[str, float]]:
    """Model-agnostic word-level explanation by occlusion.

    Removes each word in turn and measures how much the spam log-odds drop (log-odds
    stay informative even when the probability is saturated near 0% or 100%).
    Positive values push the message towards spam, negative towards ham.
    """
    words = text.split()
    if not words:
        return []
    pipe = bundle["pipeline"]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    probs = spam_probability(pipe, [text] + variants)
    deltas = logit(probs[0]) - logit(probs[1:])
    scored = sorted(zip(words, deltas), key=lambda x: abs(x[1]), reverse=True)
    seen, out = set(), []
    for w, d in scored:
        if w.lower() in seen or abs(d) < 0.05:
            continue
        seen.add(w.lower())
        out.append((w, float(d)))
        if len(out) == top:
            break
    return out
