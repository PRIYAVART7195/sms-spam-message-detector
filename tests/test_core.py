import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spam_detector.core import MODEL_PATH, explain, load_bundle, load_data, predict  # noqa: E402


def test_load_data_cleans_and_dedupes(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"label": ["HAM", "ham", "junk", "spam", "weird", None],
                  "message": ["hi", "hi", "win cash", "win cash now", "x", "y"]}).to_csv(p, index=False)
    df = load_data(p)
    assert set(df.label) <= {"ham", "spam"}
    assert len(df) == 3  # duplicate 'hi' dropped, 'weird' and None dropped
    assert len(load_data(p, dedupe=False)) == 4


def test_load_data_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_data(p)


@pytest.fixture(scope="module")
def bundle():
    if not MODEL_PATH.exists():
        pytest.skip("run train_model.py first")
    return load_bundle()


def test_predictions(bundle):
    res = predict(bundle, ["WINNER!! Claim your free £1000 prize now, call 09061701461",
                           "Are we meeting for lunch tomorrow?"])
    assert list(res.prediction) == ["spam", "ham"]
    assert res.spam_probability.between(0, 1).all()


def test_empty_and_odd_input(bundle):
    assert predict(bundle, [""]).shape[0] == 1
    assert explain(bundle, "") == []
    predict(bundle, ["😀" * 50, "a" * 5000])


def test_explain_flags_spam_words(bundle):
    cues = dict(explain(bundle, "Free prize! Claim your cash now"))
    assert any(d > 0 for d in cues.values())


def test_quality_floor(bundle):
    m = bundle["metrics"]["test"]
    assert m["f1_spam"] > 0.9 and m["precision_spam"] > 0.95
