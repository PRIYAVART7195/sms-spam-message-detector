# SMS Spam Message Detector — AIML Minor Project

Classifies SMS messages as **spam** or **ham** (legitimate). Word + character TF-IDF features and text-style cues feed a cross-validated, threshold-tuned model, served through a Streamlit dashboard and a command-line tool.

## Results (held-out test set, 1,032 unique messages)
| Metric | Original project* | Upgraded |
|---|---|---|
| Accuracy | 97.0% | **99.4%** |
| Spam precision | 99.0% | **99.2%** |
| Spam recall | 76.6% | **96.1%** |
| Spam F1 | 0.863 | **0.976** |
| ROC-AUC | 0.991 | **0.999** |

\*Original model re-evaluated on the same de-duplicated split. The original code reported F1 0.925 / accuracy 97.9%, but those numbers were inflated: the dataset has 414 exact-duplicate messages, so identical texts appeared in both train and test.
Errors on the test set: 1 legitimate message flagged as spam, 5 spam messages missed (see `outputs/misclassified.csv`). With only 128 spam messages in the test set, scores move by a point or two between splits; the 5-fold CV spam-F1 of 0.951 on training data is the more conservative estimate.

## What was upgraded
- **Leakage fixed:** exact duplicates removed before the split; model chosen by 5-fold cross-validation on training data only, so the test set is used exactly once.
- **Better features:** word 1–2-grams + character 2–5-grams (robust to obfuscation like `fr33`, `txt4`) + cues (URL, long phone number, currency, urgency words, caps ratio, digits, `!`, length). Stop-word removal was dropped because words like *now*, *free*, *call* are strong spam signals.
- **Three models compared** (Naive Bayes, Logistic Regression, calibrated Linear SVM) plus the original model as a baseline.
- **Threshold tuning:** maximises F1 subject to spam precision ≥ 98 %, because blocking a real message costs more than missing spam.
- **Fast app:** loads the saved model in a moment instead of retraining on every launch. Auto-trains once if the model file is missing.
- **New features:** word-level explanations, batch CSV scoring with download, example messages, CLI (`predict.py`), metrics JSON, confusion-matrix image, misclassified-message list, unit tests.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py             # dashboard at http://localhost:8501
```
Retrain (about 30 s): `python train_model.py`   Run tests: `python -m pytest tests`

Command line:
```bash
python predict.py "URGENT! You won a prize, call 09061701461 now"
python predict.py --file messages.csv --column message --out results.csv
```

## Project layout
```
app.py                  Streamlit dashboard (Classify / Batch / Evaluation / Dataset)
train_model.py          CV model selection, threshold tuning, evaluation, saves outputs/
predict.py              CLI single/batch prediction
spam_detector/core.py   Data loading, features, models, prediction, explanations
tests/test_core.py      Unit tests
data/                   SMSSpamCollection.csv (UCI) + README_DATA.md
outputs/                Trained model, metrics.json, comparison CSV, confusion matrix, errors
docs/screenshots/       App screenshots used in the report and slides
PROJECT_REPORT.md/.docx, PROJECT_PRESENTATION.pptx, PRESENTATION_OUTLINE.md
```

## Dataset format
CSV with `label` (`ham`/`spam`) and `message` columns; extra columns are ignored. To use your own data: `python train_model.py --data path/to/file.csv`.

## Limitations
English SMS from the mid-2000s: new scam styles, languages, or slang will reduce accuracy. The model does not check URLs or senders. The dataset's 5,160 unique messages are small, so retrain on fresh data before any real use. Prototype only: do not auto-delete messages on its output.

## Citation
Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, T. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* ACM DocEng. DOI 10.1145/2034691.2034742. Data: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
