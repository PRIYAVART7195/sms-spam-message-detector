"""Streamlit dashboard for the SMS Spam Detector.

The app loads the pre-trained model from outputs/ (instant start-up). If it is missing it
trains one automatically the first time and caches it.   Run:  streamlit run app.py
"""
import json
import subprocess
import sys

import pandas as pd
import streamlit as st

from spam_detector.core import (DATA_PATH, MODEL_PATH, OUT_DIR, ROOT, explain, load_bundle,
                                load_data, predict)

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="wide")

EXAMPLES = {
    "Prize scam": "URGENT! You have won a £1000 cash prize. Call 09061701461 now to claim. T&C apply.",
    "Friend": "Are we still meeting after class today? Let me know when you reach the library.",
    "Bank phishing": "Your account is suspended. Verify now at www.secure-bank-login.info or lose access.",
    "Reminder": "Reminder: your assignment is due tomorrow at 10am. Please submit on the portal.",
}


@st.cache_resource(show_spinner="Training model for the first time (about 30 seconds)…")
def get_bundle():
    if not MODEL_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "train_model.py")], check=True, cwd=ROOT)
    return load_bundle()


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_comparison():
    p = OUT_DIR / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else None


try:
    bundle = get_bundle()
    df = get_data()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not start: {exc}")
    st.stop()

m = bundle["metrics"]
st.title("📩 SMS Spam Message Detector")
st.write("Classifies SMS text as **spam** or **ham** (legitimate) using word + character TF-IDF "
         "features, text-style cues and a threshold-tuned machine-learning model.")

c = st.columns(5)
c[0].metric("Model", {"logistic_regression": "Logistic Reg.", "naive_bayes": "Naive Bayes", "linear_svm": "Linear SVM"}.get(bundle["model_name"], bundle["model_name"]))
c[1].metric("Test accuracy", f"{m['test']['accuracy']:.1%}")
c[2].metric("Spam precision", f"{m['test']['precision_spam']:.1%}")
c[3].metric("Spam recall", f"{m['test']['recall_spam']:.1%}")
c[4].metric("Spam F1", f"{m['test']['f1_spam']:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Classify", "📁 Batch (CSV)", "📊 Evaluation", "🗂️ Dataset"])

with tab1:
    ex = st.selectbox("Load an example", ["—"] + list(EXAMPLES))
    msg = st.text_area("SMS text", value=EXAMPLES.get(ex, ""), height=120,
                       placeholder="Paste a message to check…")
    if st.button("Check message", type="primary"):
        if not msg.strip():
            st.warning("Enter a message first.")
        else:
            r = predict(bundle, [msg]).iloc[0]
            p = float(r.spam_probability)
            (st.error if r.prediction == "spam" else st.success)(
                f"Prediction: {r.prediction.upper()}" + ("" if r.prediction == "spam" else " (likely legitimate)"))
            st.progress(min(max(p, 0.0), 1.0), text=f"Spam score {p:.1%} (decision threshold {bundle['threshold']:.0%})")
            cues = explain(bundle, msg)
            if cues:
                st.markdown("**Words that influenced the result** (change in spam log-odds when removed)")
                st.dataframe(pd.DataFrame(cues, columns=["word", "effect"]).assign(
                    direction=lambda d: d.effect.map(lambda v: "→ spam" if v > 0 else "→ ham")),
                    hide_index=True, width="stretch")
            st.caption("A model estimate, not a guarantee.")

with tab2:
    st.write("Upload a CSV with a **message** column (or a .txt with one message per line).")
    up = st.file_uploader("File", type=["csv", "txt"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                src = pd.read_csv(up)
                col = "message" if "message" in src.columns else st.selectbox("Message column", src.columns)
                texts = src[col].dropna().astype(str).tolist()
            else:
                texts = [l.strip() for l in up.read().decode("utf-8", "ignore").splitlines() if l.strip()]
            if len(texts) > 50000:
                st.warning("Only the first 50,000 messages are scored.")
                texts = texts[:50000]
            res = predict(bundle, texts)
            st.write(f"{len(res):,} messages scored — **{(res.prediction == 'spam').sum():,} spam**")
            st.dataframe(res.sort_values("spam_probability", ascending=False), width="stretch")
            st.download_button("Download results", res.to_csv(index=False), "spam_results.csv", "text/csv")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that file: {exc}")

with tab3:
    cm = m["confusion_matrix"]
    l, r_ = st.columns([1, 1])
    with l:
        st.subheader("Held-out test results")
        st.write(f"{m['test_examples']:,} test messages · ROC-AUC **{m['roc_auc']:.3f}** · "
                 f"{cm['fp']} false positive(s), {cm['fn']} missed spam")
        png = OUT_DIR / "confusion_matrix.png"
        if png.exists():
            st.image(str(png), width=380)
    with r_:
        comp = get_comparison()
        if comp is not None:
            st.subheader("Model comparison")
            show = comp[["model", "cv_f1_spam_tuned", "test_precision_spam", "test_recall_spam", "test_f1_spam"]]
            show.columns = ["Model", "CV F1", "Test precision", "Test recall", "Test F1"]
            st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[1:]}),
                         hide_index=True, width="stretch")
        err = OUT_DIR / "misclassified.csv"
        if err.exists():
            with st.expander("Misclassified test messages"):
                st.dataframe(pd.read_csv(err), width="stretch")
    st.info(f"Duplicates removed before splitting: {m['duplicates_removed']}. Model selection used "
            f"{m['folds']}-fold cross-validation on training data only; the test set was used once.")

with tab4:
    st.write(f"{len(df):,} unique messages after cleaning ({m['duplicates_removed']} duplicates removed).")
    st.bar_chart(df["label"].value_counts())
    st.dataframe(df.sample(min(100, len(df)), random_state=1), width="stretch")
    st.caption("Source: UCI SMS Spam Collection (Almeida et al., 2011).")

st.divider()
st.caption("Academic prototype. Do not use as the sole basis for blocking important messages.")
