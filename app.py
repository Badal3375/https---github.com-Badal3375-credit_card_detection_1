import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Paths to saved artifacts
MODEL_PATH = Path("model.pkl")
VECT_PATH  = Path("vectorizer.pkl")

# Load model + vectorizer
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECT_PATH, "rb") as f:
        vec = pickle.load(f)
    scaler = vec["scaler"]              # ✅ extract scaler from dict
    feature_names = vec["feature_names"]  # list of features
    return model, scaler, feature_names

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Predict fraud probability using a trained model. Upload CSV for batch scoring or enter a single record.")

model, scaler, feature_names = load_artifacts()

# Tabs: single vs batch
tab1, tab2 = st.tabs(["Single Prediction", "Batch CSV Prediction"])

# --- Single input prediction
with tab1:
    st.subheader("Enter Transaction Details")
    values = []
    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        with cols[i % 2]:
            values.append(st.number_input(feat, value=0.0, step=0.1, format="%.4f"))

    if st.button("Predict"):
        # Ensure correct feature order
        X = np.array(values, dtype=float).reshape(1, -1)
        X_sc = scaler.transform(X)
        proba = model.predict_proba(X_sc)[0, 1]
        pred = int(proba >= 0.5)
        st.metric("Fraud Probability", f"{proba:.4f}")
        st.write("Prediction:", "**🚨 FRAUD**" if pred == 1 else "✅ Legit Transaction")

# --- Batch CSV prediction
with tab2:
    st.subheader("Upload CSV for Bulk Prediction")
    file = st.file_uploader("Choose a CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            X = df[feature_names].values
            X_sc = scaler.transform(X)
            proba = model.predict_proba(X_sc)[:, 1]
            pred = (proba >= 0.5).astype(int)
            out = df.copy()
            out["fraud_proba"] = proba
            out["prediction"] = pred
            st.write(out.head())
            st.download_button(
                "Download Predictions CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

st.info("⚠️ Expected feature order: " + ", ".join(feature_names))
