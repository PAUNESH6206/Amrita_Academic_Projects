from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).parent
FEATURES = [
    "chol",
    "stab.glu",
    "hdl",
    "ratio",
    "glyhb",
    "age",
    "bp.1s",
    "bp.1d",
    "time.ppn",
]
LABELS = {0: "No diabetes predicted", 1: "Diabetes predicted"}

st.set_page_config(page_title="Diabetes Classification", page_icon="🩺", layout="wide")
st.title("Diabetes Classification")
st.caption(
    "A machine-learning demonstration using the project's saved StandardScaler, "
    "PCA transform, and XGBoost classifier."
)
st.warning(
    "This model is for educational demonstration only. It is not a medical device "
    "and cannot diagnose diabetes. Please consult a qualified healthcare professional."
)

@st.cache_resource
def load_models():
    scaler = joblib.load(PROJECT_DIR / "scaler.pkl")
    pca = joblib.load(PROJECT_DIR / "pca_model.pkl")
    classifier = joblib.load(PROJECT_DIR / "xgboost_model.pkl")
    return scaler, pca, classifier

try:
    scaler, pca, classifier = load_models()
except Exception as exc:
    st.error(f"Could not load the saved model files: {exc}")
    st.stop()

st.subheader("Enter the nine model features")
defaults = {
    "chol": 200.0,
    "stab.glu": 100.0,
    "hdl": 45.0,
    "ratio": 4.5,
    "glyhb": 5.5,
    "age": 45.0,
    "bp.1s": 120.0,
    "bp.1d": 80.0,
    "time.ppn": 180.0,
}
columns = st.columns(3)
values = {}
for i, feature in enumerate(FEATURES):
    with columns[i % len(columns)]:
        values[feature] = st.number_input(
            feature,
            value=defaults[feature],
            format="%.2f",
            help=f"Enter the measured {feature} value used by this model.",
        )

if st.button("Run classification", type="primary"):
    input_frame = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    try:
        transformed = pca.transform(scaler.transform(input_frame))
        predicted = int(classifier.predict(transformed)[0])
        probabilities = classifier.predict_proba(transformed)[0]
        positive_index = list(classifier.classes_).index(1)
        probability = float(probabilities[positive_index])
        st.subheader("Model output")
        st.metric("Predicted class", LABELS.get(predicted, str(predicted)))
        st.metric("Model probability for class 1", f"{probability:.1%}")
        st.caption(
            "This probability reflects this model's output, not an individual's "
            "clinical probability or diagnosis."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

csv_path = PROJECT_DIR / "df_final_processed.csv"
if csv_path.exists():
    with st.expander("Explore the processed project dataset"):
        data = pd.read_csv(csv_path)
        st.write(f"{len(data):,} rows · {len(data.columns)} columns")
        st.dataframe(data.head(25), use_container_width=True)
