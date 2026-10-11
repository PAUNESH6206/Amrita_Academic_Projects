from __future__ import annotations

import os
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Histopathology": ROOT / "models" / "histopathology_model.keras",
    "Ultrasound": ROOT / "models" / "ultrasound_model.keras",
    "X-ray": ROOT / "models" / "xray_model.keras",
    "Fusion · Histopathology + Ultrasound": ROOT / "models" / "hist_ultrasound_fusion_model.keras",
}
if os.getenv("BREAST_CANCER_MODEL"):
    MODEL_PATHS["Fusion · Histopathology + Ultrasound"] = Path(os.environ["BREAST_CANCER_MODEL"])
AVAILABLE_MODELS = {name: path for name, path in MODEL_PATHS.items() if path.is_file()}

st.set_page_config(
    page_title="BreastSight · Model Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#172a3a; --muted:#718096; --teal:#167d80; --mint:#e6f4f1; --line:#e8edf0; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(180deg,#f7fbfa 0,#f5f8fb 32rem,#f7f9fc 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:#112b3a; }
    [data-testid="stSidebar"] * { color:#eaf3f4 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 10px; border-radius:10px; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.03em; }
    .hero { background:linear-gradient(120deg,#123848,#177c7b); border-radius:24px; padding:30px 34px; color:white; margin:4px 0 24px; box-shadow:0 18px 45px #1b52651a; }
    .hero h1 { color:white !important; font-size:2.25rem; margin:3px 0 8px; }
    .hero p { color:#d7ebeb; font-size:1.03rem; margin:0; max-width:760px; }
    .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:#9ee4d6; }
    .metric-card { background:white; border:1px solid var(--line); border-radius:18px; padding:20px 22px; min-height:112px; box-shadow:0 7px 24px #20354a08; }
    .metric-label { color:var(--muted); font-size:.83rem; font-weight:600; }
    .metric-value { font:800 1.9rem 'Manrope',sans-serif; color:var(--ink); margin-top:8px; }
    .metric-foot { color:#8a98a5; font-size:.77rem; margin-top:2px; }
    .section-card { background:white; border:1px solid var(--line); border-radius:18px; padding:22px; }
    .soft-note { background:#eaf5f3; color:#285c5d; border-radius:12px; padding:13px 16px; font-size:.91rem; }
    .warning-note { background:#fff6e8; color:#79561e; border-radius:12px; padding:13px 16px; font-size:.9rem; }
    .stButton > button[kind="primary"] { background:#167d80; border:0; border-radius:11px; font-weight:700; min-height:46px; }
    div[data-testid="stFileUploader"] { background:#fbfdfd; border:1px dashed #a9c6c6; border-radius:14px; padding:10px; }
    footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the fusion model…")
def load_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)


def metric_card(label: str, value: str, foot: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def image_tensor(upload) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(upload)
    image = ImageOps.exif_transpose(image).convert("RGB")
    display = image.copy()
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return display, np.expand_dims(array, axis=0)


def pasted_image_tensor(component_value):
    """Decode an image pasted into the clipboard component without writing it to disk."""
    if not isinstance(component_value, dict):
        return None
    for item in component_value.get("uploadedFiles", []):
        if not isinstance(item, dict) or not str(item.get("type", "")).startswith("image/"):
            continue
        data_url = item.get("content", "")
        if not isinstance(data_url, str) or "," not in data_url:
            continue
        try:
            raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
            return image_tensor(BytesIO(raw))
        except Exception:
            continue
    return None


def image_input(label: str, key: str):
    """Offer a normal file picker or clipboard paste for one image."""
    mode = st.radio(
        f"{label} input method",
        ["Upload a file", "Copy & paste"],
        horizontal=True,
        key=f"{key}_input_method",
        label_visibility="collapsed",
    )
    if mode == "Upload a file":
        upload = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"{key}_upload")
        if upload is None:
            return None
        return image_tensor(upload)

    from st_multimodal_chatinput import multimodal_chatinput

    value = multimodal_chatinput(
        key=f"{key}_paste",
        placeholder="Click here, then paste an image with Ctrl+V / ⌘V",
    )
    pasted = pasted_image_tensor(value)
    if pasted is None:
        st.caption("Copy an image, click the box above, and paste. You can also attach an image there.")
    return pasted


def get_probability(model, hist_array: np.ndarray, us_array: np.ndarray) -> float:
    inputs = {
        "histopathology_input": hist_array,
        "ultrasound_input": us_array,
    }
    result = model(inputs, training=False)
    return float(np.asarray(result).reshape(-1)[0])


with st.sidebar:
    st.markdown("<div style='font:800 1.25rem Manrope,sans-serif;margin:12px 4px 4px'>🩺 BreastSight</div>", unsafe_allow_html=True)
    st.caption("Breast imaging · Model Studio")
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "Try a prediction", "Model evaluation", "About & guidance"], label_visibility="visible")
    st.markdown("---")
    if AVAILABLE_MODELS:
        st.success(f"{len(AVAILABLE_MODELS)} model(s) ready")
        st.caption(", ".join(AVAILABLE_MODELS.keys()))
    else:
        st.warning("Prediction model not installed")
        st.caption("Add the exported .keras file to the models folder. Evaluation results remain available.")
    st.markdown("<div style='height:22vh'></div><div style='font-size:.77rem;color:#b5c8cc'>Educational prototype<br/>Not for clinical decisions</div>", unsafe_allow_html=True)


st.markdown(
    "<div class='hero'><div class='eyebrow'>MULTIMODAL BREAST CANCER RESEARCH</div>"
    "<h1>A clearer way to explore model results.</h1>"
    "<p>Bring two image types together, review the model’s score, and understand how the validation run performed — all in one simple workspace.</p></div>",
    unsafe_allow_html=True,
)

if page == "Overview":
    st.subheader("Your project at a glance")
    st.write("Explore three individual image models and a two-input fusion model. Start with the evaluation summary, or try a prediction using one image or a matched histopathology + ultrasound pair.")
    cols = st.columns(4)
    with cols[0]: metric_card("Best single model", "85%", "Ultrasound validation accuracy")
    with cols[1]: metric_card("Fusion AUC", "0.8128", "Area under the ROC curve")
    with cols[2]: metric_card("Fusion validation", "116", "63 class 0 · 53 class 1")
    with cols[3]: metric_card("Image size", "224 × 224", "RGB input for each modality")
    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("<div class='section-card'><h3>How to use this dashboard</h3><ol><li>Choose <b>Try a prediction</b>.</li><li>Select an individual model and upload its image, or choose fusion and upload a matched pair.</li><li>Review the score and predicted class together.</li><li>Open <b>Model evaluation</b> to compare reported validation results.</li></ol></div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section-card'><h3>What is ready?</h3><p>All four model files and the notebook's validation metrics are included.</p><div class='soft-note'>Try an individual image model or compare it with the paired fusion model.</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='warning-note'>Research prototype only. Outputs are model scores, not a diagnosis or a substitute for review by a qualified clinician.</div>", unsafe_allow_html=True)

elif page == "Try a prediction":
    st.subheader("Run a model")
    st.write("Choose an individual image model or the histopathology + ultrasound fusion model, then provide the matching image input(s).")
    st.markdown("<div class='soft-note'>Accepted formats: JPG, JPEG, or PNG. Images are resized to 224 × 224 and scaled to 0–1, matching the notebook preprocessing.</div>", unsafe_allow_html=True)
    st.write("")
    selected_model = st.selectbox("Choose a model", list(MODEL_PATHS.keys()))
    model_path = MODEL_PATHS[selected_model]
    is_fusion = selected_model.startswith("Fusion")
    if not model_path.is_file():
        st.markdown(f"<div class='warning-note'><b>{selected_model} model file is not installed.</b> Add its <code>.keras</code> file under <code>models/</code> to enable this option.</div>", unsafe_allow_html=True)
    if is_fusion:
        st.caption("Upload the histopathology and ultrasound images from the same case.")
        col1, col2 = st.columns(2)
        with col1:
            hist_input = image_input("① Histopathology image", "hist")
            st.caption("Tissue slide or pathology image")
        with col2:
            us_input = image_input("② Ultrasound image", "ultrasound")
            st.caption("Corresponding ultrasound image")
    else:
        st.caption(f"Upload one {selected_model.lower()} image.")
        single_input = image_input(f"{selected_model} image", "single_image")
        hist_input = us_input = None
    hist_array = us_array = single_array = None
    if is_fusion:
        if hist_input:
            hist_display, hist_array = hist_input
            st.image(hist_display, caption="Histopathology input", use_container_width=True)
        if us_input:
            us_display, us_array = us_input
            st.image(us_display, caption="Ultrasound input", use_container_width=True)
        ready = bool(hist_input and us_input and model_path.is_file())
    else:
        if single_input:
            single_display, single_array = single_input
            st.image(single_display, caption=f"{selected_model} input", use_container_width=True)
        ready = bool(single_input and model_path.is_file())
    button_text = "Analyze image pair" if is_fusion else "Analyze image"
    if st.button(button_text, type="primary", disabled=not ready, use_container_width=True):
        try:
            model = load_model(str(model_path))
            with st.spinner("Running model inference…"):
                if is_fusion:
                    probability = get_probability(model, hist_array, us_array)
                else:
                    probability = float(np.asarray(model(single_array, training=False)).reshape(-1)[0])
            predicted_malignancy = probability >= 0.5
            st.markdown("### Model output")
            a, b = st.columns([1, 2])
            with a:
                metric_card("Malignant-class score", f"{probability:.1%}", "Model score; not a clinical probability")
            with b:
                if predicted_malignancy:
                    st.error("The model selected class 1 (malignant) at the 0.50 threshold.")
                else:
                    st.success("The model selected class 0 (benign / non-malignant) at the 0.50 threshold.")
                st.progress(min(max(probability, 0.0), 1.0), text="Score for class 1")
            st.markdown("<div class='warning-note'>This prediction is for research demonstration only. Have imaging reviewed by a qualified clinician.</div>", unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Could not run the model: {exc}")

elif page == "Model evaluation":
    st.subheader("Validation performance")
    st.write("These results are recorded in the supplied notebook. The individual experts and fusion model used separate validation sets, so compare them as project results rather than a controlled head-to-head test.")
    model_report = pd.DataFrame({
        "Model": ["Histopathology", "Ultrasound", "X-ray", "Fusion · Histopathology + Ultrasound"],
        "Validation accuracy": [0.71, 0.85, 0.63, 0.7328],
        "Validation support": [187, 121, 150, 116],
        "Class 0 F1": [0.69, 0.85, 0.56, 0.75],
        "Class 1 F1": [0.72, 0.85, 0.67, 0.71],
    })
    cols = st.columns(4)
    with cols[0]: metric_card("Histopathology", "71%", "187 validation images")
    with cols[1]: metric_card("Ultrasound", "85%", "121 validation images")
    with cols[2]: metric_card("X-ray", "63%", "150 validation images")
    with cols[3]: metric_card("Fusion", "73.28%", "116 validation pairs")
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("<div class='section-card'><h3>Compare validation results</h3></div>", unsafe_allow_html=True)
        st.dataframe(model_report, hide_index=True, use_container_width=True, column_config={
            "Validation accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1, format="%.2f"),
            "Class 0 F1": st.column_config.ProgressColumn("Class 0 F1", min_value=0, max_value=1, format="%.2f"),
            "Class 1 F1": st.column_config.ProgressColumn("Class 1 F1", min_value=0, max_value=1, format="%.2f"),
        })
    with right:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=81.28, number={"suffix":"%"}, title={"text":"ROC AUC"}, gauge={"axis":{"range":[0,100]},"bar":{"color":"#167d80"},"bgcolor":"white","borderwidth":0,"steps":[{"range":[0,50],"color":"#f0f4f6"},{"range":[50,75],"color":"#d9eeea"},{"range":[75,100],"color":"#b9dfd8"}]}))
        fig.update_layout(height=260, margin=dict(l=20,r=20,t=55,b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#172a3a")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='soft-note'><b>Reading the AUC:</b> higher values mean the model more often ranks a positive example above a negative one across possible thresholds. AUC does not choose a safe clinical threshold.</div>", unsafe_allow_html=True)
    st.markdown("<div class='warning-note'>Metrics are copied from the notebook output. They describe that validation run and should not be treated as a guarantee of performance on new hospitals, scanners, or patient groups.</div>", unsafe_allow_html=True)
    st.caption("Fusion notebook report: accuracy 0.7328 · AUC 0.8128 · class 0 F1 0.75 · class 1 F1 0.71 · total support 116. Individual expert scores come from their own validation runs.")

else:
    st.subheader("About this prototype")
    st.write("This interface wraps the notebook’s histopathology, ultrasound, X-ray, and two-input fusion models. It applies the same 224 × 224 RGB scaling used in the notebook and displays the selected model’s sigmoid output.")
    st.markdown("### Labels and score")
    st.write("The notebook maps benign and normal examples to class 0, and malignant examples to class 1. The displayed score is the model output for class 1. A 0.50 threshold is used only to show a simple predicted class.")
    st.markdown("### Bring your own data")
    st.write("For an individual model, select one image. For fusion, select the histopathology and ultrasound images from the same case. You can upload a file or paste an image from your clipboard without creating a local image file. In a hosted deployment, either method sends image bytes to the Streamlit server for inference; this app does not save the images to disk.")
    st.markdown("### Limitations")
    st.write("The notebook reports different validation sets for each model (116 paired examples for fusion). These metrics are useful for project evaluation, but do not establish clinical safety or generalization. Do not use this prototype for screening, diagnosis, treatment, or other medical decisions.")
    st.markdown("### Dataset reference")
    st.markdown("The notebook references the [Breast Cancer MSI Multimodal Image Dataset on Kaggle](https://www.kaggle.com/datasets/zoya77/breast-cancer-msi-multimodal-image-dataset). Download access and usage are governed by Kaggle and the dataset owner.")
    st.markdown("### Model artifacts")
    st.code("models/histopathology_model.keras\nmodels/ultrasound_model.keras\nmodels/xray_model.keras\nmodels/hist_ultrasound_fusion_model.keras")
    st.write("The four supplied Keras model files are installed with this dashboard. Each is loaded only when selected for a prediction.")

