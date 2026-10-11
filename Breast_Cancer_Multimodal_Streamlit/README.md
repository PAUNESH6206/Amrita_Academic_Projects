<div align="center">

# Breast Cancer Multimodal Model Studio

**An approachable Streamlit dashboard for breast-imaging model exploration**

Upload an image, explore the expert models, or combine histopathology and ultrasound with the fusion model.

</div>

---

## What you can do

- Run a histopathology, ultrasound, or X-ray expert model with one image.
- Run the fusion model with a matched histopathology + ultrasound image pair.
- Review the notebook's validation accuracy, class F1 scores, and fusion AUC.
- Upload JPG, JPEG, and PNG files, or paste an image from the clipboard in the prediction input; preprocessing resizes images to 224 × 224 RGB and scales pixels to `[0, 1]`.

## Models and validation results

| Model | Validation accuracy | Validation examples | Class 0 F1 | Class 1 F1 |
|---|---:|---:|---:|---:|
| Histopathology | 71% | 187 | 0.69 | 0.72 |
| Ultrasound | 85% | 121 | 0.85 | 0.85 |
| X-ray | 63% | 150 | 0.56 | 0.67 |
| Histopathology + ultrasound fusion | 73.28% | 116 paired | 0.75 | 0.71 |

The fusion model's recorded AUC is **0.8128**. Each model was evaluated on its own validation set, so these results are not a controlled head-to-head comparison.

## Dataset

The notebook points to the [Breast Cancer MSI Multimodal Image Dataset on Kaggle](https://www.kaggle.com/datasets/zoya77/breast-cancer-msi-multimodal-image-dataset). Download the dataset from Kaggle and follow its access and usage terms. The dataset is not copied into this repository; the app accepts images selected from your device.

## Run locally

Python 3.11 or 3.12 is recommended. Streamlit 1.65 or later is required for clipboard-paste input.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new app from this GitHub repository.
2. Select `Breast_Cancer_Multimodal_Streamlit` as the **Main file path** and `app.py` as the entry point.
3. Choose Python 3.11 or 3.12 in the advanced deployment settings.
4. Deploy. `requirements.txt`, `.streamlit/config.toml`, and the four model files are in this folder.

## Model files

```text
models/histopathology_model.keras
models/ultrasound_model.keras
models/xray_model.keras
models/hist_ultrasound_fusion_model.keras
```

Each expert expects one `(224, 224, 3)` RGB image. The fusion model expects two inputs named `histopathology_input` and `ultrasound_input`, each with shape `(224, 224, 3)`.

## Responsible use and privacy

This is an educational research prototype, not a medical device. Predictions are model scores, not diagnoses or calibrated clinical probabilities. Do not use this dashboard for screening, diagnosis, treatment, or medical decisions.

When hosted, uploaded or pasted images are sent to the Streamlit app server for inference. Pasting avoids creating a separate image file on your device, but does not keep the image on your device alone. The app code does not save the image to disk. Do not upload identifiable or sensitive images to a deployment unless you have authorization and have reviewed the hosting provider's privacy terms.

## Project structure

```text
Breast_Cancer_Multimodal_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── models/
    ├── histopathology_model.keras
    ├── ultrasound_model.keras
    ├── xray_model.keras
    └── hist_ultrasound_fusion_model.keras
```

