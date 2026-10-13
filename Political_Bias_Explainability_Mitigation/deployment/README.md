# Bias Mitigation Studio

Streamlit interface for the BART-base + LoRA political-bias mitigation experiment.

## Run locally

From this directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The LoRA adapter and BART tokenizer are included in `model/`. The public `facebook/bart-base` weights download on first launch.

## Deploy on Streamlit Community Cloud

Choose this repository, branch `main`, and set the app file to:

```
Political_Bias_Explainability_Mitigation/deployment/app.py
```

This app generates suggested rewrites; it does not classify political orientation. Review all generated output for factual accuracy, meaning, and context. The displayed BLEU/ROUGE scores are overlap metrics on a small fixed sample and do not establish neutrality or factual correctness.
