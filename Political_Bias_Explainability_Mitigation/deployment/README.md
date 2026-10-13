# Political Lens: Bias Classification and Mitigation

An interactive Streamlit research demo combining a five-class RoBERTa article classifier with a BART-base + LoRA rewrite assistant. The classifier and rewrite model load on demand so the two models do not need to occupy memory at the same time.

## Run locally

From this directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The BART LoRA adapter and tokenizer are bundled under `model/`. The public `facebook/bart-base` weights download on first use. The RoBERTa tokenizer/config are under `classifier_model/`; its 499 MB weights are downloaded on first use from the GitHub release `roberta-classifier-v1` and checked against a SHA-256 checksum.

## Deploy on Streamlit Community Cloud

Choose this repository, branch `main`, and entrypoint:

```
Political_Bias_Explainability_Mitigation/deployment/app.py
```

## Interpretation

RoBERTa predicts one of the five labels learned from the dataset: Center, Leaning-left, Leaning-right, Left, or Right. These reflect one dataset's annotations and are not objective political facts. BART suggests a rewrite; it does not classify political orientation. Review generated text for meaning and factual accuracy. BLEU/ROUGE are overlap metrics on a small fixed sample; they do not establish neutrality or factual correctness.
