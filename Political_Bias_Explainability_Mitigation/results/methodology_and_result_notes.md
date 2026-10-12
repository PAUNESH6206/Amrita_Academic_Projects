# Methods, saved results and reproducibility notes

## Shared task

The project uses article text (`page_text`) to predict one of five dataset-provided ideological leaning labels (`bias`). The paper proposes preprocessing, tokenization, class balancing, transformer fine-tuning, evaluation, SHAP explainability and a BART + LoRA mitigation experiment.

## Per-notebook implementation

- **ALBERT:** cleans missing/duplicate/empty records, balances the classes before splitting, encodes labels, uses `albert-base-v2`, max length 128, AdamW, learning rate 2e-5, batch size 8, weight decay 0.01 and early stopping (patience 4). Reports accuracy, precision, recall, weighted/macro F1, MCC, Cohen's kappa, class report and diagnostic plots. The original first cell heading mislabeled the notebook as DeBERTa; the Kaggle-ready copy corrects the heading only.
- **DistilBERT:** balances and tokenizes the classification data, fine-tunes `distilbert-base-uncased` (max length 256; up to 40 epochs; early stopping) and reports classification metrics, confusion matrix and plots. The supplied notebook has multiple separate saved evaluations: one reports accuracy 0.6673, while a later evaluation reports 0.6943. This package records both; neither should be silently selected as a single canonical run until rerun with a fixed split and versioned configuration. A later cell performs SHAP-based analysis.
- **RoBERTa:** cleans to 9,457 examples in the saved output, uses an unbalanced cleaned set with a stratified 70/15/15 split (6,619/1,419/1,419), `roberta-base`, max length 256, class-weighted loss, AdamW, learning rate 1e-5, batch size 4, gradient accumulation 2, weight decay 0.05, and early stopping (patience 4). It includes confusion matrix, ROC/PR curves, confidence/per-class analysis and SHAP explanations. It saves model and tokenizer to `final_roberta_bias_model`.
- **BART + LoRA:** creates balanced train/validation/test splits (70/15/15) and constructs target text through a hand-written word substitution/neutralization dictionary and filtering/augmentation rules. It fine-tunes `facebook/bart-base` with PEFT LoRA (r=16, alpha=32, dropout=0.10, q/v projections; learning rate 2e-4, batch size 4, gradient accumulation 2, max 40 epochs and early stopping). It saves the adapter/tokenizer to `final_bart_lora_mitigation_model`. Its ROUGE/BLEU values are overlap against heuristic generated references, not human judgments of neutrality, factual preservation, or fairness.

## Saved source-notebook metrics

These are transcribed from the outputs embedded in the supplied notebooks (which appear to have been run in different environments/splits). They are not results of the cleaned Kaggle-ready copies.

| Notebook | Test accuracy | Precision | Recall | Weighted F1 | Macro F1 | Additional |
|---|---:|---:|---:|---:|---:|---|
| ALBERT | 0.6601 | 0.6714 | 0.6601 | 0.6585 | 0.6585 | MCC 0.5779; kappa 0.5751; saved output test N=1,109 |
| DistilBERT (evaluation A) | 0.6673 | 0.6684 | 0.6673 | 0.6668 | not separately confirmed | A distinct evaluation cell reports this summary |
| DistilBERT (evaluation B) | 0.6943 | 0.7049 | 0.6943 | 0.6953 | 0.6953 | MCC 0.6199; kappa 0.6179 |
| RoBERTa | 0.7773 | 0.7818 | 0.7773 | 0.7766 | 0.7719 | MCC 0.7137; kappa 0.7122; saved output test N=1,419 |
| BART + LoRA mitigation | n/a | n/a | n/a | n/a | n/a | Average ROUGE-1 F1 0.8518; ROUGE-2 F1 0.8320; ROUGE-L F1 0.8406; BLEU is generated in the notebook. |

Metrics are in `[previous_notebook_results.csv](previous_notebook_results.csv)`. BART generation scores are not classification metrics and cannot be compared against accuracy/F1. The classifier test sets differ in sample count and balancing; the numbers are not a controlled model leaderboard.

## Kaggle rerun status (12 Oct 2026)

- **ALBERT:** Kaggle T4x2 execution completed through preprocessing, training, test evaluation and diagnostic plots. Training stopped early before the 40-epoch ceiling. The model save cell completed, and the added results exporter wrote JSON/CSV files under `/kaggle/working/results`. Kaggle notebook [Political Bias Classification — ALBERT](https://www.kaggle.com/code/ponssvvv/political-bias-classification-albert) Version 1 saved successfully. The Kaggle test metrics (N=1,109) are accuracy 0.646528, weighted F1 0.646143, macro F1 0.646151, MCC 0.558496 and Cohen's kappa 0.558155. This is a fresh run and differs from the historic embedded output; preserve both rows in the CSV.
- **DistilBERT, RoBERTa, and BART + LoRA:** not yet rerun on Kaggle. Their source-notebook figures remain historical only.
- **Artifacts:** the committed notebook contains the exporter code. Kaggle keeps the saved execution output and model artifacts with its version; no large trained weights are placed in GitHub.

## Items to verify during Kaggle reruns

1. Confirm the attached Kaggle dataset contains `bias_clean.csv` with `page_text` and `bias`; the importer now searches for the filename rather than assuming a particular mount path.
2. Record the Kaggle notebook version, accelerator, package versions, seed, split sizes and exact model checkpoint for each run. The previous outputs show different test sizes (1,109 vs 1,419) and DistilBERT duplicates.
3. Inspect label mapping and whether class balancing occurs before or after the split. Balancing before splitting can cause duplicated/resampled examples to leak across train/test when oversampling is used; confirm the actual sampling method and correct it if needed before presenting the test score.
4. Confirm each final inference pipeline uses the same label mapping as training. RoBERTa saved output from one SHAP example displays generic labels such as `LABEL_0`, indicating that label names should be wired into the saved model config before deployment.
5. Review BART-generated samples with human evaluation. Lexical substitutions may change meaning, and heuristic reference overlap cannot establish that a rewrite is neutral or preserves factual meaning.

## Export and models

Each classification notebook includes `save_pretrained`/trainer model save code and now adds an explicit `/kaggle/working/results` JSON metrics export plus confusion matrix/report CSV export. BART saves its LoRA adapter and tokenizer and now adds a compact JSON metrics export. Save a Kaggle notebook version to persist run outputs and model artifacts; do not commit multi-hundred-megabyte base-model checkpoints to GitHub.

