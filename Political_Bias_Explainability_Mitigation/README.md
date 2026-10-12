# Explainable Political Bias Classification and Mitigation

A reproducible research project for five-class political leaning classification and exploratory bias mitigation. The work compares ALBERT, DistilBERT and RoBERTa classifiers, then explores BART fine-tuned with LoRA for heuristic text neutralization. The project paper is included in [`docs/project_paper.pdf`](docs/project_paper.pdf).

> **Research caveat:** These models predict labels present in a particular dataset. Political leaning is a subjective, context-dependent annotation; model predictions must not be treated as objective fact, publisher truth, or a measure of misinformation. This is an academic prototype, not a fact-checker.

## Project files

| File | Role | Methods in the notebook |
|---|---|---|
| [`01_ALBERT_Classification.ipynb`](notebooks/01_ALBERT_Classification.ipynb) | Five-class baseline | Dataset cleaning, class balancing, stratified train/validation/test split, ALBERT fine-tuning, early stopping, weighted F1, confusion matrix and curve analysis. |
| [`02_DistilBERT_Classification.ipynb`](notebooks/02_DistilBERT_Classification.ipynb) | Lightweight classifier | DistilBERT fine-tuning, balancing, stratified splits, evaluation metrics, confusion matrix and a separate SHAP analysis cell. |
| [`03_RoBERTa_Classification_SHAP.ipynb`](notebooks/03_RoBERTa_Classification_SHAP.ipynb) | Primary classifier and explanations | RoBERTa fine-tuning on cleaned articles, stratified split, weighted loss, early stopping, detailed metrics, curves and SHAP token explanations. |
| [`04_BART_LoRA_Mitigation.ipynb`](notebooks/04_BART_LoRA_Mitigation.ipynb) | Exploratory mitigation | Heuristic source/target generation, BART sequence-to-sequence fine-tuning with LoRA, generated examples, BLEU/ROUGE overlap scoring. |
| [`results/previous_notebook_results.csv`](results/previous_notebook_results.csv) | Metrics transcribed from the supplied notebooks | Historical notebook outputs, clearly marked as not yet reproduced in Kaggle. |
| [`results/methodology_and_result_notes.md`](results/methodology_and_result_notes.md) | Methods, interpretation and run status | Scope, caveats and project-specific reproducibility notes. |

Notebook copies retain executable code but omit embedded old cell outputs to keep them compact and re-runnable. The supplied source notebooks remain unchanged on the author's computer. Each classification notebook now discovers the attached `bias_clean.csv` under `/kaggle/input` and writes metrics/confusion reports to `/kaggle/working/results`. Model save code is present in all four notebooks.

## Dataset

Attach the public [`bias_clean` dataset on Kaggle](https://www.kaggle.com/datasets/ponssvvv/bias-clean) to each notebook as an input. The notebooks locate `bias_clean.csv` by searching `/kaggle/input`, instead of depending on a fixed third-party mount path. Required columns are `page_text` (article text) and `bias` (class label). No copy of the dataset is included here; use the Kaggle source and follow its MIT license and attribution.

## Labels

The paper describes five classes: Center, Leaning-Left, Left, Leaning-Right, and Right. Label encoding and class balancing are not identical in every supplied notebook; see the methods note before comparing numbers.

## Run in Kaggle

1. Open the dataset above while signed in to Kaggle.
2. For each notebook, use Kaggle **Create → Notebook → File → Import Notebook** and upload the corresponding `.ipynb` file from this folder (or import from the repository after it is public).
3. In the notebook **Input** panel, attach `ponssvvv/bias-clean`.
4. Select a GPU accelerator for fine-tuning, then run the notebook from top to bottom. BART + LoRA and SHAP cells can take substantially longer than the classifier metrics cells.
5. Use **Save Version** after a completed run. Kaggle stores the run outputs and model files under the notebook version's output artifacts. The metrics exporter writes compact JSON and CSV files beneath `/kaggle/working/results`.

Kaggle execution has not yet been verified for these cleaned copies. The committed CSV contains the already-saved source-notebook metrics, not new Kaggle rerun results.

## Reported notebook results

RoBERTa has the strongest saved classification result (77.73% accuracy; weighted F1 0.7766). However, the notebooks differ in cleaning, balancing, sequence length, splitting and training configurations; these figures are descriptive, not a controlled architecture comparison. DistilBERT contains more than one metric summary. The full details and limitations are in [`results/methodology_and_result_notes.md`](results/methodology_and_result_notes.md).

## Tech stack

Python, PyTorch, Hugging Face Transformers, Datasets, scikit-learn, pandas, matplotlib/seaborn, SHAP, PEFT/LoRA, BART, ALBERT, DistilBERT, RoBERTa, Kaggle Notebooks.

## Future deployment

The planned next phase is an accessible Streamlit dashboard, followed by a LinkedIn project entry. Those deployment and profile steps are not part of this repository package yet.

