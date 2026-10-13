import json
from pathlib import Path
import pandas as pd
import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
ARTIFACTS = ROOT / "artifacts"
st.set_page_config(page_title="Bias Mitigation Studio", page_icon="📰", layout="wide")
st.title("📰 Bias Mitigation Studio")
st.caption("Review a BART + LoRA generated rewrite side by side with your original text.")

@st.cache_resource(show_spinner="Loading BART and the fine-tuned adapter…")
def load_model():
    with (MODEL_DIR / "deployment_config.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"])
    mdl = PeftModel.from_pretrained(base, str(MODEL_DIR)).to(device).eval()
    return tok, mdl, device, cfg

def rewrite(text, tok, mdl, device, cfg, beams, max_tokens):
    prompt = ("Neutralize political bias while preserving semantic meaning and factual consistency.\n\n"
              f"TEXT:\n{text.strip()}\n\nNEUTRAL VERSION:\n")
    batch = tok(prompt, return_tensors="pt", truncation=True,
                max_length=int(cfg.get("max_input_length", 256)))
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        ids = mdl.generate(**batch, num_beams=beams, max_new_tokens=max_tokens,
                           no_repeat_ngram_size=3, repetition_penalty=1.15, early_stopping=True)
    return tok.decode(ids[0], skip_special_tokens=True).strip()

with st.sidebar:
    st.header("Generation settings")
    beams = st.slider("Beam search", 1, 6, 4)
    max_tokens = st.slider("Maximum output tokens", 32, 256, 160, step=16)
    st.caption("Long input is truncated to the model's 256-token limit.")

try:
    tokenizer, model, device, config = load_model()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.info("Keep model/ and artifacts/ beside app.py. Start with: streamlit run app.py")
    st.stop()

rewrite_tab, evaluation_tab, about_tab = st.tabs(["✍️ Mitigate text", "📊 Evaluation", "ℹ️ About"])
with rewrite_tab:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("1 · Original text")
        text = st.text_area("Paste an article or paragraph", height=300, key="input_text")
        st.caption(f"{len(text):,} characters")
        run = st.button("Generate rewrite", type="primary", use_container_width=True)
    with right:
        st.subheader("2 · Rewritten text")
        if run and not text.strip():
            st.warning("Enter some text first.")
        elif run:
            with st.spinner("Generating rewrite…"):
                result = rewrite(text, tokenizer, model, device, config, beams, max_tokens)
            st.text_area("Generated rewrite", value=result, height=300, key="result_text")
            st.download_button("Download rewrite", result, file_name="rewritten_text.txt", mime="text/plain")
        else:
            st.info("Your result will appear here after generation.")
    st.warning("Experimental output: check factual claims, context, and meaning before using or sharing.")

with evaluation_tab:
    path = ARTIFACTS / "evaluation_metrics.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            m = json.load(f)
        st.subheader("Fixed test-sample results")
        a, b, c, d = st.columns(4)
        a.metric("BLEU", f"{m['bleu']:.3f}"); b.metric("ROUGE-1 F1", f"{m['rouge1_f1']:.3f}")
        c.metric("ROUGE-2 F1", f"{m['rouge2_f1']:.3f}"); d.metric("ROUGE-L F1", f"{m['rougeL_f1']:.3f}")
        st.caption(m["note"])
        st.bar_chart(pd.DataFrame({"Metric": ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
                                   "Score": [m["bleu"], m["rouge1_f1"], m["rouge2_f1"], m["rougeL_f1"]]}).set_index("Metric"))
        csv = ARTIFACTS / "sample_predictions.csv"
        if csv.exists():
            st.subheader("Saved sample outputs")
            st.dataframe(pd.read_csv(csv), use_container_width=True, hide_index=True)
    else:
        st.info("Evaluation artifacts were not found in the bundle.")

with about_tab:
    st.subheader("Project")
    st.write("BART-base fine-tuned with a LoRA adapter for heuristic political-bias mitigation.")
    st.write("This app generates rewrites; it does not classify political orientation.")
    st.write("The adapter is bundled. The public facebook/bart-base model downloads on first launch.")
