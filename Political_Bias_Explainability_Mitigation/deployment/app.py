import gc, hashlib, json, os, tempfile
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
BART_DIR, CLS_DIR, ARTIFACTS = ROOT / "model", ROOT / "classifier_model", ROOT / "artifacts"
WEIGHT_URL = "https://github.com/pauneshvaradharajan/Amrita_Academic_Projects/releases/download/roberta-classifier-v1/model.safetensors"
WEIGHT_SHA256 = "0A6B7C59399A93CF7658A161C856850BB1D90E71549FBBCACE29CF64155C74A7"
LABELS = {0:"Center", 1:"Leaning-left", 2:"Leaning-right", 3:"Left", 4:"Right"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(4, os.cpu_count() or 1))


st.set_page_config(page_title="Political Lens | Bias Lab", page_icon="🧭", layout="wide")
st.markdown("""
<style>
:root{color-scheme:light} html,body,.stApp,[data-testid="stAppViewContainer"]{background:#fff!important;color:#172033!important}
[data-testid="stHeader"]{background:#ffffffed!important}[data-testid="stSidebar"],[data-testid="stSidebar"]>div{background:#f5f7fc!important;color:#172033!important}
.block-container{max-width:1220px;padding-top:1.5rem;padding-bottom:3rem}[data-testid="stTextArea"] textarea{background:#fbfcff!important;color:#172033!important;border:1px solid #d8e0ed!important;border-radius:13px!important}
div.stButton>button,div.stDownloadButton>button{border-radius:10px;font-weight:650}div.stButton>button[kind="primary"]{background:#3157d5;color:white;border:0}
.hero{background:linear-gradient(115deg,#f1f5ff,#fcfdff 65%,#eff9f5);border:1px solid #e5eaf4;border-radius:22px;padding:2rem 2.2rem;margin:.2rem 0 1rem}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.78rem;font-weight:750;color:#5268a5}.hero h1{color:#17233b;font-size:clamp(2.2rem,5vw,3.7rem);line-height:1.05;margin:.6rem 0}.hero p{color:#58657c;font-size:1.05rem;max-width:720px;margin:.5rem 0}
.word-stack{display:inline-grid;position:relative;vertical-align:top;min-width:5.5em;height:1.12em;color:#3157d5}.word-stack span{grid-area:1/1;opacity:0;animation:wordcycle 8s linear infinite}.word-stack .w1{animation-delay:0s}.word-stack .w2{animation-delay:-2s}.word-stack .w3{animation-delay:-4s}.word-stack .w4{animation-delay:-6s}
@keyframes wordcycle{0%,19%{opacity:1;transform:translateY(0)}24%,100%{opacity:0;transform:translateY(-.15em)}}@media(prefers-reduced-motion:reduce){.word-stack span{animation:none;opacity:0}.word-stack .w1{opacity:1}}

/* High contrast for the light dashboard */
.stApp p,.stApp label,.stApp small,.stApp [data-testid="stMarkdownContainer"] *, .stApp [data-testid="stWidgetLabel"] *,.stApp [data-testid="stMetricLabel"] *,.stApp [data-testid="stMetricDelta"] *{color:#173b68!important}
.stApp [data-testid="stMetricValue"]{color:#176b55!important}
.stApp [data-testid="stCaptionContainer"] *{color:#52647a!important;opacity:1!important}
.stApp [role="tab"],.stApp [role="tab"] *,.stApp [data-baseweb="tab"],.stApp [data-baseweb="tab"] *{color:#173b68!important;opacity:1!important}
.stApp [role="tab"][aria-selected="true"],.stApp [role="tab"][aria-selected="true"] *,.stApp [data-baseweb="tab"][aria-selected="true"],.stApp [data-baseweb="tab"][aria-selected="true"] *{color:#174bc6!important}
</style><div class="hero"><div class="eyebrow">Political text · research demo</div><h1>Read with a<br><span class="word-stack"><span class="w1">clearer lens.</span><span class="w2">fairer lens.</span><span class="w3">curious lens.</span><span class="w4">wider lens.</span></span></h1><p>Explore a RoBERTa classifier and a BART rewrite assistant. These research tools are starting points for analysis, not verdicts about truth.</p></div>
""", unsafe_allow_html=True)
st.caption("Two models · five dataset labels · review every output")
with st.sidebar:
    st.markdown("## 🧭 Political Lens")
    st.write("Select a tool to examine political text.")
    st.divider()
    st.caption("Predictions reflect labels in one dataset. They are not objective fact, publisher intent, or misinformation checks.")


@st.cache_resource(show_spinner="Loading BART + LoRA…")
def load_bart():
    cfg=json.loads((BART_DIR/"deployment_config.json").read_text(encoding="utf-8"))
    tok=AutoTokenizer.from_pretrained(str(BART_DIR))
    base=AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"])
    return tok,PeftModel.from_pretrained(base,str(BART_DIR)).to(DEVICE).eval(),cfg


def classifier_dir():
    d=Path(tempfile.gettempdir())/"political_lens_roberta"; d.mkdir(parents=True,exist_ok=True)
    for name in ("config.json","tokenizer.json","tokenizer_config.json"):
        src=CLS_DIR/name
        if not src.exists(): raise FileNotFoundError(f"Missing classifier file: {name}")
        dst=d/name
        if not dst.exists() or dst.stat().st_size!=src.stat().st_size: dst.write_bytes(src.read_bytes())
    w=d/"model.safetensors"; ok=False
    if w.exists():
        h=hashlib.sha256()
        with w.open("rb") as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
        ok=h.hexdigest().upper()==WEIGHT_SHA256
    if not ok:
        if w.exists(): w.unlink()
        tmp=d/"model.download"; h=hashlib.sha256(); r=requests.get(WEIGHT_URL,stream=True,timeout=(20,600)); r.raise_for_status()
        try:
            with tmp.open("wb") as f:
                for chunk in r.iter_content(8*1024*1024):
                    if chunk: f.write(chunk); h.update(chunk)
            if h.hexdigest().upper()!=WEIGHT_SHA256: raise ValueError("Classifier checksum mismatch")
            tmp.replace(w)
        finally:
            r.close()
            if tmp.exists(): tmp.unlink()
    return d


@st.cache_resource(show_spinner="Loading RoBERTa classifier…")
def load_classifier():
    d=classifier_dir(); tok=AutoTokenizer.from_pretrained(str(d),local_files_only=True)
    cfg=AutoConfig.from_pretrained(str(d),local_files_only=True); cfg.id2label=LABELS; cfg.label2id={v:k for k,v in LABELS.items()}
    mdl=AutoModelForSequenceClassification.from_pretrained(str(d),config=cfg,local_files_only=True,use_safetensors=True).to(DEVICE).eval()
    return tok,mdl


def free_other(current):
    (load_bart if current=="classifier" else load_classifier).clear(); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


t_cls,t_rewrite,t_info=st.tabs(["🔎 Classify an article","✍️ Explore a rewrite","📊 Evaluation & notes"])
with t_cls:
    st.subheader("Classify political leaning")
    st.write("Paste a news passage to see its predicted dataset label and score distribution.")
    text=st.text_area("Article text",height=220,placeholder="Paste an article paragraph…",key="classify_text")
    run=st.button("Analyze text",type="primary",key="classify",width="stretch")
    st.caption("RoBERTa · maximum 256 tokens · English")
    if run:
        if not text.strip(): st.warning("Add some article text first.")
        else:
            try:
                free_other("classifier")
                with st.spinner("Preparing RoBERTa and analyzing the passage…"):
                    tok,mdl=load_classifier(); batch=tok(text,return_tensors="pt",truncation=True,max_length=256); batch={k:v.to(DEVICE) for k,v in batch.items()}
                    with torch.inference_mode(): scores=torch.softmax(mdl(**batch).logits[0].float(),dim=-1).cpu().tolist()
                pred=max(range(len(scores)),key=scores.__getitem__); a,b=st.columns([1,1.5],gap="large")
                a.metric("Predicted dataset label",LABELS[pred],f"{scores[pred]*100:.1f}% model score")
                with b:
                    st.markdown("**Score by label**"); st.bar_chart(pd.DataFrame({"Score":scores},index=[LABELS[i] for i in range(len(scores))]),color="#5275df")
                st.info("The score distribution is not the probability that a political label is objectively correct.")
            except Exception as e: st.error(f"Classification could not run: {e}")


with t_rewrite:
    st.subheader("Explore a neutralized rewrite")
    st.write("BART + LoRA suggests a less loaded version while aiming to preserve meaning.")
    a,b=st.columns(2,gap="large")
    with a:
        source=st.text_area("Original passage",height=220,placeholder="Paste a political news passage…",key="rewrite_text")
        beams=st.slider("Rewrite search",1,6,4); limit=st.slider("Maximum output tokens",32,256,160,step=16)
        do=st.button("Generate rewrite",type="primary",key="rewrite",width="stretch")
    with b:
        st.markdown("**Suggested rewrite**")
        if do:
            if not source.strip(): st.warning("Add some article text first.")
            else:
                try:
                    free_other("bart")
                    with st.spinner("Preparing BART and generating a suggestion…"):
                        tok,mdl,cfg=load_bart(); prompt=("Neutralize political bias while preserving semantic meaning and factual consistency.\n\nTEXT:\n"+source.strip()+"\n\nNEUTRAL VERSION:\n")
                        batch=tok(prompt,return_tensors="pt",truncation=True,max_length=int(cfg.get("max_input_length",256))); batch={k:v.to(DEVICE) for k,v in batch.items()}
                        with torch.inference_mode(): ids=mdl.generate(**batch,num_beams=beams,max_new_tokens=limit,no_repeat_ngram_size=3,repetition_penalty=1.15,early_stopping=True)
                        result=tok.decode(ids[0],skip_special_tokens=True).strip()
                    st.text_area("Generated text",value=result,height=220,key="rewrite_output")
                    st.download_button("Download rewrite",result,file_name="rewrite.txt",mime="text/plain",width="stretch")
                except Exception as e: st.error(f"Rewrite could not be generated: {e}")
        else: st.info("Your rewrite will appear here.")
    st.warning("Review each rewrite for factual accuracy, context, and meaning before sharing.")


with t_info:
    st.subheader("BART evaluation snapshot")
    path=ARTIFACTS/"evaluation_metrics.json"
    if path.exists():
        m=json.loads(path.read_text(encoding="utf-8")); cols=st.columns(4)
        for col,label,key in zip(cols,["BLEU","ROUGE-1 F1","ROUGE-2 F1","ROUGE-L F1"],["bleu","rouge1_f1","rouge2_f1","rougeL_f1"]): col.metric(label,f"{m[key]:.3f}")
        st.caption(m["note"]); st.bar_chart(pd.DataFrame({"Score":[m["bleu"],m["rouge1_f1"],m["rouge2_f1"],m["rougeL_f1"]]},index=["BLEU","ROUGE-1","ROUGE-2","ROUGE-L"]),color="#5275df")
        samples=ARTIFACTS/"sample_predictions.csv"
        if samples.exists():
            with st.expander("View saved sample predictions"): st.dataframe(pd.read_csv(samples),width="stretch",hide_index=True)
    st.divider(); st.subheader("How to interpret these tools")
    st.markdown("- RoBERTa predicts a label learned from the dataset: Center, Leaning-left, Left, Leaning-right, or Right.\n- BART generates a rewrite; it does not classify orientation.\n- Neither tool determines truth, intent, or neutrality as objective fact.\n- BLEU/ROUGE measure text overlap, not factual correctness.\n- Research prototype: check sources and context independently.")
