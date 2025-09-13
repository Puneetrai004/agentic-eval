# app.py
import streamlit as st
import pandas as pd
import json
import time
from io import StringIO
import os
from eval import rules, ml_model
from utils.visuals import spider_net_multi
import matplotlib.pyplot as plt
import yaml
import requests

# Force Streamlit to use local .streamlit directory
os.environ["STREAMLIT_CONFIG_DIR"] = os.path.join(os.getcwd(), ".streamlit")

# CACHING
@st.cache_data
def load_prompt_templates(path="prompt_templates.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_sample_dataset():
    # lazy generate if not exists
    sample_path = "assets/sample_dataset.csv"
    if not os.path.exists(sample_path):
        from data.mock_generator import generate_sample_csv
        generate_sample_csv(path=sample_path, n_agents=10, n_tasks_per_agent=40)
    return pd.read_csv(sample_path)

@st.cache_resource
def train_or_load_ml_model(df):
    # train lightweight model on df using rule-based labels (weak supervision)
    model = ml_model.train_lightweight_model(df, rules.apply_rule_based)
    return model

# --- LLM Judge helper (optional) ---
def call_llm_judge_openai(api_key, prompt_text):
    """
    If user provides OPENAI API KEY, call OpenAI (or you can adapt for HF Inference).
    This is optional and only called after user opts-in.
    For safety we keep this generic: do not call unless key provided.
    """
    # NOTE: we purposely don't include a heavy SDK to avoid additional deps.
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # basic gpt-3.5-turbo wrapper (user must provide key & pay)
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": "You are an unbiased evaluator."},
                     {"role": "user", "content": prompt_text}],
        "max_tokens": 200,
        "temperature": 0.0
    }
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"LLM call failed: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"LLM exception: {e}"

def llm_judge_for_row(templates, row, api_key):
    """
    Build prompt from template and call LLM judge (if key provided).
    Returns parsed dict or None on parse error.
    """
    prompt = templates["llm_judge_prompt"]
    prompt_text = prompt + "\n\nPROMPT:\n" + row.get("prompt", "") + "\n\nRESPONSE:\n" + row.get("response", "")
    raw = call_llm_judge_openai(api_key, prompt_text)
    # attempt parse JSON
    try:
        import re
        # sometimes model returns code fences or text; extract json substring
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            j = json.loads(m.group(0))
        else:
            j = json.loads(raw)
        # ensure keys exist and convert numeric
        return {
            "instruction_following": int(round(float(j.get("instruction_following", 0)))),
            "hallucination": int(round(float(j.get("hallucination", 0)))),
            "assumption_control": int(round(float(j.get("assumption_control", 0)))),
            "coherence_accuracy": int(round(float(j.get("coherence_accuracy", 0))))
        }
    except Exception as e:
        st.warning(f"LLM parse failed: {e}. Raw: {raw[:300]}")
        return None

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Agentic Evaluation Framework")
st.title("Agentic Evaluation Framework — Hybrid scoring + optional LLM judge")

templates = load_prompt_templates()

st.sidebar.header("Options")
scoring_method = st.sidebar.radio("Scoring method", ["Rule-based (fast)", "ML-based (learned)", "LLM-as-Judge (optional)"])
use_llm = False
api_key = None
if scoring_method == "LLM-as-Judge (optional)":
    use_llm = st.sidebar.checkbox("Enable LLM Judge (requires API key)")
    api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
    if use_llm and not api_key:
        st.sidebar.warning("Enter OpenAI API key to enable LLM judge.")

st.sidebar.markdown("---")
st.sidebar.markdown("Upload CSV (columns: task_id, task_type, prompt, agent, response, metadata)")
uploaded = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_sample_dataset()
    st.sidebar.info("Using sample dataset (you can upload a CSV).")

st.sidebar.markdown("---")
if st.sidebar.button("Retrain lightweight ML model (on current dataset)"):
    with st.spinner("Training small ML model..."):
        model = train_or_load_ml_model(df)
    st.sidebar.success("Model trained and cached.")

# Main UI: data preview & evaluation
tab1, tab2, tab3 = st.tabs(["Data & Run", "Leaderboard", "Per-Agent Visualization"])

with tab1:
    st.subheader("Dataset preview")
    st.dataframe(df.head(200))
    st.markdown("### Run evaluation")
    run_button = st.button("Run Evaluation (batch)")
    if run_button:
        t0 = time.time()
        st.info(f"Running {scoring_method} scoring on {len(df)} rows...")
        results = []
        if scoring_method == "Rule-based (fast)":
            # apply rules vectorized
            applied = df.apply(lambda r: rules.apply_rule_based(r), axis=1)
            # applied is series of dicts
            df_scores = pd.DataFrame(applied.tolist())
        elif scoring_method == "ML-based (learned)":
            # train or load small model
            with st.spinner("Training / loading lightweight model (weakly supervised via rules)..."):
                model = train_or_load_ml_model(df)
            preds = ml_model.predict_with_model(model, df)
            df_scores = pd.DataFrame(preds)
        else:
            # LLM as judge (optional): if use_llm==False fallback to rule-based to avoid calls
            if not use_llm or not api_key:
                st.warning("LLM Judge not enabled or API key missing. Falling back to Rule-based scoring.")
                applied = df.apply(lambda r: rules.apply_rule_based(r), axis=1)
                df_scores = pd.DataFrame(applied.tolist())
            else:
                # iterate but keep it safe: rate limit, show progress, but don't blow memory.
                out_rows = []
                for idx, row in df.iterrows():
                    st.write(f"Evaluating {idx+1}/{len(df)}: {row.get('task_id')}")
                    j = llm_judge_for_row(templates, row, api_key)
                    if j is None:
                        j = rules.apply_rule_based(row)
                    out_rows.append(j)
                df_scores = pd.DataFrame(out_rows)
        # attach scores to df
        df_eval = pd.concat([df.reset_index(drop=True), df_scores.reset_index(drop=True)], axis=1)
        # create overall score (simple average, weighting can be configurable)
        df_eval["overall_score"] = df_eval[["instruction_following","hallucination","assumption_control","coherence_accuracy"]].mean(axis=1)
        st.success(f"Evaluation completed in {time.time()-t0:.1f}s")
        st.session_state["df_eval"] = df_eval
        st.dataframe(df_eval.head(200))
        csv = df_eval.to_csv(index=False)
        st.download_button("Download evaluated CSV", csv, file_name="evaluated_results.csv", mime="text/csv")

with tab2:
    st.subheader("Leaderboard")
    df_eval = st.session_state.get("df_eval", None)
    if df_eval is None:
        st.info("Run evaluation in the 'Data & Run' tab first.")
    else:
        # group by agent
        agg = df_eval.groupby("agent").agg({
            "overall_score":"mean",
            "instruction_following":"mean",
            "hallucination":"mean",
            "assumption_control":"mean",
            "coherence_accuracy":"mean",
            "task_id":"count"
        }).rename(columns={"task_id":"n_tasks"}).reset_index().sort_values("overall_score", ascending=False)
        st.dataframe(agg)
        st.markdown("Top 10 agents")
        st.dataframe(agg.head(10))

with tab3:
    st.subheader("Per-Agent Visualization (Radar Chart)")
    df_eval = st.session_state.get("df_eval", None)
    if df_eval is None:
        st.info("Run evaluation in the 'Data & Run' tab first.")
    else:
        agents = df_eval["agent"].unique().tolist()
        sel = st.selectbox("Choose agent", options=agents)
        sub = df_eval[df_eval["agent"]==sel]
        # compute mean per dimension
        mean_vals = [
            sub["instruction_following"].mean(),
            5 - sub["hallucination"].mean(),  # invert hallucination so higher is better on radar
            5 - sub["assumption_control"].mean(), # invert assumption_control so higher is better
            sub["coherence_accuracy"].mean()
        ]
        # bounds and labels
        labels = ["Instr-Follow", "No-Hallucination", "Assumption-Control", "Coherence/Acc"]
        rows = [{"name": sel, "values": [float(round(v,2)) for v in mean_vals]}]
        fig, ax = spider_net_multi(labels, rows, title=f"{sel} (mean scores, inverted where needed)")
        st.pyplot(fig)
        # show distribution per task
        st.markdown("Detail: per-task scores head")
        st.dataframe(sub[["task_id","prompt","response","instruction_following","hallucination","assumption_control","coherence_accuracy"]].head(50))

st.markdown("---")
st.markdown("Project: Hybrid Agent Evaluation Framework. See README in repo for deployment steps.")
