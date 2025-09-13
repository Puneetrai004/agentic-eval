# eval/ml_model.py
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# We'll create a tiny ML pipeline that learns from rule-based synthetic labels.
# This is intentionally lightweight to avoid memory issues.

def _parse_metadata_field(m):
    try:
        if isinstance(m, str):
            return json.loads(m)
        return m or {}
    except Exception:
        return {}

def featurize_dataframe(df):
    """
    Create features from (prompt,response,metadata)
    Returns X (text features) and y (4-dim continuous labels 0-5)
    """
    texts = (df["prompt"].fillna("") + " ||| " + df["response"].fillna("")).astype(str).tolist()
    # Use TF-IDF on combined text
    return texts

def train_lightweight_model(df, rule_label_fn, random_state=42):
    """
    Trains a small model that predicts 4 numeric scores (0-5).
    We first create labels using the rule-based function as "weak supervision",
    then train a MultiOutputClassifier (RandomForest) on TF-IDF features.
    """
    # create training labels from rules
    labels = []
    for _, row in df.iterrows():
        lbls = rule_label_fn(row)
        labels.append([lbls["instruction_following"], lbls["hallucination"],
                       lbls["assumption_control"], lbls["coherence_accuracy"]])
    y = np.array(labels)
    texts = featurize_dataframe(df)
    # pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1,2))),
        ("clf", MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=random_state)))
    ])
    pipe.fit(texts, y)
    return pipe

def predict_with_model(model_pipe, df):
    texts = featurize_dataframe(df)
    preds = model_pipe.predict(texts)
    # convert to dicts with float/int
    out = []
    for p in preds:
        out.append({
            "instruction_following": int(max(0, min(5, round(float(p[0]))))),
            "hallucination": int(max(0, min(5, round(float(p[1]))))),
            "assumption_control": int(max(0, min(5, round(float(p[2]))))),
            "coherence_accuracy": int(max(0, min(5, round(float(p[3])))))
        })
    return out
