import pandas as pd
import numpy as np

def rule_based_scores(row):
    """Simple rule-based scoring using metadata + heuristics"""
    response = row["response"]
    prompt = row["prompt"]
    scores = {
        "instruction_following": 1 if len(response.split()) <= row.get("metadata", {}).get("max_words", 50) else 0.5,
        "hallucination": 0 if "I'm not sure" in response or "#NAME?" in response else 1,
        "assumption_control": 0.5 if "I think" in response else 1,
        "coherence_accuracy": 1 if response and not response.startswith("#") else 0.5
    }
    return scores

def ml_based_scores(row):
    """Dummy ML-based scoring (replace with real classifier if needed)"""
    # For now: random scores to simulate ML model
    rng = np.random.default_rng()
    scores = {
        "instruction_following": rng.uniform(0.6, 1.0),
        "hallucination": rng.uniform(0.4, 1.0),
        "assumption_control": rng.uniform(0.5, 1.0),
        "coherence_accuracy": rng.uniform(0.6, 1.0)
    }
    return scores

def evaluate_responses(df, method="rule"):
    all_scores = []
    for _, row in df.iterrows():
        if method == "rule":
            scores = rule_based_scores(row)
        elif method == "ml":
            scores = ml_based_scores(row)
        else:
            scores = {"instruction_following": 0, "hallucination": 0, "assumption_control": 0, "coherence_accuracy": 0}
        all_scores.append(scores)
    return pd.DataFrame(all_scores)
