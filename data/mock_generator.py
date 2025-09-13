# data/mock_generator.py
import pandas as pd
import json
import random
import os

SAMPLE_COUNTRIES = [f"Country_{i}" for i in range(1, 51)]
SAMPLE_CITIES = [f"City_{i}" for i in range(1, 51)]

def generate_sample_csv(path="assets/sample_dataset.csv", n_agents=5, n_tasks_per_agent=30, seed=42):
    random.seed(seed)
    rows = []
    for a in range(1, n_agents+1):
        agent_name = f"Agent_{a}"
        for t in range(1, n_tasks_per_agent+1):
            task_id = f"QA_{agent_name}_{t}"
            country_idx = random.randint(1, 50)
            prompt = f"Q{t}: What is the capital of Country_{country_idx}? Answer concisely."
            # create a variety of responses (some correct, some hallucinated, some uncertain)
            roll = random.random()
            if roll < 0.6:
                # "correct" structured but use City_i (we treat this as possibly correct)
                response = f"The capital of Country_{country_idx} is City_{country_idx}."
            elif roll < 0.75:
                response = "I'm not sure, but I think it's Springfield."
            elif roll < 0.9:
                # overconfident hallucination
                response = f"The capital of Country_{country_idx} is CapitalCity_{random.randint(1,100)}."
            else:
                response = "N/A"
            metadata = {"max_words": 20, "output_format": "text"}
            rows.append({
                "task_id": task_id,
                "task_type": "qa",
                "prompt": prompt,
                "agent": agent_name,
                "response": response,
                "metadata": json.dumps(metadata)
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
