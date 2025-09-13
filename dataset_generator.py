import pandas as pd
import random

def generate_synthetic_dataset(n=50, agents=3):
    rows = []
    for agent in range(1, agents+1):
        for i in range(n):
            task_type = random.choice(["qa", "summarization"])
            if task_type == "qa":
                prompt = f"Q{i}: What is the capital of Country_{i}? Answer concisely."
                response = f"The capital of Country_{i} is City_{i}." if random.random()>0.2 else "I'm not sure."
            else:
                prompt = f"S{i}: Summarize this: 'Artificial intelligence is transforming industries...'"
                response = "AI is transforming industries by improving efficiency." if random.random()>0.3 else "#NAME?"

            rows.append({
                "task_id": f"{task_type.upper()}_Agent{agent}_{i}",
                "task_type": task_type,
                "prompt": prompt,
                "agent": f"Agent_{agent}",
                "response": response,
                "metadata": {"max_words": 20, "output_format": "text"}
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_synthetic_dataset()
    df.to_csv("sample_data.csv", index=False)
    print("✅ Synthetic dataset saved to sample_data.csv")
