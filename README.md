# Agentic Evaluation Framework (Streamlit) — Hugging Face Spaces ready

This project implements a hybrid evaluation framework for many agents' responses:
- Rule-based heuristics for instruction-following, hallucination, assumption control, coherence/accuracy.
- Lightweight ML model trained on weak labels (rule-based).
- Optional LLM-as-Judge (OpenAI) that can be toggled on the UI and only called if API key provided.
- Streamlit UI, radar charts, leaderboard, CSV upload/download, synthetic dataset generator.

## Files
- `app.py` – main Streamlit app
- `eval/rules.py` – rule-based scorers
- `eval/ml_model.py` – lightweight ML pipeline (TF-IDF + RandomForest)
- `utils/visuals.py` – radar chart helper
- `data/mock_generator.py` – creates `assets/sample_dataset.csv`
- `prompt_templates.yaml` – LLM prompts
- `requirements.txt`, `.gitignore`

## Run locally
1. Clone the repo
2. Create a venv and install dependencies:
