# eval/rules.py
import re
import json

SENTENCE_TRIGGERS_LOW_CONFIDENCE = [
    "i'm not sure", "i am not sure", "i think", "might be", "could be", "possibly", "not certain"
]

def _safe_get_metadata(meta_field):
    try:
        if isinstance(meta_field, str):
            return json.loads(meta_field)
        return meta_field or {}
    except Exception:
        return {}

def instruction_following_score(prompt, response, metadata):
    """
    Heuristic:
    - If metadata.max_words exists, penalize repeated text or too long outputs.
    - If response contains explicit admission of not following instructions -> low score.
    Returns 0-5 (5 best).
    """
    meta = _safe_get_metadata(metadata)
    max_words = meta.get("max_words", None)
    resp_text = (response or "").strip()
    # penalty for repetitive or extremely long outputs
    words = resp_text.split()
    score = 5
    if any(trig in resp_text.lower() for trig in SENTENCE_TRIGGERS_LOW_CONFIDENCE):
        score -= 2
    if max_words:
        try:
            if len(words) > int(max_words) * 1.5:
                score -= 2
        except Exception:
            pass
    # nonsense markers
    if not resp_text or resp_text in ["#NAME?", "N/A"]:
        score = 0
    return max(0, min(5, score))

def hallucination_score(prompt, response, metadata):
    """
    Heuristic:
    - If response contains many named entities not present in prompt and includes overconfident claims -> higher hallucination risk.
    - If response admits uncertainty -> low hallucination score.
    Return 0-5 where 5 = high hallucination risk.
    """
    resp = (response or "").strip()
    if not resp:
        return 5
    low_conf = any(trig in resp.lower() for trig in SENTENCE_TRIGGERS_LOW_CONFIDENCE)
    if low_conf:
        # If model says "not sure", risk is lower
        return 1
    # naive check: repeated proper nouns (words starting with capital letter and not at sentence start)
    tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", resp)
    # weight by number of unique tokens
    uniq = set(tokens)
    if len(uniq) >= 4:
        return 4
    if len(uniq) >= 2:
        return 3
    return 1

def assumption_control_score(prompt, response, metadata):
    """
    Heuristic measuring how many explicit unwarranted assumptions are present.
    - Penalize when response invents details that are not in prompt (detects 'because', dates, numbers).
    Returns 0-5 where 5 = many assumptions.
    """
    resp = (response or "").lower()
    prompt_l = (prompt or "").lower()
    score = 0
    # if response asserts facts beyond prompt: e.g. naming a person, place not referenced in prompt -> +2
    # crude heuristic: presence of dates, numbers or claims "because", "therefore"
    if re.search(r"\b\d{4}\b", resp) or re.search(r"\b\d{2,}\b", resp):
        score += 2
    if "because" in resp or "therefore" in resp or "hence" in resp:
        score += 1
    # if contains proper nouns not in prompt:
    tokens = set(re.findall(r"\b[A-Z][a-z]{2,}\b", response or ""))
    if tokens:
        # if prompt does not include those tokens:
        for t in tokens:
            if t.lower() not in prompt_l:
                score += 1
                if score >= 5:
                    break
    score = min(5, score)
    # convert to "assumption control" where 0 is good, 5 is poor -> keep as-is
    return score

def coherence_accuracy_score(prompt, response, metadata):
    """
    Heuristics:
    - Short, consistent responses score higher.
    - Repetition => lower.
    - If response contains contradictory phrases -> lower.
    Returns 0-5 (5 best).
    """
    resp = (response or "")
    if not resp.strip():
        return 0
    score = 5
    # repetition detection
    if len(resp.split()) > 40 and len(set(resp.split())) < len(resp.split())*0.5:
        score -= 2
    # contradiction words
    if any(k in resp.lower() for k in ["on the other hand", "contradict", "however", "but"]):
        score -= 1
    # nonsense markers
    if resp.strip() in ["#NAME?", "N/A"]:
        score = 0
    return max(0, min(5, score))

def apply_rule_based(row):
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    metadata = row.get("metadata", "{}")
    return {
        "instruction_following": instruction_following_score(prompt, response, metadata),
        "hallucination": hallucination_score(prompt, response, metadata),
        "assumption_control": assumption_control_score(prompt, response, metadata),
        "coherence_accuracy": coherence_accuracy_score(prompt, response, metadata)
    }
