"""Explainable rule-based Translation Risk Index; not a correctness probability."""


def assess_risk(evidence, ambiguity_detected=False, language_confidence=None):
    """Return a bounded 0-100 index and rule-based decision with reasons."""
    score = 0
    reasons = []

    def add(points, condition, message):
        nonlocal score
        if condition:
            score += points
            reasons.append(message)

    add(10, ambiguity_detected, "Ambiguity was detected.")
    if language_confidence is not None and language_confidence < 0.65:
        add(10, True, "Speech language confidence is low.")
    add(25, evidence.get("meaning_preserved") is False, "Meaning preservation failed or was contradicted.")
    add(20, evidence.get("selected_meaning_preserved") is False, "The selected interpretation was not preserved.")
    add(20, evidence.get("semantic_constraint_satisfied") is False, "A semantic constraint was not satisfied.")
    add(15, evidence.get("entities_preserved") is False, "A named entity may have changed or been lost.")
    add(15, evidence.get("numbers_preserved") is False, "A number was changed or lost.")
    add(10, evidence.get("dates_preserved") is False, "A date was changed or lost.")
    add(10, evidence.get("context_consistent") is False, "Context consistency could not be established.")
    add(10, bool(evidence.get("major_additions")), "Potential major additions were detected.")
    add(10, bool(evidence.get("major_omissions")), "Potential major omissions were detected.")
    add(20, evidence.get("status") == "REVIEW", "Verification requires review.")
    score = min(100, score)
    if evidence.get("status") == "REVIEW":
        score = max(score, 30)

    if score <= 20:
        level, action = "Low", "ACCEPT"
    elif score <= 50:
        level, action = "Moderate", "VERIFY"
    elif score <= 75:
        level, action = "High", "REPAIR"
    else:
        level, action = "Critical", "HUMAN_REVIEW"
    return {
        "score": score, "level": level, "action": action,
        "reasons": reasons or ["No configured risk indicators were triggered."],
        "interpretation": "Design-based Translation Risk Index; not a calibrated probability.",
    }


def evidence_score(evidence):
    """Rank evidence for conservative candidate comparison (higher is better)."""
    points = 0
    weights = {
        "meaning_preserved": 30,
        "selected_meaning_preserved": 15,
        "semantic_constraint_satisfied": 15,
        "numbers_preserved": 10,
        "dates_preserved": 5,
        "entities_preserved": 10,
        "context_consistent": 5,
    }
    for key, weight in weights.items():
        value = evidence.get(key)
        if value is True:
            points += weight
        elif value is False:
            points -= weight
    points -= min(20, 5 * len(evidence.get("major_additions") or []))
    points -= min(20, 5 * len(evidence.get("major_omissions") or []))
    semantic_similarity = evidence.get("semantic_similarity")
    if isinstance(semantic_similarity, (int, float)):
        points += round((semantic_similarity - 0.5) * 10)
    back_similarity = evidence.get("back_translation_similarity")
    if isinstance(back_similarity, (int, float)):
        # Low weight: paraphrases can be valid despite low lexical overlap.
        points += round((back_similarity - 0.5) * 4)
    risk = evidence.get("risk_index")
    if isinstance(risk, dict) and isinstance(risk.get("score"), (int, float)):
        points -= round(risk["score"] / 10)
    points += 10 if evidence.get("status") == "PASS" else -10
    return points
