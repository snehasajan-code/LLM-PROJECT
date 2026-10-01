"""Generate and rank NLLB candidates using independent verification evidence."""

from risk_engine import assess_risk, evidence_score


def select_best_candidate(candidates, verifier, *, ambiguity_detected=False, language_confidence=None):
    """Verify each candidate and select the highest evidence score conservatively."""
    evaluated = []
    for text in candidates:
        if not text:
            continue
        evidence = verifier(text)
        risk = assess_risk(evidence, ambiguity_detected, language_confidence)
        evaluated.append({
            "translation": text,
            "verification": evidence,
            "risk": risk,
            "evidence_score": evidence_score(evidence),
        })
    if not evaluated:
        return {"translation": "", "candidates": [], "selected": None}
    # Stable ordering preserves the engine's first candidate on evidence ties.
    best = max(evaluated, key=lambda item: item["evidence_score"])
    return {"translation": best["translation"], "candidates": evaluated, "selected": best}


def translate_and_rank(text, source_language, target_language, verifier, *,
                       clarification=None, semantic_constraint=None, context="",
                       tone="Natural", candidate_count=3, translator=None):
    """Generate NLLB hypotheses and rank them against the same source constraints."""
    if translator is None:
        from translator import translate_candidates, normalize_clarification
        translator = translate_candidates
        normalized_source = normalize_clarification(
            text, clarification, semantic_constraint, source_language, context
        ) if clarification and clarification.strip() else text
    else:
        # Preserve testability and backward compatibility for injected translators.
        normalized_source = clarification.strip() if clarification and clarification.strip() else text
    candidates = translator(normalized_source, source_language, target_language, candidate_count)
    return select_best_candidate(
        candidates,
        lambda candidate: verifier(
            text, candidate, source_language, target_language,
            clarification=clarification, semantic_constraint=semantic_constraint,
            context=context,
        ),
        ambiguity_detected=bool(clarification),
    )
