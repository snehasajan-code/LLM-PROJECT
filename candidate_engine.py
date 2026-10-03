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
    from translator import interpret_idiom
    normalizer = None
    if translator is None:
        from translator import translate_candidates, normalize_clarification
        translator = translate_candidates
        normalizer = normalize_clarification
    idiom = interpret_idiom(text, source_language)
    clarification_text = clarification.strip() if clarification and clarification.strip() else ""

    def comparable(value):
        return " ".join((value or "").casefold().strip(" .!?\t\r\n").split())

    selected_idiom_meaning = bool(
        idiom and clarification_text
        and comparable(clarification_text) == comparable(idiom["meaning"])
    )

    if selected_idiom_meaning:
        # A recognized human choice is authoritative: never ask a model to
        # rewrite it, since that can restore the idiom's literal wording.
        normalized_source = idiom["normalized_text"]
    elif clarification_text:
        # Preserve the existing injected-translator contract, but normalize
        # human clarifications when running the production translator.
        normalized_source = normalizer(
            text, clarification_text, semantic_constraint, source_language, context
        ) if normalizer else clarification_text
    else:
        normalized_source = idiom["normalized_text"] if idiom else text
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
