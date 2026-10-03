"""Generate and rank NLLB candidates using independent verification evidence."""

from time import monotonic_ns

from risk_engine import assess_risk, evidence_score
from translation_request import TranslationRequest, call_with_supported_kwargs


def select_best_candidate(candidates, verifier, *, ambiguity_detected=False,
                          language_confidence=None, request=None):
    """Verify each candidate and select the highest evidence score conservatively."""
    evaluated = []
    for text in candidates:
        if not text:
            continue
        evidence = verifier(text)
        risk = assess_risk(
            evidence, ambiguity_detected, language_confidence, request=request
        )
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
    if request is not None:
        request.candidates = evaluated
        request.translation = best["translation"]
        request.verification_result = best["verification"]
        request.risk_result = best["risk"]
    return {"translation": best["translation"], "candidates": evaluated, "selected": best}


def translate_and_rank(text, source_language, target_language, verifier, *,
                       clarification=None, semantic_constraint=None, context="",
                       tone="Natural", candidate_count=3, translator=None,
                       request=None, experiment_metrics=None):
    """Generate NLLB hypotheses and rank them against the same source constraints."""
    if request is None:
        request = TranslationRequest(
            source_text=text, source_language=source_language,
            target_language=target_language, context=context, tone=tone,
        )
        request.select_meaning(clarification, semantic_constraint)
    else:
        text = request.source_text or text
        source_language = request.source_language or source_language
        target_language = request.target_language or target_language
        context = request.context or context
        tone = request.tone or tone
        clarification = clarification or request.selected_meaning
        semantic_constraint = semantic_constraint or request.semantic_constraint
        request.select_meaning(clarification, semantic_constraint)

    normalizer = None
    if translator is None:
        from translator import translate_candidates, normalize_clarification
        translator = translate_candidates
        normalizer = normalize_clarification
    idiom = request.ensure_idiom()
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
    if experiment_metrics is not None:
        experiment_metrics.update({
            "requested_candidates": candidate_count,
            "actual_candidates": 0,
            "verified_candidates": 0,
            "generation_ms": None,
            "verification_ms": 0.0,
            "total_ms": None,
            "pass_count": 0,
            "review_count": 0,
            "selected_index": None,
            "selected_evidence_score": None,
            "selected_risk_index": None,
            "repair_triggered": None,
        })
        generation_started = monotonic_ns()
        candidates = list(translator(
            normalized_source, source_language, target_language, candidate_count
        ) or [])
        experiment_metrics["generation_ms"] = (monotonic_ns() - generation_started) / 1_000_000
        experiment_metrics["actual_candidates"] = len(candidates)
    else:
        candidates = translator(normalized_source, source_language, target_language, candidate_count)

    def verify_candidate(candidate):
        started = monotonic_ns() if experiment_metrics is not None else None
        try:
            evidence = call_with_supported_kwargs(
                verifier,
                text, candidate, source_language, target_language,
                clarification=clarification,
                semantic_constraint=semantic_constraint,
                context=context,
                tone=tone,
                language_confidence=request.voice_confidence,
                request=request,
            )
        finally:
            if experiment_metrics is not None:
                experiment_metrics["verification_ms"] += (
                    monotonic_ns() - started
                ) / 1_000_000
        if experiment_metrics is not None and isinstance(evidence, dict):
            status = evidence.get("status")
            if status == "PASS":
                experiment_metrics["pass_count"] += 1
            elif status == "REVIEW":
                experiment_metrics["review_count"] += 1
        return evidence

    ranked = select_best_candidate(
        candidates,
        verify_candidate,
        ambiguity_detected=request.ambiguity_detected is True or bool(clarification),
        language_confidence=request.voice_confidence,
        request=request,
    )
    if experiment_metrics is not None:
        evaluated = ranked["candidates"]
        selected = ranked.get("selected")
        experiment_metrics["verified_candidates"] = len(evaluated)
        if selected is not None:
            experiment_metrics["selected_index"] = evaluated.index(selected)
            experiment_metrics["selected_evidence_score"] = selected["evidence_score"]
            experiment_metrics["selected_risk_index"] = (
                selected.get("risk") or {}
            ).get("score")
    request.translation = ranked["translation"]
    request.candidates = ranked["candidates"]
    return ranked
