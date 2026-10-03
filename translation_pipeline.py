"""Shared core orchestration for text and voice translation requests."""

from candidate_engine import translate_and_rank
from translation_request import TranslationRequest


def run_translation_request(request, verifier=None, translator=None, candidate_count=1):
    """Run candidate generation, verification, and risk assessment on one request."""
    if not isinstance(request, TranslationRequest):
        raise TypeError("request must be a TranslationRequest")
    if verifier is None:
        from verifier import verify_translation
        verifier = verify_translation

    request.ensure_idiom()
    ranked = translate_and_rank(
        request.source_text,
        request.source_language,
        request.target_language,
        verifier,
        clarification=request.selected_meaning,
        semantic_constraint=request.semantic_constraint,
        context=request.context,
        tone=request.tone,
        candidate_count=candidate_count,
        translator=translator,
        request=request,
    )
    request.translation = ranked.get("translation", "")
    request.candidates = ranked.get("candidates", [])
    selected = ranked.get("selected") or {}
    request.verification_result = selected.get("verification")
    request.risk_result = selected.get("risk")
    return request


def evaluate_voice_repair(request, evaluator, transcript, translation, verification):
    """Summarize an accepted voice repair using the request's semantic context."""
    return evaluator(
        original_text=transcript,
        translated_text=translation,
        source_language=request.source_language,
        target_language=request.target_language,
        context=request.context,
        tone=request.tone,
        verification=verification,
        selected_meaning=request.selected_meaning,
        semantic_constraint=request.semantic_constraint,
    )
