"""Transparent evidence summary; does not fabricate calibrated quality scores."""

import re


def normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[.!?,;:]+$", "", (text or "").lower().strip()))


def back_translation_matches(original_text, back_translation):
    """Legacy helper retained for callers; exact normalized match only."""
    return bool(normalize_text(original_text)) and normalize_text(original_text) == normalize_text(back_translation)


def extract_numbers(text):
    return re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)", text or "")


def numbers_preserved(original_text, back_translation):
    return extract_numbers(original_text) == extract_numbers(back_translation)


def evaluate_translation(original_text, translated_text, source_language, target_language,
                         context="", tone="Natural", clarification=None,
                         semantic_constraint=None, back_translation=None,
                         verification=None, **_legacy_kwargs):
    """Summarize verifier evidence with PASS/REVIEW/N/A labels, never None/5."""
    evidence = verification or {}
    if not evidence and back_translation is not None:
        # Legacy direct-call fallback is explicitly a lexical proxy, not semantic evaluation.
        meaning = back_translation_matches(original_text, back_translation)
        evidence = {
            "meaning_preserved": meaning,
            "numbers_preserved": numbers_preserved(original_text, back_translation),
            "status": "REVIEW" if not meaning else "PASS",
        }

    def label(key, applicable=True):
        if not applicable:
            return "N/A"
        value = evidence.get(key)
        return "PASS" if value is True else "REVIEW" if value is False else "N/A"

    metrics = {
        "meaning": label("meaning_preserved"),
        "context": label("context_consistent"),
        "grammar_naturalness": "N/A",
        "tone": "N/A",
        "ambiguity_resolution": label("selected_meaning_preserved", bool(clarification)),
        "numbers": label("numbers_preserved"),
        "dates": label("dates_preserved"),
        "entities": label("entities_preserved"),
    }
    strengths, issues = [], []
    for key in ("meaning_preserved", "selected_meaning_preserved", "semantic_constraint_satisfied",
                "numbers_preserved", "dates_preserved", "entities_preserved", "context_consistent"):
        if evidence.get(key) is True:
            strengths.append(key.replace("_", " ").capitalize() + " supported by verification evidence.")
        elif evidence.get(key) is False:
            issues.append(key.replace("_", " ").capitalize() + " requires review.")
    issues.extend(evidence.get("major_additions") or [])
    issues.extend(evidence.get("major_omissions") or [])
    for key, message in (("grammar_naturalness", "Grammar/naturalness is not automatically evaluated."),
                         ("tone", "Tone is not automatically evaluated.")):
        if metrics[key] == "N/A":
            issues.append(message)
    return {
        **metrics,
        # Old key retained for UI compatibility, but explicitly not a numeric score.
        "overall": evidence.get("status", "N/A"),
        "assessment_type": "Evidence summary; not a calibrated human quality score",
        "strengths": strengths,
        "issues": issues,
        "explanation": evidence.get("reason", "Evaluation is based on available verifier evidence."),
        "details": {
            "meaning": "Structured semantic assessment, when available.",
            "context": f"Context supplied: {context or 'General'}; consistency is LLM-based and may be N/A.",
            "grammar": "Not measured.", "tone": f"Requested tone: {tone}; not measured.",
            "ambiguity": "Selected interpretation preservation check." if clarification else "No selected interpretation supplied.",
            "numbers": "Deterministic back-translation preservation check.",
        },
    }
