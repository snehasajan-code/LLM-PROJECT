"""Conservative repair: every candidate is independently verified before comparison."""

import json

OLLAMA_URL = "http://localhost:11434/api/generate"
REPAIR_MODEL = "qwen2.5:3b"


def _call_ollama(prompt):
    import requests
    response = requests.post(OLLAMA_URL, json={
        "model": REPAIR_MODEL, "prompt": prompt, "stream": False,
        "format": "json", "options": {"temperature": 0},
    }, timeout=90)
    response.raise_for_status()
    raw = response.json().get("response", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}


def generate_repair_plan(original_text, translated_text, back_translation,
                         verification_reason, source_language, target_language,
                         selected_meaning=None, semantic_constraint=None, context=""):
    prompt = f"""Analyze the evidence and suggest one minimally clarified source sentence. Do not translate. Keep it in {source_language}; preserve all facts, numbers, names, and relations. Preserve this user-selected meaning and constraint. Use the supplied context only to resolve terminology, never to add facts.
CONTEXT: {context or 'General'}
SOURCE: {original_text}
CURRENT TARGET: {translated_text}
BACK-TRANSLATION: {back_translation}
VERIFICATION: {verification_reason}
SELECTED MEANING: {selected_meaning or 'None'}
CONSTRAINT: {semantic_constraint or 'None'}
Return JSON: {{"repair_needed": true, "problem": "", "repair_instruction": "", "corrected_source_sentence": "", "reason": ""}}"""
    result = _call_ollama(prompt)
    return {
        "repair_needed": bool(result.get("repair_needed", False)),
        "problem": result.get("problem", ""),
        "repair_instruction": result.get("repair_instruction", ""),
        "corrected_source_sentence": result.get("corrected_source_sentence", ""),
        "reason": result.get("reason", ""),
    }


def _evidence_score(evidence):
    from risk_engine import evidence_score
    return evidence_score(evidence)


def _safe_improvement(original, candidate):
    """Require strictly better evidence and no candidate hard-preservation failures."""
    hard_keys = ("numbers_preserved", "dates_preserved", "entities_preserved",
                 "selected_meaning_preserved", "semantic_constraint_satisfied")
    for key in hard_keys:
        if candidate.get(key) is False and original.get(key) is not False:
            return False
    original_additions = set(original.get("major_additions") or [])
    original_omissions = set(original.get("major_omissions") or [])
    if set(candidate.get("major_additions") or []) - original_additions:
        return False
    if set(candidate.get("major_omissions") or []) - original_omissions:
        return False
    old_score, new_score = _evidence_score(original), _evidence_score(candidate)
    old_risk = (original.get("risk_index") or {}).get("score", 100)
    new_risk = (candidate.get("risk_index") or {}).get("score", 100)
    return new_score > old_score and new_risk <= old_risk


def repair_translation(original_text, translated_text, verification, source_language,
                       target_language, context, tone, selected_meaning=None,
                       semantic_constraint=None, candidate_count=3,
                       verifier=None, candidate_translator=None):
    """Try candidates but never mark one accepted without improved verification evidence."""
    verification = verification or {}
    base = {
        "attempted": False, "repaired": False, "accepted": False,
        "initial_translation": translated_text, "repaired_translation": "",
        "candidate_translations": [], "candidate_verifications": [],
        "original_evidence_score": _evidence_score(verification),
        "reason": "Repair was not accepted.",
    }
    try:
        if verifier is None:
            from verifier import verify_translation as verifier
        if candidate_translator is None:
            from translator import translate_candidates
            candidate_translator = translate_candidates
        if not verification or "status" not in verification:
            verification = verifier(original_text, translated_text, source_language,
                                    target_language, clarification=selected_meaning,
                                    semantic_constraint=semantic_constraint, context=context)
        plan = generate_repair_plan(
            original_text, translated_text, verification.get("back_translation", ""),
            verification.get("reason", ""), source_language, target_language,
            selected_meaning, semantic_constraint, context,
        )
        base.update({
            "attempted": True, "problem": plan.get("problem", ""),
            "repair_instruction": plan.get("repair_instruction", ""),
            "corrected_source_sentence": plan.get("corrected_source_sentence", ""),
            "repair_reason": plan.get("reason", ""),
        })
        corrected = (plan.get("corrected_source_sentence") or "").strip()
        if not plan.get("repair_needed") or not corrected:
            base["reason"] = "No safe repair candidate could be generated; original retained for review."
            return base

        candidates = candidate_translator(corrected, source_language, target_language, candidate_count)
        base["candidate_translations"] = candidates
        scored = []
        for candidate_text in candidates:
            evidence = verifier(
                original_text, candidate_text, source_language, target_language,
                clarification=selected_meaning, semantic_constraint=semantic_constraint,
                context=context,
            )
            scored.append({"translation": candidate_text, "verification": evidence,
                           "evidence_score": _evidence_score(evidence)})
        base["candidate_verifications"] = scored
        base["original_evidence_score"] = _evidence_score(verification)
        improved = [item for item in scored if _safe_improvement(verification, item["verification"])]
        eligible = [item for item in improved if item["verification"].get("status") == "PASS"]
        if not eligible:
            if improved:
                best = max(improved, key=lambda item: item["evidence_score"])
                base["recommended_candidate"] = best["translation"]
                base["recommended_verification"] = best["verification"]
                base["reason"] = (
                    "A candidate improved the evidence but verification still requires review. "
                    "The original translation was retained; HUMAN REVIEW required."
                )
            else:
                base["reason"] = "No repair candidate improved verification evidence without adding a new preservation failure. Original retained; HUMAN REVIEW required."
            base["human_review"] = True
            return base
        accepted = max(eligible, key=lambda item: item["evidence_score"])
        base.update({
            "repaired": True, "accepted": True,
            "repaired_translation": accepted["translation"],
            "repaired_verification": accepted["verification"],
            "repaired_evidence_score": accepted["evidence_score"],
            "reason": "Repair accepted because independently verified evidence improved.",
            "human_review": accepted["verification"].get("status") != "PASS",
        })
        return base
    except Exception as exc:
        base.update({"attempted": True, "human_review": True,
                     "reason": f"Repair failed safely; original retained for HUMAN REVIEW: {exc}"})
        return base
