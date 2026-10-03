import unittest
from unittest.mock import patch

import ambiguity
import repair_engine
import translator
import verifier
from risk_engine import assess_risk
from candidate_engine import translate_and_rank
from translation_pipeline import run_translation_request, evaluate_voice_repair
from translation_request import TranslationRequest


def passing_evidence(**overrides):
    evidence = {
        "status": "PASS",
        "meaning_preserved": True,
        "selected_meaning_preserved": True,
        "semantic_constraint_satisfied": True,
        "numbers_preserved": True,
        "dates_preserved": True,
        "entities_preserved": True,
        "context_consistent": True,
        "major_additions": [],
        "major_omissions": [],
        "risk_index": {"score": 10},
        "back_translation": "preserved meaning",
    }
    evidence.update(overrides)
    return evidence


class TranslationRequestFlowTests(unittest.TestCase):
    def test_normal_text_request_reaches_candidate_verification(self):
        request = TranslationRequest(
            "Please send the report.", "English", "Malayalam",
            context="Business", tone="Formal",
        )
        request.record_ambiguity({"available": True, "ambiguous": False})
        seen = {}

        def fake_verifier(*args, **kwargs):
            seen.update(kwargs)
            return passing_evidence()

        run_translation_request(
            request, verifier=fake_verifier,
            translator=lambda text, *_args: ["translated"],
        )

        self.assertEqual(request.ambiguity_status, "false")
        self.assertEqual(seen["context"], "Business")
        self.assertEqual(seen["tone"], "Formal")
        self.assertIs(seen["request"], request)
        self.assertEqual(request.translation, "translated")

    def test_ambiguous_text_retains_both_duck_options(self):
        result = ambiguity.detect_ambiguity(
            "I saw her duck.", "English", "Malayalam", "General"
        )
        request = TranslationRequest("I saw her duck.", "English", "Malayalam")
        request.record_ambiguity(result)

        self.assertEqual(request.ambiguity_status, "true")
        self.assertEqual(len(request.ambiguity_options), 2)
        self.assertIn("bird/animal", request.ambiguity_options[0]["semantic_constraint"])
        self.assertIn("lowering/bowing", request.ambiguity_options[1]["semantic_constraint"])

    def test_bird_meaning_reaches_translation_and_verification(self):
        request = TranslationRequest("I saw her duck.", "English", "Malayalam")
        request.record_ambiguity(ambiguity._known_ambiguity("I saw her duck.", "English"))
        request.select_meaning(
            "I saw the duck that belongs to her.",
            "Duck is the bird/animal that belongs to her.",
        )
        seen = {}

        def fake_verifier(*args, **kwargs):
            seen.update(kwargs)
            return passing_evidence()

        run_translation_request(
            request, verifier=fake_verifier,
            translator=lambda text, *_args: (seen.update(source=text) or ["translated"]),
        )

        self.assertEqual(seen["source"], "I saw the duck that belongs to her.")
        self.assertEqual(seen["request"].selected_meaning, request.selected_meaning)
        self.assertEqual(request.ambiguity_status, "true")

    def test_action_meaning_and_constraint_reach_verification(self):
        request = TranslationRequest("I saw her duck.", "English", "Malayalam")
        request.record_ambiguity(ambiguity._known_ambiguity("I saw her duck.", "English"))
        request.select_meaning(
            "I saw her bow her head.",
            "Duck means lowering her head, not a bird.",
        )
        seen = {}

        def fake_verifier(*args, **kwargs):
            seen.update(kwargs)
            return passing_evidence()

        run_translation_request(
            request, verifier=fake_verifier,
            translator=lambda *_args: ["translated"],
        )

        self.assertEqual(seen["request"].selected_meaning, "I saw her bow her head.")
        self.assertEqual(seen["request"].semantic_constraint,
                         "Duck means lowering her head, not a bird.")

    def test_detector_failure_is_retained_as_unknown(self):
        request = TranslationRequest("A sentence.", "English", "Malayalam")
        request.record_ambiguity({"available": False, "ambiguous": False,
                                  "error": "Ollama offline"})
        risk = assess_risk(passing_evidence(), request=request)

        self.assertEqual(request.ambiguity_status, "unknown")
        self.assertIsNone(request.ambiguity_detected)
        self.assertEqual(risk["ambiguity_status"], "unknown")
        self.assertTrue(any("uncertainty is retained" in reason for reason in risk["reasons"]))

    def test_missing_detector_result_stays_unknown_in_candidate_engine(self):
        request = TranslationRequest("A sentence.", "English", "Malayalam")
        translate_and_rank(
            request.source_text, request.source_language, request.target_language,
            lambda *_args, **_kwargs: passing_evidence(),
            translator=lambda *_args: ["translated"], request=request,
        )

        self.assertEqual(request.ambiguity_status, "unknown")
        self.assertEqual(request.risk_result["ambiguity_status"], "unknown")

    def test_idiom_interpretation_is_shared_with_candidate_and_verifier(self):
        request = TranslationRequest(
            "It's raining cats and dogs.", "English", "Malayalam",
            context="Conversation", tone="Informal",
        )
        request.record_ambiguity({"available": True, "ambiguous": False})
        seen = {}

        def fake_verifier(*args, **kwargs):
            seen["request"] = kwargs["request"]
            return passing_evidence()

        run_translation_request(
            request, verifier=fake_verifier,
            translator=lambda text, *_args: (seen.update(source=text) or ["translated"]),
        )

        self.assertTrue(request.idiom_detected)
        self.assertEqual(request.idiom_interpretation, "It is raining very heavily.")
        self.assertEqual(seen["source"], "It is raining very heavily.")
        self.assertIs(seen["request"], request)

    def test_voice_request_uses_the_same_core_pipeline(self):
        request = TranslationRequest.for_voice(
            "Hello, നമസ്കാരം.", "English", "Malayalam",
            context="Conversation", tone="Friendly", voice_confidence=0.72,
        )
        request.record_ambiguity({"available": True, "ambiguous": False})
        observed = {}

        def fake_verifier(*args, **kwargs):
            observed.update(kwargs)
            return passing_evidence()

        run_translation_request(
            request, verifier=fake_verifier,
            translator=lambda *_args: ["translated"],
        )

        self.assertEqual(request.transcript, "Hello, നമസ്കാരം.")
        self.assertEqual(request.detected_language, "English")
        self.assertEqual(request.voice_confidence, 0.72)
        self.assertIs(observed["request"], request)

    def test_context_and_tone_reach_semantic_verification(self):
        request = TranslationRequest(
            "A source.", "English", "Malayalam",
            context="Medical", tone="Professional",
        )
        request.record_ambiguity({"available": True, "ambiguous": False})
        with patch.object(verifier, "nllb_translate", return_value="A back translation."):
            with patch.object(verifier, "semantic_check", return_value=passing_evidence()) as check:
                result = verifier.verify_translation(
                    request.source_text, "target", "English", "Malayalam",
                    request=request,
                )

        self.assertEqual(check.call_args.args[5], "Medical")
        self.assertEqual(check.call_args.kwargs["tone"], "Professional")
        self.assertIs(request.verification_result, result)

    def test_idiom_request_is_the_verification_reference(self):
        request = TranslationRequest(
            "It's raining cats and dogs.", "English", "Malayalam",
            context="Conversation", tone="Friendly",
        )
        request.record_ambiguity({"available": True, "ambiguous": False})
        request.ensure_idiom()
        with patch.object(verifier, "nllb_translate", return_value="It is raining very heavily."):
            with patch.object(verifier, "semantic_check", return_value=passing_evidence()) as check:
                result = verifier.verify_translation(
                    request.source_text, "target", "English", "Malayalam",
                    request=request,
                )

        self.assertEqual(check.call_args.kwargs["resolved_reference"],
                         "It is raining very heavily.")
        self.assertEqual(result["idiom"]["meaning"], request.idiom_interpretation)

    def test_voice_selected_meaning_and_constraint_reach_repair_plan(self):
        request = TranslationRequest.for_voice(
            "I saw her duck.", "English", "Malayalam", context="Conversation",
            tone="Friendly", voice_confidence=0.81,
        )
        request.record_ambiguity(ambiguity._known_ambiguity("I saw her duck.", "English"))
        request.select_meaning("I saw her bow her head.", "Duck means lowering her head.")
        request.translation = "original"
        request.verification_result = passing_evidence(
            status="REVIEW", risk_index={"score": 70}, back_translation="uncertain",
        )
        captured = {}

        def plan(*args):
            captured["args"] = args
            return {"repair_needed": True, "corrected_source_sentence": "clarified source"}

        def candidate_verifier(*args, **kwargs):
            captured["request"] = kwargs["request"]
            return passing_evidence(risk_index={"score": 10})

        with patch.object(repair_engine, "generate_repair_plan", side_effect=plan):
            result = repair_engine.repair_translation(
                "I saw her duck.", "original", request.verification_result,
                "English", "Malayalam", "General", "Neutral",
                verifier=candidate_verifier,
                candidate_translator=lambda *_args: ["repaired"], request=request,
            )

        self.assertEqual(captured["args"][6], "I saw her bow her head.")
        self.assertEqual(captured["args"][7], "Duck means lowering her head.")
        self.assertEqual(captured["args"][8], "Conversation")
        self.assertEqual(captured["args"][9], "Friendly")
        self.assertIs(captured["request"], request)
        self.assertTrue(result["accepted"])

    def test_voice_repair_verification_keeps_constraint_context_and_tone(self):
        request = TranslationRequest.for_voice(
            "I saw her duck.", "English", "Malayalam", context="Travel",
            tone="Polite", voice_confidence=0.61,
        )
        request.select_meaning("I saw her bow her head.", "Duck means lowering her head.")
        request.translation = "original"
        request.verification_result = passing_evidence(
            status="REVIEW", risk_index={"score": 70},
        )
        captured = []

        def candidate_verifier(*args, **kwargs):
            captured.append(kwargs)
            return passing_evidence(risk_index={"score": 10})

        with patch.object(repair_engine, "generate_repair_plan", return_value={
            "repair_needed": True, "corrected_source_sentence": "corrected source",
        }):
            repair_engine.repair_translation(
                request.source_text, "original", request.verification_result,
                request.source_language, request.target_language, request.context,
                request.tone, verifier=candidate_verifier,
                candidate_translator=lambda *_args: ["candidate"], request=request,
            )

        self.assertEqual(captured[0]["clarification"], "I saw her bow her head.")
        self.assertEqual(captured[0]["semantic_constraint"], "Duck means lowering her head.")
        self.assertEqual(captured[0]["context"], "Travel")
        self.assertEqual(captured[0]["tone"], "Polite")
        self.assertIs(captured[0]["request"], request)

    def test_voice_post_repair_evaluation_receives_selected_semantic_context(self):
        request = TranslationRequest.for_voice(
            "I saw her duck.", "English", "Malayalam", context="Travel",
            tone="Polite", voice_confidence=0.61,
        )
        request.record_ambiguity(ambiguity._known_ambiguity("I saw her duck.", "English"))
        request.select_meaning(
            "I saw her bow her head.", "Duck means lowering her head, not a bird."
        )
        evaluator = unittest.mock.Mock(return_value={"overall": "PASS"})
        verification = passing_evidence()

        evaluate_voice_repair(
            request, evaluator, request.transcript, "repaired Malayalam", verification
        )

        evaluator.assert_called_once_with(
            original_text=request.transcript,
            translated_text="repaired Malayalam",
            source_language="English",
            target_language="Malayalam",
            context="Travel",
            tone="Polite",
            verification=verification,
            selected_meaning="I saw her bow her head.",
            semantic_constraint="Duck means lowering her head, not a bird.",
        )

    def test_repair_candidates_are_reverified_with_same_idiom_and_constraints(self):
        request = TranslationRequest(
            "It's raining cats and dogs.", "English", "Malayalam",
            context="Conversation", tone="Informal",
        )
        request.record_ambiguity({"available": True, "ambiguous": False})
        request.select_meaning("It is raining very heavily.", "Interpret the idiom by its meaning.")
        request.ensure_idiom()
        request.translation = "original"
        request.verification_result = passing_evidence(
            status="REVIEW", risk_index={"score": 70},
        )
        seen = []

        def candidate_verifier(*args, **kwargs):
            req = kwargs["request"]
            seen.append((req.selected_meaning, req.semantic_constraint,
                         req.idiom_interpretation, req.context, req.tone))
            return passing_evidence(risk_index={"score": 10})

        plan_args = {}

        def plan(*args):
            plan_args["args"] = args
            return {"repair_needed": True, "corrected_source_sentence": "It is raining heavily."}

        with patch.object(repair_engine, "generate_repair_plan", side_effect=plan):
            result = repair_engine.repair_translation(
                request.source_text, "original", request.verification_result,
                request.source_language, request.target_language, request.context,
                request.tone, verifier=candidate_verifier,
                candidate_translator=lambda *_args: ["repaired"], request=request,
            )

        self.assertEqual(seen, [("It is raining very heavily.",
                                 "Interpret the idiom by its meaning.",
                                 "It is raining very heavily.", "Conversation", "Informal")])
        self.assertTrue(result["accepted"])
        self.assertEqual(request.translation, "repaired")
        self.assertEqual(plan_args["args"][10], "It is raining very heavily.")

    def test_unsafe_repair_retains_original_request_result(self):
        request = TranslationRequest("Source.", "English", "Malayalam")
        request.record_ambiguity({"available": True, "ambiguous": False})
        original_evidence = passing_evidence(
            status="REVIEW", risk_index={"score": 60}, selected_meaning_preserved=True,
        )
        request.translation = "original"
        request.verification_result = original_evidence

        with patch.object(repair_engine, "generate_repair_plan", return_value={
            "repair_needed": True, "corrected_source_sentence": "changed source",
        }):
            result = repair_engine.repair_translation(
                request.source_text, request.translation, original_evidence,
                request.source_language, request.target_language, request.context,
                request.tone, verifier=lambda *args, **kwargs: passing_evidence(
                    status="REVIEW", risk_index={"score": 90},
                ), candidate_translator=lambda *_args: ["unsafe"], request=request,
            )

        self.assertFalse(result["accepted"])
        self.assertEqual(request.translation, "original")
        self.assertIs(request.verification_result, original_evidence)


if __name__ == "__main__":
    unittest.main()
