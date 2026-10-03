import json
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidate_engine
import evaluator
import repair_engine
import risk_engine
import verifier
import translator


class RiskAndEvaluationTests(unittest.TestCase):
    def test_risk_is_explainable_and_not_probability(self):
        risk = risk_engine.assess_risk({"status": "REVIEW", "meaning_preserved": None})
        self.assertEqual(risk["action"], "VERIFY")
        self.assertIn("not a calibrated probability", risk["interpretation"])
        self.assertTrue(risk["reasons"])

    def test_evaluator_never_returns_none_scores(self):
        result = evaluator.evaluate_translation(
            "She lowered her head.", "translation", "English", "Malayalam",
            verification={"status": "PASS", "meaning_preserved": True,
                          "numbers_preserved": True, "context_consistent": None},
        )
        self.assertNotIn(None, result.values())
        self.assertEqual(result["grammar_naturalness"], "N/A")
        self.assertEqual(result["overall"], "PASS")

    def test_evaluator_reports_selected_interpretation_when_clarified(self):
        result = evaluator.evaluate_translation(
            "I saw her duck.", "target translation", "English", "Malayalam",
            clarification="I saw the duck that belongs to her.",
            verification={"status": "PASS", "meaning_preserved": True,
                          "selected_meaning_preserved": True},
        )
        self.assertEqual(result["ambiguity_resolution"], "PASS")


class VerificationTests(unittest.TestCase):
    @patch.object(verifier, "semantic_check")
    @patch.object(verifier, "nllb_translate", return_value="She bent her head downward.")
    def test_paraphrase_can_pass_semantic_check(self, _back, semantic):
        semantic.return_value = {
            "meaning_preserved": True, "selected_meaning_preserved": True,
            "semantic_constraint_satisfied": True, "context_consistent": True,
            "major_additions": [], "major_omissions": [],
            "semantic_similarity": 0.94, "reason": "Paraphrase preserves the action.",
        }
        result = verifier.verify_translation(
            "She lowered her head.", "target text", "English", "Malayalam"
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["meaning_preserved"])

    @patch.object(verifier, "semantic_check", return_value={
        "meaning_preserved": True, "selected_meaning_preserved": True,
        "semantic_constraint_satisfied": True, "context_consistent": True,
        "major_additions": [], "major_omissions": [], "semantic_similarity": 0.91,
        "reason": "The back-translation preserves her duck as the bird.",
    })
    @patch.object(verifier, "nllb_translate", return_value="I saw her duck, bird.")
    def test_selected_duck_sense_is_canonical_verification_reference(self, _back, semantic):
        result = verifier.verify_translation(
            "I saw her duck.", "Malayalam candidate", "English", "Malayalam",
            clarification="I saw her duck, the bird.",
            semantic_constraint="Duck means the animal, not lowering the head.",
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["risk_index"]["score"], 10)
        self.assertEqual(semantic.call_args.kwargs["resolved_reference"],
                         "I saw the duck that belongs to her.")

    @patch.object(verifier, "semantic_check", return_value={
        "meaning_preserved": False, "selected_meaning_preserved": False,
        "semantic_constraint_satisfied": False, "context_consistent": False,
        "major_additions": [], "major_omissions": [], "semantic_similarity": 0.42,
        "reason": "Aspect wording differs.",
    })
    @patch.object(verifier, "nllb_translate", return_value="I saw her bowing her head.")
    def test_duck_action_progressive_backtranslation_is_equivalent(self, _back, _semantic):
        result = verifier.verify_translation(
            "I saw her duck.", "അവൾ തല കുനിക്കുന്നത് ഞാൻ കണ്ടു.",
            "English", "Malayalam",
            clarification="I saw her bow her head.",
            semantic_constraint="Duck refers to her bowing her head, not a bird.",
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["meaning_preserved"])
        self.assertTrue(result["selected_meaning_preserved"])
        self.assertTrue(result["semantic_constraint_satisfied"])
        self.assertTrue(result["context_consistent"])
        self.assertIn("recognized action paraphrase", result["evidence_source"])

    @patch.object(verifier, "semantic_check", return_value={
        "meaning_preserved": False, "selected_meaning_preserved": False,
        "semantic_constraint_satisfied": False, "context_consistent": False,
        "major_additions": [], "major_omissions": [], "semantic_similarity": 0.2,
        "reason": "Meaning differs.",
    })
    @patch.object(verifier, "nllb_translate", return_value="I saw her bowing.")
    def test_non_equivalent_duck_action_backtranslation_stays_review(self, _back, _semantic):
        result = verifier.verify_translation(
            "I saw her duck.", "target", "English", "Malayalam",
            clarification="I saw her bow her head.",
            semantic_constraint="Duck refers to bowing her head, not a bird.",
        )
        self.assertEqual(result["status"], "REVIEW")


class SemanticResponseValidationTests(unittest.TestCase):
    @staticmethod
    def response_for(payload):
        return SimpleNamespace(
            json=lambda: {"response": json.dumps(payload)},
            raise_for_status=lambda: None,
        )

    @staticmethod
    def valid_payload():
        return {
            "meaning_preserved": True,
            "selected_meaning_preserved": True,
            "semantic_constraint_satisfied": True,
            "context_consistent": True,
            "major_additions": [], "major_omissions": [],
            "semantic_similarity": 0.9, "reason": "Preserved.",
        }

    def parse(self, payload):
        requests_stub = SimpleNamespace(post=Mock(return_value=self.response_for(payload)))
        with patch.dict(sys.modules, {"requests": requests_stub}):
            return verifier._ollama_semantic_check("source", "back", "English")

    def test_only_json_booleans_are_accepted(self):
        for value in (True, False):
            with self.subTest(value=value):
                payload = self.valid_payload()
                payload["meaning_preserved"] = value
                self.assertIs(self.parse(payload)["meaning_preserved"], value)

        for value in (0, 1, "true", "false", "yes", "no", None):
            with self.subTest(value=value):
                payload = self.valid_payload()
                payload["meaning_preserved"] = value
                self.assertIsNone(self.parse(payload)["meaning_preserved"])

        payload = self.valid_payload()
        del payload["meaning_preserved"]
        self.assertIsNone(self.parse(payload)["meaning_preserved"])

    def test_additions_and_omissions_preserve_valid_lists_and_reject_other_types(self):
        payload = self.valid_payload()
        self.assertEqual(self.parse(payload)["major_additions"], [])
        payload["major_additions"] = ["an extra event"]
        self.assertEqual(self.parse(payload)["major_additions"], ["an extra event"])
        payload["major_omissions"] = ["a missing event"]
        self.assertEqual(self.parse(payload)["major_omissions"], ["a missing event"])

        for field in ("major_additions", "major_omissions"):
            for value in ("none", {"item": "extra"}, None, 1, True):
                with self.subTest(field=field, value=value):
                    payload = self.valid_payload()
                    payload[field] = value
                    self.assertIsNone(self.parse(payload)[field])

    @patch.object(verifier, "nllb_translate", return_value="Hello.")
    def test_malformed_additions_evidence_forces_review_in_verifier(self, _back):
        payload = self.valid_payload()
        payload["major_additions"] = "none"
        requests_stub = SimpleNamespace(post=Mock(return_value=self.response_for(payload)))
        with patch.dict(sys.modules, {"requests": requests_stub}):
            result = verifier.verify_translation("Hello.", "ഹലോ.", "English", "Malayalam")
        self.assertEqual(result["status"], "REVIEW")
        self.assertIsNone(result["major_additions"])

    @patch.object(verifier, "nllb_translate", return_value="Hello.")
    def test_missing_required_boolean_forces_review_in_verifier(self, _back):
        payload = self.valid_payload()
        del payload["meaning_preserved"]
        requests_stub = SimpleNamespace(post=Mock(return_value=self.response_for(payload)))
        with patch.dict(sys.modules, {"requests": requests_stub}):
            result = verifier.verify_translation("Hello.", "ഹലോ.", "English", "Malayalam")
        self.assertEqual(result["status"], "REVIEW")
        self.assertIsNone(result["meaning_preserved"])

    @patch.object(verifier, "nllb_translate", return_value="Good morning.")
    def test_transport_failure_stays_unknown_and_does_not_pass(self, _back):
        requests_stub = SimpleNamespace(post=Mock(side_effect=RuntimeError("Ollama unavailable")))
        with patch.dict(sys.modules, {"requests": requests_stub}):
            evidence = verifier.semantic_check("Good morning.", "Good morning.", "English")
            result = verifier.verify_translation(
                "Good morning.", "സുപ്രഭാതം.", "English", "Malayalam"
            )
        self.assertIsNone(evidence["meaning_preserved"])
        self.assertIsNone(evidence["major_additions"])
        self.assertEqual(result["status"], "REVIEW")
        self.assertIsNone(result["meaning_preserved"])

    @patch.object(verifier, "nllb_translate", return_value="I saw her bowing her head.")
    def test_duck_action_unknown_flags_are_not_overridden(self, _back):
        payload = self.valid_payload()
        payload["meaning_preserved"] = 1
        requests_stub = SimpleNamespace(post=Mock(return_value=self.response_for(payload)))
        with patch.dict(sys.modules, {"requests": requests_stub}):
            result = verifier.verify_translation(
                "I saw her duck.", "action translation", "English", "Malayalam",
                clarification="I saw her bow her head.",
                semantic_constraint="Duck refers to lowering her head, not a bird.",
            )
        self.assertEqual(result["status"], "REVIEW")
        self.assertIsNone(result["meaning_preserved"])

    @patch.object(verifier, "nllb_translate", return_value="I saw her bowing her head.")
    def test_duck_action_malformed_json_is_not_overridden(self, _back):
        response = SimpleNamespace(
            json=lambda: {"response": "not valid JSON"},
            raise_for_status=lambda: None,
        )
        requests_stub = SimpleNamespace(post=Mock(return_value=response))
        with patch.dict(sys.modules, {"requests": requests_stub}):
            result = verifier.verify_translation(
                "I saw her duck.", "action translation", "English", "Malayalam",
                clarification="I saw her bow her head.",
                semantic_constraint="Duck refers to lowering her head, not a bird.",
            )
        self.assertEqual(result["status"], "REVIEW")
        self.assertIsNone(result["meaning_preserved"])


class SourceNormalizationTests(unittest.TestCase):
    def test_short_translations_use_bounded_single_beam_generation(self):
        generated = {}

        class Tokens:
            shape = (1, 8)

            def to(self, _device):
                return self

        class Tokenizer:
            src_lang = None

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": Tokens()}

            def convert_tokens_to_ids(self, _language):
                return 7

            def batch_decode(self, _output, **_kwargs):
                return ["translated"]

        class Model:
            def parameters(self):
                return iter([SimpleNamespace(device="cpu")])

            def generate(self, **kwargs):
                generated.update(kwargs)
                return [object()]

        fake_torch = SimpleNamespace(inference_mode=nullcontext)
        with patch.dict(sys.modules, {"torch": fake_torch}), \
             patch.object(translator, "load_translation_model", return_value=(Tokenizer(), Model())):
            translated = translator._generate("short input", "English", "Malayalam", variants=1)

        self.assertEqual(translated, ["translated"])
        self.assertEqual(generated["num_beams"], 3)
        self.assertEqual(generated["num_return_sequences"], 1)
        self.assertEqual(generated["max_new_tokens"], 64)

    def test_clarification_is_naturalized_before_nllb(self):
        normalized = translator.normalize_clarification(
            "I saw her duck.", "I saw her duck, the bird.",
            "Duck is the animal, not the action.", "English", "General",
        )
        self.assertEqual(normalized, "I saw the duck that belongs to her.")

    def test_duck_action_clarification_is_canonical_without_ollama(self):
        normalized = translator.normalize_clarification(
            "I saw her duck.", "I saw her lower her head.",
            "Duck is the action of lowering her head, not a bird.",
            "English", "General",
        )
        self.assertEqual(normalized, "I saw her bow her head.")

    @patch.object(verifier, "semantic_check", return_value={
        "meaning_preserved": True, "selected_meaning_preserved": True,
        "semantic_constraint_satisfied": True, "context_consistent": True,
        "major_additions": [], "major_omissions": [], "semantic_similarity": 0.9,
    })
    @patch.object(verifier, "nllb_translate", return_value="Temperature is 39 degrees on 15 October 2026.")
    def test_changed_number_triggers_review(self, _back, _semantic):
        result = verifier.verify_translation(
            "Temperature is 38 degrees on 15 October 2026.", "target", "English", "Hindi"
        )
        self.assertEqual(result["status"], "REVIEW")
        self.assertFalse(result["numbers_preserved"])


class CandidateAndRepairTests(unittest.TestCase):
    def test_candidate_ranking_uses_verifier_evidence(self):
        candidates = ["weak", "strong"]
        def verify(text):
            return {"status": "PASS" if text == "strong" else "REVIEW",
                    "meaning_preserved": text == "strong",
                    "numbers_preserved": True, "entities_preserved": True,
                    "dates_preserved": True, "selected_meaning_preserved": True,
                    "semantic_constraint_satisfied": True, "context_consistent": True}
        result = candidate_engine.select_best_candidate(candidates, verify)
        self.assertEqual(result["translation"], "strong")

    def test_clarification_constraint_and_context_reach_candidate_verifier(self):
        calls = []
        def translator(text, source, target, count):
            self.assertEqual(text, "I saw her duck, the bird.")
            return ["candidate"]
        def verify(*args, **kwargs):
            calls.append((args, kwargs))
            return {"status": "PASS", "meaning_preserved": True,
                    "selected_meaning_preserved": True,
                    "semantic_constraint_satisfied": True,
                    "numbers_preserved": True, "dates_preserved": True,
                    "entities_preserved": True, "context_consistent": True}
        candidate_engine.translate_and_rank(
            "I saw her duck.", "English", "Malayalam", verify,
            clarification="I saw her duck, the bird.",
            semantic_constraint="Duck refers to the animal, not the action.",
            context="General", translator=translator,
        )
        self.assertEqual(calls[0][1]["semantic_constraint"],
                         "Duck refers to the animal, not the action.")
        self.assertEqual(calls[0][1]["clarification"], "I saw her duck, the bird.")
        self.assertEqual(calls[0][1]["context"], "General")

    @patch.object(repair_engine, "generate_repair_plan", return_value={
        "repair_needed": True, "problem": "semantic mismatch",
        "corrected_source_sentence": "clarified source",
    })
    def test_bad_repair_is_rejected_and_original_retained(self, _plan):
        original_evidence = {
            "status": "REVIEW", "meaning_preserved": False,
            "selected_meaning_preserved": True, "semantic_constraint_satisfied": True,
            "numbers_preserved": True, "dates_preserved": True,
            "entities_preserved": True, "context_consistent": True,
        }
        candidate_evidence = dict(original_evidence)
        result = repair_engine.repair_translation(
            "source", "original target", original_evidence,
            "English", "Malayalam", "General", "Natural",
            candidate_count=1,
            verifier=lambda *args, **kwargs: candidate_evidence,
            candidate_translator=lambda *args: ["same quality candidate"],
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["repaired"])
        self.assertEqual(result["initial_translation"], "original target")
        self.assertTrue(result["human_review"])

    @patch.object(repair_engine, "generate_repair_plan", return_value={
        "repair_needed": True, "problem": "meaning lost",
        "corrected_source_sentence": "clarified source",
    })
    def test_repair_accepted_only_when_evidence_improves(self, _plan):
        original = {"status": "REVIEW", "meaning_preserved": False,
                    "selected_meaning_preserved": True, "semantic_constraint_satisfied": True,
                    "numbers_preserved": True, "dates_preserved": True,
                    "entities_preserved": True, "context_consistent": True}
        improved = {"status": "PASS", "meaning_preserved": True,
                    "selected_meaning_preserved": True, "semantic_constraint_satisfied": True,
                    "numbers_preserved": True, "dates_preserved": True,
                    "entities_preserved": True, "context_consistent": True,
                    "risk_index": {"score": 0}}
        result = repair_engine.repair_translation(
            "source", "original", original, "English", "Malayalam", "General", "Natural",
            candidate_count=1, verifier=lambda *args, **kwargs: improved,
            candidate_translator=lambda *args: ["better"],
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["repaired_translation"], "better")

    @patch.object(repair_engine, "generate_repair_plan", return_value={
        "repair_needed": True, "problem": "uncertain meaning",
        "corrected_source_sentence": "clarified source",
    })
    def test_improved_review_candidate_is_suggested_but_original_retained(self, _plan):
        original = {"status": "REVIEW", "meaning_preserved": False,
                    "selected_meaning_preserved": True, "semantic_constraint_satisfied": True,
                    "numbers_preserved": True, "dates_preserved": True,
                    "entities_preserved": True, "context_consistent": True}
        improved_but_uncertain = {"status": "REVIEW", "meaning_preserved": True,
                    "selected_meaning_preserved": True, "semantic_constraint_satisfied": True,
                    "numbers_preserved": True, "dates_preserved": True,
                    "entities_preserved": True, "context_consistent": True,
                    "risk_index": {"score": 20}}
        result = repair_engine.repair_translation(
            "source", "original translation", original,
            "English", "Malayalam", "General", "Natural", candidate_count=1,
            verifier=lambda *args, **kwargs: improved_but_uncertain,
            candidate_translator=lambda *args: ["suggested candidate"],
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["initial_translation"], "original translation")
        self.assertEqual(result["recommended_candidate"], "suggested candidate")
        self.assertTrue(result["human_review"])


class IdiomInterpretationTests(unittest.TestCase):
    def test_raining_cats_and_dogs_is_paraphrased_before_translation(self):
        result = translator.interpret_idiom("its raining cats and dogs.")
        self.assertEqual(result["meaning"], "It is raining very heavily.")
        self.assertEqual(result["normalized_text"], "It is raining very heavily.")

    def test_idiom_phrase_inside_sentence_is_paraphrased(self):
        result = translator.interpret_idiom("It is raining cats and dogs outside.")
        self.assertEqual(result["normalized_text"], "It is raining very heavily outside.")

    def test_selected_idiom_meaning_is_not_normalized_back_to_literal_phrase(self):
        source = "its raining cats and dogs."
        meaning = "It is raining very heavily."
        self.assertEqual(
            translator.normalize_clarification(source, meaning, None, "English"),
            "It is raining very heavily.",
        )

    def test_candidate_flow_translates_selected_idiom_meaning(self):
        seen = {}
        def fake_candidates(text, source, target, count):
            seen["source"] = text
            return ["മഴ ശക്തമായി പെയ്യുന്നു."]
        with patch("translator.translate_candidates", side_effect=fake_candidates):
            with patch("translator.normalize_clarification", side_effect=AssertionError("should bypass model normalization")):
                candidate_engine.translate_and_rank(
                    "its raining cats and dogs.", "English", "Malayalam",
                    lambda *args, **kwargs: {"status": "REVIEW", "evidence_score": 0},
                    clarification="It is raining very heavily.", candidate_count=1,
                )
        self.assertEqual(seen["source"], "It is raining very heavily.")

    def test_unknown_and_non_english_text_are_untouched(self):
        self.assertIsNone(translator.interpret_idiom("A normal sentence."))
        self.assertIsNone(translator.interpret_idiom("raining cats and dogs", "Malayalam"))

    def test_known_bad_nllb_examples_use_curated_malayalam_meaning(self):
        cases = [
            ("I saw the duck that belongs to her.", "English", "Malayalam", "ഞാൻ അവളുടെ താറാവിനെ കണ്ടു."),
            ("I saw her bow her head.", "English", "Malayalam", "അവൾ തല കുനിക്കുന്നത് ഞാൻ കണ്ടു."),
            ("It is raining very heavily.", "English", "Malayalam", "കനത്ത മഴ പെയ്യുന്നു."),
            ("ഞാൻ അവളുടെ താറാവിനെ കണ്ടു.", "Malayalam", "English", "I saw the duck that belongs to her."),
            ("അവൾ തല കുനിക്കുന്നത് ഞാൻ കണ്ടു.", "Malayalam", "English", "I saw her bow her head."),
            ("കനത്ത മഴ പെയ്യുന്നു.", "Malayalam", "English", "It is raining very heavily."),
        ]
        with patch.object(translator, "load_translation_model", side_effect=AssertionError("model should not be needed")):
            for source, source_language, target_language, expected in cases:
                with self.subTest(source=source):
                    self.assertEqual(
                        translator.nllb_translate(source, source_language, target_language),
                        expected,
                    )

    def test_candidate_generation_receives_idiom_meaning(self):
        seen = {}
        def fake_translator(text, source, target, count):
            seen["text"] = text
            return ["മഴ ശക്തമായി പെയ്യുന്നു."]
        result = candidate_engine.translate_and_rank(
            "its raining cats and dogs.", "English", "Malayalam",
            lambda *args, **kwargs: {"status": "REVIEW", "evidence_score": 0},
            translator=fake_translator,
        )
        self.assertEqual(seen["text"], "It is raining very heavily.")
        self.assertTrue(result["translation"])

    def test_verification_uses_idiom_meaning_as_reference(self):
        with patch.object(verifier, "nllb_translate", return_value="It is raining very heavily."):
            with patch.object(verifier, "semantic_check") as check:
                check.return_value = {
                    "meaning_preserved": True, "selected_meaning_preserved": True,
                    "semantic_constraint_satisfied": True, "context_consistent": True,
                    "major_additions": [], "major_omissions": [],
                    "semantic_similarity": 0.95, "reason": "Meaning preserved.",
                }
                result = verifier.verify_translation(
                    "its raining cats and dogs.", "മഴ ശക്തമായി പെയ്യുന്നു.",
                    "English", "Malayalam",
                )
        self.assertEqual(check.call_args.kwargs["resolved_reference"], "It is raining very heavily.")
        self.assertEqual(result["idiom"]["phrase"], "It is raining cats and dogs")


if __name__ == "__main__":
    unittest.main()
