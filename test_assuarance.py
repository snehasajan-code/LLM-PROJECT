import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidate_engine
import evaluator
import repair_engine
import risk_engine
import verifier


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


if __name__ == "__main__":
    unittest.main()
