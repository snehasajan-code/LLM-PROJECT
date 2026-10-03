import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_experiment import experiment_enabled, run_experiment_case
from translation_pipeline import run_translation_request
from translation_request import TranslationRequest


def evidence(status, preserved=True, ranking_risk=None):
    result = {
        "status": status,
        "meaning_preserved": preserved,
        "selected_meaning_preserved": preserved,
        "semantic_constraint_satisfied": preserved,
        "numbers_preserved": True,
        "dates_preserved": True,
        "entities_preserved": True,
        "context_consistent": preserved,
        "major_additions": [],
        "major_omissions": [],
    }
    if ranking_risk is not None:
        result["risk_index"] = {"score": ranking_risk}
    return result


def request():
    return TranslationRequest("Hello.", "English", "Malayalam")


class CandidateExperimentTests(unittest.TestCase):
    def test_one_candidate_reports_requested_actual_and_timing(self):
        received = []
        metrics = run_experiment_case(
            request(), 1,
            verifier=lambda *_args, **_kwargs: evidence("PASS"),
            translator=lambda _text, _source, _target, count: (
                received.append(count) or ["നമസ്കാരം."]
            ),
        )
        self.assertEqual(received, [1])
        self.assertEqual(metrics["requested_candidates"], 1)
        self.assertEqual(metrics["actual_candidates"], 1)
        self.assertEqual(metrics["verified_candidates"], 1)
        self.assertGreaterEqual(metrics["generation_ms"], 0)
        self.assertGreaterEqual(metrics["verification_ms"], 0)
        self.assertGreaterEqual(metrics["total_ms"], 0)

    def test_three_candidates_count_statuses_and_existing_ranking(self):
        received = []
        candidates = ["weak", "high-risk pass", "low-risk pass"]
        statuses = {
            "weak": evidence("REVIEW", preserved=False),
            "high-risk pass": evidence("PASS", ranking_risk=100),
            "low-risk pass": evidence("PASS", ranking_risk=0),
        }
        metrics = run_experiment_case(
            request(), 3,
            verifier=lambda _source, text, *_args, **_kwargs: statuses[text],
            translator=lambda _text, _source, _target, count: (
                received.append(count) or candidates
            ),
        )
        self.assertEqual(received, [3])
        self.assertEqual(metrics["actual_candidates"], 3)
        self.assertEqual(metrics["pass_count"], 2)
        self.assertEqual(metrics["review_count"], 1)
        self.assertEqual(metrics["selected_index"], 2)
        self.assertEqual(metrics["translation"], "low-risk pass")
        self.assertIsNotNone(metrics["selected_evidence_score"])
        self.assertIsNotNone(metrics["selected_risk_index"])

    def test_duplicate_or_empty_candidates_report_actual_and_verified_counts(self):
        metrics = run_experiment_case(
            request(), 3,
            verifier=lambda *_args, **_kwargs: evidence("PASS"),
            translator=lambda *_args: ["first", "", "first"],
        )
        self.assertEqual(metrics["actual_candidates"], 3)
        self.assertEqual(metrics["verified_candidates"], 2)

    def test_repair_trigger_metric_records_actual_invocation_or_unavailable(self):
        runner = Mock()
        triggered = run_experiment_case(
            request(), 1,
            verifier=lambda *_args, **_kwargs: evidence("REVIEW", preserved=False),
            translator=lambda *_args: ["uncertain"],
            run_repair=True,
            repair_runner=runner,
        )
        self.assertTrue(triggered["repair_triggered"])
        runner.assert_called_once()

        not_triggered = run_experiment_case(
            request(), 1,
            verifier=lambda *_args, **_kwargs: evidence("PASS"),
            translator=lambda *_args: ["good"],
            run_repair=True,
            repair_runner=runner,
        )
        self.assertFalse(not_triggered["repair_triggered"])
        self.assertIsNone(run_experiment_case(
            request(), 1,
            verifier=lambda *_args, **_kwargs: evidence("PASS"),
            translator=lambda *_args: ["good"],
        )["repair_triggered"])

    def test_pipeline_production_default_stays_one_candidate(self):
        received = []
        run_translation_request(
            request(),
            verifier=lambda *_args, **_kwargs: evidence("PASS"),
            translator=lambda _text, _source, _target, count: (
                received.append(count) or ["നമസ്കാരം."]
            ),
        )
        self.assertEqual(received, [1])

    def test_experiment_requires_explicit_environment_opt_in(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(experiment_enabled())
        with patch.dict("os.environ", {"TRANSLATION_CANDIDATE_EXPERIMENT": "1"}):
            self.assertTrue(experiment_enabled())


if __name__ == "__main__":
    unittest.main()
