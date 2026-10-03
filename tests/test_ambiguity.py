import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

import ambiguity


class AmbiguityFallbackTests(unittest.TestCase):
    def test_unavailable_ollama_is_reported_as_unknown(self):
        fake_requests = SimpleNamespace(post=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Ollama offline")
        ))
        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = ambiguity.detect_ambiguity(
                "A test sentence.", "English", "Malayalam", "General"
            )

        self.assertFalse(result["available"])
        self.assertFalse(result["ambiguous"])
        self.assertIn("Ollama offline", result["error"])

    def test_valid_no_ambiguity_response_is_available(self):
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": '{"ambiguous": false}'},
        )
        fake_requests = SimpleNamespace(post=lambda *args, **kwargs: response)
        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = ambiguity.detect_ambiguity(
                "A test sentence.", "English", "Malayalam", "General"
            )

        self.assertTrue(result["available"])
        self.assertFalse(result["ambiguous"])

    def test_known_duck_ambiguity_is_available_without_ollama(self):
        fake_requests = SimpleNamespace(post=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("known ambiguity should not call Ollama")
        ))
        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = ambiguity.detect_ambiguity(
                "I saw her duck.", "English", "Malayalam", "General"
            )

        self.assertTrue(result["available"])
        self.assertTrue(result["ambiguous"])
        self.assertEqual(result["interpretations"], [
            "I saw the duck that belongs to her.",
            "I saw her bow her head.",
        ])

    def test_model_duck_choices_are_canonicalized_to_complete_paraphrases(self):
        model_result = {
            "ambiguous": True,
            "interpretations": ["I saw her duck, the bird.", "I saw her lower her head."],
            "semantic_constraints": ["Duck means the bird.", "Duck means lowering her head."],
        }
        result = ambiguity._canonicalize_known_duck_result(
            "I saw her duck.", model_result, "English"
        )

        self.assertEqual(result["interpretations"], [
            "I saw the duck that belongs to her.",
            "I saw her bow her head.",
        ])
        self.assertEqual(result["semantic_constraints"], [
            "Duck refers to the bird/animal that belongs to her, not the action of lowering a head.",
            "Duck refers to her lowering/bowing her head, not to a bird/animal.",
        ])


if __name__ == "__main__":
    unittest.main()
