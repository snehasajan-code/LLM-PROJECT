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


if __name__ == "__main__":
    unittest.main()
