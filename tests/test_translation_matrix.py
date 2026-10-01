"""Required project test matrix; model-backed checks opt in with RUN_MODEL_TESTS=1."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_CASES = [
    ("A", "I am going to college tomorrow.", "English", "Malayalam", "General"),
    ("B", "I saw her duck.", "English", "Malayalam", "General"),
    ("C", "I saw her duck before the ball hit her.", "English", "Malayalam", "General"),
    ("D", "The temperature is 38 degrees.", "English", "Hindi", "General"),
    ("E", "The meeting is on 15 October 2026.", "English", "Tamil", "Business"),
    ("F", "Rahul works at Microsoft.", "English", "Malayalam", "Business"),
    ("G", "The model was trained using a neural network.", "English", "Malayalam", "Technical"),
    ("H", "The patient has high blood pressure.", "English", "Malayalam", "Medical"),
    ("I", "The company reported a 15 percent increase in revenue.", "English", "Hindi", "Business"),
    ("J1", "The model is accurate.", "English", "Malayalam", "Technical"),
    ("J2", "The model is accurate.", "English", "Malayalam", "Fashion"),
    ("K", "Although the train was delayed by heavy rain, the students reached the college before the examination began.", "English", "Malayalam", "Academic"),
    ("L", "നാളെ ഞാൻ കോളേജിലേക്ക് പോകുന്നു.", "Malayalam", "English", "General"),
    ("M", "I am going to college tomorrow.", "English", "Malayalam", "General"),
    ("N", "मैं कल कॉलेज जा रहा हूँ।", "Hindi", "English", "General"),
    ("O", "I am going to college tomorrow.", "English", "Tamil", "General"),
    ("P", "I am going to college tomorrow.", "English", "Telugu", "General"),
    ("Q", "I am going to college tomorrow.", "English", "Kannada", "General"),
]


class MatrixDefinitionTests(unittest.TestCase):
    def test_all_required_cases_are_present(self):
        self.assertEqual(len(TEST_CASES), 18)
        self.assertEqual({case[0] for case in TEST_CASES} - {"J1", "J2"},
                         set("ABCDEFGHI") | set("KLMNOPQ"))
        self.assertTrue(any(case[2] == "Malayalam" and case[3] == "English" for case in TEST_CASES))
        self.assertTrue(all(case[3] in {"English", "Malayalam", "Hindi", "Tamil", "Telugu", "Kannada"}
                            for case in TEST_CASES))


@unittest.skipUnless(os.getenv("RUN_MODEL_TESTS") == "1", "Set RUN_MODEL_TESTS=1 when NLLB and Ollama are configured.")
class NLLBIntegrationTests(unittest.TestCase):
    def test_required_translation_pairs_generate_nonempty_candidates(self):
        from translator import translate_candidates
        for case_id, text, source, target, _context in TEST_CASES:
            with self.subTest(case=case_id):
                self.assertTrue(translate_candidates(text, source, target, 1)[0])


if __name__ == "__main__":
    unittest.main()
