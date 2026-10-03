import importlib.util
import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

import voice_engine


def make_wav(samples=(0, 1000, -1000, 0), sample_rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))
    return buffer.getvalue()


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is required by faster-whisper")
class VoiceTranscriptionCompatibilityTests(unittest.TestCase):
    def test_recorded_wav_is_decoded_to_float_samples(self):
        samples = voice_engine._decode_recorded_wav(make_wav())

        self.assertEqual(samples.dtype.name, "float32")
        self.assertEqual(samples.shape, (4,))
        self.assertAlmostEqual(float(samples[1]), 1000 / 32768, places=6)

    def test_transcription_passes_samples_instead_of_audio_path(self):
        captured = {}

        class Model:
            def transcribe(self, audio, **kwargs):
                captured["audio"] = audio
                return [SimpleNamespace(start=0.0, end=0.2, text=" Hello ")], SimpleNamespace(
                    language="en", language_probability=0.98
                )

        with patch.object(voice_engine, "load_whisper_model", return_value=Model()):
            result = voice_engine.transcribe_audio(make_wav())

        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "Hello")
        self.assertNotIsInstance(captured["audio"], (str, bytes))


class LanguageMixTests(unittest.TestCase):
    def test_english_and_malayalam_scripts_are_reported_as_mixed(self):
        result = voice_engine.detect_language_mix("Hello, നമസ്കാരം.")

        self.assertTrue(result["mixed"])
        self.assertEqual(result["languages"], ["English", "Malayalam"])

    def test_latin_only_transcript_does_not_claim_mixed_language(self):
        result = voice_engine.detect_language_mix("Hallo, namens Gare.")

        self.assertFalse(result["mixed"])
        self.assertEqual(result["languages"], ["English"])


if __name__ == "__main__":
    unittest.main()
