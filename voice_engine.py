import io
import tempfile
import asyncio
import wave



# ============================================================
# WHISPER CONFIGURATION
# ============================================================

WHISPER_MODEL = "base"

# CPU configuration
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


# ============================================================
# LANGUAGE INFORMATION
# ============================================================

LANGUAGES = {
    "en": "English",
    "ml": "Malayalam",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
}


# ============================================================
# EDGE TTS VOICES
# ============================================================

TTS_VOICES = {
    "English": "en-IN-NeerjaNeural",
    "Malayalam": "ml-IN-SobhanaNeural",
    "Hindi": "hi-IN-SwaraNeural",
    "Tamil": "ta-IN-PallaviNeural",
    "Telugu": "te-IN-ShrutiNeural",
    "Kannada": "kn-IN-SapnaNeural",
}


# ============================================================
# WHISPER MODEL
# ============================================================

_whisper_model = None


def load_whisper_model():

    global _whisper_model

    if _whisper_model is None:

        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )

    return _whisper_model


# ============================================================
# SPEECH → TEXT
# ============================================================

def _decode_recorded_wav(audio_bytes):
    """Decode Streamlit's PCM WAV recording into faster-whisper's 16 kHz array.

    Passing samples directly avoids faster-whisper's PyAV file decoder, which
    is incompatible with PyAV releases that removed ``metadata_errors``.
    """
    import numpy as np

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError("The voice recording must use uncompressed PCM WAV audio.")
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
    elif sample_width == 3:
        packed = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = packed[:, 0].astype(np.int32)
        values |= packed[:, 1].astype(np.int32) << 8
        values |= packed[:, 2].astype(np.int32) << 16
        values = (values ^ 0x800000) - 0x800000
        samples = values.astype(np.float32) / 8388608
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if sample_rate != 16000:
        if sample_rate <= 0:
            raise ValueError("The voice recording has an invalid sample rate.")
        output_length = round(len(samples) * 16000 / sample_rate)
        if output_length:
            old_positions = np.arange(len(samples), dtype=np.float32)
            new_positions = np.arange(output_length, dtype=np.float32) * sample_rate / 16000
            samples = np.interp(new_positions, old_positions, samples).astype(np.float32)

    return np.ascontiguousarray(samples, dtype=np.float32)

def transcribe_audio(audio_bytes):

    """
    Converts recorded speech into text.

    Also automatically detects the spoken language.
    """

    try:
        audio_samples = _decode_recorded_wav(audio_bytes)

        model = load_whisper_model()

        segments, info = model.transcribe(
            audio_samples,
            language=None,
            beam_size=5,
            vad_filter=True
        )

        segments = list(segments)

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()


        detected_code = info.language

        detected_language = LANGUAGES.get(
            detected_code,
            detected_code.upper()
        )

        confidence = float(
            info.language_probability
        )


        return {
            "success": True,
            "text": transcript,
            "language_code": detected_code,
            "language": detected_language,
            "confidence": confidence,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                for segment in segments
            ]
        }


    except Exception as e:

        return {
            "success": False,
            "text": "",
            "language_code": "",
            "language": "",
            "confidence": 0.0,
            "segments": [],
            "error": str(e)
        }


# ============================================================
# SCRIPT-BASED LANGUAGE MIX DETECTION
# ============================================================

def detect_language_mix(text):

    """
    Detects possible mixed-language text using Unicode scripts.

    This is complementary to Whisper's audio-level language
    detection. It should be described as script-based detection,
    not as a second speech-recognition confidence score.
    """

    counts = {
        "English": 0,
        "Malayalam": 0,
        "Hindi": 0,
        "Tamil": 0,
        "Telugu": 0,
        "Kannada": 0,
    }


    for char in text:

        code = ord(char)

        # Malayalam
        if 0x0D00 <= code <= 0x0D7F:
            counts["Malayalam"] += 1

        # Devanagari
        elif 0x0900 <= code <= 0x097F:
            counts["Hindi"] += 1

        # Tamil
        elif 0x0B80 <= code <= 0x0BFF:
            counts["Tamil"] += 1

        # Telugu
        elif 0x0C00 <= code <= 0x0C7F:
            counts["Telugu"] += 1

        # Kannada
        elif 0x0C80 <= code <= 0x0CFF:
            counts["Kannada"] += 1

        # Basic Latin
        elif (
            "A" <= char <= "Z"
            or "a" <= char <= "z"
        ):
            counts["English"] += 1


    active_languages = [
        language
        for language, count in counts.items()
        if count > 0
    ]


    total = sum(counts.values())

    percentages = {}

    if total > 0:

        percentages = {
            language: round(
                (count / total) * 100,
                1
            )
            for language, count in counts.items()
            if count > 0
        }


    return {
        "mixed": len(active_languages) > 1,
        "languages": active_languages,
        "counts": counts,
        "percentages": percentages
    }


# ============================================================
# TEXT → SPEECH
# ============================================================

async def _generate_speech(
    text,
    voice,
    output_file
):

    import edge_tts

    communicate = edge_tts.Communicate(
        text,
        voice
    )

    await communicate.save(
        output_file
    )


def text_to_speech(
    text,
    language
):

    """
    Converts translated text into spoken audio.
    """

    if not text or not text.strip():

        return {
            "success": False,
            "audio_path": None,
            "error": "No text supplied."
        }


    voice = TTS_VOICES.get(language)

    if voice is None:

        return {
            "success": False,
            "audio_path": None,
            "error": f"No TTS voice configured for {language}."
        }


    try:

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ).name


        asyncio.run(
            _generate_speech(
                text,
                voice,
                output_file
            )
        )


        return {
            "success": True,
            "audio_path": output_file,
            "voice": voice,
            "language": language
        }


    except Exception as e:

        return {
            "success": False,
            "audio_path": None,
            "error": str(e)
        }


# ============================================================
# COMPLETE VOICE PIPELINE
# ============================================================

def process_voice_input(audio_bytes):

    """
    Main voice-processing entry point.
    """

    result = transcribe_audio(
        audio_bytes
    )


    if not result["success"]:
        return result


    mix_info = detect_language_mix(
        result["text"]
    )


    result["language_mix"] = mix_info


    return result
