# AI Multilingual Translator

A Streamlit project for context-aware multilingual translation with ambiguity clarification, back-translation verification, evidence-based candidate selection, conservative repair, and voice input/output.

## Requirements

- Python 3.10 or newer
- Ollama running locally with `qwen2.5:3b` available for semantic checks and clarification
- The NLLB model downloads on first use; the initial model load may take several minutes depending on your connection and hardware

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5:3b
streamlit run app.py
```

The app runs locally at the URL printed by Streamlit. Translation uses the NLLB model configured in `translator.py`; Ollama provides local semantic checks. Voice transcription uses faster-whisper and speech output uses edge-tts.

## Tests

Run deterministic tests with:

```powershell
python -m unittest discover -s tests -v
```

The model integration test is skipped unless `RUN_MODEL_TESTS=1` is set and NLLB/Ollama are configured.

## Optional API smoke test

`test_api.py` is a separate OpenAI API connectivity check, not part of the translation pipeline. To run it, install `openai` and `python-dotenv`, set `OPENAI_API_KEY` in a local `.env` file, then run `python test_api.py`. The `.env` file is ignored by Git.

## Project modules

- `app.py`: Streamlit interface and pipeline wiring
- `translator.py`, `candidate_engine.py`: NLLB translation and candidate selection
- `verifier.py`, `evaluator.py`, `risk_engine.py`: evidence-based checks and reporting
- `ambiguity.py`, `prompts.py`: ambiguity detection and LLM prompt templates
- `repair_engine.py`: candidate repair with conservative acceptance
- `voice_engine.py`: speech recognition, language detection, and speech synthesis
- `tests/`: deterministic regression tests
