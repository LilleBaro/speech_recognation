# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech-to-text + sentiment analysis prototype for French audio: a FastAPI backend accepts an audio upload, transcribes it with a Wav2Vec2 ASR model, then classifies the sentiment of the transcription with a fine-tuned BERT classifier. The repo is a work-in-progress prototype — several pieces are stubbed, inconsistently wired, or empty (see "Known issues" below). When making changes, prefer small fixes that make the pipeline runnable over large rewrites.

## Commands

Dependencies are managed with `uv` (see `uv.lock`); Python `>=3.12` is required (`.python-version` pins 3.12).

```bash
uv sync                          # install dependencies into .venv
uv run uvicorn api:app --reload  # start the FastAPI backend (must run from repo root)
uv run python app.py             # start the Gradio UI (requires the backend above to be running)
uv run python -m models.asr      # run ASR module's own smoke test (transcribes models/sample.mp3)
uv run python -m preprocessing.audio_preprocessing  # smoke-test audio preprocessing
```

There is no test suite. Exploration/validation happens ad hoc in the notebooks under `notebooks/` (`test_audio_processing.ipynb`, `test_wav2vec.ipynb`, `test_sentiment_analysis_model.ipynb`).

## Architecture

Request flow: `api.py` → `models/pipeline.py::predict()` → `preprocessing/audio_preprocessing.py` → `models/asr.py` → `models/sentiment.py`.

- `api.py` — FastAPI app. `POST /predict` accepts a `.wav`/`.mp3` upload, saves it to `uploads/`, and calls `models.pipeline.predict(path)`, returning `{transcription, sentiment, confidence}` as JSON.
- `models/pipeline.py` — orchestrates the full pipeline: loads/resamples/normalizes audio, runs ASR, loads `BertClassifier` + checkpoint, runs sentiment prediction, maps the class index through `LABEL_MAP` (`0/1/2` → negative/neutral/positive).
- `models/asr.py` — loads `jonatasgrosman/wav2vec2-large-xlsr-53-french` (Wav2Vec2ForCTC) at import time and exposes `transcribe_audio(waveform, sample_rate)`.
- `models/sentiment.py` — `BertClassifier` (bert-base-uncased + linear head, 3 classes) and `predict(text, model, tokenizer, device)`. Has its own `if __name__ == "__main__"` smoke test using a 2-class `label_map`, inconsistent with the pipeline's 3-class map — don't assume the two are in sync.
- `preprocessing/audio_preprocessing.py` — `load_audio` (torchaudio), `convert_to_mono`, `audio_resampling` (→16kHz), `normalize_audio`, and the composed `preprocess_audio()` used by both `asr.py`'s smoke test and `pipeline.py`.
- `app.py` — Gradio UI that calls the FastAPI backend's `POST /predict` over HTTP (via `httpx`), so the API server must be running first. Backend URL defaults to `http://127.0.0.1:8000`, overridable with env var `API_URL`.

### Model checkpoint

The BERT checkpoint and tokenizer are always fetched from the Hugging Face repo `LilleBaro/fr_sentiment_analysis` (override with env var `HF_REPO_ID`) via `huggingface_hub.hf_hub_download` / `AutoTokenizer.from_pretrained` in `models/pipeline.py::predict()`. There is no Git LFS tracking anymore — `models/mon_checkpoint.pth` is gitignored and only used as the local `hf_hub_download` cache target, not committed.

## Known issues / gotchas

- **Import style requires running from the repo root.** Modules mix absolute imports (`from preprocessing.audio_preprocessing import ...`, `from models.pipeline import ...`) with relative imports (`from .sentiment import ...`, `from .asr import transcribe_audio` inside `models/pipeline.py`). This only resolves correctly when the repo root is on `sys.path` — i.e. run commands from the repo root as shown above, not from inside `models/` or `preprocessing/`.
- **`models.py` (top-level) vs. `models/` (package) name collision.** There is both a top-level `models.py` (a Pydantic schema, `Post`, unrelated to the ML models) and a `models/` package (`asr.py`, `sentiment.py`, `pipeline.py`). Only one can be imported as `models` at a time depending on Python's import resolution — be deliberate about which one you intend when adding `import models` or `from models import ...` anywhere in the repo.
- **Empty files**: `README.md` and `utils/helpers.py` are both empty — don't assume either contains real content or shared logic.
- `app.py`'s Gradio demo is a disconnected placeholder, not a UI for the actual pipeline.
