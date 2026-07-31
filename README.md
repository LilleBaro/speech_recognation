# Speech Recognition & Sentiment Analysis Prototype

A prototype that analyzes French audio recordings in two steps:

1. **Transcription** a Wav2Vec2 ASR model converts speech to text.
2. **Sentiment analysis** a fine-tuned DistilBERT classifier predicts whether the transcription is negative, neutral, or positive.

This is a Master's 2 AI coursework project. The codebase is a working prototype, not production software, see [Known limitations](#known-limitations) for the rough edges.

## Architecture

Two deployment shapes exist for the same underlying pipeline:

### Local development: FastAPI backend + Gradio frontend (this repository)

1. User Interface 
   ┌─────────────┐
   │  app.py     │
   │ (Gradio UI) │
   └───────┬─────┘
           │
           │ HTTP POST /predict (multipart audio)
           ▼

2. API
   ┌──────────────┐
   │   api.py     │
   │  (FastAPI)   │
   └───────┬──────┘
           │
           │ pipeline.predict(path)
           ▼

3. Pipeline
   ┌──────────────────────┐
   │ models/pipeline.py   │
   └──────────┬───────────┘
              │
              ▼

4. Modules called by the pipeline
   ┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
   ▼                          ▼                          ▼
   preprocessing/             models/asr.py              models/sentiment.py
   audio_preprocessing.py     (Wav2Vec2ForCTC,           (BertClassifier sur
   (mono → 16kHz →            jonatasgrosman/            distilbert-base-
   normalize)                 wav2vec2-large-xlsr-       multilingual-cased,
                              53-french)                 fine-tuned checkpoint)


`app.py` and `api.py` are two separate processes that talk over HTTP, you must run both. This mirrors a real client/server split and lets the API be tested independently (curl, Postman, another frontend) of the demo UI.

### Hosted demo: standalone Gradio app on Hugging Face Spaces

A second, single-process variant is deployed at **https://huggingface.co/spaces/LilleBaro/Speech_recognation**. It lives in its own repository (not a subfolder of this one) because a Space runs one process: its `app.py` calls `models/pipeline.py::predict()` directly instead of proxying over HTTP to a FastAPI server. That Space also runs on ZeroGPU hardware, which requires the inference entry point to be decorated with `@spaces.GPU` and requires the CUDA device to be resolved *inside* that decorated call rather than at import time (ZeroGPU only grants GPU access for the duration of the decorated function).

### Model details

| Stage | Model | Source |
|---|---|---|
| ASR | `jonatasgrosman/wav2vec2-large-xlsr-53-french` (Wav2Vec2ForCTC) | Downloaded from the Hugging Face Hub at import time |
| Sentiment | DistilBERT (`distilbert-base-multilingual-cased` backbone + linear classification head, 3 classes) fine-tuned on [SetFit/amazon_reviews_multi_fr](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_fr) (ratings 0–4, collapsed to negative/neutral/positive) | Weights (`mon_checkpoint.pth`) and tokenizer downloaded from `LilleBaro/DistilBert_sentiment_analysis` via `huggingface_hub.hf_hub_download`, overridable with the `HF_REPO_ID` env var |

## Repository structure

- [api.py](api.py) - FastAPI application exposing `POST /predict`; saves the upload to `uploads/` and calls the pipeline.
- [app.py](app.py) - Gradio demo frontend; sends the recorded/uploaded audio to the FastAPI backend over HTTP.
- [models/pipeline.py](models/pipeline.py) - orchestrates preprocessing → ASR → sentiment classification.
- [models/asr.py](models/asr.py) - Wav2Vec2 model/processor and `transcribe_audio()`.
- [models/sentiment.py](models/sentiment.py) - `BertClassifier` definition, checkpoint/tokenizer loading, and `predict()`.
- [preprocessing/audio_preprocessing.py](preprocessing/audio_preprocessing.py) - load, mono-conversion, resampling to 16kHz, amplitude normalization.
- [exceptions.py](exceptions.py) - `AudioProcessingError`, raised for corrupt/empty/silent audio.
- [models.py](models.py) - a Pydantic response schema (`Post`); unrelated to the `models/` ML package (see [Known limitations](#known-limitations)).
- [notebooks/](notebooks/) - exploratory/validation notebooks (no automated test suite exists).
- [uploads/](uploads/) - where `api.py` persists incoming audio files.

## Requirements

- Python 3.12+ (pinned via `.python-version`)
- Dependencies managed with [`uv`](https://docs.astral.sh/uv/)
- A Hugging Face account is **not** required to run it (the model repos used are public), but setting `HF_TOKEN` avoids Hub rate limits on repeated downloads.

## Installation

```bash
git clone https://github.com/LilleBaro/speech_recognation.git
cd speech_recognation
uv sync
```

This creates `.venv/` and installs everything pinned in `uv.lock` (FastAPI, Gradio, PyTorch, TorchAudio, Transformers, etc.).

## Reproduction steps

All commands must be run from the repository root, several modules mix absolute (`from preprocessing...`, `from models...`) and relative (`from .asr import ...`) imports, which only resolve correctly when the root is on `sys.path`.

### 1. Start the FastAPI backend

```bash
uv run uvicorn api:app --reload
```

Available at `http://127.0.0.1:8000`. First request triggers model downloads (ASR + sentiment weights, a few GB combined), expect a slow first call.

### 2. Start the Gradio UI (second terminal)

```bash
uv run python app.py
```

Opens a local UI that uploads/records audio and forwards it to the backend above. Requires `API_URL` to point at a reachable backend if not running on the default `http://127.0.0.1:8000`.

### 3. Call the API directly (optional, no UI needed)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/path/to/audio.wav"
```

Actual response shape (see the label-mapping caveat in [Known limitations](#known-limitations)):

```json
{
  "transcription": "...",
  "sentiment": "Positif",
  "confidence": 0.92
}
```

### Standalone module smoke tests

```bash
uv run python -m models.asr                          # transcribes models/sample.mp3, prints the text
uv run python -m preprocessing.audio_preprocessing    # loads + preprocesses the same sample, prints tensor shape
uv run python -m models.sentiment                     # runs sentiment inference on a hardcoded French sentence
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `API_URL` | `http://127.0.0.1:8000` | Backend URL the Gradio UI (`app.py`) sends requests to |
| `HF_REPO_ID` | `LilleBaro/DistilBert_sentiment_analysis` | Hugging Face repo the sentiment checkpoint + tokenizer are pulled from |
| `HF_MODEL_NAME` | `distilbert-base-multilingual-cased` | Backbone name used when instantiating `BertClassifier` (must match the checkpoint's architecture) |

## Use cases

- **Coursework demo**: an end-to-end example of chaining a pretrained ASR model with a fine-tuned sentiment classifier, for the Deep Learning 2 M2 curriculum.
- **French customer-feedback triage**: transcribe short voice notes or call-center clips and get a quick negative/neutral/positive read without a human listening first.
- **API integration testing**: `api.py` exposes a plain JSON endpoint, so it can be called from any HTTP client (curl, Postman, another service) independent of the Gradio UI.
- **Standalone hosted demo**: anyone can try the pipeline without any local setup via the [Hugging Face Space](https://huggingface.co/spaces/LilleBaro/Speech_recognation).

Not a fit for: real-time/streaming transcription (whole-file only), non-French audio (both models are French-specific), or long recordings (single-pass in-memory processing, no chunking).

## Known limitations

- **No automated test suite.** Validation is ad hoc, via the notebooks in [notebooks/](notebooks/) (`test_audio_processing.ipynb`, `test_wav2vec.ipynb`, `test_sentiment_analysis_model.ipynb`).
- **Import style requires running from the repo root.** Mixed absolute/relative imports across `models/` and `preprocessing/` only resolve correctly when invoked from the repository root (see [Reproduction steps](#reproduction-steps)).
- **No GPU by default locally.** Both models resolve `torch.device("cuda" if torch.cuda.is_available() else "cpu")` at call time; if no local CUDA GPU is present (the common case for this coursework setup), everything runs on CPU, and a full ASR + sentiment pass on a several-second clip can take a noticeable amount of time.
- **Fresh cold-start downloads.** ASR weights (~1.2GB) and the sentiment checkpoint (~540MB) are fetched from the Hugging Face Hub on first use, not vendored in the repo, the first request after starting the backend will be slow, and repeated cold starts without `HF_TOKEN` risk Hub rate limiting.
- **No batching, chunking, or streaming.** Audio is loaded and processed entirely in memory in a single pass; very long recordings are not split, which can exhaust memory or produce degraded ASR quality (Wav2Vec2 wasn't trained on very long single utterances).
- **Naive silence detection.** `preprocessing/audio_preprocessing.py::normalize_audio()` rejects audio below a fixed peak-amplitude threshold (`1e-4`) as "silent", a legitimately quiet-but-valid recording could be rejected.
- **Uploads are never cleaned up.** `api.py` writes every upload to `uploads/<original filename>` and never deletes it; disk usage grows unbounded, and two concurrent uploads with the same filename can race and overwrite each other.
- **Confidence is an uncalibrated softmax max.** `confidence` is `max(softmax(logits))` from the classifier, a raw probability, not a calibrated confidence score; it can be overconfident, a known property of neural classifiers without temperature scaling or calibration.
- **French-only.** Both the ASR model and the sentiment classifier's training data are French-specific; other languages will transcribe and classify poorly, if at all.
