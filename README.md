# Speech Recognition & Sentiment Analysis Prototype

This repository contains a prototype for analyzing French audio recordings in two steps:

1. Transcribing speech with a Wav2Vec2-based ASR model.
2. Predicting the sentiment of the transcribed text with a fine-tuned BERT classifier.

The project is currently a work-in-progress prototype with a FastAPI backend and a Gradio-based demo interface.

## Project goals

- Accept an audio file upload in WAV or MP3 format.
- Transcribe the audio to text.
- Classify the transcription as negative, neutral, or positive.
- Return a JSON payload with the transcription, sentiment, and confidence score.

## Architecture overview

The main flow is:

- API layer: [api.py](api.py)
- Processing pipeline: [models/pipeline.py](models/pipeline.py)
- Audio preprocessing: [preprocessing/audio_preprocessing.py](preprocessing/audio_preprocessing.py)
- ASR model: [models/asr.py](models/asr.py)
- Sentiment model: [models/sentiment.py](models/sentiment.py)
- Gradio UI: [app.py](app.py)

## Model description

2 models are used i  this project:

- A Camembert model finetuned with **SetFit/amazon_reviews_multi_fr** dataset from huggingface https://huggingface.co/datasets/SetFit/amazon_reviews_multi_fr, which contains the following variables:
    - id
    - text 
    - label: ratings from 0 to 4
    - label_text: string ratings from 0 to 4
- **jonatasgrosman/wav2vec2-large-xlsr-53-french** specialised in french transcriptions.
  

## Repository structure

- [api.py](api.py) — FastAPI application exposing the `/predict` endpoint.
- [app.py](app.py) — Gradio demo frontend that calls the backend.
- [models/](models/) — ASR, sentiment, and pipeline orchestration code.
- [preprocessing/](preprocessing/) — audio loading and preprocessing utilities.
- [notebooks/](notebooks/) — exploratory notebooks for testing the pipeline.

## Requirements

- Python 3.12+
- Dependencies managed with `uv`

## Setup

From the repository root, install dependencies:

```bash
uv sync
```

## Running the project

### 1. Start the backend

```bash
uv run uvicorn api:app --reload
```

The API will be available at:

- http://127.0.0.1:8000

### 2. Start the Gradio UI

In a second terminal:

```bash
uv run python app.py
```

This UI sends requests to the running FastAPI backend.

## API usage

You can test the backend with a sample audio file using curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/path/to/audio.wav"
```

Expected response shape:

```json
{
  "transcription": "...",
  "sentiment": "positive",
  "confidence": 0.92
}
```

## Environment variables

- `API_URL` — base URL used by the Gradio UI to reach the backend. Default: `http://127.0.0.1:8000`
- `HF_REPO_ID` — override the Hugging Face repository used for the sentiment checkpoint and tokenizer.

## Notes and limitations

- The project is part of my Deep Learning exam
- Commands should be run from the repository root so Python imports resolve correctly.
- The Gradio interface is a basic demo and may need further refinement.

## Validation and exploration

There is no formal test suite yet. The notebooks in [notebooks/](notebooks/) are the main place for ad-hoc experimentation and validation.
