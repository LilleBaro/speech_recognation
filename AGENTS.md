# AI Coding Agent Instructions

## Purpose
This project is a speech recognition and sentiment analysis prototype built with FastAPI, Gradio, PyTorch, and Hugging Face Transformers. The repo currently has several incomplete components and inconsistent module imports, so agents should focus on making the application runnable and coherent.

## Key files
- `api.py`: FastAPI backend entrypoint that should accept audio uploads, validate audio format, save files, and forward them to the speech+sentiment pipeline.
- `app.py`: Gradio demo stub with a placeholder `greet()` function. It is not the main speech pipeline.
- `models.py`: experimental Wav2Vec2 ASR code and processor setup for French transcription.
- `models/pipeline.py`: end-to-end pipeline that should glue audio preprocessing, ASR, and sentiment classification.
- `models/asr.py`: speech-to-text model and transcription logic.
- `models/sentiment.py`: BERT classifier and sentiment prediction helper.
- `preprocessing/audio_preprocessing.py`: audio loading and preprocessing utilities.

## What agents should do first
- Make the app runnable by fixing import paths and package structure.
- Connect `api.py` to the pipeline so uploaded audio is transcribed and analyzed.
- Ensure the pipeline uses the actual model files and correct path names.
- Add or correct error handling for unsupported formats and missing files.
- Keep the interface simple: upload audio, transcribe, classify sentiment, return JSON.

## Important notes
- The repository has no meaningful `README.md`; do not rely on it for behavior or requirements.
- `utils/helpers.py` is currently empty in the workspace and should not be assumed to contain shared logic.
- Some imports use inconsistent package paths, e.g. `from asr import transcribe_audio` inside `models/pipeline.py`; verify actual module locations and use package imports consistently.
- `pyproject.toml` requires Python `>=3.12` and lists dependencies such as `fastapi`, `gradio`, `torch`, `transformers`, and `torchaudio`.
- The model checkpoint file appears as `mon_checkpoint (1).pth` under `models/`; verify that code references match the actual filename.

## Development commands
- Start the FastAPI backend: `uvicorn api:app --reload` from the repo root.
- Run the Gradio demo: `python app.py` from the repo root.
- Use the Python package imports from the repo root; the project is not currently a well-defined installable package.

## Agent behavior guidance
- Prefer small, incremental fixes that preserve the current architecture.
- Avoid introducing large new frameworks or unrelated features.
- When possible, keep the existing FastAPI + Gradio prototypes rather than replacing them.
- Document any assumptions in code comments or a brief repo note if the current project intent is unclear.
