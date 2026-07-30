from pathlib import Path

import os
import torch

from preprocessing.audio_preprocessing import preprocess_audio, load_audio
from .asr import transcribe_audio
from .sentiment import load_model_and_tokenizer, predict as predict_sentiment

MODEL_NAME = os.environ.get("HF_MODEL_NAME", "distilbert-base-multilingual-cased")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Hugging Face repo hosting the fine-tuned checkpoint + tokenizer 
HF_REPO_ID = os.environ.get("HF_REPO_ID", "LilleBaro/DistilBert_sentiment_analysis")


def predict(audio_path):
    """Complete pipeline  : Audio -> Preprocessing -> ASR -> Prediction"""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    model, tokenizer, _ = load_model_and_tokenizer(HF_REPO_ID, device=DEVICE)

    waveform, sample_rate = load_audio(str(audio_path))
    waveform, sample_rate = preprocess_audio((waveform, sample_rate))

    transcription = transcribe_audio(waveform, sample_rate)
    prediction, confidence = predict_sentiment(
        transcription,
        model,
        tokenizer,
        device=DEVICE,
        max_length=256
    )

    return {
        "transcription": transcription,
        "sentiment": LABEL_MAP.get(prediction, str(prediction)),
        "confidence": confidence
    }

