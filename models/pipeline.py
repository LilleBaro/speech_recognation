from pathlib import Path

from preprocessing.audio_preprocessing import preprocess_audio, load_audio
from .sentiment import BertClassifier, predict as predict_sentiment
from .asr import transcribe_audio
from transformers import AutoTokenizer
import torch
import os
from huggingface_hub import hf_hub_download

MODEL_NAME = "bert-base-uncased"
CHECKPOINT_FILENAME = "mon_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Hugging Face repo hosting the fine-tuned checkpoint + tokenizer (override with env HF_REPO_ID)
HF_REPO_ID = os.environ.get("HF_REPO_ID", "LilleBaro/fr_sentiment_analysis")


def predict(audio_path):
    """Complete pipeline  : Audio -> Preprocessing -> ASR -> Prediction"""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, subfolder="tokenizer")

    model = BertClassifier(MODEL_NAME, n_classes=3)
    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CHECKPOINT_FILENAME)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)

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

