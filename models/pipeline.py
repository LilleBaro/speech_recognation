from pathlib import Path

from preprocessing.audio_preprocessing import preprocess_audio, load_audio
from .sentiment import BertClassifier, predict as predict_sentiment
from .asr import transcribe_audio
from transformers import AutoTokenizer
import torch

MODEL_NAME = "bert-base-uncased"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "mon_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def predict(audio_path):
    """Pipeline complet : Audio -> Prétraitement -> ASR -> Sentiment"""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"checkpoint file not found: {CHECKPOINT_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertClassifier(MODEL_NAME, n_classes=3)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
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

