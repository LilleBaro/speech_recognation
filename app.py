import os
from pathlib import Path

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"


def transcribe_and_analyze(audio_path):
    if not audio_path:
        return "", "", None

    filename = Path(audio_path).name
    try:
        with open(audio_path, "rb") as f:
            response = httpx.post(
                PREDICT_ENDPOINT,
                files={"file": (filename, f, "audio/wav")},
                timeout=120.0,
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", exc.response.text)
        raise gr.Error(f"Backend error ({exc.response.status_code}): {detail}")
    except httpx.RequestError as exc:
        raise gr.Error(f"Could not reach backend at {PREDICT_ENDPOINT}: {exc}")

    result = response.json()
    return result["transcription"], result["sentiment"], result["confidence"]


demo = gr.Interface(
    fn=transcribe_and_analyze,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio (wav/mp3)"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Sentiment"),
        gr.Number(label="Confidence"),
    ],
    title="Speech Sentiment Analysis",
    description=f"Transcribes French audio and predicts sentiment",
    api_name="predict",
)
demo.launch(theme=gr.Theme.from_hub("Yntec/YntecDarkTheme"))

if __name__ == "__main__":
    demo.launch()
