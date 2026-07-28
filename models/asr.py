from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from preprocessing.audio_preprocessing import preprocess_audio, load_audio, convert_to_mono, audio_resampling, normalize_audio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
lang = "fr"
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

def transcribe_audio(waveform, sample_rate):
    if waveform.dim() > 1 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0) # Remove the channel dimension if it's 1
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]
def main():
    audio_file_path = PROJECT_ROOT / "models" / "sample.mp3"
    if not audio_file_path.exists():
        audio_file_path = PROJECT_ROOT / "notebooks" / "sample.mp3"

    print(f"Loading audio file: {audio_file_path}")

    loaded_waveform, loaded_sample_rate = load_audio(audio_file_path)

    # Preprocess the loaded audio
    waveform, sample_rate = preprocess_audio((loaded_waveform, loaded_sample_rate))

    print(f"Original sample rate (after resampling): {sample_rate} Hz")
    print(f"Waveform shape (after preprocessing): {waveform.shape}")

    # Transcribe the audio
    transcription = transcribe_audio(waveform, sample_rate)

    print("\n--- Transcription Result ---")
    print(transcription)

if __name__ == "__main__":
    main()