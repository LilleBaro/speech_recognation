from pathlib import Path

import torch
import torchaudio
from torchaudio.transforms import Resample

def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate

def convert_to_mono(waveform):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def audio_resampling(waveform, orig_freq, new_freq=16000):
    waveform = convert_to_mono(waveform)
    resampler = Resample(orig_freq=orig_freq, new_freq=new_freq, dtype=torch.float32)
    resampled_audio = resampler(waveform)
    return resampled_audio, new_freq

def normalize_audio(sample):
    return sample / torch.max(torch.abs(sample))

def preprocess_audio(audio):
    waveform, sample_rate = audio
    waveform = convert_to_mono(waveform)
    resampled_audio, resampled_sample_rate = audio_resampling(waveform, sample_rate)
    normalized_audio = normalize_audio(resampled_audio)
    return normalized_audio, resampled_sample_rate

if __name__ == "__main__":
    sample_path = Path(__file__).resolve().parents[1] / "notebooks" / "sample.mp3"
    if not sample_path.exists():
        sample_path = Path(__file__).resolve().parents[1] / "models" / "sample.mp3"

    if sample_path.exists():
        sample = load_audio(str(sample_path))
        processed_audio = preprocess_audio(sample)
        print(processed_audio)
    else:
        print("No sample audio file found.")