from pathlib import Path

import torch
import torchaudio
from torchaudio.transforms import Resample

from exceptions import AudioProcessingError

# Below this peak amplitude, audio is considered silent rather than just quiet.
SILENCE_THRESHOLD = 1e-4

def load_audio(path):
    """
    Load an audio file .

    This function reads an audio file using torchaudio and returns the
    corresponding waveform along with its sampling rate.

    Args:
        path (str): Path to the audio file.

    Returns:
        tuple[torch.Tensor, int]:
            - waveform (torch.Tensor): Audio waveform of shape
              (channels, num_samples).
            - sample_rate (int): Original sampling rate of the audio file.

    Raises:
        AudioProcessingError: If the file is missing, empty, corrupt, or in
            an unsupported/unreadable format.
    """
    try:
        waveform, sample_rate = torchaudio.load(path)
    except Exception as exc:
        raise AudioProcessingError(f"unable to read audio file (corrupt or unsupported format): {exc}") from exc

    if waveform.numel() == 0:
        raise AudioProcessingError("audio file is empty (no samples)")

    return waveform, sample_rate

def convert_to_mono(waveform):
    """
    Convert a multi-channel audio waveform to mono.

    If the input waveform contains more than one channel, the channels are
    averaged to produce a single mono channel. If the waveform is already
    mono, it is returned unchanged.

    Args:
        waveform (torch.Tensor): Audio waveform of shape
            (channels, num_samples).

    Returns:
        torch.Tensor: Mono audio waveform of shape
        (1, num_samples).
    """
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def audio_resampling(waveform, orig_freq, new_freq=16000):
    """
    Resample an audio waveform to a target sampling rate.

    The waveform is first converted to mono (if necessary) and then
    resampled to the specified sampling rate.

    Args:
        waveform (torch.Tensor): Input audio waveform.
        orig_freq (int): Original sampling rate of the audio.
        new_freq (int, optional): Target sampling rate.
            Defaults to 16000 Hz.

    Returns:
        tuple[torch.Tensor, int]:
            - resampled_audio (torch.Tensor): Resampled audio waveform.
            - new_freq (int): Target sampling rate.
    """
    waveform = convert_to_mono(waveform)
    resampler = Resample(orig_freq=orig_freq, new_freq=new_freq, dtype=torch.float32)
    resampled_audio = resampler(waveform)
    return resampled_audio, new_freq

def normalize_audio(sample):
    """
    Normalize an audio waveform.

    The waveform amplitudes are scaled to the range [-1, 1] by dividing
    all samples by the maximum absolute amplitude.

    Args:
        sample (torch.Tensor): Audio waveform.

    Returns:
        torch.Tensor: Normalized audio waveform.

    Raises:
        AudioProcessingError: If the audio is silent (peak amplitude below
            SILENCE_THRESHOLD), since normalizing it would divide by ~zero.
    """
    peak = torch.max(torch.abs(sample))
    if peak < SILENCE_THRESHOLD:
        raise AudioProcessingError("audio is silent (no meaningful signal detected)")
    return sample / peak

def preprocess_audio(audio):
    """
    Apply the complete audio preprocessing pipeline.

    This function performs all preprocessing steps required before passing
    the audio to the speech recognition model:
        1. Convert the waveform to mono.
        2. Resample it to 16 kHz.
        3. Normalize its amplitude.

    Args:
        audio (tuple[torch.Tensor, int]):
            Tuple containing:
                - waveform (torch.Tensor): Input audio waveform.
                - sample_rate (int): Original sampling rate.

    Returns:
        tuple[torch.Tensor, int]:
            - normalized_audio (torch.Tensor): Preprocessed audio waveform.
            - resampled_sample_rate (int): Output sampling rate (16 kHz).
    """
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