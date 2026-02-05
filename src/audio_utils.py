import io
import wave

import numpy as np
from scipy.signal import resample


def calculate_rms(audio_bytes: bytes) -> float:
    """Calculate the RMS (Root Mean Square) energy of PCM audio data."""
    if not audio_bytes:
        return 0.0
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if len(audio_array) == 0:
        return 0.0
    return np.sqrt(np.mean(audio_array**2))


def pcm_to_wav(
    pcm_data: bytes, channels: int, sample_width: int, frame_rate: int
) -> bytes:
    """Convert raw PCM data to WAV format in memory."""
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(frame_rate)
            wf.writeframes(pcm_data)
        return wav_buffer.getvalue()


def normalize_audio(audio_array: np.ndarray, target_rms: float = 0.2) -> np.ndarray:
    """Normalize audio array to a target RMS loudness."""
    rms = np.sqrt(np.mean(audio_array**2))
    if rms < 0.000001:
        return audio_array
    return audio_array * (target_rms / rms)


def resample_audio(
    audio_array: np.ndarray, original_rate: int, target_rate: int
) -> np.ndarray:
    """Resample audio to the target sample rate."""
    if original_rate == target_rate:
        return audio_array
    num_samples = int(len(audio_array) * (target_rate / original_rate))
    return resample(audio_array, num_samples)
