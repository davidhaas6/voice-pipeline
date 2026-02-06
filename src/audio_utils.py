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


def wav_bytes_to_pcm_and_meta(wav_bytes: bytes):
    """Extract raw PCM and metadata from WAV bytes."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, channels, sample_width, frame_rate


def pcm16_to_float32_mono(pcm: bytes, channels: int) -> np.ndarray:
    """Convert 16-bit PCM to float32 mono array."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    return x


def float32_to_pcm16(x: np.ndarray) -> bytes:
    """Convert float32 array back to 16-bit PCM bytes."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def to_mono_16k_wav(audio_bytes: bytes) -> bytes:
    """
    Accept either WAV container bytes or raw PCM16 stereo/48k.
    Returns 16-bit mono 16kHz WAV bytes optimized for STT.
    """
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        pcm, ch, sw, sr = wav_bytes_to_pcm_and_meta(audio_bytes)
        if sw != 2:
            raise ValueError(f"Expected 16-bit WAV, got sample_width={sw}")
        x = pcm16_to_float32_mono(pcm, ch)
    else:
        # Assume raw PCM16 stereo @ 48k (current configuration)
        ch, sr = 2, 48000
        x = pcm16_to_float32_mono(audio_bytes, ch)

    # Resample to 16k for STT robustness
    target_sr = 16000
    if sr != target_sr:
        num = int(len(x) * (target_sr / sr))
        x = resample(x, num)

    pcm16 = float32_to_pcm16(x)

    return pcm_to_wav(pcm16, channels=1, sample_width=2, frame_rate=target_sr)


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
