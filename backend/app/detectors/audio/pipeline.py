"""Shared deterministic preprocessing for training, evaluation and inference."""
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
SECONDS = 4
N_FFT = 512
HOP = 160
N_MELS = 80
CONFIG = dict(sample_rate=SAMPLE_RATE, seconds=SECONDS, n_fft=N_FFT, hop=HOP,
              n_mels=N_MELS, normalization="fixed_db_80_v1")


def load_audio(path: Path) -> np.ndarray:
    try:
        with sf.SoundFile(path) as f:
            if f.format not in {"WAV", "FLAC", "OGG"}:
                raise ValueError("Chỉ hỗ trợ WAV, FLAC hoặc OGG.")
            if not 8000 <= f.samplerate <= 192000 or not 1 <= f.channels <= 8:
                raise ValueError("Tần số lấy mẫu hoặc số kênh không được hỗ trợ.")
            duration = len(f) / f.samplerate
            if not 0.5 <= duration <= 60:
                raise ValueError("Âm thanh phải dài từ 0,5 đến 60 giây.")
            sr = f.samplerate
            y = f.read(dtype="float32", always_2d=True).mean(axis=1)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError("Không đọc được âm thanh. Hãy dùng WAV, FLAC hoặc OGG hợp lệ.") from exc
    if not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-5:
        raise ValueError("Âm thanh trống, im lặng hoặc có giá trị không hợp lệ.")
    divisor = gcd(sr, SAMPLE_RATE)
    y = resample_poly(y, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    # Peak normalization is identical in all stages; no inference-only denoising.
    return y / max(float(np.max(np.abs(y))), 1e-8)


def segments(y: np.ndarray):
    width = SAMPLE_RATE * SECONDS
    for start in range(0, len(y), width):
        clip = y[start:start + width]
        yield start / SAMPLE_RATE, len(clip) / SAMPLE_RATE, np.pad(clip, (0, width-len(clip)))


def mel_frequencies():
    high = 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700)
    return 700 * (10 ** (np.linspace(0, high, N_MELS + 2) / 2595) - 1)


def spectrogram(y: np.ndarray) -> torch.Tensor:
    frequencies = mel_frequencies()
    bins = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    filters = np.maximum(0, np.minimum(
        (bins[None, :] - frequencies[:-2, None]) / np.diff(frequencies)[:-1, None],
        (frequencies[2:, None] - bins[None, :]) / np.diff(frequencies)[1:, None]))
    x = torch.as_tensor(y, dtype=torch.float32)
    power = torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT),
                       return_complex=True).abs().square()
    mel = torch.as_tensor(filters, dtype=torch.float32) @ power
    db = 10 * torch.log10(mel.clamp_min(1e-10))
    db = (db - db.max()).clamp(-80, 0)
    return ((db + 80) / 80).unsqueeze(0)
