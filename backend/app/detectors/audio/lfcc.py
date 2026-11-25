"""LFCC feature extraction compatible with the ASVspoof subset experiment."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly, stft

FEATURE_VERSION = "lfcc20_mean_std_delta_6sec_v1"
SAMPLE_RATE = 16_000
MAX_SAMPLES = 96_000


def read_audio(path: Path) -> tuple[np.ndarray, float]:
    try:
        with sf.SoundFile(path) as stream:
            source_rate = stream.samplerate
            duration = len(stream) / source_rate
            audio = stream.read(dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise ValueError("Không giải mã được tệp. Hãy dùng WAV hoặc FLAC.") from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Tệp âm thanh rỗng hoặc chứa dữ liệu không hợp lệ.")
    audio = audio.mean(axis=1)
    if source_rate != SAMPLE_RATE:
        factor = gcd(SAMPLE_RATE, source_rate)
        audio = resample_poly(audio, SAMPLE_RATE // factor, source_rate // factor).astype(np.float32)
    audio = audio[:MAX_SAMPLES]
    if len(audio) < 400 or float(np.max(np.abs(audio))) < 1e-7:
        raise ValueError("Âm thanh quá ngắn hoặc im lặng; cần tối thiểu 25 ms có tín hiệu.")
    return audio.astype(np.float32), min(duration, 6.0)


def log_filterbank(audio: np.ndarray) -> np.ndarray:
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    _, _, spectrum = stft(
        emphasized, fs=SAMPLE_RATE, window="hann", nperseg=400,
        noverlap=240, nfft=512, boundary=None, padded=False,
    )
    power = np.abs(spectrum) ** 2
    bins = np.arange(257)
    edges = np.linspace(0, 256, 42)
    banks = np.stack([
        np.maximum(0, np.minimum(
            (bins - edges[i]) / (edges[i + 1] - edges[i]),
            (edges[i + 2] - bins) / (edges[i + 2] - edges[i + 1]),
        )) for i in range(40)
    ])
    # Keep the float64 bank multiplication/log values through the DCT, matching
    # run_experiment.features exactly; cast only after statistical pooling.
    return np.log(np.maximum(banks @ power, 1e-12))


def features_from_logbank(log_banks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cepstra = dct(log_banks, type=2, axis=0, norm="ortho")[:20]
    delta = np.gradient(cepstra, axis=1)
    delta2 = np.gradient(delta, axis=1)
    vector = np.concatenate([
        stat for block in (cepstra, delta, delta2)
        for stat in (block.mean(axis=1), block.std(axis=1))
    ]).astype(np.float32)
    return vector, cepstra


def extract_features(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    audio, duration = read_audio(path)
    log_banks = log_filterbank(audio)
    vector, _ = features_from_logbank(log_banks)
    return vector, duration, log_banks
