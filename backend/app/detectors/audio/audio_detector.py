from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from app.models.schemas import Analysis
from app.detectors.audio.lfcc import FEATURE_VERSION, extract_features, features_from_logbank


class AudioUnavailable(RuntimeError):
    """Raised when the trained Extra Trees bundle is missing or incompatible."""


def _model_path() -> Path:
    configured = os.getenv("DEEPGUARD_AUDIO_MODEL_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "outputs" / "audio_official_20261026" / "model.joblib"


@lru_cache(maxsize=2)
def _load_bundle(path_text: str, modified: int):
    del modified  # Include mtime in the cache key so replacing the model reloads it.
    bundle = joblib.load(path_text)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError("Model audio không khớp phiên bản đặc trưng LFCC.")
    return bundle


def _predict(bundle, vector: np.ndarray) -> float:
    return float(bundle["model"].predict_proba(vector[None, :])[0, 1])


def _pool_time(matrix: np.ndarray, columns: int) -> np.ndarray:
    chunks = np.array_split(matrix, columns, axis=1)
    return np.stack([chunk.mean(axis=1) if chunk.shape[1] else np.zeros(matrix.shape[0]) for chunk in chunks], axis=1)


def _occlusion_explanation(bundle, log_banks: np.ndarray, baseline_score: float) -> dict:
    """Perturb groups of LFCC filter-bank cells and report score changes."""
    frequency_groups, time_groups = 10, 12
    time_chunks = np.array_split(np.arange(log_banks.shape[1]), time_groups)
    freq_chunks = np.array_split(np.arange(log_banks.shape[0]), frequency_groups)
    attribution = np.zeros((frequency_groups, time_groups), dtype=np.float32)
    perturbations: list[np.ndarray] = []
    for fi, frequencies in enumerate(freq_chunks):
        for ti, frames in enumerate(time_chunks):
            occluded = log_banks.copy()
            # Replace local evidence by each filterbank's utterance median.
            occluded[np.ix_(frequencies, frames)] = np.median(log_banks[frequencies], axis=1)[:, None]
            vector, _ = features_from_logbank(occluded)
            perturbations.append(vector)

    # Batch the occlusion predictions: the Extra Trees estimator is much faster
    # with one matrix call than with hundreds of one-row calls.
    masked_scores = bundle["model"].predict_proba(np.asarray(perturbations))[:, 1]
    attribution[:] = baseline_score - masked_scores.reshape(frequency_groups, time_groups)

    # Keep the actual sign and compact the plot data for the browser.
    spec = _pool_time(log_banks, 60)
    low, high = np.percentile(spec, [5, 95])
    spec = np.clip((spec - low) / max(float(high - low), 1e-6), 0, 1)
    attribution_scale = max(float(np.max(np.abs(attribution))), 1e-6)
    return {
        "method": "LFCC filter-bank occlusion",
        "note": "Giải thích hậu nghiệm: mỗi ô che một vùng thời gian–dải tần rồi đo thay đổi điểm spoof; đây không phải bằng chứng nhân quả.",
        "frequencyBins": 40,
        "timeBins": 60,
        "spectrogram": np.round(spec, 3).tolist(),
        "attribution": np.round(attribution, 4).tolist(),
        "attributionScale": round(attribution_scale, 6),
        "attributionMeaning": "Giá trị dương: vùng đó làm điểm spoof giảm khi bị che (ủng hộ spoof). Giá trị âm: vùng đó làm điểm spoof tăng khi bị che.",
    }


class AudioDetector:
    def analyze(self, path: Path) -> Analysis:
        started = perf_counter()
        model_path = _model_path()
        if not model_path.is_file():
            raise AudioUnavailable(
                f"Không tìm thấy model audio: {model_path}. "
                "Tải ASVspoof 2019 LA và chạy backend/experiments/audio_asvspoof/run_experiment.py."
            )
        bundle = _load_bundle(str(model_path.resolve()), model_path.stat().st_mtime_ns)
        vector, duration, log_banks = extract_features(path)
        score = _predict(bundle, vector)
        threshold = float(bundle["threshold"])
        is_spoof = score >= threshold
        label = "Nghi là giọng giả" if is_spoof else "Nghi là giọng thật"
        xai = _occlusion_explanation(bundle, log_banks, score)
        latency_ms = (perf_counter() - started) * 1000
        return Analysis(
            analysisId=f"audio-{path.stem.lower()}",
            modality="Audio",
            label=label,
            riskScore=round(score * 100),
            # Extra Trees scores are not calibrated probabilities; the UI labels this as a score.
            confidence=0.0,
            evidence=[
                f"Điểm spoof của Extra Trees: {score:.4f}; ngưỡng đã chọn trên tập dev: {threshold:.6f}.",
                f"Phân tích {duration:.2f} giây đầu sau khi chuyển mono 16 kHz; đặc trưng LFCC 120 chiều.",
                "Bản đồ XAI dùng occlusion hậu nghiệm trên vùng filter-bank và thời gian.",
            ],
            detector="lfcc-extra-trees-asvspoof2019-la-v1",
            input=path.name,
            createdAt=__import__("datetime").datetime.now(),
            audioScore=score,
            audioThreshold=threshold,
            audioLatencyMs=round(latency_ms, 2),
            audioXai=xai,
        )
