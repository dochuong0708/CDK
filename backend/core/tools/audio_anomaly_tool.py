"""Reuse the ASVspoof CNN without changing its trained preprocessing."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import soundfile as sf
import librosa
import numpy as np

from app.detectors.audio.inference import load_checkpoint, predict
from core.tools.audio_features_tool import decode_audio


class AudioAnomalyTool:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("DEEPGUARD_AUDIO_MODEL", "models/audio_cnn.pt")
        self.model = None
        self.lock = Lock()

    def __call__(self, path: Path, *, explain=True) -> dict:
        y, metadata = decode_audio(path)
        flags = ["clipping"] if metadata["clippingFraction"] > .01 else []
        power = np.abs(librosa.stft(y, n_fft=512, hop_length=160)) ** 2
        high_ratio = float(power[192:].sum() / max(float(power.sum()), 1e-12))
        if high_ratio > .35:
            flags.append("high_energy_above_6kHz")
        base = dict(signalFlags=flags, signalMetadata=metadata,
                    frequencyDiagnostics=dict(highFrequencyEnergyRatio=high_ratio, bandHz=[6000, 8000],
                                              heuristicThreshold=.35, validatedForSpoofDetection=False),
                    warning="Clipping/bất thường tín hiệu không chứng minh giả mạo. Điểm CNN chưa được hiệu chuẩn.")
        if not Path(self.model_path).is_file():
            return dict(status="unavailable", modelStatus="missing", fakeScore=None, **base)
        with self.lock:
            if self.model is None:
                self.model, self.threshold = load_checkpoint(self.model_path)
            # Preserve original waveform preprocessing for formats the existing model supports.
            # MP3 is decoded to float WAV, then goes through that same existing preprocessing.
            if path.suffix.lower() == ".mp3":
                with TemporaryDirectory(prefix="audio-tool-") as directory:
                    wav = Path(directory) / "decoded.wav"
                    original, info = decode_audio(path, target_rate=None)
                    sf.write(wav, original, info["originalSampleRate"], subtype="FLOAT")
                    result = predict(wav, self.model, self.threshold, explain=explain)
            else:
                result = predict(path, self.model, self.threshold, explain=explain)
        return dict(status="ok", modelStatus="unvalidated", **base, **result)
