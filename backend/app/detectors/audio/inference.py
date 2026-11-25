import base64
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from matplotlib.figure import Figure

from .model import AudioCNN, grad_cam
from .pipeline import CONFIG, HOP, SAMPLE_RATE, mel_frequencies, load_audio, segments, spectrogram

PLOT_LOCK = Lock()


def load_checkpoint(path):
    # Load only locally configured, trusted checkpoint paths; never an uploaded model.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("config") != CONFIG or checkpoint.get("classes") != ["real", "fake"]
            or checkpoint.get("architecture") != "audio-cnn-v1"):
        raise ValueError("Checkpoint không tương thích với pipeline audio-cnn-v1.")
    threshold = float(checkpoint["threshold"])
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Ngưỡng checkpoint không hợp lệ.")
    model = AudioCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, threshold


def plot_explanation(y, spec, cam, start, valid, probability):
    with PLOT_LOCK:
        fig = Figure(figsize=(11, 8), layout="constrained")
        axes = fig.subplots(3, 1)
        # Min/max envelope preserves peaks; simple decimation aliases high frequencies.
        stride = max(1, int(np.ceil(len(y) / 4000)))
        blocks = [y[i:i+stride] for i in range(0, len(y), stride)]
        t = np.arange(len(blocks)) * stride / SAMPLE_RATE
        axes[0].fill_between(t, [b.min() for b in blocks], [b.max() for b in blocks],
                             linewidth=0.3, color="#087b74")
        axes[0].axvspan(start, start + valid, alpha=.18, color="orange")
        axes[0].set(title="Waveform envelope | highlighted window explained below", xlabel="Time (s)", ylabel="Amplitude")
        frames = min(spec.shape[1], int(np.ceil(valid * SAMPLE_RATE / HOP)))
        times = start + np.arange(frames) * HOP / SAMPLE_RATE
        frequencies = mel_frequencies()[1:-1]
        db = spec[:, :frames] * 80 - 80
        for ax in axes[1:]:
            mesh = ax.pcolormesh(times, frequencies, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
            ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, 8000))
        fig.colorbar(mesh, ax=axes[1], label="Relative energy (dB)")
        axes[1].set_title("Mel spectrogram | 80 bands, 16 kHz audio")
        overlay = axes[2].pcolormesh(times, frequencies, cam[:, :frames], shading="auto",
                                     cmap="YlGnBu", vmin=0, vmax=1, alpha=.6)
        fig.colorbar(overlay, ax=axes[2], label="Relative attribution")
        axes[2].set_title(f"Grad-CAM for FAKE class | window score={probability:.3f}")
        fig.suptitle("Model attribution is not proof of manipulation", fontsize=13)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def predict(path: Path, model, threshold, explain=True):
    y = load_audio(path)
    windows = []
    specs = []
    for start, valid, clip in segments(y):
        x = spectrogram(clip).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            probability = float(model(x).softmax(dim=1)[0, 1])
        windows.append(dict(start=start, end=start + valid, fakeScore=probability))
        specs.append(x)
    # Duration weighting prevents a short zero-padded tail dominating a file.
    score = float(np.average([w["fakeScore"] for w in windows],
                             weights=[w["end"]-w["start"] for w in windows]))
    result = dict(fakeScore=score, threshold=threshold, predictedClass="fake" if score >= threshold else "real",
                  duration=len(y)/SAMPLE_RATE, sampleRate=SAMPLE_RATE, windows=windows,
                  scoreMeaning="Uncalibrated model score; not probability of factual authenticity")
    if explain:
        chosen = int(np.argmax([w["fakeScore"] for w in windows]))
        window = windows[chosen]
        _, cam = grad_cam(model, specs[chosen], target=1)
        result.update(explainedWindow=chosen, xaiTarget="fake", xaiMethod="Grad-CAM",
                      xaiNote="Window with highest fake score; heatmap is relative attribution, not localization ground truth.",
                      xaiPngBase64=plot_explanation(y, specs[chosen][0, 0].cpu().numpy(), cam,
                                                  window["start"], window["end"]-window["start"], window["fakeScore"]))
    return result
