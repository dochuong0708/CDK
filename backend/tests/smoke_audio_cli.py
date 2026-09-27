"""End-to-end mechanics only: artificial tones are NOT a voice-deepfake benchmark.

Run from backend: python tests/smoke_audio_cli.py
Outputs stay under ignored outputs/smoke; never use its model in production.
"""
import csv
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

root = Path("outputs/smoke") / uuid4().hex[:12]
root.mkdir(parents=True, exist_ok=True)
rows = []
rng = np.random.default_rng(42)
for split_index, split in enumerate(["train", "dev", "test"]):
    for label_index, label in enumerate(["real", "fake"]):
        for example in range(2):
            name = f"{split}_{label}_{example}.wav"
            t = np.arange(16000) / 16000
            hz = 200 + 1700*label_index + 25*example + 10*split_index
            signal = .5*np.sin(2*np.pi*hz*t) + .01*rng.normal(size=len(t))
            sf.write(root/name, signal, 16000)
            rows.append([name, label, f"speaker_{split}_{label}_{example}", split])
with (root/"manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["path", "label", "speaker", "split"])
    writer.writerows(rows)


def run(*arguments):
    subprocess.run([sys.executable, "audio_cli.py", *map(str, arguments)], check=True)


run("train", "--manifest", root/"manifest.csv", "--epochs", 1, "--batch-size", 2,
    "--output", root/"smoke_only.pt")
run("evaluate", "--manifest", root/"manifest.csv", "--model", root/"smoke_only.pt",
    "--output", root/"metrics_NOT_A_BENCHMARK.json")
run("predict", "--audio", root/"test_fake_0.wav", "--model", root/"smoke_only.pt",
    "--output", root/"prediction")
print("PASS: train/evaluate/predict. Synthetic smoke test only; no voice accuracy claim.")
