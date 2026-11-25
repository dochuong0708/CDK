# DeepGuard audio detector

The web audio endpoint uses LFCC features and an Extra Trees classifier. The training pipeline and web inference share the same feature implementation: mono audio at 16 kHz, at most the first six seconds, pre-emphasis with α = 0.97, a 25 ms Hann STFT with 10 ms hop and 512-point FFT, 40 linear filter banks, 20 LFCC coefficients with first/second deltas, and mean/std pooling into 120 values.

## Train the model

Download ASVspoof 2019 Logical Access from its official source and extract it so these paths exist:

```text
data/audio/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/
data/audio/asvspoof2019/LA/ASVspoof2019_LA_train/flac/
data/audio/asvspoof2019/LA/ASVspoof2019_LA_dev/flac/
data/audio/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/
```

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-audio.txt
$data = (Resolve-Path .\data\audio\asvspoof2019\LA).Path
.\.venv\Scripts\python.exe .\backend\experiments\audio_asvspoof\run_experiment.py --data $data --output .\backend\outputs\audio_official_20261026 --seed 42
```

The script samples 6,000 train, 2,000 dev, and 4,000 eval utterances with seed 42. It fits Logistic Regression and Extra Trees on train, selects the candidate with the highest dev ROC-AUC, chooses the threshold on dev with Youden J, and evaluates once on eval. The saved bundle is written to `backend/outputs/audio_official_20261026/model.joblib`; `backend/outputs/` is ignored by Git, so download the corpus and train the model locally before running audio inference.

To use a model bundle stored at another path, set `DEEPGUARD_AUDIO_MODEL_PATH` before starting the backend:

```powershell
$env:DEEPGUARD_AUDIO_MODEL_PATH = (Resolve-Path .\backend\outputs\audio_official_20261026\model.joblib).Path
```

## Web inference and XAI

The API converts supported audio to mono 16 kHz, extracts the same 120 LFCC statistics, and compares the Extra Trees spoof score with the frozen dev threshold. The score is not calibrated as a probability. The frontend shows a log filter-bank view and an occlusion attribution map. For each attribution cell, the detector replaces a local time/frequency region with the utterance median and records how the spoof score changes. This is a post-hoc explanation, not causal evidence or ground-truth localization.

If the model bundle is missing, the API returns HTTP 503 with the expected training command. Invalid or undecodable audio returns HTTP 422.

## Reproduce the version comparison

After the initial run, the comparison script can compare the old Git heuristic with saved and retrained Extra Trees results on the same eval IDs:

```powershell
.\.venv\Scripts\python.exe .\backend\experiments\audio_asvspoof\compare_versions.py --old-output .\backend\outputs\audio_official_20261026 --new-output .\backend\outputs\audio_repro_20261125 --output .\backend\outputs\audio_repro_20261125
```

It verifies that protocol hashes, labels, decisions, and eval metrics match before writing the comparison files.
