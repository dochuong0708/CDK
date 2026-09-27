"""Run from backend: python audio_cli.py --help."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from asvspoof_data import read_manifest as manifest, discover_partitions, prepare_manifest, audit_manifest

from app.detectors.audio.inference import load_checkpoint, predict
from app.detectors.audio.model import AudioCNN
from app.detectors.audio.pipeline import CONFIG, SAMPLE_RATE, SECONDS, load_audio, spectrogram


class AudioDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        y = load_audio(row["path"])
        width = SAMPLE_RATE * SECONDS
        start = random.randint(0, max(0, len(y)-width))
        y = y[start:start+width]
        y = np.pad(y, (0, width-len(y)))
        return spectrogram(y), int(row["label"] == "fake")


def scores(model, rows):
    labels, probabilities = [], []
    for index, row in enumerate(rows, 1):
        if index % 500 == 0:
            print(f"Scoring {index}/{len(rows)} files", flush=True)
        result = predict(row["path"], model, .5, explain=False)
        labels.append(int(row["label"] == "fake"))
        probabilities.append(result["fakeScore"])
    return np.array(labels), np.array(probabilities)


def metrics(labels, probabilities, threshold):
    if len(set(labels)) != 2:
        raise ValueError("Mỗi tập đánh giá phải có cả real và fake.")
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    index = int(np.argmin(np.abs(fpr - (1-tpr))))
    prediction = probabilities >= threshold
    return dict(accuracy=float(accuracy_score(labels, prediction)),
                f1_fake=float(f1_score(labels, prediction, zero_division=0)),
                roc_auc=float(roc_auc_score(labels, probabilities)),
                eer_approx=float((fpr[index]+1-tpr[index])/2),
                threshold=float(threshold), confusion_matrix=confusion_matrix(labels, prediction).tolist(),
                confusion_order=["real", "fake"], samples=len(labels))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = manifest(args.manifest)
    training = [r for r in rows if r["split"] == "train"]
    development = [r for r in rows if r["split"] == "dev"]
    for partition in [training, development]:
        if {r["label"] for r in partition} != {"real", "fake"}:
            raise ValueError("Train và dev phải có cả real và fake.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA không khả dụng. Dùng --device cpu hoặc cài đúng PyTorch CUDA.")
    if args.output.exists():
        raise ValueError(f"Checkpoint đã tồn tại; chọn --output mới: {args.output}")
    device = torch.device(args.device)
    model = AudioCNN().to(device)
    counts = np.bincount([int(r["label"] == "fake") for r in training], minlength=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(training)/(2*counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(AudioDataset(training), batch_size=args.batch_size, shuffle=True, num_workers=0)
    history, best = [], float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch, (x, labels) in enumerate(loader, 1):
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if batch % 50 == 0 or batch == len(loader):
                print(f"Epoch {epoch+1}/{args.epochs} | batch {batch}/{len(loader)} | loss={np.mean(losses):.4f}", flush=True)
        model.eval()
        labels, probabilities = scores(model, development)
        report = metrics(labels, probabilities, .5)
        report.update(epoch=epoch+1, train_loss=float(np.mean(losses)))
        history.append(report)
        print(json.dumps(report), flush=True)
        if report["eer_approx"] < best:
            best = report["eer_approx"]
            fpr, tpr, thresholds = roc_curve(labels, probabilities)
            finite = np.flatnonzero(np.isfinite(thresholds) & (thresholds > 0) & (thresholds < 1))
            threshold = float(thresholds[finite[np.argmin(np.abs(fpr[finite]-(1-tpr[finite])))]] ) if len(finite) else .5
            torch.save(dict(architecture="audio-cnn-v1", state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=CONFIG,
                            classes=["real", "fake"], threshold=threshold, seed=args.seed,
                            dev_metrics=metrics(labels, probabilities, threshold)), args.output)
        args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def evaluate(args):
    rows = [r for r in manifest(args.manifest) if r["split"] == "test"]
    model, threshold = load_checkpoint(args.model)
    labels, probabilities = scores(model, rows)
    report = metrics(labels, probabilities, threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.output.with_suffix(".scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "fake_score"])
        writer.writerows((str(r["path"]), r["label"], float(p)) for r, p in zip(rows, probabilities))
    print(json.dumps(report, indent=2))


def infer(args):
    model, threshold = load_checkpoint(args.model)
    result = predict(args.audio, model, threshold)
    import base64
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "xai.png").write_bytes(base64.b64decode(result.pop("xaiPngBase64")))
    (args.output / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def prepare(args):
    partitions = discover_partitions(args.root, args.splits) if args.root else args.partition
    report = prepare_manifest(partitions, args.output, args.limit_per_class, args.seed)
    print(json.dumps(report, indent=2))


def validate(args):
    print(json.dumps(audit_manifest(args.manifest, args.output, args.decode_audio), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-asvspoof2019")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--partition", nargs=3, action="append", metavar=("SPLIT", "PROTOCOL", "FLAC_DIR"))
    source.add_argument("--root", type=Path, help="Extracted ASVspoof 2019 LA directory")
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    p.add_argument("--limit-per-class", type=int, help="Small deterministic subset per split/class, for smoke runs only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("validate-manifest")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/data_audit.json"))
    p.add_argument("--decode-audio", action="store_true", help="Decode every file using training preprocessing")
    p.set_defaults(func=validate)
    p = sub.add_parser("train")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("models/audio_cnn.pt"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=train)
    p = sub.add_parser("evaluate")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/test_metrics.json"))
    p.set_defaults(func=evaluate)
    p = sub.add_parser("predict")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/prediction"))
    p.set_defaults(func=infer)
    args = parser.parse_args()
    if args.command == "train" and (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0):
        parser.error("epochs, batch-size và lr phải lớn hơn 0")
    torch.set_num_threads(2)
    try:
        args.func(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
