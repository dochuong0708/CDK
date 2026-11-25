"""Week 6 labelled ASVspoof evaluation of the actual audio tool, without training/tuning."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from core.tools.audio_anomaly_tool import AudioAnomalyTool
from asvspoof_data import discover_partitions, prepare_manifest, read_manifest
from audio_cli import metrics

ROOT = Path(__file__).resolve().parent


def evaluate(root, model, output, per_class, seed):
    if not model.is_file():
        raise ValueError("Checkpoint không tồn tại.")
    if per_class < 1:
        raise ValueError("per-class phải lớn hơn 0.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "manifest.csv"
    prepare_manifest(discover_partitions(root, ["test"]), manifest, per_class, seed)
    rows = read_manifest(manifest)
    tool = AudioAnomalyTool(str(model))
    scored = []
    for i, row in enumerate(rows, 1):
        start = time.perf_counter()
        result = tool(row["path"], explain=False)
        if result["status"] != "ok":
            raise ValueError("Không thể đánh giá model unavailable.")
        scored.append(dict(utterance=row["utterance"], label=row["label"], attack=row["attack"],
                           fake_score=result["fakeScore"], predicted=result["predictedClass"],
                           latency_ms=(time.perf_counter()-start)*1000))
        if i % 20 == 0:
            print(f"Evaluated {i}/{len(rows)}", flush=True)
    labels = np.array([int(r["label"] == "fake") for r in scored])
    probabilities = np.array([r["fake_score"] for r in scored])
    threshold = tool.threshold
    report = metrics(labels, probabilities, threshold)
    attacks = {}
    for attack in sorted({r["attack"] for r in scored if r["label"] == "fake"}):
        group = [r for r in scored if r["attack"] == attack and r["label"] == "fake"]
        attacks[attack] = dict(samples=len(group), false_negative_rate=sum(r["predicted"] == "real" for r in group)/len(group))
    latency = [r["latency_ms"] for r in scored]
    report.update(createdAt=datetime.now(timezone.utc).isoformat(), seed=seed,
                  model=str(model.resolve()), modelSHA256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  manifestSHA256=hashlib.sha256(manifest.read_bytes()).hexdigest(), split="ASVspoof2019 LA eval",
                  classCounts=dict(Counter(r["label"] for r in scored)), perAttack=attacks,
                  latencyMs=dict(median=float(np.median(latency)), p95=float(np.percentile(latency,95))),
                  scorePath="AudioAnomalyTool(explain=False)",
                  thresholdPolicy="Frozen checkpoint threshold; no threshold tuning on eval",
                  scope="Balanced subset diagnostic only; not the full ASVspoof benchmark; XAI disabled for timing",
                  qualityStatus="unvalidated", acceptanceCriteria="Not specified by the workbook; no fabricated accuracy gate")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT.parent / "data/audio/asvspoof2019/LA")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.model is None:
        args.model = Path(json.loads((ROOT / "outputs/asvspoof_demo/latest.json").read_text(encoding="utf-8"))["model"])
    if args.output is None:
        args.output = ROOT / "outputs/week6_eval" / uuid4().hex
    torch.set_num_threads(2)
    evaluate(args.root, args.model, args.output, args.per_class, args.seed)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
