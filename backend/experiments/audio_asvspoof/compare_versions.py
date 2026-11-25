"""Compare the committed audio heuristic with ASVspoof ExtraTrees runs.

The historical detector in the selected Git revision classifies every non-empty
file as bonafide. This script evaluates that exact behavior against the same
labels used by the saved and freshly retrained ASVspoof eval subset.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def confusion_and_metrics(labels: list[str], predictions: list[str]):
    tn = sum(y == "bonafide" and p == "bonafide" for y, p in zip(labels, predictions))
    fp = sum(y == "bonafide" and p == "spoof" for y, p in zip(labels, predictions))
    fn = sum(y == "spoof" and p == "bonafide" for y, p in zip(labels, predictions))
    tp = sum(y == "spoof" and p == "spoof" for y, p in zip(labels, predictions))
    total = len(labels)
    recall_bonafide = tn / (tn + fp) if tn + fp else 0.0
    recall_spoof = tp / (tp + fn) if tp + fn else 0.0
    precision_spoof = tp / (tp + fp) if tp + fp else 0.0
    f1_spoof = (2 * precision_spoof * recall_spoof /
                (precision_spoof + recall_spoof)
                if precision_spoof + recall_spoof else 0.0)
    return {
        "samples": total,
        "accuracy": (tn + tp) / total if total else 0.0,
        "balanced_accuracy": (recall_bonafide + recall_spoof) / 2,
        "precision_spoof": precision_spoof,
        "recall_spoof": recall_spoof,
        "f1_spoof": f1_spoof,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_order": ["bonafide", "spoof"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-output", type=Path,
                        default=ROOT / "backend/outputs/audio_official_20261026")
    parser.add_argument("--new-output", type=Path,
                        default=ROOT / "backend/outputs/audio_repro_20261125")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "backend/outputs/audio_repro_20261125")
    args = parser.parse_args()

    old_rows = read_csv(args.old_output / "test_predictions.csv")
    new_rows = read_csv(args.new_output / "test_predictions.csv")
    if len(old_rows) != len(new_rows):
        raise ValueError("Số mẫu giữa hai lần chạy không trùng nhau")
    for index, (old, new) in enumerate(zip(old_rows, new_rows)):
        if (old["id"], old["label"]) != (new["id"], new["label"]):
            raise ValueError(f"Mẫu/nhãn khác nhau ở dòng {index + 1}")

    labels = [row["label"] for row in new_rows]
    heuristic = confusion_and_metrics(labels, ["bonafide"] * len(labels))
    old_run = read_json(args.old_output / "metrics.json")
    new_run = read_json(args.new_output / "metrics.json")
    score_deltas = [
        abs(float(left["spoof_score"]) - float(right["spoof_score"]))
        for left, right in zip(old_rows, new_rows)
    ]
    decisions_match = all(
        left["prediction"] == right["prediction"]
        for left, right in zip(old_rows, new_rows)
    )
    original_scores_match = max(score_deltas, default=0.0) <= 1e-12
    reproducible = (
        old_run["protocol_sha256"] == new_run["protocol_sha256"]
        and old_run["test"] == new_run["test"]
        and original_scores_match
    )
    if not reproducible:
        raise ValueError("Lần chạy mới không khớp protocol/kết quả đã lưu")

    result = {
        "dataset": "ASVspoof 2019 LA eval subset, seed 42",
        "samples": len(labels),
        "label_counts": {label: labels.count(label) for label in ("bonafide", "spoof")},
        "old_git_heuristic": heuristic,
        "saved_extra_trees": old_run["test"],
        "retrained_extra_trees": new_run["test"],
        "retraining_reproduced_saved_run": reproducible,
        "all_predictions_match": decisions_match,
        "max_absolute_score_delta": max(score_deltas, default=0.0),
        "protocol_sha256": new_run["protocol_sha256"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for name, values in (
        ("old_git_heuristic", heuristic),
        ("saved_extra_trees", old_run["test"]),
        ("retrained_extra_trees", new_run["test"]),
    ):
        rows.append({
            "version": name,
            "samples": values["samples"],
            "accuracy": values["accuracy"],
            "balanced_accuracy": values["balanced_accuracy"],
            "precision_spoof": values["precision_spoof"],
            "recall_spoof": values["recall_spoof"],
            "f1_spoof": values["f1_spoof"],
            "roc_auc": values.get("roc_auc", ""),
            "eer_approx": values.get("eer_approx", ""),
            "confusion_matrix": json.dumps(values["confusion_matrix"]),
        })
    with (args.output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
