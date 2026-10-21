"""Fine-tune model kiểm chứng claim (NLI): (claim, evidence) -> SUPPORTED/REFUTED/NEI.

Đầu vào là các file claims_{train,val,test}.csv do data_pipeline tạo ra.
Mặc định dùng vinai/phobert-base (giấy phép MIT).

    python -m app.detectors.language.train_nli
    python -m app.detectors.language.train_nli --evaluate-only data/models/nli-phobert
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.detectors.language.data_pipeline import CLAIM_LABELS, DATASET_LICENSES, DEFAULT_OUT_DIR

BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "vinai/phobert-base"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "data" / "models" / "nli-phobert"
MODEL_LICENSES = {
    "vinai/phobert-base": "MIT",
    "vinai/phobert-large": "MIT",
    "FacebookAI/xlm-roberta-base": "MIT",
    "microsoft/mdeberta-v3-base": "MIT",
}
LABEL2ID = {label: index for index, label in enumerate(CLAIM_LABELS)}
ID2LABEL = dict(enumerate(CLAIM_LABELS))
REQUIRED_COLUMNS = ("claim", "evidence", "label", "dataset")


def load_claims(
    path: Path, max_samples: int | None = None, seed: int = 42
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
        rows = []
        for row in reader:
            if row["label"] not in LABEL2ID:
                raise ValueError(f"{path}: nhãn '{row['label']}' không thuộc {CLAIM_LABELS}")
            rows.append({column: row[column] for column in REQUIRED_COLUMNS})
    if max_samples is not None and len(rows) > max_samples:
        rows = random.Random(seed).sample(rows, max_samples)
    return rows


def needs_word_segmentation(model_name: str) -> bool:
    return "phobert" in model_name.casefold()


def make_segmenter(enabled: bool):
    if not enabled:
        return lambda text: unicodedata.normalize("NFC", text)
    try:
        from pyvi import ViTokenizer
    except ImportError as error:
        raise RuntimeError("PhoBERT cần tách từ: python -m pip install pyvi") from error
    return lambda text: ViTokenizer.tokenize(unicodedata.normalize("NFC", text))


def classification_report(
    gold: Sequence[str], predicted: Sequence[str], datasets: Sequence[str] | None = None
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    def summarize(indices: Sequence[int]) -> dict[str, Any]:
        y_true = [gold[i] for i in indices]
        y_pred = [predicted[i] for i in indices]
        per_label = f1_score(
            y_true, y_pred, labels=list(CLAIM_LABELS), average=None, zero_division=0
        )
        return {
            "n": len(indices),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(
                float(
                    f1_score(
                        y_true, y_pred, labels=list(CLAIM_LABELS), average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "f1_per_label": {
                label: round(float(score), 4) for label, score in zip(CLAIM_LABELS, per_label)
            },
            "confusion_matrix": {
                "labels": list(CLAIM_LABELS),
                "rows_gold_cols_pred": confusion_matrix(
                    y_true, y_pred, labels=list(CLAIM_LABELS)
                ).tolist(),
            },
        }

    report = {"overall": summarize(range(len(gold)))}
    if datasets is not None:
        for name in sorted(set(datasets)):
            report[name] = summarize([i for i, value in enumerate(datasets) if value == name])
    return report


class ClaimPairDataset:
    def __init__(self, rows, tokenizer, segment, max_length: int):
        self.encodings = tokenizer(
            [segment(row["claim"]) for row in rows],
            [segment(row["evidence"]) for row in rows],
            truncation="longest_first",
            max_length=max_length,
        )
        self.labels = [LABEL2ID[row["label"]] for row in rows]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def predict(model, tokenizer, rows, segment, max_length: int, batch_size: int) -> list[str]:
    import torch

    device = next(model.parameters()).device
    model.eval()
    predictions: list[str] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs = tokenizer(
            [segment(row["claim"]) for row in batch],
            [segment(row["evidence"]) for row in batch],
            truncation="longest_first",
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        predictions.extend(ID2LABEL[int(i)] for i in logits.argmax(dim=-1).tolist())
    return predictions


def precision_flags(requested: str) -> dict[str, bool]:
    import torch

    if requested == "fp32" or not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    if requested == "bf16" or (requested == "auto" and torch.cuda.is_bf16_supported()):
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


def evaluate_split(model, tokenizer, rows, segment, args) -> dict[str, Any]:
    predicted = predict(model, tokenizer, rows, segment, args.max_length, args.eval_batch_size)
    return classification_report(
        [row["label"] for row in rows], predicted, [row["dataset"] for row in rows]
    )


def print_report(name: str, report: dict[str, Any]) -> None:
    for scope, metrics in report.items():
        print(
            f"[{name}] {scope:<12} n={metrics['n']:<6} "
            f"acc={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluate-only", type=Path, help="Chỉ đánh giá model đã lưu trên test")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--segment",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Tách từ bằng pyvi (auto: bật cho PhoBERT)",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    test_rows = load_claims(
        args.data_dir / "claims_test.csv", args.max_eval_samples, args.seed
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    if args.evaluate_only:
        config_path = args.evaluate_only / "deepguard_nli.json"
        saved = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        segment = make_segmenter(saved.get("word_segmentation", False))
        tokenizer = AutoTokenizer.from_pretrained(args.evaluate_only)
        model = AutoModelForSequenceClassification.from_pretrained(args.evaluate_only).to(device)
        args.max_length = saved.get("max_length", args.max_length)
        report = evaluate_split(model, tokenizer, test_rows, segment, args)
        print_report("test", report)
        return {"test": report}

    train_rows = load_claims(
        args.data_dir / "claims_train.csv", args.max_train_samples, args.seed
    )
    val_rows = load_claims(args.data_dir / "claims_val.csv", args.max_eval_samples, args.seed)
    segmentation = args.segment == "yes" or (
        args.segment == "auto" and needs_word_segmentation(args.model)
    )
    segment = make_segmenter(segmentation)
    print(
        f"Model: {args.model} | tách từ: {segmentation} | "
        f"train {len(train_rows)} | val {len(val_rows)} | test {len(test_rows)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(CLAIM_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    print("Đang tokenize...", flush=True)
    train_set = ClaimPairDataset(train_rows, tokenizer, segment, args.max_length)
    val_set = ClaimPairDataset(val_rows, tokenizer, segment, args.max_length)

    def compute_metrics(eval_prediction) -> dict[str, float]:
        predicted = [ID2LABEL[int(i)] for i in eval_prediction.predictions.argmax(axis=-1)]
        gold = [ID2LABEL[int(i)] for i in eval_prediction.label_ids]
        overall = classification_report(gold, predicted)["overall"]
        return {"accuracy": overall["accuracy"], "macro_f1": overall["macro_f1"]}

    checkpoints = args.output / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
        report_to=[],
        seed=args.seed,
        **precision_flags(args.precision),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    started = time.time()
    trainer.train()
    train_seconds = round(time.time() - started, 1)

    model = trainer.model
    report = {
        "val": evaluate_split(model, tokenizer, val_rows, segment, args),
        "test": evaluate_split(model, tokenizer, test_rows, segment, args),
    }
    for name, split_report in report.items():
        print_report(name, split_report)

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_model": args.model,
        "base_model_license": MODEL_LICENSES.get(args.model, "chưa xác minh"),
        "training_data": DATASET_LICENSES,
        "labels": list(CLAIM_LABELS),
        "word_segmentation": segmentation,
        "max_length": args.max_length,
        "hyperparameters": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "train_seconds": train_seconds,
        "metrics": report,
    }
    (args.output / "deepguard_nli.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã lưu model vào {args.output}")
    return report


if __name__ == "__main__":
    main()
