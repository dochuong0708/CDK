"""Fine-tune PhoBERT on explicitly labeled safe/risky Vietnamese CSV datasets."""

import argparse
from pathlib import Path
from typing import Any

from app.detectors.language.preprocessing import preprocess_text
from app.detectors.language.train import load_dataset, resolve_label_sets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--validation-dataset", action="append", type=Path, default=[])
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    parser.add_argument("--risky-label", action="append", default=[])
    parser.add_argument("--safe-label", action="append", default=[])
    parser.add_argument("--model", default="vinai/phobert-base-v2")
    parser.add_argument("--output", type=Path, default=Path("data/models/phobert-fraud-finetuned"))
    parser.add_argument("--min-samples", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise SystemExit(
            "Fine-tuning requires torch, transformers, accelerate, and scikit-learn."
        ) from error

    
    risky_labels, safe_labels = resolve_label_sets(args.risky_label, args.safe_label)

    def collect_unique_rows(dataset_paths: list[Path]) -> tuple[list[tuple[str, str]], set[str]]:
        unique_rows: dict[str, str] = {}
        conflicts: set[str] = set()

        for dataset_path in dataset_paths:
            texts, labels = load_dataset(
                dataset_path,
                args.text_column,
                args.label_column,
                risky_labels,
                safe_labels,
            )

            for text, label in zip(texts, labels):
                if text in unique_rows and unique_rows[text] != label:
                    conflicts.add(text)
                else:
                    unique_rows[text] = label

        rows = [
            (text, label)
            for text, label in unique_rows.items()
            if text not in conflicts
        ]
        return rows, conflicts

    # Deduplicate training data independently.
    train_rows, train_conflicts = collect_unique_rows(args.dataset)

    if not args.min_samples <= len(train_rows) <= args.max_samples:
        raise SystemExit(
            f"After training deduplication/conflict removal found {len(train_rows)} "
            f"samples; expected {args.min_samples}-{args.max_samples}."
        )

    if args.validation_dataset:
        # Keep validation data completely separate from training data.
        validation_rows, validation_conflicts = collect_unique_rows(
            args.validation_dataset
        )

        if not validation_rows:
            raise SystemExit("Validation data contains no valid samples.")

        train_text_set = {text for text, _ in train_rows}
        validation_text_set = {text for text, _ in validation_rows}
        overlapping_texts = train_text_set & validation_text_set

        if overlapping_texts:
            raise SystemExit(
                f"Found {len(overlapping_texts)} duplicate texts between "
                "training and validation. Resolve the overlap before training."
            )

        if len({label for _, label in validation_rows}) != 2:
            raise SystemExit(
                "Validation data must contain both safe and risky labels."
            )

        print("Validation mode: independent dataset")
        print(f"Training samples: {len(train_rows)}")
        print(f"Training conflicts removed: {len(train_conflicts)}")
        print(f"Validation samples: {len(validation_rows)}")
        print(f"Validation conflicts removed: {len(validation_conflicts)}")

    else:
        # Preserve the original internal 85/15 split as fallback.
        validation_rows = []

        print("Validation mode: internal stratified 85/15 split")

    texts = [text for text, _ in train_rows]
    labels = [1 if label == "risky" else 0 for _, label in train_rows]

    if len(set(labels)) != 2:
        raise SystemExit("Training data must contain both safe and risky labels.")

    if args.validation_dataset:
        train_texts = texts
        train_labels = labels

        validation_texts = [text for text, _ in validation_rows]
        validation_labels = [
            1 if label == "risky" else 0
            for _, label in validation_rows
        ]
    else:
        train_texts, validation_texts, train_labels, validation_labels = (
            train_test_split(
                texts,
                labels,
                test_size=0.15,
                random_state=args.seed,
                stratify=labels,
            )
        )


    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, examples: list[str], targets: list[int]):
            self.encodings = [
                tokenizer(
                    " ".join(preprocess_text(text)["tokens"]),
                    truncation=True,
                    max_length=args.max_length,
                )
                for text in examples
            ]
            self.targets = targets

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {**self.encodings[index], "labels": self.targets[index]}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "safe", 1: "risky"},
        label2id={"safe": 0, "risky": 1},
    )

    def metrics(prediction: Any) -> dict[str, float]:
        predicted = np.argmax(prediction.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            prediction.label_ids, predicted, average="binary", zero_division=0
        )
        return {
            "accuracy": accuracy_score(prediction.label_ids, predicted),
            "precision_risky": float(precision),
            "recall_risky": float(recall),
            "f1_risky": float(f1),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_risky",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TextDataset(train_texts, train_labels),
        eval_dataset=TextDataset(validation_texts, validation_labels),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    evaluation = trainer.evaluate()

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(validation_texts)}")
    print("Final validation metrics:")
    print(evaluation)


if __name__ == "__main__":
    main()