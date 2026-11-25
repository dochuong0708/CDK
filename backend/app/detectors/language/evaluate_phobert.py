"""Evaluate a fine-tuned PhoBERT model on an independent test CSV."""

import argparse
import csv
from pathlib import Path

from app.detectors.language.preprocessing import preprocess_text
from app.detectors.language.train import load_dataset, resolve_label_sets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/models/phobert-fraud-originalsplit"),
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=Path("../data/TEXT/Fake_news_data/fix_test_data.csv"),
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path("../data/TEXT/Fake_news_data/update_train_data.csv"),
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=Path("../data/TEXT/Fake_news_data/update_val_data.csv"),
    )
    parser.add_argument("--text-column", default="Maintext")
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--risky-label", action="append", default=None)
    parser.add_argument("--safe-label", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/reports/phobert_originalsplit_test_predictions.csv"
        ),
    )

    return parser


def load_unique_rows(
    csv_path: Path,
    text_column: str,
    label_column: str,
    risky_labels: set[str],
    safe_labels: set[str],
) -> tuple[list[tuple[str, str]], set[str]]:
    texts, labels = load_dataset(
        csv_path,
        text_column,
        label_column,
        risky_labels,
        safe_labels,
    )

    unique_rows: dict[str, str] = {}
    conflicts: set[str] = set()

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


def main() -> None:
    args = build_parser().parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
        )
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
    except ImportError as error:
        raise SystemExit(
            "Evaluation requires torch, transformers, and scikit-learn."
        ) from error

    if not args.model.exists():
        raise SystemExit(f"Model directory does not exist: {args.model}")

    if not args.test_dataset.exists():
        raise SystemExit(
            f"Test dataset does not exist: {args.test_dataset}"
        )

    risky_labels, safe_labels = resolve_label_sets(
        args.risky_label or ["0"],
        args.safe_label or ["1"],
    )

    # Load the three original splits using the same CSV loader.
    train_rows, _ = load_unique_rows(
        args.train_dataset,
        args.text_column,
        args.label_column,
        risky_labels,
        safe_labels,
    )

    validation_rows, _ = load_unique_rows(
        args.validation_dataset,
        args.text_column,
        args.label_column,
        risky_labels,
        safe_labels,
    )

    test_rows, test_conflicts = load_unique_rows(
        args.test_dataset,
        args.text_column,
        args.label_column,
        risky_labels,
        safe_labels,
    )

    if not test_rows:
        raise SystemExit("No valid test samples were found.")

    # Verify that the test set remains independent.
    train_texts = {text for text, _ in train_rows}
    validation_texts = {text for text, _ in validation_rows}
    test_texts = {text for text, _ in test_rows}

    train_overlap = test_texts & train_texts
    validation_overlap = test_texts & validation_texts

    if train_overlap or validation_overlap:
        raise SystemExit(
            "Test data overlaps with training or validation data. "
            f"Train overlap: {len(train_overlap)}; "
            f"validation overlap: {len(validation_overlap)}."
        )

    true_labels = [
        1 if label == "risky" else 0
        for _, label in test_rows
    ]

    if len(set(true_labels)) != 2:
        raise SystemExit(
            "Test dataset must contain both safe and risky labels."
        )

    print("=" * 65)
    print("PHOBERT INDEPENDENT TEST EVALUATION")
    print("=" * 65)
    print(f"Model: {args.model}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Training samples checked: {len(train_rows)}")
    print(f"Validation samples checked: {len(validation_rows)}")
    print(f"Test samples evaluated: {len(test_rows)}")
    print(f"Test conflicting duplicates removed: {len(test_conflicts)}")
    print("Train-test overlap: 0")
    print("Validation-test overlap: 0")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model)
    )

    # Ensure the model uses the expected label mapping.
    expected_id2label = {
        0: "safe",
        1: "risky",
    }

    actual_id2label = {
        int(key): value.casefold()
        for key, value in model.config.id2label.items()
    }

    if actual_id2label != expected_id2label:
        raise SystemExit(
            "Unexpected model label mapping. "
            f"Expected {expected_id2label}, got {actual_id2label}."
        )

    class TestDataset(Dataset):
        def __init__(self, examples: list[str]):
            self.encodings = [
                tokenizer(
                    " ".join(preprocess_text(text)["tokens"]),
                    truncation=True,
                    max_length=args.max_length,
                )
                for text in examples
            ]

        def __len__(self) -> int:
            return len(self.encodings)

        def __getitem__(self, index: int) -> dict:
            return self.encodings[index]

    test_text_list = [text for text, _ in test_rows]

    test_dataset = TestDataset(test_text_list)

    data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    predicted_labels: list[int] = []
    risky_probabilities: list[float] = []

    with torch.inference_mode():
        for batch in data_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            probabilities = torch.softmax(outputs.logits, dim=-1)

            predictions = torch.argmax(
                probabilities,
                dim=-1,
            )

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

            risky_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )

    # Calculate independent test metrics.
    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["safe", "risky"],
        digits=4,
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")
    print(report)

    print("CONFUSION MATRIX")
    print("Rows: actual labels | Columns: predicted labels")
    print("Label order: safe, risky")
    print(matrix)

    print(f"\nTEST ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Save individual predictions for error analysis.
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "text",
                "true_label",
                "predicted_label",
                "risky_probability",
                "correct",
            ],
        )

        writer.writeheader()

        for text, true_id, predicted_id, risky_probability in zip(
            test_text_list,
            true_labels,
            predicted_labels,
            risky_probabilities,
        ):
            writer.writerow(
                {
                    "text": text,
                    "true_label": "risky" if true_id == 1 else "safe",
                    "predicted_label": (
                        "risky" if predicted_id == 1 else "safe"
                    ),
                    "risky_probability": f"{risky_probability:.6f}",
                    "correct": true_id == predicted_id,
                }
            )

    print(f"\nPrediction file saved to: {args.output}")
    print("Evaluation completed. Model weights were not modified.")


if __name__ == "__main__":
    main()