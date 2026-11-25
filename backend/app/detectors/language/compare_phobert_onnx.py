import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from app.detectors.language.train import (
    load_dataset,
    resolve_label_sets,
)

from app.detectors.language.preprocessing import preprocess_text


BASE_DIR = Path(__file__).resolve().parents[3]

PYTORCH_DIR = BASE_DIR / "data/models/phobert-fraud-originalsplit"
ONNX_DIR = BASE_DIR / "data/models/phobert-onnx-originalsplit"

TEST_CSV = (
    BASE_DIR.parent
    / "data/TEXT/Fake_news_data/fix_test_data.csv"
)

TEXT_COLUMN = "Maintext"
LABEL_COLUMN = "Label"

MAX_LENGTH = 256
BATCH_SIZE = 4


def load_unique_rows():
    risky_labels, safe_labels = resolve_label_sets(
        ["0"],
        ["1"],
    )

    texts, labels = load_dataset(
        TEST_CSV,
        TEXT_COLUMN,
        LABEL_COLUMN,
        risky_labels,
        safe_labels,
    )

    unique_rows = {}
    conflicts = set()

    for text, label in zip(texts, labels):
        if text in unique_rows and unique_rows[text] != label:
            conflicts.add(text)
        else:
            unique_rows[text] = label

    rows = [
        {
            "text": text,
            "true_label": 1 if label == "risky" else 0,
        }
        for text, label in unique_rows.items()
        if text not in conflicts
    ]

    return rows, conflicts


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)



def main():
    print("=" * 70)
    print("PYTORCH VS ONNX PARITY TEST")
    print("=" * 70)

    print("\nLoading test data...")
    rows, conflicts = load_unique_rows()

    print("Test samples:", len(rows))
    print("Conflicting duplicates removed:", len(conflicts))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))

    print("\nLoading PyTorch model...")
    torch_model = AutoModelForSequenceClassification.from_pretrained(
        str(PYTORCH_DIR)
    )
    torch_model.eval()
    torch_model.to("cpu")

    print("\nLoading ONNX model...")
    session = ort.InferenceSession(
        str(ONNX_DIR / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    class TestDataset(Dataset):
        def __init__(self, examples):
            self.encodings = [
                tokenizer(
                    " ".join(preprocess_text(text)["tokens"]),
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                for text in examples
            ]

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, index):
            return self.encodings[index]

    test_dataset = TestDataset([row["text"] for row in rows])

    data_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    torch_logits_all = []
    onnx_logits_all = []

    print("\nRunning comparison...")

    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader):
            torch_outputs = torch_model(**batch)
            torch_logits = torch_outputs.logits.cpu().numpy()

            input_ids = batch["input_ids"].numpy().astype(np.int64)
            attention_mask = batch["attention_mask"].numpy().astype(np.int64)

            onnx_outputs = session.run(
                ["logits"],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
            )

            torch_logits_all.append(torch_logits)
            onnx_logits_all.append(onnx_outputs[0])

            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(rows),
            )
            print(f"Processed {processed}/{len(rows)}")

    torch_logits = np.concatenate(torch_logits_all, axis=0)
    onnx_logits = np.concatenate(onnx_logits_all, axis=0)

    torch_probs = softmax(torch_logits)
    onnx_probs = softmax(onnx_logits)

    torch_predictions = np.argmax(torch_logits, axis=1)
    onnx_predictions = np.argmax(onnx_logits, axis=1)

    true_labels = np.asarray([
        row["true_label"] for row in rows
    ])

    # Numerical differences
    logit_diff = np.abs(torch_logits - onnx_logits)
    probability_diff = np.abs(torch_probs - onnx_probs)

    max_logit_diff = float(np.max(logit_diff))
    mean_logit_diff = float(np.mean(logit_diff))

    max_probability_diff = float(np.max(probability_diff))
    mean_probability_diff = float(np.mean(probability_diff))

    # Prediction agreement
    agreement = float(
        np.mean(torch_predictions == onnx_predictions)
    )

    torch_accuracy = float(
        np.mean(torch_predictions == true_labels)
    )

    onnx_accuracy = float(
        np.mean(onnx_predictions == true_labels)
    )

    print("\n" + "=" * 70)
    print("PARITY RESULTS")
    print("=" * 70)

    print("Samples:", len(rows))

    print("\nLOGIT DIFFERENCE")
    print(f"Maximum: {max_logit_diff:.10f}")
    print(f"Mean:    {mean_logit_diff:.10f}")

    print("\nPROBABILITY DIFFERENCE")
    print(f"Maximum: {max_probability_diff:.10f}")
    print(f"Mean:    {mean_probability_diff:.10f}")

    print("\nPREDICTION AGREEMENT")
    print(f"Agreement: {agreement * 100:.2f}%")
    print(
        "Different predictions:",
        int(np.sum(torch_predictions != onnx_predictions)),
    )

    print("\nACCURACY ON TEST DATA")
    print(f"PyTorch: {torch_accuracy * 100:.2f}%")
    print(f"ONNX:    {onnx_accuracy * 100:.2f}%")

    print("\nRESULT:")
    if np.array_equal(torch_predictions, onnx_predictions):
        print("PASS: All predicted labels are identical.")
    else:
        print("WARNING: Some predicted labels are different.")

    print("\nComparison completed.")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
