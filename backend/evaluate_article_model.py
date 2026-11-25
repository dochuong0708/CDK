import time
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from app.main import article_analyzer

DATASETS = {
    "validation": "../data/language/processed/news_val.csv",
    "test": "../data/language/processed/news_test.csv",
}

LABEL_MAP = {"real": 0, "fake": 1}
THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65
]


def evaluate_dataset(name, path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"]).copy()

    results = []

    print(f"\n=== {name.upper()} ===")
    print("Samples:", len(df))

    for count, (_, row) in enumerate(df.iterrows(), start=1):
        actual_label = str(row["label"]).strip().lower()

        if actual_label not in LABEL_MAP:
            raise ValueError(f"Unknown label: {actual_label}")

        start = time.perf_counter()

        result = article_analyzer.analyze_article(
            text=str(row["text"]),
            title="",
            check_facts=False,
        )

        elapsed = time.perf_counter() - start

        paragraph_scores = [
            p.get("model_score")
            for p in result.get("paragraphs", [])
            if p.get("model_score") is not None
        ]

        if not paragraph_scores:
            raise RuntimeError(
                f"No Joblib scores found in sample {count}"
            )

        article_score = sum(paragraph_scores) / len(paragraph_scores)

        results.append({
            "sample_index": count,
            "actual_label": actual_label,
            "actual_binary": LABEL_MAP[actual_label],
            "article_model_score": article_score,
            "paragraph_count": len(paragraph_scores),
            "processing_time_seconds": elapsed,
        })

        print(
            f"[{count}/{len(df)}] "
            f"Actual={actual_label:<5} "
            f"Model score={article_score:.4f}"
        )

    return pd.DataFrame(results)


def calculate_metrics(df, threshold):
    y_true = df["actual_binary"]
    y_pred = (
        df["article_model_score"] >= threshold
    ).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }


start_total = time.perf_counter()

validation = evaluate_dataset(
    "validation",
    DATASETS["validation"],
)

test = evaluate_dataset(
    "test",
    DATASETS["test"],
)

# Chọn ngưỡng chỉ trên validation.
validation_results = [
    calculate_metrics(validation, threshold)
    for threshold in THRESHOLDS
]

best = max(validation_results, key=lambda x: x["f1"])
best_threshold = best["threshold"]

# Đánh giá test bằng ngưỡng đã chọn.
test_predictions = (
    test["article_model_score"] >= best_threshold
).astype(int)

test_metrics = calculate_metrics(test, best_threshold)

matrix = confusion_matrix(
    test["actual_binary"],
    test_predictions,
    labels=[0, 1],
)

report = classification_report(
    test["actual_binary"],
    test_predictions,
    labels=[0, 1],
    target_names=["real", "fake"],
    output_dict=True,
    zero_division=0,
)

test["predicted_binary"] = test_predictions
test["predicted_label"] = test_predictions.map({
    0: "real",
    1: "fake",
})

validation.to_csv(
    "article_model_validation.csv",
    index=False,
    encoding="utf-8-sig",
)

test.to_csv(
    "article_model_test.csv",
    index=False,
    encoding="utf-8-sig",
)

output = {
    "evaluation_method": "mean_paragraph_joblib_score",
    "threshold_selection": "validation_only",
    "selected_threshold": best_threshold,
    "validation_best_metrics": best,
    "test_metrics": test_metrics,
    "test_confusion_matrix": matrix.tolist(),
    "test_classification_report": report,
    "test_average_processing_time_seconds": float(
        test["processing_time_seconds"].mean()
    ),
    "test_sample_count": len(test),
    "validation_sample_count": len(validation),
    "threshold_comparison": validation_results,
}

Path("article_model_evaluation.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n=== VALIDATION THRESHOLD COMPARISON ===")
for item in validation_results:
    print(
        f"Threshold={item['threshold']:.2f} "
        f"Precision={item['precision']:.4f} "
        f"Recall={item['recall']:.4f} "
        f"F1={item['f1']:.4f}"
    )

print("\n=== SELECTED THRESHOLD ===")
print(best)

print("\n=== FINAL TEST RESULTS ===")
print(test_metrics)

print("\n=== TEST CONFUSION MATRIX ===")
print("Rows: actual [real, fake]")
print("Columns: predicted [real, fake]")
print(matrix)

print("\n=== TEST CLASSIFICATION REPORT ===")
print(classification_report(
    test["actual_binary"],
    test_predictions,
    labels=[0, 1],
    target_names=["real", "fake"],
    zero_division=0,
))

print("\nAverage processing time:")
print(
    round(
        test["processing_time_seconds"].mean(),
        4
    ),
    "seconds/sample"
)

print("\nTotal execution time:")
print(round(time.perf_counter() - start_total, 4), "seconds")

print("\nSaved:")
print("article_model_validation.csv")
print("article_model_test.csv")
print("article_model_evaluation.json")
