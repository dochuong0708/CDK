import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from app.main import article_analyzer


DATA_PATH = Path("../data/language/processed/news_test.csv")
OUTPUT_PATH = Path("article_test_enhanced_scores.csv")
REPORT_PATH = Path("article_test_enhanced_metrics.json")
DECISION_THRESHOLD = 55

LABEL_MAP = {
    "real": 0,
    "fake": 1,
}

PREDICTION_MAP = {
    "SAFE": 0,
    "SUSPICIOUS": 1,
    "DANGEROUS": 1,
}
DECISION_THRESHOLD = 55

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)

df["actual"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(LABEL_MAP)
)

if df["actual"].isna().any():
    raise ValueError("Dataset contains unmapped labels.")

results = []
errors = []

print("=== DEEPGUARD ARTICLE EVALUATION ===")
print("Total samples:", len(df))
print("\nActual label distribution:")
print(df["label"].value_counts())

total_start = time.perf_counter()

for count, (index, row) in enumerate(df.iterrows(), start=1):
    text = row["text"]
    actual_label = str(row["label"]).strip().lower()

    start = time.perf_counter()

    try:
        result = article_analyzer.analyze_article(
            text=text,
            title="",
            check_facts=False,
        )

        elapsed = time.perf_counter() - start

        summary = result["summary"]
        overall_risk = summary.get("overall_risk_score")

        if overall_risk is None:
            raise ValueError("Missing overall_risk_score")

        predicted_binary = int(
            float(overall_risk) >= DECISION_THRESHOLD
        )

        predicted_label = (
            "SUSPICIOUS" if predicted_binary == 1 else "SAFE"
        )

        ai_components = result.get("ai_components", [])

        results.append({
            "sample_index": index,
            "actual_label": actual_label,
            "actual_binary": LABEL_MAP[actual_label],
            "predicted_label": predicted_label,
            "predicted_binary": predicted_binary,
            "overall_risk_score": summary.get("overall_risk_score"),
            "credibility_score": summary.get("credibility_score"),
            "style_risk_score": summary.get("style_risk_score"),
            "fraud_risk_score": summary.get("fraud_risk_score"),
            "factuality_risk_score": summary.get("factuality_risk_score"),
            "total_paragraphs": summary.get("stats", {}).get(
                "total_paragraphs"
            ),
            "ai_components": ";".join(ai_components),
            "processing_time_seconds": round(elapsed, 6),
        })

        print(
            f"[{count}/{len(df)}] "
            f"Actual={actual_label:<5} "
            f"Predicted={predicted_label:<10} "
            f"Risk={summary.get('overall_risk_score')} "
            f"Time={elapsed:.4f}s"
        )

    except Exception as error:
        elapsed = time.perf_counter() - start

        errors.append({
            "sample_index": index,
            "actual_label": actual_label,
            "error": str(error),
        })

        print(
            f"[{count}/{len(df)}] ERROR: {error}"
        )

total_elapsed = time.perf_counter() - total_start

if not results:
    raise RuntimeError("No samples were successfully evaluated.")

result_df = pd.DataFrame(results)
result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

y_true = result_df["actual_binary"]
y_pred = result_df["predicted_binary"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, zero_division=0
)
recall = recall_score(
    y_true, y_pred, zero_division=0
)
f1 = f1_score(
    y_true, y_pred, zero_division=0
)

matrix = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["real", "fake"],
    output_dict=True,
    zero_division=0,
)

component_counts = {}

for components in result_df["ai_components"]:
    for component in components.split(";"):
        if component:
            component_counts[component] = (
                component_counts.get(component, 0) + 1
            )

metrics = {
    "evaluation_mode": "article_analysis_without_online_fact_check",
    "prediction_mapping": {
        "SAFE": "real",
        "SUSPICIOUS": "fake",
        "DANGEROUS": "fake",
    },
    "total_samples": len(df),
    "successful_samples": len(result_df),
    "failed_samples": len(errors),
    "accuracy": accuracy,
    "precision_fake": precision,
    "recall_fake": recall,
    "f1_fake": f1,
    "macro_f1": report["macro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"],
    "confusion_matrix": matrix.tolist(),
    "classification_report": report,
    "actual_distribution": {
        str(k): int(v)
        for k, v in df["label"].value_counts().items()
    },
    "predicted_three_class_distribution": {
        str(k): int(v)
        for k, v in result_df["predicted_label"].value_counts().items()
    },
    "active_component_sample_counts": component_counts,
    "average_processing_time_seconds": float(
        result_df["processing_time_seconds"].mean()
    ),
    "total_processing_time_seconds": total_elapsed,
    "errors": errors,
}

REPORT_PATH.write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n=== BINARY CLASSIFICATION RESULTS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")
print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

print("\n=== CONFUSION MATRIX ===")
print("Rows: actual [real, fake]")
print("Columns: predicted [real, fake]")
print(matrix)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["real", "fake"],
    zero_division=0,
))

print("\n=== PREDICTED THREE-CLASS DISTRIBUTION ===")
print(result_df["predicted_label"].value_counts())

print("\n=== ACTIVE AI COMPONENTS ===")
print(component_counts)

print("\n=== PROCESSING TIME ===")
print(
    "Average:",
    round(result_df["processing_time_seconds"].mean(), 4),
    "seconds/sample",
)
print("Total:", round(total_elapsed, 4), "seconds")

print("\n=== EXECUTION STATUS ===")
print("Successful:", len(results))
print("Failed:", len(errors))

print("\nSaved:", OUTPUT_PATH.resolve())
print("Saved:", REPORT_PATH.resolve())
