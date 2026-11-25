
import json
import time
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


DATA_PATH = Path("../data/language/processed/news_val.csv")
OUTPUT_PATH = Path("article_validation_enhanced_scores.csv")
REPORT_PATH = Path("article_validation_enhanced_metrics.json")

LABEL_MAP = {
    "real": 0,
    "fake": 1,
}

# Các ngưỡng thử nghiệm, tính theo thang điểm 0-100.
THRESHOLDS = list(range(30, 71, 5))


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

print("=== DEEPGUARD VALIDATION THRESHOLD EVALUATION ===")
print("Total samples:", len(df))
print("\nActual label distribution:")
print(df["label"].value_counts())

start_time = time.perf_counter()

for count, (index, row) in enumerate(df.iterrows(), start=1):
    try:
        result = article_analyzer.analyze_article(
            text=row["text"],
            title="",
            check_facts=False,
        )

        summary = result["summary"]

        risk_score = summary.get("overall_risk_score")

        if risk_score is None:
            raise ValueError("Missing overall_risk_score")

        results.append({
            "sample_index": index,
            "actual_label": str(row["label"]).strip().lower(),
            "actual_binary": int(row["actual"]),
            "overall_risk_score": float(risk_score),
            "credibility_score": summary.get("credibility_score"),
            "style_risk_score": summary.get("style_risk_score"),
            "fraud_risk_score": summary.get("fraud_risk_score"),
            "factuality_risk_score": summary.get(
                "factuality_risk_score"
            ),
            "original_predicted_label": summary.get("label"),
        })

        print(
            f"[{count}/{len(df)}] "
            f"Actual={row['label']} "
            f"Risk={risk_score}"
        )

    except Exception as exc:
        errors.append({
            "sample_index": index,
            "error": str(exc),
        })
        print(f"[ERROR] Sample {index}: {exc}")


elapsed = time.perf_counter() - start_time

if not results:
    raise RuntimeError("No validation samples were evaluated.")

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

y_true = result_df["actual_binary"].to_numpy()

threshold_results = []

for threshold in THRESHOLDS:
    y_pred = (
        result_df["overall_risk_score"] >= threshold
    ).astype(int).to_numpy()

    metrics = {
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
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }

    threshold_results.append(metrics)


print("\n=== THRESHOLD COMPARISON ===")

for item in threshold_results:
    print(
        f"T={item['threshold']:2d} | "
        f"Accuracy={item['accuracy']:.4f} | "
        f"Precision={item['precision']:.4f} | "
        f"Recall={item['recall']:.4f} | "
        f"F1={item['f1']:.4f} | "
        f"CM={item['confusion_matrix']}"
    )


best_result = max(
    threshold_results,
    key=lambda item: item["f1"],
)

print("\n=== BEST VALIDATION THRESHOLD BY F1 ===")
print(json.dumps(best_result, indent=4, ensure_ascii=False))

best_pred = (
    result_df["overall_risk_score"] >= best_result["threshold"]
).astype(int)

print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        y_true,
        best_pred,
        target_names=["real", "fake"],
        zero_division=0,
    )
)

report = {
    "evaluation_mode": "validation_threshold_selection",
    "total_samples": len(result_df),
    "failed_samples": len(errors),
    "processing_time_seconds": elapsed,
    "threshold_results": threshold_results,
    "best_threshold_by_f1": best_result,
    "errors": errors,
}

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4, ensure_ascii=False)

print("\nSaved:", OUTPUT_PATH)
print("Saved:", REPORT_PATH)
print(f"Total processing time: {elapsed:.4f}s")