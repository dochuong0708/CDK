import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

model = joblib.load("data/models/vietnam_fake_news_risk_v3.joblib")
df = pd.read_csv("../data/language/processed/news_test.csv", encoding="utf-8-sig")

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)

label_map = {"fake": 1, "real": 0}
df["actual"] = df["label"].astype(str).str.strip().str.lower().map(label_map)

if df["actual"].isna().any():
    raise ValueError("Dataset contains unmapped labels")

classes = [str(x).lower() for x in model.classes_]
risky_index = classes.index("risky")

probabilities = model.predict_proba(df["text"].tolist())
df["risk_probability"] = probabilities[:, risky_index]

df["predicted"] = (df["risk_probability"] >= 0.5).astype(int)
df["error"] = df["actual"] != df["predicted"]

print("\n=== ERROR DISTRIBUTION BY SOURCE ===")
if "source" in df.columns:
    print(
        df[df["error"]]
        .groupby(["source", "label"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

print("\n=== ERROR DISTRIBUTION BY DATASET ===")
if "dataset" in df.columns:
    print(
        df[df["error"]]
        .groupby(["dataset", "label"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

print("\n=== LABEL ORIGIN OF ERROR SAMPLES ===")
if "label_origin" in df.columns:
    print(df[df["error"]]["label_origin"].value_counts(dropna=False))

print("\n=== THRESHOLD COMPARISON ===")
print("Threshold | Precision | Recall | F1")

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    predicted = (df["risk_probability"] >= threshold).astype(int)

    precision = precision_score(df["actual"], predicted, zero_division=0)
    recall = recall_score(df["actual"], predicted, zero_division=0)
    f1 = f1_score(df["actual"], predicted, zero_division=0)

    print(
        f"{threshold:.2f}      | "
        f"{precision:.4f}    | "
        f"{recall:.4f} | "
        f"{f1:.4f}"
    )

print("\n=== ERROR SAMPLE METADATA ===")
columns = [
    col for col in [
        "id", "title", "source", "dataset",
        "label", "label_origin", "risk_probability"
    ] if col in df.columns
]

print(
    df[df["error"]][columns]
    .to_string(index=False, max_colwidth=100)
)
