import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_PATH = "data/models/vietnam_fake_news_risk_v3.joblib"
DATA_PATH = "../data/language/processed/news_val.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)

label_map = {"fake": 1, "real": 0}
df["actual"] = (
    df["label"].astype(str).str.strip().str.lower().map(label_map)
)

if df["actual"].isna().any():
    raise ValueError("Dataset contains unmapped labels")

classes = [str(x).lower() for x in model.classes_]
risky_index = classes.index("risky")

probabilities = model.predict_proba(df["text"].tolist())
df["risk_probability"] = probabilities[:, risky_index]

print("=== VALIDATION DATASET ===")
print("Number of samples:", len(df))
print(df["actual"].value_counts())

print("\n=== THRESHOLD COMPARISON ===")
print("Threshold | Precision | Recall | F1")

results = []

for threshold in [
    0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65
]:
    predicted = (
        df["risk_probability"] >= threshold
    ).astype(int)

    precision = precision_score(
        df["actual"], predicted, zero_division=0
    )
    recall = recall_score(
        df["actual"], predicted, zero_division=0
    )
    f1 = f1_score(
        df["actual"], predicted, zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(
        f"{threshold:.2f}      | "
        f"{precision:.4f}    | "
        f"{recall:.4f} | "
        f"{f1:.4f}"
    )

best = max(results, key=lambda x: x["f1"])

print("\n=== BEST VALIDATION THRESHOLD ===")
print("Threshold:", best["threshold"])
print("Precision:", round(best["precision"], 4))
print("Recall:", round(best["recall"], 4))
print("F1:", round(best["f1"], 4))
