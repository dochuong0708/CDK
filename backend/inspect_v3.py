import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "data/models/vietnam_fake_news_risk_v3.joblib"
DATA_PATH = "../data/language/processed/news_test.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)

label_map = {
    "fake": "risky",
    "real": "safe",
}

df["actual"] = df["label"].astype(str).str.strip().str.lower().map(label_map)

if df["actual"].isna().any():
    print("ERROR: Found labels that are not mapped:")
    print(df.loc[df["actual"].isna(), "label"].value_counts())
    raise SystemExit(1)

predicted = model.predict(df["text"].tolist())

probabilities = model.predict_proba(df["text"].tolist())
classes = [str(x).lower() for x in model.classes_]
risky_index = classes.index("risky")

df["predicted"] = predicted
df["risk_probability"] = probabilities[:, risky_index]

print("\n=== DATASET INFORMATION ===")
print("Number of samples:", len(df))
print("\nActual labels:")
print(df["actual"].value_counts())

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    df["actual"],
    df["predicted"],
    labels=["safe", "risky"],
    digits=4,
    zero_division=0
))

print("\n=== CONFUSION MATRIX ===")
print("Rows = actual labels; columns = predicted labels")
print("Label order: safe, risky")
print(confusion_matrix(
    df["actual"],
    df["predicted"],
    labels=["safe", "risky"]
))

print("\n=== MEAN RISK PROBABILITY BY ACTUAL LABEL ===")
print(df.groupby("actual")["risk_probability"].agg(
    ["count", "mean", "min", "max"]
).round(4))

false_positive = df[
    (df["actual"] == "safe") &
    (df["predicted"] == "risky")
]

false_negative = df[
    (df["actual"] == "risky") &
    (df["predicted"] == "safe")
]

print("\n=== FALSE POSITIVES: REAL NEWS PREDICTED AS FAKE ===")
print("Count:", len(false_positive))

for _, row in false_positive.head(10).iterrows():
    print("\nRisk probability:", round(row["risk_probability"], 4))
    print("Text:", row["text"][:700])

print("\n=== FALSE NEGATIVES: FAKE NEWS PREDICTED AS REAL ===")
print("Count:", len(false_negative))

for _, row in false_negative.head(10).iterrows():
    print("\nRisk probability:", round(row["risk_probability"], 4))
    print("Text:", row["text"][:700])
