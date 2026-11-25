import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("article_test_results.csv")

df["score"] = df["overall_risk_score"] / 100

def classify(score):
    if score < 0.55:
        return "SAFE"
    elif score < 0.70:
        return "SUSPICIOUS"
    return "DANGEROUS"

df["new_prediction"] = df["score"].apply(classify)

print("=== THREE-CLASS DISTRIBUTION ===")
print(df["new_prediction"].value_counts())

print("\n=== PREDICTION BY ACTUAL LABEL ===")
print(pd.crosstab(
    df["actual_label"],
    df["new_prediction"]
))

df["predicted_binary"] = (
    df["new_prediction"] != "SAFE"
).astype(int)

print("\n=== BINARY EVALUATION ===")
print(confusion_matrix(
    df["actual_binary"],
    df["predicted_binary"],
    labels=[0, 1]
))

print(classification_report(
    df["actual_binary"],
    df["predicted_binary"],
    labels=[0, 1],
    target_names=["real", "fake"],
    zero_division=0
))
