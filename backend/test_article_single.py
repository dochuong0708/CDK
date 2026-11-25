import json
import time
import pandas as pd

from app.main import article_analyzer

DATA_PATH = "../data/language/processed/news_test.csv"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = df.dropna(subset=["text", "label"]).copy()

sample = df.iloc[0]

print("=== INPUT ===")
print("Label:", sample["label"])
print("Text length:", len(str(sample["text"])))

start = time.perf_counter()

result = article_analyzer.analyze_article(
    text=str(sample["text"]),
    title="",
    check_facts=False
)

elapsed = time.perf_counter() - start

print("\n=== RESULT KEYS ===")
print(list(result.keys()))

print("\n=== RESULT ===")

for key, value in result.items():
    if key in {"paragraphs", "claims", "evidence"}:
        print(f"{key}: {len(value) if value else 0} items")
    else:
        print(f"{key}: {value}")

print("\n=== EXECUTION TIME ===")
print(f"{elapsed:.4f} seconds")
