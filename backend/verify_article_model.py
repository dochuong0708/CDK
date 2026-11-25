from pathlib import Path
from app.main import article_analyzer

model = article_analyzer.model

print("Model class:", type(model).__name__)
print("Configured path:", model.model_path)
print("Model loaded:", model.model is not None)

if model.model_path:
    print("File exists:", Path(model.model_path).is_file())
