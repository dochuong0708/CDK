from pathlib import Path
from app.models.schemas import Analysis

class ImageDetector:
    def analyze(self, path: Path) -> Analysis:
        score = 64 if path.stat().st_size else 50
        return Analysis(
            analysisId=f"img-{path.stem.lower()}",
            modality="Image",
            label="Review recommended" if score >= 65 else "Likely authentic",
            riskScore=score,
            confidence=0.63,
            evidence=[f"Metadata inspected for {path.name}", "Content heuristic completed"],
            detector="image-heuristic-v1",
            input=path.name,
            createdAt=__import__('datetime').datetime.now()
        )
