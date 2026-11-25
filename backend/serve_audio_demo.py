"""Start the API with the latest completed demo checkpoint."""
import json
import os
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
PROJECT = BACKEND.parent


if __name__ == "__main__":
    latest = BACKEND / "outputs/asvspoof_demo/latest.json"
    if not latest.is_file():
        raise SystemExit("Run run_audio_demo.py successfully before starting this demo API.")
    model = Path(json.loads(latest.read_text(encoding="utf-8"))["model"])
    if not model.is_file():
        raise SystemExit(f"Demo checkpoint missing: {model}")
    os.environ["DEEPGUARD_AUDIO_MODEL"] = str(model)
    os.environ["DEEPGUARD_DB_PATH"] = str(PROJECT / "data/deepguard.db")
    os.environ["DEEPGUARD_UPLOAD_DIR"] = str(PROJECT / "data/uploads")
    os.environ["MPLCONFIGDIR"] = str(BACKEND / "outputs/mpl-cache")
    import uvicorn
    print("Starting API with a small DEMO model, not a validated deepfake detector.")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
