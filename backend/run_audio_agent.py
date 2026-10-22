"""Start the local week 5 Audio Agent, with the existing demo checkpoint if available."""
import json
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.setdefault("DEEPGUARD_DB_PATH", str(ROOT / "data/deepguard.db"))
    os.environ.setdefault("DEEPGUARD_UPLOAD_DIR", str(ROOT / "data/uploads"))
    os.environ.setdefault("DEEPGUARD_AGENT_DIR", str(ROOT / "data/agent"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "backend/outputs/mpl-cache"))
    # Keep student recordings local. External tracing requires a separate explicit setup.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    latest = ROOT / "backend/outputs/asvspoof_demo/latest.json"
    if latest.is_file():
        os.environ.setdefault("DEEPGUARD_AUDIO_MODEL", json.loads(latest.read_text(encoding="utf-8"))["model"])
    import uvicorn
    print(f"Audio Agent: http://127.0.0.1:{args.port}/docs | CNN remains unvalidated.")
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, workers=1)
