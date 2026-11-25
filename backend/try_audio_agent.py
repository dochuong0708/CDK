"""Exercise the real HTTP streaming workflow and save its XAI PNG locally."""
import argparse
import json
from pathlib import Path
from uuid import uuid4

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    backend = Path(__file__).resolve().parent
    if args.audio is None:
        latest = json.loads((backend / "outputs/asvspoof_demo/latest.json").read_text(encoding="utf-8"))
        args.audio = Path(latest["examples"]["real"]["audio"])
    output = backend / "outputs/agent_demo" / uuid4().hex
    output.mkdir(parents=True)
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        response = client.post("/api/agent/sessions")
        response.raise_for_status()
        session = response.json()["sessionID"]
        prefix = f"/api/agent/sessions/{session}"
        with args.audio.open("rb") as audio:
            response = client.post(prefix + "/files", files={"file": (args.audio.name, audio, "audio/octet-stream")})
        response.raise_for_status()
        payload = dict(file_id=response.json()["file_id"], message="Phan tich am thanh, MFCC va Grad-CAM.")
        result = None
        with client.stream("POST", prefix + "/runs/stream", json=payload) as stream:
            stream.raise_for_status()
            event = None
            for line in stream.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                    print(event, flush=True)
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    if event == "error":
                        raise RuntimeError(data)
                    if event == "result":
                        result = data
        if result is None:
            raise RuntimeError("Stream ended without a result.")
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        xai = result["metadata"]["anomaly"].get("xaiUrl")
        if xai:
            response = client.get(xai)
            response.raise_for_status()
            (output / "xai.png").write_bytes(response.content)
        for endpoint in ("history", "memory?q=MFCC"):
            response = client.get(prefix + "/" + endpoint)
            response.raise_for_status()
            (output / (endpoint.split("?")[0]+".json")).write_text(json.dumps(response.json(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Result: {output}\nSession: {session}\nModel status: {result['metadata']['modelStatus']}")


if __name__ == "__main__":
    main()
