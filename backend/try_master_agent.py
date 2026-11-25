"""Week 7 demo: route text + ASVspoof audio, read a trace, exercise memory CRUD."""
import argparse
import json
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if args.audio is None:
        latest = json.loads((ROOT/"outputs/asvspoof_demo/latest.json").read_text(encoding="utf-8"))
        args.audio = Path(latest["examples"]["real"]["audio"])
    output = ROOT/"outputs/week7_demo"/uuid4().hex
    output.mkdir(parents=True)
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        def checked(response):
            response.raise_for_status()
            return response.json()
        session = checked(client.post("/api/agent/sessions"))["sessionID"]
        prefix = f"/api/agent/sessions/{session}"
        master = f"/api/master/sessions/{session}"
        checked(client.patch(prefix, json={"title":"Week 7 Audio integration demo"}))
        ids = []
        for path in (args.audio, args.image):
            if path:
                with path.open("rb") as recording:
                    uploaded = checked(client.post(master+"/files", files={"file":(path.name, recording)}))
                ids.append(uploaded["file_id"])
        request = dict(file_ids=ids, text="Student-supplied text for the multimodal routing demo.", message="Analyze the attached inputs.")
        result = None
        with client.stream("POST", master+"/runs/stream", json=request) as response:
            response.raise_for_status()
            event = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                    print(event, flush=True)
                elif line.startswith("data: "):
                    payload = json.loads(line[6:])
                    if event == "error":
                        raise RuntimeError(payload)
                    if event == "result":
                        result = payload
        if result is None:
            raise RuntimeError("No final result received.")
        trace_url = prefix+f"/traces/{result['traced']}"
        trace = checked(client.get(trace_url))
        assert trace["status"] == "completed"
        checked(client.patch(trace_url, json={"note":"Demo reviewed; model is unvalidated."}))
        note = checked(client.post(prefix+"/messages", json={"text":"Temporary CRUD verification note"}))
        note_url = prefix+f"/messages/{note['id']}"
        checked(client.patch(note_url, json={"text":"Updated CRUD verification note"}))
        checked(client.delete(note_url))  # Only the note created by this script.
        assert client.get(note_url).status_code == 404
        for name, value in (("result", result), ("trace", checked(client.get(trace_url))),
                            ("history", checked(client.get(prefix+"/history"))),
                            ("memory", checked(client.get(prefix+"/memory", params={"q":"audio_anomaly_tool"})))):
            (output/f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        for item in result["metadata"]["outputs"]:
            url = item["output"].get("xaiUrl")
            if url:
                response = client.get(url)
                response.raise_for_status()
                (output/f"{item['arguments']['file_id']}-xai.png").write_bytes(response.content)
    print(f"Saved: {output}\nSession: {session}\nTrace: {result['traced']}\nCRUD: passed")


if __name__ == "__main__":
    main()
