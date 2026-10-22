"""Local HTTP smoke test with the existing ASVspoof demo checkpoint; no accuracy claim."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import soundfile as sf

BACKEND = Path(__file__).resolve().parents[1]


def main():
    output = BACKEND / "outputs/agent_validation" / uuid4().hex
    output.mkdir(parents=True)
    latest = json.loads((BACKEND / "outputs/asvspoof_demo/latest.json").read_text(encoding="utf-8"))
    audio = Path(latest["examples"]["real"]["audio"])
    y, rate = sf.read(audio)
    mp3 = output / "asvspoof_sample.mp3"
    sf.write(mp3, y, rate)
    env = os.environ.copy()
    env.update(DEEPGUARD_AGENT_DIR=str(output / "memory"), DEEPGUARD_DB_PATH=str(output / "legacy.db"),
               DEEPGUARD_UPLOAD_DIR=str(output / "uploads"), DEEPGUARD_AUDIO_MODEL=latest["model"],
               NUMBA_CACHE_DIR=str(BACKEND / "outputs/numba-cache"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log = (output / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(BACKEND / "run_audio_agent.py"), "--port", str(port)],
                               env=env, cwd=BACKEND, stdout=log, stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
            deadline = time.monotonic()+90
            while True:
                try:
                    client.get("/api/health").raise_for_status()
                    break
                except httpx.TransportError:
                    if time.monotonic() > deadline or process.poll() is not None:
                        raise RuntimeError(f"Server did not start; see {output / 'server.log'}")
                    time.sleep(.5)
            response = client.post("/api/agent/sessions")
            response.raise_for_status()
            session = response.json()["sessionID"]
            prefix = f"/api/agent/sessions/{session}"
            report = dict(sessionID=session, source=str(audio), model=latest["model"], cases=[])
            for source in (audio, mp3):
                with source.open("rb") as recording:
                    response = client.post(prefix+"/files", files={"file": (source.name, recording)})
                response.raise_for_status()
                file_id = response.json()["file_id"]
                events, result = [], None
                with client.stream("POST", prefix+"/runs/stream", json=dict(file_id=file_id)) as stream:
                    stream.raise_for_status()
                    event = None
                    for line in stream.iter_lines():
                        if line.startswith("event: "):
                            event = line[7:]
                            events.append(event)
                        if line.startswith("data: ") and event == "result":
                            result = json.loads(line[6:])
                assert events == ["started", "tool_completed", "tool_completed", "result"], events
                assert result["metadata"]["anomaly"]["status"] == "ok", result
                response = client.get(result["metadata"]["anomaly"]["xaiUrl"])
                response.raise_for_status()
                (output / (source.suffix[1:]+"-xai.png")).write_bytes(response.content)
                (output / (source.suffix[1:]+"-result.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                report["cases"].append(dict(format=source.suffix, events=events, fakeScore=result["riskScore"],
                                             modelStatus=result["metadata"]["modelStatus"]))
                print(source.suffix, "HTTP stream and XAI OK", flush=True)
            response = client.get(prefix+"/history")
            response.raise_for_status()
            assert len(response.json()["runs"]) == 2
            report["status"] = "passed"
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(output, flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == "__main__":
    main()
