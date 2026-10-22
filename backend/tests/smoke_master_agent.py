"""Real HTTP + restart + deletion test, using a separate local database."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]


def main():
    output = BACKEND/"outputs/week7_validation"/uuid4().hex
    output.mkdir(parents=True)
    image = output/"synthetic-image.png"
    Image.new("RGB", (32,32), "blue").save(image)
    env = os.environ.copy()
    env.update(DEEPGUARD_AGENT_DIR=str(output/"memory"), DEEPGUARD_DB_PATH=str(output/"legacy.db"),
               DEEPGUARD_UPLOAD_DIR=str(output/"uploads"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1",0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = (output/"server.log").open("w", encoding="utf-8")
    process = None
    def stop():
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    def start():
        child = subprocess.Popen([sys.executable,str(BACKEND/"run_audio_agent.py"),"--port",str(port)],
                                 cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+90
        try:
            while time.monotonic() < deadline:
                try:
                    response = httpx.get(url+"/api/health", timeout=2)
                    response.raise_for_status()
                    return child
                except httpx.TransportError:
                    if child.poll() is not None:
                        break
                    time.sleep(.5)
            raise RuntimeError(f"Server failed to start; see {output/'server.log'}")
        except BaseException:
            child.terminate()
            child.wait(timeout=10)
            raise
    try:
        process = start()
        demo = subprocess.run([sys.executable,str(BACKEND/"try_master_agent.py"),"--base-url",url,"--image",str(image)],
                              env=env,cwd=BACKEND,capture_output=True,text=True,encoding="utf-8",timeout=240)
        (output/"demo.log").write_text(demo.stdout+demo.stderr, encoding="utf-8")
        if demo.returncode:
            raise RuntimeError(demo.stdout+demo.stderr)
        demo_dir = Path(next(line[7:] for line in demo.stdout.splitlines() if line.startswith("Saved: ")))
        result = json.loads((demo_dir/"result.json").read_text(encoding="utf-8"))
        session, trace = result["sessionID"], result["traced"]
        stop()
        process = start()
        with httpx.Client(base_url=url,timeout=90) as client:
            prefix = f"/api/agent/sessions/{session}"
            saved = client.get(prefix+f"/traces/{trace}")
            saved.raise_for_status()
            assert saved.json()["status"] == "completed"
            hits = client.get(prefix+"/memory",params={"q":"audio_anomaly_tool"})
            hits.raise_for_status()
            assert hits.json()["items"]
            deletion = client.delete(prefix)
            deletion.raise_for_status()
            assert deletion.json()["vectorsPending"] == 0
            assert deletion.json()["filesPendingCleanup"] == 0
            assert client.get(prefix+"/history").status_code == 404
        report = dict(status="passed", tools=[s["tool"] for s in result["toolsExecuted"]],
                      tracePersistedAfterRestart=True, memoryRetrievedAfterRestart=True,
                      sessionDeletion=deletion.json(), demoOutput=str(demo_dir))
        (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2),flush=True)
        print(f"Saved: {output}")
    finally:
        stop()
        log.close()


if __name__ == "__main__":
    main()
