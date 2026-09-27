"""Start the installed frontend using Node on PATH or the local bundled runtime."""
import shutil
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    node = shutil.which("node")
    if node is None:
        bundled = Path(sys.base_prefix).parent / "node/bin/node.exe"
        if bundled.is_file():
            node = str(bundled)
    if node is None:
        raise SystemExit("Node.js not found. Install Node.js, then reopen VS Code.")
    vite = frontend / "node_modules/vite/bin/vite.js"
    if not vite.is_file():
        raise SystemExit("Frontend dependencies missing. Run npm install in frontend first.")
    try:
        subprocess.run([node, str(vite), "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
                       cwd=frontend, check=True)
    except KeyboardInterrupt:
        pass
