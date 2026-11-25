"""One-command ASVspoof smoke run on real dataset files; not a benchmark."""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

BACKEND = Path(__file__).resolve().parent
PROJECT = BACKEND.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "data/audio/asvspoof2019/LA")
    parser.add_argument("--limit-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.limit_per_class < 1 or args.epochs < 1:
        parser.error("Sample limit and epochs must be positive")
    folder = BACKEND / "outputs/asvspoof_demo"
    run = folder / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
    run.mkdir(parents=True)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(BACKEND / "outputs/mpl-cache")

    def command(*arguments):
        print("\n>>> " + " ".join(map(str, arguments)), flush=True)
        subprocess.run([sys.executable, "-X", "utf8", str(BACKEND / "audio_cli.py"),
                        *map(str, arguments)], cwd=BACKEND, env=environment, check=True)

    manifest = run / "manifest.csv"
    model = run / "model.pt"
    command("prepare-asvspoof2019", "--root", args.root, "--limit-per-class", args.limit_per_class,
            "--output", manifest)
    command("validate-manifest", "--manifest", manifest, "--decode-audio", "--output", run/"audit.json")
    command("train", "--manifest", manifest, "--epochs", args.epochs, "--batch-size", 8,
            "--device", "cpu", "--output", model)
    command("evaluate", "--manifest", manifest, "--model", model, "--output", run/"metrics.json")
    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    examples = {}
    for label in ("real", "fake"):
        row = next(r for r in rows if r["split"] == "test" and r["label"] == label)
        command("predict", "--audio", row["path"], "--model", model, "--output", run/label)
        examples[label] = dict(audio=row["path"], xai=str(run/label/"xai.png"))
    result = dict(run=str(run), model=str(model), manifest=str(manifest), examples=examples,
                  notice="Small subset, one training run: pipeline demo only, not an ASVspoof benchmark")
    (folder/"latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nDONE. Open this folder in VS Code: {run}")
    print("Images: real/xai.png and fake/xai.png. Scores: metrics.json")
    print("This small model can misclassify speech. Do not report these scores as full ASVspoof results.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        sys.exit(1)
