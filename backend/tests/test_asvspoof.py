import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asvspoof_data import discover_partitions, prepare_manifest, read_manifest, audit_manifest


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "LA"
    protocols = root / "ASVspoof2019_LA_cm_protocols"
    protocols.mkdir(parents=True)
    for split_index, (split, prefix, suffix) in enumerate([
        ("train", "T", "trn"), ("dev", "D", "trl"), ("eval", "E", "trl")
    ]):
        audio_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
        audio_dir.mkdir(parents=True)
        lines = []
        for index in range(4):
            utterance = f"LA_{prefix}_{1000000+index}"
            label = "bonafide" if index < 2 else "spoof"
            attack = "-" if index < 2 else "A01"
            lines.append(f"speaker_{split}_{index} {utterance} - {attack} {label}")
            t = np.arange(16000) / 16000
            sf.write(audio_dir / (utterance + ".flac"), .5*np.sin(2*np.pi*(200+split_index*100+index*20)*t), 16000)
        (protocols / f"ASVspoof2019.LA.cm.{split}.{suffix}.txt").write_text("\n".join(lines))
    return root


def test_discovery_mapping_and_reproducible_subset(corpus, tmp_path):
    partitions = discover_partitions(corpus.parent, ["train", "dev", "test"])
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    report = prepare_manifest(partitions, first, limit_per_class=1, seed=42)
    prepare_manifest(partitions, second, limit_per_class=1, seed=42)
    assert first.read_bytes() == second.read_bytes()
    assert report["subset"] is True
    rows = read_manifest(first)
    assert len(rows) == 6
    assert {r["label"] for r in rows} == {"real", "fake"}
    assert all(r["utterance"].startswith("LA_E_") for r in rows if r["split"] == "test")
    audit = audit_manifest(first, tmp_path / "audit.json", decode_audio=True)
    assert audit["valid"] and audit["decoded"]
    with pytest.raises(ValueError, match="tồn tại"):
        prepare_manifest(partitions, first)


def test_missing_audio_fails_before_output(corpus, tmp_path):
    missing = next((corpus / "ASVspoof2019_LA_train" / "flac").glob("*.flac"))
    missing.unlink()
    out = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="Thiếu file"):
        prepare_manifest(discover_partitions(corpus, ["train"]), out)
    assert not out.exists()


@pytest.mark.parametrize("line", [
    "speaker LA_T_1000000 - A01 unknown",
    "speaker LA_D_1000000 - A01 spoof",
    "speaker ../../outside - A01 spoof",
    "speaker LA_T_1000000 - A01 spoof extra_column",
])
def test_reject_wrong_protocol(corpus, tmp_path, line):
    partitions = discover_partitions(corpus, ["train"])
    partitions[0][1].write_text(line)
    with pytest.raises(ValueError):
        prepare_manifest(partitions, tmp_path / "bad.csv")


def test_hash_duplicates_and_decode_error(corpus, tmp_path):
    out = tmp_path / "full.csv"
    prepare_manifest(discover_partitions(corpus, ["train"]), out)
    rows = read_manifest(out)
    first, second = rows[0]["path"], rows[1]["path"]
    original = second.read_bytes()
    second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="Nội dung file trùng"):
        read_manifest(out)
    second.write_bytes(original)
    first.write_bytes(b"corrupt audio")
    with pytest.raises(ValueError, match="file không hợp lệ"):
        audit_manifest(out, tmp_path / "audit.json", decode_audio=True)
    report = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert not report["valid"] and len(report["errors"]) == 1


def test_cli_end_to_end(corpus, tmp_path):
    """Protocol-shaped synthetic corpus checks plumbing, never detection accuracy."""
    cli = Path(__file__).resolve().parents[1] / "audio_cli.py"
    out = tmp_path / "manifest.csv"
    checkpoint = tmp_path / "model.pt"

    def run(*args):
        result = subprocess.run([sys.executable, "-X", "utf8", str(cli), *map(str, args)],
                                capture_output=True, text=True, encoding="utf-8", timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr

    run("prepare-asvspoof2019", "--root", corpus, "--output", out, "--limit-per-class", 1)
    run("validate-manifest", "--manifest", out, "--output", tmp_path/"audit.json", "--decode-audio")
    run("train", "--manifest", out, "--epochs", 1, "--batch-size", 2, "--output", checkpoint)
    run("evaluate", "--manifest", out, "--model", checkpoint, "--output", tmp_path/"metrics.json")
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    run("predict", "--audio", rows[0]["path"], "--model", checkpoint, "--output", tmp_path/"prediction")
    assert (tmp_path/"prediction"/"xai.png").read_bytes().startswith(b"\x89PNG")
    assert json.loads((tmp_path/"metrics.json").read_text())["samples"] == 2
