"""ASVspoof 2019 LA protocol preparation and manifest integrity checks."""
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

FIELDS = ["path", "label", "speaker", "split", "attack", "utterance"]
PARTITIONS = {"train": ("train", "trn"), "dev": ("dev", "trl"), "test": ("eval", "trl")}


def read_manifest(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"path", "label", "speaker", "split"} <= rows[0].keys():
        raise ValueError("CSV cần các cột path,label,speaker,split và ít nhất một dòng.")
    seen_paths, speakers, hashes = set(), {}, set()
    for index, row in enumerate(rows, 2):
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ("path", "label", "speaker", "split")):
            raise ValueError(f"CSV dòng {index}: thiếu giá trị bắt buộc.")
        if row["label"] not in {"real", "fake"} or row["split"] not in PARTITIONS:
            raise ValueError("label phải là real/fake; split phải là train/dev/test.")
        row["path"] = (path.parent / row["path"]).resolve()
        if row["path"] in seen_paths:
            raise ValueError(f"File trùng lặp: {row['path']}")
        seen_paths.add(row["path"])
        previous = speakers.setdefault(row["speaker"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Người nói xuất hiện ở nhiều split: {row['speaker']}")
        digest = hashlib.sha256()
        with row["path"].open("rb") as audio:
            for chunk in iter(lambda: audio.read(1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        if checksum in hashes:
            raise ValueError(f"Nội dung file trùng lặp: {row['path']}")
        hashes.add(checksum)
    return rows


def discover_partitions(root, splits):
    """Accept the extracted LA directory or its immediate parent, never guess recursively."""
    root = Path(root).resolve()
    if not (root / "ASVspoof2019_LA_cm_protocols").is_dir():
        root = root / "LA"
    protocols = root / "ASVspoof2019_LA_cm_protocols"
    if not protocols.is_dir():
        raise ValueError("Không tìm thấy ASVspoof2019_LA_cm_protocols. --root phải trỏ tới thư mục LA đã giải nén hoặc thư mục cha.")
    partitions = []
    for split in splits:
        name, suffix = PARTITIONS[split]
        protocol = protocols / f"ASVspoof2019.LA.cm.{name}.{suffix}.txt"
        audio_dir = root / f"ASVspoof2019_LA_{name}" / "flac"
        if not protocol.is_file() or not audio_dir.is_dir():
            raise ValueError(f"Thiếu protocol hoặc thư mục flac cho {split}: {protocol}; {audio_dir}")
        partitions.append((split, protocol, audio_dir))
    return partitions


def protocol_rows(partitions):
    rows, utterances, speakers = [], set(), {}
    for split, protocol, audio_dir in partitions:
        if split not in PARTITIONS:
            raise ValueError("Split phải là train, dev hoặc test.")
        audio_dir = Path(audio_dir).resolve()
        for number, line in enumerate(Path(protocol).read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5 or fields[4] not in {"bonafide", "spoof"}:
                raise ValueError(f"{protocol}:{number}: cần protocol CM ASVspoof 2019 LA 5 cột; không dùng protocol ASV hoặc 2021.")
            speaker, utterance, _, attack, label = fields
            if not re.fullmatch(r"LA_[TDE]_\d+", utterance):
                raise ValueError(f"ID không phải ASVspoof 2019 LA: {utterance}")
            expected_prefix = {"train": "LA_T_", "dev": "LA_D_", "test": "LA_E_"}[split]
            if not utterance.startswith(expected_prefix):
                raise ValueError(f"Protocol {utterance} không khớp split {split}.")
            if utterance in utterances:
                raise ValueError(f"Utterance trùng: {utterance}")
            utterances.add(utterance)
            if speakers.setdefault(speaker, split) != split:
                raise ValueError(f"Người nói xuất hiện ở nhiều split: {speaker}")
            audio = audio_dir / (utterance + ".flac")
            if not audio.is_file():
                raise ValueError(f"Thiếu file: {audio}")
            rows.append(dict(path=str(audio), label="real" if label == "bonafide" else "fake",
                             speaker=speaker, split=split, attack=attack, utterance=utterance))
    return rows


def summary(rows):
    report = {}
    for split in PARTITIONS:
        partition = [r for r in rows if r["split"] == split]
        if partition:
            report[split] = dict(files=len(partition), labels=dict(Counter(r["label"] for r in partition)),
                                 speakers=len({r["speaker"] for r in partition}),
                                 attacks=dict(Counter(r.get("attack", "unknown") for r in partition)))
    return report


def prepare_manifest(partitions, output, limit_per_class=None, seed=42):
    if limit_per_class is not None and limit_per_class < 1:
        raise ValueError("--limit-per-class phải lớn hơn 0.")
    rows = protocol_rows(partitions)
    full_summary = summary(rows)
    if not rows:
        raise ValueError("Protocol không có bản ghi.")
    for split, counts in full_summary.items():
        if set(counts["labels"]) != {"real", "fake"}:
            raise ValueError(f"Split {split} phải có cả real và fake.")
    if limit_per_class is not None:
        rng = random.Random(seed)
        selected = []
        for split in PARTITIONS:
            for label in ("real", "fake"):
                group = [r for r in rows if r["split"] == split and r["label"] == label]
                selected.extend(rng.sample(group, min(limit_per_class, len(group))))
        rows = selected
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an existing experiment manifest by accident.
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise ValueError(f"Đầu ra đã tồn tại; chọn tên manifest mới: {output}")
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = dict(dataset="ASVspoof2019-LA", subset=limit_per_class is not None,
                  limit_per_class=limit_per_class, seed=seed, selected=summary(rows), full_protocol=full_summary,
                  note="Subset for pipeline checks only, not an official benchmark" if limit_per_class else "Official split assignments preserved")
    output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def audit_manifest(path, output, decode_audio=False):
    """Hash integrity plus optional full decoding with exactly the inference preprocessor."""
    rows = read_manifest(path)
    partitions = summary(rows)
    for split, counts in partitions.items():
        if set(counts["labels"]) != {"real", "fake"}:
            raise ValueError(f"Split {split} thiếu real hoặc fake.")
    errors = []
    if decode_audio:
        from app.detectors.audio.pipeline import load_audio
        for index, row in enumerate(rows, 1):
            try:
                load_audio(row["path"])
            except (ValueError, OSError) as exc:
                errors.append(dict(path=str(row["path"]), reason=str(exc)))
            if index % 1000 == 0:
                print(f"Decoded {index}/{len(rows)} files", flush=True)
    report = dict(manifest=str(Path(path).resolve()), splits=partitions, sha256_checked=True,
                  decoded=decode_audio, errors=errors, valid=not errors)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise ValueError(f"{len(errors)} file không hợp lệ. Xem {output}. Không tự động bỏ file khỏi benchmark.")
    return report
