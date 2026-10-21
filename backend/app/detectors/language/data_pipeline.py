"""Tải, chuẩn hóa nhãn, khử trùng lặp và chia train/val/test cho dữ liệu tin giả tiếng Việt.

Sinh hai bộ dữ liệu trong ``data/language/processed``:

* ``news_{train,val,test}.csv``: phân loại văn bản ``fake``/``real`` (dùng trực tiếp với train.py).
* ``claims_{train,val,test}.csv``: kiểm chứng claim–evidence ``SUPPORTED``/``REFUTED``/``NEI``.

Chạy từ thư mục backend: ``python -m app.detectors.language.data_pipeline``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "language" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "language" / "processed"
LOCAL_FAKE_NEWS_DIR = REPO_ROOT / "data" / "TEXT" / "Fake_news_data"
USER_AGENT = "DeepGuard-data-pipeline/0.1 (+https://github.com/dochuong0708/CDK)"
HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"

SPLITS = ("train", "val", "test")
NEWS_LABELS = ("fake", "real")
CLAIM_LABELS = ("SUPPORTED", "REFUTED", "NEI")
MIN_NEWS_CHARS = 20

# Ý nghĩa nhãn lấy từ dataset card của từng bộ; nhãn lạ sẽ báo lỗi thay vì đoán.
LOCAL_FAKE_NEWS_LABELS = {"0": "fake", "1": "real"}
VIFACTCHECK_LABELS = {"0": "SUPPORTED", "1": "REFUTED", "2": "NEI"}
VIWIKIFC_LABELS = {
    "supports": "SUPPORTED",
    "refutes": "REFUTED",
    "not_enough_information": "NEI",
}

HF_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "vifactcheck": (
        "tranthaihoa/vifactcheck",
        (
            "data/train-00000-of-00001.parquet",
            "data/dev-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ),
    ),
    "viwikifc": ("High-Will/ViWikiFC", ("data/train.csv", "data/dev.csv", "data/test.csv")),
}
# Chỉ dùng bộ dữ liệu có giấy phép mở rõ ràng. Không thêm bộ không ghi giấy phép
# hoặc phải xin tác giả (ví dụ ReINTEL, VFND, binhvq_news_vi) hay nội dung báo chí có bản quyền.
DATASET_LICENSES: dict[str, dict[str, str]] = {
    "vifactcheck": {
        "license": "MIT",
        "url": "https://huggingface.co/datasets/tranthaihoa/vifactcheck",
        "attribution": "Tran, Tran, Tran, Nguyen. ViFactCheck (AAAI-25)",
    },
    "viwikifc": {
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/High-Will/ViWikiFC",
        "attribution": "Le, To, Nguyen, Nguyen. ViWikiFC; nội dung gốc từ Wikipedia tiếng Việt",
    },
}
CLAIM_SOURCES = ("vifactcheck", "viwikifc")
# Dữ liệu cục bộ chưa rõ giấy phép: chỉ bật bằng --sources khi chủ dự án đã xác nhận quyền sử dụng.
OPT_IN_SOURCES = ("fake_news_data",)
DEFAULT_SOURCES = CLAIM_SOURCES
SUPPORTED_SOURCES = (*CLAIM_SOURCES, *OPT_IN_SOURCES)


@dataclass
class NewsRecord:
    id: str
    text: str
    label: str
    dataset: str
    source: str = ""
    title: str = ""
    url: str = ""
    published_at: str = ""
    label_origin: str = "gold"


@dataclass
class ClaimRecord:
    id: str
    claim: str
    evidence: str
    context: str
    label: str
    dataset: str
    source: str = ""
    topic: str = ""
    url: str = ""


Record = TypeVar("Record", NewsRecord, ClaimRecord)


# ---------------------------------------------------------------- chuẩn hóa văn bản và nhãn


def clean_text(text: str) -> str:
    """NFC, bỏ ký tự điều khiển, gom khoảng trắng nhưng giữ ranh giới đoạn văn."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = "".join(
        char for char in normalized if char == "\n" or unicodedata.category(char)[0] != "C"
    )
    lines = [re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in normalized.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def map_label(mapping: dict[str, str], raw: object, dataset: str) -> str:
    key = unicodedata.normalize("NFC", str(raw)).strip().casefold()
    if key in mapping:
        return mapping[key]
    raise ValueError(f"{dataset}: nhãn '{raw}' không nằm trong bảng quy đổi {sorted(mapping)}")


# ---------------------------------------------------------------- chuyển từng nguồn về schema chung


def local_fake_news_records(rows: Iterable[dict[str, Any]], prefix: str) -> Iterator[NewsRecord]:
    for index, row in enumerate(rows):
        text = clean_text(str(row.get("Maintext") or ""))
        if len(text) < MIN_NEWS_CHARS:
            continue
        yield NewsRecord(
            id=f"fake_news_data-{prefix}-{index}",
            text=text,
            label=map_label(LOCAL_FAKE_NEWS_LABELS, row.get("Label"), "Fake_news_data"),
            dataset="fake_news_data",
        )


def vifactcheck_records(rows: Iterable[dict[str, Any]]) -> Iterator[ClaimRecord]:
    for row in rows:
        claim = clean_text(str(row.get("Statement") or ""))
        if not claim:
            continue
        yield ClaimRecord(
            id=f"vifactcheck-{row.get('annotation_id')}",
            claim=claim,
            evidence=clean_text(str(row.get("Evidence") or "")),
            context=clean_text(str(row.get("Context") or "")),
            label=map_label(VIFACTCHECK_LABELS, row.get("labels"), "ViFactCheck"),
            dataset="vifactcheck",
            source=str(row.get("Author") or "").strip(),
            topic=str(row.get("Topic") or "").strip(),
            url=str(row.get("Url") or "").strip(),
        )


def viwikifc_records(rows: Iterable[dict[str, Any]]) -> Iterator[ClaimRecord]:
    for row in rows:
        claim = clean_text(str(row.get("claim") or ""))
        if not claim:
            continue
        title = str(row.get("title") or "").strip()
        yield ClaimRecord(
            id=f"viwikifc-{row.get('pairID')}",
            claim=claim,
            evidence=clean_text(str(row.get("evidence") or "")),
            context=clean_text(str(row.get("context") or "")),
            label=map_label(VIWIKIFC_LABELS, row.get("gold_label"), "ViWikiFC"),
            dataset="viwikifc",
            source=f"wikipedia:{title}" if title else "",
            url=str(row.get("link") or "").strip(),
        )


# ---------------------------------------------------------------- tải và đọc file


def download(url: str, target: Path, force: bool = False) -> Path:
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(target)
    return target


def fetch_hf_files(source: str, raw_dir: Path, revision: str, force: bool) -> list[Path]:
    repo, paths = HF_SOURCES[source]
    return [
        download(
            HF_RESOLVE_URL.format(repo=repo, revision=revision, path=path),
            raw_dir / source / Path(path).name,
            force=force,
        )
        for path in paths
    ]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def read_parquet_rows(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Cần pyarrow để đọc parquet: pip install -r requirements-data.txt"
        ) from error
    return pq.read_table(path, columns=list(columns) if columns else None).to_pylist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- khử trùng lặp gần đúng (MinHash + LSH)

_HASH_PRIME = np.uint64(4294967311)  # số nguyên tố > 2^32, để a*x + b không tràn uint64
_MAX_HASH = np.uint64(0xFFFFFFFF)


def dedup_tokens(text: str) -> list[str]:
    lowered = unicodedata.normalize("NFC", text).casefold()
    lowered = re.sub(r"https?://\S+|www\.\S+", " ", lowered)
    return re.findall(r"\w+", lowered)


def shingles(tokens: Sequence[str], size: int) -> set[str]:
    if len(tokens) <= size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, int(_MAX_HASH), num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MAX_HASH), num_perm, dtype=np.uint64)

    def signature(self, items: set[str]) -> np.ndarray:
        hashes = np.fromiter(
            (
                int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=4).digest(), "little")
                for item in items
            ),
            dtype=np.uint64,
            count=len(items),
        )
        permuted = (hashes[:, None] * self.a[None, :] + self.b[None, :]) % _HASH_PRIME
        return permuted.min(axis=0)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def cluster_near_duplicates(
    texts: Sequence[str],
    threshold: float = 0.85,
    shingle_size: int = 5,
    num_perm: int = 128,
    bands: int = 16,
) -> list[int]:
    """Trả về mã cụm cho từng văn bản; các văn bản có Jaccard ước lượng >= threshold chung cụm."""
    if num_perm % bands:
        raise ValueError("num_perm phải chia hết cho bands")
    rows_per_band = num_perm // bands
    hasher = MinHasher(num_perm)
    union = _UnionFind(len(texts))
    exact: dict[str, int] = {}
    signatures: dict[int, np.ndarray] = {}
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)

    for index, text in enumerate(texts):
        tokens = dedup_tokens(text)
        if not tokens:
            continue
        fingerprint = " ".join(tokens)
        if fingerprint in exact:
            union.union(exact[fingerprint], index)
            continue
        exact[fingerprint] = index
        signature = hasher.signature(shingles(tokens, shingle_size))
        signatures[index] = signature
        for band in range(bands):
            key = (band, signature[band * rows_per_band : (band + 1) * rows_per_band].tobytes())
            for other in buckets[key]:
                if union.find(other) == union.find(index):
                    continue
                if float(np.mean(signatures[other] == signature)) >= threshold:
                    union.union(other, index)
            buckets[key].append(index)

    return [union.find(index) for index in range(len(texts))]


@dataclass
class DedupStats:
    input: int = 0
    kept: int = 0
    duplicates_removed: int = 0
    conflicting_removed: int = 0


def deduplicate(
    records: Sequence[Record],
    key: Callable[[Record], str],
    threshold: float,
    shingle_size: int,
) -> tuple[list[Record], DedupStats]:
    """Giữ một bản đại diện mỗi cụm; bỏ cả cụm nếu các bản trùng mang nhãn mâu thuẫn."""
    clusters = cluster_near_duplicates([key(r) for r in records], threshold, shingle_size)
    members: dict[int, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        members[cluster].append(index)

    stats = DedupStats(input=len(records))
    keep: list[int] = []
    for indexes in members.values():
        if len({records[i].label for i in indexes}) > 1:
            stats.conflicting_removed += len(indexes)
            continue
        representative = min(
            indexes, key=lambda i: (getattr(records[i], "label_origin", "gold") != "gold", i)
        )
        keep.append(representative)
        stats.duplicates_removed += len(indexes) - 1
    kept = [records[i] for i in sorted(keep)]
    stats.kept = len(kept)
    return kept, stats


def news_dedup_key(record: NewsRecord) -> str:
    return record.text


def claim_dedup_key(record: ClaimRecord) -> str:
    return f"{record.claim}\n{record.evidence}"


# ---------------------------------------------------------------- chia theo nguồn


def _stable_hash(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha1(f"{seed}:{value}".encode()).digest()[:8], "big")


def group_split(
    records: Sequence[Record],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[Record]]:
    """Chia theo nhóm nguồn: mọi bản ghi cùng ``source`` nằm trọn trong một split.

    Bản ghi không có nguồn được coi là một nhóm riêng. Nhóm lớn được xếp trước, mỗi nhóm vào
    split đang thiếu nhiều nhất so với tỉ lệ mục tiêu, nên kết quả ổn định theo ``seed``.
    """
    if len(ratios) != len(SPLITS) or any(r < 0 for r in ratios) or not sum(ratios):
        raise ValueError("ratios phải gồm 3 số không âm cho train/val/test")
    weights = {name: ratio / sum(ratios) for name, ratio in zip(SPLITS, ratios, strict=True)}
    active = [name for name in SPLITS if weights[name] > 0]

    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.source or f"record:{record.id}"].append(record)
    if len(groups) < len(active):
        groups = {f"record:{r.id}": [r] for r in records}

    order = sorted(groups, key=lambda group: (-len(groups[group]), _stable_hash(group, seed)))
    total = len(records)
    result: dict[str, list[Record]] = {name: [] for name in SPLITS}
    for group in order:
        target = max(
            active,
            key=lambda name: (weights[name] * total - len(result[name])) / (weights[name] * total),
        )
        result[target].extend(groups[group])
    return result


def split_by_dataset(
    records: Sequence[Record], ratios: Sequence[float], seed: int
) -> dict[str, list[Record]]:
    """Chia riêng từng dataset để split nào cũng có đủ các nguồn dữ liệu."""
    by_dataset: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_dataset[record.dataset].append(record)
    combined: dict[str, list[Record]] = {name: [] for name in SPLITS}
    for dataset in sorted(by_dataset):
        for name, items in group_split(by_dataset[dataset], ratios, seed).items():
            combined[name].extend(items)
    return combined


# ---------------------------------------------------------------- ghi kết quả


def write_split(path: Path, records: Sequence[Record], record_type: type) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field.name for field in fields(record_type)]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def split_summary(splits: dict[str, Sequence[Record]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, items in splits.items():
        by_dataset: dict[str, Counter] = defaultdict(Counter)
        for record in items:
            by_dataset[record.dataset][record.label] += 1
        summary[name] = {
            "total": len(items),
            "labels": dict(Counter(record.label for record in items)),
            "sources": len({record.source or record.id for record in items}),
            "by_dataset": {dataset: dict(counts) for dataset, counts in sorted(by_dataset.items())},
        }
    return summary


def leaked_sources(splits: dict[str, Sequence[Record]]) -> set[str]:
    seen: dict[str, set[str]] = defaultdict(set)
    for name, items in splits.items():
        for record in items:
            if record.source:
                seen[record.source].add(name)
    return {source for source, names in seen.items() if len(names) > 1}


# ---------------------------------------------------------------- điều phối


def load_news(sources: Sequence[str], files: dict[str, str]) -> list[NewsRecord]:
    records: list[NewsRecord] = []
    if "fake_news_data" in sources:
        for path in sorted(LOCAL_FAKE_NEWS_DIR.glob("*.csv")):
            files[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
            records.extend(local_fake_news_records(read_csv_rows(path), path.stem))
    return records


def load_claims(
    sources: Sequence[str], raw_dir: Path, revision: str, force: bool, files: dict[str, str]
) -> list[ClaimRecord]:
    records: list[ClaimRecord] = []
    if "vifactcheck" in sources:
        for path in fetch_hf_files("vifactcheck", raw_dir, revision, force):
            files[f"vifactcheck/{path.name}"] = sha256_file(path)
            records.extend(vifactcheck_records(read_parquet_rows(path)))
    if "viwikifc" in sources:
        for path in fetch_hf_files("viwikifc", raw_dir, revision, force):
            files[f"viwikifc/{path.name}"] = sha256_file(path)
            records.extend(viwikifc_records(read_csv_rows(path)))
    return records


def process(
    records: Sequence[Record],
    prefix: str,
    record_type: type,
    key: Callable[[Record], str],
    threshold: float,
    shingle_size: int,
    ratios: Sequence[float],
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    raw_counts = Counter(record.dataset for record in records)
    kept, stats = deduplicate(records, key, threshold, shingle_size)
    splits = split_by_dataset(kept, ratios, seed)
    for name, items in splits.items():
        write_split(out_dir / f"{prefix}_{name}.csv", items, record_type)
    leaks = leaked_sources(splits)
    if leaks:
        raise AssertionError(f"Nguồn xuất hiện ở nhiều split: {sorted(leaks)[:5]}")
    return {
        "raw_by_dataset": dict(raw_counts),
        "dedup": asdict(stats),
        "splits": split_summary(splits),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=(
            f"Danh sách nguồn, phân tách bằng dấu phẩy. Mặc định: {','.join(DEFAULT_SOURCES)}."
            f" Nguồn chỉ bật khi đã xác nhận giấy phép: {','.join(OPT_IN_SOURCES)}"
        ),
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-dup-threshold", type=float, default=0.85)
    parser.add_argument("--claim-near-dup-threshold", type=float, default=0.9)
    parser.add_argument("--hf-revision", default="main", help="Branch/commit trên Hugging Face")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(sources) - set(SUPPORTED_SOURCES)
    if unknown:
        raise SystemExit(f"Nguồn không hỗ trợ: {sorted(unknown)}")
    files: dict[str, str] = {}

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "config": {
            "sources": sources,
            "ratios": list(args.ratios),
            "seed": args.seed,
            "near_dup_threshold": args.near_dup_threshold,
            "claim_near_dup_threshold": args.claim_near_dup_threshold,
            "hf_revision": args.hf_revision,
        },
        "label_maps": {
            "fake_news_data": LOCAL_FAKE_NEWS_LABELS,
            "vifactcheck": VIFACTCHECK_LABELS,
            "viwikifc": VIWIKIFC_LABELS,
        },
        "licenses": {name: DATASET_LICENSES[name] for name in sources if name in DATASET_LICENSES},
    }

    news = load_news(sources, files)
    if news:
        manifest["news"] = process(
            news, "news", NewsRecord, news_dedup_key, args.near_dup_threshold, 5,
            args.ratios, args.seed, args.out_dir,
        )
    claims = load_claims(sources, args.raw_dir, args.hf_revision, args.force_download, files)
    if claims:
        manifest["claims"] = process(
            claims, "claims", ClaimRecord, claim_dedup_key, args.claim_near_dup_threshold, 3,
            args.ratios, args.seed, args.out_dir,
        )
    manifest["input_files_sha256"] = files

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for task in ("news", "claims"):
        if task in manifest:
            dedup = manifest[task]["dedup"]
            print(
                f"[{task}] vào {dedup['input']} | giữ {dedup['kept']} | trùng {dedup['duplicates_removed']}"
                f" | mâu thuẫn nhãn {dedup['conflicting_removed']}"
            )
            for name, summary in manifest[task]["splits"].items():
                print(f"  {name:5s} {summary['total']:6d} {summary['labels']}")
    print(f"Đã ghi {args.out_dir} (manifest: {manifest_path})")
    return manifest


if __name__ == "__main__":
    main()
