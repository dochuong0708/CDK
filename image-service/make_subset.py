"""
Chép một tập con nhỏ từ bộ CIFAKE (hoặc bộ dữ liệu có cấu trúc tương tự)
sang thư mục data/ để huấn luyện thử nhanh.

Cấu trúc nguồn (--src) cần có:
    train/REAL  train/FAKE  test/REAL  test/FAKE

Kết quả (--dst):
    data/train/real  data/train/fake  data/val/real  data/val/fake

Ví dụ:
    python make_subset.py --src "C:\\Users\\PC\\Downloads\\CIFAKE" --dst data --train_n 1500 --val_n 300
"""

import argparse
import random
import shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_dir(parent, names):
    """Tìm thư mục con có tên nằm trong `names` (không phân biệt hoa/thường)."""
    for d in parent.iterdir():
        if d.is_dir() and d.name.lower() in names:
            return d
    raise SystemExit(f"Không tìm thấy thư mục {sorted(names)} trong: {parent}")


def copy_subset(src_split, dst_split, n, seed):
    for cls in ("real", "fake"):
        src_dir = find_dir(src_split, {cls})
        files = [p for p in src_dir.iterdir() if p.suffix.lower() in EXTS]
        random.Random(seed).shuffle(files)
        out = dst_split / cls
        out.mkdir(parents=True, exist_ok=True)
        chosen = files[:n]
        for p in chosen:
            shutil.copy2(p, out / p.name)
        print(f"  {out}: đã chép {len(chosen)}/{len(files)} ảnh")


def main():
    parser = argparse.ArgumentParser(description="Tạo tập con nhỏ từ CIFAKE")
    parser.add_argument("--src", required=True, help="Thư mục chứa train và test")
    parser.add_argument("--dst", default="data", help="Thư mục đích (mặc định: data)")
    parser.add_argument("--train_n", type=int, default=1500, help="Số ảnh mỗi lớp cho train")
    parser.add_argument("--val_n", type=int, default=300, help="Số ảnh mỗi lớp cho val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    src_train = find_dir(src, {"train"})
    src_val = find_dir(src, {"test", "val", "valid", "validation"})

    print("Train:")
    copy_subset(src_train, dst / "train", args.train_n, args.seed)
    print("Val:")
    copy_subset(src_val, dst / "val", args.val_n, args.seed)
    print("Xong. Giờ có thể chạy: python ai_image_detector.py train --data_dir data")


if __name__ == "__main__":
    main()
