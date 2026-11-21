import sys
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FAKE_DIR = SCRIPT_DIR.parents[2] / "data" / "train_tmp" / "fake"
TEST_FAKE_DIR = SCRIPT_DIR / "test_data" / "fake"
DOWNLOAD_COUNT = 600


def download_fake_images() -> None:
    TRAIN_FAKE_DIR.mkdir(parents=True, exist_ok=True)
    TEST_FAKE_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 Đang tải 600 ảnh DiffusionDB qua streaming...")
    dataset = load_dataset(
        "poloclub/diffusiondb",
        "large_random_1k",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    processed = 0
    for item in dataset:
        if processed < 500:
            save_path = TRAIN_FAKE_DIR / f"diffdb_train_{processed + 1:04d}.jpg"
        else:
            save_path = TEST_FAKE_DIR / f"diffdb_test_{processed - 499:04d}.jpg"

        if save_path.exists() and save_path.stat().st_size > 0:
            processed += 1
        else:
            image = item["image"].convert("RGB")
            image = image.resize((224, 224))
            image.save(save_path, "JPEG", quality=85)
            processed += 1

        if processed % 25 == 0:
            print(f"  - Đã xử lý: {processed}/{DOWNLOAD_COUNT}")

        if processed >= DOWNLOAD_COUNT:
            break

    if processed < DOWNLOAD_COUNT:
        raise RuntimeError(
            f"Dataset chỉ cung cấp {processed}/{DOWNLOAD_COUNT} ảnh"
        )

    print("🎉 HOÀN THÀNH TẢI VÀ CHIA 600 ẢNH FAKE!")
    print(f"  - Train (500 ảnh): {TRAIN_FAKE_DIR}")
    print(f"  - Test (100 ảnh): {TEST_FAKE_DIR}")


if __name__ == "__main__":
    try:
        download_fake_images()
    except Exception as error:
        print(f"❌ Có lỗi xảy ra: {error}", file=sys.stderr)
        raise SystemExit(1)