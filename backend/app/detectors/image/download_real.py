import sys
import urllib.request
import io
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent

# Đường dẫn Train ở ổ C, Test ở dự án (ổ D)
TRAIN_REAL_DIR = Path(r"C:\Users\PC\train_tmp\data\real")
TEST_REAL_DIR = SCRIPT_DIR / "test_data" / "real"
DOWNLOAD_COUNT = 600

def download_real_images() -> None:
    TRAIN_REAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_REAL_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 Đang tải 600 ảnh REAL trực tiếp (Bỏ qua hoàn toàn Hugging Face)...")

    processed = 0
    # Vòng lặp tải từng ảnh
    for i in range(DOWNLOAD_COUNT):
        if processed < 500:
            save_path = TRAIN_REAL_DIR / f"real_train_{processed + 1:04d}.jpg"
        else:
            save_path = TEST_REAL_DIR / f"real_test_{processed - 499:04d}.jpg"

        # Bỏ qua nếu ảnh đã tồn tại (hỗ trợ tải tiếp nếu dừng giữa chừng)
        if save_path.exists() and save_path.stat().st_size > 0:
            processed += 1
        else:
            # Lấy ảnh chụp thực tế ngẫu nhiên theo ID cố định (đảm bảo file không thay đổi)
            url = f"https://picsum.photos/seed/real_img_{i}/512/512"
            
            # Gửi request tải ảnh trực tiếp
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                img_bytes = response.read()

            # Mở, Resize 224x224 và nén JPG
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image = image.resize((224, 224))
            image.save(save_path, "JPEG", quality=85)
            processed += 1

        # Báo tiến độ mỗi 25 ảnh giống hệt file Fake
        if processed % 25 == 0:
            print(f"  - Đã xử lý: {processed}/{DOWNLOAD_COUNT}")

    if processed < DOWNLOAD_COUNT:
        raise RuntimeError(f"Chỉ tải được {processed}/{DOWNLOAD_COUNT} ảnh")

    print("🎉 HOÀN THÀNH TẢI VÀ CHIA 600 ẢNH REAL!")
    print(f"  - Train Real (500 ảnh): {TRAIN_REAL_DIR}")
    print(f"  - Test Real (100 ảnh): {TEST_REAL_DIR}")

if __name__ == "__main__":
    try:
        download_real_images()
    except Exception as error:
        print(f"❌ Có lỗi xảy ra: {error}", file=sys.stderr)
        raise SystemExit(1)