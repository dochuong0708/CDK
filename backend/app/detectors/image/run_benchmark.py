import sys
import time
from pathlib import Path

if __package__:
    from .image_detector import FAKE_THRESHOLD, ImageDetector
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from app.detectors.image.image_detector import FAKE_THRESHOLD, ImageDetector

TEST_DIR = Path(__file__).resolve().parent / "test_data"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_benchmark():
    real_dir = TEST_DIR / "real"
    fake_dir = TEST_DIR / "fake"

    if not TEST_DIR.exists():
        print(f"❌ Không tìm thấy thư mục '{TEST_DIR.resolve()}'!")
        return

    try:
        detector = ImageDetector()
    except Exception as error:
        print(f"❌ Lỗi khởi tạo ImageDetector: {error}")
        return

    # TP: Fake đoán đúng là Fake | FN: Fake đoán sai thành Real
    # TN: Real đoán đúng là Real | FP: Real đoán sai thành Fake
    tp, fp, tn, fn = 0, 0, 0, 0
    total_time_ms = 0.0
    processed_images = 0

    print("🚀 Đang chạy đánh giá Benchmark chi tiết (Accuracy, Precision, Recall, F1, Latency)...\n")

    # Đánh giá tập REAL
    for img_path in real_dir.glob("*"):
        if img_path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            file_bytes = img_path.read_bytes()
            start_time = time.perf_counter()
            result = detector.analyze(file_bytes, img_path.name)
            latency = (time.perf_counter() - start_time) * 1000
            total_time_ms += latency
            processed_images += 1

            if result.riskScore >= (FAKE_THRESHOLD * 100):
                fp += 1
            else:
                tn += 1
        except Exception as error:
            print(f"Lỗi ảnh {img_path.name}: {error}")

    # Đánh giá tập FAKE
    for img_path in fake_dir.glob("*"):
        if img_path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            file_bytes = img_path.read_bytes()
            start_time = time.perf_counter()
            result = detector.analyze(file_bytes, img_path.name)
            latency = (time.perf_counter() - start_time) * 1000
            total_time_ms += latency
            processed_images += 1

            if result.riskScore >= (FAKE_THRESHOLD * 100):
                tp += 1
            else:
                fn += 1
        except Exception as error:
            print(f"Lỗi ảnh {img_path.name}: {error}")

    accuracy = (tp + tn) / processed_images if processed_images > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    avg_latency = total_time_ms / processed_images if processed_images > 0 else 0

    print("=" * 60)
    print("📊 BÁO CÁO KẾT QUẢ BENCHMARK DỰ ÁN (DETAILED EVALUATION)")
    print("=" * 60)
    print(f"🔢 Confusion Matrix        : TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print("-" * 60)
    print(f"🎯 Accuracy (Độ chính xác tổng) : {accuracy * 100:.2f}%")
    print(f"🎯 Precision (Độ chính xác Fake): {precision * 100:.2f}%")
    print(f"🎯 Recall (Độ nhạy)             : {recall * 100:.2f}%")
    print(f"🎯 F1-Score                     : {f1_score:.4f} (hoặc {f1_score * 100:.2f}%)")
    print(f"⚡ Latency (Thời gian phản hồi) : {avg_latency:.2f} ms/ảnh")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()