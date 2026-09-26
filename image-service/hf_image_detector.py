"""
Phát hiện ảnh do AI tạo bằng mô hình đã huấn luyện sẵn trên Hugging Face,
thay vì tự huấn luyện từ đầu.

Mô hình dùng: dima806/ai_vs_human_generated_image_detection
- Kiến trúc: Vision Transformer (ViT)
- Giấy phép: Apache 2.0 (được phép dùng và tinh chỉnh, cần ghi nguồn)
- Huấn luyện trên dữ liệu mới hơn CIFAKE, nên đa dạng và chuẩn hơn với ảnh ngoài đời

CÁCH DÙNG:
    pip install -r requirements.txt

    python hf_image_detector.py --input anh.jpg
    python hf_image_detector.py --input thu_muc_anh/

Lần chạy đầu tiên cần mạng để tải mô hình (khoảng 300-400 MB), các lần sau
mô hình được lưu cache nên không cần tải lại.
"""

import argparse
from pathlib import Path

from transformers import pipeline

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_NAME = "dima806/ai_vs_human_generated_image_detection"

FAKE_THRESHOLD = 0.65
REAL_THRESHOLD = 0.35

# Nhãn "AI-generated" có thể viết khác nhau tuỳ mô hình, liệt kê để nhận diện chắc chắn
FAKE_LABELS = {"ai", "ai-generated", "ai_generated", "fake", "generated"}


_detector = None


def get_detector():
    """Tải mô hình một lần và tái sử dụng cho các lần gọi sau."""
    global _detector
    if _detector is None:
        print(f"Đang tải mô hình {MODEL_NAME} (chỉ tải lần đầu)...")
        _detector = pipeline("image-classification", model=MODEL_NAME)
    return _detector


def fake_probability(image_path):
    """Trả về xác suất ảnh do AI tạo, dựa trên toàn bộ điểm số của mô hình."""
    detector = get_detector()
    results = detector(str(image_path))  # ví dụ: [{"label": "AI", "score": 0.95}, {"label": "Real", "score": 0.05}]
    prob_fake = 0.0
    for r in results:
        if r["label"].strip().lower() in FAKE_LABELS:
            prob_fake += r["score"]
    return prob_fake


def verdict(prob_fake):
    if prob_fake >= FAKE_THRESHOLD:
        return "Nhiều khả năng do AI tạo"
    if prob_fake <= REAL_THRESHOLD:
        return "Nhiều khả năng là ảnh thật"
    return "Không chắc chắn - cần người xem xét"


def predict(input_path):
    target = Path(input_path)
    if target.is_dir():
        paths = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMG_EXTS)
    else:
        paths = [target]
    if not paths:
        raise SystemExit("Không tìm thấy ảnh nào.")

    for p in paths:
        prob_fake = fake_probability(p)
        print(f"{p}: xác suất AI = {prob_fake:.1%} -> {verdict(prob_fake)}")


def main():
    parser = argparse.ArgumentParser(description="Phát hiện ảnh AI bằng mô hình Hugging Face có sẵn")
    parser.add_argument("--input", required=True, help="Đường dẫn ảnh hoặc thư mục")
    args = parser.parse_args()
    predict(args.input)


if __name__ == "__main__":
    main()
