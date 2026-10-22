"""
Phát hiện ảnh do AI tạo bằng mô hình đã huấn luyện sẵn trên Hugging Face,
thay vì tự huấn luyện từ đầu.

Mô hình dùng: dima806/ai_vs_human_generated_image_detection
- Kiến trúc: Vision Transformer (ViT)
- Giấy phép: Apache 2.0 (được phép dùng và tinh chỉnh, cần ghi nguồn)
- Huấn luyện trên dữ liệu mới hơn CIFAKE, nên đa dạng và chuẩn hơn với ảnh ngoài đời

CẢI TIẾN THÊM SO VỚI BẢN GỐC (Test-Time Augmentation - TTA):
Thay vì chỉ đưa 1 ảnh gốc vào mô hình, chương trình tạo thêm vài phiên bản
biến đổi nhẹ của cùng một ảnh (lật ngang, xoay nhẹ, tăng/giảm sáng nhẹ),
chạy mô hình trên từng phiên bản rồi lấy trung bình. Cách này giúp giảm
sai số khi ảnh gốc rơi đúng vào "điểm mù" của mô hình, vì các phiên bản
biến đổi ít khi cùng rơi vào điểm mù giống hệt nhau.

CÁCH DÙNG:
    pip install -r requirements.txt

    python hf_image_detector.py --input anh.jpg
    python hf_image_detector.py --input thu_muc_anh/
    python hf_image_detector.py --input anh.jpg --no_tta   # tắt TTA, chạy như bản gốc

Lần chạy đầu tiên cần mạng để tải mô hình (khoảng 300-400 MB), các lần sau
mô hình được lưu cache nên không cần tải lại.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageEnhance
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


def _prob_from_results(results):
    """Cộng dồn điểm số của các nhãn được coi là 'AI-generated'."""
    prob_fake = 0.0
    for r in results:
        if r["label"].strip().lower() in FAKE_LABELS:
            prob_fake += r["score"]
    return prob_fake


def make_tta_variants(image):
    """Tạo các phiên bản biến đổi nhẹ của cùng một ảnh để giảm sai số ngẫu nhiên."""
    variants = [image]
    variants.append(image.transpose(Image.FLIP_LEFT_RIGHT))  # lật ngang
    variants.append(image.rotate(5, expand=False, fillcolor=(128, 128, 128)))   # xoay nhẹ +5 độ
    variants.append(image.rotate(-5, expand=False, fillcolor=(128, 128, 128)))  # xoay nhẹ -5 độ
    variants.append(ImageEnhance.Brightness(image).enhance(1.1))  # sáng hơn 10%
    variants.append(ImageEnhance.Brightness(image).enhance(0.9))  # tối hơn 10%
    return variants


def fake_probability(image_path, use_tta=True):
    """
    Trả về xác suất ảnh do AI tạo.

    use_tta=True (mặc định): chạy mô hình trên nhiều phiên bản biến đổi nhẹ của
    ảnh gốc (test-time augmentation) rồi lấy trung bình, giúp kết quả ổn định hơn.
    use_tta=False: chạy như bản gốc, chỉ 1 lần trên ảnh gốc.
    """
    detector = get_detector()

    if not use_tta:
        results = detector(str(image_path))
        return _prob_from_results(results)

    image = Image.open(image_path).convert("RGB")
    variants = make_tta_variants(image)
    probs = [_prob_from_results(detector(v)) for v in variants]
    return sum(probs) / len(probs)


def verdict(prob_fake):
    if prob_fake >= FAKE_THRESHOLD:
        return "Nhiều khả năng do AI tạo"
    if prob_fake <= REAL_THRESHOLD:
        return "Nhiều khả năng là ảnh thật"
    return "Không chắc chắn - cần người xem xét"


def predict(input_path, use_tta=True):
    target = Path(input_path)
    if target.is_dir():
        paths = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMG_EXTS)
    else:
        paths = [target]
    if not paths:
        raise SystemExit("Không tìm thấy ảnh nào.")

    for p in paths:
        prob_fake = fake_probability(p, use_tta=use_tta)
        print(f"{p}: xác suất AI = {prob_fake:.1%} -> {verdict(prob_fake)}")


def main():
    parser = argparse.ArgumentParser(description="Phát hiện ảnh AI bằng mô hình Hugging Face có sẵn")
    parser.add_argument("--input", required=True, help="Đường dẫn ảnh hoặc thư mục")
    parser.add_argument("--no_tta", action="store_true",
                         help="Tắt test-time augmentation, chạy như bản gốc (nhanh hơn nhưng kém ổn định hơn)")
    args = parser.parse_args()
    predict(args.input, use_tta=not args.no_tta)


if __name__ == "__main__":
    main()
