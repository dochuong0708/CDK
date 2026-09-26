"""
Phát hiện video deepfake bằng mô hình đã huấn luyện sẵn trên Hugging Face.

Mô hình dùng: Naman712/Deep-fake-detection
- Kiến trúc: ResNext50 + LSTM (phân tích chuỗi khung hình theo thời gian)
- Nhận trực tiếp file video, không cần tự tách khung hình

CÁCH DÙNG:
    pip install -r requirements.txt

    python hf_video_detector.py --input video.mp4
    python hf_video_detector.py --input thu_muc_video/

Lần chạy đầu tiên cần mạng để tải mô hình, các lần sau dùng cache nên nhanh hơn.
"""

import argparse
from pathlib import Path

from transformers import pipeline

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MODEL_NAME = "Naman712/Deep-fake-detection"

FAKE_THRESHOLD = 0.65
REAL_THRESHOLD = 0.35
FAKE_LABELS = {"fake", "ai", "ai-generated", "deepfake"}

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        print(f"Đang tải mô hình {MODEL_NAME} (chỉ tải lần đầu)...")
        _detector = pipeline("video-classification", model=MODEL_NAME)
    return _detector


def fake_probability(video_path):
    detector = get_detector()
    results = detector(str(video_path))  # ví dụ: [{"label": "fake", "score": 0.92}, {"label": "real", "score": 0.08}]
    prob_fake = 0.0
    for r in results:
        if r["label"].strip().lower() in FAKE_LABELS:
            prob_fake += r["score"]
    return prob_fake


def verdict(prob_fake):
    if prob_fake >= FAKE_THRESHOLD:
        return "Nhiều khả năng là video deepfake/AI"
    if prob_fake <= REAL_THRESHOLD:
        return "Nhiều khả năng là video thật"
    return "Không chắc chắn - cần người xem xét"


def predict(input_path):
    target = Path(input_path)
    if target.is_dir():
        paths = sorted(p for p in target.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    else:
        paths = [target]
    if not paths:
        raise SystemExit("Không tìm thấy video nào.")

    for p in paths:
        try:
            prob_fake = fake_probability(p)
        except Exception as e:
            print(f"{p}: LỖI khi xử lý - {e}")
            continue
        print(f"{p}: xác suất deepfake = {prob_fake:.1%} -> {verdict(prob_fake)}")


def main():
    parser = argparse.ArgumentParser(description="Phát hiện video deepfake bằng mô hình Hugging Face có sẵn")
    parser.add_argument("--input", required=True, help="Đường dẫn video hoặc thư mục")
    args = parser.parse_args()
    predict(args.input)


if __name__ == "__main__":
    main()
