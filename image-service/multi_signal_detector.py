"""
Phát hiện ảnh AI bằng cách kết hợp NHIỀU tín hiệu độc lập, thay vì chỉ dựa vào
một mô hình duy nhất. Ý tưởng: mỗi tín hiệu có điểm mạnh/yếu khác nhau, kết hợp
lại giúp bù trừ sai số cho nhau.

BA TÍN HIỆU:
1. Mô hình ViT (dima806/ai_vs_human_generated_image_detection) - đọc NỘI DUNG ảnh,
   nhưng có thể bị đánh lừa bởi ảnh lạ với dữ liệu huấn luyện của nó.
2. Metadata EXIF - ảnh thật chụp từ máy ảnh/điện thoại thường có thông tin máy ảnh,
   ống kính, ngày giờ... Ảnh AI tạo ra hoặc đã qua chỉnh sửa thường THIẾU các
   trường này. Đây là tín hiệu không liên quan gì đến nội dung ảnh, nên không bị
   đánh lừa bởi cùng điểm yếu của mô hình ViT.
3. Phân tích tần số (FFT) - ảnh AI thường để lại hoạ tiết lặp lại ở tần số cao
   mà mắt thường khó thấy nhưng có thể đo được bằng biến đổi Fourier.

CÁCH DÙNG:
    pip install -r requirements.txt

    python multi_signal_detector.py --input anh.jpg
    python multi_signal_detector.py --input thu_muc_anh/

LƯU Ý QUAN TRỌNG: đây vẫn chỉ là công cụ hỗ trợ, không phải kết luận chắc chắn.
Metadata có thể bị xóa/giả mạo cố ý, và phân tích tần số có thể bị ảnh hưởng
bởi việc nén/chỉnh sửa ảnh thông thường (không liên quan đến AI).
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ExifTags

from hf_image_detector import fake_probability as model_fake_probability

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Trọng số của từng tín hiệu khi kết hợp - mô hình ViT đáng tin nhất nên có
# trọng số cao nhất, hai tín hiệu kia chỉ đóng vai trò hỗ trợ/điều chỉnh.
WEIGHTS = {"model": 0.70, "metadata": 0.15, "frequency": 0.15}

FAKE_THRESHOLD = 0.65
REAL_THRESHOLD = 0.35

# Các trường EXIF cho thấy ảnh chụp từ thiết bị thật
CAMERA_TAGS = {"Make", "Model", "LensModel", "FNumber", "ExposureTime", "ISOSpeedRatings", "FocalLength"}


def check_metadata(image_path):
    """
    Trả về điểm nghi ngờ 'AI tạo' dựa trên metadata EXIF (0 = chắc chắn có vẻ thật,
    1 = chắc chắn có vẻ AI/đã qua xử lý). Ảnh càng thiếu thông tin máy ảnh, điểm
    nghi ngờ càng cao.
    """
    try:
        img = Image.open(image_path)
        exif_raw = img.getexif()
        if not exif_raw:
            return 0.75, "Không có dữ liệu EXIF nào"

        tags_found = set()
        for tag_id, _ in exif_raw.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            tags_found.add(tag_name)

        camera_tags_present = tags_found & CAMERA_TAGS
        if len(camera_tags_present) >= 3:
            return 0.1, f"Có {len(camera_tags_present)} trường thông tin máy ảnh (Make, Model...)"
        if len(camera_tags_present) >= 1:
            return 0.4, f"Chỉ có {len(camera_tags_present)} trường thông tin máy ảnh"
        return 0.7, "Có EXIF nhưng không có thông tin máy ảnh cụ thể"
    except Exception as e:
        return 0.5, f"Không đọc được metadata ({e})"


def check_frequency(image_path):
    """
    Trả về điểm nghi ngờ 'AI tạo' dựa trên phân tích tần số (FFT). Ảnh AI thường
    có năng lượng bất thường ở vùng tần số cao do cách mạng sinh ảnh (GAN/diffusion)
    tạo ra hoạ tiết lặp lại rất nhỏ.

    LƯU Ý: đây là ước lượng đơn giản hoá, không phải phương pháp pháp y ảnh đầy đủ.
    Ảnh đã bị nén JPEG mạnh cũng có thể cho điểm số lệch, không chỉ ảnh AI.
    """
    try:
        img = Image.open(image_path).convert("L")  # chuyển ảnh xám để tính FFT đơn giản hơn
        arr = np.array(img, dtype=np.float32)

        # Biến đổi Fourier 2 chiều, dời tần số 0 vào giữa để dễ phân vùng
        f = np.fft.fft2(arr)
        f_shifted = np.fft.fftshift(f)
        magnitude = np.abs(f_shifted)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4  # bán kính chia vùng tần số thấp / cao

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        low_freq_mask = dist_from_center <= radius
        high_freq_mask = dist_from_center > radius

        low_energy = magnitude[low_freq_mask].mean()
        high_energy = magnitude[high_freq_mask].mean()

        # Tỷ lệ năng lượng tần số cao / thấp bất thường (quá cao hoặc quá thấp
        # so với mức thường thấy ở ảnh chụp tự nhiên) được coi là đáng nghi.
        ratio = high_energy / (low_energy + 1e-8)

        # Ngưỡng dưới đây là ước lượng thô, nên dùng để tham khảo, không phải
        # con số đã được kiểm chứng khoa học chặt chẽ.
        if ratio < 0.015 or ratio > 0.08:
            return 0.65, f"Tỷ lệ năng lượng tần số cao/thấp bất thường ({ratio:.4f})"
        return 0.3, f"Tỷ lệ năng lượng tần số cao/thấp trong mức bình thường ({ratio:.4f})"
    except Exception as e:
        return 0.5, f"Không phân tích được tần số ({e})"


def combined_verdict(prob):
    if prob >= FAKE_THRESHOLD:
        return "Nhiều khả năng do AI tạo"
    if prob <= REAL_THRESHOLD:
        return "Nhiều khả năng là ảnh thật"
    return "Không chắc chắn - cần người xem xét"


def analyze(image_path):
    model_prob = model_fake_probability(image_path, use_tta=True)
    meta_prob, meta_reason = check_metadata(image_path)
    freq_prob, freq_reason = check_frequency(image_path)

    combined = (
        WEIGHTS["model"] * model_prob
        + WEIGHTS["metadata"] * meta_prob
        + WEIGHTS["frequency"] * freq_prob
    )

    return {
        "model_prob": model_prob,
        "metadata_prob": meta_prob,
        "metadata_reason": meta_reason,
        "frequency_prob": freq_prob,
        "frequency_reason": freq_reason,
        "combined_prob": combined,
        "verdict": combined_verdict(combined),
    }


def predict(input_path):
    target = Path(input_path)
    if target.is_dir():
        paths = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMG_EXTS)
    else:
        paths = [target]
    if not paths:
        raise SystemExit("Không tìm thấy ảnh nào.")

    for p in paths:
        r = analyze(p)
        print(f"\n{p}")
        print(f"  Mô hình ViT:        {r['model_prob']:.1%}")
        print(f"  Metadata EXIF:       {r['metadata_prob']:.1%}  ({r['metadata_reason']})")
        print(f"  Phân tích tần số:    {r['frequency_prob']:.1%}  ({r['frequency_reason']})")
        print(f"  --> Kết hợp:         {r['combined_prob']:.1%}  -> {r['verdict']}")


def main():
    parser = argparse.ArgumentParser(description="Phát hiện ảnh AI bằng cách kết hợp nhiều tín hiệu")
    parser.add_argument("--input", required=True, help="Đường dẫn ảnh hoặc thư mục")
    args = parser.parse_args()
    predict(args.input)


if __name__ == "__main__":
    main()
