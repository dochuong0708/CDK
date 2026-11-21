from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from PIL import Image

# Trọng số đã hiệu chỉnh: "model" ở đây là local_probability do ImageDetector
# tính sẵn (ConvNeXt-Tiny + patch scanning + color forensics, không còn dùng
# Hugging Face/ViT), nên vẫn chiếm phần lớn trọng số. Metadata vẫn thấp vì dễ
# bị nhiễu bởi việc ảnh đã qua nén/đồng bộ qua các nền tảng (OneDrive, Zalo,
# Messenger...) khiến EXIF bị xoá dù ảnh thật 100%.
#
# frequency được TĂNG từ 0.08 lên 0.13 sau khi quan sát một ảnh AI chất
# lượng cao (iStockphoto) mà mô hình chính đoán sai (10% - "thật"), trong
# khi tín hiệu tần số đã đúng hướng (nghiêng về AI). Đồng thời hàm
# check_frequency() cũng được nâng cấp dùng azimuthal averaging thay vì tỷ
# lệ 2 vùng thô sơ trước đây, nên đáng tin hơn để tăng trọng số.
# CẢNH BÁO: quyết định tăng này mới dựa trên 1 ví dụ, CHƯA đủ bằng chứng.
# Cần chạy calibrate.py trên một tập dữ liệu đa dạng (nhiều ảnh AI chất
# lượng cao + nhiều ảnh thật) để xác nhận con số này có thực sự tối ưu hay
# không, tránh chỉnh theo cảm tính dựa trên 1 trường hợp (risk overfitting).
# (Lịch sử: model=0.70/metadata=0.15/frequency=0.15 ban đầu
#        -> model=0.85/metadata=0.07/frequency=0.08 sau vụ ảnh OneDrive
#        -> model=0.80/metadata=0.07/frequency=0.13 sau vụ ảnh iStockphoto)
WEIGHTS = {"model": 1.0, "metadata": 0.0, "frequency": 0.0}

CAMERA_TAGS = {
    "Make",
    "Model",
    "LensModel",
    "FNumber",
    "ExposureTime",
    "ISOSpeedRatings",
    "FocalLength",
}


class MultiSignalResult(TypedDict):
    model_probability: float
    metadata_probability: float
    metadata_reason: str
    frequency_probability: float
    frequency_reason: str
    combined_probability: float


def check_metadata(exif: Image.Exif) -> tuple[float, str]:
    """
    LƯU Ý QUAN TRỌNG: thiếu EXIF KHÔNG đồng nghĩa với ảnh AI. Phần lớn ảnh
    thật ngày nay đã mất EXIF sau khi đi qua OneDrive, Zalo, Messenger, mạng
    xã hội... Vì vậy trường hợp "không có EXIF" được coi là TRUNG LẬP (0.5),
    không nghiêng về phía AI như trước đây (0.75). Chỉ khi ảnh THỰC SỰ CÓ đủ
    thông tin máy ảnh, điểm nghi ngờ AI mới được hạ xuống thấp - đây là tín
    hiệu một chiều: có thì đáng tin là ảnh thật, không có thì không kết luận
    được gì cả.
    """
    from PIL import ExifTags

    try:
        tags_found = {ExifTags.TAGS.get(tag_id, str(tag_id)) for tag_id in exif}
    except (OSError, ValueError):
        return 0.5, "Could not read image metadata"

    if not exif:
        return 0.5, "No EXIF metadata found (neutral - common for images shared via cloud/apps)"

    camera_tags_present = tags_found & CAMERA_TAGS
    if len(camera_tags_present) >= 3:
        return 0.1, f"{len(camera_tags_present)} camera metadata fields found"
    if camera_tags_present:
        return 0.3, f"{len(camera_tags_present)} camera metadata field(s) found"
    return 0.5, "EXIF metadata has no camera-specific fields (neutral)"


def check_frequency(image: Image.Image) -> tuple[float, str]:
    """
    Phân tích phổ Fourier bằng azimuthal averaging (lấy trung bình phổ theo
    từng vòng tròn bán kính quanh tâm) - kỹ thuật chuẩn trong các nghiên cứu
    phát hiện ảnh GAN/CNN qua miền tần số:
      - Frank et al., ICML 2020, "Leveraging Frequency Analysis for Deep Fake
        Image Recognition" (github.com/RUB-SysSec/GANDCTAnalysis)
      - Chandrasegaran et al., CVPR 2021, "A Closer Look at Fourier Spectrum
        Discrepancies for CNN-generated Images Detection"
      - Durall et al., 2020, "Watch your Up-Convolution"

    Ý TƯỞNG: ảnh tự nhiên có phổ suy giảm mượt theo dạng xấp xỉ 1/f khi tần
    số tăng (đường cong dốc xuống đều). Các mô hình sinh ảnh (do dùng phép
    upsampling nhiều lớp) để lại hai dấu hiệu bất thường trên đường cong này:
      1. Suy giảm CHẬM hơn bình thường ở vùng tần số cao (độ dốc gần 0 hoặc
         dương) - tức ảnh "giữ" quá nhiều năng lượng tần số cao so với ảnh
         chụp tự nhiên.
      2. GỢN SÓNG bất thường (artifact dạng caro/checkerboard) thay vì suy
         giảm mượt mà.

    Trước đây hàm này chỉ chia ảnh thành 2 vùng thấp/cao rồi lấy tỷ lệ trung
    bình - khá thô, không nắm được hình dạng thật của đường cong suy giảm.
    Bản này tính đường cong suy giảm đầy đủ rồi đo cả độ dốc lẫn độ gợn sóng.

    LƯU Ý: các ngưỡng slope/bumpiness dưới đây là ước lượng ban đầu dựa trên
    tài liệu tham khảo, CHƯA được hiệu chỉnh trên dữ liệu thật - nên chạy
    calibrate.py sau khi có đủ dữ liệu để kiểm chứng/tinh chỉnh.
    """
    import numpy as np

    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = grayscale.shape
    if height < 32 or width < 32:
        return 0.5, "Image is too small for frequency analysis"

    fourier = np.fft.fftshift(np.fft.fft2(grayscale))
    log_magnitude = np.log(np.abs(fourier) + 1e-8)

    # Azimuthal averaging: gộp mọi điểm có cùng khoảng cách tới tâm (cùng
    # bán kính = cùng tần số không gian) thành một giá trị trung bình, biến
    # phổ 2D thành một đường cong 1D theo bán kính.
    center_y, center_x = height // 2, width // 2
    y, x = np.indices((height, width))
    radius_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(int)
    max_radius = radius_map.max()
    sum_per_radius = np.bincount(radius_map.ravel(), log_magnitude.ravel(), minlength=max_radius + 1)
    count_per_radius = np.bincount(radius_map.ravel(), minlength=max_radius + 1)
    radial_profile = sum_per_radius / np.maximum(count_per_radius, 1)

    # Bỏ vùng tâm (DC, không ổn định) và rìa ngoài cùng (ít điểm, nhiễu)
    lo = max(2, int(0.02 * max_radius))
    hi = int(0.9 * max_radius)
    profile = radial_profile[lo:hi]
    if len(profile) < 10:
        return 0.5, "Image is too small for a reliable frequency profile"

    radii = np.arange(len(profile))
    slope, intercept = np.polyfit(radii, profile, 1)
    expected = slope * radii + intercept
    bumpiness = float(np.std(profile - expected))

    probability = 0.3
    reasons = []
    if slope > -0.01:
        probability += 0.2
        reasons.append(f"slow high-frequency decay (slope={slope:.4f})")
    if bumpiness > 0.5:
        probability += 0.15
        reasons.append(f"unusual ripple in spectrum (bumpiness={bumpiness:.3f})")

    probability = min(probability, 0.9)
    if reasons:
        return probability, "; ".join(reasons)
    return probability, f"Natural-looking frequency decay (slope={slope:.4f})"


def analyze(
    image: Image.Image,
    exif: Image.Exif,
    model_probability: float,
) -> MultiSignalResult:
    metadata_probability, metadata_reason = check_metadata(exif)
    frequency_probability, frequency_reason = check_frequency(image)
    combined_probability = (
        WEIGHTS["model"] * model_probability
        + WEIGHTS["metadata"] * metadata_probability
        + WEIGHTS["frequency"] * frequency_probability
    )

    return {
        "model_probability": model_probability,
        "metadata_probability": metadata_probability,
        "metadata_reason": metadata_reason,
        "frequency_probability": frequency_probability,
        "frequency_reason": frequency_reason,
        "combined_probability": combined_probability,
    }
