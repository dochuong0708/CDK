import cv2
import numpy as np
from PIL import Image


def _to_cv2_bgr(image_input):
    if isinstance(image_input, Image.Image):
        rgb = image_input.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    return None


def _compute_npr_energy(image_bgr):
    """
    NPR (Neighboring Pixel Relationships) - kỹ thuật từ bài báo "Rethinking the
    Up-Sampling Operations in CNN-based Generative Network for Generalizable
    Deepfake Detection" (Tan et al., CVPR 2024).
    https://github.com/chuangchuangtan/NPR-DeepfakeDetection

    Ý tưởng: hầu hết mô hình sinh ảnh (GAN lẫn Diffusion) dùng phép upsampling
    (thu nhỏ rồi phóng to) ở nhiều lớp để tạo ảnh độ phân giải cao, để lại dấu
    vết đặc trưng ở quan hệ giữa các pixel liền kề. Cách đo: thu nhỏ ảnh rồi
    phóng to lại bằng nội suy (mô phỏng đúng thao tác upsampling), rồi so
    sánh với ảnh gốc - phần chênh lệch khuếch đại dấu vết đó.

    Điểm mạnh được báo cáo trong bài: dù chỉ huấn luyện trên ảnh GAN, NPR vẫn
    tổng quát tốt sang nhiều mô hình diffusion chưa từng thấy - đúng vấn đề
    CIFAKE (chỉ có 1 nguồn Stable Diffusion) đang gặp phải.
    """
    h, w = image_bgr.shape[:2]
    if h < 4 or w < 4:
        return 0.0
    small = cv2.resize(image_bgr, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
    reconstructed = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    diff = image_bgr.astype(np.float32) - reconstructed.astype(np.float32)
    return float(np.mean(np.abs(diff)))


def get_color_forensic_adjustment(image_input):
    image = _to_cv2_bgr(image_input)
    if image is None or image.size == 0:
        return {"adjustment": 0.0}

    green = image[:, :, 1].astype(np.float32)
    cfa_score = float(np.std(cv2.Laplacian(green, cv2.CV_32F)))

    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance, cb, cr = cv2.split(ycbcr)
    luminance_variance = cv2.Laplacian(luminance, cv2.CV_64F).var()
    cb_variance = cv2.Laplacian(cb, cv2.CV_64F).var()
    cr_variance = cv2.Laplacian(cr, cv2.CV_64F).var()
    chroma_ratio = ((cb_variance + cr_variance) / 2.0) / (luminance_variance + 1e-5)

    adjustment = 0.0
    if cfa_score < 18.0 and chroma_ratio < 0.06:
        adjustment += 0.12
    elif cfa_score < 25.0 and chroma_ratio < 0.1:
        adjustment += 0.06
    if cfa_score > 35.0 and chroma_ratio > 0.15:
        adjustment -= 0.08

    # NPR: artifact upsampling thường "mượt" bất thường ở ảnh do mô hình sinh
    # ảnh tạo ra (độ chênh lệch thấp sau khi thu-phóng lại), trong khi ảnh
    # chụp thật có kết cấu nhiễu tự nhiên nhiều hơn.
    # LƯU Ý: ngưỡng dưới đây là ước lượng ban đầu, CHƯA được hiệu chỉnh trên
    # dữ liệu thật (nên chạy calibrate.py để kiểm chứng/tinh chỉnh con số này,
    # giống cách đã làm với trọng số trong multi_signal_detector.py).
    npr_energy = _compute_npr_energy(image)
    if npr_energy < 2.0:
        adjustment += 0.05
    elif npr_energy > 8.0:
        adjustment -= 0.03

    # Trả thêm các số thô (cfa_score, chroma_ratio, npr_energy) để debug_pipeline.py
    # (và bất kỳ ai cần soi từng bước) có thể in ra riêng từng tín hiệu, thay vì
    # chỉ thấy mỗi "adjustment" tổng hợp cuối cùng.
    return {
        "adjustment": adjustment,
        "cfa_score": cfa_score,
        "chroma_ratio": chroma_ratio,
        "npr_energy": npr_energy,
    }
