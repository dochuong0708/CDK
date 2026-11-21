import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from app.models.schemas import Analysis
from .multi_signal_detector import analyze as analyze_multi_signal
from .vision_xai import generate_visual_xai

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
FAKE_THRESHOLD = 0.50
REAL_THRESHOLD = 0.40


class ImageDetector:
    def __init__(self, model_path: Path | None = None) -> None:
        configured_path = os.getenv("DEEPGUARD_IMAGE_MODEL_PATH")
        self.model_path = model_path or (
            Path(configured_path)
            if configured_path
            else Path(__file__).resolve().parents[3] / "models" / "detector.pt"
        )
        self._model = None
        self._device = None
        self._fake_idx = None
        self._eval_transform = None
        self._patch_transform = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
        except ImportError as error:
            raise RuntimeError(
                "Image detection dependencies are missing; install backend/requirements-image.txt"
            ) from error

        if not self.model_path.is_file():
            raise RuntimeError(f"Image detector checkpoint not found: {self.model_path}")

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["state_dict"]
            fake_idx = checkpoint["fake_idx"]
            if not isinstance(fake_idx, int) or fake_idx not in (0, 1):
                raise ValueError("checkpoint has an invalid fake class index")
            if not isinstance(state_dict, dict):
                raise ValueError("checkpoint state_dict is invalid")
            if "features.0.0.weight" not in state_dict:
                raise ValueError(
                    "checkpoint is not a ConvNeXt-Tiny model; train/export a compatible "
                    "two-class checkpoint before using this detector"
                )
            classifier_weight = state_dict.get("classifier.2.weight")
            if not isinstance(classifier_weight, torch.Tensor) or tuple(classifier_weight.shape) != (
                2,
                768,
            ):
                raise ValueError(
                    "checkpoint must contain a ConvNeXt-Tiny classifier with two output classes"
                )

            model = models.convnext_tiny(weights=None)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
            model.load_state_dict(state_dict)
            model.to(device).eval()
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Could not load ConvNeXt-Tiny image detector checkpoint: {self.model_path}. "
                "A compatible two-class ConvNeXt-Tiny checkpoint is required at "
                "backend/models/detector.pt."
            ) from error

        self._model = model
        self._device = device
        self._fake_idx = fake_idx
        self._eval_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        self._patch_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def analyze(self, file_bytes: bytes, filename: str) -> Analysis:
        if Path(filename).suffix.lower() not in IMG_EXTS:
            raise ValueError("Unsupported image format")

        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Image detection dependencies are missing; install backend/requirements-image.txt"
            ) from error

        try:
            with Image.open(BytesIO(file_bytes)) as source:
                source.verify()
            with Image.open(BytesIO(file_bytes)) as source:
                exif = source.getexif()
                image = source.convert("RGB")
        except (OSError, ValueError) as error:
            raise ValueError("The uploaded file is not a valid supported image") from error

        self._load_model()
        # Dự đoán kèm theo thông tin giải thích cục bộ (tọa độ patch & chỉ số pháp y màu sắc)
        local_probability, xai_details = self._predict_with_details(image)
        multi_signal = analyze_multi_signal(image, exif, local_probability)
        probability = multi_signal["combined_probability"]

        risk_score = round(probability * 100)
        if probability >= FAKE_THRESHOLD:
            label = "Review recommended"
        elif probability <= REAL_THRESHOLD:
            label = "Likely authentic"
        else:
            label = "Uncertain"

        # Tổng hợp danh sách bằng chứng giải thích (Textual XAI)
        evidence = [
            f"Combined AI-generation risk score: {risk_score}%",
            f"ConvNeXt-Tiny + color forensics (local model): {local_probability:.1%}",
            *xai_details["reasons"],
            f"Metadata: {multi_signal['metadata_reason']}",
            f"Frequency: {multi_signal['frequency_reason']}",
        ]

        # Tạo bản đồ nhiệt Grad-CAM & ảnh khoanh vùng vi phạm nếu ảnh có dấu hiệu nghi ngờ
        visual_xai = None
        if probability > REAL_THRESHOLD:
            visual_xai = generate_visual_xai(
                image=image,
                model=self._model,
                device=self._device,
                fake_idx=self._fake_idx,
                eval_transform=self._eval_transform,
                suspicious_boxes=xai_details["suspicious_boxes"],
            )

        analysis_payload = {
            "analysisId": f"img-{uuid4().hex}",
            "modality": "Image",
            "label": label,
            "riskScore": risk_score,
            "confidence": max(probability, 1 - probability),
            "evidence": evidence,
            "detector": "image-convnexttiny-multisignal-v1",
            "input": filename,
            "createdAt": datetime.now(),
        }

        # Đính kèm dữ liệu trực quan nếu schema Analysis hỗ trợ extra fields hoặc thuộc tính xai
        if visual_xai is not None:
            if "xai" in Analysis.model_fields:
                analysis_payload["xai"] = visual_xai
            else:
                # Dự phòng: lưu Base64 vào cuối danh sách evidence nếu chưa cập nhật schemas.py
                evidence.append(f"XAI_HEATMAP_BASE64:{visual_xai['heatmap_base64']}")
                if xai_details["suspicious_boxes"]:
                    evidence.append(f"XAI_BBOX_BASE64:{visual_xai['annotated_base64']}")

        return Analysis(**analysis_payload)

    def _predict_with_details(self, image) -> tuple[float, dict]:
        try:
            import numpy as np
            import torch

            from .color_analysis import get_color_forensic_adjustment
        except ImportError as error:
            raise RuntimeError(
                "Image detection dependencies are missing; install backend/requirements-image.txt"
            ) from error

        width, height = image.size
        reasons = []
        suspicious_boxes = []

        global_tensor = self._eval_transform(image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            global_output = self._model(global_tensor)
            global_probability = float(
                torch.softmax(global_output, dim=1)[0, self._fake_idx].cpu().item()
            )

            if width < 224 or height < 224:
                model_probability = global_probability
                reasons.append("Image is smaller than a patch; global image analysis completed")
            else:
                image_tensor = self._patch_transform(image).unsqueeze(0)
                patches = image_tensor.unfold(2, 224, 112).unfold(3, 224, 112)
                patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 224, 224)

                patch_probabilities = []
                for start in range(0, patches.size(0), 32):
                    batch = patches[start : start + 32].to(self._device)
                    output = self._model(batch)
                    patch_probabilities.extend(
                        torch.softmax(output, dim=1)[:, self._fake_idx].cpu().tolist()
                    )

                scores_arr = np.array(patch_probabilities)
                k = min(3, len(scores_arr))
                top_indices = np.argsort(scores_arr)[::-1][:k]
                top_k_probs = scores_arr[top_indices]
                top_k_mean = float(np.mean(top_k_probs))

                # Lưu tọa độ các vùng nghi vấn nhất (score >= 0.60) để vẽ Bounding Box
                num_cols = (width - 224) // 112 + 1
                for idx in top_indices:
                    score = float(scores_arr[idx])
                    if score >= 0.60:
                        row, col = int(idx) // num_cols, int(idx) % num_cols
                        x, y = col * 112, row * 112
                        suspicious_boxes.append([x, y, x + 224, y + 224])

                if any(p >= 0.85 for p in patch_probabilities):
                    model_probability = top_k_mean
                    max_p = float(top_k_probs[0])
                    box_info = f" at region {suspicious_boxes[0][:2]}" if suspicious_boxes else ""
                    reasons.append(
                        f"Localized anomaly override triggered: max patch risk {max_p:.1%}{box_info}"
                    )
                else:
                    model_probability = 0.6 * global_probability + 0.4 * top_k_mean
                    reasons.append("Global image and sliding-window patch analysis completed")

        forensic = get_color_forensic_adjustment(image)
        forensic_adjustment = forensic["adjustment"]
        cfa_score = forensic.get("cfa_score", 99.0)
        npr_energy = forensic.get("npr_energy", 99.0)

        if cfa_score < 18.0:
            reasons.append(f"Color forensics: missing natural camera CFA pattern (score={cfa_score:.2f})")
        if npr_energy < 2.0:
            reasons.append(f"Color forensics: overly smooth pixel upsampling artifact (NPR={npr_energy:.2f})")

        color_probability = float(np.clip(model_probability + forensic_adjustment, 0.0, 1.0))
        return color_probability, {
            "reasons": reasons,
            "suspicious_boxes": suspicious_boxes,
        }