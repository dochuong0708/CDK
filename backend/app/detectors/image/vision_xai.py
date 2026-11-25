import base64
import io
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_visual_xai(
    image: Image.Image,
    model,
    device,
    fake_idx: int,
    eval_transform,
    suspicious_boxes: list[list[int]] | None = None,
) -> dict[str, str]:
    """Tạo bản đồ nhiệt Grad-CAM và ảnh khoanh vùng vi phạm dưới dạng Base64."""
    with torch.enable_grad():
        model.eval()
        target_layers = [model.layer4[-1]] if hasattr(model, "layer4") else [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        input_tensor = eval_transform(image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        grayscale_cam = cam(
            input_tensor=input_tensor, targets=[ClassifierOutputTarget(fake_idx)]
        )[0, :]

    img_float = np.float32(image.resize((224, 224))) / 255.0
    heatmap_pil = Image.fromarray(show_cam_on_image(img_float, grayscale_cam, use_rgb=True))

    annotated_pil = image.copy()
    if suspicious_boxes:
        draw = ImageDraw.Draw(annotated_pil)
        for box in suspicious_boxes:
            draw.rectangle(box, outline="red", width=4)

    return {
        "heatmap_base64": _pil_to_base64(heatmap_pil),
        "annotated_base64": _pil_to_base64(annotated_pil),
    }