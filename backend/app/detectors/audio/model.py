"""Small spectrogram CNN baseline; class 0=real, class 1=fake."""
import torch
from torch import nn
from torch.nn import functional as F


class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [1, 16, 32, 64]
        for incoming, outgoing in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(incoming, outgoing, 3, padding=1),
                           nn.ReLU(), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features.mean(dim=(2, 3)))


def grad_cam(model, x, target=1):
    """Positive attribution to a specified class, NOT a forgery segmentation mask."""
    with torch.enable_grad():
        features = model.features(x.detach().requires_grad_(True))
        logits = model.classifier(features.mean(dim=(2, 3)))
        gradients, = torch.autograd.grad(logits[:, target].sum(), features)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        maximum = cam.amax(dim=(2, 3), keepdim=True)
        cam = cam / maximum.clamp_min(1e-12)
    return logits.detach(), cam.detach()[0, 0].cpu().numpy()
