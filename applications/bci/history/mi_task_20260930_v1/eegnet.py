"""PyTorch EEGNet-8,2 reproduction: temporal, depthwise-spatial, separable blocks.

Architecture reference: Lawhern et al. 2018, DOI 10.1088/1741-2552/aace8c.
Implemented independently; logits replace softmax for CrossEntropyLoss.
"""
import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, channels, samples, F1=8, D=2, F2=16, dropout=0.5, kernel_length=64):
        super().__init__()
        self.spatial = nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False)
        self.features = nn.Sequential(
            nn.ZeroPad2d(((kernel_length-1)//2, kernel_length//2, 0, 0)),
            nn.Conv2d(1, F1, (1, kernel_length), bias=False),
            nn.BatchNorm2d(F1, eps=1e-3, momentum=0.01),
            self.spatial,
            nn.BatchNorm2d(F1 * D, eps=1e-3, momentum=0.01),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, eps=1e-3, momentum=0.01),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout), nn.Flatten(),
        )
        self.classifier = nn.Linear(F2 * (samples // 4 // 8), 2)

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))

    @torch.no_grad()
    def constrain_weights(self):
        for weight, limit in ((self.spatial.weight, 1.0), (self.classifier.weight, 0.25)):
            shape = weight.shape
            rows = weight.reshape(shape[0], -1)
            norms = torch.linalg.vector_norm(rows, dim=1, keepdim=True).clamp_min(1e-8)
            weight.mul_((limit / norms).clamp_max(1).reshape(shape[0], *([1] * (weight.ndim - 1))))
