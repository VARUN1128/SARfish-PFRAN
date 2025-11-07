import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Shape-Scale Convolution ----
class SSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# ---- Pyramid Feature Aggregation (PFA) ----
class PFA(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.fuse = nn.Conv2d(in_channels_list[-1], out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        # Expecting list of feature maps from different levels
        fused = self.fuse(features[-1])
        return [self.relu(fused)]

# ---- Center-Aware Head (CAC) ----
class CACHead(nn.Module):
    def __init__(self, original_predictor):
        super().__init__()
        self.predictor = original_predictor  # Start with existing head

    def forward(self, x):
        # Later, you’ll add center-offset prediction here
        return self.predictor(x)
