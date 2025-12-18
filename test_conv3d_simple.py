#!/usr/bin/env python3
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(0)}")

x = torch.randn(1, 3, 8, 64, 64, device=device)
print(f"Input: {x.shape}")

# 1. nn.Conv3d
print("\n1. nn.Conv3d:")
conv = torch.nn.Conv3d(3, 16, 3, padding=1).to(device)
y1 = conv(x)
print(f"   Output: {y1.shape}")

# 2. F.conv3d
print("\n2. F.conv3d:")
weight = torch.randn(16, 3, 3, 3, 3, device=device)
bias = torch.randn(16, device=device)
y2 = F.conv3d(x, weight, bias, padding=1)
print(f"   Output: {y2.shape}")

