#!/usr/bin/env python3
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")
dtype = torch.bfloat16

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Dtype: {dtype}")

x = torch.randn(1, 3, 8, 64, 64, device=device, dtype=dtype)
print(f"Input: {x.shape}, dtype: {x.dtype}")

# 1. nn.Conv3d - contiguous
print("\n1. nn.Conv3d (contiguous, bf16):")
conv = torch.nn.Conv3d(3, 16, 3, padding=1).to(device).to(dtype)
y1 = conv(x)
print(f"   Output: {y1.shape}, dtype: {y1.dtype}")

# 2. nn.Conv3d - channels_last_3d
print("\n2. nn.Conv3d (channels_last_3d, bf16):")
conv_cl = torch.nn.Conv3d(3, 16, 3, padding=1).to(device).to(dtype)
conv_cl.weight.data = conv_cl.weight.data.to(memory_format=torch.channels_last_3d)
x_cl = x.to(memory_format=torch.channels_last_3d)
y2 = conv_cl(x_cl)
print(f"   Output: {y2.shape}, dtype: {y2.dtype}")
print(f"   Is channels_last_3d: {y2.is_contiguous(memory_format=torch.channels_last_3d)}")

# 3. F.conv3d - contiguous
print("\n3. F.conv3d (contiguous, bf16):")
weight = torch.randn(16, 3, 3, 3, 3, device=device, dtype=dtype)
bias = torch.randn(16, device=device, dtype=dtype)
y3 = F.conv3d(x, weight, bias, padding=1)
print(f"   Output: {y3.shape}, dtype: {y3.dtype}")

# 4. F.conv3d - channels_last_3d
print("\n4. F.conv3d (channels_last_3d, bf16):")
weight_cl = weight.to(memory_format=torch.channels_last_3d)
y4 = F.conv3d(x_cl, weight_cl, bias, padding=1)
print(f"   Output: {y4.shape}, dtype: {y4.dtype}")
print(f"   Is channels_last_3d: {y4.is_contiguous(memory_format=torch.channels_last_3d)}")
