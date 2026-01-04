import torch
import logging

def apply_amd_conv_fix():
    if not (hasattr(torch, 'version') and torch.version.hip):
        logging.info("[AMD Conv Fix] Not an AMD GPU, skipping patch")
        return False
    
    if not torch.cuda.is_available():
        logging.info("[AMD Conv Fix] No CUDA device available, skipping patch")
        return False
    
    # prefer bf16
    target_dtype = torch.bfloat16
    
    logging.info("=" * 60)
    logging.info("[AMD Conv Fix] Applying AMD ROCm Conv2d/Conv3d/ConvTranspose/GridSample patch")
    logging.info(f"[AMD Conv Fix] Target dtype: {target_dtype}")
    logging.info(f"[AMD Conv Fix] ROCm version: {torch.version.hip}")
    logging.info(f"[AMD Conv Fix] Device: {torch.cuda.get_device_name(0)}")
    
    _orig_conv2d = torch.nn.functional.conv2d
    _orig_conv3d = torch.nn.functional.conv3d
    _orig_conv_transpose2d = torch.nn.functional.conv_transpose2d
    _orig_conv_transpose3d = torch.nn.functional.conv_transpose3d
    _orig_grid_sample = torch.nn.functional.grid_sample
    
    def _amd_conv2d_wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """Conv2d wrapper:  fp32 -> bf16/fp16"""
        if input.dtype == torch.float32 and input.is_cuda:
            input = input.to(target_dtype)
        if weight.dtype == torch.float32 and weight.is_cuda:
            weight = weight.to(target_dtype)
        if bias is not None and bias.dtype == torch.float32 and bias.is_cuda:
            bias = bias.to(target_dtype)
        
        return _orig_conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    def _amd_conv3d_wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """Conv3d wrapper:  fp32 -> bf16/fp16"""
        if input.dtype == torch.float32 and input.is_cuda:
            input = input.to(target_dtype)
        if weight.dtype == torch.float32 and weight.is_cuda:
            weight = weight.to(target_dtype)
        if bias is not None and bias.dtype == torch.float32 and bias.is_cuda:
            bias = bias.to(target_dtype)
        
        return _orig_conv3d(input, weight, bias, stride, padding, dilation, groups)
    
    import torch.nn.functional as F

    def _to_2tuple(v):
        if isinstance(v, tuple):
            return v
        return (v, v)

    def zero_insert_upsample(x, stride):
        if stride == (1, 1):
            return x

        B, C, H, W = x.shape
        sH, sW = stride

        out = torch.zeros(
            B, C,
            H * sH,
            W * sW,
            device=x.device,
            dtype=x.dtype,
        )

        out[:, :, ::sH, ::sW] = x
        return out


    def _amd_conv_transpose2d_wrapper(
        x, weight, bias=None,
        stride=1, padding=0, output_padding=0,
        groups=1, dilation=1
    ):
        stride = _to_2tuple(stride)
        padding = _to_2tuple(padding)
        output_padding = _to_2tuple(output_padding)
        dilation = _to_2tuple(dilation)

        # forward-only
        x = x.detach().to(torch.bfloat16)
        weight = weight.detach().to(torch.bfloat16)
        if bias is not None:
            bias = bias.detach().to(torch.bfloat16)

        # zero-insert upsample
        x = zero_insert_upsample(x, stride)

        # flip kernel
        weight_flip = weight.flip([2, 3]).permute(1, 0, 2, 3)

        kH, kW = weight.shape[2:]

        pad_h = kH - 1 - padding[0]
        pad_w = kW - 1 - padding[1]

        y = F.conv2d(
            x,
            weight_flip,
            bias=bias,
            stride=1,
            padding=(pad_h, pad_w),
            dilation=dilation,
            groups=groups,
        )

        # ===== 關鍵：PyTorch 官方輸出尺寸 =====
        H_in, W_in = x.shape[2] // stride[0], x.shape[3] // stride[1]

        H_out = (
            (H_in - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kH - 1)
            + output_padding[0]
            + 1
        )

        W_out = (
            (W_in - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (kW - 1)
            + output_padding[1]
            + 1
        )

        # ===== 強制裁切（避免 1152 / 1153 問題）=====
        y = y[:, :, :H_out, :W_out]

        return y

    def _amd_conv_transpose3d_wrapper(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        """ConvTranspose3d wrapper:  fp32 -> bf16/fp16"""
        if input.dtype == torch.float32 and input.is_cuda:
            input = input.to(target_dtype)
        if weight.dtype == torch.float32 and weight.is_cuda:
            weight = weight.to(target_dtype)
        if bias is not None and bias.dtype == torch.float32 and bias.is_cuda:
            bias = bias.to(target_dtype)
        
        return _orig_conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)
    
    def _amd_grid_sample_wrapper(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
        """grid_sample wrapper: ensure input and grid dtypes match for bf16/fp16"""
        # If input is bf16/fp16 on CUDA, convert grid to match
        if input.is_cuda and input.dtype in (torch.bfloat16, torch.float16):
            if grid.dtype == torch.float32:
                grid = grid.to(input.dtype)
        # If grid is bf16/fp16 on CUDA, convert input to match
        elif grid.is_cuda and grid.dtype in (torch.bfloat16, torch.float16):
            if input.dtype == torch.float32:
                input = input.to(grid.dtype)
        
        return _orig_grid_sample(input, grid, mode, padding_mode, align_corners)
    
    torch.nn.functional.conv2d = _amd_conv2d_wrapper
    torch.nn.functional.conv3d = _amd_conv3d_wrapper
    torch.nn.functional.conv_transpose2d = _amd_conv_transpose2d_wrapper
    torch.nn.functional.conv_transpose3d = _amd_conv_transpose3d_wrapper
    torch.nn.functional.grid_sample = _amd_grid_sample_wrapper
    
    return True


try:
    _patch_applied = apply_amd_conv_fix()
    

    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
    __AMD_CONV_FIX_LOADED__ = True
    
except Exception as e:
    logging.error(f"[AMD Conv Fix] Failed to apply patch: {e}")
    import traceback
    traceback.print_exc()

