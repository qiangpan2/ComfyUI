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
    logging.info("[AMD Conv Fix] Applying AMD ROCm Conv2d/Conv3d patch")
    logging.info(f"[AMD Conv Fix] Target dtype: {target_dtype}")
    logging.info(f"[AMD Conv Fix] ROCm version: {torch.version.hip}")
    logging.info(f"[AMD Conv Fix] Device: {torch.cuda.get_device_name(0)}")
    
    _orig_conv2d = torch.nn.functional.conv2d
    _orig_conv3d = torch.nn.functional.conv3d
    
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
    
    torch.nn.functional.conv2d = _amd_conv2d_wrapper
    torch.nn.functional.conv3d = _amd_conv3d_wrapper
    
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

