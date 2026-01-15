export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_LOG_LEVEL=7
export MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0   
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export AMD_LOG_LEVEL=3

#ShaderName : void ck_tile::kentry<ck_tile::gfx11_t, 2, ck_tile::GroupedConvolutionForwardKernel
uv run  main.py --listen --port 2828 --force-channels-last --bf16-vae --bf16-unet 2>&1 | tee output.log
