export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_LOG_LEVEL=7
export MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0   
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export AMD_LOG_LEVEL=3

mkdir -p models/vae && wget -O models/vae/wan_2.1_vae.safetensors "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

MIOpenDriver convfp16 -n 1 -c 16 --in_d 5 -H 104 -W 60 -k 16 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1 -i 1 

uv run test_conv3d_simple.py > test.log  2>&1 
#ShaderName : void ck_tile::kentry<ck_tile::gfx11_t, 2, ck_tile::GroupedConvolutionForwardKernel
uv run  main.py --listen --port 2828 --force-channels-last --bf16-vae --bf16-unet --use-flash-attention  2>&1 | tee output.log

python main.py --listen --port 2828 --force-channels-last --bf16-vae --bf16-unet --use-flash-attention 2>&1 | tee output.log
