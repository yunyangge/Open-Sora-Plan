CUDA_VISIBLE_DEVICES=7 python opensora/sample/sample_4d.py \
    --model_path /storage/gyy/hw/Open-Sora-Plan/test_image_styled_vae_256x256x29/checkpoint-15000/model_ema \
    --num_frames 29 \
    --height 256 \
    --width 256 \
    --cache_dir "cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_1.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --save_img_path "./test_image_styled_vae" \
    --fps 24 \
    --guidance_scale 5.0 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method PNDM \
    --model_type "dit" \
    # --compile