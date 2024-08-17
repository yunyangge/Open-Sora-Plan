torchrun --nnodes=1 --nproc_per_node 2  --master_port 29503 \
    -m opensora.sample.sample_4d \
    --model_path /storage/gyy/hw/Open-Sora-Plan/runs/4d_480p_f24_bs1x8x1_lr1e-5_snrgamma5_0_noiseoffset0_02_ema0_999/checkpoint-10000/model \
    --num_frames 24 \
    --height 480 \
    --width 640 \
    --cache_dir "cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt /storage/gyy/hw/Open-Sora-Plan/test_prompt.txt \
    --conditional_images_path /storage/gyy/hw/Open-Sora-Plan/test_cond_imgs_path.txt \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --save_img_path "./test_4d_pndm" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method PNDM \
    # --compile