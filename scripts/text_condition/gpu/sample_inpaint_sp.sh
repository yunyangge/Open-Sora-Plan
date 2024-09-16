
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m opensora.sample.sample_inpaint_sp  \
    --model_path /storage/gyy/hw/Open-Sora-Plan/runs/inpaint_93x1280x1280_stage3_gpu/checkpoint-1692/model \
    --num_frames 93 \
    --height 736 \
    --width 1280 \
    --cache_dir "../cache_dir" \
    --text_encoder_name /storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl \
    --text_prompt /storage/gyy/hw/Open-Sora-Plan/prompt_inpaint.txt \
    --conditional_images_path /storage/gyy/hw/Open-Sora-Plan/cond_imgs_path_inpaint.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/wf-vae_trilinear" \
    --save_img_path "./test_video_inpaint_final_test_normal" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --sample_method EulerAncestralDiscrete \
    --motion_score 0.9 \
    --seed 26468453 \
    --enable_tiling \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr 