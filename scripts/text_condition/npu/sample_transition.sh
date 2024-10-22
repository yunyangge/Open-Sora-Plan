export TASK_QUEUE_ENABLE=0
torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
    -m opensora.sample.sample \
    --model_type "transition" \
    --model_path "/home/save_dir/pretrained/i2v_ckpt14777_ema" \
    --version v1_3 \
    --num_frames 93 \
    --crop_for_hw \
    --height 352 \
    --width 640 \
    --max_hxw 236544 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/home/image_data/mt5-xxl" \
    --text_prompt "/home/image_data/chengxinhua/Open-Sora-Plan/inference/prompt.txt" \
    --conditional_pixel_values_path "/home/image_data/chengxinhua/Open-Sora-Plan/inference/cond_paths.txt" \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/home/save_dir/lzj/formal_8dim/latent8" \
    --save_img_path "./transition_inference" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --seed 519 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr
# 2>&1 | tee -a "logs/log_$(date +'%Y%m%d_%H%M%S').txt"