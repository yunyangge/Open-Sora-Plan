export TASK_QUEUE_ENABLE=0
torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
    -m opensora.sample.sample \
    --model_path /home/save_dir/pretrained/93x640x640_144k_ema \
    --version v1_2 \
    --num_frames 93 \
    --height 352 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/home/save_dir/pretrained/mt5-xxl" \
    --text_prompt examples/dense_output.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/home/save_dir/lzj/wf-vae_trilinear" \
    --save_img_path "./test_t2v_on_npu" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --motion_score 0.95 \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr 