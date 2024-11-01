
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29512 \
    -m opensora.sample.sample \
    --model_path mmdit13b_dense_rf_bs8192_lr1e-4_max1x256x256_min1x192x192_emaclip99_recap_coyo_merge_1025/checkpoint-6705/model_ema \
    --version v1_5 \
    --num_frames 1 \
    --height 256 \
    --width 256 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/storage/cache_dir/t5-v1_1-xl" \
    --text_encoder_name_2 "/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
    --text_prompt examples/sora.txt \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/Middle888" \
    --save_img_path "./rf_1x256x256_v1_5_13b_cfg7.0_s100_dense46k_7kema" \
    --fps 18 \
    --guidance_scale 7.0 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method FlowMatchEulerDiscrete \
    --seed 1234 \
D