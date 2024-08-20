# --save_img_path "./sample_video_26500ema_61x480p_k333_s122_cfg5.0_step50" \
# CUDA_VISIBLE_DEVICES=7 python opensora/sample/sample_t2v.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8  --master_port 29504 \
    -m opensora.sample.sample_t2v_ddp \
    --model_path /storage/ongoing/new/7.19anyres/Open-Sora-Plan/bs32x8x1_anyx93x320x320_fps16_lr1e-5_snr5_noioff0.02_ema9999_sparse1d4_dit_l_mt5xxl_alldata100m/checkpoint-250000/model_ema \
    --version 65x512x512 \
    --num_frames 93 \
    --height 320 \
    --width 160 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/sora.txt \
    --ae CausalVAEModel_D8_4x8x8 \
    --ae_path "/storage/dataset/new488dim8/last" \
    --save_img_path "./sample_video_dit_vae8_newmodel_anyx93x320x160_sora_m0.05_368k" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "sparsedit" \
    --motion_score 0.05