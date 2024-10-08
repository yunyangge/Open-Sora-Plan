
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
    -m opensora.sample.sample \
    --model_path /storage/ongoing/9.29/mmdit/Open-Sora-Plan/train_v1_3_any93x352x640_min320_vpre_nomotion_fps18_allvid_hqimg_aes5.25_nozsnr_snr5.0/checkpoint-2000/model_ema \
    --version v1_3 \
    --num_frames 93 \
    --height 352 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl" \
    --text_prompt examples/sora_refine.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/latent8" \
    --save_img_path "./test_v1_3_nomo_93x352x640_sora_s200_9k_after_newnew_ddim_nozsnr_snr5" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 200 \
    --max_sequence_length 512 \
    --sample_method DDIM \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction"