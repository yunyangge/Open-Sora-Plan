export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 1 --master_port 29512 \
    -m opensora.sample.sample \
    --model_path /home/ma-user/work/checkpoint/gyy/runs/t2v_suv_from_dit_288x512_384x384_32x8x16_lr1e-4_wd1e-5_eps1e-15_total_recap/checkpoint-52500/model_ema \
    --version v1_5 \
    --num_frames 1 \
    --height 288 \
    --width 512 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/home/ma-user/work/checkpoint/pretrained/t5/t5-v1_1-xl" \
    --text_encoder_name_2 "/home/ma-user/work/checkpoint/pretrained/clip" \
    --text_prompt examples/sora.txt \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/home/ma-user/work/checkpoint/pretrained/vae_32dim" \
    --save_img_path "./test_suv_9_16_53k_step100_lq1000_cfg7.5_res0.7_hq_ema" \
    --fps 18 \
    --guidance_scale 7.5 \
    --guidance_rescale 0.7 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method FlowMatchEulerDiscrete \
    --seed 1024 \
    --num_samples_per_prompt 1  \
    --weight_dtype fp16 \
    --vae_dtype fp16 \
    --use_linear_quadratic_schedule