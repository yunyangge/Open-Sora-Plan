export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29512 \
    -m opensora.sample.sample \
    --model_path /home/save_dir/runs/SUV/suv/suv_video_bs32x8x2_lr1e-4_wd1e-4_eps1e-15_f65_288x512_final/checkpoint-57963/model_ema \
    --version v1_5 \
    --num_frames 65 \
    --height 288 \
    --width 512 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/home/save_dir/pretrained/t5/t5-v1_1-xl" \
    --text_encoder_name_2 "/home/save_dir/pretrained/clip/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189" \
    --text_prompt examples/sora.txt \
    --ae WFVAEModel_D32_8x8x8 \
    --ae_path "/home/save_dir/lzj/Middle888" \
    --save_img_path "./SUV/suv/video_final_288_512_step100_ema" \
    --fps 18 \
    --guidance_scale 4.0 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method FlowMatchEulerDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --ae_dtype 'fp16' \
    --weight_dtype 'fp16' \
    --enable_tiling \
    --use_linear_quadratic_schedule