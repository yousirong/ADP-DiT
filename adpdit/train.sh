#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29500
export PYTHONPATH=IndexKits:$PYTHONPATH

# Training configuration
task_flag="dit_g_2"
index_file=./dataset/AD_meta/jsons/AD_meta.json
results_dir="./log_EXP_dit_g_2_AD_meta"

batch_size=128
image_size=256
grad_accu_steps=1
lr=0.0001
ckpt_every=100
ckpt_latest_every=100
ckpt_every_n_epoch=1
epochs=100
warmup_num_steps=0

# CosineAnnealingWarmupRestarts parameters
max_steps=22100
num_cycles=4
warmup_ratio=0.05
max_lr=0.0001
min_lr=0.00001
t_mult=1.5
gamma=0.6

# Run the training script
sh $(dirname "$0")/run_g.sh \
    --task-flag ${task_flag} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0 \
    --uncond-p-t5 0 \
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --max-training-steps ${max_steps} \
    --num-cycles ${num_cycles} \
    --warmup-ratio ${warmup_ratio} \
    --max_lr ${max_lr} \
    --min_lr ${min_lr} \
    --t_mult ${t_mult} \
    --gamma ${gamma} \
    --use-fp16 \
    --extra-fp16 \
    --results-dir ${results_dir} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --ckpt-every-n-epoch ${ckpt_every_n_epoch} \
    --epochs ${epochs} \
    --log-every 10 \
    --deepspeed \
    --use-zero-stage 2 \
    --gradient-checkpointing \
    "$@"
