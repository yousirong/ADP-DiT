#!/bin/bash
#
# Cosine Annealing with Warm Restarts - GENERATION ONLY MODE (Memory Efficient)
# ============================================================================
# - Generation Only: MSE loss only (no classification)
# - Memory Optimized: Uses less VRAM than classification version
# ============================================================================
# Features:
#   - Classification: DISABLED
#   - Validation: CONFIGURED (requires --use-classification to run)
#   - VRAM Usage: ~38-40 GiB (more stable, less peak memory)
#   - Loss: MSE only (pure generation)
#
# Usage:
#   ./train_cosine_restarts.sh                    # Start new training
#   ./train_cosine_restarts.sh --resume \         # Resume training
#     --resume-module-root checkpoints/latest.pt
#

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29500
export PYTHONPATH=IndexKits:$PYTHONPATH

# Training configuration
task_flag="dit_g_2_cosine_restarts"
index_file=./dataset/AD_meta/jsons/AD_meta.json
train_csv_path=./dataset/AD_meta/csvfile/image_text.csv
results_dir="./log_EXP_dit_g_2_AD_meta_cosine_restarts"

batch_size=128              # Training batch size per GPU
image_size=256
grad_accu_steps=1
lr=0.00005
ckpt_every=1000
ckpt_latest_every=1000
ckpt_every_n_epoch=1

# Sample generation during training
sample_every=100
num_samples=4
sample_infer_steps=20
sample_cfg_scale=6.0
sample_strength=0.1
epochs=100
warmup_num_steps=0

# Validation settings (will be used only if --use-classification is enabled)
val_every=100
val_patience=3
val_start_step=100
val_batch_size=64
val_grad_accu_steps=1

# ============================================================================
# Cosine Annealing with Warm Restarts Parameters
# ============================================================================
# Iters per epoch: 204,360 / 1,024 ~ 199
# Total steps: 199 iters/epoch x 100 epochs = 19,900 steps
# Schedule (4 cycles with t_mult=1.2, gamma=0.75):
#   first_cycle ~ 3,707 steps
#   Cycle 0: 3,707 steps (~19 epochs)
#   Cycle 1: 4,448 steps (~22 epochs)
#   Cycle 2: 5,338 steps (~27 epochs)
#   Cycle 3: 6,406 steps (~32 epochs)
#   Total: ~19,900 steps
# ============================================================================
max_steps=19900
num_cycles=4
warmup_ratio=0.05
max_lr=0.00005
min_lr=0.000005
t_mult=1.2
gamma=0.75

echo "=========================================="
echo "  Generation Only Mode (No Classification)"
echo "=========================================="
echo "Task: ${task_flag}"
echo "Results: ${results_dir}"
echo ""
echo "Training Mode:"
echo "  - Classification: DISABLED"
echo "  - Validation: CONFIGURED (runs only with --use-classification)"
echo "  - Loss Type: MSE (generation only)"
echo "  - VRAM Usage: Stable 38-40 GiB (memory efficient)"
echo ""

echo "Scheduler Settings:"
echo "  - Strategy: Multi-Cycle with Restarts"
echo "  - Total Steps: ${max_steps}"
echo "  - Epochs: ${epochs}"
echo "  - Number of Cycles: ${num_cycles}"
echo "  - Cycle Multiplier (t_mult): ${t_mult}"
echo "  - LR Decay (gamma): ${gamma}"
echo "  - Max LR: ${max_lr}"
echo "  - Min LR: ${min_lr}"
echo "  - Warmup: ${warmup_ratio} (5% of each cycle)"
echo "=========================================="
echo ""

echo "Loss Strategy:"
echo "  - Generation Loss:     MSE (L2 pixel distance)"
echo "  - Total Loss:          MSE only"
echo "  - Optimizer:           AdamW"
echo "  - Focus:               Pure image generation quality"
echo ""

echo "Sample Generation:"
echo "  - Sample Every: ${sample_every} steps"
echo "  - Num Samples: ${num_samples}"
echo "  - Infer Steps: ${sample_infer_steps}"
echo "  - CFG Scale: ${sample_cfg_scale}"
echo "  - Strength: ${sample_strength}"
echo ""
echo "Validation Settings (no-op without --use-classification):"
echo "  - Train CSV: ${train_csv_path}"
echo "  - Validation from train: ENABLED"
echo "  - Validation every: ${val_every} steps"
echo "  - Validation start step: ${val_start_step}"
echo "  - Validation batch: ${val_batch_size} per GPU"
echo "  - Validation grad accu: ${val_grad_accu_steps}"
echo "  - Early stopping patience: ${val_patience} validations"
echo "=========================================="
echo ""

# Run training with CosineAnnealingWarmupRestarts
sh $(dirname "$0")/run_g.sh \
    --task-flag ${task_flag} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0.1 \
    --uncond-p-t5 0.1 \
    --index-file ${index_file} \
    --valid-from-train \
    --train-csv-path ${train_csv_path} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --val-batch-size ${val_batch_size} \
    --val-grad-accu-steps ${val_grad_accu_steps} \
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
    --val-every ${val_every} \
    --val-patience ${val_patience} \
    --val-start-step ${val_start_step} \
    --sample-every ${sample_every} \
    --num-samples ${num_samples} \
    --sample-infer-steps ${sample_infer_steps} \
    --sample-cfg-scale ${sample_cfg_scale} \
    --sample-strength ${sample_strength} \
    --deepspeed \
    --use-zero-stage 2 \
    --gradient-checkpointing \
    "$@"
