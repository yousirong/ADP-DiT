#!/bin/bash
#
# Test Evaluation Script for ADP-DiT
# ============================================================================
# Evaluates test.csv using 8 GPUs with the trained model checkpoint.
# Computes PSNR, MSE, SSIM metrics and saves results to CSV.
# ============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.:$PYTHONPATH

# Configuration - set these paths before running
TEST_CSV="./dataset/AD_meta/csvfile/test.csv"
CHECKPOINT="./checkpoints/final.pt/mp_rank_00_model_states.pt"

# Output configuration
OUTPUT_DIR="./test_outputs"
OUTPUT_CSV="${OUTPUT_DIR}/test_results.csv"

# Inference settings for Alzheimer's progression (subtle changes)
NUM_GPUS=8
BATCH_SIZE=2
INFER_STEPS=30        # High steps for precise medical image generation
STRENGTH=0.1          # Low strength: preserve input structure (subtle progression)
CFG_SCALE=6           # Lower CFG: reduce prompt influence, more natural anatomical changes
SAMPLER="dpmpp_2m_karras"  # DPM++ 2M with Karras sigmas - best quality/speed balance

# Note on strength for different progression types:
# - CN->CN, MCI->MCI, AD->AD: 0.3 (minimal change, same diagnosis)
# - CN->MCI, MCI->AD: 0.4-0.5 (moderate atrophy progression)
# - Using 0.4 as balanced default for mixed test set

echo "============================================================================"
echo "ADP-DiT Test Evaluation"
echo "============================================================================"
echo "Test CSV:        ${TEST_CSV}"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "  - Generated:   ${OUTPUT_DIR}/generated/"
echo "  - Comparisons: ${OUTPUT_DIR}/comparisons/"
echo "Num GPUs:        ${NUM_GPUS}"
echo "Inference Steps: ${INFER_STEPS}"
echo "Strength:        ${STRENGTH}"
echo "CFG Scale:       ${CFG_SCALE}"
echo "Sampler:         ${SAMPLER}"
echo "============================================================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

python -m adpdit.evaluate_test_csv \
    --test-csv "${TEST_CSV}" \
    --checkpoint "${CHECKPOINT}" \
    --output-csv "${OUTPUT_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-gpus ${NUM_GPUS} \
    --batch-size ${BATCH_SIZE} \
    --infer-steps ${INFER_STEPS} \
    --strength ${STRENGTH} \
    --cfg-scale ${CFG_SCALE} \
    --sampler "${SAMPLER}" \
    --save-images \
    --save-comparisons

echo ""
echo "============================================================================"
echo "Evaluation Complete!"
echo "Results saved to: ${OUTPUT_CSV}"
echo "Generated images saved to: ${OUTPUT_DIR}/generated/"
echo "Comparison images saved to: ${OUTPUT_DIR}/comparisons/"
echo "============================================================================"
