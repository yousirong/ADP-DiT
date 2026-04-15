#!/bin/bash
#
# Selected Patient Evaluation Script for ADP-DiT
# ============================================================================
# Evaluates 5 transition types with representative patients:
# - CNtoCN, CNtoMCI, MCItoMCI, MCItoAD, ADtoAD
#
# Creates separate filtered CSVs for each transition type.
# Results are saved in separate directories:
#   ${OUTPUT_BASE}/cntocn/
#   ${OUTPUT_BASE}/cntomci/
#   ${OUTPUT_BASE}/mcitomci/
#   ${OUTPUT_BASE}/mcitoad/
#   ${OUTPUT_BASE}/adtoad/
# ============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.:$PYTHONPATH

# NUMA memory interleaving for better memory performance
numactl --interleave=all

# Configuration - set these paths before running
TEST_CSV="./dataset/AD_meta/csvfile/test_selected_all.csv"
CHECKPOINT="./checkpoints/final.pt/mp_rank_00_model_states.pt"

# Output configuration
OUTPUT_BASE="./test_outputs_by_transition"

# Inference settings
NUM_GPUS=8
BATCH_SIZE=2
INFER_STEPS=30
STRENGTH=0.1
CFG_SCALE=6
SAMPLER="dpmpp_2m_karras"


echo "============================================================================"
echo "ADP-DiT Selected Patient Evaluation (5 Transition Types)"
echo "============================================================================"
echo ""

echo "============================================================================"
echo "Configuration"
echo "============================================================================"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Output Base:     ${OUTPUT_BASE}"
echo "Num GPUs:        ${NUM_GPUS}"
echo "Inference Steps: ${INFER_STEPS}"
echo "Strength:        ${STRENGTH}"
echo "CFG Scale:       ${CFG_SCALE}"
echo "Sampler:         ${SAMPLER}"
echo "Error Threshold: ${ERROR_THRESHOLD}"
echo "Border Thickness: ${BORDER_THICKNESS}"
echo "Test CSV:        ${TEST_CSV}"
echo "============================================================================"

# Check if CSV exists and has data
if [ ! -f "${TEST_CSV}" ]; then
    echo "Error: CSV not found - ${TEST_CSV}"
    exit 1
fi

LINE_COUNT=$(wc -l < "${TEST_CSV}")
echo ""
echo "Total samples: $((LINE_COUNT - 1))"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Run evaluation
python -m adpdit.evaluate_test_csv_tb \
    --test-csv "${TEST_CSV}" \
    --checkpoint "${CHECKPOINT}" \
    --output-csv "${OUTPUT_BASE}/test_results.csv" \
    --output-dir "${OUTPUT_BASE}" \
    --num-gpus ${NUM_GPUS} \
    --batch-size ${BATCH_SIZE} \
    --infer-steps ${INFER_STEPS} \
    --strength ${STRENGTH} \
    --cfg-scale ${CFG_SCALE} \
    --sampler "${SAMPLER}" \
    --error-threshold ${ERROR_THRESHOLD} \
    --border-thickness ${BORDER_THICKNESS} \
    --save-images \
    --save-comparisons

echo ""
echo "============================================================================"
echo "Evaluation Complete!"
echo "============================================================================"
echo "Results saved to: ${OUTPUT_BASE}/"
echo "============================================================================"
