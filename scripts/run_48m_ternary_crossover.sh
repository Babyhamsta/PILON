#!/bin/bash
# PILON-R BitNet b1.58 Ternary Crossover Experiment (v2 — no MoE)
#
# Runs ternary configs only (dense/fp16 baselines already have logs).
# No MoE — matches the static-composition fp16 PILON baseline.
# Uses custom autograd STE functions for lower VRAM (no checkpoint_ffn needed).
#
# Configs:
# 1. PILON-48M-ternary (compositional + ternary + SubLN)
# 2. PILON-48M-ternary-sqrelu (compositional + ternary + SubLN + squared ReLU)
#
# Purpose: validate ternary quantization on PILON primitives.
# Success criteria: ternary PILON within ~1.5x loss of fp16 PILON.
#
# Hardware target: RTX 4070 (12GB VRAM)

set -e

TOTAL_TOKENS=500000000  # 500M tokens
OUTPUT_BASE="outputs/48m_ternary_crossover_v2"
BATCH_SIZE=8
GRAD_ACCUM=8
SEQ_LEN=512
DATASET="HuggingFaceFW/fineweb-edu"

# Throughput/data flags shared across configs
DATA_FLAGS="--num-workers 4 --prefetch-factor 4 --persistent-workers --tokenize-batch-size 128 --log-timing"

# PILON-specific training flags (NO MoE, no checkpoint_ffn)
PILON_FLAGS="--freeze-primitives-phase2 --topk-cache-steps 10 --comp-lr-mult 2.0 --forward-fast-mode on --forward-fast-min-topk 1 --band-diversity-weight 0.01 --no-checkpoint-ffn ${DATA_FLAGS}"

echo "=========================================="
echo "PILON-R 48M Ternary Crossover v2 (no MoE)"
echo "Total tokens: ${TOTAL_TOKENS}"
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} x ${SEQ_LEN} = $((BATCH_SIZE * GRAD_ACCUM * SEQ_LEN)) tokens/step"
echo "Expected steps: $((TOTAL_TOKENS / (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN)))"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_BASE}"
echo "=========================================="

# Config 1: PILON-48M-ternary (ternary + SubLN)
echo ""
echo "[1/2] PILON-48M-ternary (ternary + SubLN, no MoE)"
echo "=========================================="
if [ -f "${OUTPUT_BASE}/pilon_48m_ternary/final_model.pt" ]; then
    echo "SKIPPING: PILON-48M-ternary already complete (final_model.pt exists)"
else
    python -m pilon_r.train \
        --model-size 48m \
        --ffn-type compositional \
        --phase1-sparse \
        --phase1-top-k 8 \
        --ternary \
        --use-subln \
        ${PILON_FLAGS} \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir "${OUTPUT_BASE}/pilon_48m_ternary" \
        --log-comp-stats
fi

# Config 2: PILON-48M-ternary-sqrelu (ternary + SubLN + squared ReLU)
echo ""
echo "[2/2] PILON-48M-ternary-sqrelu (ternary + SubLN + squared ReLU, no MoE)"
echo "=========================================="
if [ -f "${OUTPUT_BASE}/pilon_48m_ternary_sqrelu/final_model.pt" ]; then
    echo "SKIPPING: PILON-48M-ternary-sqrelu already complete (final_model.pt exists)"
else
    python -m pilon_r.train \
        --model-size 48m \
        --ffn-type compositional \
        --phase1-sparse \
        --phase1-top-k 8 \
        --ternary \
        --use-subln \
        --use-squared-relu \
        ${PILON_FLAGS} \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir "${OUTPUT_BASE}/pilon_48m_ternary_sqrelu" \
        --log-comp-stats
fi

echo ""
echo "=========================================="
echo "48M Ternary Crossover v2 complete!"
echo ""
echo "Results in:"
echo "  ${OUTPUT_BASE}/pilon_48m_ternary/"
echo "  ${OUTPUT_BASE}/pilon_48m_ternary_sqrelu/"
echo ""
echo "Compare against existing fp16 baseline logs in outputs/48m_crossover/."
echo "Success: ternary PILON within ~1.5x loss of fp16 PILON."
echo "=========================================="
