#!/bin/bash
# PILON-R BitNet b1.58 Ternary Crossover Experiment — 360M
#
# Scaled-up version of run_48m_ternary_crossover.sh.
# Uses 360M config: d_model=1024, n_layers=24, d_ff=4096, n_primitives=80, rank=80, top_k=8
#
# Configs:
# 1. PILON-360M-ternary (compositional + ternary + SubLN)
# 2. PILON-360M-ternary-sqrelu (compositional + ternary + SubLN + squared ReLU)
#
# Purpose: validate ternary PILON at scale.
# Success criteria: ternary PILON within ~1.5x loss of fp16 PILON at 360M.
#
# Hardware target: RTX 4070 (12GB VRAM) — may need --checkpoint-ffn if OOM.
# torch.compile enabled for throughput (~1.9x speedup).
#
# If you hit OOM, try:
#   1. Reduce BATCH_SIZE to 1
#   2. Add --checkpoint-ffn (removes --no-checkpoint-ffn from PILON_FLAGS)
#   3. Reduce SEQ_LEN to 1024

set -e

TOTAL_TOKENS=1000000000  # 1B tokens
OUTPUT_BASE="outputs/360m_ternary_crossover"
BATCH_SIZE=2
GRAD_ACCUM=32
SEQ_LEN=2048
DATASET="HuggingFaceFW/fineweb-edu"

# Throughput/data flags shared across configs
DATA_FLAGS="--num-workers 4 --prefetch-factor 4 --persistent-workers --tokenize-batch-size 128 --log-timing"

# PILON-specific training flags
PILON_FLAGS="--freeze-primitives-phase2 --topk-cache-steps 10 --comp-lr-mult 2.0 --forward-fast-mode on --forward-fast-min-topk 1 --band-diversity-weight 0.01 --no-checkpoint-ffn --compile ${DATA_FLAGS}"

echo "=========================================="
echo "PILON-R 360M Ternary Crossover"
echo "Total tokens: ${TOTAL_TOKENS}"
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} x ${SEQ_LEN} = $((BATCH_SIZE * GRAD_ACCUM * SEQ_LEN)) tokens/step"
echo "Expected steps: $((TOTAL_TOKENS / (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN)))"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_BASE}"
echo "=========================================="

# Config 1: PILON-360M-ternary (ternary + SubLN)
echo ""
echo "[1/2] PILON-360M-ternary (ternary + SubLN)"
echo "=========================================="
if [ -f "${OUTPUT_BASE}/pilon_360m_ternary/final_model.pt" ]; then
    echo "SKIPPING: PILON-360M-ternary already complete (final_model.pt exists)"
else
    python -m pilon_r.train \
        --model-size 360m \
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
        --output-dir "${OUTPUT_BASE}/pilon_360m_ternary" \
        --log-comp-stats
fi

# Config 2: PILON-360M-ternary-sqrelu (ternary + SubLN + squared ReLU)
echo ""
echo "[2/2] PILON-360M-ternary-sqrelu (ternary + SubLN + squared ReLU)"
echo "=========================================="
if [ -f "${OUTPUT_BASE}/pilon_360m_ternary_sqrelu/final_model.pt" ]; then
    echo "SKIPPING: PILON-360M-ternary-sqrelu already complete (final_model.pt exists)"
else
    python -m pilon_r.train \
        --model-size 360m \
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
        --output-dir "${OUTPUT_BASE}/pilon_360m_ternary_sqrelu" \
        --log-comp-stats
fi

echo ""
echo "=========================================="
echo "360M Ternary Crossover complete!"
echo ""
echo "Results in:"
echo "  ${OUTPUT_BASE}/pilon_360m_ternary/"
echo "  ${OUTPUT_BASE}/pilon_360m_ternary_sqrelu/"
echo ""
echo "Compare against fp16 baselines once available."
echo "Success: ternary PILON within ~1.5x loss of fp16 PILON."
echo "=========================================="
