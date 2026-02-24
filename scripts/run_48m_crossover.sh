#!/bin/bash
# PILON-R Phase B.5: 48M Crossover Experiment (Validation Run)
#
# Runs 4 configurations head-to-head on 500M tokens:
# 1. Dense-48M (standard FFN baseline)
# 2. PILON-48M-full (compositional, all primitives in VRAM)
# 3. PILON-48M-tiered (compositional with n_hot=12)
# 4. PILON-48M-exit (tiered model + post-hoc gate training)
#
# PILON configs use throughput-oriented flags:
#   --moe-experts/--moe-top-k: token-level routing instead of static per-layer mixing
#   --forward-fast-mode on: force fused fast-path for compositional FFN
#   --no-checkpoint-ffn: avoid recompute-heavy backward on compositional FFN
#   --band-diversity-weight: exploit shared banks by reducing in-band redundancy
#
# Hardware target: RTX 4070 (12GB VRAM)
# Purpose: validate full 4-config pipeline before the longer 360M run.

set -e

TOTAL_TOKENS=500000000  # 500M tokens
OUTPUT_BASE="outputs/48m_crossover"
BATCH_SIZE=8
GRAD_ACCUM=8
SEQ_LEN=512
DATASET="HuggingFaceFW/fineweb-edu"

# Throughput/data flags shared across configs
DATA_FLAGS="--num-workers 4 --prefetch-factor 4 --persistent-workers --tokenize-batch-size 128 --log-timing"

# PILON-specific training flags (sparse/token-routed, throughput-biased)
PILON_FLAGS="--freeze-primitives-phase2 --topk-cache-steps 10 --comp-lr-mult 2.0 --forward-fast-mode on --forward-fast-min-topk 1 --moe-experts 8 --moe-top-k 2 --moe-aux-loss-weight 0.01 --enable-early-exit --exit-threshold 0.5 --band-diversity-weight 0.01 --joint-exit-loss-weight 0.02 --no-checkpoint-ffn ${DATA_FLAGS}"

echo "=========================================="
echo "PILON-R 48M Crossover Experiment"
echo "Total tokens: ${TOTAL_TOKENS}"
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} x ${SEQ_LEN} = $((BATCH_SIZE * GRAD_ACCUM * SEQ_LEN)) tokens/step"
echo "Expected steps: $((TOTAL_TOKENS / (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN)))"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_BASE}"
echo "=========================================="

# Config 1: Dense-48M
echo ""
echo "[1/4] Dense-48M (standard FFN baseline)"
echo "=========================================="
if [ -f "${OUTPUT_BASE}/dense_48m/final_model.pt" ]; then
    echo "SKIPPING: Dense-48M already complete (final_model.pt exists)"
else
    python -m pilon_r.train \
        --model-size 48m \
        --ffn-type standard \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir "${OUTPUT_BASE}/dense_48m" \
        --no-checkpoint-ffn ${DATA_FLAGS}
fi

# Config 2: PILON-48M-full
echo ""
echo "[2/4] PILON-48M-full (compositional, all primitives in VRAM)"
echo "=========================================="
python -m pilon_r.train \
    --model-size 48m \
    --ffn-type compositional \
    --phase1-sparse \
    --phase1-top-k 2 \
    ${PILON_FLAGS} \
    --total-tokens ${TOTAL_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --seq-len ${SEQ_LEN} \
    --dataset ${DATASET} \
    --output-dir "${OUTPUT_BASE}/pilon_48m_full" \
    --log-comp-stats

# Config 3: PILON-48M-tiered (n_hot=12)
echo ""
echo "[3/4] PILON-48M-tiered (n_hot=12, swap_interval=100)"
echo "=========================================="
python -m pilon_r.train \
    --model-size 48m \
    --ffn-type compositional \
    --phase1-sparse \
    --phase1-top-k 2 \
    --n-hot 12 \
    --swap-interval 100 \
    --hot-tier-bias-weight 0.01 \
    ${PILON_FLAGS} \
    --total-tokens ${TOTAL_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --seq-len ${SEQ_LEN} \
    --dataset ${DATASET} \
    --output-dir "${OUTPUT_BASE}/pilon_48m_tiered" \
    --log-comp-stats

# Config 4: PILON-48M-exit (tiered + post-hoc gate training)
echo ""
echo "[4/4] PILON-48M-exit (tiered + early exit gate training)"
echo "=========================================="
# Train gates on the tiered model checkpoint
TIERED_CKPT="${OUTPUT_BASE}/pilon_48m_tiered/final_model.pt"
python -m pilon_r.train \
    --model-size 48m \
    --ffn-type compositional \
    --n-hot 12 \
    --swap-interval 100 \
    --enable-early-exit \
    --exit-threshold 0.5 \
    --moe-experts 8 \
    --moe-top-k 2 \
    --moe-aux-loss-weight 0.01 \
    --forward-fast-mode on \
    --forward-fast-min-topk 1 \
    --no-checkpoint-ffn \
    ${DATA_FLAGS} \
    --train-exit-gates \
    --resume "${TIERED_CKPT}" \
    --batch-size ${BATCH_SIZE} \
    --dataset ${DATASET} \
    --output-dir "${OUTPUT_BASE}/pilon_48m_exit" \
    --seq-len ${SEQ_LEN}

echo ""
echo "=========================================="
echo "48M Crossover experiment complete!"
echo ""
echo "Results in:"
echo "  ${OUTPUT_BASE}/dense_48m/"
echo "  ${OUTPUT_BASE}/pilon_48m_full/"
echo "  ${OUTPUT_BASE}/pilon_48m_tiered/"
echo "  ${OUTPUT_BASE}/pilon_48m_exit/"
echo ""
echo "Run benchmarks:"
echo "  python -m pilon_r.benchmark ${OUTPUT_BASE}/dense_48m/final_model.pt --device cuda"
echo "  python -m pilon_r.benchmark ${OUTPUT_BASE}/pilon_48m_full/final_model.pt --device cuda"
echo "  python -m pilon_r.benchmark ${OUTPUT_BASE}/pilon_48m_tiered/final_model.pt --device cuda"
echo "  python -m pilon_r.benchmark ${OUTPUT_BASE}/pilon_48m_exit/model_with_gates.pt --device cuda"
echo "=========================================="
