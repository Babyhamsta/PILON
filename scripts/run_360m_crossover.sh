#!/bin/bash
# PILON-R Phase B.5d: 360M Crossover Experiment
#
# Runs 4 configurations head-to-head on 1B tokens:
# 1. Dense-360M (standard FFN baseline)
# 2. PILON-360M-full (compositional, all primitives in VRAM)
# 3. PILON-360M-tiered (compositional with n_hot=16)
# 4. PILON-360M-exit (tiered model + post-hoc gate training)
#
# PILON runs are configured for token-routed sparse training (MoE) and
# throughput-first settings (no FFN checkpointing, forward_fast on).
#
# Hardware target: RTX 4070 (12GB VRAM)
# Estimated time: ~1.4 hours per config, ~6 hours total

set -e

TOTAL_TOKENS=1000000000  # 1B tokens
OUTPUT_BASE="outputs/360m_crossover"
BATCH_SIZE=4
GRAD_ACCUM=16
SEQ_LEN=2048
DATA_FLAGS="--num-workers 4 --prefetch-factor 4 --persistent-workers --tokenize-batch-size 128 --log-timing"
PILON_FLAGS="--freeze-primitives-phase2 --topk-cache-steps 10 --comp-lr-mult 2.0 --forward-fast-mode on --forward-fast-min-topk 1 --moe-experts 8 --moe-top-k 2 --moe-aux-loss-weight 0.01 --enable-early-exit --exit-threshold 0.5 --band-diversity-weight 0.01 --joint-exit-loss-weight 0.02 --no-checkpoint-ffn ${DATA_FLAGS}"

echo "=========================================="
echo "PILON-R 360M Crossover Experiment"
echo "Total tokens: ${TOTAL_TOKENS}"
echo "Output: ${OUTPUT_BASE}"
echo "=========================================="

# Config 1: Dense-360M
echo ""
echo "[1/4] Dense-360M (standard FFN baseline)"
echo "=========================================="
python -m pilon_r.train \
    --model-size 360m \
    --ffn-type standard \
    --total-tokens ${TOTAL_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --seq-len ${SEQ_LEN} \
    --output-dir "${OUTPUT_BASE}/dense_360m" \
    --no-checkpoint-ffn ${DATA_FLAGS}

# Config 2: PILON-360M-full
echo ""
echo "[2/4] PILON-360M-full (compositional, all primitives in VRAM)"
echo "=========================================="
python -m pilon_r.train \
    --model-size 360m \
    --ffn-type compositional \
    --phase1-sparse \
    --phase1-top-k 2 \
    ${PILON_FLAGS} \
    --total-tokens ${TOTAL_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --seq-len ${SEQ_LEN} \
    --output-dir "${OUTPUT_BASE}/pilon_360m_full"

# Config 3: PILON-360M-tiered (n_hot=16)
echo ""
echo "[3/4] PILON-360M-tiered (n_hot=16, swap_interval=100)"
echo "=========================================="
python -m pilon_r.train \
    --model-size 360m \
    --ffn-type compositional \
    --phase1-sparse \
    --phase1-top-k 2 \
    --n-hot 16 \
    --swap-interval 100 \
    --hot-tier-bias-weight 0.01 \
    ${PILON_FLAGS} \
    --total-tokens ${TOTAL_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --seq-len ${SEQ_LEN} \
    --output-dir "${OUTPUT_BASE}/pilon_360m_tiered"

# Config 4: PILON-360M-exit (tiered + post-hoc gate training)
echo ""
echo "[4/4] PILON-360M-exit (tiered + early exit gate training)"
echo "=========================================="
# First: copy tiered model checkpoint, then train gates on it
TIERED_CKPT="${OUTPUT_BASE}/pilon_360m_tiered/final_model.pt"
python -m pilon_r.train \
    --model-size 360m \
    --ffn-type compositional \
    --n-hot 16 \
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
    --output-dir "${OUTPUT_BASE}/pilon_360m_exit" \
    --seq-len ${SEQ_LEN}

echo ""
echo "=========================================="
echo "Crossover experiment complete!"
echo ""
echo "Results in:"
echo "  ${OUTPUT_BASE}/dense_360m/"
echo "  ${OUTPUT_BASE}/pilon_360m_full/"
echo "  ${OUTPUT_BASE}/pilon_360m_tiered/"
echo "  ${OUTPUT_BASE}/pilon_360m_exit/"
echo ""
echo "Run benchmarks:"
echo "  python -m pilon_r.benchmark ${OUTPUT_BASE}/dense_360m/final_model.pt --device cuda"
echo "  python -m pilon_r.benchmark ${OUTPUT_BASE}/pilon_360m_full/final_model.pt --device cuda"
echo "  python -m pilon_r.benchmark ${OUTPUT_BASE}/pilon_360m_tiered/final_model.pt --device cuda"
echo "  python -m pilon_r.benchmark ${OUTPUT_BASE}/pilon_360m_exit/model_with_gates.pt --device cuda"
echo "=========================================="
