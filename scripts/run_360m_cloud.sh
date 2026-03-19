#!/bin/bash
# PILON-R 360M Cloud Training Script
#
# Run on A100 40GB (Vast.ai, Lambda, etc.)
# Trains Dense baseline + PILON C2 (gated recurrence), 2 seeds each.
#
# Setup:
#   git clone https://github.com/Babyhamsta/PILON.git && cd PILON
#   pip install -r requirements.txt
#   pip install flash-linear-attention
#   bash scripts/run_360m_cloud.sh
#
# Estimated time on A100 40GB: ~60-80 hours total (4 runs)
# Estimated cost at $0.472/hr: ~$30-40

set -e

TOTAL_TOKENS=1000000000  # 1B tokens
SEQ_LEN=2048
DATASET="HuggingFaceFW/fineweb-edu"

# Auto-detect batch size for this GPU
echo "=========================================="
echo "Auto-detecting optimal batch size..."
echo "=========================================="

# --- Step 1: Find batch size for Dense 360M ---
echo ""
echo "Probing Dense 360M..."
DENSE_BATCH=$(python scripts/auto_batch.py \
    --model-size 360m \
    --ffn-type standard \
    --attention-type standard_mha \
    --seq-len ${SEQ_LEN} \
    --no-ternary 2>&1 | grep "batch-size" | awk '{print $NF}')

if [ -z "$DENSE_BATCH" ]; then
    echo "Auto-detect failed for Dense, defaulting to batch=2"
    DENSE_BATCH=2
fi

# --- Step 2: Find batch size for PILON C2 ---
echo ""
echo "Probing PILON C2 (GLA + ternary)..."
PILON_BATCH=$(python scripts/auto_batch.py \
    --model-size 360m \
    --ffn-type compositional \
    --attention-type gated_recurrence \
    --seq-len ${SEQ_LEN} 2>&1 | grep "batch-size" | awk '{print $NF}')

if [ -z "$PILON_BATCH" ]; then
    echo "Auto-detect failed for PILON, defaulting to batch=2"
    PILON_BATCH=2
fi

# Compute grad_accum to target ~64k tokens per effective batch
DENSE_ACCUM=$((65536 / (DENSE_BATCH * SEQ_LEN)))
PILON_ACCUM=$((65536 / (PILON_BATCH * SEQ_LEN)))
[ "$DENSE_ACCUM" -lt 1 ] && DENSE_ACCUM=1
[ "$PILON_ACCUM" -lt 1 ] && PILON_ACCUM=1

echo ""
echo "=========================================="
echo "Configuration:"
echo "  Dense:  batch=${DENSE_BATCH}, grad_accum=${DENSE_ACCUM}, seq=${SEQ_LEN}"
echo "  PILON:  batch=${PILON_BATCH}, grad_accum=${PILON_ACCUM}, seq=${SEQ_LEN}"
echo "  Tokens: ${TOTAL_TOKENS}"
echo "  Dataset: ${DATASET}"
echo "=========================================="

# Shared flags
DATA_FLAGS="--num-workers 4 --prefetch-factor 4 --persistent-workers --tokenize-batch-size 128"
PILON_FLAGS="--freeze-primitives-phase2 --topk-cache-steps 10 --comp-lr-mult 2.0 --forward-fast-mode on --forward-fast-min-topk 1 --band-diversity-weight 0.01 --no-checkpoint-ffn --compile --recompile-phases --sparse-lr-max-mult 8.0 ${DATA_FLAGS}"

# ==========================================
# Run 1: Dense 360M, seed=42
# ==========================================
echo ""
echo "[1/4] Dense 360M, seed=42"
echo "=========================================="
if [ -f "outputs/360m_dense_seed42/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m \
        --ffn-type standard \
        --attention-type standard_mha \
        --compile \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${DENSE_BATCH} \
        --grad-accum ${DENSE_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir outputs/360m_dense_seed42 \
        --seed 42 \
        ${DATA_FLAGS}
fi

# ==========================================
# Run 2: Dense 360M, seed=123
# ==========================================
echo ""
echo "[2/4] Dense 360M, seed=123"
echo "=========================================="
if [ -f "outputs/360m_dense_seed123/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m \
        --ffn-type standard \
        --attention-type standard_mha \
        --compile \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${DENSE_BATCH} \
        --grad-accum ${DENSE_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir outputs/360m_dense_seed123 \
        --seed 123 \
        ${DATA_FLAGS}
fi

# ==========================================
# Run 3: PILON C2 (GLA + ternary) 360M, seed=42
# ==========================================
echo ""
echo "[3/4] PILON C2 360M (GLA + ternary), seed=42"
echo "=========================================="
if [ -f "outputs/360m_pilon_c2_seed42/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m \
        --ffn-type compositional \
        --ternary \
        --use-subln \
        --use-squared-relu \
        --attention-type gated_recurrence \
        --phase1-sparse \
        --phase1-top-k 8 \
        ${PILON_FLAGS} \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${PILON_BATCH} \
        --grad-accum ${PILON_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir outputs/360m_pilon_c2_seed42 \
        --seed 42 \
        --log-comp-stats
fi

# ==========================================
# Run 4: PILON C2 (GLA + ternary) 360M, seed=123
# ==========================================
echo ""
echo "[4/4] PILON C2 360M (GLA + ternary), seed=123"
echo "=========================================="
if [ -f "outputs/360m_pilon_c2_seed123/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m \
        --ffn-type compositional \
        --ternary \
        --use-subln \
        --use-squared-relu \
        --attention-type gated_recurrence \
        --phase1-sparse \
        --phase1-top-k 8 \
        ${PILON_FLAGS} \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${PILON_BATCH} \
        --grad-accum ${PILON_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir outputs/360m_pilon_c2_seed123 \
        --seed 123 \
        --log-comp-stats
fi

echo ""
echo "=========================================="
echo "All 360M runs complete!"
echo ""
echo "Results:"
for d in outputs/360m_*/; do
    if [ -f "${d}trainv2.log" ]; then
        last_val=$(grep "val_loss=" "${d}trainv2.log" | tail -1)
        echo "  ${d}: ${last_val}"
    fi
done
echo "=========================================="
