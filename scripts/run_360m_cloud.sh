#!/bin/bash
# PILON-R 360M Cloud Training Script
#
# Run on A800 80GB / A100 (Vast.ai, Lambda, etc.)
# Trains Dense baseline + PILON C2 (gated recurrence), 2 seeds each.
#
# Setup:
#   git clone https://github.com/Babyhamsta/PILON.git && cd PILON
#   git checkout phase_c_experiments
#   pip install -r requirements.txt
#   pip install flash-linear-attention
#   bash scripts/run_360m_cloud.sh
#
# Override batch size: BATCH=16 bash scripts/run_360m_cloud.sh
#
# To find optimal batch for your GPU first:
#   PYTHONPATH=. python scripts/auto_batch.py --model-size 360m --ffn-type standard --no-ternary
#   PYTHONPATH=. python scripts/auto_batch.py --model-size 360m --ffn-type compositional

set -e

# Configurable via environment or defaults
BATCH=${BATCH:-12}
GRAD_ACCUM=${GRAD_ACCUM:-3}
SEQ_LEN=${SEQ_LEN:-2048}
TOTAL_TOKENS=${TOTAL_TOKENS:-1000000000}
DATASET=${DATASET:-"HuggingFaceFW/fineweb-edu"}

# Ensure PYTHONPATH includes repo root
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

echo "=========================================="
echo "PILON-R 360M Cloud Training"
echo "  Batch: ${BATCH}"
echo "  Grad accum: ${GRAD_ACCUM}"
echo "  Seq len: ${SEQ_LEN}"
echo "  Effective batch: $((BATCH * GRAD_ACCUM * SEQ_LEN)) tokens/step"
echo "  Total tokens: ${TOTAL_TOKENS}"
echo "  Dataset: ${DATASET}"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'unknown')"
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
        --batch-size ${BATCH} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir outputs/360m_dense_seed42 \
        --seed 42 --save-every 0 \
        ${DATA_FLAGS}
fi
# Clean intermediate checkpoints
rm -f outputs/360m_dense_seed42/checkpoint_step_*.pt 2>/dev/null

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
        --batch-size ${BATCH} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir outputs/360m_dense_seed123 \
        --seed 123 --save-every 0 \
        ${DATA_FLAGS}
fi
# Clean intermediate checkpoints
rm -f outputs/360m_dense_seed123/checkpoint_step_*.pt 2>/dev/null

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
        --batch-size ${BATCH} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir outputs/360m_pilon_c2_seed42 \
        --seed 42 --save-every 0 \
        --log-comp-stats
fi
# Clean intermediate checkpoints
rm -f outputs/360m_pilon_c2_seed42/checkpoint_step_*.pt 2>/dev/null

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
        --batch-size ${BATCH} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir outputs/360m_pilon_c2_seed123 \
        --seed 123 --save-every 0 \
        --log-comp-stats
fi
# Clean intermediate checkpoints
rm -f outputs/360m_pilon_c2_seed123/checkpoint_step_*.pt 2>/dev/null

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
