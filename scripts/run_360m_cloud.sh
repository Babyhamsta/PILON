#!/bin/bash
# PILON-R 360M Cloud Training Script
#
# Trains Dense + PILON C0 (standard MHA) + PILON C2 (gated recurrence), 2 seeds each.
# Designed for A800 80GB / A100 80GB.
#
# Setup:
#   git clone https://github.com/Babyhamsta/PILON.git && cd PILON
#   git checkout phase_c_experiments
#   pip install -r requirements.txt
#   pip install flash-linear-attention
#   bash scripts/run_360m_cloud.sh

set -e

# Prevent VRAM fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Batch configs: Dense fits batch=12, PILON needs batch=8 (more VRAM for primitive banks)
DENSE_BATCH=${DENSE_BATCH:-12}
DENSE_ACCUM=${DENSE_ACCUM:-3}
PILON_BATCH=${PILON_BATCH:-8}
PILON_ACCUM=${PILON_ACCUM:-4}
SEQ_LEN=${SEQ_LEN:-2048}
TOTAL_TOKENS=${TOTAL_TOKENS:-1000000000}
DATASET=${DATASET:-"HuggingFaceFW/fineweb-edu"}

echo "=========================================="
echo "PILON-R 360M Cloud Training (6 runs)"
echo "  Dense:  batch=${DENSE_BATCH} x accum=${DENSE_ACCUM} x seq=${SEQ_LEN} = $((DENSE_BATCH * DENSE_ACCUM * SEQ_LEN)) tok/step"
echo "  PILON:  batch=${PILON_BATCH} x accum=${PILON_ACCUM} x seq=${SEQ_LEN} = $((PILON_BATCH * PILON_ACCUM * SEQ_LEN)) tok/step"
echo "  Tokens: ${TOTAL_TOKENS}"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'unknown')"
echo "=========================================="

DATA_FLAGS="--num-workers 4 --prefetch-factor 4 --persistent-workers --tokenize-batch-size 128"
PILON_FLAGS="--freeze-primitives-phase2 --topk-cache-steps 10 --comp-lr-mult 2.0 --forward-fast-mode on --forward-fast-min-topk 1 --band-diversity-weight 0.01 --no-checkpoint-ffn --compile --recompile-phases --sparse-lr-max-mult 8.0 ${DATA_FLAGS}"

# ==========================================
# Run 1: Dense 360M, seed=42
# ==========================================
echo ""
echo "[1/6] Dense 360M, seed=42"
echo "=========================================="
if [ -f "outputs/360m_dense_seed42/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m --ffn-type standard --attention-type standard_mha \
        --compile --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${DENSE_BATCH} --grad-accum ${DENSE_ACCUM} --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} --output-dir outputs/360m_dense_seed42 \
        --seed 42 --save-every 0 ${DATA_FLAGS}
fi
rm -f outputs/360m_dense_seed42/checkpoint_step_*.pt 2>/dev/null

# ==========================================
# Run 2: Dense 360M, seed=123
# ==========================================
echo ""
echo "[2/6] Dense 360M, seed=123"
echo "=========================================="
if [ -f "outputs/360m_dense_seed123/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m --ffn-type standard --attention-type standard_mha \
        --compile --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${DENSE_BATCH} --grad-accum ${DENSE_ACCUM} --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} --output-dir outputs/360m_dense_seed123 \
        --seed 123 --save-every 0 ${DATA_FLAGS}
fi
rm -f outputs/360m_dense_seed123/checkpoint_step_*.pt 2>/dev/null

# ==========================================
# Run 3: PILON C0 (standard MHA + ternary) 360M, seed=42
# ==========================================
echo ""
echo "[3/6] PILON C0 360M (standard MHA + ternary), seed=42"
echo "=========================================="
if [ -f "outputs/360m_pilon_c0_seed42/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m --ffn-type compositional \
        --ternary --use-subln --use-squared-relu \
        --attention-type standard_mha \
        --phase1-sparse --phase1-top-k 8 \
        ${PILON_FLAGS} --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${PILON_BATCH} --grad-accum ${PILON_ACCUM} --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} --output-dir outputs/360m_pilon_c0_seed42 \
        --seed 42 --save-every 0 --log-comp-stats
fi
rm -f outputs/360m_pilon_c0_seed42/checkpoint_step_*.pt 2>/dev/null

# ==========================================
# Run 4: PILON C0 (standard MHA + ternary) 360M, seed=123
# ==========================================
echo ""
echo "[4/6] PILON C0 360M (standard MHA + ternary), seed=123"
echo "=========================================="
if [ -f "outputs/360m_pilon_c0_seed123/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m --ffn-type compositional \
        --ternary --use-subln --use-squared-relu \
        --attention-type standard_mha \
        --phase1-sparse --phase1-top-k 8 \
        ${PILON_FLAGS} --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${PILON_BATCH} --grad-accum ${PILON_ACCUM} --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} --output-dir outputs/360m_pilon_c0_seed123 \
        --seed 123 --save-every 0 --log-comp-stats
fi
rm -f outputs/360m_pilon_c0_seed123/checkpoint_step_*.pt 2>/dev/null

# ==========================================
# Run 5: PILON C2 (GLA + ternary) 360M, seed=42
# ==========================================
echo ""
echo "[5/6] PILON C2 360M (GLA + ternary), seed=42"
echo "=========================================="
if [ -f "outputs/360m_pilon_c2_seed42/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m --ffn-type compositional \
        --ternary --use-subln --use-squared-relu \
        --attention-type gated_recurrence \
        --phase1-sparse --phase1-top-k 8 \
        ${PILON_FLAGS} --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${PILON_BATCH} --grad-accum ${PILON_ACCUM} --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} --output-dir outputs/360m_pilon_c2_seed42 \
        --seed 42 --save-every 0 --log-comp-stats
fi
rm -f outputs/360m_pilon_c2_seed42/checkpoint_step_*.pt 2>/dev/null

# ==========================================
# Run 6: PILON C2 (GLA + ternary) 360M, seed=123
# ==========================================
echo ""
echo "[6/6] PILON C2 360M (GLA + ternary), seed=123"
echo "=========================================="
if [ -f "outputs/360m_pilon_c2_seed123/final_model.pt" ]; then
    echo "SKIPPING: already complete"
else
    python -m pilon_r.train \
        --model-size 360m --ffn-type compositional \
        --ternary --use-subln --use-squared-relu \
        --attention-type gated_recurrence \
        --phase1-sparse --phase1-top-k 8 \
        ${PILON_FLAGS} --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${PILON_BATCH} --grad-accum ${PILON_ACCUM} --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} --output-dir outputs/360m_pilon_c2_seed123 \
        --seed 123 --save-every 0 --log-comp-stats
fi
rm -f outputs/360m_pilon_c2_seed123/checkpoint_step_*.pt 2>/dev/null

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "All 360M runs complete!"
echo ""
echo "Results:"
for d in outputs/360m_*/; do
    if [ -f "${d}trainv2.log" ]; then
        name=$(basename "$d")
        last_val=$(grep "val_loss=" "${d}trainv2.log" | tail -1 | grep -o "val_loss=[0-9.]*" || echo "no val")
        last_ppl=$(grep "val_ppl=" "${d}trainv2.log" | tail -1 | grep -o "val_ppl=[0-9.]*" || echo "no ppl")
        echo "  ${name}: ${last_val} ${last_ppl}"
    fi
done
echo "=========================================="
