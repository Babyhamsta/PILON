#!/bin/bash
# PILON-R 360M Crossover — RunPod A40 Setup + Training Script
#
# Self-contained script for RunPod A40 (48GB VRAM). Handles:
#   1. Environment setup (pip, tokenizer, GPU check)
#   2. Smoke test (10 steps → throughput + ETA)
#   3. Four training configs back-to-back
#   4. Post-training benchmarks + results archive
#
# Usage:
#   Upload PILON-R.zip to RunPod, unzip, cd PILON-R
#   bash scripts/runpod_360m.sh
#
# Resume after crash:
#   Just re-run: bash scripts/runpod_360m.sh
#   It auto-detects checkpoints and resumes from the latest one.
#
# Key differences from run_360m_crossover.sh:
#   - No MoE flags (training path broken, 3.3x slower)
#   - phase1-top-k=8 (best loss quality per 48M tests)
#   - python -u (unbuffered output for monitoring)
#   - Skip-if-done on all 4 configs (safe restart)
#   - Auto-resume from latest checkpoint on crash
#   - Checkpoints every 250 steps (~1h) to /workspace
#   - Smoke test + ETA tracking for cost visibility
#   - Auto batch-size based on VRAM
#   - Results archive (tar.gz) at end

set -euo pipefail

SCRIPT_START=$(date +%s)

# ============================================================
# Helper Functions
# ============================================================

log() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

elapsed_since() {
    local start=$1
    local now=$(date +%s)
    local secs=$((now - start))
    printf "%dh %dm %ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

print_eta() {
    local configs_done=$1
    local configs_total=$2
    local now=$(date +%s)
    local elapsed=$((now - SCRIPT_START))

    if [ "$configs_done" -gt 0 ]; then
        local avg=$((elapsed / configs_done))
        local remaining=$(( avg * (configs_total - configs_done) ))
        echo ""
        echo "--- Progress: ${configs_done}/${configs_total} configs done ---"
        echo "    Elapsed:   $(elapsed_since $SCRIPT_START)"
        echo "    Avg/config: $(printf '%dh %dm %ds' $((avg/3600)) $(((avg%3600)/60)) $((avg%60)))"
        echo "    ETA:        $(printf '%dh %dm %ds' $((remaining/3600)) $(((remaining%3600)/60)) $((remaining%60)))"
        echo ""
    fi
}

check_done() {
    local checkpoint_path=$1
    local config_name=$2
    if [ -f "$checkpoint_path" ]; then
        echo "SKIPPING: ${config_name} already complete ($(basename "$checkpoint_path") exists)"
        return 0
    fi
    return 1
}

# Find the latest checkpoint_step_*.pt in a directory, return path or empty string
find_latest_checkpoint() {
    local dir=$1
    local latest=""
    local max_step=-1
    if [ -d "$dir" ]; then
        for ckpt in "$dir"/checkpoint_step_*.pt; do
            [ -f "$ckpt" ] || continue
            # Extract step number from filename
            step=$(echo "$ckpt" | grep -oP 'checkpoint_step_\K[0-9]+')
            if [ "$step" -gt "$max_step" ]; then
                max_step=$step
                latest=$ckpt
            fi
        done
    fi
    echo "$latest"
}

# ============================================================
# Section 1: Environment Setup
# ============================================================

log "Section 1: Environment Setup"

echo "[setup] Installing Python dependencies..."
pip install --no-cache-dir \
    "transformers>=4.30.0" \
    "datasets>=2.14.0" \
    numpy \
    tqdm \
    matplotlib 2>&1 | tail -5
echo "[setup] pip install done."

echo "[setup] Pre-downloading GPT-2 tokenizer..."
python -u -c "from transformers import GPT2TokenizerFast; GPT2TokenizerFast.from_pretrained('gpt2')"
echo "[setup] Tokenizer cached."

echo "[setup] GPU verification..."
GPU_INFO=$(python -u -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: No CUDA GPU found')
    exit(1)
name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
bf16 = torch.cuda.is_bf16_supported()
print(f'GPU: {name}')
print(f'VRAM: {vram_gb:.1f} GB')
print(f'BF16: {bf16}')
print(f'VRAM_GB:{vram_gb:.0f}')
")
echo "$GPU_INFO"

# Auto-select batch size based on VRAM
VRAM_GB=$(echo "$GPU_INFO" | grep "^VRAM_GB:" | cut -d: -f2)
if [ "${VRAM_GB:-0}" -ge 40 ]; then
    BATCH_SIZE=8
elif [ "${VRAM_GB:-0}" -ge 20 ]; then
    BATCH_SIZE=4
else
    BATCH_SIZE=2
fi
echo "[setup] Auto-selected BATCH_SIZE=${BATCH_SIZE} for ${VRAM_GB}GB VRAM"

echo "[setup] Dataset streaming warm-up..."
python -u -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)
for i, sample in enumerate(ds):
    if i >= 10:
        break
del ds
print('Dataset streaming OK')
" 2>/dev/null || echo "[setup] warm-up OK (ignoring cleanup crash)"
echo "[setup] Environment ready."

# ============================================================
# Section 2: Shared Config Variables
# ============================================================

TOTAL_TOKENS=500000000   # 500M tokens
GRAD_ACCUM=16
SEQ_LEN=2048
DATASET="HuggingFaceFW/fineweb-edu"
SAVE_EVERY=250           # checkpoint every 250 steps (~1h on A40)

# Use /workspace for output so checkpoints survive pod crashes
# /workspace is the persistent volume on RunPod
if [ -d "/workspace" ]; then
    OUTPUT_BASE="/workspace/360m_crossover"
    echo "[setup] Using persistent volume: ${OUTPUT_BASE}"
else
    OUTPUT_BASE="outputs/360m_crossover"
    echo "[setup] No /workspace found, using local: ${OUTPUT_BASE}"
fi

TOKENS_PER_STEP=$((BATCH_SIZE * GRAD_ACCUM * SEQ_LEN))
TOTAL_STEPS=$((TOTAL_TOKENS / TOKENS_PER_STEP))

DATA_FLAGS="--num-workers 4 --prefetch-factor 4 --persistent-workers --tokenize-batch-size 128 --log-timing --log-comp-stats"

# No MoE flags. phase1-top-k=8 set per-config.
PILON_FLAGS="--freeze-primitives-phase2 --topk-cache-steps 10 --comp-lr-mult 2.0 --forward-fast-mode on --forward-fast-min-topk 1 --band-diversity-weight 0.01 --no-checkpoint-ffn"

log "PILON-R 360M Crossover — RunPod A40"
echo "Total tokens:   ${TOTAL_TOKENS} (500M)"
echo "Batch size:     ${BATCH_SIZE}"
echo "Grad accum:     ${GRAD_ACCUM}"
echo "Seq len:        ${SEQ_LEN}"
echo "Tokens/step:    ${TOKENS_PER_STEP}"
echo "Total steps:    ${TOTAL_STEPS}"
echo "Save every:     ${SAVE_EVERY} steps"
echo "Dataset:        ${DATASET}"
echo "Output:         ${OUTPUT_BASE}"

# ============================================================
# Section 3: Smoke Test (10 steps)
# ============================================================

# Skip smoke test if any config already has progress (i.e. this is a resume)
HAS_PROGRESS=false
for d in dense_360m pilon_360m_full pilon_360m_tiered pilon_360m_exit; do
    if ls "${OUTPUT_BASE}/${d}"/checkpoint_step_*.pt 2>/dev/null | head -1 | grep -q . || \
       [ -f "${OUTPUT_BASE}/${d}/final_model.pt" ] || \
       [ -f "${OUTPUT_BASE}/${d}/model_with_gates.pt" ]; then
        HAS_PROGRESS=true
        break
    fi
done

if [ "$HAS_PROGRESS" = true ]; then
    echo "Resuming previous run — skipping smoke test."
else
    log "Section 3: Smoke Test (10 steps of Dense-360M)"

    SMOKE_DIR="${OUTPUT_BASE}/_smoke_test"
    mkdir -p "${SMOKE_DIR}"

    SMOKE_START=$(date +%s)
    python -u -m pilon_r.train \
    --model-size 360m \
    --ffn-type standard \
    --steps 10 \
    --total-tokens $((10 * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN)) \
    --save-every 0 \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --seq-len ${SEQ_LEN} \
    --dataset ${DATASET} \
    --no-checkpoint-ffn --log-timing \
    --output-dir "${SMOKE_DIR}" \
    2>&1 | tee "${SMOKE_DIR}/smoke_stdout.log"
SMOKE_END=$(date +%s)
SMOKE_SECS=$((SMOKE_END - SMOKE_START))

echo ""
echo "Smoke test: 10 steps in ${SMOKE_SECS}s"

# Estimate total time
SECS_PER_STEP=$(python -u -c "print(f'{${SMOKE_SECS}/10:.1f}')")
EST_PER_CONFIG=$(python -u -c "print(f'{${SMOKE_SECS}/10 * ${TOTAL_STEPS}:.0f}')")
EST_TOTAL=$(python -u -c "print(f'{${SMOKE_SECS}/10 * ${TOTAL_STEPS} * 3.5:.0f}')")  # ~3.5 configs worth (config4 is short)
EST_HOURS=$(python -u -c "print(f'{${SMOKE_SECS}/10 * ${TOTAL_STEPS} * 3.5 / 3600:.1f}')")

echo "Throughput:     ~${SECS_PER_STEP}s/step"
echo "Est. per config: ~$((EST_PER_CONFIG / 3600))h $((EST_PER_CONFIG % 3600 / 60))m"
echo "Est. total:      ~${EST_HOURS}h (3 full configs + 1 short gate training)"

TOTAL_HOURS_INT=$(python -u -c "print(int(${SMOKE_SECS}/10 * ${TOTAL_STEPS} * 3.5 / 3600))")
if [ "$TOTAL_HOURS_INT" -gt 12 ]; then
    echo ""
    echo "NOTE: Estimated total > 12h. Checkpoints saved every ${SAVE_EVERY} steps."
    echo "      Safe to restart: bash scripts/runpod_360m.sh"
fi

    # Clean up smoke test
    rm -rf "${SMOKE_DIR}"
    echo "Smoke test output cleaned up."
fi

# ============================================================
# Section 4: Training Configs
# ============================================================

CONFIGS_DONE=0
CONFIGS_TOTAL=4

# --- Config 1: Dense-360M ---
log "[1/4] Dense-360M (standard FFN baseline)"
CONFIG1_DIR="${OUTPUT_BASE}/dense_360m"
mkdir -p "${CONFIG1_DIR}"

if check_done "${CONFIG1_DIR}/final_model.pt" "Dense-360M"; then
    CONFIGS_DONE=$((CONFIGS_DONE + 1))
else
    RESUME_FLAGS=""
    LATEST=$(find_latest_checkpoint "${CONFIG1_DIR}")
    if [ -n "$LATEST" ]; then
        echo "RESUMING from ${LATEST}"
        RESUME_FLAGS="--resume ${LATEST}"
    fi
    CONFIG1_START=$(date +%s)
    python -u -m pilon_r.train \
        --model-size 360m \
        --ffn-type standard \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --save-every ${SAVE_EVERY} \
        --no-checkpoint-ffn --log-timing \
        --output-dir "${CONFIG1_DIR}" \
        ${RESUME_FLAGS} \
        2>&1 | tee -a "${CONFIG1_DIR}/train_stdout.log"
    CONFIG1_END=$(date +%s)
    echo "Dense-360M completed in $(elapsed_since $CONFIG1_START)"
    CONFIGS_DONE=$((CONFIGS_DONE + 1))
fi
print_eta $CONFIGS_DONE $CONFIGS_TOTAL

# --- Config 2: PILON-360M-full ---
log "[2/4] PILON-360M-full (compositional, all primitives in VRAM)"
CONFIG2_DIR="${OUTPUT_BASE}/pilon_360m_full"
mkdir -p "${CONFIG2_DIR}"

if check_done "${CONFIG2_DIR}/final_model.pt" "PILON-360M-full"; then
    CONFIGS_DONE=$((CONFIGS_DONE + 1))
else
    RESUME_FLAGS=""
    LATEST=$(find_latest_checkpoint "${CONFIG2_DIR}")
    if [ -n "$LATEST" ]; then
        echo "RESUMING from ${LATEST}"
        RESUME_FLAGS="--resume ${LATEST}"
    fi
    CONFIG2_START=$(date +%s)
    python -u -m pilon_r.train \
        --model-size 360m \
        --ffn-type compositional \
        --phase1-sparse \
        --phase1-top-k 8 \
        ${PILON_FLAGS} ${DATA_FLAGS} \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --save-every ${SAVE_EVERY} \
        --output-dir "${CONFIG2_DIR}" \
        ${RESUME_FLAGS} \
        2>&1 | tee -a "${CONFIG2_DIR}/train_stdout.log"
    CONFIG2_END=$(date +%s)
    echo "PILON-360M-full completed in $(elapsed_since $CONFIG2_START)"
    CONFIGS_DONE=$((CONFIGS_DONE + 1))
fi
print_eta $CONFIGS_DONE $CONFIGS_TOTAL

# --- Config 3: PILON-360M-tiered ---
log "[3/4] PILON-360M-tiered (n_hot=16, swap_interval=100)"
CONFIG3_DIR="${OUTPUT_BASE}/pilon_360m_tiered"
mkdir -p "${CONFIG3_DIR}"

if check_done "${CONFIG3_DIR}/final_model.pt" "PILON-360M-tiered"; then
    CONFIGS_DONE=$((CONFIGS_DONE + 1))
else
    RESUME_FLAGS=""
    LATEST=$(find_latest_checkpoint "${CONFIG3_DIR}")
    if [ -n "$LATEST" ]; then
        echo "RESUMING from ${LATEST}"
        RESUME_FLAGS="--resume ${LATEST}"
    fi
    CONFIG3_START=$(date +%s)
    python -u -m pilon_r.train \
        --model-size 360m \
        --ffn-type compositional \
        --phase1-sparse \
        --phase1-top-k 8 \
        --n-hot 16 \
        --swap-interval 100 \
        --hot-tier-bias-weight 0.01 \
        ${PILON_FLAGS} ${DATA_FLAGS} \
        --total-tokens ${TOTAL_TOKENS} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum ${GRAD_ACCUM} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --save-every ${SAVE_EVERY} \
        --output-dir "${CONFIG3_DIR}" \
        ${RESUME_FLAGS} \
        2>&1 | tee -a "${CONFIG3_DIR}/train_stdout.log"
    CONFIG3_END=$(date +%s)
    echo "PILON-360M-tiered completed in $(elapsed_since $CONFIG3_START)"
    CONFIGS_DONE=$((CONFIGS_DONE + 1))
fi
print_eta $CONFIGS_DONE $CONFIGS_TOTAL

# --- Config 4: PILON-360M-exit (gate training on Config 3 checkpoint) ---
log "[4/4] PILON-360M-exit (tiered + early exit gate training)"
CONFIG4_DIR="${OUTPUT_BASE}/pilon_360m_exit"
mkdir -p "${CONFIG4_DIR}"
TIERED_CKPT="${CONFIG3_DIR}/final_model.pt"

if check_done "${CONFIG4_DIR}/model_with_gates.pt" "PILON-360M-exit"; then
    CONFIGS_DONE=$((CONFIGS_DONE + 1))
else
    if [ ! -f "${TIERED_CKPT}" ]; then
        echo "ERROR: Config 3 checkpoint not found at ${TIERED_CKPT}"
        echo "       Config 4 depends on Config 3. Cannot continue."
        exit 1
    fi
    CONFIG4_START=$(date +%s)
    python -u -m pilon_r.train \
        --model-size 360m \
        --ffn-type compositional \
        --n-hot 16 \
        --swap-interval 100 \
        --enable-early-exit \
        --exit-threshold 0.5 \
        --forward-fast-mode on \
        --forward-fast-min-topk 1 \
        --no-checkpoint-ffn \
        ${DATA_FLAGS} \
        --train-exit-gates \
        --resume "${TIERED_CKPT}" \
        --batch-size ${BATCH_SIZE} \
        --seq-len ${SEQ_LEN} \
        --dataset ${DATASET} \
        --output-dir "${CONFIG4_DIR}" \
        2>&1 | tee -a "${CONFIG4_DIR}/train_stdout.log"
    CONFIG4_END=$(date +%s)
    echo "PILON-360M-exit completed in $(elapsed_since $CONFIG4_START)"
    CONFIGS_DONE=$((CONFIGS_DONE + 1))
fi
print_eta $CONFIGS_DONE $CONFIGS_TOTAL

# ============================================================
# Section 5: Post-Training
# ============================================================

log "Section 5: Post-Training Results"

# Print final val_loss/ppl from each config's trainv2.log
echo "--- Final Metrics (from trainv2.log) ---"
for config_dir in dense_360m pilon_360m_full pilon_360m_tiered pilon_360m_exit; do
    logfile="${OUTPUT_BASE}/${config_dir}/trainv2.log"
    echo ""
    echo "${config_dir}:"
    if [ -f "$logfile" ]; then
        # Print last lines containing val_loss or perplexity
        grep -i "val_loss\|perplexity\|ppl" "$logfile" | tail -3 || echo "  (no val_loss/ppl lines found)"
    else
        echo "  (trainv2.log not found)"
    fi
done

# Run benchmarks on all checkpoints
log "Running Benchmarks"
BENCHMARKS=(
    "${OUTPUT_BASE}/dense_360m/final_model.pt"
    "${OUTPUT_BASE}/pilon_360m_full/final_model.pt"
    "${OUTPUT_BASE}/pilon_360m_tiered/final_model.pt"
    "${OUTPUT_BASE}/pilon_360m_exit/model_with_gates.pt"
)

for ckpt in "${BENCHMARKS[@]}"; do
    if [ -f "$ckpt" ]; then
        echo ""
        echo "Benchmarking: $ckpt"
        python -u -m pilon_r.benchmark "$ckpt" --device cuda 2>&1 || echo "  (benchmark failed for $ckpt)"
    else
        echo "SKIP benchmark: $ckpt not found"
    fi
done

# Create results archive
log "Creating Results Archive"
ARCHIVE_NAME="360m_crossover_results.tar.gz"
ARCHIVE_PATH="${OUTPUT_BASE}/${ARCHIVE_NAME}"
tar -czf "${ARCHIVE_PATH}" \
    --exclude="${ARCHIVE_NAME}" \
    -C "$(dirname "${OUTPUT_BASE}")" \
    "$(basename "${OUTPUT_BASE}")" \
    2>&1 || echo "WARNING: tar archive creation failed"

if [ -f "${ARCHIVE_PATH}" ]; then
    ARCHIVE_SIZE=$(du -h "${ARCHIVE_PATH}" | cut -f1)
    echo "Archive created: ${ARCHIVE_PATH} (${ARCHIVE_SIZE})"
    echo "Download with:   rsync or scp from RunPod"
else
    echo "WARNING: Archive not created"
fi

# Final summary
SCRIPT_END=$(date +%s)
TOTAL_SECS=$((SCRIPT_END - SCRIPT_START))

log "COMPLETE"
echo "Total wall-clock time: $(elapsed_since $SCRIPT_START)"
echo ""
echo "Results:"
echo "  ${OUTPUT_BASE}/dense_360m/"
echo "  ${OUTPUT_BASE}/pilon_360m_full/"
echo "  ${OUTPUT_BASE}/pilon_360m_tiered/"
echo "  ${OUTPUT_BASE}/pilon_360m_exit/"
echo ""
echo "Archive: ${ARCHIVE_PATH}"
echo ""
echo "Done! You can now download the results and shut down the pod."
