#!/bin/bash
# PILON-R 360M Comparison Experiment
# Train and compare PILON-360M vs Dense-360M on 300B tokens of FineWeb-Edu

set -e  # Exit on error

# Configuration
OUTPUT_BASE="outputs/360m_comparison"
DATASET="HuggingFaceFW/fineweb-edu"
TOTAL_TOKENS=1000000000  # 1B tokens
DEVICE="cuda"

# =============================================================================
# Step 1: Train PILON-360M
# =============================================================================
train_pilon() {
    echo "=============================================="
    echo "Step 1: Training PILON-360M"
    echo "=============================================="

    OUTPUT_DIR="${OUTPUT_BASE}/360m_pilon"

    python -m pilon_r.train \
        --output-dir "$OUTPUT_DIR" \
        --model-size 360m \
        --ffn-type compositional \
        --dataset "$DATASET" \
        --total-tokens "$TOTAL_TOKENS" \
        --device "$DEVICE" \
        --phase1-sparse \
        --phase1-top-k 4 \
        --freeze-primitives-phase2 \
        --composition-temp 0.5 \
        --comp-lr-mult 4.0 \
        --comp-entropy-weight 0.001 \
        --log-timing \
        --log-comp-stats

    echo "PILON-360M training complete!"
    echo "Checkpoint saved to: ${OUTPUT_DIR}/final_model.pt"
}

# =============================================================================
# Step 2: Train Dense-360M
# =============================================================================
train_dense() {
    echo "=============================================="
    echo "Step 2: Training Dense-360M"
    echo "=============================================="

    OUTPUT_DIR="${OUTPUT_BASE}/360m_dense"

    python -m pilon_r.train \
        --output-dir "$OUTPUT_DIR" \
        --model-size 360m \
        --ffn-type standard \
        --dataset "$DATASET" \
        --total-tokens "$TOTAL_TOKENS" \
        --device "$DEVICE" \
        --log-timing

    echo "Dense-360M training complete!"
    echo "Checkpoint saved to: ${OUTPUT_DIR}/final_model.pt"
}

# =============================================================================
# Step 3: Benchmark Inference
# =============================================================================
benchmark() {
    echo "=============================================="
    echo "Step 3: Benchmarking Inference"
    echo "=============================================="

    PILON_CKPT="${OUTPUT_BASE}/360m_pilon/final_model.pt"
    DENSE_CKPT="${OUTPUT_BASE}/360m_dense/final_model.pt"

    # Benchmark PILON
    echo "Benchmarking PILON-360M..."
    python -m pilon_r.benchmark "$PILON_CKPT" \
        --device "$DEVICE" \
        --num-runs 100 \
        --max-tokens 100 \
        --output "${OUTPUT_BASE}/benchmark_pilon.json"

    # Benchmark Dense
    echo "Benchmarking Dense-360M..."
    python -m pilon_r.benchmark "$DENSE_CKPT" \
        --device "$DEVICE" \
        --num-runs 100 \
        --max-tokens 100 \
        --output "${OUTPUT_BASE}/benchmark_dense.json"

    # Compare
    echo "Comparing models..."
    python -m pilon_r.benchmark "$PILON_CKPT" \
        --compare "$DENSE_CKPT" \
        --device "$DEVICE" \
        --output "${OUTPUT_BASE}/benchmark_comparison.json"

    echo "Benchmarking complete!"
    echo "Results saved to: ${OUTPUT_BASE}/benchmark_*.json"
}

# =============================================================================
# Step 4: Prefill Benchmark
# =============================================================================
benchmark_prefill() {
    echo "=============================================="
    echo "Step 4: Prefill Benchmark"
    echo "=============================================="

    PILON_CKPT="${OUTPUT_BASE}/360m_pilon/final_model.pt"
    DENSE_CKPT="${OUTPUT_BASE}/360m_dense/final_model.pt"

    echo "Benchmarking PILON-360M prefill..."
    python -m pilon_r.benchmark "$PILON_CKPT" \
        --device "$DEVICE" \
        --prefill \
        --output "${OUTPUT_BASE}/prefill_pilon.json"

    echo "Benchmarking Dense-360M prefill..."
    python -m pilon_r.benchmark "$DENSE_CKPT" \
        --device "$DEVICE" \
        --prefill \
        --output "${OUTPUT_BASE}/prefill_dense.json"

    echo "Prefill benchmarking complete!"
}

# =============================================================================
# Main Script
# =============================================================================
print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  pilon        Train PILON-360M only"
    echo "  dense        Train Dense-360M only"
    echo "  benchmark    Run inference benchmarks"
    echo "  prefill      Run prefill benchmarks"
    echo "  all          Run complete pipeline"
    echo ""
    echo "Examples:"
    echo "  $0 pilon         # Train PILON model"
    echo "  $0 dense         # Train Dense model"
    echo "  $0 all           # Run everything"
}

# Parse command
case "${1:-all}" in
    pilon)
        train_pilon
        ;;
    dense)
        train_dense
        ;;
    benchmark)
        benchmark
        ;;
    prefill)
        benchmark_prefill
        ;;
    all)
        train_pilon
        train_dense
        benchmark
        benchmark_prefill
        echo ""
        echo "=============================================="
        echo "All steps complete!"
        echo "=============================================="
        echo "Results:"
        echo "  - PILON-360M: ${OUTPUT_BASE}/360m_pilon/"
        echo "  - Dense-360M: ${OUTPUT_BASE}/360m_dense/"
        echo "  - Benchmarks: ${OUTPUT_BASE}/benchmark_*.json"
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        echo "Unknown command: $1"
        print_usage
        exit 1
        ;;
esac
