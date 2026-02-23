# PILON-R 360M Comparison Experiment (PowerShell)
# Train and compare PILON-360M vs Dense-360M on 300B tokens of FineWeb-Edu

param(
    [Parameter(Position=0)]
    [ValidateSet("pilon", "dense", "benchmark", "prefill", "all", "help")]
    [string]$Command = "all"
)

$ErrorActionPreference = "Stop"

# Configuration
$OUTPUT_BASE = "outputs/360m_comparison"
$DATASET = "HuggingFaceFW/fineweb-edu"
$TOTAL_TOKENS = 1000000000  # 1B tokens
$DEVICE = "cuda"

# =============================================================================
# Step 1: Train PILON-360M
# =============================================================================
function Train-PILON {
    Write-Host "=============================================="
    Write-Host "Step 1: Training PILON-360M"
    Write-Host "=============================================="

    $OUTPUT_DIR = "$OUTPUT_BASE/360m_pilon"

    python -m pilon_r.train `
        --output-dir $OUTPUT_DIR `
        --model-size 360m `
        --ffn-type compositional `
        --dataset $DATASET `
        --total-tokens $TOTAL_TOKENS `
        --device $DEVICE `
        --phase1-sparse `
        --phase1-top-k 4 `
        --freeze-primitives-phase2 `
        --composition-temp 0.5 `
        --comp-lr-mult 4.0 `
        --comp-entropy-weight 0.001 `
        --log-timing `
        --log-comp-stats

    if ($LASTEXITCODE -ne 0) { throw "PILON-360M training failed" }

    Write-Host "PILON-360M training complete!"
    Write-Host "Checkpoint saved to: $OUTPUT_DIR/final_model.pt"
}

# =============================================================================
# Step 2: Train Dense-360M
# =============================================================================
function Train-Dense {
    Write-Host "=============================================="
    Write-Host "Step 2: Training Dense-360M"
    Write-Host "=============================================="

    $OUTPUT_DIR = "$OUTPUT_BASE/360m_dense"

    python -m pilon_r.train `
        --output-dir $OUTPUT_DIR `
        --model-size 360m `
        --ffn-type standard `
        --dataset $DATASET `
        --total-tokens $TOTAL_TOKENS `
        --device $DEVICE `
        --log-timing

    if ($LASTEXITCODE -ne 0) { throw "Dense-360M training failed" }

    Write-Host "Dense-360M training complete!"
    Write-Host "Checkpoint saved to: $OUTPUT_DIR/final_model.pt"
}

# =============================================================================
# Step 3: Benchmark Inference
# =============================================================================
function Run-Benchmark {
    Write-Host "=============================================="
    Write-Host "Step 3: Benchmarking Inference"
    Write-Host "=============================================="

    $PILON_CKPT = "$OUTPUT_BASE/360m_pilon/final_model.pt"
    $DENSE_CKPT = "$OUTPUT_BASE/360m_dense/final_model.pt"

    # Benchmark PILON
    Write-Host "Benchmarking PILON-360M..."
    python -m pilon_r.benchmark $PILON_CKPT `
        --device $DEVICE `
        --num-runs 100 `
        --max-tokens 100 `
        --output "$OUTPUT_BASE/benchmark_pilon.json"

    if ($LASTEXITCODE -ne 0) { throw "PILON benchmark failed" }

    # Benchmark Dense
    Write-Host "Benchmarking Dense-360M..."
    python -m pilon_r.benchmark $DENSE_CKPT `
        --device $DEVICE `
        --num-runs 100 `
        --max-tokens 100 `
        --output "$OUTPUT_BASE/benchmark_dense.json"

    if ($LASTEXITCODE -ne 0) { throw "Dense benchmark failed" }

    # Compare
    Write-Host "Comparing models..."
    python -m pilon_r.benchmark $PILON_CKPT `
        --compare $DENSE_CKPT `
        --device $DEVICE `
        --output "$OUTPUT_BASE/benchmark_comparison.json"

    if ($LASTEXITCODE -ne 0) { throw "Comparison failed" }

    Write-Host "Benchmarking complete!"
    Write-Host "Results saved to: $OUTPUT_BASE/benchmark_*.json"
}

# =============================================================================
# Step 4: Prefill Benchmark
# =============================================================================
function Run-PrefillBenchmark {
    Write-Host "=============================================="
    Write-Host "Step 4: Prefill Benchmark"
    Write-Host "=============================================="

    $PILON_CKPT = "$OUTPUT_BASE/360m_pilon/final_model.pt"
    $DENSE_CKPT = "$OUTPUT_BASE/360m_dense/final_model.pt"

    Write-Host "Benchmarking PILON-360M prefill..."
    python -m pilon_r.benchmark $PILON_CKPT `
        --device $DEVICE `
        --prefill `
        --output "$OUTPUT_BASE/prefill_pilon.json"

    if ($LASTEXITCODE -ne 0) { throw "PILON prefill benchmark failed" }

    Write-Host "Benchmarking Dense-360M prefill..."
    python -m pilon_r.benchmark $DENSE_CKPT `
        --device $DEVICE `
        --prefill `
        --output "$OUTPUT_BASE/prefill_dense.json"

    if ($LASTEXITCODE -ne 0) { throw "Dense prefill benchmark failed" }

    Write-Host "Prefill benchmarking complete!"
}

# =============================================================================
# Help
# =============================================================================
function Show-Help {
    Write-Host @"
PILON-R 360M Comparison Experiment

Usage: .\train_360m.ps1 [command]

Commands:
  pilon        Train PILON-360M only
  dense        Train Dense-360M only
  benchmark    Run inference benchmarks
  prefill      Run prefill benchmarks
  all          Run complete pipeline (default)
  help         Show this help message

Examples:
  .\train_360m.ps1 pilon         # Train PILON model
  .\train_360m.ps1 dense         # Train Dense model
  .\train_360m.ps1 all           # Run everything
"@
}

# =============================================================================
# Main Script
# =============================================================================
switch ($Command) {
    "pilon" { Train-PILON }
    "dense" { Train-Dense }
    "benchmark" { Run-Benchmark }
    "prefill" { Run-PrefillBenchmark }
    "all" {
        Train-PILON
        Train-Dense
        Run-Benchmark
        Run-PrefillBenchmark

        Write-Host ""
        Write-Host "=============================================="
        Write-Host "All steps complete!"
        Write-Host "=============================================="
        Write-Host "Results:"
        Write-Host "  - PILON-360M: $OUTPUT_BASE/360m_pilon/"
        Write-Host "  - Dense-360M: $OUTPUT_BASE/360m_dense/"
        Write-Host "  - Benchmarks: $OUTPUT_BASE/benchmark_*.json"
    }
    "help" { Show-Help }
    default {
        Write-Host "Unknown command: $Command"
        Show-Help
    }
}
