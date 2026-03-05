"""
PILON Throughput Profiler

Profiles the forward+backward pass of PILON vs Dense FFN to identify
where the throughput gap comes from.

Usage:
    python scripts/profile_pilon.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import time
from contextlib import contextmanager

from pilon_r.core.config import ModelConfig, PrimitiveConfig, BandConfig
from pilon_r.core.model import create_model, create_baseline_model


# ---------------------------------------------------------------------------
# Config matching the 48M ternary training run
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
BATCH = 8
SEQ = 512
WARMUP = 5
ITERS = 20


def get_48m_ternary_config():
    """Match the 48M ternary crossover config."""
    return ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        vocab_size=50257,
        max_seq_len=512,
        dropout=0.0,
        ffn_type="compositional",
        primitive_config=PrimitiveConfig(
            n_primitives=48,
            rank=48,
            top_k=8,
            bands=[
                BandConfig(name="early", layers=[0, 1, 2]),
                BandConfig(name="middle", layers=[3, 4, 5]),
                BandConfig(name="late", layers=[6, 7]),
            ],
            forward_fast_mode="on",
            forward_fast_min_topk=1,
            ternary_primitives=True,
            activation_bits=8,
            use_subln=True,
            use_squared_relu=True,
        ),
    )


def get_48m_dense_config():
    return ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        vocab_size=50257,
        max_seq_len=512,
        dropout=0.0,
        ffn_type="standard",
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@contextmanager
def cuda_timer(name, results):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    results.setdefault(name, []).append(elapsed_ms)


def summarize(results, label=""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    for name, times in results.items():
        times = times[WARMUP:]  # skip warmup
        if not times:
            continue
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        print(f"  {name:40s}  avg={avg:7.2f}ms  min={mn:7.2f}ms  max={mx:7.2f}ms")


# ---------------------------------------------------------------------------
# Test 1: Full model forward+backward throughput
# ---------------------------------------------------------------------------

def _enable_ternary_cache(model):
    """Pre-quantize ternary weights (simulates training loop cache)."""
    if not hasattr(model, 'primitive_banks') or model.primitive_banks is None:
        return
    for bank in model.primitive_banks.fc1_banks.values():
        bank.prepare_q_cache()
    for bank in model.primitive_banks.fc2_banks.values():
        bank.prepare_q_cache()


def _invalidate_ternary_cache(model):
    if not hasattr(model, 'primitive_banks') or model.primitive_banks is None:
        return
    for bank in model.primitive_banks.fc1_banks.values():
        bank.invalidate_q_cache()
    for bank in model.primitive_banks.fc2_banks.values():
        bank.invalidate_q_cache()


def benchmark_model(model, label, n_iters=WARMUP + ITERS):
    model.eval()
    model.to(DEVICE)
    results = {}
    use_cache = hasattr(model, 'primitive_banks')

    for i in range(n_iters):
        if use_cache:
            _enable_ternary_cache(model)
        x = torch.randint(0, 50257, (BATCH, SEQ), device=DEVICE)
        with cuda_timer("full_fwd+bwd", results):
            with torch.amp.autocast("cuda", dtype=DTYPE):
                out = model(x)
                logits = out["logits"]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    x.view(-1),
                )
            loss.backward()
        # Clear grads + invalidate cache (simulates optimizer.step())
        model.zero_grad(set_to_none=True)
        if use_cache:
            _invalidate_ternary_cache(model)

    summarize(results, label)
    times = results["full_fwd+bwd"][WARMUP:]
    avg_ms = sum(times) / len(times)
    tokens_per_iter = BATCH * SEQ
    tok_s = tokens_per_iter / (avg_ms / 1000)
    print(f"  --> {tok_s:,.0f} tok/s  ({avg_ms:.2f} ms/iter, {BATCH}x{SEQ}={tokens_per_iter} tokens)")
    return avg_ms


# ---------------------------------------------------------------------------
# Test 2: Component-level profiling of PrimitiveBank
# ---------------------------------------------------------------------------

def profile_primitive_bank(model, label):
    """Profile individual components of the PILON FFN path."""
    model.eval()
    model.to(DEVICE)

    # Find a CompositionalFFN layer
    from pilon_r.core.ffn import CompositionalFFN
    comp_ffn = None
    for module in model.modules():
        if isinstance(module, CompositionalFFN):
            comp_ffn = module
            break

    if comp_ffn is None:
        print(f"  No CompositionalFFN found in {label}, skipping component profile")
        return

    fc1_bank = comp_ffn.primitive_banks.get_fc1_bank(comp_ffn.layer_idx)
    fc2_bank = comp_ffn.primitive_banks.get_fc2_bank(comp_ffn.layer_idx)
    comp_weights = comp_ffn.composition_weights

    results = {}
    n_iters = WARMUP + ITERS

    for i in range(n_iters):
        x = torch.randn(BATCH, SEQ, model.config.d_model, device=DEVICE, dtype=DTYPE)

        # 1. Composition weight computation (softmax + topk)
        with cuda_timer("comp_weights_softmax", results):
            fc1_w = comp_weights.get_fc1_weights()
            fc2_w = comp_weights.get_fc2_weights()

        with cuda_timer("comp_weights_topk", results):
            k = comp_ffn.top_k_fc1
            _, fc1_idx = torch.topk(fc1_w, k, sorted=False)
            _, fc2_idx = torch.topk(fc2_w, k, sorted=False)
            fc1_top_w = fc1_w.index_select(0, fc1_idx)
            fc2_top_w = fc2_w.index_select(0, fc2_idx)

        # 2. Index_select (gathering primitives)
        with cuda_timer("gather_primitives", results):
            A1 = fc1_bank.A.index_select(0, fc1_idx)
            B1 = fc1_bank.B.index_select(0, fc1_idx)
            A2 = fc2_bank.A.index_select(0, fc2_idx)
            B2 = fc2_bank.B.index_select(0, fc2_idx)

        # 3. Ternary quantization (if enabled)
        if fc1_bank.ternary:
            with cuda_timer("ternary_quantize_fc1", results):
                A1_q, B1_q, _, _ = fc1_bank._quantize_weights(A1, B1)
            with cuda_timer("ternary_quantize_fc2", results):
                A2_q, B2_q, _, _ = fc2_bank._quantize_weights(A2, B2)
            with cuda_timer("activation_quantize", results):
                x_q, _, _ = fc1_bank._quantize_input(x)
        else:
            A1_q, B1_q = A1, B1
            x_q = x

        # 4. sqrt(w) scaling + concatenation
        with cuda_timer("sqrt_scale_concat", results):
            sqrt_w1 = torch.sqrt(fc1_top_w.float() / (fc1_top_w.sum() + 1e-8) + 1e-8).to(dtype=A1_q.dtype)
            A1_s = A1_q * sqrt_w1[:, None, None]
            B1_s = B1_q * sqrt_w1[:, None, None]
            r = A1_s.shape[2]
            A1_cat = A1_s.permute(1, 0, 2).contiguous().view(fc1_bank.d_in, k * r)
            B1_cat = B1_s.contiguous().view(k * r, fc1_bank.d_out)

        # 5. First GEMM (x @ A_cat)
        x_flat = x_q.reshape(-1, fc1_bank.d_in)
        with cuda_timer("gemm1_x@A", results):
            U = x_flat @ A1_cat  # (T, k*r)

        # 6. Latent scale/bias
        with cuda_timer("latent_scale_bias", results):
            scale = fc1_bank.latent_scale.to(dtype=U.dtype)
            bias = fc1_bank.latent_bias.to(dtype=U.dtype)
            U2 = U.view(-1, k, r)
            U2.mul_(scale)
            U2.add_(bias)
            U2 = U2.view(-1, k * r)

        # 7. Second GEMM (U @ B_cat)
        with cuda_timer("gemm2_U@B", results):
            h = U2 @ B1_cat  # (T, d_ff)

        # 8. Activation
        with cuda_timer("activation_fn", results):
            h_act = F.relu(h).square()  # squared_relu

        # 9. Full forward_topk_fused for fc1 (end-to-end)
        with cuda_timer("fc1_forward_topk_fused", results):
            fc1_out = fc1_bank.forward_topk_fused(
                x, fc1_w, top_k=k,
                top_indices=fc1_idx,
                active_rank=None,
                active_primitives=None,
            )

        # 10. Full forward_topk_fused for fc2
        h_for_fc2 = comp_ffn.activation(fc1_out)
        with cuda_timer("fc2_forward_topk_fused", results):
            fc2_out = fc2_bank.forward_topk_fused(
                h_for_fc2, fc2_w, top_k=k,
                top_indices=fc2_idx,
                active_rank=None,
                active_primitives=None,
            )

        # 11. Dense FFN equivalent for comparison
        with cuda_timer("dense_equiv_fc1", results):
            dense_h = x.reshape(-1, fc1_bank.d_in) @ torch.randn(
                fc1_bank.d_in, fc1_bank.d_out, device=DEVICE, dtype=DTYPE
            )
        with cuda_timer("dense_equiv_fc2", results):
            dense_out = dense_h @ torch.randn(
                fc2_bank.d_in, fc2_bank.d_out, device=DEVICE, dtype=DTYPE
            )

    summarize(results, f"{label} — Component Breakdown")


# ---------------------------------------------------------------------------
# Test 3: torch.profiler trace
# ---------------------------------------------------------------------------

def torch_profiler_trace(model, label, n_steps=3):
    """Run torch.profiler and print top CUDA operations."""
    model.eval()
    model.to(DEVICE)

    print(f"\n{'='*60}")
    print(f"  torch.profiler: {label}")
    print(f"{'='*60}")

    use_cache = hasattr(model, 'primitive_banks')

    # Warmup
    for _ in range(3):
        if use_cache:
            _enable_ternary_cache(model)
        x = torch.randint(0, 50257, (BATCH, SEQ), device=DEVICE)
        with torch.amp.autocast("cuda", dtype=DTYPE):
            out = model(x)
            logits = out["logits"]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
        if use_cache:
            _invalidate_ternary_cache(model)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(n_steps):
            if use_cache:
                _enable_ternary_cache(model)
            x = torch.randint(0, 50257, (BATCH, SEQ), device=DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                out = model(x)
                logits = out["logits"]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
            if use_cache:
                _invalidate_ternary_cache(model)

    print("\nTop 30 CUDA ops by total CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\nTop 20 CUDA ops by self CUDA time:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Batch: {BATCH}, Seq: {SEQ}")
    print(f"Warmup: {WARMUP}, Iters: {ITERS}")

    # Create models
    ternary_cfg = get_48m_ternary_config()
    dense_cfg = get_48m_dense_config()

    print("\nCreating ternary PILON model...")
    pilon_model = create_model(ternary_cfg)
    pilon_params = sum(p.numel() for p in pilon_model.parameters())
    print(f"  Parameters: {pilon_params:,}")

    print("Creating dense baseline model...")
    dense_model = create_model(dense_cfg)
    dense_params = sum(p.numel() for p in dense_model.parameters())
    print(f"  Parameters: {dense_params:,}")

    # Test 1: Full model throughput (eager)
    print("\n" + "="*60)
    print("  TEST 1: Full Model Throughput — Eager (fwd+bwd)")
    print("="*60)
    dense_ms = benchmark_model(dense_model, "Dense-48M (eager)")
    pilon_ms = benchmark_model(pilon_model, "PILON-48M-Ternary (eager)")
    print(f"\n  Ratio: PILON is {pilon_ms/dense_ms:.2f}x slower than Dense")

    # Test 1b: Full model throughput (torch.compile on PrimitiveBank)
    print("\n" + "="*60)
    print("  TEST 1b: Full Model Throughput — compiled PrimitiveBank (fwd+bwd)")
    print("="*60)
    try:
        from pilon_r.core.primitives import PrimitiveBank
        from pilon_r.core.ffn import CompositionalFFN
        # Compile key methods on PrimitiveBanks
        for module in pilon_model.modules():
            if isinstance(module, PrimitiveBank):
                module.forward_topk_fused = torch.compile(
                    module.forward_topk_fused, mode="reduce-overhead", fullgraph=False
                )
        pilon_c_ms = benchmark_model(pilon_model, "PILON-48M-Ternary (compiled banks)")
        print(f"\n  PILON compile speedup: {pilon_ms/pilon_c_ms:.2f}x")
        print(f"  Ratio vs Dense: PILON is {pilon_c_ms/dense_ms:.2f}x vs Dense")
    except Exception as e:
        import traceback
        print(f"  torch.compile failed: {e}")
        traceback.print_exc()

    # Test 2: Component-level breakdown
    print("\n" + "="*60)
    print("  TEST 2: Component-Level Profiling")
    print("="*60)
    profile_primitive_bank(pilon_model, "PILON-48M-Ternary")

    # Test 3: torch.profiler
    print("\n" + "="*60)
    print("  TEST 3: torch.profiler Trace")
    print("="*60)
    torch_profiler_trace(pilon_model, "PILON-48M-Ternary", n_steps=3)
    torch_profiler_trace(dense_model, "Dense-48M", n_steps=3)


if __name__ == "__main__":
    main()
