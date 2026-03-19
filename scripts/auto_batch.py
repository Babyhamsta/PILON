"""
Auto batch size finder for PILON-R training.

Probes increasing batch sizes until OOM, then returns the largest
that fits with headroom. Used to automatically configure training
for different GPU VRAM sizes.

Usage:
    python scripts/auto_batch.py --model-size 360m --attention-type gated_recurrence --seq-len 2048
    python scripts/auto_batch.py --model-size 360m --ffn-type standard --seq-len 2048
"""

import argparse
import gc
import sys
import torch


def find_max_batch_size(
    model_size: str = "360m",
    ffn_type: str = "compositional",
    attention_type: str = "gated_recurrence",
    seq_len: int = 2048,
    ternary: bool = True,
    compile_model: bool = True,
    headroom_gb: float = 1.0,
    grad_accum: int = 1,
) -> dict:
    """
    Find the maximum micro batch size that fits in GPU VRAM.

    Returns dict with optimal batch_size, grad_accum, throughput, and VRAM usage.
    """
    from pilon_r.core.config import ModelConfig, PrimitiveConfig, BandConfig
    from pilon_r.core.model import PILONTransformer

    if model_size == "360m":
        from pilon_r.configs.model_360m import get_360m_config
        config = get_360m_config(ffn_type)
    elif model_size == "48m":
        config = ModelConfig(ffn_type=ffn_type)
    else:
        raise ValueError(f"Unknown model_size: {model_size}")

    config.attention_type = attention_type
    config.max_seq_len = seq_len

    if ffn_type == "compositional" and ternary:
        config.primitive_config.ternary_primitives = True
        config.primitive_config.use_subln = True
        config.primitive_config.use_squared_relu = True
        config.primitive_config.activation_bits = 8
        config.primitive_config.forward_fast_mode = "on"
        config.primitive_config.forward_fast_min_topk = 1

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    target_vram = total_vram - headroom_gb

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_vram:.1f} GB, target max: {target_vram:.1f} GB")
    print(f"Model: {model_size}, FFN: {ffn_type}, Attention: {attention_type}")
    print(f"Seq len: {seq_len}, Ternary: {ternary}, Compile: {compile_model}")
    print()

    # Probe batch sizes
    candidates = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    best = None

    for batch_size in candidates:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            model = PILONTransformer(config).cuda()

            if compile_model:
                torch._dynamo.reset()
                torch._dynamo.config.capture_scalar_outputs = True
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
                model = torch.compile(model)

            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
            x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")

            # Run 2 steps to measure stable VRAM
            for _ in range(2):
                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, labels=x)
                    loss = out["loss"]
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / 1e9

            # Quick throughput estimate (3 steps)
            import time
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, labels=x)
                    out["loss"].backward()
                optimizer.step()
            torch.cuda.synchronize()
            tps = batch_size * seq_len * 3 / (time.time() - t0)

            fits = peak_vram < target_vram
            status = "OK" if fits else "OVER"
            print(f"  batch={batch_size:>3d}: {peak_vram:>5.1f} GB, {tps:>7.0f} tok/s  [{status}]")

            if fits:
                best = {
                    "batch_size": batch_size,
                    "peak_vram_gb": peak_vram,
                    "tok_per_sec": tps,
                    "seq_len": seq_len,
                }

            del model, optimizer, x
            torch.cuda.empty_cache()

            if not fits:
                break  # No point trying larger

        except torch.cuda.OutOfMemoryError:
            print(f"  batch={batch_size:>3d}: OOM")
            try:
                del model
            except:
                pass
            torch.cuda.empty_cache()
            break

        except Exception as e:
            print(f"  batch={batch_size:>3d}: ERROR ({e})")
            try:
                del model
            except:
                pass
            torch.cuda.empty_cache()
            break

    if best is None:
        print("\nERROR: Could not fit even batch_size=1!")
        return None

    # Compute recommended grad_accum to hit ~32-64k tokens per effective batch
    tokens_per_micro = best["batch_size"] * seq_len
    target_effective = 65536  # ~64k tokens per effective batch
    recommended_accum = max(1, round(target_effective / tokens_per_micro))

    best["grad_accum"] = recommended_accum
    best["effective_batch_tokens"] = tokens_per_micro * recommended_accum

    print()
    print(f"=== Recommended Configuration ===")
    print(f"  --batch-size {best['batch_size']}")
    print(f"  --grad-accum {best['grad_accum']}")
    print(f"  --seq-len {seq_len}")
    print(f"  Effective batch: {best['effective_batch_tokens']:,} tokens/step")
    print(f"  Peak VRAM: {best['peak_vram_gb']:.1f} GB / {total_vram:.1f} GB")
    print(f"  Throughput: {best['tok_per_sec']:.0f} tok/s")

    for tokens in [1_000_000_000, 2_000_000_000]:
        hours = tokens / best["tok_per_sec"] / 3600
        print(f"  {tokens/1e9:.0f}B tokens: {hours:.1f} hours")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto batch size finder")
    parser.add_argument("--model-size", type=str, default="360m", choices=["48m", "360m"])
    parser.add_argument("--ffn-type", type=str, default="compositional", choices=["compositional", "standard"])
    parser.add_argument("--attention-type", type=str, default="gated_recurrence")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--no-ternary", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--headroom", type=float, default=1.0, help="VRAM headroom in GB")
    args = parser.parse_args()

    find_max_batch_size(
        model_size=args.model_size,
        ffn_type=args.ffn_type,
        attention_type=args.attention_type,
        seq_len=args.seq_len,
        ternary=not args.no_ternary and args.ffn_type == "compositional",
        compile_model=not args.no_compile,
        headroom_gb=args.headroom,
    )
