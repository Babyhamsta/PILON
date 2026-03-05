"""
PILON-R Primitive Bank
Low-rank primitives for compositional FFN layers.

Key design decisions:
- Separate banks for fc1 and fc2 (different operations)
- Band-based sharing (layers in same band share primitives)
- Low-rank structure: each primitive is A @ B where A: (d_in, rank), B: (rank, d_out)
- Static per-layer composition weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# Ternary quantization helpers (BitNet b1.58)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """Minimal RMSNorm (avoids circular import with model.py)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stay in native dtype (fp16/bf16) — avoids expensive full-tensor fp32 copy.
        # Fuse two muls into one: x * (rms_inv * weight) saves a kernel launch.
        rms_inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * (rms_inv * self.weight)


class _TernaryWeightSTE(torch.autograd.Function):
    """Quantize weights to {-1, 0, +1} in a single autograd node.

    Forward: compute ternary weights scaled by absmean.
    Backward: STE — pass gradients straight through to shadow weights.

    All computation uses tensor ops (no .item()) so torch.compile can
    fuse them into a single kernel.
    """

    @staticmethod
    def forward(ctx, w, scale):
        s = scale.to(dtype=w.dtype)
        w_ternary = (w / s).round().clamp(-1, 1)
        return w_ternary * s

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # STE for w, None for scale


class _ActivationSTE(torch.autograd.Function):
    """Quantize activations in a single autograd node.

    Forward: quantize to [-Qb, Qb] range, rescale back to original range.
    Backward: STE — pass gradients straight through.

    All computation uses tensor ops (no .item()) so torch.compile can
    fuse them into a single kernel.
    """

    @staticmethod
    def forward(ctx, x, scale, Qb):
        s = scale.to(dtype=x.dtype)
        x_quant = (x * (Qb / s)).round().clamp(-Qb, Qb)
        return x_quant * (s / Qb)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None  # STE for x, None for scale/Qb


def ternary_quantize(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to {-1, 0, +1} using absmean scaling + STE.

    Uses a custom autograd Function to collapse quantization into a single
    autograd node, reducing memory from ~7-8 intermediate tensors to 1.

    Scale is computed in native dtype (fp16 mean is sufficient for absmean),
    then stored as fp32 for the return value.

    Returns:
        (w_ternary_ste, scale) where scale is the absmean in fp32.
        During backward, gradients flow through w unchanged (straight-through).
    """
    with torch.no_grad():
        scale = w.abs().mean().float().clamp(min=1e-8)
    w_ste = _TernaryWeightSTE.apply(w, scale)
    return w_ste, scale


def activation_quantize(x: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Quantize activations to `bits`-bit using absmax scaling + STE.

    Uses a custom autograd Function to collapse the round+clamp+rescale
    into a single autograd node.

    Scale is computed in native dtype to avoid full-tensor float32 conversion.

    Returns:
        (x_quant_ste, scale, Qb) where Qb = 2^(bits-1).
        During backward, gradients flow through x unchanged.
    """
    if bits <= 0:
        return x, torch.ones(1, device=x.device, dtype=torch.float32), 1.0
    Qb = 2 ** (bits - 1)
    with torch.no_grad():
        scale = x.abs().amax().float().clamp(min=1e-8)
    x_ste = _ActivationSTE.apply(x, scale, Qb)
    return x_ste, scale, Qb


class LowRankPrimitive(nn.Module):
    """
    A single low-rank primitive: W = A @ B

    A: (d_in, rank)
    B: (rank, d_out)
    Full weight would be: (d_in, d_out)
    """

    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank

        # Initialize with scaled random values
        # Using Kaiming-like initialization adapted for low-rank
        std_a = 1.0 / math.sqrt(d_in)
        std_b = 1.0 / math.sqrt(rank)

        self.A = nn.Parameter(torch.randn(d_in, rank) * std_a)
        self.B = nn.Parameter(torch.randn(rank, d_out) * std_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply primitive: x @ A @ B"""
        return x @ self.A @ self.B

    def get_full_weight(self) -> torch.Tensor:
        """Reconstruct full weight matrix (for analysis only)."""
        return self.A @ self.B

    def parameter_count(self) -> int:
        """Number of parameters in this primitive."""
        return self.d_in * self.rank + self.rank * self.d_out


class PrimitiveBank(nn.Module):
    """
    A bank of low-rank primitives shared across layers in a band.

    The bank contains n_primitives, each a low-rank matrix.
    Layers compose these primitives using learned weights.

    Optimized: Stores packed A/B tensors directly for efficient vectorized computation.
    - A: (n_primitives, d_in, rank)
    - B: (n_primitives, rank, d_out)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_primitives: int,
        rank: int,
        name: str = "bank",
        ternary: bool = False,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_primitives = n_primitives
        self.rank = rank
        self.name = name
        self.ternary = ternary
        self.activation_bits = activation_bits

        # Packed tensors for vectorized computation
        # Using Kaiming-like initialization adapted for low-rank
        std_b = 1.0 / math.sqrt(rank)

        # Orthogonalize A across primitives at init (CPU-side is fine)
        flat_cols = n_primitives * rank
        A_flat = torch.randn(d_in, flat_cols)
        if d_in >= flat_cols:
            A_ortho, _ = torch.linalg.qr(A_flat, mode="reduced")
        else:
            A_ortho, _ = torch.linalg.qr(A_flat.T, mode="reduced")
            A_ortho = A_ortho.T
        A_ortho = A_ortho.reshape(d_in, n_primitives, rank).permute(1, 0, 2).contiguous()

        self.A = nn.Parameter(A_ortho)
        self.B = nn.Parameter(torch.randn(n_primitives, rank, d_out) * std_b)

        # Latent affine transform in rank space (shared across primitives)
        self.latent_scale = nn.Parameter(torch.ones(rank))
        self.latent_bias = nn.Parameter(torch.zeros(rank))

        # Ternary quantization support
        if ternary:
            self.input_norm = _RMSNorm(d_in)

        # Ternary weight cache: pre-quantize all primitives once per
        # optimizer step, then index_select from cached quantized tensors.
        # This avoids redundant quantization across micro-batches and layers.
        self._A_q_cache: Optional[torch.Tensor] = None
        self._B_q_cache: Optional[torch.Tensor] = None
        self._q_cache_valid: bool = False

    # ------------------------------------------------------------------
    # Ternary weight cache management
    # ------------------------------------------------------------------

    def prepare_q_cache(self) -> None:
        """Pre-quantize all primitives with autograd. Call once per step.

        The cached tensors are part of the autograd graph so gradients
        from all micro-batches accumulate correctly through the STE.
        """
        if not self.ternary:
            return
        self._A_q_cache, _ = ternary_quantize(self.A)
        self._B_q_cache, _ = ternary_quantize(self.B)
        self._q_cache_valid = True

    def invalidate_q_cache(self) -> None:
        """Invalidate cache. Call after optimizer.step()."""
        self._A_q_cache = None
        self._B_q_cache = None
        self._q_cache_valid = False

    # ------------------------------------------------------------------
    # Ternary quantization helpers (no-ops when self.ternary is False)
    # ------------------------------------------------------------------

    def _quantize_weights(
        self, A: torch.Tensor, B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize A and B to ternary. No-op when ternary=False."""
        if not self.ternary:
            z = torch.ones(1, device=A.device, dtype=torch.float32)
            return A, B, z, z
        A_q, scale_A = ternary_quantize(A)
        B_q, scale_B = ternary_quantize(B)
        return A_q, B_q, scale_A, scale_B

    def _quantize_weights_or_cache(
        self, top_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get quantized weights, using cache if available.

        If the full-bank cache is valid, index_select from it (cheap).
        Otherwise, fall back to quantizing the selected primitives.
        """
        z = torch.ones(1, device=self.A.device, dtype=torch.float32)
        if not self.ternary:
            A_sel = self.A.index_select(0, top_indices)
            B_sel = self.B.index_select(0, top_indices)
            return A_sel, B_sel, z, z
        if self._q_cache_valid and self._A_q_cache is not None:
            A_sel = self._A_q_cache.index_select(0, top_indices)
            B_sel = self._B_q_cache.index_select(0, top_indices)
            return A_sel, B_sel, z, z
        # Fallback: gather then quantize
        A_sel = self.A.index_select(0, top_indices)
        B_sel = self.B.index_select(0, top_indices)
        A_q, scale_A = ternary_quantize(A_sel)
        B_q, scale_B = ternary_quantize(B_sel)
        return A_q, B_q, scale_A, scale_B

    def _quantize_input(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply input_norm + activation quantization. No-op when ternary=False."""
        if not self.ternary:
            return x, torch.ones(1, device=x.device, dtype=torch.float32), 1.0
        x = self.input_norm(x)
        x_q, scale_x, Qb = activation_quantize(x, self.activation_bits)
        return x_q, scale_x, Qb

    def _rescale_output(
        self,
        output: torch.Tensor,
        scale_A: torch.Tensor,
        scale_B: torch.Tensor,
        scale_x: torch.Tensor,
        Qb: float,
    ) -> torch.Tensor:
        """No-op: STE already embeds scale factors in forward values."""
        return output

    def quantize_external(
        self, A: torch.Tensor, B: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Public API for MoE code that accesses bank.A/B directly.

        Returns:
            (A_q, B_q, x_q, scale_A, scale_B, scale_x, Qb)
        """
        A_q, B_q, scale_A, scale_B = self._quantize_weights(A, B)
        x_q, scale_x, Qb = self._quantize_input(x)
        return A_q, B_q, x_q, scale_A, scale_B, scale_x, Qb

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_k: Optional[int] = None,
        active_rank: Optional[int] = None,
        active_primitives: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply weighted combination of primitives (vectorized).

        Args:
            x: Input tensor (batch, seq, d_in)
            weights: Composition weights (n_primitives,) - already softmaxed
            top_k: If provided, only use top_k primitives (sparse composition)
            active_rank: If provided, only use first active_rank dimensions of rank (for rank scheduling)

        Returns:
            Output tensor (batch, seq, d_out)
        """
        # Support active primitive scheduling
        A = self.A
        B = self.B
        if active_primitives is not None and active_primitives < self.n_primitives:
            active_primitives = max(1, active_primitives)
            A = A[:active_primitives]
            B = B[:active_primitives]
            weights = weights[:active_primitives]
            if top_k is not None:
                top_k = min(top_k, active_primitives)

        n_primitives = A.size(0)
        if top_k is not None and top_k < n_primitives:
            # Sparse composition: only use top_k primitives
            top_weights, top_indices = torch.topk(weights, top_k, sorted=False)
            top_weights = top_weights / (top_weights.sum() + 1e-8)
            A = A.index_select(0, top_indices)
            B = B.index_select(0, top_indices)
            weights = top_weights

        # Ternary quantization on full rank (matches cached path behaviour)
        A, B, scale_A, scale_B = self._quantize_weights(A, B)

        # Rank scheduling: slice after quantization so scale is consistent
        # with forward_topk_fused/forward_sparse which quantize full then slice
        if active_rank is not None:
            A = A[:, :, :active_rank]
            B = B[:, :active_rank, :]
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
        else:
            scale = self.latent_scale
            bias = self.latent_bias
        x, scale_x, Qb = self._quantize_input(x)

        batch, seq, _ = x.shape
        x_flat = x.reshape(-1, self.d_in)

        # Vectorized: U = X @ A for all primitives, Y = U @ B with weights
        U = torch.einsum("td,pdr->tpr", x_flat, A)  # [T, P, r]
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U.mul_(scale)
        U.add_(bias)
        out_flat = torch.einsum("tpr,pro,p->to", U, B, weights)  # [T, d_out]

        out_flat = self._rescale_output(out_flat, scale_A, scale_B, scale_x, Qb)
        return out_flat.view(batch, seq, self.d_out)

    def select_topk_primitives(
        self,
        top_indices: torch.Tensor,
        active_rank: Optional[int] = None,
        active_primitives: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather selected primitive factors for reuse across micro-batches.
        """
        A = self.A[:, :, :active_rank] if active_rank is not None else self.A
        B = self.B[:, :active_rank, :] if active_rank is not None else self.B
        if active_primitives is not None and active_primitives < self.n_primitives:
            active_primitives = max(1, active_primitives)
            A = A[:active_primitives]
            B = B[:active_primitives]
            top_indices = top_indices[top_indices < active_primitives]
            if top_indices.numel() == 0:
                top_indices = torch.arange(
                    min(1, A.size(0)), device=A.device, dtype=torch.long
                )
        return A.index_select(0, top_indices), B.index_select(0, top_indices)

    def forward_topk_fused(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_k: int,
        active_rank: Optional[int] = None,
        top_indices: Optional[torch.Tensor] = None,
        active_primitives: Optional[int] = None,
        preselected: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Sparse fused forward: build a single low-rank map from top-k primitives.

        Computes x @ A_cat @ B_cat where A_cat/B_cat are formed by
        concatenating top-k primitives scaled by sqrt(weights).
        """
        if top_k <= 0:
            raise ValueError("top_k must be >= 1")

        # Support rank scheduling
        A = self.A[:, :, :active_rank] if active_rank is not None else self.A
        B = self.B[:, :active_rank, :] if active_rank is not None else self.B
        if active_rank is not None:
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
            rank = active_rank
        else:
            scale = self.latent_scale
            bias = self.latent_bias
            rank = self.rank

        # Support active primitive scheduling
        if active_primitives is not None and active_primitives < self.n_primitives:
            active_primitives = max(1, active_primitives)
            A = A[:active_primitives]
            B = B[:active_primitives]
            weights = weights[:active_primitives]
            top_k = min(top_k, active_primitives)

        if top_indices is None:
            top_weights, top_indices = torch.topk(weights, top_k, sorted=False)
        else:
            top_weights = weights.index_select(0, top_indices)
        # Normalize weights in native dtype (avoid float32 round-trip)
        top_weights = top_weights / (top_weights.sum() + 1e-8)

        # Gather top-k primitives (or reuse preselected factors when provided).
        if preselected is not None:
            if top_indices is None:
                raise ValueError("preselected factors require top_indices")
            A_sel, B_sel = preselected
            top_k = int(A_sel.size(0))
            if top_weights.numel() != top_k:
                top_weights = top_weights[:top_k]
                top_weights = top_weights / (top_weights.sum() + 1e-8)
            # Quantize preselected (can't use bank cache for these)
            A_sel, B_sel, scale_A, scale_B = self._quantize_weights(A_sel, B_sel)
        else:
            # Use pre-quantized cache if available (index_select is cheap)
            A_sel, B_sel, scale_A, scale_B = self._quantize_weights_or_cache(top_indices)
            # Apply rank scheduling to selected primitives
            if active_rank is not None:
                A_sel = A_sel[:, :, :active_rank]
                B_sel = B_sel[:, :active_rank, :]

        x, scale_x, Qb = self._quantize_input(x)

        # Scale each primitive by sqrt(weight) and concatenate into a single low-rank map
        # Stay in native dtype — sqrt of small positive weights is numerically fine in fp16
        sqrt_w = torch.sqrt(top_weights.to(dtype=A_sel.dtype) + 1e-8)
        A_sel = A_sel * sqrt_w[:, None, None]
        B_sel = B_sel * sqrt_w[:, None, None]

        # A_cat: (d_in, k*r), B_cat: (k*r, d_out)
        A_cat = A_sel.permute(1, 0, 2).contiguous().view(self.d_in, top_k * rank)
        B_cat = B_sel.contiguous().view(top_k * rank, self.d_out)

        batch, seq, _ = x.shape
        x_flat = x.reshape(-1, self.d_in)

        # Two GEMMs with shared latent scale/bias
        U = x_flat @ A_cat  # (T, k*r)
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U = U.view(-1, top_k, rank)
        U.mul_(scale)
        U.add_(bias)
        U = U.view(-1, top_k * rank)

        out_flat = U @ B_cat  # (T, d_out)
        out_flat = self._rescale_output(out_flat, scale_A, scale_B, scale_x, Qb)
        return out_flat.view(batch, seq, self.d_out)

    def forward_sparse(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_k: int,
        active_rank: Optional[int] = None,
        top_indices: Optional[torch.Tensor] = None,
        active_primitives: Optional[int] = None
    ) -> torch.Tensor:
        """
        Truly sparse forward: only compute selected top-k primitive outputs.

        Unlike compute_all_outputs(), this never materializes non-selected
        primitive outputs.
        """
        if top_k <= 0:
            raise ValueError("top_k must be >= 1")

        # Support rank scheduling
        A = self.A[:, :, :active_rank] if active_rank is not None else self.A
        B = self.B[:, :active_rank, :] if active_rank is not None else self.B
        if active_rank is not None:
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
        else:
            scale = self.latent_scale
            bias = self.latent_bias

        # Support active primitive scheduling
        if active_primitives is not None and active_primitives < self.n_primitives:
            active_primitives = max(1, active_primitives)
            A = A[:active_primitives]
            B = B[:active_primitives]
            weights = weights[:active_primitives]
            top_k = min(top_k, active_primitives)

        if top_indices is None:
            top_weights, top_indices = torch.topk(weights, top_k, sorted=False)
        else:
            top_weights = weights.index_select(0, top_indices)
            top_k = top_indices.numel()
        top_weights = top_weights / (top_weights.sum() + 1e-8)

        # Only gather the selected primitives (use cache if available)
        A_sel, B_sel, scale_A, scale_B = self._quantize_weights_or_cache(top_indices)
        # Apply rank scheduling to selected primitives (cache stores full rank)
        if active_rank is not None:
            A_sel = A_sel[:, :, :active_rank]
            B_sel = B_sel[:, :active_rank, :]
        x, scale_x, Qb = self._quantize_input(x)

        batch, seq, _ = x.shape
        x_flat = x.reshape(-1, self.d_in)

        # Batched sparse primitive computation over selected k
        U = torch.einsum("td,kdr->tkr", x_flat, A_sel)  # (T, k, r)
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U.mul_(scale)
        U.add_(bias)

        Y = torch.einsum("tkr,kro->tko", U, B_sel)  # (T, k, d_out)
        out_flat = torch.einsum("tko,k->to", Y, top_weights.to(dtype=Y.dtype))
        out_flat = self._rescale_output(out_flat, scale_A, scale_B, scale_x, Qb)
        return out_flat.view(batch, seq, self.d_out)

    def forward_fast(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_k: Optional[int] = None,
        active_rank: Optional[int] = None,
        active_primitives: Optional[int] = None
    ) -> torch.Tensor:
        """
        Faster forward: compute all outputs once, then mix.

        This is more efficient when the same primitive outputs could be reused,
        and for the top_k selection path.

        Args:
            x: Input tensor (batch, seq, d_in)
            weights: Composition weights (n_primitives,) - already softmaxed
            top_k: If provided, only use top_k primitives (sparse composition)
            active_rank: If provided, only use first active_rank dimensions of rank

        Returns:
            Output tensor (batch, seq, d_out)
        """
        # Support active primitive scheduling (slice weights and clamp top_k)
        if active_primitives is not None and active_primitives < self.n_primitives:
            active_primitives = max(1, active_primitives)
            weights = weights[:active_primitives]
            if top_k is not None:
                top_k = min(top_k, active_primitives)

        n_primitives = weights.numel()
        if top_k is not None and top_k < n_primitives:
            # Sparse path without materializing all primitive outputs
            return self.forward_sparse(
                x,
                weights,
                top_k=top_k,
                active_rank=active_rank,
                active_primitives=active_primitives
            )

        # Dense path: compute all primitive outputs once, then mix
        all_outputs = self.compute_all_outputs(
            x,
            active_rank=active_rank,
            active_primitives=active_primitives
        )
        return torch.einsum("bspd,p->bsd", all_outputs, weights)

    def _forward_loop(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_k: Optional[int] = None,
        active_rank: Optional[int] = None,
        active_primitives: Optional[int] = None
    ) -> torch.Tensor:
        """
        Loop-based implementation for correctness verification.

        This is slower but easier to verify correctness.
        """
        batch, seq, _ = x.shape

        A = self.A[:, :, :active_rank] if active_rank is not None else self.A
        B = self.B[:, :active_rank, :] if active_rank is not None else self.B
        if active_rank is not None:
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
        else:
            scale = self.latent_scale
            bias = self.latent_bias

        if active_primitives is not None and active_primitives < self.n_primitives:
            active_primitives = max(1, active_primitives)
            A = A[:active_primitives]
            B = B[:active_primitives]
            weights = weights[:active_primitives]
            if top_k is not None:
                top_k = min(top_k, active_primitives)

        n_primitives = A.size(0)
        if top_k is not None and top_k < n_primitives:
            top_weights, top_indices = torch.topk(weights, top_k, sorted=False)
            top_weights = top_weights / (top_weights.sum() + 1e-8)
        else:
            top_indices = torch.arange(n_primitives, device=x.device)
            top_weights = weights

        # Ternary quantization (no-op when ternary=False)
        A, B, scale_A, scale_B = self._quantize_weights(A, B)
        x, scale_x, Qb = self._quantize_input(x)
        out = torch.zeros(batch, seq, self.d_out, device=x.device, dtype=x.dtype)

        for i, idx in enumerate(top_indices):
            A_i = A[idx]  # [d_in, r]
            B_i = B[idx]  # [r, d_out]
            w_i = top_weights[i]
            latent = x @ A_i
            latent = latent * scale + bias
            prim_out = latent @ B_i  # [batch, seq, d_out]
            out = out + w_i * prim_out

        out = self._rescale_output(out, scale_A, scale_B, scale_x, Qb)
        return out

    def compute_all_outputs(
        self,
        x: torch.Tensor,
        active_rank: Optional[int] = None,
        active_primitives: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute outputs for all primitives in a batched fashion.

        Args:
            x: Input tensor (batch, seq, d_in)
            active_rank: If provided, only use first active_rank dimensions of rank

        Returns:
            Tensor of shape (batch, seq, n_primitives, d_out)
        """
        batch, seq, _ = x.shape

        # Support rank scheduling
        A = self.A[:, :, :active_rank] if active_rank is not None else self.A
        B = self.B[:, :active_rank, :] if active_rank is not None else self.B
        if active_rank is not None:
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
        else:
            scale = self.latent_scale
            bias = self.latent_bias

        # Support active primitive scheduling
        if active_primitives is not None and active_primitives < self.n_primitives:
            active_primitives = max(1, active_primitives)
            A = A[:active_primitives]
            B = B[:active_primitives]

        # Ternary quantization (no-op when ternary=False)
        A, B, scale_A, scale_B = self._quantize_weights(A, B)
        x, scale_x, Qb = self._quantize_input(x)
        x_flat = x.reshape(-1, self.d_in)

        U = torch.einsum("td,pdr->tpr", x_flat, A)  # [T, P, r]
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U.mul_(scale)
        U.add_(bias)
        Y = torch.einsum("tpr,pro->tpo", U, B)  # [T, P, d_out]

        Y = self._rescale_output(Y, scale_A, scale_B, scale_x, Qb)
        n_primitives = A.size(0)
        return Y.view(batch, seq, n_primitives, self.d_out)

    def get_composed_weight(
        self,
        weights: torch.Tensor,
        top_k: Optional[int] = None,
        active_primitives: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get the composed full weight matrix (for analysis).

        This reconstructs what the effective weight would be
        without actually computing the forward pass.
        """
        A, B = self.A, self.B

        if active_primitives is not None and active_primitives < self.n_primitives:
            active_primitives = max(1, active_primitives)
            A = A[:active_primitives]
            B = B[:active_primitives]
            weights = weights[:active_primitives]
            if top_k is not None:
                top_k = min(top_k, active_primitives)

        n_primitives = weights.numel()
        if top_k is not None and top_k < n_primitives:
            top_weights, top_indices = torch.topk(weights, top_k, sorted=False)
            top_weights = top_weights / (top_weights.sum() + 1e-8)
            A = A.index_select(0, top_indices)
            B = B.index_select(0, top_indices)
            weights = top_weights

        composed = torch.einsum("p,pir,pro->io", weights, A, B)
        return composed

    def parameter_count(self) -> int:
        """Total parameters in this bank."""
        # A: (n_primitives, d_in, rank) + B: (n_primitives, rank, d_out)
        return self.n_primitives * (self.d_in * self.rank + self.rank * self.d_out) + 2 * self.rank


class BandPrimitiveBanks(nn.Module):
    """
    Primitive banks organized by bands.

    Each band has TWO banks (one for fc1, one for fc2).
    Layers in the same band share the same banks.

    This is the main interface for the compositional FFN.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_primitives: int,
        rank: int,
        bands: List[Dict],  # [{"name": "early", "layers": [0,1,2]}, ...]
        share_fc1_fc2: bool = False,  # Should be False for Phase A
        n_hot: Optional[int] = None,  # Phase B.5b: tiered primitive loading
        swap_interval: int = 100,     # Steps between hot/warm swaps
        ternary: bool = False,        # Ternary weight quantization
        activation_bits: int = 8,     # Activation quantization bitwidth
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_primitives = n_primitives
        self.rank = rank
        self.bands = bands
        self.share_fc1_fc2 = share_fc1_fc2
        self.n_hot = n_hot
        self.ternary = ternary
        self.activation_bits = activation_bits

        # Create layer -> band mapping
        self.layer_to_band: Dict[int, str] = {}
        for band in bands:
            for layer_idx in band["layers"]:
                self.layer_to_band[layer_idx] = band["name"]

        # Determine whether to use tiered banks
        use_tiered = (n_hot is not None and n_hot < n_primitives)
        if use_tiered and ternary:
            raise NotImplementedError(
                "Ternary quantization is not yet supported with tiered primitive banks. "
                "Disable tiering (remove --n-hot) or disable ternary (remove --ternary)."
            )

        # Create banks for each band
        # fc1: d_model -> d_ff (expansion)
        # fc2: d_ff -> d_model (compression)
        self.fc1_banks = nn.ModuleDict()
        self.fc2_banks = nn.ModuleDict()

        for band in bands:
            band_name = band["name"]

            if use_tiered:
                from .tiered_bank import TieredPrimitiveBank
                self.fc1_banks[band_name] = TieredPrimitiveBank(
                    d_in=d_model,
                    d_out=d_ff,
                    n_primitives=n_primitives,
                    rank=rank,
                    n_hot=n_hot,
                    swap_interval=swap_interval,
                    name=f"{band_name}_fc1",
                )
            else:
                # fc1 bank: projects up
                self.fc1_banks[band_name] = PrimitiveBank(
                    d_in=d_model,
                    d_out=d_ff,
                    n_primitives=n_primitives,
                    rank=rank,
                    name=f"{band_name}_fc1",
                    ternary=ternary,
                    activation_bits=activation_bits,
                )

            if share_fc1_fc2:
                # Share bank (NOT RECOMMENDED for Phase A)
                # Would need to handle dimension mismatch
                raise NotImplementedError(
                    "Shared fc1/fc2 banks not implemented for Phase A. "
                    "fc1 and fc2 have different dimensions."
                )
            else:
                if use_tiered:
                    from .tiered_bank import TieredPrimitiveBank
                    self.fc2_banks[band_name] = TieredPrimitiveBank(
                        d_in=d_ff,
                        d_out=d_model,
                        n_primitives=n_primitives,
                        rank=rank,
                        n_hot=n_hot,
                        swap_interval=swap_interval,
                        name=f"{band_name}_fc2",
                    )
                else:
                    # fc2 bank: projects down (separate)
                    self.fc2_banks[band_name] = PrimitiveBank(
                        d_in=d_ff,
                        d_out=d_model,
                        n_primitives=n_primitives,
                        rank=rank,
                        name=f"{band_name}_fc2",
                        ternary=ternary,
                        activation_bits=activation_bits,
                    )

    def get_fc1_bank(self, layer_idx: int) -> PrimitiveBank:
        """Get the fc1 bank for a given layer."""
        band_name = self.layer_to_band[layer_idx]
        return self.fc1_banks[band_name]

    def get_fc2_bank(self, layer_idx: int) -> PrimitiveBank:
        """Get the fc2 bank for a given layer."""
        band_name = self.layer_to_band[layer_idx]
        return self.fc2_banks[band_name]

    def get_band_name(self, layer_idx: int) -> str:
        """Get band name for a layer."""
        return self.layer_to_band[layer_idx]

    def parameter_count(self) -> Dict[str, int]:
        """Count parameters by bank."""
        counts = {}
        for name, bank in self.fc1_banks.items():
            counts[f"fc1_{name}"] = bank.parameter_count()
        for name, bank in self.fc2_banks.items():
            counts[f"fc2_{name}"] = bank.parameter_count()
        counts["total"] = sum(counts.values())
        return counts


class LayerCompositionWeights(nn.Module):
    """
    Per-layer composition weights for combining primitives.

    Each layer has its own learned weights for how to combine
    primitives from the shared bank.
    """

    def __init__(
        self,
        n_primitives: int,
        top_k: int,
        layer_idx: int,
        temperature: float = 1.0
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.temperature = temperature

        # Learnable logits for fc1 and fc2
        # Initialize uniformly to start with diverse usage
        self.fc1_logits = nn.Parameter(torch.zeros(n_primitives))
        self.fc2_logits = nn.Parameter(torch.zeros(n_primitives))

        # Small random perturbation to break symmetry
        nn.init.normal_(self.fc1_logits, mean=0, std=0.01)
        nn.init.normal_(self.fc2_logits, mean=0, std=0.01)

    def get_fc1_weights(self) -> torch.Tensor:
        """Get softmax weights for fc1 composition."""
        return F.softmax(self.fc1_logits / self.temperature, dim=0)

    def get_fc2_weights(self) -> torch.Tensor:
        """Get softmax weights for fc2 composition."""
        return F.softmax(self.fc2_logits / self.temperature, dim=0)

    def get_fc1_logits(self) -> torch.Tensor:
        """Get raw fc1 logits (for top-k selection)."""
        return self.fc1_logits

    def get_fc2_logits(self) -> torch.Tensor:
        """Get raw fc2 logits (for top-k selection)."""
        return self.fc2_logits

    def compute_entropy(self) -> Dict[str, float]:
        """Compute entropy of composition weights (for monitoring)."""
        def entropy(logits: torch.Tensor) -> float:
            probs = F.softmax(logits / self.temperature, dim=0)
            log_probs = F.log_softmax(logits / self.temperature, dim=0)
            return -(probs * log_probs).sum().item()

        return {
            "fc1_entropy": entropy(self.fc1_logits),
            "fc2_entropy": entropy(self.fc2_logits),
        }

    def get_top_k_indices(self) -> Dict[str, torch.Tensor]:
        """Get indices of top_k primitives for each projection."""
        fc1_weights = self.get_fc1_weights()
        fc2_weights = self.get_fc2_weights()

        _, fc1_top = torch.topk(fc1_weights, self.top_k, sorted=False)
        _, fc2_top = torch.topk(fc2_weights, self.top_k, sorted=False)

        return {
            "fc1_top_k": fc1_top,
            "fc2_top_k": fc2_top,
        }


class ExpertCompositionBank(nn.Module):
    """
    Bank of expert composition weights for MoE (Phase B).

    Each expert has its own composition weights over the shared primitives.
    This is like having n_experts copies of LayerCompositionWeights, where the
    router selects which experts to use per token.

    Key difference from LayerCompositionWeights:
    - LayerCompositionWeights: (n_primitives,) - one composition per layer
    - ExpertCompositionBank: (n_experts, n_primitives) - multiple compositions
    """

    def __init__(
        self,
        n_experts: int,
        n_primitives: int,
        top_k_primitives: int,
        layer_idx: int,
        temperature: float = 1.0
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_primitives = n_primitives
        self.top_k_primitives = top_k_primitives
        self.layer_idx = layer_idx
        self.temperature = temperature

        # Expert composition logits: (n_experts, n_primitives)
        # Each row is like the logits in LayerCompositionWeights
        self.fc1_logits = nn.Parameter(torch.zeros(n_experts, n_primitives))
        self.fc2_logits = nn.Parameter(torch.zeros(n_experts, n_primitives))

        # Initialize with small random values to break symmetry
        nn.init.normal_(self.fc1_logits, mean=0, std=0.01)
        nn.init.normal_(self.fc2_logits, mean=0, std=0.01)

    def get_expert_weights(self, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get composition weights for a specific expert."""
        fc1_weights = F.softmax(self.fc1_logits[expert_idx] / self.temperature, dim=-1)
        fc2_weights = F.softmax(self.fc2_logits[expert_idx] / self.temperature, dim=-1)
        return fc1_weights, fc2_weights

    def get_fc1_logits(self) -> torch.Tensor:
        """Get raw fc1 expert logits for top-k selection/cache updates."""
        return self.fc1_logits

    def get_fc2_logits(self) -> torch.Tensor:
        """Get raw fc2 expert logits for top-k selection/cache updates."""
        return self.fc2_logits

    def get_all_expert_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get composition weights for all experts. Shape: (n_experts, n_primitives)"""
        fc1_weights = F.softmax(self.fc1_logits / self.temperature, dim=-1)
        fc2_weights = F.softmax(self.fc2_logits / self.temperature, dim=-1)
        return fc1_weights, fc2_weights

    def compute_expert_entropy(self) -> Dict[str, torch.Tensor]:
        """Compute entropy of each expert's composition weights."""
        fc1_weights, fc2_weights = self.get_all_expert_weights()

        fc1_entropy = -(fc1_weights * torch.log(fc1_weights + 1e-8)).sum(dim=-1)
        fc2_entropy = -(fc2_weights * torch.log(fc2_weights + 1e-8)).sum(dim=-1)

        return {
            "fc1_entropy": fc1_entropy,  # (n_experts,)
            "fc2_entropy": fc2_entropy,
            "fc1_mean_entropy": fc1_entropy.mean(),
            "fc2_mean_entropy": fc2_entropy.mean(),
        }

    def compute_expert_similarity(self) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between experts.
        Shape: (n_experts, n_experts)

        Low similarity = experts have specialized (good)
        High similarity = experts converged to similar compositions (bad)
        """
        fc1_weights, _ = self.get_all_expert_weights()
        # Normalize for cosine similarity
        fc1_norm = F.normalize(fc1_weights, p=2, dim=-1)
        similarity = fc1_norm @ fc1_norm.T
        return similarity

    def get_similarity_stats(self) -> Dict[str, float]:
        """Get summary statistics for expert similarity."""
        similarity = self.compute_expert_similarity()

        # Get upper triangle (excluding diagonal)
        n = similarity.shape[0]
        mask = torch.triu(torch.ones(n, n, device=similarity.device), diagonal=1).bool()
        pairwise = similarity[mask]

        return {
            "mean_similarity": pairwise.mean().item(),
            "min_similarity": pairwise.min().item(),
            "max_similarity": pairwise.max().item(),
        }


def compute_bank_entropy(bank: PrimitiveBank, all_weights: List[torch.Tensor]) -> float:
    """
    Compute how uniformly primitives are used across all layers.

    High entropy = primitives used evenly (good)
    Low entropy = few primitives dominate (collapse, bad)
    """
    # Aggregate usage across all layers
    usage = torch.zeros(bank.n_primitives)
    for weights in all_weights:
        usage = usage + weights.detach().cpu()

    # Normalize to probability
    usage = usage / (usage.sum() + 1e-8)

    # Compute entropy
    log_usage = torch.log(usage + 1e-8)
    entropy = -(usage * log_usage).sum().item()

    return entropy


if __name__ == "__main__":
    # Test the primitive bank
    print("Testing PrimitiveBank...")

    d_model = 512
    d_ff = 2048
    n_primitives = 32
    rank = 32
    top_k = 8

    bands = [
        {"name": "early", "layers": [0, 1, 2]},
        {"name": "middle", "layers": [3, 4, 5]},
        {"name": "late", "layers": [6, 7]},
    ]

    # Create banks
    banks = BandPrimitiveBanks(
        d_model=d_model,
        d_ff=d_ff,
        n_primitives=n_primitives,
        rank=rank,
        bands=bands,
        share_fc1_fc2=False
    )

    # Create composition weights for each layer
    layer_weights = nn.ModuleList([
        LayerCompositionWeights(n_primitives, top_k, layer_idx=i)
        for i in range(8)
    ])

    # Test forward pass
    x = torch.randn(2, 128, d_model)  # batch=2, seq=128

    for layer_idx in range(8):
        fc1_bank = banks.get_fc1_bank(layer_idx)
        fc2_bank = banks.get_fc2_bank(layer_idx)
        weights = layer_weights[layer_idx]

        # fc1: d_model -> d_ff
        h = fc1_bank(x, weights.get_fc1_weights(), top_k=top_k)
        print(f"Layer {layer_idx} fc1 output shape: {h.shape}")

        # Apply activation
        h = F.gelu(h)

        # fc2: d_ff -> d_model
        out = fc2_bank(h, weights.get_fc2_weights(), top_k=top_k)
        print(f"Layer {layer_idx} fc2 output shape: {out.shape}")

        # Check entropy
        entropy = weights.compute_entropy()
        print(f"Layer {layer_idx} entropy: fc1={entropy['fc1_entropy']:.2f}, fc2={entropy['fc2_entropy']:.2f}")

    # Parameter count
    print(f"\nParameter counts: {banks.parameter_count()}")
