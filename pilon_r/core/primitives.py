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
        name: str = "bank"
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_primitives = n_primitives
        self.rank = rank
        self.name = name

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

        return out_flat.view(batch, seq, self.d_out)

    def forward_topk_fused(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_k: int,
        active_rank: Optional[int] = None,
        top_indices: Optional[torch.Tensor] = None,
        active_primitives: Optional[int] = None
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
        top_weights = top_weights.float()
        top_weights = top_weights / (top_weights.sum() + 1e-8)

        # Gather top-k primitives
        A_sel = A.index_select(0, top_indices)  # (k, d_in, r)
        B_sel = B.index_select(0, top_indices)  # (k, r, d_out)

        # Scale each primitive by sqrt(weight) and concatenate into a single low-rank map
        sqrt_w = torch.sqrt(top_weights + 1e-8).to(dtype=A_sel.dtype)
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

        # Get all primitive outputs: (B, S, P, d_out)
        all_outputs = self.compute_all_outputs(x, active_rank=active_rank, active_primitives=active_primitives)

        n_primitives = weights.numel()
        if top_k is not None and top_k < n_primitives:
            # Sparse: get top-k primitives and weighted sum
            top_weights, top_indices = torch.topk(weights, top_k, sorted=False)
            top_weights = top_weights / (top_weights.sum() + 1e-8)
            # Gather selected primitives: (B, S, k, d_out)
            selected = all_outputs.index_select(2, top_indices)
            # Weighted sum: (B, S, d_out)
            return torch.einsum("bskd,k->bsd", selected, top_weights)
        else:
            # Full weighted sum across all primitives
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
        out = torch.zeros(batch, seq, self.d_out, device=x.device, dtype=x.dtype)

        n_primitives = weights.numel()
        if top_k is not None and top_k < n_primitives:
            top_weights, top_indices = torch.topk(weights, top_k, sorted=False)
            top_weights = top_weights / (top_weights.sum() + 1e-8)
        else:
            top_indices = torch.arange(self.n_primitives, device=x.device)
            top_weights = weights

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

        for i, idx in enumerate(top_indices):
            A_i = A[idx]  # [d_in, r]
            B_i = B[idx]  # [r, d_out]
            w_i = top_weights[i]
            latent = x @ A_i
            latent.mul_(scale)
            latent.add_(bias)
            prim_out = latent @ B_i  # [batch, seq, d_out]
            out = out + w_i * prim_out

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
        x_flat = x.reshape(-1, self.d_in)

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

        U = torch.einsum("td,pdr->tpr", x_flat, A)  # [T, P, r]
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U.mul_(scale)
        U.add_(bias)
        Y = torch.einsum("tpr,pro->tpo", U, B)  # [T, P, d_out]

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
        share_fc1_fc2: bool = False  # Should be False for Phase A
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_primitives = n_primitives
        self.rank = rank
        self.bands = bands
        self.share_fc1_fc2 = share_fc1_fc2

        # Create layer -> band mapping
        self.layer_to_band: Dict[int, str] = {}
        for band in bands:
            for layer_idx in band["layers"]:
                self.layer_to_band[layer_idx] = band["name"]

        # Create banks for each band
        # fc1: d_model -> d_ff (expansion)
        # fc2: d_ff -> d_model (compression)
        self.fc1_banks = nn.ModuleDict()
        self.fc2_banks = nn.ModuleDict()

        for band in bands:
            band_name = band["name"]

            # fc1 bank: projects up
            self.fc1_banks[band_name] = PrimitiveBank(
                d_in=d_model,
                d_out=d_ff,
                n_primitives=n_primitives,
                rank=rank,
                name=f"{band_name}_fc1"
            )

            if share_fc1_fc2:
                # Share bank (NOT RECOMMENDED for Phase A)
                # Would need to handle dimension mismatch
                raise NotImplementedError(
                    "Shared fc1/fc2 banks not implemented for Phase A. "
                    "fc1 and fc2 have different dimensions."
                )
            else:
                # fc2 bank: projects down (separate)
                self.fc2_banks[band_name] = PrimitiveBank(
                    d_in=d_ff,
                    d_out=d_model,
                    n_primitives=n_primitives,
                    rank=rank,
                    name=f"{band_name}_fc2"
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
