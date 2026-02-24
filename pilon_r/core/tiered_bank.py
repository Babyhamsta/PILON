"""
PILON-R Tiered Primitive Bank (Phase B.5b)

VRAM-efficient primitive loading: only n_hot primitives live in VRAM with
gradients/optimizer states. The rest live in CPU pinned memory.

Duck-types PrimitiveBank (same interface for forward methods).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class TieredPrimitiveBank(nn.Module):
    """
    Tiered primitive bank with hot (VRAM) and warm (CPU) tiers.

    Hot primitives receive gradients and optimizer states.
    Warm primitives are stored in CPU pinned memory (no gradients).

    Provides the same forward interface as PrimitiveBank so existing
    CompositionalFFN code works unchanged.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_primitives: int,
        rank: int,
        n_hot: int,
        swap_interval: int = 100,
        name: str = "tiered_bank",
    ):
        super().__init__()
        assert n_hot > 0, "n_hot must be >= 1"
        self.d_in = d_in
        self.d_out = d_out
        self.n_primitives = n_primitives
        self.rank = rank
        self.n_hot = min(n_hot, n_primitives)
        self.n_warm = n_primitives - self.n_hot
        self.swap_interval = swap_interval
        self.name = name
        self._step_counter = 0

        # --- Hot tier (VRAM, receives gradients) ---
        std_b = 1.0 / math.sqrt(rank)
        # Initialize all primitives first, then split
        flat_cols = n_primitives * rank
        A_flat = torch.randn(d_in, flat_cols)
        if d_in >= flat_cols:
            A_ortho, _ = torch.linalg.qr(A_flat, mode="reduced")
        else:
            A_ortho, _ = torch.linalg.qr(A_flat.T, mode="reduced")
            A_ortho = A_ortho.T
        A_all = A_ortho.reshape(d_in, n_primitives, rank).permute(1, 0, 2).contiguous()
        B_all = torch.randn(n_primitives, rank, d_out) * std_b

        # Hot primitives (first n_hot) - on GPU with gradients
        self.A_hot = nn.Parameter(A_all[: self.n_hot].clone())
        self.B_hot = nn.Parameter(B_all[: self.n_hot].clone())

        # Warm primitives (remaining) - CPU pinned, no gradients
        if self.n_warm > 0:
            A_warm = A_all[self.n_hot :].clone()
            B_warm = B_all[self.n_hot :].clone()
            # Try pin_memory; fall back to regular CPU tensors on Windows if needed
            try:
                A_warm = A_warm.pin_memory()
                B_warm = B_warm.pin_memory()
            except RuntimeError:
                pass  # pin_memory not supported, use regular CPU tensors
            self.register_buffer("A_warm", A_warm, persistent=True)
            self.register_buffer("B_warm", B_warm, persistent=True)
        else:
            self.register_buffer("A_warm", torch.empty(0, d_in, rank), persistent=True)
            self.register_buffer("B_warm", torch.empty(0, rank, d_out), persistent=True)

        # Latent affine transform (always on GPU, shared across all primitives)
        self.latent_scale = nn.Parameter(torch.ones(rank))
        self.latent_bias = nn.Parameter(torch.zeros(rank))

        # hot_indices: maps hot slot i -> global primitive index
        self.register_buffer(
            "hot_indices", torch.arange(self.n_hot, dtype=torch.long), persistent=True
        )
        # warm_indices: maps warm slot i -> global primitive index
        # (explicit tracking to avoid corruption after multiple swaps)
        self.register_buffer(
            "warm_indices",
            torch.arange(self.n_hot, self.n_primitives, dtype=torch.long),
            persistent=True,
        )

        # Usage counts: track selection frequency per global primitive
        self.register_buffer(
            "usage_counts", torch.zeros(n_primitives), persistent=True
        )

    # --- Properties to match PrimitiveBank interface ---

    @property
    def A(self) -> nn.Parameter:
        """Return hot A parameters (used by existing forward methods)."""
        return self.A_hot

    @property
    def B(self) -> nn.Parameter:
        """Return hot B parameters (used by existing forward methods)."""
        return self.B_hot

    def _remap_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Remap global primitive weights to hot-only weights.

        Args:
            weights: Composition weights over all n_primitives

        Returns:
            Weights over n_hot primitives (re-normalized)
        """
        # Accumulate usage counts (all primitives, based on softmax weights)
        if self.training:
            self.usage_counts.add_(weights.detach())

        # Extract weights for hot primitives only
        hot_weights = weights[self.hot_indices]
        # Re-normalize so hot weights sum to 1
        hot_sum = hot_weights.sum()
        if hot_sum > 1e-8:
            hot_weights = hot_weights / hot_sum
        return hot_weights

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_k: Optional[int] = None,
        active_rank: Optional[int] = None,
        active_primitives: Optional[int] = None,
    ) -> torch.Tensor:
        """Weighted combination of hot primitives (vectorized)."""
        hot_weights = self._remap_weights(weights)

        # Support rank scheduling
        A = self.A_hot[:, :, :active_rank] if active_rank is not None else self.A_hot
        B = self.B_hot[:, :active_rank, :] if active_rank is not None else self.B_hot
        if active_rank is not None:
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
        else:
            scale = self.latent_scale
            bias = self.latent_bias

        n_hot = A.size(0)
        effective_top_k = top_k
        if effective_top_k is not None:
            effective_top_k = min(effective_top_k, n_hot)

        if effective_top_k is not None and effective_top_k < n_hot:
            top_w, top_idx = torch.topk(hot_weights, effective_top_k, sorted=False)
            top_w = top_w / (top_w.sum() + 1e-8)
            A = A.index_select(0, top_idx)
            B = B.index_select(0, top_idx)
            hot_weights = top_w

        batch, seq, _ = x.shape
        x_flat = x.reshape(-1, self.d_in)

        U = torch.einsum("td,pdr->tpr", x_flat, A)
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U.mul_(scale)
        U.add_(bias)
        out_flat = torch.einsum("tpr,pro,p->to", U, B, hot_weights)

        return out_flat.view(batch, seq, self.d_out)

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
        """Sparse fused forward over hot tier."""
        hot_weights = self._remap_weights(weights)

        A = self.A_hot[:, :, :active_rank] if active_rank is not None else self.A_hot
        B = self.B_hot[:, :active_rank, :] if active_rank is not None else self.B_hot
        if active_rank is not None:
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
            rank = active_rank
        else:
            scale = self.latent_scale
            bias = self.latent_bias
            rank = self.rank

        if top_indices is not None:
            top_idx = self._global_to_hot_indices(top_indices)
            if top_idx is None:
                # Some indices aren't hot; fall back to topk on hot weights
                actual_k = min(top_k, self.n_hot)
                top_w, top_idx = torch.topk(hot_weights, actual_k, sorted=False)
            else:
                top_w = hot_weights.index_select(0, top_idx)
        else:
            actual_k = min(top_k, self.n_hot)
            top_w, top_idx = torch.topk(hot_weights, actual_k, sorted=False)

        # Use actual number of selected indices (may differ from top_k)
        actual_k = top_idx.numel()

        top_w = top_w.float()
        top_w = top_w / (top_w.sum() + 1e-8)

        if preselected is not None:
            A_sel, B_sel = preselected
            actual_k = int(A_sel.size(0))
            if top_w.numel() != actual_k:
                top_w = top_w[:actual_k]
                top_w = top_w / (top_w.sum() + 1e-8)
        else:
            A_sel = A.index_select(0, top_idx)
            B_sel = B.index_select(0, top_idx)

        sqrt_w = torch.sqrt(top_w + 1e-8).to(dtype=A_sel.dtype)
        A_sel = A_sel * sqrt_w[:, None, None]
        B_sel = B_sel * sqrt_w[:, None, None]

        A_cat = A_sel.permute(1, 0, 2).contiguous().view(self.d_in, actual_k * rank)
        B_cat = B_sel.contiguous().view(actual_k * rank, self.d_out)

        batch, seq, _ = x.shape
        x_flat = x.reshape(-1, self.d_in)

        U = x_flat @ A_cat
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U = U.view(-1, actual_k, rank)
        U.mul_(scale)
        U.add_(bias)
        U = U.view(-1, actual_k * rank)

        out_flat = U @ B_cat
        return out_flat.view(batch, seq, self.d_out)

    def select_topk_primitives(
        self,
        top_indices: torch.Tensor,
        active_rank: Optional[int] = None,
        active_primitives: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather selected hot-tier primitive factors for reuse across micro-batches.
        """
        del active_primitives  # Tiered bank only materializes hot primitives on device.
        A = self.A_hot[:, :, :active_rank] if active_rank is not None else self.A_hot
        B = self.B_hot[:, :active_rank, :] if active_rank is not None else self.B_hot
        hot_idx = self._global_to_hot_indices(top_indices)
        if hot_idx is None or hot_idx.numel() == 0:
            raise ValueError("Top-k selection contains warm primitives; cannot cache fused hot selection.")
        return A.index_select(0, hot_idx), B.index_select(0, hot_idx)

    def forward_sparse(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_k: int,
        active_rank: Optional[int] = None,
        top_indices: Optional[torch.Tensor] = None,
        active_primitives: Optional[int] = None,
    ) -> torch.Tensor:
        """Truly sparse forward over hot tier."""
        hot_weights = self._remap_weights(weights)
        effective_top_k = min(top_k, self.n_hot)

        A = self.A_hot[:, :, :active_rank] if active_rank is not None else self.A_hot
        B = self.B_hot[:, :active_rank, :] if active_rank is not None else self.B_hot
        if active_rank is not None:
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
        else:
            scale = self.latent_scale
            bias = self.latent_bias

        if top_indices is not None:
            top_idx = self._global_to_hot_indices(top_indices)
            if top_idx is None:
                top_w, top_idx = torch.topk(
                    hot_weights, effective_top_k, sorted=False
                )
            else:
                top_w = hot_weights.index_select(0, top_idx)
                effective_top_k = top_idx.numel()
        else:
            top_w, top_idx = torch.topk(hot_weights, effective_top_k, sorted=False)

        top_w = top_w / (top_w.sum() + 1e-8)

        A_sel = A.index_select(0, top_idx)
        B_sel = B.index_select(0, top_idx)

        batch, seq, _ = x.shape
        x_flat = x.reshape(-1, self.d_in)

        U = torch.einsum("td,kdr->tkr", x_flat, A_sel)
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U.mul_(scale)
        U.add_(bias)

        Y = torch.einsum("tkr,kro->tko", U, B_sel)
        out_flat = torch.einsum("tko,k->to", Y, top_w.to(dtype=Y.dtype))
        return out_flat.view(batch, seq, self.d_out)

    def forward_fast(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_k: Optional[int] = None,
        active_rank: Optional[int] = None,
        active_primitives: Optional[int] = None,
    ) -> torch.Tensor:
        """Fast forward: delegates to sparse for top-k, else compute all hot outputs."""
        hot_weights = self._remap_weights(weights)
        n_hot = hot_weights.numel()

        if top_k is not None and top_k < n_hot:
            # Use _forward_sparse_with_hot_weights to avoid double _remap_weights call
            return self._forward_sparse_with_hot_weights(
                x, hot_weights, top_k=top_k, active_rank=active_rank
            )

        all_outputs = self.compute_all_outputs(x, active_rank=active_rank)
        return torch.einsum("bspd,p->bsd", all_outputs, hot_weights)

    def _forward_sparse_with_hot_weights(
        self,
        x: torch.Tensor,
        hot_weights: torch.Tensor,
        top_k: int,
        active_rank: Optional[int] = None,
    ) -> torch.Tensor:
        """Sparse forward using pre-remapped hot weights (avoids double remap)."""
        effective_top_k = min(top_k, self.n_hot)

        A = self.A_hot[:, :, :active_rank] if active_rank is not None else self.A_hot
        B = self.B_hot[:, :active_rank, :] if active_rank is not None else self.B_hot
        if active_rank is not None:
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
        else:
            scale = self.latent_scale
            bias = self.latent_bias

        top_w, top_idx = torch.topk(hot_weights, effective_top_k, sorted=False)
        top_w = top_w / (top_w.sum() + 1e-8)

        A_sel = A.index_select(0, top_idx)
        B_sel = B.index_select(0, top_idx)

        batch, seq, _ = x.shape
        x_flat = x.reshape(-1, self.d_in)

        U = torch.einsum("td,kdr->tkr", x_flat, A_sel)
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U.mul_(scale)
        U.add_(bias)

        Y = torch.einsum("tkr,kro->tko", U, B_sel)
        out_flat = torch.einsum("tko,k->to", Y, top_w.to(dtype=Y.dtype))
        return out_flat.view(batch, seq, self.d_out)

    def compute_all_outputs(
        self,
        x: torch.Tensor,
        active_rank: Optional[int] = None,
        active_primitives: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute outputs for all hot primitives."""
        batch, seq, _ = x.shape
        x_flat = x.reshape(-1, self.d_in)

        A = self.A_hot[:, :, :active_rank] if active_rank is not None else self.A_hot
        B = self.B_hot[:, :active_rank, :] if active_rank is not None else self.B_hot
        if active_rank is not None:
            scale = self.latent_scale[:active_rank]
            bias = self.latent_bias[:active_rank]
        else:
            scale = self.latent_scale
            bias = self.latent_bias

        U = torch.einsum("td,pdr->tpr", x_flat, A)
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U.mul_(scale)
        U.add_(bias)
        Y = torch.einsum("tpr,pro->tpo", U, B)

        return Y.view(batch, seq, self.n_hot, self.d_out)

    def _global_to_hot_indices(
        self, global_indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Convert global primitive indices to hot-local indices. Returns None if any are warm."""
        # Vectorized: (n_indices, 1) == (1, n_hot) -> (n_indices, n_hot)
        matches = (global_indices.unsqueeze(1) == self.hot_indices.unsqueeze(0))
        if not matches.any(dim=1).all():
            return None  # Some global indices are not in hot tier
        return matches.float().argmax(dim=1)

    def maybe_swap(self, optimizer) -> bool:
        """
        Swap hot/warm primitives based on usage frequency.

        Called every step; only performs swap at swap_interval boundaries.
        Transfers optimizer Adam m/v states to maintain training continuity.

        Args:
            optimizer: The optimizer (AdamW) to transfer states for

        Returns:
            True if a swap occurred, False otherwise
        """
        self._step_counter += 1
        if self._step_counter % self.swap_interval != 0:
            return False
        if self.n_warm == 0:
            return False

        warm_global = self.warm_indices.clone()
        hot_global = self.hot_indices.clone()

        warm_usage = self.usage_counts[warm_global]
        hot_usage = self.usage_counts[hot_global]

        # Sort: find warm primitives with highest usage
        warm_sorted = warm_usage.argsort(descending=True)
        # Sort: find hot primitives with lowest usage
        hot_sorted = hot_usage.argsort(descending=False)

        # Determine how many to swap (swap those where warm usage > hot usage)
        n_swap = 0
        for i in range(min(len(warm_sorted), len(hot_sorted))):
            warm_idx = warm_sorted[i]
            hot_idx = hot_sorted[i]
            if warm_usage[warm_idx] > hot_usage[hot_idx]:
                n_swap += 1
            else:
                break

        if n_swap == 0:
            self.usage_counts.zero_()
            return False

        device = self.A_hot.device

        # Perform swaps
        for i in range(n_swap):
            warm_local = warm_sorted[i].item()
            hot_local = hot_sorted[i].item()

            warm_gi = warm_global[warm_local].item()
            hot_gi = hot_global[hot_local].item()

            # Save hot data to CPU
            A_hot_data = self.A_hot.data[hot_local].cpu().clone()
            B_hot_data = self.B_hot.data[hot_local].cpu().clone()

            # Promote warm -> hot (CPU -> GPU)
            self.A_hot.data[hot_local] = self.A_warm[warm_local].to(device)
            self.B_hot.data[hot_local] = self.B_warm[warm_local].to(device)

            # Demote hot -> warm (GPU -> CPU)
            self.A_warm[warm_local] = A_hot_data
            self.B_warm[warm_local] = B_hot_data

            # Update BOTH index mappings
            self.hot_indices[hot_local] = warm_gi
            self.warm_indices[warm_local] = hot_gi

            # Transfer optimizer states (Adam m/v)
            self._transfer_optimizer_states(optimizer, hot_local, warm_gi, hot_gi)

        # Reset usage counts
        self.usage_counts.zero_()
        return True

    def _transfer_optimizer_states(
        self, optimizer, hot_slot: int, new_global: int, old_global: int
    ) -> None:
        """Transfer Adam m/v states for swapped primitives."""
        if optimizer is None:
            return

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param is self.A_hot or param is self.B_hot:
                    state = optimizer.state.get(param)
                    if state is None:
                        continue
                    # Zero out the states for the swapped slot
                    # (warm primitive has no prior optimizer state)
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key][hot_slot].zero_()

    @classmethod
    def from_primitive_bank(
        cls,
        bank,  # PrimitiveBank
        n_hot: int,
        swap_interval: int = 100,
    ) -> "TieredPrimitiveBank":
        """
        Create a TieredPrimitiveBank from an existing PrimitiveBank.

        Copies the weights from the PrimitiveBank into hot/warm tiers.
        """
        tiered = cls.__new__(cls)
        nn.Module.__init__(tiered)

        tiered.d_in = bank.d_in
        tiered.d_out = bank.d_out
        tiered.n_primitives = bank.n_primitives
        tiered.rank = bank.rank
        tiered.n_hot = min(n_hot, bank.n_primitives)
        tiered.n_warm = bank.n_primitives - tiered.n_hot
        tiered.swap_interval = swap_interval
        tiered.name = f"tiered_{bank.name}"
        tiered._step_counter = 0

        # Copy hot primitives (first n_hot)
        tiered.A_hot = nn.Parameter(bank.A.data[: tiered.n_hot].clone())
        tiered.B_hot = nn.Parameter(bank.B.data[: tiered.n_hot].clone())

        # Copy warm primitives
        if tiered.n_warm > 0:
            A_warm = bank.A.data[tiered.n_hot :].cpu().clone()
            B_warm = bank.B.data[tiered.n_hot :].cpu().clone()
            try:
                A_warm = A_warm.pin_memory()
                B_warm = B_warm.pin_memory()
            except RuntimeError:
                pass
            tiered.register_buffer("A_warm", A_warm, persistent=True)
            tiered.register_buffer("B_warm", B_warm, persistent=True)
        else:
            tiered.register_buffer(
                "A_warm", torch.empty(0, bank.d_in, bank.rank), persistent=True
            )
            tiered.register_buffer(
                "B_warm", torch.empty(0, bank.rank, bank.d_out), persistent=True
            )

        # Copy latent affine
        tiered.latent_scale = nn.Parameter(bank.latent_scale.data.clone())
        tiered.latent_bias = nn.Parameter(bank.latent_bias.data.clone())

        # Initialize indices
        tiered.register_buffer(
            "hot_indices",
            torch.arange(tiered.n_hot, dtype=torch.long),
            persistent=True,
        )
        tiered.register_buffer(
            "warm_indices",
            torch.arange(tiered.n_hot, bank.n_primitives, dtype=torch.long),
            persistent=True,
        )
        tiered.register_buffer(
            "usage_counts",
            torch.zeros(bank.n_primitives),
            persistent=True,
        )

        return tiered

    def parameter_count(self) -> int:
        """Total parameters (hot + warm) in this bank."""
        return self.n_primitives * (self.d_in * self.rank + self.rank * self.d_out) + 2 * self.rank
