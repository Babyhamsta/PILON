"""
PILON-R Feed-Forward Network Implementations

Contains:
- StandardFFN: Dense baseline for comparison
- CompositionalFFN: Uses primitive banks (Phase A - static composition)
- MoECompositionalFFN: MoE routing over primitive compositions (Phase B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, Tuple

from .primitives import BandPrimitiveBanks, LayerCompositionWeights, ExpertCompositionBank


class StandardFFN(nn.Module):
    """
    Standard dense FFN for baseline comparison.

    FFN(x) = fc2(activation(fc1(x)))

    fc1: d_model -> d_ff
    fc2: d_ff -> d_model
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with standard scheme."""
        nn.init.kaiming_uniform_(self.fc1.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.fc2.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Output tensor (batch, seq, d_model)
        """
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return h

    def parameter_count(self) -> int:
        """Count parameters."""
        return sum(p.numel() for p in self.parameters())


class CompositionalFFN(nn.Module):
    """
    Compositional FFN using primitive banks.

    Instead of dense fc1/fc2, this uses:
    - A shared bank of low-rank primitives (per band)
    - Per-layer composition weights to combine primitives
    - Sparse top_k selection for efficiency

    This is THE EXPERIMENT for Phase A.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        layer_idx: int,
        primitive_banks: BandPrimitiveBanks,
        n_primitives: int,
        top_k: int,
        top_k_fc1: Optional[int] = None,
        top_k_fc2: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        temperature: float = 1.0,
        forward_fast_mode: str = "auto",
        forward_fast_min_topk: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.top_k_fc1 = top_k_fc1 if top_k_fc1 is not None else top_k
        self.top_k_fc2 = top_k_fc2 if top_k_fc2 is not None else top_k
        self.n_primitives = n_primitives
        self.forward_fast_mode = forward_fast_mode
        self.forward_fast_min_topk = forward_fast_min_topk

        # Reference to shared banks (not owned by this module)
        self.primitive_banks = primitive_banks

        # Per-layer composition weights (owned by this module)
        self.composition_weights = LayerCompositionWeights(
            n_primitives=n_primitives,
            top_k=top_k,
            layer_idx=layer_idx,
            temperature=temperature
        )

        self.dropout = nn.Dropout(dropout)

        # Runtime overrides (set by training loop for phase training)
        self.runtime_top_k: Optional[int] = None  # Overrides self.top_k when set
        self.runtime_top_k_fc1: Optional[int] = None  # Overrides per-projection
        self.runtime_top_k_fc2: Optional[int] = None
        self.active_rank: Optional[int] = None    # For rank scheduling
        self.active_primitives: Optional[int] = None  # For primitive scheduling
        self.runtime_uniform_topk: bool = False  # Uniform weights for top-k (phase1)
        self.runtime_step: Optional[int] = None   # Set by training loop
        self.topk_cache_steps: Optional[int] = None  # Update cached top-k every N steps
        self.use_fused_topk: bool = True
        self._fc1_topk_cache: Optional[Dict[str, torch.Tensor]] = None
        self._fc2_topk_cache: Optional[Dict[str, torch.Tensor]] = None
        self.cache_selected_primitives: bool = False
        self._fc1_selected_cache: Optional[Dict[str, Any]] = None
        self._fc2_selected_cache: Optional[Dict[str, Any]] = None

        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def update_topk_cache(self) -> None:
        """
        Explicitly update top-k cache based on current weights.
        Called by training loop before forward pass to ensure deterministic
        behavior for checkpointing.
        """
        if self.topk_cache_steps is None or self.runtime_step is None:
            return

        # Check if update is needed based on step count
        # Update if step % cache_steps == 0
        should_update = (self.runtime_step % self.topk_cache_steps == 0)
        
        # Or if cache is empty
        if self._fc1_topk_cache is None or self._fc1_topk_cache.get("step") is None:
            should_update = True
            if self._fc1_topk_cache is None:
                self._fc1_topk_cache = {}
            if self._fc2_topk_cache is None:
                self._fc2_topk_cache = {}

        if should_update:
            active_prims = self.active_primitives or self.n_primitives
            active_prims = min(active_prims, self.n_primitives)

            # Update FC1
            fc1_logits = self.composition_weights.get_fc1_logits()
            k_fc1 = self.runtime_top_k_fc1
            if k_fc1 is None:
                k_fc1 = self.runtime_top_k if self.runtime_top_k is not None else self.top_k_fc1
            if k_fc1 is None:
                k_fc1 = self.top_k_fc1
            k_fc1 = min(k_fc1, active_prims)
            if k_fc1 < active_prims:
                _, indices = torch.topk(fc1_logits[:active_prims], k_fc1, sorted=False)
                self._fc1_topk_cache["indices"] = indices
                self._fc1_topk_cache["step"] = self.runtime_step
                self._fc1_topk_cache["top_k"] = k_fc1
                self._fc1_topk_cache["active_primitives"] = active_prims

            # Update FC2
            fc2_logits = self.composition_weights.get_fc2_logits()
            k_fc2 = self.runtime_top_k_fc2
            if k_fc2 is None:
                k_fc2 = self.runtime_top_k if self.runtime_top_k is not None else self.top_k_fc2
            if k_fc2 is None:
                k_fc2 = self.top_k_fc2
            k_fc2 = min(k_fc2, active_prims)
            if k_fc2 < active_prims:
                _, indices = torch.topk(fc2_logits[:active_prims], k_fc2, sorted=False)
                self._fc2_topk_cache["indices"] = indices
                self._fc2_topk_cache["step"] = self.runtime_step
                self._fc2_topk_cache["top_k"] = k_fc2
                self._fc2_topk_cache["active_primitives"] = active_prims

        if not self.cache_selected_primitives:
            self._fc1_selected_cache = None
            self._fc2_selected_cache = None
            return

        # Build selected primitive-factor cache every step (enables reuse across
        # gradient accumulation micro-batches and checkpoint recomputation).
        active_prims = self.active_primitives or self.n_primitives
        active_prims = min(active_prims, self.n_primitives)
        active_rank = self.active_rank
        fc1_bank = self.primitive_banks.get_fc1_bank(self.layer_idx)
        fc2_bank = self.primitive_banks.get_fc2_bank(self.layer_idx)

        k_fc1 = self.runtime_top_k_fc1
        if k_fc1 is None:
            k_fc1 = self.runtime_top_k if self.runtime_top_k is not None else self.top_k_fc1
        if k_fc1 is None:
            k_fc1 = self.top_k_fc1
        k_fc1 = int(min(k_fc1, active_prims))
        if k_fc1 < active_prims and hasattr(fc1_bank, "select_topk_primitives"):
            fc1_scores = self.composition_weights.get_fc1_logits()
            fc1_indices = self._get_cached_indices(
                fc1_scores, k_fc1, self._fc1_topk_cache, active_prims
            )
            try:
                with torch.no_grad():
                    A_sel, B_sel = fc1_bank.select_topk_primitives(
                        fc1_indices,
                        active_rank=active_rank,
                        active_primitives=active_prims,
                    )
                self._fc1_selected_cache = {
                    "step": self.runtime_step,
                    "top_k": k_fc1,
                    "active_primitives": active_prims,
                    "active_rank": active_rank,
                    "A_sel": A_sel.detach(),
                    "B_sel": B_sel.detach(),
                }
            except Exception:
                self._fc1_selected_cache = None
        else:
            self._fc1_selected_cache = None

        k_fc2 = self.runtime_top_k_fc2
        if k_fc2 is None:
            k_fc2 = self.runtime_top_k if self.runtime_top_k is not None else self.top_k_fc2
        if k_fc2 is None:
            k_fc2 = self.top_k_fc2
        k_fc2 = int(min(k_fc2, active_prims))
        if k_fc2 < active_prims and hasattr(fc2_bank, "select_topk_primitives"):
            fc2_scores = self.composition_weights.get_fc2_logits()
            fc2_indices = self._get_cached_indices(
                fc2_scores, k_fc2, self._fc2_topk_cache, active_prims
            )
            try:
                with torch.no_grad():
                    A_sel, B_sel = fc2_bank.select_topk_primitives(
                        fc2_indices,
                        active_rank=active_rank,
                        active_primitives=active_prims,
                    )
                self._fc2_selected_cache = {
                    "step": self.runtime_step,
                    "top_k": k_fc2,
                    "active_primitives": active_prims,
                    "active_rank": active_rank,
                    "A_sel": A_sel.detach(),
                    "B_sel": B_sel.detach(),
                }
            except Exception:
                self._fc2_selected_cache = None
        else:
            self._fc2_selected_cache = None

    def _get_cached_indices(
        self,
        scores: torch.Tensor,
        top_k: int,
        cache: Optional[Dict[str, torch.Tensor]],
        active_primitives: Optional[int]
    ) -> torch.Tensor:
        """
        Get indices from cache if valid, otherwise compute transiently.
        Does NOT update cache (no side effects).
        """
        # Try to use cache
        if cache is not None:
            indices = cache.get("indices")
            cache_k = cache.get("top_k")
            cache_active = cache.get("active_primitives")
            # Only use if k matches
            if indices is not None and cache_k == top_k and cache_active == active_primitives:
                return indices

        # Fallback: compute but don't save
        if active_primitives is not None and active_primitives < scores.numel():
            scores = scores[:active_primitives]
        _, indices = torch.topk(scores, top_k, sorted=False)
        return indices

    def _get_cached_selected(
        self,
        cache: Optional[Dict[str, Any]],
        top_k: int,
        active_primitives: int,
        active_rank: Optional[int],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if cache is None:
            return None
        if cache.get("step") != self.runtime_step:
            return None
        if cache.get("top_k") != top_k:
            return None
        if cache.get("active_primitives") != active_primitives:
            return None
        if cache.get("active_rank") != active_rank:
            return None
        A_sel = cache.get("A_sel")
        B_sel = cache.get("B_sel")
        if A_sel is None or B_sel is None:
            return None
        return A_sel, B_sel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using compositional primitives.

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Output tensor (batch, seq, d_model)
        """
        # Get banks for this layer
        fc1_bank = self.primitive_banks.get_fc1_bank(self.layer_idx)
        fc2_bank = self.primitive_banks.get_fc2_bank(self.layer_idx)

        # Get composition weights
        fc1_weights = self.composition_weights.get_fc1_weights()
        fc2_weights = self.composition_weights.get_fc2_weights()

        # Use runtime overrides if set, else config values
        effective_top_k_fc1 = self.runtime_top_k_fc1
        if effective_top_k_fc1 is None:
            effective_top_k_fc1 = self.runtime_top_k if self.runtime_top_k is not None else self.top_k_fc1
        effective_top_k_fc2 = self.runtime_top_k_fc2
        if effective_top_k_fc2 is None:
            effective_top_k_fc2 = self.runtime_top_k if self.runtime_top_k is not None else self.top_k_fc2
        active_rank = getattr(self, "active_rank", None)
        active_primitives = getattr(self, "active_primitives", None)
        effective_n_primitives = self.n_primitives
        if active_primitives is not None and active_primitives < self.n_primitives:
            effective_n_primitives = max(1, active_primitives)
        if effective_top_k_fc1 is not None:
            effective_top_k_fc1 = min(effective_top_k_fc1, effective_n_primitives)
        if effective_top_k_fc2 is not None:
            effective_top_k_fc2 = min(effective_top_k_fc2, effective_n_primitives)

        def _use_fast_path(top_k: Optional[int], effective_n_primitives: int, force_uniform: bool) -> bool:
            mode = self.forward_fast_mode
            if mode == "off":
                return False
            if mode == "on":
                return True
            if mode != "auto":
                raise ValueError(f"Unknown forward_fast_mode: {mode}. Use 'auto', 'on', or 'off'.")
            if top_k is None:
                return True
            if force_uniform:
                return False
            min_k = self.forward_fast_min_topk
            if min_k is None:
                min_k = effective_n_primitives
            return top_k >= min_k

        # fc1: d_model -> d_ff (expansion)
        force_uniform_fc1 = (
            self.runtime_uniform_topk
            and effective_top_k_fc1 is not None
            and effective_top_k_fc1 < effective_n_primitives
        )
        if (
            (self.use_fused_topk or force_uniform_fc1)
            and effective_top_k_fc1 is not None
            and effective_top_k_fc1 < effective_n_primitives
        ):
            fc1_preselected = self._get_cached_selected(
                self._fc1_selected_cache,
                effective_top_k_fc1,
                effective_n_primitives,
                active_rank,
            )
            if force_uniform_fc1:
                fc1_scores = self.composition_weights.get_fc1_logits()
                fc1_indices = self._get_cached_indices(
                    fc1_scores, effective_top_k_fc1, self._fc1_topk_cache, effective_n_primitives
                )
                fc1_weights = torch.full_like(fc1_scores, 1.0 / self.n_primitives)
            else:
                fc1_indices = self._get_cached_indices(
                    fc1_weights, effective_top_k_fc1, self._fc1_topk_cache, effective_n_primitives
                )
            h = fc1_bank.forward_topk_fused(
                x,
                fc1_weights,
                top_k=effective_top_k_fc1,
                active_rank=active_rank,
                top_indices=fc1_indices,
                active_primitives=active_primitives,
                preselected=fc1_preselected,
            )
        elif _use_fast_path(effective_top_k_fc1, effective_n_primitives, force_uniform_fc1):
            h = fc1_bank.forward_fast(
                x,
                fc1_weights,
                top_k=effective_top_k_fc1,
                active_rank=active_rank,
                active_primitives=active_primitives
            )
        else:
            h = fc1_bank.forward(
                x,
                fc1_weights,
                top_k=effective_top_k_fc1,
                active_rank=active_rank,
                active_primitives=active_primitives
            )

        # Activation
        h = self.activation(h)
        h = self.dropout(h)

        # fc2: d_ff -> d_model (compression)
        force_uniform_fc2 = (
            self.runtime_uniform_topk
            and effective_top_k_fc2 is not None
            and effective_top_k_fc2 < effective_n_primitives
        )
        if (
            (self.use_fused_topk or force_uniform_fc2)
            and effective_top_k_fc2 is not None
            and effective_top_k_fc2 < effective_n_primitives
        ):
            fc2_preselected = self._get_cached_selected(
                self._fc2_selected_cache,
                effective_top_k_fc2,
                effective_n_primitives,
                active_rank,
            )
            if force_uniform_fc2:
                fc2_scores = self.composition_weights.get_fc2_logits()
                fc2_indices = self._get_cached_indices(
                    fc2_scores, effective_top_k_fc2, self._fc2_topk_cache, effective_n_primitives
                )
                fc2_weights = torch.full_like(fc2_scores, 1.0 / self.n_primitives)
            else:
                fc2_indices = self._get_cached_indices(
                    fc2_weights, effective_top_k_fc2, self._fc2_topk_cache, effective_n_primitives
                )
            out = fc2_bank.forward_topk_fused(
                h,
                fc2_weights,
                top_k=effective_top_k_fc2,
                active_rank=active_rank,
                top_indices=fc2_indices,
                active_primitives=active_primitives,
                preselected=fc2_preselected,
            )
        elif _use_fast_path(effective_top_k_fc2, effective_n_primitives, force_uniform_fc2):
            out = fc2_bank.forward_fast(
                h,
                fc2_weights,
                top_k=effective_top_k_fc2,
                active_rank=active_rank,
                active_primitives=active_primitives
            )
        else:
            out = fc2_bank.forward(
                h,
                fc2_weights,
                top_k=effective_top_k_fc2,
                active_rank=active_rank,
                active_primitives=active_primitives
            )

        return out

    def get_entropy(self) -> Dict[str, float]:
        """Get composition weight entropy (for monitoring)."""
        return self.composition_weights.compute_entropy()

    def get_top_k_usage(self) -> Dict[str, torch.Tensor]:
        """Get which primitives are in top_k (for monitoring)."""
        return self.composition_weights.get_top_k_indices()

    def get_composition_weights(self) -> Dict[str, torch.Tensor]:
        """Get raw composition weights (for analysis)."""
        return {
            "fc1_weights": self.composition_weights.get_fc1_weights(),
            "fc2_weights": self.composition_weights.get_fc2_weights(),
        }


class Router(nn.Module):
    """
    Router for MoE - selects experts per token.

    Supports:
    - Linear router (simple, recommended for Phase B)
    - MLP router (more capacity, optional)
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        router_type: str = "linear",
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.router_type = router_type

        if router_type == "linear":
            self.router = nn.Linear(d_model, n_experts, bias=False)
        elif router_type == "mlp":
            hidden = hidden_dim or d_model // 4
            self.router = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_experts)
            )
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

        self._init_weights()

    def _init_weights(self):
        if self.router_type == "linear":
            nn.init.xavier_uniform_(self.router.weight)
        else:
            for module in self.router:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute router logits.

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Router logits (batch, seq, n_experts)
        """
        return self.router(x)


def compute_load_balancing_loss(
    router_probs: torch.Tensor,
    expert_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute load balancing auxiliary loss (from Switch Transformer).

    Loss = n_experts * sum_i(f_i * P_i)
    where:
    - f_i = fraction of tokens routed to expert i
    - P_i = average router probability for expert i

    This encourages uniform distribution of tokens across experts.

    Args:
        router_probs: Router probabilities (batch, seq, n_experts)
        expert_mask: Binary mask of selected experts (batch, seq, n_experts)

    Returns:
        Scalar auxiliary loss
    """
    n_experts = router_probs.shape[-1]

    # Flatten batch and seq dimensions
    router_probs_flat = router_probs.view(-1, n_experts)
    expert_mask_flat = expert_mask.view(-1, n_experts)

    # f_i: fraction of tokens routed to each expert
    tokens_per_expert = expert_mask_flat.sum(dim=0)
    total_tokens = tokens_per_expert.sum()
    f = tokens_per_expert / (total_tokens + 1e-8)

    # P_i: average probability assigned to each expert
    P = router_probs_flat.mean(dim=0)

    # Load balancing loss
    aux_loss = n_experts * (f * P).sum()

    return aux_loss


class MoECompositionalFFN(nn.Module):
    """
    Mixture-of-Experts Compositional FFN for Phase B.

    Key innovation: Each "expert" is a composition recipe over shared primitives,
    not a separate FFN. This maintains parameter efficiency while adding
    token-dependent routing.

    Architecture:
    1. Router decides which 2 experts (compositions) to use per token
    2. Each expert applies its composition weights to the shared primitive bank
    3. Outputs are weighted-summed based on router probabilities
    4. Load balancing loss encourages even expert usage
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        layer_idx: int,
        primitive_banks: BandPrimitiveBanks,
        n_primitives: int,
        top_k_primitives: int,
        n_experts: int = 8,
        top_k_experts: int = 2,
        router_type: str = "linear",
        router_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        temperature: float = 1.0,
        load_balancing: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layer_idx = layer_idx
        self.n_experts = n_experts
        self.top_k_experts = top_k_experts
        self.top_k_primitives = top_k_primitives
        self.n_primitives = n_primitives
        self.load_balancing = load_balancing

        # Shared primitive banks (reference, not owned)
        self.primitive_banks = primitive_banks

        # Router
        self.router = Router(d_model, n_experts, router_type, router_hidden_dim)

        # Expert composition bank (n_experts sets of composition weights)
        self.expert_compositions = ExpertCompositionBank(
            n_experts=n_experts,
            n_primitives=n_primitives,
            top_k_primitives=top_k_primitives,
            layer_idx=layer_idx,
            temperature=temperature
        )

        self.dropout = nn.Dropout(dropout)

        # Activation
        self.activation = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
        }[activation]

        # Cache for monitoring (set during forward)
        self._last_router_logits = None
        self._last_router_probs = None
        # Runtime override for expert top-k (default: config top_k_experts).
        self.runtime_top_k: Optional[int] = top_k_experts
        self.runtime_step: Optional[int] = None
        self.topk_cache_steps: Optional[int] = None
        self._fc1_expert_topk_cache: Optional[Dict[str, torch.Tensor]] = None
        self._fc2_expert_topk_cache: Optional[Dict[str, torch.Tensor]] = None
        self._fc1_dense_fused_cache: Optional[Dict[str, Any]] = None
        self._fc2_dense_fused_cache: Optional[Dict[str, Any]] = None

    def update_topk_cache(self) -> None:
        """
        Cache per-expert primitive top-k indices for a short step window.
        This avoids repeated top-k scans on composition weights.
        """
        if self.topk_cache_steps is None or self.runtime_step is None:
            return

        should_update = (self.runtime_step % self.topk_cache_steps == 0)
        if self._fc1_expert_topk_cache is None or self._fc1_expert_topk_cache.get("step") is None:
            should_update = True
            if self._fc1_expert_topk_cache is None:
                self._fc1_expert_topk_cache = {}
            if self._fc2_expert_topk_cache is None:
                self._fc2_expert_topk_cache = {}

        if not should_update:
            return

        primitive_top_k = max(1, min(self.top_k_primitives, self.n_primitives))
        if primitive_top_k < self.n_primitives:
            with torch.no_grad():
                _, fc1_idx = torch.topk(
                    self.expert_compositions.get_fc1_logits(),
                    primitive_top_k,
                    dim=-1,
                    sorted=False,
                )
                _, fc2_idx = torch.topk(
                    self.expert_compositions.get_fc2_logits(),
                    primitive_top_k,
                    dim=-1,
                    sorted=False,
                )
        else:
            all_idx = torch.arange(self.n_primitives, device=self.expert_compositions.fc1_logits.device)
            fc1_idx = all_idx.unsqueeze(0).expand(self.n_experts, -1)
            fc2_idx = all_idx.unsqueeze(0).expand(self.n_experts, -1)

        self._fc1_expert_topk_cache["indices"] = fc1_idx
        self._fc1_expert_topk_cache["top_k"] = primitive_top_k
        self._fc1_expert_topk_cache["step"] = self.runtime_step
        self._fc2_expert_topk_cache["indices"] = fc2_idx
        self._fc2_expert_topk_cache["top_k"] = primitive_top_k
        self._fc2_expert_topk_cache["step"] = self.runtime_step

    def _get_cached_expert_indices(
        self,
        scores: torch.Tensor,
        top_k: int,
        cache: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if cache is not None:
            indices = cache.get("indices")
            cache_k = cache.get("top_k")
            if indices is not None and cache_k == top_k:
                return indices
        _, indices = torch.topk(scores, top_k, dim=-1, sorted=False)
        return indices

    def _bank_forward_experts_fused(
        self,
        x: torch.Tensor,
        bank: Any,
        expert_weights: Optional[torch.Tensor],
        expert_indices: torch.Tensor,
        top_k: int,
        expert_top_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batched fused primitive-bank forward across all experts.

        Args:
            x: Expert inputs of shape (n_experts, n_tokens, d_in)
            bank: PrimitiveBank module (fc1 or fc2)
            expert_weights: (n_experts, n_primitives), optional if expert_top_weights provided
            expert_indices: Top-k primitive indices per expert (n_experts, top_k)
            top_k: Number of primitives per expert to use
            expert_top_weights: Optional pre-gathered top-k primitive weights (n_experts, top_k)

        Returns:
            Tensor of shape (n_experts, n_tokens, d_out)
        """
        n_experts, n_tokens, _ = x.shape
        rank = bank.rank

        # Gather selected primitives for all experts in one pass
        flat_idx = expert_indices.reshape(-1)
        A_sel = bank.A.index_select(0, flat_idx).view(n_experts, top_k, bank.d_in, rank)
        B_sel = bank.B.index_select(0, flat_idx).view(n_experts, top_k, rank, bank.d_out)

        # Convert per-expert top-k mixture into a single low-rank map
        if expert_top_weights is not None:
            top_weights = expert_top_weights
        else:
            if expert_weights is None:
                raise ValueError("expert_weights must be provided when expert_top_weights is None")
            top_weights = expert_weights.gather(1, expert_indices)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        sqrt_w = torch.sqrt(top_weights + 1e-8).to(dtype=A_sel.dtype)
        A_sel = A_sel * sqrt_w[:, :, None, None]
        B_sel = B_sel * sqrt_w[:, :, None, None]

        A_cat = A_sel.permute(0, 2, 1, 3).contiguous().view(n_experts, bank.d_in, top_k * rank)
        B_cat = B_sel.contiguous().view(n_experts, top_k * rank, bank.d_out)

        return self._apply_fused_maps(
            x=x,
            bank=bank,
            A_cat=A_cat,
            B_cat=B_cat,
            top_k=top_k,
        )

    def _prepare_tiered_expert_topk(
        self,
        bank: Any,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Map global primitive selections to hot-local indices for tiered banks.

        If any selected primitive is warm for an expert, fall back to that
        expert's hot-only top-k selection, matching TieredPrimitiveBank behavior.
        """
        if not hasattr(bank, "hot_indices"):
            raise ValueError("Tiered top-k preparation requires a tiered bank.")

        k_eff = int(max(1, min(int(top_k), int(bank.n_hot))))
        hot_indices = bank.hot_indices
        if hot_indices.device != expert_weights.device:
            hot_indices = hot_indices.to(expert_weights.device)

        hot_weights = expert_weights.index_select(1, hot_indices)
        hot_weights = hot_weights / (hot_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Preserve tiered-bank usage accounting used for hot/warm swapping.
        if bank.training and hasattr(bank, "usage_counts"):
            with torch.no_grad():
                usage_add = expert_weights.detach().sum(dim=0)
                if usage_add.device != bank.usage_counts.device:
                    usage_add = usage_add.to(bank.usage_counts.device)
                bank.usage_counts.add_(usage_add.to(dtype=bank.usage_counts.dtype))

        matches = (expert_indices.unsqueeze(-1) == hot_indices.view(1, 1, -1))
        valid_rows = matches.any(dim=-1).all(dim=-1)
        local_idx_full = matches.float().argmax(dim=-1)

        n_experts = expert_indices.size(0)
        local_idx = torch.empty(
            n_experts, k_eff, dtype=torch.long, device=expert_indices.device
        )
        local_w = torch.empty(
            n_experts, k_eff, dtype=hot_weights.dtype, device=hot_weights.device
        )

        valid_ids = torch.nonzero(valid_rows, as_tuple=False).squeeze(-1)
        if valid_ids.numel() > 0:
            idx_valid = local_idx_full.index_select(0, valid_ids)
            if idx_valid.size(1) > k_eff:
                idx_valid = idx_valid[:, :k_eff]
            w_valid = hot_weights.index_select(0, valid_ids).gather(1, idx_valid)
            local_idx.index_copy_(0, valid_ids, idx_valid)
            local_w.index_copy_(0, valid_ids, w_valid)

        invalid_ids = torch.nonzero(~valid_rows, as_tuple=False).squeeze(-1)
        if invalid_ids.numel() > 0:
            hot_invalid = hot_weights.index_select(0, invalid_ids)
            w_fallback, idx_fallback = torch.topk(hot_invalid, k_eff, dim=-1, sorted=False)
            local_idx.index_copy_(0, invalid_ids, idx_fallback)
            local_w.index_copy_(0, invalid_ids, w_fallback)

        local_w = local_w / (local_w.sum(dim=-1, keepdim=True) + 1e-8)
        return local_idx, local_w, k_eff

    def _apply_fused_maps(
        self,
        x: torch.Tensor,
        bank: Any,
        A_cat: torch.Tensor,
        B_cat: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Apply pre-fused low-rank maps for a set of experts."""
        n_experts, n_tokens, _ = x.shape
        rank = bank.rank

        U = torch.bmm(x, A_cat)  # (E, T, top_k * rank)
        U = U.view(n_experts, n_tokens, top_k, rank)

        scale = bank.latent_scale
        bias = bank.latent_bias
        if scale.dtype != U.dtype:
            scale = scale.to(dtype=U.dtype)
            bias = bias.to(dtype=U.dtype)
        U.mul_(scale)
        U.add_(bias)
        U = U.view(n_experts, n_tokens, top_k * rank)
        return torch.bmm(U, B_cat)  # (E, T, d_out)

    def _fused_cache_bucket(self) -> int:
        if self.topk_cache_steps is not None and self.runtime_step is not None and self.topk_cache_steps > 0:
            return int(self.runtime_step // self.topk_cache_steps)
        return -1

    def _get_dense_fused_maps(
        self,
        bank: Any,
        expert_indices: torch.Tensor,
        expert_top_weights: torch.Tensor,
        top_k: int,
        cache_attr: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build (or reuse) fused expert maps for dense expert execution.

        Cache is inference-only to avoid stale training graphs.
        """
        cache = getattr(self, cache_attr)
        can_cache = not self.training
        bucket = self._fused_cache_bucket()
        device = expert_indices.device
        dtype = bank.A.dtype

        if can_cache and isinstance(cache, dict):
            if (
                cache.get("top_k") == top_k
                and cache.get("bucket") == bucket
                and cache.get("device") == str(device)
                and cache.get("dtype") == str(dtype)
                and cache.get("A_cat") is not None
                and cache.get("B_cat") is not None
            ):
                return cache["A_cat"], cache["B_cat"]

        n_experts = expert_indices.size(0)
        rank = bank.rank
        flat_idx = expert_indices.reshape(-1)
        A_sel = bank.A.index_select(0, flat_idx).view(n_experts, top_k, bank.d_in, rank)
        B_sel = bank.B.index_select(0, flat_idx).view(n_experts, top_k, rank, bank.d_out)

        top_weights = expert_top_weights / (expert_top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        sqrt_w = torch.sqrt(top_weights + 1e-8).to(dtype=A_sel.dtype)
        A_sel = A_sel * sqrt_w[:, :, None, None]
        B_sel = B_sel * sqrt_w[:, :, None, None]

        A_cat = A_sel.permute(0, 2, 1, 3).contiguous().view(n_experts, bank.d_in, top_k * rank)
        B_cat = B_sel.contiguous().view(n_experts, top_k * rank, bank.d_out)

        if can_cache:
            new_cache: Dict[str, Any] = {
                "top_k": top_k,
                "bucket": bucket,
                "device": str(device),
                "dtype": str(dtype),
                "A_cat": A_cat,
                "B_cat": B_cat,
            }
            setattr(self, cache_attr, new_cache)

        return A_cat, B_cat

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MoE routing.

        Efficient implementation:
        1. Route tokens to experts
        2. Run expert FFNs through fused top-k primitive composition
        3. Aggregate with router probabilities

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Dictionary with:
            - output: Output tensor (batch, seq, d_model)
            - aux_loss: Load balancing loss (scalar)
            - router_logits: Router logits for monitoring (batch, seq, n_experts)
        """
        batch_size, seq_len, _ = x.shape

        # Get banks for this layer
        fc1_bank = self.primitive_banks.get_fc1_bank(self.layer_idx)
        fc2_bank = self.primitive_banks.get_fc2_bank(self.layer_idx)

        # Router forward
        router_logits = self.router(x)
        self._last_router_logits = router_logits.detach()
        router_probs = F.softmax(router_logits, dim=-1)

        # Runtime top-k (None falls back to config top_k_experts).
        effective_top_k = self.runtime_top_k
        top_k_probs: Optional[torch.Tensor] = None
        top_k_indices: Optional[torch.Tensor] = None
        if effective_top_k is None:
            effective_top_k = self.top_k_experts
        effective_top_k = int(max(1, min(self.n_experts, int(effective_top_k))))

        if effective_top_k >= self.n_experts:
            effective_top_k = self.n_experts
            effective_router_probs = router_probs
            expert_mask = torch.ones_like(router_probs)
        else:
            # Select top-k experts per token and mask others
            top_k_probs, top_k_indices = torch.topk(
                router_probs, effective_top_k, dim=-1, sorted=False
            )
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
            masked = torch.zeros_like(router_probs)
            masked.scatter_(-1, top_k_indices, top_k_probs)
            effective_router_probs = masked

            expert_mask = torch.zeros_like(router_probs)
            expert_mask.scatter_(-1, top_k_indices, 1.0)

        # Store effective probs for logging (masked when hard routing active)
        self._last_router_probs = effective_router_probs.detach()

        # Get all expert composition weights: (n_experts, n_primitives)
        fc1_expert_weights, fc2_expert_weights = self.expert_compositions.get_all_expert_weights()

        primitive_top_k = max(1, min(self.top_k_primitives, self.n_primitives))

        # Apply top-k primitives per expert (not per token).
        if primitive_top_k < self.n_primitives:
            fc1_idx = self._get_cached_expert_indices(
                fc1_expert_weights, primitive_top_k, self._fc1_expert_topk_cache
            )
            fc2_idx = self._get_cached_expert_indices(
                fc2_expert_weights, primitive_top_k, self._fc2_expert_topk_cache
            )
            fc1_top_weights = fc1_expert_weights.gather(1, fc1_idx)
            fc2_top_weights = fc2_expert_weights.gather(1, fc2_idx)
            fc1_top_weights = fc1_top_weights / (fc1_top_weights.sum(dim=-1, keepdim=True) + 1e-8)
            fc2_top_weights = fc2_top_weights / (fc2_top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            all_idx = torch.arange(self.n_primitives, device=x.device)
            fc1_idx = all_idx.unsqueeze(0).expand(self.n_experts, -1)
            fc2_idx = all_idx.unsqueeze(0).expand(self.n_experts, -1)
            fc1_top_weights = fc1_expert_weights
            fc2_top_weights = fc2_expert_weights

        # Flatten once; used by both dense and sparse expert paths.
        n_tokens = batch_size * seq_len
        x_flat = x.reshape(n_tokens, self.d_model)
        router_flat = effective_router_probs.reshape(n_tokens, self.n_experts)
        use_tiered_bank_fallback = hasattr(fc1_bank, "hot_indices") or hasattr(fc2_bank, "hot_indices")

        if effective_top_k < self.n_experts and top_k_indices is not None and top_k_probs is not None:
            # Token-grouped sparse expert execution:
            # each active expert only processes tokens routed to it.
            output_flat = torch.zeros(n_tokens, self.d_model, device=x.device, dtype=x.dtype)
            assign_expert = top_k_indices.reshape(-1)
            assign_token = (
                torch.arange(n_tokens, device=x.device, dtype=torch.long)
                .unsqueeze(1)
                .expand(-1, effective_top_k)
                .reshape(-1)
            )
            assign_weight = top_k_probs.reshape(-1)

            sort_idx = torch.argsort(assign_expert)
            assign_expert = assign_expert.index_select(0, sort_idx)
            assign_token = assign_token.index_select(0, sort_idx)
            assign_weight = assign_weight.index_select(0, sort_idx)

            counts = torch.bincount(assign_expert, minlength=self.n_experts)
            active_experts = torch.nonzero(counts > 0, as_tuple=False).squeeze(-1)
            if active_experts.numel() > 0:
                counts_active = counts.index_select(0, active_experts)
                max_tokens_per_expert = int(counts_active.max().item())
                n_active = int(active_experts.numel())

                expert_inputs = torch.zeros(
                    n_active,
                    max_tokens_per_expert,
                    self.d_model,
                    device=x.device,
                    dtype=x.dtype,
                )
                token_buf = torch.zeros(
                    n_active,
                    max_tokens_per_expert,
                    device=x.device,
                    dtype=torch.long,
                )
                weight_buf = torch.zeros(
                    n_active,
                    max_tokens_per_expert,
                    device=x.device,
                    dtype=assign_weight.dtype,
                )

                start = 0
                for row_idx, cnt in enumerate(counts_active.tolist()):
                    cnt_i = int(cnt)
                    if cnt_i <= 0:
                        continue
                    end = start + cnt_i
                    tok = assign_token[start:end]
                    expert_inputs[row_idx, :cnt_i] = x_flat.index_select(0, tok)
                    token_buf[row_idx, :cnt_i] = tok
                    weight_buf[row_idx, :cnt_i] = assign_weight[start:end]
                    start = end

                active_fc1_idx = fc1_idx.index_select(0, active_experts)
                active_fc2_idx = fc2_idx.index_select(0, active_experts)

                if use_tiered_bank_fallback:
                    active_fc1_w_global = fc1_expert_weights.index_select(0, active_experts)
                    active_fc2_w_global = fc2_expert_weights.index_select(0, active_experts)
                    active_fc1_idx, active_fc1_w, fc1_top_k = self._prepare_tiered_expert_topk(
                        fc1_bank, active_fc1_w_global, active_fc1_idx, primitive_top_k
                    )
                    active_fc2_idx, active_fc2_w, fc2_top_k = self._prepare_tiered_expert_topk(
                        fc2_bank, active_fc2_w_global, active_fc2_idx, primitive_top_k
                    )
                else:
                    active_fc1_w = fc1_top_weights.index_select(0, active_experts)
                    active_fc2_w = fc2_top_weights.index_select(0, active_experts)
                    fc1_top_k = primitive_top_k
                    fc2_top_k = primitive_top_k

                fc1_e = self._bank_forward_experts_fused(
                    expert_inputs,
                    fc1_bank,
                    expert_weights=None,
                    expert_indices=active_fc1_idx,
                    top_k=fc1_top_k,
                    expert_top_weights=active_fc1_w,
                )
                fc1_e = self.activation(fc1_e)
                fc2_e = self._bank_forward_experts_fused(
                    fc1_e,
                    fc2_bank,
                    expert_weights=None,
                    expert_indices=active_fc2_idx,
                    top_k=fc2_top_k,
                    expert_top_weights=active_fc2_w,
                )

                valid_mask = (
                    torch.arange(max_tokens_per_expert, device=x.device).unsqueeze(0)
                    < counts_active.unsqueeze(1)
                )
                token_idx_flat = token_buf.masked_select(valid_mask)
                contrib = fc2_e * weight_buf.to(dtype=fc2_e.dtype).unsqueeze(-1)
                contrib_flat = contrib[valid_mask]
                output_flat.index_add_(0, token_idx_flat, contrib_flat.to(dtype=output_flat.dtype))
        elif use_tiered_bank_fallback:
            # Tiered banks keep only hot primitives on-device; use bank-native
            # fused forward so global primitive indices can be remapped safely.
            expert_outputs = []
            for expert_idx in range(self.n_experts):
                h_expert = fc1_bank.forward_topk_fused(
                    x,
                    fc1_expert_weights[expert_idx],
                    top_k=primitive_top_k,
                    top_indices=fc1_idx[expert_idx],
                )
                h_expert = self.activation(h_expert)
                out_expert = fc2_bank.forward_topk_fused(
                    h_expert,
                    fc2_expert_weights[expert_idx],
                    top_k=primitive_top_k,
                    top_indices=fc2_idx[expert_idx],
                )
                expert_outputs.append(out_expert.view(n_tokens, self.d_model))

            expert_outs = torch.stack(expert_outputs, dim=0)  # (E, T, d_model)
            router_dense = router_flat.transpose(0, 1).unsqueeze(-1).to(dtype=expert_outs.dtype)
            output_flat = (expert_outs * router_dense).sum(dim=0)
        else:
            # Dense expert execution across all experts.
            x_expert = x_flat.unsqueeze(0).expand(self.n_experts, -1, -1)  # (E, T, d_model)
            fc1_A_cat, fc1_B_cat = self._get_dense_fused_maps(
                fc1_bank,
                fc1_idx,
                fc1_top_weights,
                primitive_top_k,
                cache_attr="_fc1_dense_fused_cache",
            )
            h_all = self._apply_fused_maps(
                x=x_expert,
                bank=fc1_bank,
                A_cat=fc1_A_cat,
                B_cat=fc1_B_cat,
                top_k=primitive_top_k,
            )
            h_all = self.activation(h_all)

            fc2_A_cat, fc2_B_cat = self._get_dense_fused_maps(
                fc2_bank,
                fc2_idx,
                fc2_top_weights,
                primitive_top_k,
                cache_attr="_fc2_dense_fused_cache",
            )
            expert_outs = self._apply_fused_maps(
                x=h_all,
                bank=fc2_bank,
                A_cat=fc2_A_cat,
                B_cat=fc2_B_cat,
                top_k=primitive_top_k,
            )  # (E, T, d_model)

            router_dense = router_flat.transpose(0, 1).unsqueeze(-1).to(dtype=expert_outs.dtype)
            output_flat = (expert_outs * router_dense).sum(dim=0)

        output = output_flat.view(batch_size, seq_len, self.d_model)

        output = self.dropout(output)

        # Compute auxiliary loss
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.load_balancing and self.training:
            aux_loss = compute_load_balancing_loss(router_probs, expert_mask)

        return {
            "output": output,
            "aux_loss": aux_loss,
            "router_logits": router_logits,
        }

    def get_entropy(self) -> Dict[str, float]:
        """Get expert composition weight entropy (for monitoring)."""
        entropy = self.expert_compositions.compute_expert_entropy()
        return {
            "fc1_entropy": entropy["fc1_mean_entropy"].item(),
            "fc2_entropy": entropy["fc2_mean_entropy"].item(),
        }

    def get_expert_similarity(self) -> Dict[str, float]:
        """Get pairwise similarity between experts."""
        return self.expert_compositions.get_similarity_stats()

    def get_router_entropy(self) -> float:
        """Compute router entropy from last forward pass."""
        if self._last_router_probs is not None:
            probs = self._last_router_probs
            avg_probs = probs.mean(dim=(0, 1))
            entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum().item()
            return entropy

        if self._last_router_logits is None:
            return -1.0

        # Average router probs over batch and seq
        probs = F.softmax(self._last_router_logits, dim=-1)
        avg_probs = probs.mean(dim=(0, 1))
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum().item()
        return entropy


def create_ffn(
    ffn_type: str,
    d_model: int,
    d_ff: int,
    layer_idx: int,
    dropout: float = 0.0,
    activation: str = "gelu",
    primitive_banks: Optional[BandPrimitiveBanks] = None,
    n_primitives: int = 32,
    top_k: int = 8,
    top_k_fc1: Optional[int] = None,
    top_k_fc2: Optional[int] = None,
    temperature: float = 1.0,
    forward_fast_mode: str = "auto",
    forward_fast_min_topk: Optional[int] = None,
    moe_config: Optional[Any] = None  # MoEConfig from config.py
) -> nn.Module:
    """
    Factory function to create FFN layer.

    Args:
        ffn_type: "standard" or "compositional"
        d_model: Model dimension
        d_ff: FFN hidden dimension
        layer_idx: Index of this layer
        dropout: Dropout rate
        activation: Activation function name
        primitive_banks: Shared primitive banks (required if compositional)
        n_primitives: Number of primitives per bank
        top_k: Number of primitives to use per forward pass
        temperature: Softmax temperature for composition weights
        moe_config: MoE configuration (if provided, uses MoECompositionalFFN)

    Returns:
        FFN module (StandardFFN, CompositionalFFN, or MoECompositionalFFN)
    """
    if ffn_type == "standard":
        return StandardFFN(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
    elif ffn_type == "compositional":
        if primitive_banks is None:
            raise ValueError("primitive_banks required for compositional FFN")

        # Use MoE if config provided
        if moe_config is not None:
            return MoECompositionalFFN(
                d_model=d_model,
                d_ff=d_ff,
                layer_idx=layer_idx,
                primitive_banks=primitive_banks,
                n_primitives=n_primitives,
                top_k_primitives=top_k,
                n_experts=moe_config.n_experts,
                top_k_experts=moe_config.top_k,
                router_type=moe_config.router_type,
                router_hidden_dim=moe_config.router_hidden_dim,
                dropout=dropout,
                activation=activation,
                temperature=temperature,
                load_balancing=moe_config.load_balancing
            )
        else:
            # Phase A: Static composition
            return CompositionalFFN(
                d_model=d_model,
                d_ff=d_ff,
                layer_idx=layer_idx,
                primitive_banks=primitive_banks,
                n_primitives=n_primitives,
                top_k=top_k,
                top_k_fc1=top_k_fc1,
                top_k_fc2=top_k_fc2,
                dropout=dropout,
                activation=activation,
                temperature=temperature,
                forward_fast_mode=forward_fast_mode,
                forward_fast_min_topk=forward_fast_min_topk
            )
    else:
        raise ValueError(f"Unknown ffn_type: {ffn_type}")


if __name__ == "__main__":
    # Test both FFN types
    print("Testing FFN implementations...")

    d_model = 512
    d_ff = 2048
    batch_size = 2
    seq_len = 128

    x = torch.randn(batch_size, seq_len, d_model)

    # Test StandardFFN
    print("\n--- StandardFFN ---")
    standard_ffn = StandardFFN(d_model, d_ff, dropout=0.1)
    out = standard_ffn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {standard_ffn.parameter_count():,}")

    # Test CompositionalFFN
    print("\n--- CompositionalFFN ---")
    from .primitives import BandPrimitiveBanks

    bands = [
        {"name": "early", "layers": [0, 1, 2]},
        {"name": "middle", "layers": [3, 4, 5]},
        {"name": "late", "layers": [6, 7]},
    ]

    primitive_banks = BandPrimitiveBanks(
        d_model=d_model,
        d_ff=d_ff,
        n_primitives=32,
        rank=32,
        bands=bands,
        share_fc1_fc2=False
    )

    comp_ffn = CompositionalFFN(
        d_model=d_model,
        d_ff=d_ff,
        layer_idx=0,  # Early band
        primitive_banks=primitive_banks,
        n_primitives=32,
        top_k=8,
        dropout=0.1
    )

    out = comp_ffn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Check entropy
    entropy = comp_ffn.get_entropy()
    print(f"Entropy: fc1={entropy['fc1_entropy']:.2f}, fc2={entropy['fc2_entropy']:.2f}")

    # Check top-k usage
    top_k_usage = comp_ffn.get_top_k_usage()
    print(f"Top-k fc1: {top_k_usage['fc1_top_k'].tolist()}")
    print(f"Top-k fc2: {top_k_usage['fc2_top_k'].tolist()}")
