"""
Legacy Mixture-of-Experts (MoE) components.

This module is kept for backwards compatibility and reference only.
The active PILON MoE implementation is `core.ffn.MoECompositionalFFN`.
"""

import warnings
import torch
import torch.nn as nn
from typing import Optional


class MoERouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]
        logits = self.linear(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


class MoEExpert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoEFFN(nn.Module):
    """
    Simple MoE FFN with soft routing.

    Returns (output, router_probs).
    """

    def __init__(self, d_model: int, d_hidden: int, num_experts: int):
        super().__init__()
        warnings.warn(
            "pilon_r.core.moe.MoEFFN is legacy; use core.ffn.MoECompositionalFFN for active training/inference.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.router = MoERouter(d_model, num_experts)
        self.num_experts = num_experts
        self.runtime_top_k: Optional[int] = None
        self._last_router_probs = None
        self.d_model = d_model
        self.d_hidden = d_hidden

        # Vectorized expert parameters: (E, D, H) and (E, H, D)
        self.fc1_weight = nn.Parameter(torch.empty(num_experts, d_model, d_hidden))
        self.fc1_bias = nn.Parameter(torch.zeros(num_experts, d_hidden))
        self.fc2_weight = nn.Parameter(torch.empty(num_experts, d_hidden, d_model))
        self.fc2_bias = nn.Parameter(torch.zeros(num_experts, d_model))
        self.act = nn.GELU()

        nn.init.xavier_uniform_(self.fc1_weight)
        nn.init.xavier_uniform_(self.fc2_weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        router_probs = self.router(x)  # [B, T, E]

        # Apply runtime top-k masking (None = dense routing)
        effective_top_k = self.runtime_top_k
        if effective_top_k is not None and effective_top_k < self.num_experts:
            top_k_probs, top_k_indices = torch.topk(
                router_probs, effective_top_k, dim=-1, sorted=False
            )
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
            masked = torch.zeros_like(router_probs)
            masked.scatter_(-1, top_k_indices, top_k_probs)
            router_probs = masked

        # Store effective probs for logging (masked when hard routing active)
        self._last_router_probs = router_probs.detach()

        # Vectorized expert compute: [T, D] -> [T, E, D]
        batch, seq, d_model = x.shape
        x_flat = x.reshape(-1, d_model)
        probs_flat = router_probs.reshape(-1, self.num_experts)

        fc1 = torch.einsum("td,edh->teh", x_flat, self.fc1_weight) + self.fc1_bias
        fc1 = self.act(fc1)
        expert_outputs = torch.einsum("teh,ehd->ted", fc1, self.fc2_weight) + self.fc2_bias

        out_flat = torch.einsum("te,ted->td", probs_flat, expert_outputs)
        y = out_flat.view(batch, seq, d_model)

        # Legacy API returns only model output and router probabilities.
        return y, router_probs
