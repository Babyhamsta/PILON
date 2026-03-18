"""
Phase C Attention Modules for PILON-R.

Four attention variants plus a factory function:

1. CompositionalMHA        -- MHA with compositional Q/K/V projections via primitive banks
2. GatedLinearRecurrence   -- RWKV-inspired gated linear recurrence (sequential prototype)
3. CompositionalGatedRecurrence -- GatedLinearRecurrence with compositional projections
4. HybridAttention         -- Layer-dependent dispatch (recurrence for early, MHA for late)

The standard MultiHeadAttention lives in model.py and is reused via the factory.

NOTE on torch.compile: The training path uses a parallel linear recurrence via
cumulative sums in log-space (no sequential loops), making it fully compatible with
torch.compile. The sequential loop is only used during incremental generation (T=1)
where compile is not needed.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pilon_r.core.primitives import PrimitiveBank


# ---------------------------------------------------------------------------
# Attention Primitive Banks (analogous to BandPrimitiveBanks for FFN)
# ---------------------------------------------------------------------------

class AttentionPrimitiveBanks(nn.Module):
    """
    Primitive banks for compositional attention Q/K/V projections.

    Organized by bands, mirroring BandPrimitiveBanks. Each band has three
    banks: one each for Q, K, and V projections. Layers in the same band
    share the same banks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_primitives: int,
        rank: int,
        bands: List[Dict],  # [{"name": "early", "layers": [0,1,2]}, ...]
        ternary: bool = False,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_out = n_heads * d_head
        self.n_primitives = n_primitives
        self.rank = rank
        self.bands = bands

        # Layer -> band mapping
        self.layer_to_band: Dict[int, str] = {}
        for band in bands:
            for layer_idx in band["layers"]:
                self.layer_to_band[layer_idx] = band["name"]

        # Three banks per band: Q, K, V (all project d_model -> n_heads * d_head)
        self.q_banks = nn.ModuleDict()
        self.k_banks = nn.ModuleDict()
        self.v_banks = nn.ModuleDict()

        for band in bands:
            name = band["name"]
            for prefix, bank_dict in [("q", self.q_banks),
                                      ("k", self.k_banks),
                                      ("v", self.v_banks)]:
                bank_dict[name] = PrimitiveBank(
                    d_in=d_model,
                    d_out=self.d_out,
                    n_primitives=n_primitives,
                    rank=rank,
                    name=f"{name}_{prefix}",
                    ternary=ternary,
                    activation_bits=activation_bits,
                )

    def get_q_bank(self, layer_idx: int) -> PrimitiveBank:
        return self.q_banks[self.layer_to_band[layer_idx]]

    def get_k_bank(self, layer_idx: int) -> PrimitiveBank:
        return self.k_banks[self.layer_to_band[layer_idx]]

    def get_v_bank(self, layer_idx: int) -> PrimitiveBank:
        return self.v_banks[self.layer_to_band[layer_idx]]

    def get_band_name(self, layer_idx: int) -> str:
        return self.layer_to_band[layer_idx]

    def parameter_count(self) -> Dict[str, int]:
        counts = {}
        for prefix, banks in [("q", self.q_banks), ("k", self.k_banks), ("v", self.v_banks)]:
            for name, bank in banks.items():
                counts[f"{prefix}_{name}"] = bank.parameter_count()
        counts["total"] = sum(counts.values())
        return counts


# ---------------------------------------------------------------------------
# Attention Composition Weights (analogous to LayerCompositionWeights)
# ---------------------------------------------------------------------------

class AttentionCompositionWeights(nn.Module):
    """
    Per-layer composition weights for combining attention primitives.

    Each layer has three sets of learned logits (Q, K, V) that determine
    how to combine primitives from the shared banks.
    """

    def __init__(
        self,
        n_primitives: int,
        top_k: int,
        layer_idx: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.temperature = temperature

        # Learnable logits for Q, K, V
        self.q_logits = nn.Parameter(torch.zeros(n_primitives))
        self.k_logits = nn.Parameter(torch.zeros(n_primitives))
        self.v_logits = nn.Parameter(torch.zeros(n_primitives))

        # Small perturbation to break symmetry
        nn.init.normal_(self.q_logits, mean=0, std=0.01)
        nn.init.normal_(self.k_logits, mean=0, std=0.01)
        nn.init.normal_(self.v_logits, mean=0, std=0.01)

    def get_q_weights(self) -> torch.Tensor:
        return F.softmax(self.q_logits / self.temperature, dim=0)

    def get_k_weights(self) -> torch.Tensor:
        return F.softmax(self.k_logits / self.temperature, dim=0)

    def get_v_weights(self) -> torch.Tensor:
        return F.softmax(self.v_logits / self.temperature, dim=0)

    def compute_entropy(self) -> Dict[str, float]:
        """Compute entropy of composition weights (for monitoring)."""
        def _entropy(logits: torch.Tensor) -> float:
            probs = F.softmax(logits / self.temperature, dim=0)
            log_probs = F.log_softmax(logits / self.temperature, dim=0)
            return -(probs * log_probs).sum().item()

        return {
            "q_entropy": _entropy(self.q_logits),
            "k_entropy": _entropy(self.k_logits),
            "v_entropy": _entropy(self.v_logits),
        }


# ---------------------------------------------------------------------------
# Compositional Multi-Head Attention
# ---------------------------------------------------------------------------

class CompositionalMHA(nn.Module):
    """
    Multi-head attention with compositional Q/K/V projections.

    The attention mechanism is unchanged (softmax(QK^T/sqrt(d)) @ V via SDPA).
    Only the projection weights are shared/composed via primitive banks.
    The output projection remains a standard nn.Linear.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        attn_primitive_banks: Optional[AttentionPrimitiveBanks] = None,
        composition_weights: Optional[AttentionCompositionWeights] = None,
        top_k: int = 4,
        layer_idx: int = 0,
    ):
        super().__init__()
        if attn_primitive_banks is None:
            raise ValueError("CompositionalMHA requires attn_primitive_banks")
        if composition_weights is None:
            raise ValueError("CompositionalMHA requires composition_weights")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.top_k = top_k
        self.layer_idx = layer_idx

        self.attn_banks = attn_primitive_banks
        self.composition_weights = composition_weights

        # Output projection is standard (not compositional)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight)

        self.dropout = nn.Dropout(dropout)

        # Causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

    def _project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input through compositional primitive banks."""
        q_bank = self.attn_banks.get_q_bank(self.layer_idx)
        k_bank = self.attn_banks.get_k_bank(self.layer_idx)
        v_bank = self.attn_banks.get_v_bank(self.layer_idx)

        q_weights = self.composition_weights.get_q_weights()
        k_weights = self.composition_weights.get_k_weights()
        v_weights = self.composition_weights.get_v_weights()

        q = q_bank.forward_topk_fused(x, q_weights, self.top_k)
        k = k_bank.forward_topk_fused(x, k_weights, self.top_k)
        v = v_bank.forward_topk_fused(x, v_weights, self.top_k)

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Compositional Q/K/V projections
        q, k, v = self._project_qkv(x)

        # Reshape for multi-head attention: (B, T, H*D) -> (B, H, T, D)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # KV cache: concatenate with past keys/values
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        total_k = k.size(2)
        q_len = q.size(2)

        # SDPA path (same logic as MultiHeadAttention)
        use_sdpa = hasattr(F, "scaled_dot_product_attention")
        if use_sdpa:
            attn_mask = None
            is_causal = False

            if attention_mask is None and past_len == 0:
                is_causal = True
            else:
                # Build additive causal mask
                q_pos = past_len + torch.arange(q_len, device=x.device)
                k_pos = torch.arange(total_k, device=x.device)
                causal_mask = k_pos[None, :] > q_pos[:, None]  # (q_len, total_k)

                min_val = torch.finfo(q.dtype).min
                attn_mask = causal_mask.to(dtype=q.dtype) * min_val
                attn_mask = attn_mask[None, None, :, :]  # (1, 1, Q, K)

                # Add padding mask if provided
                pad_mask = self._prepare_attention_mask(attention_mask, q_len)
                if pad_mask is not None:
                    if pad_mask.dtype != attn_mask.dtype:
                        pad_mask = pad_mask.to(dtype=attn_mask.dtype)
                    attn_mask = attn_mask + pad_mask

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Manual attention fallback
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Causal mask
            if past_len == 0 and q_len == total_k and q_len <= self.causal_mask.size(0):
                causal = self.causal_mask[:q_len, :q_len]
            else:
                q_pos = past_len + torch.arange(q_len, device=x.device)
                k_pos = torch.arange(total_k, device=x.device)
                causal = k_pos[None, :] > q_pos[:, None]

            attn_scores = attn_scores.masked_fill(causal, float("-inf"))

            pad_mask = self._prepare_attention_mask(attention_mask, q_len)
            if pad_mask is not None:
                if pad_mask.dtype != attn_scores.dtype:
                    pad_mask = pad_mask.to(dtype=attn_scores.dtype)
                attn_scores = attn_scores + pad_mask

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            out = torch.matmul(attn_probs, v)

        # Reshape and output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        if use_cache:
            return out, present
        return out

    @staticmethod
    def _prepare_attention_mask(
        attention_mask: Optional[torch.Tensor], q_len: int
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return attention_mask.expand(-1, -1, q_len, -1)
        if attention_mask.dim() == 3:
            return attention_mask.unsqueeze(1)
        if attention_mask.dim() == 2:
            return attention_mask[:, None, None, :].expand(-1, -1, q_len, -1)
        return attention_mask


# ---------------------------------------------------------------------------
# Gated Linear Recurrence
# ---------------------------------------------------------------------------

class GatedLinearRecurrence(nn.Module):
    """
    Gated linear recurrence for token mixing (RWKV-inspired).

    This is a sequential prototype. The recurrence loop is a plain Python
    for-loop over the time dimension, which is NOT compatible with
    torch.compile graph capture. For phases C2/C3, fall back to eager mode
    or replace with a parallel scan / fused CUDA kernel.

    The recurrent state serves as the "cache" for autoregressive generation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.hidden_dim = n_heads * d_head

        # Standard projections
        self.q_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, d_model, bias=False)

        # Decay gate: d_model -> n_heads (one decay scalar per head)
        # Griffin-style log-space parameterization: log_decay = -softplus(w)
        # This keeps decay in (0, 1) and backward flows through log/exp
        # (additive gradients) instead of multiplicative sigmoid chains.
        self.decay_proj = nn.Linear(d_model, n_heads, bias=True)
        # Initialize bias so softplus(2.0) ≈ 2.13, decay = exp(-2.13) ≈ 0.12
        # (conservative — retains ~12% per step, similar to sigmoid(2.0) ≈ 0.88
        # but in the "amount forgotten" direction rather than "amount retained")
        nn.init.constant_(self.decay_proj.bias, -2.0)  # softplus(-2.0) ≈ 0.13, decay ≈ 0.88

        # Input gate: controls how much of k*v enters the state
        self.gate_proj = nn.Linear(d_model, self.hidden_dim, bias=False)

        # Output gate: controls how much of the recurrent output passes through
        self.output_gate = nn.Linear(d_model, self.hidden_dim, bias=False)

        # RMSNorm on flattened recurrence output to stabilize gradient flow
        # into downstream ternary FFN. Without this, recurrence state
        # accumulation can produce values that cause NaN in STE backward.
        self.recurrence_norm = nn.RMSNorm(self.hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj,
                     self.gate_proj, self.output_gate]:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.decay_proj.weight)

    @torch.compiler.disable
    def _forward_parallel(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel linear recurrence using Griffin-style log-space gating
        with chunked scan for forward numerical stability.

        Griffin log-space gating: decay = exp(-softplus(w))
        - Gradients flow through softplus/exp (additive in log-space)
        - No multiplicative gradient chains

        Chunked scan via reshape: reshape (B, T, ...) into (B*n_chunks, C, ...)
        and use vectorized cumsum. Cross-chunk state carry is sequential but
        only n_chunks iterations (8 for T=512, C=64).

        Reference: Griffin (ICLR 2025)
        """
        batch_size, seq_len, _, _ = q.shape

        # Griffin-style log-space decay: always negative, numerically stable
        log_decay = -F.softplus(self.decay_proj(x)).unsqueeze(-1)  # (B, T, H, 1)

        # Input gate: (B, T, H, D)
        gate = torch.sigmoid(
            self.gate_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        )

        # Gated key-value product: (B, T, H, D)
        kv = gate * k * v

        # --- Chunked parallel scan (vectorized within chunks) ---
        CHUNK_SIZE = 64
        # Pad to multiple of CHUNK_SIZE
        pad = (CHUNK_SIZE - seq_len % CHUNK_SIZE) % CHUNK_SIZE
        if pad > 0:
            log_decay = F.pad(log_decay, (0, 0, 0, 0, 0, pad))  # pad time dim
            kv = F.pad(kv, (0, 0, 0, 0, 0, pad))
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
        padded_len = seq_len + pad
        n_chunks = padded_len // CHUNK_SIZE

        # Reshape into chunks: (B, n_chunks, C, H, D/1)
        log_d = log_decay.float().view(batch_size, n_chunks, CHUNK_SIZE, self.n_heads, 1)
        kv_c = kv.float().view(batch_size, n_chunks, CHUNK_SIZE, self.n_heads, self.d_head)
        q_c = q.view(batch_size, n_chunks, CHUNK_SIZE, self.n_heads, self.d_head)

        # Within-chunk cumulative log-decay: (B, n_chunks, C, H, 1)
        cum_log_d = torch.cumsum(log_d, dim=2)

        # Within-chunk parallel scan using max-shift for numerical stability.
        # Instead of exp(-cum) which can overflow, compute:
        #   new_t = sum_{i=0}^{t} kv_i * exp(cum_t - cum_i)
        # By shifting by cum_t (the last position's cumulative), we get:
        #   kv_shifted_i = kv_i * exp(cum_C - cum_i)  (bounded: max = exp(0) = 1 at i=C)
        #   new_t = exp(cum_t - cum_C) * cumsum(kv_shifted)[t]
        # where cum_C - cum_i is always <= 0 (since cum is monotonically decreasing)
        # and cum_t - cum_C <= 0, so all exponents are <= 0. No overflow possible.
        cum_last = cum_log_d[:, :, -1:, :, :]  # (B, nc, 1, H, 1)
        kv_shifted = kv_c * torch.exp(cum_last - cum_log_d)  # exp <= 0, bounded
        new = torch.exp(cum_log_d - cum_last) * torch.cumsum(kv_shifted, dim=2)

        # Cross-chunk state carry (sequential, but only n_chunks iterations)
        state = torch.zeros(
            batch_size, 1, 1, self.n_heads, self.d_head,
            device=x.device, dtype=torch.float32,
        )
        carries = []
        for c in range(n_chunks):
            carry = state * torch.exp(cum_log_d[:, c:c+1])
            carries.append(carry)
            state = carry[:, :, -1:] + new[:, c:c+1, -1:]

        carries = torch.cat(carries, dim=1)  # (B, nc, C, H, D)
        chunk_states = carries + new  # (B, nc, C, H, D)

        # Query-weighted readout
        out = q_c * chunk_states.to(q_c.dtype)  # (B, nc, C, H, D)

        # Reshape back and trim padding
        out = out.view(batch_size, padded_len, self.n_heads, self.d_head)
        if pad > 0:
            out = out[:, :seq_len]

        final_state = chunk_states[:, -1, -1].to(kv.dtype)  # (B, H, D)
        # Trim padding positions from final state if needed
        if pad > 0:
            # Recompute from the actual last position
            actual_last_chunk = (seq_len - 1) // CHUNK_SIZE
            actual_last_pos = (seq_len - 1) % CHUNK_SIZE
            final_state = chunk_states[:, actual_last_chunk, actual_last_pos].to(kv.dtype)

        return out, final_state

    def _forward_sequential(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential fallback for incremental generation (T is typically 1)."""
        batch_size, seq_len, _, _ = q.shape
        state = initial_state

        outputs = []
        for t in range(seq_len):
            # Griffin-style: decay via exp(-softplus(w))
            decay_t = torch.exp(-F.softplus(self.decay_proj(x[:, t]))).unsqueeze(-1)
            gate_t = torch.sigmoid(
                self.gate_proj(x[:, t]).view(batch_size, self.n_heads, self.d_head)
            )
            kv_t = gate_t * k[:, t] * v[:, t]
            state = decay_t * state + kv_t
            outputs.append(q[:, t] * state)

        out = torch.stack(outputs, dim=1)
        return out, state

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V and reshape to (B, T, H, D)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        if past_kv is not None:
            # Incremental generation: use sequential path (typically T=1)
            out, state = self._forward_sequential(x, q, k, v, past_kv)
        else:
            # Training / prefill: use parallel path (torch.compile friendly)
            out, state = self._forward_parallel(x, q, k, v)

        # Flatten heads: (B, T, H*D)
        out = out.reshape(batch_size, seq_len, self.hidden_dim)

        # Normalize recurrence output to stabilize ternary FFN interaction
        out = self.recurrence_norm(out)

        # Output gate
        out_gate = torch.sigmoid(self.output_gate(x))
        out = out * out_gate

        out = self.dropout(out)
        out = self.out_proj(out)

        if use_cache:
            return out, state
        return out


# ---------------------------------------------------------------------------
# Compositional Gated Linear Recurrence
# ---------------------------------------------------------------------------

class CompositionalGatedRecurrence(nn.Module):
    """
    Gated linear recurrence with compositional Q/K/V/gate projections.

    Same recurrence as GatedLinearRecurrence, but the Q, K, V, and gate
    projections use PrimitiveBank.forward_topk_fused() instead of nn.Linear.
    Decay and output projections remain standard nn.Linear (they are small).
    Uses the same parallel linear recurrence as GatedLinearRecurrence.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        attn_primitive_banks: Optional[AttentionPrimitiveBanks] = None,
        composition_weights: Optional["CompositionalRecurrenceWeights"] = None,
        top_k: int = 4,
        layer_idx: int = 0,
    ):
        super().__init__()
        if attn_primitive_banks is None:
            raise ValueError("CompositionalGatedRecurrence requires attn_primitive_banks")
        if composition_weights is None:
            raise ValueError("CompositionalGatedRecurrence requires composition_weights")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.hidden_dim = n_heads * d_head
        self.top_k = top_k
        self.layer_idx = layer_idx

        self.attn_banks = attn_primitive_banks
        self.composition_weights = composition_weights

        # Decay projection: Griffin-style log-space parameterization
        self.decay_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.xavier_uniform_(self.decay_proj.weight)
        nn.init.constant_(self.decay_proj.bias, -2.0)  # softplus(-2) ≈ 0.13, decay ≈ 0.88

        # Output gate and output projection stay standard
        self.output_gate = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, d_model, bias=False)
        nn.init.xavier_uniform_(self.output_gate.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        self.recurrence_norm = nn.RMSNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Compositional projections via primitive banks
        q_bank = self.attn_banks.get_q_bank(self.layer_idx)
        k_bank = self.attn_banks.get_k_bank(self.layer_idx)
        v_bank = self.attn_banks.get_v_bank(self.layer_idx)

        q_weights = self.composition_weights.get_q_weights()
        k_weights = self.composition_weights.get_k_weights()
        v_weights = self.composition_weights.get_v_weights()

        # Project full sequences through banks: (B, T, H*D)
        q = q_bank.forward_topk_fused(x, q_weights, self.top_k)
        k = k_bank.forward_topk_fused(x, k_weights, self.top_k)
        v = v_bank.forward_topk_fused(x, v_weights, self.top_k)

        # Gate projection through the V bank with separate composition weights
        gate_weights = self.composition_weights.get_gate_weights()
        gate = v_bank.forward_topk_fused(x, gate_weights, self.top_k)

        # Reshape to (B, T, H, D)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)
        gate = gate.view(batch_size, seq_len, self.n_heads, self.d_head)

        if past_kv is not None:
            # Incremental generation: sequential (T is typically 1)
            state = past_kv
            outputs = []
            for t in range(seq_len):
                decay_t = torch.exp(-F.softplus(self.decay_proj(x[:, t]))).unsqueeze(-1)
                gate_t = torch.sigmoid(gate[:, t])
                kv_t = gate_t * k[:, t] * v[:, t]
                state = decay_t * state + kv_t
                outputs.append(q[:, t] * state)
            out = torch.stack(outputs, dim=1)
        else:
            # Training / prefill: Griffin log-space + chunked parallel scan
            log_decay = -F.softplus(self.decay_proj(x)).unsqueeze(-1)  # (B, T, H, 1)
            gate_act = torch.sigmoid(gate)  # (B, T, H, D)
            kv = gate_act * k * v  # (B, T, H, D)

            CHUNK_SIZE = 64
            n_chunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            state = torch.zeros(
                batch_size, self.n_heads, self.d_head,
                device=x.device, dtype=torch.float32,
            )
            all_outputs = []
            for c in range(n_chunks):
                s = c * CHUNK_SIZE
                e = min(s + CHUNK_SIZE, seq_len)
                ld = log_decay[:, s:e].float()
                kvc = kv[:, s:e].float()
                qc = q[:, s:e]
                cum_ld = torch.cumsum(ld, dim=1)
                # Max-shift trick: all exponents <= 0, no overflow
                cum_last = cum_ld[:, -1:, :, :]
                kv_shifted = kvc * torch.exp(cum_last - cum_ld)
                new = torch.exp(cum_ld - cum_last) * torch.cumsum(kv_shifted, dim=1)
                carry = state.unsqueeze(1) * torch.exp(cum_ld)
                chunk_states = carry + new
                all_outputs.append(qc * chunk_states.to(qc.dtype))
                state = chunk_states[:, -1]
            out = torch.cat(all_outputs, dim=1)

        out = out.reshape(batch_size, seq_len, self.hidden_dim)
        out = self.recurrence_norm(out)

        out_gate = torch.sigmoid(self.output_gate(x))
        out = out * out_gate
        out = self.dropout(out)
        out = self.out_proj(out)

        if use_cache:
            return out, state
        return out


class CompositionalRecurrenceWeights(nn.Module):
    """
    Per-layer composition weights for CompositionalGatedRecurrence.

    Four sets of logits: Q, K, V, and gate.
    """

    def __init__(
        self,
        n_primitives: int,
        top_k: int,
        layer_idx: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.temperature = temperature

        self.q_logits = nn.Parameter(torch.zeros(n_primitives))
        self.k_logits = nn.Parameter(torch.zeros(n_primitives))
        self.v_logits = nn.Parameter(torch.zeros(n_primitives))
        self.gate_logits = nn.Parameter(torch.zeros(n_primitives))

        for logits in [self.q_logits, self.k_logits, self.v_logits, self.gate_logits]:
            nn.init.normal_(logits, mean=0, std=0.01)

    def get_q_weights(self) -> torch.Tensor:
        return F.softmax(self.q_logits / self.temperature, dim=0)

    def get_k_weights(self) -> torch.Tensor:
        return F.softmax(self.k_logits / self.temperature, dim=0)

    def get_v_weights(self) -> torch.Tensor:
        return F.softmax(self.v_logits / self.temperature, dim=0)

    def get_gate_weights(self) -> torch.Tensor:
        return F.softmax(self.gate_logits / self.temperature, dim=0)

    def compute_entropy(self) -> Dict[str, float]:
        def _entropy(logits: torch.Tensor) -> float:
            probs = F.softmax(logits / self.temperature, dim=0)
            log_probs = F.log_softmax(logits / self.temperature, dim=0)
            return -(probs * log_probs).sum().item()

        return {
            "q_entropy": _entropy(self.q_logits),
            "k_entropy": _entropy(self.k_logits),
            "v_entropy": _entropy(self.v_logits),
            "gate_entropy": _entropy(self.gate_logits),
        }


# ---------------------------------------------------------------------------
# Hybrid Attention
# ---------------------------------------------------------------------------

class HybridAttention(nn.Module):
    """
    Layer-dependent attention dispatch.

    Selects the attention type at __init__ time based on layer_idx and band
    configuration. No control flow in forward -- the inner module is fixed.

    Default policy:
    - Early/middle bands: GatedLinearRecurrence (cheap, linear complexity)
    - Late band: MultiHeadAttention (full quadratic attention for final layers)

    Override via hybrid_config["band_attention_types"] mapping.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        layer_idx: int = 0,
        n_layers: int = 8,
        bands: Optional[List[Dict]] = None,
        hybrid_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Determine which band this layer belongs to
        band_name = self._resolve_band(layer_idx, n_layers, bands)

        # Determine attention type for this band
        config = hybrid_config or {}
        band_types = config.get("band_attention_types", {})
        default_policy = config.get("default_policy", "recurrence_early_mha_late")

        if band_name in band_types:
            attn_type = band_types[band_name]
        elif default_policy == "recurrence_early_mha_late":
            # Last band gets MHA, everything else gets recurrence
            attn_type = "mha" if band_name == "late" else "recurrence"
        else:
            attn_type = "mha"

        # Instantiate the inner module
        if attn_type == "recurrence":
            self.inner = GatedLinearRecurrence(
                d_model=d_model,
                n_heads=n_heads,
                d_head=d_head,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
        else:
            from pilon_r.core.model import MultiHeadAttention
            self.inner = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                d_head=d_head,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )

    @staticmethod
    def _resolve_band(
        layer_idx: int,
        n_layers: int,
        bands: Optional[List[Dict]],
    ) -> str:
        """Determine the band name for a layer."""
        if bands is not None:
            for band in bands:
                if layer_idx in band["layers"]:
                    return band["name"]
        # Fallback: simple thirds heuristic
        third = n_layers / 3.0
        if layer_idx < third:
            return "early"
        elif layer_idx < 2 * third:
            return "middle"
        else:
            return "late"

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv=None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, object]]:
        return self.inner(x, attention_mask=attention_mask, past_kv=past_kv, use_cache=use_cache)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_attention(
    attention_type: str,
    d_model: int,
    n_heads: int,
    d_head: int,
    dropout: float = 0.0,
    max_seq_len: int = 512,
    # Compositional attention params (None for standard)
    attn_primitive_banks: Optional[AttentionPrimitiveBanks] = None,
    n_attn_primitives: int = 16,
    attn_top_k: int = 4,
    attn_temperature: float = 1.0,
    layer_idx: int = 0,
    n_layers: int = 8,
    # Hybrid params
    hybrid_config: Optional[Dict] = None,
    # Band config (for hybrid and compositional)
    bands: Optional[List[Dict]] = None,
) -> nn.Module:
    """
    Factory function for attention modules.

    Args:
        attention_type: One of:
            - "standard_mha": Standard MultiHeadAttention from model.py
            - "compositional_mha": CompositionalMHA with primitive banks
            - "gated_recurrence": GatedLinearRecurrence
            - "compositional_recurrence": CompositionalGatedRecurrence
            - "hybrid": HybridAttention (layer-dependent dispatch)
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_head: Dimension per head.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length (for causal mask buffer).
        attn_primitive_banks: Shared AttentionPrimitiveBanks (required for compositional types).
        n_attn_primitives: Number of attention primitives (for creating composition weights).
        attn_top_k: Top-k primitives to select.
        attn_temperature: Temperature for composition weight softmax.
        layer_idx: Index of this layer.
        n_layers: Total number of layers.
        hybrid_config: Configuration dict for HybridAttention.
        bands: Band configuration list.

    Returns:
        An nn.Module with the standard attention forward signature.
    """
    if attention_type == "standard_mha":
        from pilon_r.core.model import MultiHeadAttention
        return MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    elif attention_type == "compositional_mha":
        if attn_primitive_banks is None:
            raise ValueError(
                "compositional_mha requires attn_primitive_banks. "
                "Create an AttentionPrimitiveBanks instance and pass it to all layers."
            )
        comp_weights = AttentionCompositionWeights(
            n_primitives=n_attn_primitives,
            top_k=attn_top_k,
            layer_idx=layer_idx,
            temperature=attn_temperature,
        )
        return CompositionalMHA(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            max_seq_len=max_seq_len,
            attn_primitive_banks=attn_primitive_banks,
            composition_weights=comp_weights,
            top_k=attn_top_k,
            layer_idx=layer_idx,
        )

    elif attention_type == "gated_recurrence":
        return GatedLinearRecurrence(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    elif attention_type == "compositional_recurrence":
        if attn_primitive_banks is None:
            raise ValueError(
                "compositional_recurrence requires attn_primitive_banks."
            )
        comp_weights = CompositionalRecurrenceWeights(
            n_primitives=n_attn_primitives,
            top_k=attn_top_k,
            layer_idx=layer_idx,
            temperature=attn_temperature,
        )
        return CompositionalGatedRecurrence(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            max_seq_len=max_seq_len,
            attn_primitive_banks=attn_primitive_banks,
            composition_weights=comp_weights,
            top_k=attn_top_k,
            layer_idx=layer_idx,
        )

    elif attention_type == "hybrid":
        return HybridAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
            n_layers=n_layers,
            bands=bands,
            hybrid_config=hybrid_config,
        )

    else:
        raise ValueError(
            f"Unknown attention_type '{attention_type}'. "
            f"Expected one of: standard_mha, compositional_mha, gated_recurrence, "
            f"compositional_recurrence, hybrid."
        )
