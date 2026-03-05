"""
PILON-R Transformer Model

Full transformer implementation with compositional FFN.
This is the main model for Phase A experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import math
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig, PrimitiveConfig, MoEConfig
from .primitives import BandPrimitiveBanks, LayerCompositionWeights
from .ffn import create_ffn, CompositionalFFN, MoECompositionalFFN
from .early_exit import ExitGate


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * weight / rms(x)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention.

    Using standard attention (not compositional) for Phase A
    to isolate the FFN experiment.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq, d_model)
            attention_mask: Optional attention mask

        Returns:
            Output tensor (batch, seq, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        total_k = k.size(2)
        q_len = q.size(2)

        def _build_causal_mask() -> torch.Tensor:
            # Mask keys that are in the "future" relative to query positions
            q_pos = past_len + torch.arange(q_len, device=x.device)
            k_pos = torch.arange(total_k, device=x.device)
            mask = k_pos[None, :] > q_pos[:, None]
            return mask  # (q_len, total_k) bool

        def _prepare_attention_mask() -> Optional[torch.Tensor]:
            if attention_mask is None:
                return None
            if attention_mask.dim() == 4:
                # (B, 1, 1, K) -> (B, 1, Q, K)
                return attention_mask.expand(-1, -1, q_len, -1)
            if attention_mask.dim() == 3:
                # (B, Q, K) -> (B, 1, Q, K)
                return attention_mask.unsqueeze(1)
            if attention_mask.dim() == 2:
                # (B, K) -> (B, 1, Q, K)
                return attention_mask[:, None, None, :].expand(-1, -1, q_len, -1)
            return attention_mask

        # Fast path: SDPA when available
        use_sdpa = hasattr(F, "scaled_dot_product_attention")
        if use_sdpa:
            attn_mask = None
            is_causal = False

            if attention_mask is None and past_len == 0:
                # Let SDPA handle causality directly
                is_causal = True
            else:
                # Build additive mask: causal + optional padding
                causal_mask = _build_causal_mask()
                min_val = torch.finfo(q.dtype).min
                causal_mask = causal_mask.to(dtype=q.dtype)
                causal_mask = causal_mask * min_val  # 0 for allowed, min_val for masked
                attn_mask = causal_mask[None, None, :, :]  # (1,1,Q,K)

                pad_mask = _prepare_attention_mask()
                if pad_mask is not None:
                    if pad_mask.dtype != attn_mask.dtype:
                        pad_mask = pad_mask.to(dtype=attn_mask.dtype)
                    attn_mask = attn_mask + pad_mask

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Manual attention (fallback)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply causal mask
            if past_len == 0 and q_len == total_k and q_len <= self.causal_mask.size(0):
                causal_mask = self.causal_mask[:q_len, :q_len]
            else:
                causal_mask = _build_causal_mask()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            # Apply optional attention mask
            attn_mask = _prepare_attention_mask()
            if attn_mask is not None:
                if attn_mask.dtype != attn_scores.dtype:
                    attn_mask = attn_mask.to(dtype=attn_scores.dtype)
                attn_scores = attn_scores + attn_mask

            # Softmax and dropout
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Apply attention to values
            out = torch.matmul(attn_probs, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        if use_cache:
            return out, present
        return out


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and FFN.

    Pre-norm architecture:
    x = x + Attention(Norm(x))
    x = x + FFN(Norm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        layer_idx: int,
        ffn_type: str = "compositional",
        dropout: float = 0.1,
        norm_type: str = "rmsnorm",
        checkpoint_ffn: bool = False,
        # Compositional FFN params
        primitive_banks: Optional[BandPrimitiveBanks] = None,
        n_primitives: int = 32,
        top_k: int = 8,
        top_k_fc1: Optional[int] = None,
        top_k_fc2: Optional[int] = None,
        activation: str = "gelu",
        max_seq_len: int = 512,
        temperature: float = 1.0,
        forward_fast_mode: str = "auto",
        forward_fast_min_topk: Optional[int] = None,
        # MoE params (Phase B)
        moe_config: Optional[MoEConfig] = None,
        # Early exit params (Phase B.5c)
        enable_early_exit: bool = False,
        exit_threshold: float = 0.5,
        # Ternary stability
        use_subln: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._last_aux_loss = 0.0  # Store aux_loss from last forward pass
        self.checkpoint_ffn = checkpoint_ffn
        self.enable_early_exit = enable_early_exit
        self.exit_threshold = exit_threshold
        self._last_skip_count = 0
        self._last_total_count = 0
        self._last_exit_confidence: Optional[torch.Tensor] = None

        # Normalization
        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        # Attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

        # FFN (compositional, MoE compositional, or standard)
        self.ffn = create_ffn(
            ffn_type=ffn_type,
            d_model=d_model,
            d_ff=d_ff,
            layer_idx=layer_idx,
            dropout=dropout,
            activation=activation,
            primitive_banks=primitive_banks,
            n_primitives=n_primitives,
            top_k=top_k,
            top_k_fc1=top_k_fc1,
            top_k_fc2=top_k_fc2,
            moe_config=moe_config,
            temperature=temperature,
            forward_fast_mode=forward_fast_mode,
            forward_fast_min_topk=forward_fast_min_topk,
            use_subln=use_subln,
        )

        self.dropout = nn.Dropout(dropout)

        # Early exit gate (Phase B.5c)
        self.exit_gate = ExitGate(d_model) if enable_early_exit else None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq, d_model)
            attention_mask: Optional attention mask

        Returns:
            Output tensor (batch, seq, d_model)
        """
        # Attention with residual
        h = self.norm1(x)
        attn_out = self.attention(h, attention_mask, past_kv=past_kv, use_cache=use_cache)
        if use_cache:
            h, present_kv = attn_out
        else:
            h = attn_out
            present_kv = None
        x = x + self.dropout(h)

        # Early-exit confidence (always computed if gate exists so gates can be trained jointly).
        skip_all = False
        skip_mask = None
        if self.enable_early_exit and self.exit_gate is not None:
            confidence = self.exit_gate(x)  # (B, S, 1)
            self._last_exit_confidence = confidence
            if not self.training:
                skip_mask = (confidence > self.exit_threshold)  # (B, S, 1)
                n_total = skip_mask.numel()
                n_skip = skip_mask.sum().item()
                self._last_skip_count = int(n_skip)
                self._last_total_count = int(n_total)
                if skip_mask.all():
                    skip_all = True
            else:
                self._last_skip_count = 0
                self._last_total_count = 0
        else:
            self._last_skip_count = 0
            self._last_total_count = 0
            self._last_exit_confidence = None

        if skip_all:
            # All tokens skip FFN
            if use_cache:
                return x, present_kv
            return x

        # FFN with residual
        h = self.norm2(x)

        if skip_mask is not None and skip_mask.any() and not skip_mask.all():
            # Mixed skip: only run FFN on non-skip tokens
            B, S, D = x.shape
            flat_mask = (~skip_mask.squeeze(-1)).reshape(-1)  # (B*S,) True = compute
            compute_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)

            h_flat = h.reshape(-1, D)
            h_compute = h_flat.index_select(0, compute_indices)  # (N_compute, D)
            h_compute = h_compute.unsqueeze(0)  # (1, N_compute, D) for FFN

            ffn_result = self.ffn(h_compute)
            if isinstance(ffn_result, dict):
                ffn_out = ffn_result["output"]
                self._last_aux_loss = ffn_result.get("aux_loss", 0.0)
            else:
                ffn_out = ffn_result
                self._last_aux_loss = 0.0

            ffn_out = ffn_out.squeeze(0)  # (N_compute, D)

            # Scatter back
            full_ffn = torch.zeros_like(h_flat)
            full_ffn.index_copy_(0, compute_indices, ffn_out)
            h = full_ffn.view(B, S, D)
        else:
            if self.checkpoint_ffn and self.training:
                def ffn_forward(t: torch.Tensor):
                    out = self.ffn(t)
                    if isinstance(out, dict):
                        aux = out.get("aux_loss", t.new_zeros(()))
                        if not torch.is_tensor(aux):
                            aux = t.new_tensor(aux)
                        return out["output"], aux
                    return out, t.new_zeros(())

                ffn_out, aux_loss = checkpoint(ffn_forward, h, use_reentrant=False)
                h = ffn_out
                self._last_aux_loss = aux_loss
            else:
                ffn_result = self.ffn(h)

                # Handle MoE dict return or standard tensor return
                if isinstance(ffn_result, dict):
                    h = ffn_result["output"]
                    self._last_aux_loss = ffn_result.get("aux_loss", 0.0)
                else:
                    h = ffn_result
                    self._last_aux_loss = 0.0

        x = x + self.dropout(h)

        if use_cache:
            return x, present_kv
        return x

    def get_aux_loss(self) -> float:
        """Get auxiliary loss from last forward pass (MoE only)."""
        return self._last_aux_loss

    def get_exit_confidence(self) -> Optional[torch.Tensor]:
        """Get latest exit-gate confidence tensor for this block."""
        return self._last_exit_confidence

    def get_ffn_entropy(self) -> Optional[Dict[str, float]]:
        """Get FFN entropy if compositional."""
        if isinstance(self.ffn, CompositionalFFN):
            return self.ffn.get_entropy()
        elif isinstance(self.ffn, MoECompositionalFFN):
            return self.ffn.get_entropy()
        return None

    def get_moe_metrics(self) -> Optional[Dict[str, Any]]:
        """Get MoE-specific metrics if MoE FFN."""
        if isinstance(self.ffn, MoECompositionalFFN):
            return {
                "router_entropy": self.ffn.get_router_entropy(),
                "expert_similarity": self.ffn.get_expert_similarity(),
                "primitive_entropy": self.ffn.get_entropy(),
            }
        return None


class PILONTransformer(nn.Module):
    """
    Full PILON Transformer for language modeling.

    Architecture:
    - Token embedding + position embedding
    - N transformer blocks
    - Final norm
    - LM head (tied with embedding)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Create primitive banks if compositional
        self.primitive_banks = None
        if config.ffn_type == "compositional":
            pc = config.primitive_config
            bands = [{"name": b.name, "layers": b.layers} for b in pc.bands]
            self.primitive_banks = BandPrimitiveBanks(
                d_model=config.d_model,
                d_ff=config.d_ff,
                n_primitives=pc.n_primitives,
                rank=pc.rank,
                bands=bands,
                share_fc1_fc2=pc.share_fc1_fc2,
                n_hot=pc.n_hot,
                swap_interval=pc.swap_interval,
                ternary=pc.ternary_primitives,
                activation_bits=pc.activation_bits,
            )

        # Get MoE config if present
        moe_config = None
        if config.ffn_type == "compositional" and config.primitive_config.moe_config is not None:
            moe_config = config.primitive_config.moe_config

        # Resolve activation string (handle use_squared_relu override)
        if config.ffn_type == "compositional":
            pc = config.primitive_config
            activation_str = pc.activation
            if pc.use_squared_relu:
                activation_str = "squared_relu"
            use_subln = pc.use_subln
        else:
            activation_str = "gelu"
            use_subln = False

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_head=config.d_head,
                d_ff=config.d_ff,
                layer_idx=i,
                ffn_type=config.ffn_type,
                dropout=config.dropout,
                norm_type=config.norm_type,
                checkpoint_ffn=config.checkpoint_ffn,
                primitive_banks=self.primitive_banks,
                n_primitives=config.primitive_config.n_primitives if config.ffn_type == "compositional" else 0,
                top_k=config.primitive_config.top_k if config.ffn_type == "compositional" else 0,
                top_k_fc1=config.primitive_config.top_k_fc1 if config.ffn_type == "compositional" else None,
                top_k_fc2=config.primitive_config.top_k_fc2 if config.ffn_type == "compositional" else None,
                activation=activation_str,
                max_seq_len=config.max_seq_len,
                temperature=config.primitive_config.temperature if config.ffn_type == "compositional" else 1.0,
                forward_fast_mode=config.primitive_config.forward_fast_mode if config.ffn_type == "compositional" else "auto",
                forward_fast_min_topk=config.primitive_config.forward_fast_min_topk if config.ffn_type == "compositional" else None,
                moe_config=moe_config,
                enable_early_exit=config.enable_early_exit,
                exit_threshold=config.exit_threshold,
                use_subln=use_subln,
            )
            for i in range(config.n_layers)
        ])

        # Final norm
        if config.norm_type == "rmsnorm":
            self.final_norm = RMSNorm(config.d_model)
        else:
            self.final_norm = nn.LayerNorm(config.d_model)

        # LM head (tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation

        Returns:
            Dictionary with 'logits', optionally 'loss', and 'aux_loss' (for MoE)
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings (offset positions if using KV cache)
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].size(2)
        positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Transform attention mask if provided
        attn_mask = None
        if attention_mask is not None:
            # If using KV cache, extend mask to cover past tokens (assume past tokens are valid)
            if past_len > 0 and attention_mask.size(1) == seq_len:
                pad = torch.ones(
                    attention_mask.size(0),
                    past_len,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([pad, attention_mask], dim=1)
            # Convert (batch, seq) mask to (batch, 1, 1, seq) for broadcasting
            attn_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

        # Apply transformer blocks and collect aux_losses
        total_aux_loss = 0.0
        exit_confidences: List[torch.Tensor] = []
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = None
            if past_key_values is not None:
                past_kv = past_key_values[i]
            if use_cache:
                x, present_kv = layer(x, attn_mask, use_cache=True, past_kv=past_kv)
                present_key_values.append(present_kv)
            else:
                x = layer(x, attn_mask)
            total_aux_loss = total_aux_loss + layer.get_aux_loss()
            if self.training:
                layer_exit_conf = layer.get_exit_confidence()
                if layer_exit_conf is not None:
                    exit_confidences.append(layer_exit_conf)

        # Final norm
        x = self.final_norm(x)

        # LM head
        logits = self.lm_head(x)

        output = {"logits": logits}
        if use_cache:
            output["past_key_values"] = present_key_values
        if self.training and exit_confidences:
            output["exit_confidences"] = torch.stack(exit_confidences, dim=0)

        # Include aux_loss for MoE (even when currently zero) so train loop can
        # consistently apply configured aux weighting.
        if isinstance(total_aux_loss, torch.Tensor):
            output["aux_loss"] = total_aux_loss
        elif total_aux_loss != 0.0:
            output["aux_loss"] = total_aux_loss

        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output["loss"] = loss

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        precision: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs (batch, seq)
            attention_mask: Optional attention mask for prefill
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            precision: Optional autocast precision ("bf16", "fp16", "fp32")

        Returns:
            Generated token IDs (batch, seq + max_new_tokens)
        """
        was_training = self.training
        self.eval()
        device = input_ids.device

        # Resolve autocast dtype
        amp_dtype = None
        if device.type == "cuda":
            if precision is None:
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                prec = precision.lower()
                if prec == "bf16":
                    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                elif prec == "fp16":
                    amp_dtype = torch.float16
                else:
                    amp_dtype = None

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        past_key_values = None

        try:
            from contextlib import nullcontext
        except ImportError:
            class nullcontext:
                def __enter__(self):
                    return None
                def __exit__(self, *args):
                    return False

        autocast_ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else nullcontext()

        with torch.inference_mode(), autocast_ctx:
            for _ in range(max_new_tokens):
                # Truncate to max_seq_len if needed
                if input_ids.size(1) > self.config.max_seq_len:
                    input_ids = input_ids[:, -self.config.max_seq_len:]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, -self.config.max_seq_len:]
                    past_key_values = None

                if past_key_values is None:
                    idx_cond = input_ids
                    attn = attention_mask
                else:
                    idx_cond = input_ids[:, -1:]
                    attn = None

                outputs = self.forward(
                    idx_cond,
                    attention_mask=attn,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs["logits"][:, -1, :]  # Get last token logits
                past_key_values = outputs.get("past_key_values", None)

                # Apply temperature (skip if 0, will use argmax below)
                if temperature > 0 and temperature != 1.0:
                    logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample or greedy (use argmax if temperature=0 or do_sample=False)
                if do_sample and temperature > 0:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if attention_mask is not None:
                    one = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, one], dim=1)

        if was_training:
            self.train()

        return input_ids

    def update_caches(self) -> None:
        """
        Update caches for all layers (e.g. top-k indices).
        Should be called at the start of each training step.
        """
        for layer in self.layers:
            if hasattr(layer.ffn, "update_topk_cache"):
                layer.ffn.update_topk_cache()

    def get_all_entropy(self) -> Dict[str, float]:
        """Get entropy for all compositional FFN layers."""
        entropy = {}
        for i, layer in enumerate(self.layers):
            layer_entropy = layer.get_ffn_entropy()
            if layer_entropy is not None:
                entropy[f"layer_{i}_fc1"] = layer_entropy["fc1_entropy"]
                entropy[f"layer_{i}_fc2"] = layer_entropy["fc2_entropy"]
        return entropy

    def get_primitive_usage(self) -> Dict[str, torch.Tensor]:
        """Get primitive usage weights for all layers."""
        usage = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer.ffn, CompositionalFFN):
                weights = layer.ffn.get_composition_weights()
                usage[f"layer_{i}_fc1"] = weights["fc1_weights"].detach()
                usage[f"layer_{i}_fc2"] = weights["fc2_weights"].detach()
            elif isinstance(layer.ffn, MoECompositionalFFN):
                # For MoE, get expert composition weights
                fc1_weights, fc2_weights = layer.ffn.expert_compositions.get_all_expert_weights()
                usage[f"layer_{i}_fc1_experts"] = fc1_weights.detach()
                usage[f"layer_{i}_fc2_experts"] = fc2_weights.detach()
        return usage

    def get_moe_metrics(self) -> Optional[Dict[str, Any]]:
        """Get MoE-specific metrics for all layers."""
        metrics = {}
        has_moe = False

        for i, layer in enumerate(self.layers):
            layer_metrics = layer.get_moe_metrics()
            if layer_metrics is not None:
                has_moe = True
                metrics[f"layer_{i}"] = layer_metrics

        if not has_moe:
            return None

        # Compute aggregate metrics from layer metrics only
        layer_metrics = [m for k, m in metrics.items() if k.startswith("layer_")]
        router_entropies = [m["router_entropy"] for m in layer_metrics]
        metrics["mean_router_entropy"] = sum(router_entropies) / len(router_entropies)

        expert_sims = [m["expert_similarity"]["mean_similarity"] for m in layer_metrics]
        metrics["mean_expert_similarity"] = sum(expert_sims) / len(expert_sims)

        return metrics

    def swap_tiers(self, optimizer) -> int:
        """
        Iterate all primitive banks and call maybe_swap() on tiered banks.

        Args:
            optimizer: The optimizer to transfer Adam states

        Returns:
            Number of banks that performed a swap
        """
        n_swapped = 0
        if self.primitive_banks is not None:
            from .tiered_bank import TieredPrimitiveBank
            for bank in list(self.primitive_banks.fc1_banks.values()) + list(self.primitive_banks.fc2_banks.values()):
                if isinstance(bank, TieredPrimitiveBank):
                    if bank.maybe_swap(optimizer):
                        n_swapped += 1
        return n_swapped

    def get_early_exit_metrics(self) -> Optional[Dict[str, Any]]:
        """Get skip counts per layer for early exit."""
        has_exit = False
        skip_ratios = {}
        total_skips = 0
        total_tokens = 0
        for i, layer in enumerate(self.layers):
            if layer.enable_early_exit and layer.exit_gate is not None:
                has_exit = True
                skip = layer._last_skip_count
                total = layer._last_total_count
                ratio = skip / max(total, 1)
                skip_ratios[f"layer_{i}"] = ratio
                total_skips += skip
                total_tokens += total

        if not has_exit:
            return None

        n_layers = len(self.layers)
        avg_skip_ratio = total_skips / max(total_tokens, 1)
        avg_layers = n_layers * (1.0 - avg_skip_ratio)

        return {
            "skip_ratios": skip_ratios,
            "avg_layers_per_token": avg_layers,
            "total_skips": total_skips,
            "total_tokens": total_tokens,
        }

    def is_moe_model(self) -> bool:
        """Check if this model uses MoE."""
        if len(self.layers) > 0:
            return isinstance(self.layers[0].ffn, MoECompositionalFFN)
        return False

    def parameter_count(self) -> Dict[str, int]:
        """Count parameters by component."""
        ffn_params = 0
        router_params = 0
        expert_params = 0

        for layer in self.layers:
            if isinstance(layer.ffn, CompositionalFFN):
                ffn_params += sum(p.numel() for p in layer.ffn.composition_weights.parameters())
            elif isinstance(layer.ffn, MoECompositionalFFN):
                # Count router and expert composition params separately
                router_params += sum(p.numel() for p in layer.ffn.router.parameters())
                expert_params += sum(p.numel() for p in layer.ffn.expert_compositions.parameters())
            else:
                ffn_params += sum(p.numel() for p in layer.ffn.parameters())

        counts = {
            "embedding": self.token_embedding.weight.numel() + self.position_embedding.weight.numel(),
            "attention": sum(
                sum(p.numel() for p in layer.attention.parameters())
                for layer in self.layers
            ),
            "ffn": ffn_params,
            "norm": sum(
                sum(p.numel() for p in layer.norm1.parameters()) +
                sum(p.numel() for p in layer.norm2.parameters())
                for layer in self.layers
            ) + sum(p.numel() for p in self.final_norm.parameters()),
        }

        if self.primitive_banks is not None:
            counts["primitive_banks"] = sum(
                p.numel() for p in self.primitive_banks.parameters()
            )

        # Add MoE-specific parameter counts
        if router_params > 0:
            counts["router"] = router_params
        if expert_params > 0:
            counts["expert_compositions"] = expert_params

        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


def create_model(config: ModelConfig) -> PILONTransformer:
    """Create a PILON Transformer from config."""
    return PILONTransformer(config)


def create_baseline_model(config: ModelConfig) -> PILONTransformer:
    """Create a dense baseline model from config."""
    baseline_config = config.get_baseline_config()
    return PILONTransformer(baseline_config)


if __name__ == "__main__":
    # Test the model
    print("Testing PILONTransformer...")

    from .config import ModelConfig

    # Create model
    config = ModelConfig()
    model = PILONTransformer(config)

    print(f"\nModel config:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  ffn_type: {config.ffn_type}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, do_sample=False)
    print(f"\nGeneration:")
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")

    # Parameter count
    param_counts = model.parameter_count()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Entropy
    entropy = model.get_all_entropy()
    print(f"\nEntropy (should be ~3.5 for uniform 32 primitives):")
    for name, value in list(entropy.items())[:4]:
        print(f"  {name}: {value:.2f}")

    # Test baseline
    print("\n--- Testing Baseline Model ---")
    baseline = create_baseline_model(config)
    baseline_outputs = baseline(input_ids, labels=labels)
    print(f"Baseline loss: {baseline_outputs['loss'].item():.4f}")
    print(f"Baseline params: {baseline.parameter_count()['total']:,}")
