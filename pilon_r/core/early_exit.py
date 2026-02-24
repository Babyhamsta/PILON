"""
PILON-R Early Exit for Easy Tokens (Phase B.5c)

Lightweight exit gates on TransformerBlocks. During inference, tokens
skip the FFN if gate confidence exceeds a threshold. Gates are trained
post-hoc with all other parameters frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class ExitGate(nn.Module):
    """
    Lightweight gate that predicts whether a token can skip the FFN.

    Output: sigmoid probability in [0, 1] where higher = more confident
    the token can skip. Bias initialized to -2.0 to start conservative
    (mostly don't skip).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        # Conservative initialization: start by not skipping
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute skip confidence.

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Confidence tensor (batch, seq, 1) in [0, 1]
        """
        return torch.sigmoid(self.linear(x))


@dataclass
class EarlyExitMetrics:
    """Track early exit statistics per layer."""

    skip_counts: Dict[int, int] = field(default_factory=dict)
    total_tokens: Dict[int, int] = field(default_factory=dict)

    def update(self, layer_idx: int, n_skipped: int, n_total: int) -> None:
        self.skip_counts[layer_idx] = self.skip_counts.get(layer_idx, 0) + n_skipped
        self.total_tokens[layer_idx] = self.total_tokens.get(layer_idx, 0) + n_total

    def skip_ratio_per_layer(self) -> Dict[int, float]:
        """Fraction of tokens skipped at each layer."""
        ratios = {}
        for layer_idx in sorted(self.skip_counts.keys()):
            total = self.total_tokens.get(layer_idx, 1)
            ratios[layer_idx] = self.skip_counts[layer_idx] / max(total, 1)
        return ratios

    def avg_layers_per_token(self, n_layers: int) -> float:
        """Average number of FFN layers each token passes through."""
        if not self.skip_counts:
            return float(n_layers)
        total_skips = sum(self.skip_counts.values())
        # Use the total from any layer (they should all be equal)
        total_tokens = max(self.total_tokens.values()) if self.total_tokens else 1
        avg_skipped = total_skips / max(total_tokens, 1)
        return n_layers - avg_skipped

    def reset(self) -> None:
        self.skip_counts.clear()
        self.total_tokens.clear()


def compute_layer_kl_divergence(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute KL divergence between intermediate logits at layer_idx and final logits.

    Forward through all layers collecting intermediate representations.
    Branch at layer_idx to compute logits via final_norm + lm_head,
    compare with final logits.

    Args:
        model: PILONTransformer
        input_ids: Token IDs (batch, seq)
        layer_idx: Layer index to branch at
        device: Device
        attention_mask: Optional attention mask for padding

    Returns:
        KL divergence scalar
    """
    model.eval()
    with torch.no_grad():
        # Full forward to get final logits
        outputs = model(input_ids, attention_mask=attention_mask)
        final_logits = outputs["logits"]

        # Partial forward to get intermediate representation
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = model.token_embedding(input_ids) + model.position_embedding(positions)
        x = model.dropout(x)

        for i, layer in enumerate(model.layers):
            x = layer(x, attention_mask=attention_mask)
            if i == layer_idx:
                break

        # Branch: norm + lm_head at this intermediate point
        intermediate_x = model.final_norm(x)
        intermediate_logits = model.lm_head(intermediate_x)

    # KL(intermediate || final) - per-token, averaged
    log_p = F.log_softmax(intermediate_logits, dim=-1)
    q = F.softmax(final_logits, dim=-1)
    kl = F.kl_div(log_p, q, reduction="batchmean")
    return kl


def train_exit_gates(
    model,
    dataloader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    exit_threshold: float = 0.5,
) -> Dict[str, List[float]]:
    """
    Train exit gates post-hoc with all other parameters frozen.

    The training signal is: a token should skip at layer i if the KL divergence
    between the intermediate logits (at layer i) and the final logits is small.

    Args:
        model: PILONTransformer with enable_early_exit=True
        dataloader: Training data
        device: Device
        epochs: Training epochs
        lr: Learning rate for gate parameters
        exit_threshold: Target skip confidence threshold

    Returns:
        Training history dict with per-epoch losses
    """
    # Save requires_grad state for all parameters so we can restore after training
    saved_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    # Freeze everything except exit gates
    for param in model.parameters():
        param.requires_grad = False

    gate_params = []
    for layer in model.layers:
        if hasattr(layer, "exit_gate") and layer.exit_gate is not None:
            for param in layer.exit_gate.parameters():
                param.requires_grad = True
                gate_params.append(param)

    if not gate_params:
        # Restore requires_grad before raising
        for name, param in model.named_parameters():
            if name in saved_requires_grad:
                param.requires_grad = saved_requires_grad[name]
        raise ValueError(
            "No exit gates found. Ensure model was created with enable_early_exit=True"
        )

    import time
    from datetime import datetime

    def _log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [INFO] {msg}", flush=True)

    optimizer = torch.optim.Adam(gate_params, lr=lr)
    history = {"epoch_losses": []}

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        model.train()  # gates need training mode
        _log(f"Gate epoch {epoch+1}/{epochs}: waiting for first batch...")

        for batch in dataloader:
            if n_batches == 0:
                _log(f"Gate epoch {epoch+1}/{epochs}: first batch received, training started")
            # Support both dict (streaming) and tuple (TensorDataset) formats
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else None
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

            # Get final logits (no grad needed for this)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                final_logits = outputs["logits"].detach()

            # Forward through layers collecting intermediate reps
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=device).unsqueeze(0)

            with torch.no_grad():
                x = model.token_embedding(input_ids) + model.position_embedding(positions)
                x = model.dropout(x)

            batch_loss = torch.tensor(0.0, device=device)
            n_gates = 0

            for i, layer in enumerate(model.layers):
                with torch.no_grad():
                    # Run attention (pass attention_mask for padding support)
                    h = layer.norm1(x)
                    h = layer.attention(h, attention_mask=attention_mask)
                    x_post_attn = x + layer.dropout(h)

                if hasattr(layer, "exit_gate") and layer.exit_gate is not None:
                    # Gate prediction (needs grad)
                    confidence = layer.exit_gate(x_post_attn)  # (B, S, 1)

                    # Compute KL target: how similar are intermediate logits to final?
                    with torch.no_grad():
                        intermediate_norm = model.final_norm(x_post_attn)
                        intermediate_logits = model.lm_head(intermediate_norm)

                        # KL divergence per token
                        log_p = F.log_softmax(intermediate_logits, dim=-1)
                        q = F.softmax(final_logits, dim=-1)
                        kl_per_token = F.kl_div(
                            log_p, q, reduction="none"
                        ).sum(dim=-1, keepdim=True)  # (B, S, 1)

                        # Target: skip if KL is small (token is "easy")
                        # Binary target: 1 if KL < threshold, 0 otherwise
                        # Use a soft target based on scaled KL
                        target = torch.exp(-kl_per_token).detach()  # (B, S, 1)

                    # BCE loss: gate should predict skip confidence
                    gate_loss = F.binary_cross_entropy(
                        confidence, target, reduction="mean"
                    )
                    batch_loss = batch_loss + gate_loss
                    n_gates += 1

                with torch.no_grad():
                    # Continue forward through FFN
                    h = layer.norm2(x_post_attn)
                    ffn_result = layer.ffn(h)
                    if isinstance(ffn_result, dict):
                        h = ffn_result["output"]
                    else:
                        h = ffn_result
                    x = x_post_attn + layer.dropout(h)

            if n_gates > 0:
                batch_loss = batch_loss / n_gates
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                n_batches += 1

                if n_batches % 50 == 0:
                    avg_so_far = total_loss / n_batches
                    elapsed = time.time() - epoch_start
                    _log(
                        f"Gate epoch {epoch+1}/{epochs} batch {n_batches}: "
                        f"loss={batch_loss.item():.4f} avg={avg_so_far:.4f} "
                        f"elapsed={elapsed:.1f}s"
                    )

        avg_loss = total_loss / max(n_batches, 1)
        history["epoch_losses"].append(avg_loss)
        elapsed = time.time() - epoch_start
        _log(f"Gate epoch {epoch+1}/{epochs} complete: loss={avg_loss:.4f} batches={n_batches} time={elapsed:.1f}s")

    # Restore requires_grad for all parameters
    for name, param in model.named_parameters():
        if name in saved_requires_grad:
            param.requires_grad = saved_requires_grad[name]

    return history
