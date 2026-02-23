"""
PILON-R Model Compression (v3 - Production Ready)

Converts pretrained dense models (e.g., Qwen2.5-1.5B) to PILON-R compositional format.

v3 Fixes:
1. Single-device loading (no device_map="auto" breakage)
2. Correct top-k annealing (linear from n_primitives to top_k)
3. Proper phase training flags (freeze_primitives_phase2)
4. Complete reconstruction error reporting (all projections, all layers)
5. Reloadable save format with rebuild script
6. Top-k aware composition initialization
7. Correct device placement throughout

Compression Levels:
    minimal    - n_primitives=64, rank=64, top_k=12 (highest quality)
    baseline   - n_primitives=48, rank=48, top_k=12
    mild       - n_primitives=36, rank=36, top_k=10
    moderate   - n_primitives=24, rank=24, top_k=8  (recommended start)
    aggressive - n_primitives=16, rank=16, top_k=8
    extreme    - n_primitives=8,  rank=8,  top_k=4  (maximum compression)

Usage:
    # List available compression levels
    python -m pilon_r.compress --list-levels
    
    # Compress with preset level
    python -m pilon_r.compress Qwen/Qwen2.5-1.5B --compression-level moderate
    
    # Compress with custom settings
    python -m pilon_r.compress Qwen/Qwen2.5-1.5B --n-primitives 32 --rank 32 --top-k 8
    
    # Compress and fine-tune
    python -m pilon_r.compress Qwen/Qwen2.5-1.5B --compression-level moderate --finetune
    
    # Reload and evaluate
    python -m pilon_r.compress --load compressed_model/ --eval-only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast
from pathlib import Path
import argparse
import math
import time
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from contextlib import nullcontext
import json

from .core.config import (
    ModelConfig, TrainingConfig, PrimitiveConfig, BandConfig,
    COMPRESSION_LEVELS, get_compression_config, get_all_compression_levels
)
from .core.primitives import PrimitiveBank, BandPrimitiveBanks, LayerCompositionWeights
from .core.ffn import CompositionalFFN, StandardFFN
from .core.data import load_text_dataset, get_tokenizer, create_dataloader
from .core.metrics import Logger, compute_entropy


@dataclass
class CompressionConfig:
    """Configuration for model compression."""
    
    source_model: str = "Qwen/Qwen2.5-1.5B"
    
    # Compression level (if using preset)
    compression_level: Optional[str] = None  # e.g., "moderate", "aggressive"
    
    # Target compression settings
    n_primitives: int = 32
    rank: int = 32
    top_k: int = 8
    
    # Band configuration
    n_bands: int = 4
    
    # SVD decomposition
    scale_by_singular_values: bool = True
    svd_non_overlapping: bool = False  # Non-overlapping SVD slices for orthogonal primitives
    
    # Neuron permutation alignment (pre-compression reparameterization)
    neuron_alignment: bool = False  # Align intermediate neurons across band layers before SVD
    neuron_alignment_signature: str = "down"  # "down" or "gate_up"
    
    # Composition mode
    composition_mode: str = "softmax"  # "softmax" (original) or "unconstrained" (signed weights)
    
    # Composition initialization
    composition_init_steps: int = 200
    composition_init_lr: float = 0.5
    composition_init_with_topk: bool = True  # Optimize with top-k masking
    
    # Fine-tuning
    finetune_steps: int = 5000
    finetune_lr: float = 1e-5
    finetune_dataset: str = "Elriggs/openwebtext-100k"
    finetune_batch_size: int = 4
    finetune_max_seq_len: int = 512
    finetune_mode: str = "ffn-distill"
    
    # What to train
    train_primitives: bool = True
    train_compositions: bool = True
    freeze_attention: bool = True
    freeze_embeddings: bool = True
    
    # Phase training
    use_phase_training: bool = True
    phase1_ratio: float = 0.2
    freeze_primitives_phase2: bool = True  # Proper flag for phase 2
    
    # Top-k annealing
    topk_anneal: bool = True
    topk_anneal_start_ratio: float = 0.1  # Start annealing at 10% of training
    
    # Output
    output_dir: str = "compressed_model"
    device: str = "cuda"
    precision: str = "bf16"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "CompressionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass 
class CompressionResult:
    """Results from compression."""
    
    source_model: str
    source_params: int
    source_ffn_params: int
    compressed_ffn_params: int
    ffn_compression_ratio: float
    
    # Complete per-layer, per-projection errors
    reconstruction_errors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mean_gate_error: float = 0.0
    mean_up_error: float = 0.0
    mean_down_error: float = 0.0
    
    source_ppl: Optional[float] = None
    compressed_ppl: Optional[float] = None
    ppl_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Model Detection and Configuration
# =============================================================================

def detect_model_type(model) -> str:
    """Detect the type of model architecture."""
    model_class = model.__class__.__name__.lower()
    
    if "qwen" in model_class:
        return "qwen"
    elif "llama" in model_class or "mistral" in model_class:
        return "llama"
    elif "gpt2" in model_class:
        return "gpt2"
    else:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            if hasattr(model.model.layers[0].mlp, "gate_proj"):
                return "llama"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return "gpt2"
        raise ValueError(f"Unknown model architecture: {model_class}")


def get_model_info(model) -> Dict[str, Any]:
    """Extract model configuration."""
    model_type = detect_model_type(model)
    
    if model_type in ["qwen", "llama"]:
        return {
            "n_layers": len(model.model.layers),
            "d_model": model.config.hidden_size,
            "d_ff": model.config.intermediate_size,
            "model_type": model_type,
            "is_glu": True,
        }
    else:
        return {
            "n_layers": len(model.transformer.h),
            "d_model": model.config.n_embd,
            "d_ff": model.config.n_embd * 4,
            "model_type": model_type,
            "is_glu": False,
        }


def get_ffn_weights(model, layer_idx: int) -> Dict[str, torch.Tensor]:
    """Extract FFN weights from a layer."""
    model_type = detect_model_type(model)
    
    if model_type in ["qwen", "llama"]:
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp
        return {
            "gate_proj": mlp.gate_proj.weight.data.clone(),
            "up_proj": mlp.up_proj.weight.data.clone(),
            "down_proj": mlp.down_proj.weight.data.clone(),
        }
    else:
        layer = model.transformer.h[layer_idx]
        return {
            "fc1": layer.mlp.c_fc.weight.data.clone().T,
            "fc2": layer.mlp.c_proj.weight.data.clone().T,
        }


def count_ffn_params(model) -> int:
    """Count FFN-only parameters."""
    model_type = detect_model_type(model)
    total = 0
    
    if model_type in ["qwen", "llama"]:
        for layer in model.model.layers:
            mlp = layer.mlp
            total += mlp.gate_proj.weight.numel()
            total += mlp.up_proj.weight.numel()
            total += mlp.down_proj.weight.numel()
    else:
        for layer in model.transformer.h:
            total += layer.mlp.c_fc.weight.numel()
            total += layer.mlp.c_proj.weight.numel()
    
    return total


def estimate_model_memory(model_name: str) -> float:
    """Estimate model memory in GB (rough)."""
    # Rough estimates for common models
    estimates = {
        "Qwen/Qwen2.5-0.5B": 1.0,
        "Qwen/Qwen2.5-1.5B": 3.0,
        "Qwen/Qwen2.5-3B": 6.0,
        "Qwen/Qwen2.5-7B": 14.0,
    }
    for name, size in estimates.items():
        if name in model_name:
            return size
    return 8.0  # Default assumption


def get_available_gpu_memory() -> float:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


# =============================================================================
# SVD-Based Primitive Initialization
# =============================================================================

def create_diverse_primitives_from_svd(
    weight: torch.Tensor,
    n_primitives: int,
    rank: int,
    scale_by_singular: bool = True,
    device: Optional[torch.device] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create DIVERSE primitives by slicing DIFFERENT singular subspaces.
    
    Each primitive gets a different slice of the SVD spectrum.
    """
    device = device or weight.device
    weight_f32 = weight.float().cpu()  # SVD on CPU for stability
    
    # Full SVD
    max_rank_needed = min(n_primitives * rank, min(weight.shape))
    U_full, S_full, Vh_full = torch.linalg.svd(weight_f32, full_matrices=False)
    
    primitives = []
    available_dims = min(U_full.shape[1], max_rank_needed)
    
    # Calculate stride for diversity
    if n_primitives > 1:
        stride = max(1, (available_dims - rank) // (n_primitives - 1))
    else:
        stride = 0
    
    for i in range(n_primitives):
        start_idx = min(i * stride, max(0, available_dims - rank))
        end_idx = min(start_idx + rank, available_dims)
        actual_rank = end_idx - start_idx
        
        # Handle case where we don't have enough dims
        if actual_rank < rank:
            # Pad with zeros or wrap around
            start_idx = 0
            end_idx = rank
        
        U_slice = U_full[:, start_idx:end_idx].clone()
        S_slice = S_full[start_idx:end_idx].clone()
        V_slice = Vh_full[start_idx:end_idx, :].T.clone()
        
        if scale_by_singular:
            sqrt_S = torch.sqrt(S_slice + 1e-8)
            A = U_slice * sqrt_S.unsqueeze(0)
            B = (V_slice * sqrt_S.unsqueeze(0)).T
        else:
            A = U_slice
            B = (V_slice * S_slice.unsqueeze(0)).T
        
        primitives.append((A.to(device).contiguous(), B.to(device).contiguous()))
    
    return primitives


def create_orthogonal_primitives_from_svd(
    weight: torch.Tensor,
    n_primitives: int,
    rank: int,
    scale_by_singular: bool = True,
    device: Optional[torch.device] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create ORTHOGONAL primitives using non-overlapping SVD slices.
    
    Each primitive gets a unique, non-overlapping slice of the SVD spectrum.
    Primitive i gets singular components [i*rank : (i+1)*rank].
    
    This guarantees the primitives span orthogonal subspaces, enabling
    analytical closed-form weight computation and perfect reconstruction
    when n_primitives * rank >= effective_rank(weight).
    
    For Qwen2.5-0.5B: 8 primitives × rank 112 = 896 = full d_model coverage.
    """
    device = device or weight.device
    weight_f32 = weight.float().cpu()
    
    U_full, S_full, Vh_full = torch.linalg.svd(weight_f32, full_matrices=False)
    available_dims = min(U_full.shape[1], S_full.shape[0])
    
    primitives = []
    for i in range(n_primitives):
        start_idx = i * rank
        end_idx = min(start_idx + rank, available_dims)
        actual_rank = end_idx - start_idx
        
        if actual_rank <= 0:
            # Beyond available dims — create near-zero primitive
            A = torch.zeros(weight.shape[0], rank, device=device)
            B = torch.zeros(rank, weight.shape[1], device=device)
            primitives.append((A.contiguous(), B.contiguous()))
            continue
        
        U_slice = U_full[:, start_idx:end_idx].clone()
        S_slice = S_full[start_idx:end_idx].clone()
        V_slice = Vh_full[start_idx:end_idx, :].T.clone()  # V not Vh
        
        # Pad if actual_rank < rank (last primitive may be short)
        if actual_rank < rank:
            pad_a = torch.zeros(weight.shape[0], rank - actual_rank)
            pad_b = torch.zeros(rank - actual_rank, weight.shape[1])
            U_slice = torch.cat([U_slice, pad_a], dim=1)
            V_slice_padded = torch.zeros(weight.shape[1], rank)
            V_slice_padded[:, :actual_rank] = V_slice
            V_slice = V_slice_padded
            S_padded = torch.zeros(rank)
            S_padded[:actual_rank] = S_slice
            S_slice = S_padded
        
        if scale_by_singular:
            sqrt_S = torch.sqrt(S_slice + 1e-8)
            A = U_slice * sqrt_S.unsqueeze(0)       # (d_in, rank)
            B = (V_slice * sqrt_S.unsqueeze(0)).T    # (rank, d_out)
        else:
            A = U_slice
            B = (V_slice * S_slice.unsqueeze(0)).T
        
        primitives.append((A.to(device).contiguous(), B.to(device).contiguous()))
    
    return primitives


def _create_primitives_from_stacked_svd(
    weight_matrices: List[torch.Tensor],
    n_primitives: int,
    rank: int,
    scale_by_singular: bool = True,
    device: Optional[torch.device] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create orthogonal primitives from STACKED (horizontally concatenated) weight matrices.
    
    Instead of SVD(mean([W_0, W_1, ...])) which loses inter-layer differences,
    computes SVD([W_0 | W_1 | ...]) which captures directions used by ANY layer.
    
    The left singular vectors U span the union of all layers' column spaces.
    For each primitive, we extract U slices (shared basis) and compute per-primitive
    B matrices by projecting the average weight back through U.
    
    Args:
        weight_matrices: List of (d_in, d_out) weight matrices from layers in a band
        n_primitives: Number of primitives to create
        rank: Rank of each primitive
        scale_by_singular: Whether to distribute singular values to A and B
        device: Target device
    
    Returns:
        List of (A, B) tuples where A: (d_in, rank), B: (rank, d_out)
    """
    device = device or weight_matrices[0].device
    d_in, d_out = weight_matrices[0].shape
    
    # Stack horizontally: (d_in, n_layers * d_out)
    stacked = torch.cat([w.float().cpu() for w in weight_matrices], dim=1)
    
    # SVD of stacked matrix — U captures directions important to ANY layer
    U_full, S_full, Vh_full = torch.linalg.svd(stacked, full_matrices=False)
    available_dims = min(U_full.shape[1], S_full.shape[0])
    
    # We only need U (left singular vectors) from the stacked SVD.
    # These are the shared basis directions. For the B matrices, we project
    # the average weight through U to get the output-space mapping.
    avg_weight = sum(w.float().cpu() for w in weight_matrices) / len(weight_matrices)
    
    primitives = []
    for i in range(n_primitives):
        start_idx = i * rank
        end_idx = min(start_idx + rank, available_dims)
        actual_rank = end_idx - start_idx
        
        if actual_rank <= 0:
            A = torch.zeros(d_in, rank, device=device)
            B = torch.zeros(rank, d_out, device=device)
            primitives.append((A.contiguous(), B.contiguous()))
            continue
        
        # U slice from stacked SVD — these are the shared basis directions
        U_slice = U_full[:, start_idx:end_idx].clone()  # (d_in, actual_rank)
        
        # Project average weight onto this U subspace to get B:
        # W_avg ≈ U_slice @ (U_slice^T @ W_avg) = U_slice @ B
        B_proj = U_slice.T @ avg_weight  # (actual_rank, d_out)
        
        # Compute the effective singular values for this subspace
        # ||U_slice^T @ W_avg|| gives the energy in this subspace
        S_eff = torch.linalg.svdvals(B_proj)  # singular values of the projection
        
        # Do SVD of the projection to get clean factored form
        U_b, S_b, Vh_b = torch.linalg.svd(B_proj, full_matrices=False)
        
        # Pad if needed
        if actual_rank < rank:
            pad_u = torch.zeros(d_in, rank - actual_rank)
            U_slice = torch.cat([U_slice, pad_u], dim=1)
            pad_ub = torch.eye(rank)
            pad_ub[:actual_rank, :actual_rank] = U_b[:actual_rank, :actual_rank]
            U_b = pad_ub
            pad_s = torch.zeros(rank)
            pad_s[:actual_rank] = S_b[:actual_rank]
            S_b = pad_s
            pad_vh = torch.zeros(rank, d_out)
            pad_vh[:actual_rank, :] = Vh_b[:actual_rank, :]
            Vh_b = pad_vh
        
        if scale_by_singular:
            sqrt_S = torch.sqrt(S_b + 1e-8)
            # A = U_slice @ U_b @ diag(sqrt_S)  — rotated left basis scaled
            A = (U_slice @ U_b) * sqrt_S.unsqueeze(0)   # (d_in, rank)
            # B = diag(sqrt_S) @ Vh_b
            B = Vh_b * sqrt_S.unsqueeze(1)                # (rank, d_out)
        else:
            A = U_slice @ U_b                              # (d_in, rank)
            B = Vh_b * S_b.unsqueeze(1)                    # (rank, d_out)
        
        primitives.append((A.to(device).contiguous(), B.to(device).contiguous()))
    
    return primitives


def compute_analytical_composition_weights(
    primitives: List[Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor
) -> torch.Tensor:
    """
    Compute optimal unconstrained composition weights via closed-form least squares.
    
    For orthogonal primitives (non-overlapping SVD slices), the primitives are
    mutually orthogonal so the solution reduces to independent projections:
        w_i = <M_i, W_target> / ||M_i||^2
    
    For non-orthogonal primitives, solves the full linear system:
        w = (G^-1) @ b  where G_ij = <M_i, M_j>, b_i = <M_i, target>
    
    Returns raw weights (not logits) — these can be positive or negative.
    """
    device = target.device
    n = len(primitives)
    target_f32 = target.float()
    
    # Precompute full weight matrices M_i = A_i @ B_i
    Ms = []
    for A, B in primitives:
        M = (A.to(device).float() @ B.to(device).float())
        # Ensure M matches target dimensions
        M = M[:target_f32.shape[0], :target_f32.shape[1]]
        Ms.append(M)
    
    # Gram matrix: G_ij = <M_i, M_j> = tr(M_i^T M_j)
    # For orthogonal primitives this is diagonal
    G = torch.zeros(n, n, device=device)
    b = torch.zeros(n, device=device)
    
    for i in range(n):
        b[i] = (Ms[i] * target_f32).sum()  # <M_i, target>
        for j in range(i, n):
            val = (Ms[i] * Ms[j]).sum()  # <M_i, M_j>
            G[i, j] = val
            G[j, i] = val
    
    # Solve G @ w = b
    # Add small regularization for numerical stability
    G += 1e-6 * torch.eye(n, device=device)
    
    try:
        w = torch.linalg.solve(G, b)
    except RuntimeError:
        # Fallback to pseudo-inverse if singular
        w = torch.linalg.lstsq(G, b).solution
    
    return w


def compute_reconstruction_error(
    original: torch.Tensor,
    primitives: List[Tuple[torch.Tensor, torch.Tensor]],
    weights: torch.Tensor,
    top_k: Optional[int] = None,
    unconstrained: bool = False
) -> Dict[str, float]:
    """
    Compute reconstruction error with optional top-k masking.
    
    Args:
        unconstrained: If True, use magnitude-based top-k and no renormalization.
    """
    device = original.device
    
    A_stack = torch.stack([A.to(device) for A, _ in primitives])
    B_stack = torch.stack([B.to(device) for _, B in primitives])
    weights = weights.to(device)
    
    if top_k is not None and top_k < len(primitives):
        if unconstrained:
            # Select by magnitude, no renormalization
            top_weights, top_indices = torch.topk(weights.abs(), top_k)
            top_weights = weights[top_indices]  # Keep original signs
        else:
            top_weights, top_indices = torch.topk(weights, top_k)
            top_weights = top_weights / (top_weights.sum() + 1e-8)
        A_stack = A_stack[top_indices]
        B_stack = B_stack[top_indices]
        weights = top_weights
    
    reconstructed = torch.einsum("p,pir,pro->io", weights, A_stack, B_stack)
    
    orig = original.float()
    recon = reconstructed[:orig.shape[0], :orig.shape[1]]
    
    frob_error = torch.norm(orig - recon, p='fro').item()
    frob_orig = torch.norm(orig, p='fro').item()
    relative_error = frob_error / (frob_orig + 1e-8)
    mse = F.mse_loss(recon, orig).item()
    cos_sim = F.cosine_similarity(orig.flatten().unsqueeze(0), recon.flatten().unsqueeze(0)).item()
    
    return {
        "relative_error": relative_error,
        "mse": mse,
        "cosine_similarity": cos_sim,
    }


# =============================================================================
# Composition Weight Initialization (Top-K Aware)
# =============================================================================

def optimize_composition_weights(
    primitives: List[Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    top_k: Optional[int] = None,
    n_iterations: int = 200,
    lr: float = 0.5
) -> torch.Tensor:
    """
    Find optimal composition weights with optional top-k masking.
    
    If top_k is specified, optimizes for sparse selection.
    """
    device = target.device
    n_primitives = len(primitives)
    target_f32 = target.float()
    
    # Move primitives to target device
    A_stack = torch.stack([A.to(device).float() for A, _ in primitives])
    B_stack = torch.stack([B.to(device).float() for _, B in primitives])
    
    logits = torch.zeros(n_primitives, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    
    best_loss = float('inf')
    best_logits = logits.detach().clone()
    
    for _ in range(n_iterations):
        optimizer.zero_grad()
        
        weights = F.softmax(logits, dim=0)
        
        # Apply top-k masking if specified (differentiable approximation)
        if top_k is not None and top_k < n_primitives:
            # Straight-through estimator for top-k
            top_weights, top_indices = torch.topk(weights, top_k)
            mask = torch.zeros_like(weights)
            mask.scatter_(0, top_indices, 1.0)
            # STE: forward uses mask, backward uses soft weights
            weights_sparse = weights * mask
            weights_sparse = weights_sparse / (weights_sparse.sum() + 1e-8)
            # But for gradient, use soft version
            weights_for_recon = weights_sparse.detach() + weights - weights.detach()
        else:
            weights_for_recon = weights
        
        recon = torch.einsum("p,pir,pro->io", weights_for_recon, A_stack, B_stack)
        recon = recon[:target_f32.shape[0], :target_f32.shape[1]]
        
        loss = F.mse_loss(recon, target_f32)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_logits = logits.detach().clone()
        
        loss.backward()
        optimizer.step()
    
    return best_logits


# =============================================================================
# GLU-Aware Primitive Banks
# =============================================================================

class GLUPrimitiveBanks(nn.Module):
    """
    Primitive banks for GLU-style FFN (Qwen, LLaMA).
    
    Three separate banks: gate, up, down.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_primitives: int,
        rank: int,
        bands: List[Dict]
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_primitives = n_primitives
        self.rank = rank
        self.bands = bands
        
        self.layer_to_band = {}
        for band in bands:
            for layer_idx in band["layers"]:
                self.layer_to_band[layer_idx] = band["name"]
        
        self.gate_banks = nn.ModuleDict()
        self.up_banks = nn.ModuleDict()
        self.down_banks = nn.ModuleDict()
        
        for band in bands:
            name = band["name"]
            
            self.gate_banks[name] = PrimitiveBank(
                d_in=d_model, d_out=d_ff, n_primitives=n_primitives,
                rank=rank, name=f"{name}_gate"
            )
            self.up_banks[name] = PrimitiveBank(
                d_in=d_model, d_out=d_ff, n_primitives=n_primitives,
                rank=rank, name=f"{name}_up"
            )
            self.down_banks[name] = PrimitiveBank(
                d_in=d_ff, d_out=d_model, n_primitives=n_primitives,
                rank=rank, name=f"{name}_down"
            )
    
    def get_gate_bank(self, layer_idx: int) -> PrimitiveBank:
        return self.gate_banks[self.layer_to_band[layer_idx]]
    
    def get_up_bank(self, layer_idx: int) -> PrimitiveBank:
        return self.up_banks[self.layer_to_band[layer_idx]]
    
    def get_down_bank(self, layer_idx: int) -> PrimitiveBank:
        return self.down_banks[self.layer_to_band[layer_idx]]
    
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GLUCompositionWeights(nn.Module):
    """Composition weights for GLU-style FFN.
    
    Supports two modes:
    - "softmax": Original mode. Stores logits, returns softmax(logits).
      Weights are positive and sum to 1. For from-scratch training.
    - "unconstrained": Stores raw scalar weights (can be negative).
      Enables proper linear combination of primitives. For pretrained conversion.
    """
    
    def __init__(self, n_primitives: int, top_k: int, layer_idx: int,
                 temperature: float = 1.0, mode: str = "softmax"):
        super().__init__()
        
        self.n_primitives = n_primitives
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.temperature = temperature
        self.mode = mode
        
        if mode == "unconstrained":
            # Raw scalar weights — initialized to uniform, set analytically later
            self.gate_weights_raw = nn.Parameter(torch.ones(n_primitives) / n_primitives)
            self.up_weights_raw = nn.Parameter(torch.ones(n_primitives) / n_primitives)
            self.down_weights_raw = nn.Parameter(torch.ones(n_primitives) / n_primitives)
        else:
            self.gate_logits = nn.Parameter(torch.zeros(n_primitives))
            self.up_logits = nn.Parameter(torch.zeros(n_primitives))
            self.down_logits = nn.Parameter(torch.zeros(n_primitives))
            nn.init.normal_(self.gate_logits, mean=0, std=0.01)
            nn.init.normal_(self.up_logits, mean=0, std=0.01)
            nn.init.normal_(self.down_logits, mean=0, std=0.01)
    
    def get_gate_weights(self) -> torch.Tensor:
        if self.mode == "unconstrained":
            return self.gate_weights_raw
        return F.softmax(self.gate_logits / self.temperature, dim=0)
    
    def get_up_weights(self) -> torch.Tensor:
        if self.mode == "unconstrained":
            return self.up_weights_raw
        return F.softmax(self.up_logits / self.temperature, dim=0)
    
    def get_down_weights(self) -> torch.Tensor:
        if self.mode == "unconstrained":
            return self.down_weights_raw
        return F.softmax(self.down_logits / self.temperature, dim=0)
    
    def compute_entropy(self) -> Dict[str, float]:
        if self.mode == "unconstrained":
            # Report weight magnitude statistics instead of entropy
            return {
                "gate_l2": self.gate_weights_raw.norm().item(),
                "up_l2": self.up_weights_raw.norm().item(),
                "down_l2": self.down_weights_raw.norm().item(),
            }
        def entropy(logits):
            probs = F.softmax(logits / self.temperature, dim=0)
            log_probs = F.log_softmax(logits / self.temperature, dim=0)
            return -(probs * log_probs).sum().item()
        return {
            "gate_entropy": entropy(self.gate_logits),
            "up_entropy": entropy(self.up_logits),
            "down_entropy": entropy(self.down_logits),
        }


class CompositionalGLUMLP(nn.Module):
    """
    Drop-in replacement for GLU MLP: down(silu(gate(x)) * up(x))
    
    Matches Qwen/LLaMA SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    
    def __init__(
        self,
        layer_idx: int,
        primitive_banks: GLUPrimitiveBanks,
        composition_weights: GLUCompositionWeights,
        top_k: int = 8
    ):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.primitive_banks = primitive_banks
        self.composition_weights = composition_weights
        self.top_k = top_k
        self.runtime_top_k: Optional[int] = None
        self.active_rank: Optional[int] = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_bank = self.primitive_banks.get_gate_bank(self.layer_idx)
        up_bank = self.primitive_banks.get_up_bank(self.layer_idx)
        down_bank = self.primitive_banks.get_down_bank(self.layer_idx)
        
        gate_weights = self.composition_weights.get_gate_weights()
        up_weights = self.composition_weights.get_up_weights()
        down_weights = self.composition_weights.get_down_weights()
        
        active_rank = self.active_rank
        
        # For unconstrained weights, don't use top-k (PrimitiveBank's top-k
        # assumes positive weights and renormalizes). Use all primitives.
        if getattr(self.composition_weights, 'mode', 'softmax') == "unconstrained":
            effective_top_k = None
        else:
            effective_top_k = self.runtime_top_k if self.runtime_top_k is not None else self.top_k
        
        gate = gate_bank.forward(x, gate_weights, top_k=effective_top_k, active_rank=active_rank)
        up = up_bank.forward(x, up_weights, top_k=effective_top_k, active_rank=active_rank)
        hidden = F.silu(gate) * up
        out = down_bank.forward(hidden, down_weights, top_k=effective_top_k, active_rank=active_rank)
        
        return out
    
    def get_entropy(self) -> Dict[str, float]:
        return self.composition_weights.compute_entropy()


class CompositionalStandardMLP(nn.Module):
    """Drop-in replacement for standard MLP: fc2(gelu(fc1(x)))"""
    
    def __init__(
        self,
        layer_idx: int,
        primitive_banks: BandPrimitiveBanks,
        composition_weights: LayerCompositionWeights,
        top_k: int = 8
    ):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.primitive_banks = primitive_banks
        self.composition_weights = composition_weights
        self.top_k = top_k
        self.runtime_top_k: Optional[int] = None
        self.active_rank: Optional[int] = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_bank = self.primitive_banks.get_fc1_bank(self.layer_idx)
        fc2_bank = self.primitive_banks.get_fc2_bank(self.layer_idx)
        
        fc1_weights = self.composition_weights.get_fc1_weights()
        fc2_weights = self.composition_weights.get_fc2_weights()
        
        effective_top_k = self.runtime_top_k if self.runtime_top_k is not None else self.top_k
        
        h = fc1_bank.forward(x, fc1_weights, top_k=effective_top_k, active_rank=self.active_rank)
        h = F.gelu(h)
        out = fc2_bank.forward(h, fc2_weights, top_k=effective_top_k, active_rank=self.active_rank)
        
        return out
    
    def get_entropy(self) -> Dict[str, float]:
        return self.composition_weights.compute_entropy()


# =============================================================================
# Band Configuration
# =============================================================================

def create_band_config(n_layers: int, n_bands: int) -> List[Dict]:
    """Create band configuration for layer grouping."""
    layers_per_band = n_layers // n_bands
    remainder = n_layers % n_bands
    
    bands = []
    current_layer = 0
    
    for i in range(n_bands):
        n_layers_this_band = layers_per_band + (1 if i < remainder else 0)
        
        if i == 0:
            name = "early"
        elif i == n_bands - 1:
            name = "late"
        else:
            name = f"middle_{i}"
        
        layers = list(range(current_layer, current_layer + n_layers_this_band))
        bands.append({"name": name, "layers": layers})
        current_layer += n_layers_this_band
    
    return bands


# =============================================================================
# Neuron Permutation Alignment (Pre-compression Reparameterization)
# =============================================================================

def compute_per_layer_svd_upper_bound(
    source_model,
    rank: int,
    device: torch.device,
    logger: Optional['Logger'] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute reconstruction error for per-layer truncated SVD (no sharing).
    
    This is the theoretical BEST reconstruction at a given rank, since each
    layer gets its own optimal basis. Serves as a comparison upper bound:
    if band-shared primitives can't approach these numbers, sharing is the problem.
    """
    model_info = get_model_info(source_model)
    n_layers = model_info["n_layers"]
    
    if logger:
        logger.info(f"Computing per-layer SVD upper bound (rank={rank})...")
    
    results = {}
    
    for layer_idx in range(n_layers):
        weights = get_ffn_weights(source_model, layer_idx)
        layer_results = {}
        
        for proj_name, w in weights.items():
            w = w.T.to(device).float()  # (d_in, d_out)
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            # Truncate to rank
            r = min(rank, len(S))
            recon = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
            
            frob_err = torch.norm(w - recon, p='fro').item()
            frob_orig = torch.norm(w, p='fro').item()
            rel_err = frob_err / (frob_orig + 1e-8)
            cos = torch.nn.functional.cosine_similarity(
                w.flatten().unsqueeze(0), recon.flatten().unsqueeze(0)
            ).item()
            
            layer_results[proj_name] = {"relative_error": rel_err, "cosine_similarity": cos}
        
        results[f"layer_{layer_idx}"] = layer_results
        
        if logger:
            parts = [f"{k}={v['relative_error']:.4f} (cos={v['cosine_similarity']:.4f})" 
                     for k, v in layer_results.items()]
            logger.info(f"  Layer {layer_idx}: {', '.join(parts)}")
    
    # Summary
    if logger:
        all_errs = [v["relative_error"] for lr in results.values() for v in lr.values()]
        all_cos = [v["cosine_similarity"] for lr in results.values() for v in lr.values()]
        logger.info(f"  UPPER BOUND mean error: {sum(all_errs)/len(all_errs):.4f}, "
                    f"mean cos: {sum(all_cos)/len(all_cos):.4f}")
    
    return results

def align_band_neurons(
    band_gate_weights: List[torch.Tensor],
    band_up_weights: List[torch.Tensor],
    band_down_weights: List[torch.Tensor],
    signature_mode: str = "down",
    ref_layer_idx: Optional[int] = None,
    logger: Optional['Logger'] = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    """
    Align intermediate neurons across layers in a band via permutation.
    
    GLU MLPs have a permutation symmetry: reordering the d_ff intermediate
    neurons (consistently across gate, up, and down projections) does not
    change the layer's function. This alignment step ensures that "neuron j"
    means the same thing across all layers in a band BEFORE averaging for SVD.
    
    Without alignment, averaging unrelated neurons across layers destroys
    information and produces a poor shared basis.
    
    Args:
        band_gate_weights: List of (d_model, d_ff) tensors, one per layer
        band_up_weights: List of (d_model, d_ff) tensors, one per layer
        band_down_weights: List of (d_ff, d_model) tensors, one per layer
        signature_mode: "down" uses down_proj rows, "gate_up" uses concat of
                        gate/up columns as neuron signatures
        ref_layer_idx: Which layer index (within band) to use as reference.
                       None = middle layer.
        logger: Optional logger
    
    Returns:
        Tuple of (aligned_gate, aligned_up, aligned_down, diagnostics)
        where diagnostics contains alignment quality metrics.
    """
    n_layers = len(band_gate_weights)
    if n_layers <= 1:
        return band_gate_weights, band_up_weights, band_down_weights, {"skipped": True}
    
    device = band_gate_weights[0].device
    d_ff = band_gate_weights[0].shape[1]  # (d_model, d_ff)
    
    # Choose reference layer (middle of band)
    if ref_layer_idx is None:
        ref_layer_idx = n_layers // 2
    
    # Build reference signatures
    ref_sigs = _build_neuron_signatures(
        band_gate_weights[ref_layer_idx],
        band_up_weights[ref_layer_idx],
        band_down_weights[ref_layer_idx],
        mode=signature_mode
    )
    # Normalize reference signatures
    ref_sigs_norm = torch.nn.functional.normalize(ref_sigs, dim=1)  # (d_ff, sig_dim)
    
    aligned_gate = list(band_gate_weights)
    aligned_up = list(band_up_weights)
    aligned_down = list(band_down_weights)
    
    diagnostics = {
        "n_layers": n_layers,
        "ref_layer_idx": ref_layer_idx,
        "signature_mode": signature_mode,
        "per_layer": []
    }
    
    for i in range(n_layers):
        if i == ref_layer_idx:
            diagnostics["per_layer"].append({
                "layer_in_band": i,
                "is_reference": True,
                "mean_match_cosine": 1.0,
                "matches_above_0.9": d_ff,
                "matches_above_0.7": d_ff,
            })
            continue
        
        # Build signatures for this layer
        layer_sigs = _build_neuron_signatures(
            band_gate_weights[i], band_up_weights[i], band_down_weights[i],
            mode=signature_mode
        )
        layer_sigs_norm = torch.nn.functional.normalize(layer_sigs, dim=1)
        
        # Find permutation via greedy cosine matching
        perm, match_cosines = _greedy_neuron_match(ref_sigs_norm, layer_sigs_norm)
        
        # Apply permutation to this layer's weights
        # gate/up are (d_model, d_ff) — permute columns
        # down is (d_ff, d_model) — permute rows
        aligned_gate[i] = band_gate_weights[i][:, perm].contiguous()
        aligned_up[i] = band_up_weights[i][:, perm].contiguous()
        aligned_down[i] = band_down_weights[i][perm, :].contiguous()
        
        # Diagnostics
        mean_cos = match_cosines.mean().item()
        above_09 = (match_cosines > 0.9).sum().item()
        above_07 = (match_cosines > 0.7).sum().item()
        above_05 = (match_cosines > 0.5).sum().item()
        
        layer_diag = {
            "layer_in_band": i,
            "is_reference": False,
            "mean_match_cosine": mean_cos,
            "matches_above_0.9": above_09,
            "matches_above_0.7": above_07,
            "matches_above_0.5": above_05,
        }
        diagnostics["per_layer"].append(layer_diag)
        
        if logger:
            logger.info(
                f"      Layer {i} alignment: mean_cos={mean_cos:.4f}, "
                f">0.9={above_09}/{d_ff}, >0.7={above_07}/{d_ff}, >0.5={above_05}/{d_ff}"
            )
    
    return aligned_gate, aligned_up, aligned_down, diagnostics


def _build_neuron_signatures(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    mode: str = "down"
) -> torch.Tensor:
    """
    Build a signature vector for each intermediate neuron.
    
    Args:
        gate_weight: (d_model, d_ff)
        up_weight: (d_model, d_ff)
        down_weight: (d_ff, d_model)
        mode: "down" = use down_proj rows (how neuron writes to residual)
              "gate_up" = concat gate/up columns (how neuron reads from residual)
    
    Returns:
        (d_ff, signature_dim) tensor of neuron signatures
    """
    if mode == "down":
        # down_weight[j, :] = how neuron j writes back to residual stream
        return down_weight.float()  # (d_ff, d_model)
    elif mode == "gate_up":
        # gate_weight[:, j] and up_weight[:, j] = how neuron j reads from residual
        # Transpose to get (d_ff, d_model) then concat
        gate_sigs = gate_weight.float().T  # (d_ff, d_model)
        up_sigs = up_weight.float().T      # (d_ff, d_model)
        return torch.cat([gate_sigs, up_sigs], dim=1)  # (d_ff, 2*d_model)
    else:
        raise ValueError(f"Unknown signature mode: {mode}")


def _greedy_neuron_match(
    ref_sigs: torch.Tensor,
    layer_sigs: torch.Tensor,
    block_size: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Greedy nearest-neighbor matching of neurons to reference.
    
    Computes cosine similarity in blocks (memory-efficient for 4864×4864),
    then greedily assigns each neuron to its best available match.
    
    Args:
        ref_sigs: (d_ff, sig_dim) normalized reference signatures
        layer_sigs: (d_ff, sig_dim) normalized layer signatures
        block_size: Block size for blocked matmul
    
    Returns:
        (permutation, match_cosines) where permutation[i] = which neuron in
        the current layer maps to reference neuron i, and match_cosines[i] =
        the cosine similarity of that match.
    """
    d_ff = ref_sigs.shape[0]
    device = ref_sigs.device
    
    # Compute full similarity matrix in blocks to avoid OOM
    # sim[i, j] = cosine(ref_neuron_i, layer_neuron_j)
    sim = torch.zeros(d_ff, d_ff, device=device)
    for start in range(0, d_ff, block_size):
        end = min(start + block_size, d_ff)
        sim[start:end] = ref_sigs[start:end] @ layer_sigs.T
    
    # Greedy matching: for each ref neuron (in order of best available match),
    # assign the highest-similarity unmatched layer neuron
    perm = torch.full((d_ff,), -1, dtype=torch.long, device=device)
    match_cosines = torch.zeros(d_ff, device=device)
    used = torch.zeros(d_ff, dtype=torch.bool, device=device)
    
    # Get the best match score for each ref neuron
    best_scores, _ = sim.max(dim=1)
    # Process ref neurons from highest best-score to lowest (greedy priority)
    order = best_scores.argsort(descending=True)
    
    for ref_idx in order:
        # Find best unmatched layer neuron for this ref neuron
        scores = sim[ref_idx].clone()
        scores[used] = -2.0  # Mask already-used neurons
        best_layer_idx = scores.argmax()
        
        perm[ref_idx] = best_layer_idx
        match_cosines[ref_idx] = sim[ref_idx, best_layer_idx]
        used[best_layer_idx] = True
    
    return perm, match_cosines


# =============================================================================
# Main Compression
# =============================================================================

def initialize_glu_banks(
    source_model,
    config: CompressionConfig,
    device: torch.device,
    logger: Optional[Logger] = None
) -> Tuple[GLUPrimitiveBanks, nn.ModuleList, Dict[str, Dict[str, float]]]:
    """
    Initialize GLU primitive banks with complete error tracking.
    
    Supports two modes:
    - Default (softmax + overlapping SVD): Original PILON-R behavior
    - Orthogonal + unconstrained: Non-overlapping SVD + analytical signed weights.
      Required for pretrained model conversion.
    """
    model_info = get_model_info(source_model)
    n_layers = model_info["n_layers"]
    d_model = model_info["d_model"]
    d_ff = model_info["d_ff"]
    
    use_orthogonal = config.svd_non_overlapping
    use_unconstrained = config.composition_mode == "unconstrained"
    
    if logger:
        logger.info(f"Initializing GLU banks: {n_layers} layers, d_model={d_model}, d_ff={d_ff}")
        if use_orthogonal:
            coverage = config.n_primitives * config.rank
            logger.info(f"  Mode: orthogonal SVD (non-overlapping), {config.n_primitives} × rank {config.rank} = {coverage}/{d_model} dims")
        if use_unconstrained:
            logger.info(f"  Composition: unconstrained (signed weights, analytical init)")
        if config.neuron_alignment:
            logger.info(f"  Neuron alignment: ENABLED (signature={config.neuron_alignment_signature})")
    
    bands = create_band_config(n_layers, config.n_bands)
    
    primitive_banks = GLUPrimitiveBanks(
        d_model=d_model, d_ff=d_ff, n_primitives=config.n_primitives,
        rank=config.rank, bands=bands
    ).to(device)
    
    composition_weights = nn.ModuleList([
        GLUCompositionWeights(
            n_primitives=config.n_primitives, top_k=config.top_k, layer_idx=i,
            mode=config.composition_mode
        ).to(device)
        for i in range(n_layers)
    ])
    
    # Select SVD function
    svd_fn = create_orthogonal_primitives_from_svd if use_orthogonal else create_diverse_primitives_from_svd
    
    all_errors: Dict[str, Dict[str, float]] = {}
    
    for band in bands:
        band_name = band["name"]
        band_layers = band["layers"]
        
        if logger:
            logger.info(f"  Initializing band '{band_name}' (layers {band_layers[0]}-{band_layers[-1]})")
        
        # Collect all layer weights in this band
        band_gate_weights = []
        band_up_weights = []
        band_down_weights = []
        for layer_idx in band_layers:
            weights = get_ffn_weights(source_model, layer_idx)
            band_gate_weights.append(weights["gate_proj"].T.to(device).float())
            band_up_weights.append(weights["up_proj"].T.to(device).float())
            band_down_weights.append(weights["down_proj"].T.to(device).float())
        
        # ---- Neuron Permutation Alignment ----
        # Align intermediate neurons across layers BEFORE averaging/SVD.
        # This is a function-preserving reparameterization that ensures
        # "neuron j" means the same thing across all layers in the band.
        alignment_diag = None
        if config.neuron_alignment and len(band_layers) > 1:
            if logger:
                logger.info(f"    Aligning neurons (signature={config.neuron_alignment_signature})...")
            band_gate_weights, band_up_weights, band_down_weights, alignment_diag = align_band_neurons(
                band_gate_weights, band_up_weights, band_down_weights,
                signature_mode=config.neuron_alignment_signature,
                logger=logger
            )
        
        if use_orthogonal:
            # For orthogonal mode, use averaged weights for SVD basis.
            # (Stacked SVD was tested but hurt down_proj reconstruction.)
            if len(band_layers) == 1:
                ref_gate = band_gate_weights[0]
                ref_up = band_up_weights[0]
                ref_down = band_down_weights[0]
            else:
                ref_gate = sum(band_gate_weights) / len(band_layers)
                ref_up = sum(band_up_weights) / len(band_layers)
                ref_down = sum(band_down_weights) / len(band_layers)
            
            gate_primitives = create_orthogonal_primitives_from_svd(
                ref_gate, config.n_primitives, config.rank,
                config.scale_by_singular_values, device)
            up_primitives = create_orthogonal_primitives_from_svd(
                ref_up, config.n_primitives, config.rank,
                config.scale_by_singular_values, device)
            down_primitives = create_orthogonal_primitives_from_svd(
                ref_down, config.n_primitives, config.rank,
                config.scale_by_singular_values, device)
        else:
            if len(band_layers) == 1:
                ref_gate = band_gate_weights[0]
                ref_up = band_up_weights[0]
                ref_down = band_down_weights[0]
            else:
                ref_gate = sum(band_gate_weights) / len(band_layers)
                ref_up = sum(band_up_weights) / len(band_layers)
                ref_down = sum(band_down_weights) / len(band_layers)
            
            gate_primitives = svd_fn(ref_gate, config.n_primitives, config.rank,
                                     config.scale_by_singular_values, device)
            up_primitives = svd_fn(ref_up, config.n_primitives, config.rank,
                                   config.scale_by_singular_values, device)
            down_primitives = svd_fn(ref_down, config.n_primitives, config.rank,
                                     config.scale_by_singular_values, device)
        
        # Assign to banks
        gate_bank = primitive_banks.gate_banks[band_name]
        up_bank = primitive_banks.up_banks[band_name]
        down_bank = primitive_banks.down_banks[band_name]
        
        with torch.no_grad():
            gate_bank.A.copy_(torch.stack([A for A, _ in gate_primitives]))
            gate_bank.B.copy_(torch.stack([B for _, B in gate_primitives]))
            up_bank.A.copy_(torch.stack([A for A, _ in up_primitives]))
            up_bank.B.copy_(torch.stack([B for _, B in up_primitives]))
            down_bank.A.copy_(torch.stack([A for A, _ in down_primitives]))
            down_bank.B.copy_(torch.stack([B for _, B in down_primitives]))
        
        # Initialize composition weights for every layer in band
        # IMPORTANT: We use band_*_weights (which are already aligned if
        # neuron_alignment is enabled) rather than re-extracting from source_model.
        for band_local_idx, layer_idx in enumerate(band["layers"]):
            # Use already-collected (and possibly aligned) weights
            layer_gate = band_gate_weights[band_local_idx]
            layer_up = band_up_weights[band_local_idx]
            layer_down = band_down_weights[band_local_idx]
            
            comp = composition_weights[layer_idx]
            
            if use_unconstrained:
                # Analytical closed-form: w = (G^-1) @ b
                gate_w = compute_analytical_composition_weights(
                    gate_primitives, layer_gate
                )
                up_w = compute_analytical_composition_weights(
                    up_primitives, layer_up
                )
                down_w = compute_analytical_composition_weights(
                    down_primitives, layer_down
                )
                
                with torch.no_grad():
                    comp.gate_weights_raw.copy_(gate_w)
                    comp.up_weights_raw.copy_(up_w)
                    comp.down_weights_raw.copy_(down_w)
                
                # Compute reconstruction errors using raw weights (no top-k for unconstrained)
                gate_error = compute_reconstruction_error(
                    layer_gate, gate_primitives,
                    gate_w, top_k=None, unconstrained=True
                )
                up_error = compute_reconstruction_error(
                    layer_up, up_primitives,
                    up_w, top_k=None, unconstrained=True
                )
                down_error = compute_reconstruction_error(
                    layer_down, down_primitives,
                    down_w, top_k=None, unconstrained=True
                )
            else:
                # Original iterative softmax optimization
                init_top_k = config.top_k if config.composition_init_with_topk else None
                
                gate_logits = optimize_composition_weights(
                    gate_primitives, layer_gate,
                    top_k=init_top_k, n_iterations=config.composition_init_steps, lr=config.composition_init_lr
                )
                up_logits = optimize_composition_weights(
                    up_primitives, layer_up,
                    top_k=init_top_k, n_iterations=config.composition_init_steps, lr=config.composition_init_lr
                )
                down_logits = optimize_composition_weights(
                    down_primitives, layer_down,
                    top_k=init_top_k, n_iterations=config.composition_init_steps, lr=config.composition_init_lr
                )
                
                with torch.no_grad():
                    comp.gate_logits.copy_(gate_logits)
                    comp.up_logits.copy_(up_logits)
                    comp.down_logits.copy_(down_logits)
                
                gate_error = compute_reconstruction_error(
                    layer_gate, gate_primitives,
                    F.softmax(gate_logits, dim=0), top_k=config.top_k
                )
                up_error = compute_reconstruction_error(
                    layer_up, up_primitives,
                    F.softmax(up_logits, dim=0), top_k=config.top_k
                )
                down_error = compute_reconstruction_error(
                    layer_down, down_primitives,
                    F.softmax(down_logits, dim=0), top_k=config.top_k
                )
            
            layer_errors = {"gate": gate_error, "up": up_error, "down": down_error}
            all_errors[f"layer_{layer_idx}"] = layer_errors
            
            if logger:
                logger.info(f"    Layer {layer_idx}: gate_err={gate_error['relative_error']:.4f} (cos={gate_error['cosine_similarity']:.4f}), "
                           f"up_err={up_error['relative_error']:.4f} (cos={up_error['cosine_similarity']:.4f}), "
                           f"down_err={down_error['relative_error']:.4f} (cos={down_error['cosine_similarity']:.4f})")
    
    return primitive_banks, composition_weights, all_errors


def initialize_standard_banks(
    source_model,
    config: CompressionConfig,
    device: torch.device,
    logger: Optional[Logger] = None
) -> Tuple[BandPrimitiveBanks, nn.ModuleList, Dict[str, Dict[str, float]]]:
    """Initialize standard (non-GLU) primitive banks."""
    model_info = get_model_info(source_model)
    n_layers = model_info["n_layers"]
    d_model = model_info["d_model"]
    d_ff = model_info["d_ff"]
    
    bands = create_band_config(n_layers, config.n_bands)
    
    primitive_banks = BandPrimitiveBanks(
        d_model=d_model, d_ff=d_ff, n_primitives=config.n_primitives,
        rank=config.rank, bands=bands, share_fc1_fc2=False
    ).to(device)
    
    composition_weights = nn.ModuleList([
        LayerCompositionWeights(
            n_primitives=config.n_primitives, top_k=config.top_k, layer_idx=i
        ).to(device)
        for i in range(n_layers)
    ])
    
    all_errors = {}
    
    for band in bands:
        band_name = band["name"]
        band_layers = band["layers"]
        
        # Average weight matrices across all layers in band
        avg_fc1 = None
        avg_fc2 = None
        for layer_idx in band_layers:
            weights = get_ffn_weights(source_model, layer_idx)
            fc1_t = weights["fc1"].to(device).float()
            fc2_t = weights["fc2"].to(device).float()
            if avg_fc1 is None:
                avg_fc1 = fc1_t
                avg_fc2 = fc2_t
            else:
                avg_fc1 = avg_fc1 + fc1_t
                avg_fc2 = avg_fc2 + fc2_t
        n_band_layers = len(band_layers)
        avg_fc1 = avg_fc1 / n_band_layers
        avg_fc2 = avg_fc2 / n_band_layers
        
        fc1_primitives = create_diverse_primitives_from_svd(
            avg_fc1, config.n_primitives, config.rank,
            config.scale_by_singular_values, device
        )
        fc2_primitives = create_diverse_primitives_from_svd(
            avg_fc2, config.n_primitives, config.rank,
            config.scale_by_singular_values, device
        )
        
        fc1_bank = primitive_banks.get_fc1_bank(band["layers"][0])
        fc2_bank = primitive_banks.get_fc2_bank(band["layers"][0])
        
        with torch.no_grad():
            fc1_bank.A.copy_(torch.stack([A for A, _ in fc1_primitives]))
            fc1_bank.B.copy_(torch.stack([B for _, B in fc1_primitives]))
            fc2_bank.A.copy_(torch.stack([A for A, _ in fc2_primitives]))
            fc2_bank.B.copy_(torch.stack([B for _, B in fc2_primitives]))
        
        for layer_idx in band["layers"]:
            layer_weights = get_ffn_weights(source_model, layer_idx)
            comp = composition_weights[layer_idx]
            
            init_top_k = config.top_k if config.composition_init_with_topk else None
            
            fc1_logits = optimize_composition_weights(
                fc1_primitives, layer_weights["fc1"].to(device),
                top_k=init_top_k, n_iterations=config.composition_init_steps
            )
            fc2_logits = optimize_composition_weights(
                fc2_primitives, layer_weights["fc2"].to(device),
                top_k=init_top_k, n_iterations=config.composition_init_steps
            )
            
            with torch.no_grad():
                comp.fc1_logits.copy_(fc1_logits)
                comp.fc2_logits.copy_(fc2_logits)
            
            fc1_error = compute_reconstruction_error(
                layer_weights["fc1"].to(device), fc1_primitives,
                F.softmax(fc1_logits, dim=0), top_k=config.top_k
            )
            fc2_error = compute_reconstruction_error(
                layer_weights["fc2"].to(device), fc2_primitives,
                F.softmax(fc2_logits, dim=0), top_k=config.top_k
            )
            
            all_errors[f"layer_{layer_idx}"] = {"fc1": fc1_error, "fc2": fc2_error}
    
    return primitive_banks, composition_weights, all_errors


# =============================================================================
# Compressed Model Wrapper
# =============================================================================

class CompressedModel(nn.Module):
    """Compressed model with PILON primitive banks."""
    
    def __init__(
        self,
        source_model,
        primitive_banks: Union[GLUPrimitiveBanks, BandPrimitiveBanks],
        composition_weights: nn.ModuleList,
        config: CompressionConfig
    ):
        super().__init__()
        
        self.source_model = source_model
        self.primitive_banks = primitive_banks
        self.composition_weights = composition_weights
        self.compression_config = config
        self.config = source_model.config
        self.model_info = get_model_info(source_model)
        self._dense_mlps: List[nn.Module] = []
        
        self._replace_ffn_layers()
    
    def _replace_ffn_layers(self):
        """Replace FFN layers with compositional versions."""
        n_layers = self.model_info["n_layers"]
        
        if self.model_info["is_glu"]:
            for layer_idx in range(n_layers):
                layer = self.source_model.model.layers[layer_idx]
                self._dense_mlps.append(layer.mlp)
                layer.mlp = CompositionalGLUMLP(
                    layer_idx=layer_idx,
                    primitive_banks=self.primitive_banks,
                    composition_weights=self.composition_weights[layer_idx],
                    top_k=self.compression_config.top_k
                )
        else:
            for layer_idx in range(n_layers):
                layer = self.source_model.transformer.h[layer_idx]
                self._dense_mlps.append(layer.mlp)
                layer.mlp = CompositionalStandardMLP(
                    layer_idx=layer_idx,
                    primitive_banks=self.primitive_banks,
                    composition_weights=self.composition_weights[layer_idx],
                    top_k=self.compression_config.top_k
                )

    def get_dense_mlps(self) -> List[nn.Module]:
        return self._dense_mlps
    
    def forward(self, *args, **kwargs):
        return self.source_model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.source_model.generate(*args, **kwargs)
    
    def get_all_entropy(self) -> Dict[str, float]:
        entropy = {}
        for i, comp in enumerate(self.composition_weights):
            for key, value in comp.compute_entropy().items():
                entropy[f"layer_{i}_{key}"] = value
        return entropy
    
    def get_ffn_param_count(self) -> int:
        return (sum(p.numel() for p in self.primitive_banks.parameters()) +
                sum(p.numel() for p in self.composition_weights.parameters()))
    
    def set_runtime_top_k(self, top_k: int):
        """Set runtime top-k for all layers."""
        n_layers = self.model_info["n_layers"]
        if self.model_info["is_glu"]:
            for layer in self.source_model.model.layers:
                layer.mlp.runtime_top_k = top_k
        else:
            for layer in self.source_model.transformer.h:
                layer.mlp.runtime_top_k = top_k


# =============================================================================
# Save/Load (Reloadable Format)
# =============================================================================

def save_compressed_model(
    model: CompressedModel,
    result: CompressionResult,
    config: CompressionConfig,
    output_dir: Path,
    logger: Optional[Logger] = None
):
    """
    Save compressed model in a reloadable format.
    
    Saves:
    - primitive_banks.pt: State dict of primitive banks
    - composition_weights.pt: State dict of composition weights  
    - compression_config.json: Full compression config
    - compression_result.json: Compression results/metrics
    - model_info.json: Source model info needed for rebuilding
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save state dicts
    torch.save(model.primitive_banks.state_dict(), output_dir / "primitive_banks.pt")
    torch.save({i: comp.state_dict() for i, comp in enumerate(model.composition_weights)},
               output_dir / "composition_weights.pt")
    
    # Save configs
    with open(output_dir / "compression_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    with open(output_dir / "compression_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    
    with open(output_dir / "model_info.json", "w") as f:
        json.dump(model.model_info, f, indent=2)
    
    if logger:
        logger.info(f"Saved compressed model to {output_dir}")


def load_compressed_model(
    output_dir: Path,
    device: str = "cuda",
    logger: Optional[Logger] = None
) -> Tuple[CompressedModel, CompressionConfig, CompressionResult]:
    """
    Load a previously saved compressed model.
    """
    from transformers import AutoModelForCausalLM
    
    output_dir = Path(output_dir)
    
    # Load configs
    with open(output_dir / "compression_config.json") as f:
        config = CompressionConfig.from_dict(json.load(f))
    
    with open(output_dir / "model_info.json") as f:
        model_info = json.load(f)
    
    with open(output_dir / "compression_result.json") as f:
        result_dict = json.load(f)
        result = CompressionResult(**{k: v for k, v in result_dict.items() 
                                      if k in CompressionResult.__dataclass_fields__})
    
    if logger:
        logger.info(f"Loading source model: {config.source_model}")
    
    # Load source model (single device)
    source_model = AutoModelForCausalLM.from_pretrained(
        config.source_model,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True
    ).to(device)
    
    # Recreate primitive banks
    bands = create_band_config(model_info["n_layers"], config.n_bands)
    
    if model_info["is_glu"]:
        primitive_banks = GLUPrimitiveBanks(
            d_model=model_info["d_model"],
            d_ff=model_info["d_ff"],
            n_primitives=config.n_primitives,
            rank=config.rank,
            bands=bands
        ).to(device)
        
        composition_weights = nn.ModuleList([
            GLUCompositionWeights(
                n_primitives=config.n_primitives,
                top_k=config.top_k,
                layer_idx=i,
                mode=config.composition_mode
            ).to(device)
            for i in range(model_info["n_layers"])
        ])
    else:
        primitive_banks = BandPrimitiveBanks(
            d_model=model_info["d_model"],
            d_ff=model_info["d_ff"],
            n_primitives=config.n_primitives,
            rank=config.rank,
            bands=bands,
            share_fc1_fc2=False
        ).to(device)
        
        composition_weights = nn.ModuleList([
            LayerCompositionWeights(
                n_primitives=config.n_primitives,
                top_k=config.top_k,
                layer_idx=i
            ).to(device)
            for i in range(model_info["n_layers"])
        ])
    
    # Load state dicts
    primitive_banks.load_state_dict(torch.load(output_dir / "primitive_banks.pt", map_location=device))
    
    comp_states = torch.load(output_dir / "composition_weights.pt", map_location=device)
    for i, comp in enumerate(composition_weights):
        comp.load_state_dict(comp_states[i])

    # Match dtypes with source model
    model_dtype = next(source_model.parameters()).dtype
    primitive_banks = primitive_banks.to(device=device, dtype=model_dtype)
    composition_weights = composition_weights.to(device=device, dtype=model_dtype)
    
    # Create compressed model
    compressed_model = CompressedModel(
        source_model=source_model,
        primitive_banks=primitive_banks,
        composition_weights=composition_weights,
        config=config
    )
    
    if logger:
        logger.info(f"Loaded compressed model from {output_dir}")
    
    return compressed_model, config, result


# =============================================================================
# Main Compression Function
# =============================================================================

def compress_model(
    source_model_name: str,
    config: CompressionConfig,
    logger: Optional[Logger] = None
) -> Tuple[CompressedModel, CompressionResult]:
    """Main compression function."""
    from transformers import AutoModelForCausalLM
    
    device = torch.device(config.device)
    
    # Check memory
    estimated_mem = estimate_model_memory(source_model_name)
    available_mem = get_available_gpu_memory()
    
    if logger:
        logger.info(f"Loading source model: {source_model_name}")
        logger.info(f"  Estimated memory: {estimated_mem:.1f} GB")
        logger.info(f"  Available GPU memory: {available_mem:.1f} GB")
    
    if estimated_mem > available_mem * 0.8:
        if logger:
            logger.warning(f"Model may not fit in GPU memory! Consider using a smaller model.")
    
    # Load on single device (no device_map="auto")
    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        torch_dtype=torch.float16,
        device_map=None,  # IMPORTANT: single device
        trust_remote_code=True
    ).to(device)
    
    source_params = sum(p.numel() for p in source_model.parameters())
    source_ffn_params = count_ffn_params(source_model)
    model_info = get_model_info(source_model)
    
    if logger:
        logger.info(f"Source model loaded:")
        logger.info(f"  Total params: {source_params:,}")
        logger.info(f"  FFN params: {source_ffn_params:,} ({100*source_ffn_params/source_params:.1f}%)")
        logger.info(f"  Model type: {model_info['model_type']}, GLU: {model_info['is_glu']}")
    
    # Initialize banks
    if model_info["is_glu"]:
        primitive_banks, composition_weights, errors = initialize_glu_banks(
            source_model, config, device, logger
        )
    else:
        primitive_banks, composition_weights, errors = initialize_standard_banks(
            source_model, config, device, logger
        )
    
    # Create compressed model
    compressed_model = CompressedModel(
        source_model=source_model,
        primitive_banks=primitive_banks,
        composition_weights=composition_weights,
        config=config
    )
    
    # Compute stats
    compressed_ffn_params = compressed_model.get_ffn_param_count()
    ffn_compression_ratio = compressed_ffn_params / source_ffn_params
    
    # Compute mean errors
    gate_errors = [e["gate"]["relative_error"] for e in errors.values() if "gate" in e]
    up_errors = [e["up"]["relative_error"] for e in errors.values() if "up" in e]
    down_errors = [e["down"]["relative_error"] for e in errors.values() if "down" in e]
    fc1_errors = [e["fc1"]["relative_error"] for e in errors.values() if "fc1" in e]
    fc2_errors = [e["fc2"]["relative_error"] for e in errors.values() if "fc2" in e]
    
    result = CompressionResult(
        source_model=source_model_name,
        source_params=source_params,
        source_ffn_params=source_ffn_params,
        compressed_ffn_params=compressed_ffn_params,
        ffn_compression_ratio=ffn_compression_ratio,
        reconstruction_errors=errors,
        mean_gate_error=sum(gate_errors) / len(gate_errors) if gate_errors else sum(fc1_errors) / len(fc1_errors) if fc1_errors else 0,
        mean_up_error=sum(up_errors) / len(up_errors) if up_errors else 0,
        mean_down_error=sum(down_errors) / len(down_errors) if down_errors else sum(fc2_errors) / len(fc2_errors) if fc2_errors else 0,
    )
    
    if logger:
        logger.info(f"\nCompression complete:")
        logger.info(f"  Source FFN params: {source_ffn_params:,}")
        logger.info(f"  Compressed FFN params: {compressed_ffn_params:,}")
        logger.info(f"  FFN compression ratio: {ffn_compression_ratio:.3f}")
        if gate_errors:
            logger.info(f"  Mean gate error: {result.mean_gate_error:.4f}")
            logger.info(f"  Mean up error: {result.mean_up_error:.4f}")
            logger.info(f"  Mean down error: {result.mean_down_error:.4f}")
    
    return compressed_model, result


# =============================================================================
# Fine-tuning
# =============================================================================

def finetune_compressed_model(
    model: CompressedModel,
    config: CompressionConfig,
    logger: Optional[Logger] = None
) -> Dict[str, List]:
    """Fine-tune compressed model."""
    if config.finetune_mode == "ffn-distill":
        return finetune_compressed_model_ffn_distill(model, config, logger)
    elif config.finetune_mode == "lm-loss":
        return finetune_compressed_model_lm_loss(model, config, logger)
    raise ValueError(f"Unsupported finetune_mode: {config.finetune_mode}. Use 'ffn-distill' or 'lm-loss'.")


def _get_model_layers(model: CompressedModel) -> List[Any]:
    if model.model_info["is_glu"]:
        return list(model.source_model.model.layers)
    return list(model.source_model.transformer.h)


def _build_hidden_state_cache(
    model: CompressedModel,
    tokenizer,
    device: torch.device,
    max_seq_len: int,
    batch_size: int,
    n_batches: int = 8,
    samples_per_batch: int = 512
) -> List[torch.Tensor]:
    """
    Build hidden state cache from real dataset samples for FFN distillation.
    
    Uses actual text data for diverse coverage of the input distribution
    each FFN layer sees during inference.
    """
    layers = _get_model_layers(model)
    n_layers = len(layers)
    cache: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    dense_mlps = model.get_dense_mlps()
    if len(dense_mlps) != n_layers:
        raise ValueError("Dense MLP cache does not match layer count.")
    # Swap in dense MLPs for cache build to avoid compositional compute
    original_mlps = [layer.mlp for layer in layers]
    for i, layer in enumerate(layers):
        layer.mlp = dense_mlps[i]

    # Load real calibration data from dataset
    calibration_texts = _load_calibration_texts(
        model.compression_config.finetune_dataset,
        tokenizer,
        n_samples=n_batches * batch_size * 2,  # Extra samples to ensure coverage
        max_seq_len=max_seq_len
    )

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_texts = calibration_texts[start:end]
        if len(batch_texts) < batch_size:
            # Wrap around if we run out
            batch_texts = batch_texts + calibration_texts[:batch_size - len(batch_texts)]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        ffn_inputs: List[Optional[torch.Tensor]] = [None for _ in range(n_layers)]
        hooks = []

        def make_hook(idx: int):
            def hook(module, inputs, output):
                ffn_inputs[idx] = inputs[0].detach()
            return hook

        for i, layer in enumerate(layers):
            hooks.append(layer.mlp.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            model.source_model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()

        for i, h in enumerate(ffn_inputs):
            if h is None:
                continue
            flat = h.reshape(-1, h.shape[-1])
            if flat.size(0) > samples_per_batch:
                idx = torch.randperm(flat.size(0), device=flat.device)[:samples_per_batch]
                flat = flat.index_select(0, idx)
            cache[i].append(flat.to(dtype=torch.float16))

    stacked: List[torch.Tensor] = []
    for i in range(n_layers):
        if len(cache[i]) == 0:
            stacked.append(torch.empty(0, model.model_info["d_model"], device=device))
        else:
            stacked.append(torch.cat(cache[i], dim=0))

    # Restore compositional MLPs
    for i, layer in enumerate(layers):
        layer.mlp = original_mlps[i]

    return stacked


def _load_calibration_texts(
    dataset_name: str,
    tokenizer,
    n_samples: int = 64,
    max_seq_len: int = 256
) -> List[str]:
    """
    Load diverse text samples for calibration.
    
    Falls back to synthetic prompts if dataset loading fails.
    """
    try:
        from datasets import load_dataset
        
        ds = load_dataset(dataset_name, split="train", streaming=True)
        texts = []
        for item in ds:
            # Try common text field names
            text = None
            for key in ["text", "content", "passage", "document"]:
                if key in item and isinstance(item[key], str) and len(item[key]) > 50:
                    text = item[key]
                    break
            if text is None:
                # Take the first string field that looks like text
                for key, val in item.items():
                    if isinstance(val, str) and len(val) > 50:
                        text = val
                        break
            if text is not None:
                texts.append(text[:max_seq_len * 4])  # Rough char limit
            if len(texts) >= n_samples:
                break
        
        if len(texts) >= n_samples // 2:
            return texts
    except Exception:
        pass
    
    # Fallback: diverse synthetic prompts covering different domains
    fallback_prompts = [
        "The history of computing began with the development of mechanical calculators in the 17th century. Charles Babbage designed the first programmable computer in the 1830s.",
        "In molecular biology, DNA replication is the process by which a double-stranded DNA molecule is copied to produce two identical copies.",
        "The global economy experienced significant shifts during the 2020s, with supply chain disruptions, inflation concerns, and the rapid adoption of digital currencies.",
        "Machine learning algorithms can be broadly categorized into supervised learning, unsupervised learning, and reinforcement learning approaches.",
        "The Renaissance period in Europe, spanning from the 14th to 17th century, marked a cultural rebirth characterized by advances in art, science, and literature.",
        "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to process information in fundamentally different ways.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human activities since the Industrial Revolution.",
        "The principles of object-oriented programming include encapsulation, inheritance, polymorphism, and abstraction, forming the foundation of modern software design.",
        "Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy stored in glucose.",
        "The development of transformer architectures revolutionized natural language processing, enabling models to capture long-range dependencies in text sequences.",
        "Constitutional law governs the interpretation and implementation of a nation's constitution, defining the powers of government and the rights of citizens.",
        "In astrophysics, black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape from within the event horizon.",
        "The stock market operates through a network of exchanges where buyers and sellers trade shares of publicly listed companies at prices determined by supply and demand.",
        "Neural networks consist of interconnected layers of artificial neurons that process information through weighted connections and activation functions.",
        "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful pathogens.",
        "Software architecture refers to the fundamental structural choices made in designing a software system, including the decomposition into components and their interactions.",
    ]
    # Repeat to fill n_samples
    result = []
    while len(result) < n_samples:
        result.extend(fallback_prompts)
    return result[:n_samples]


def finetune_compressed_model_ffn_distill(
    model: CompressedModel,
    config: CompressionConfig,
    logger: Optional[Logger] = None
) -> Dict[str, List]:
    """FFN-only distillation fine-tune (no LM loss)."""
    from transformers import AutoTokenizer
    device = next(model.parameters()).device

    tokenizer = AutoTokenizer.from_pretrained(config.source_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = _get_model_layers(model)
    n_layers = len(layers)
    dense_mlps = model.get_dense_mlps()
    if len(dense_mlps) != n_layers:
        raise ValueError("Dense MLP cache does not match layer count.")

    # Freeze everything
    for param in model.source_model.parameters():
        param.requires_grad = False
    for dense in dense_mlps:
        for param in dense.parameters():
            param.requires_grad = False

    # Train only PILON FFN params
    for param in model.primitive_banks.parameters():
        param.requires_grad = config.train_primitives
    for param in model.composition_weights.parameters():
        param.requires_grad = config.train_compositions

    train_params = []
    if config.train_primitives:
        train_params.append({
            "params": list(model.primitive_banks.parameters()),
            "lr": config.finetune_lr * 2.0,
            "name": "primitives"
        })
    if config.train_compositions:
        train_params.append({
            "params": list(model.composition_weights.parameters()),
            "lr": config.finetune_lr * 0.5,
            "weight_decay": 0.0,
            "name": "compositions"
        })

    assert any(p.requires_grad for g in train_params for p in g["params"]), (
        "No trainable PILON FFN parameters found."
    )

    optimizer = AdamW(train_params, weight_decay=0.01)

    # Set fixed runtime top-k
    model.set_runtime_top_k(config.top_k)

    # Build hidden-state cache (no streaming datasets)
    model.source_model.eval()
    model.primitive_banks.train()
    model.composition_weights.train()

    cache = _build_hidden_state_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_seq_len=min(config.finetune_max_seq_len, 256),
        batch_size=config.finetune_batch_size,
        n_batches=8,
        samples_per_batch=512
    )

    history = {"ffn_mse_loss": [], "cos_sim": [], "step": []}
    start_time = time.time()

    if logger:
        logger.info(f"Starting fine-tuning (ffn-distill): {config.finetune_steps} steps")
        logger.info(f"  Top-k fixed: {config.top_k}")

    layers_per_step = min(8, n_layers)

    for step in range(1, config.finetune_steps + 1):
        step_start = time.time()
        optimizer.zero_grad()

        total_loss = 0.0
        total_cos = 0.0
        if n_layers <= layers_per_step:
            step_layers = list(range(n_layers))
        else:
            step_layers = torch.randperm(n_layers)[:layers_per_step].tolist()

        for i in step_layers:
            layer = layers[i]
            if cache[i].numel() == 0:
                continue
            samples = cache[i]
            n_samples = samples.size(0)
            take = min(256, n_samples)
            idx = torch.randint(0, n_samples, (take,), device=samples.device)
            h = samples.index_select(0, idx).to(device=device, dtype=torch.float32)
            h = h.unsqueeze(1)

            with torch.no_grad():
                dense_dtype = next(dense_mlps[i].parameters()).dtype
                y_dense = dense_mlps[i](h.to(dtype=dense_dtype))
                y_dense = y_dense.to(dtype=torch.float32)
            pilon_dtype = next(model.primitive_banks.parameters()).dtype
            y_pilon = layer.mlp(h.to(dtype=pilon_dtype))
            y_pilon = y_pilon.to(dtype=torch.float32)

            mse = F.mse_loss(y_pilon, y_dense)
            cos = F.cosine_similarity(
                y_pilon.reshape(-1, y_pilon.shape[-1]),
                y_dense.reshape(-1, y_dense.shape[-1]),
                dim=-1
            ).mean()

            total_loss = total_loss + mse
            total_cos = total_cos + cos

        if len(step_layers) == 0:
            raise ValueError("No layers found for distillation.")

        loss = total_loss / len(step_layers)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for g in train_params for p in g["params"] if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        step_time = time.time() - step_start
        if step_time > 5.0:
            raise RuntimeError(f"FFN distill step time too slow: {step_time:.2f}s")

        history["ffn_mse_loss"].append(loss.item())
        history["cos_sim"].append((total_cos / len(step_layers)).item())
        history["step"].append(step)

        if logger and (step == 1 or step % 100 == 0):
            # Grad norms
            def grad_norm(params):
                grads = [p.grad for p in params if p.grad is not None]
                if not grads:
                    return 0.0
                return torch.stack([g.norm(2) for g in grads]).mean().item()

            prim_gn = grad_norm(model.primitive_banks.parameters())
            comp_gn = grad_norm(model.composition_weights.parameters())
            elapsed = time.time() - start_time

            logger.info(
                f"Step {step}/{config.finetune_steps} | "
                f"ffn_mse_loss={loss.item():.6f} | cos_sim={(total_cos / len(step_layers)).item():.4f} | "
                f"grad_norm_primitives={prim_gn:.4f} | grad_norm_composition={comp_gn:.4f} | "
                f"step_time_seconds={step_time:.2f} | {step/elapsed:.1f} steps/s"
            )

    return history


# =============================================================================
# End-to-End LM Loss Fine-tuning
# =============================================================================

def finetune_compressed_model_lm_loss(
    model: CompressedModel,
    config: CompressionConfig,
    logger: Optional[Logger] = None
) -> Dict[str, List]:
    """
    End-to-end LM loss fine-tuning.
    
    Backpropagates through the full model using next-token prediction loss.
    More powerful than FFN-distill but slower and uses more memory.
    Recommended after ffn-distill as a second refinement pass.
    """
    from transformers import AutoTokenizer
    
    device = next(model.parameters()).device
    
    tokenizer = AutoTokenizer.from_pretrained(config.source_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_text_dataset(
        config.finetune_dataset, tokenizer, max_seq_len=config.finetune_max_seq_len,
        max_tokens=config.finetune_steps * config.finetune_batch_size * config.finetune_max_seq_len * 2,
        streaming=True
    )
    dataloader = create_dataloader(dataset, batch_size=config.finetune_batch_size, shuffle=False, num_workers=0)
    
    # Freeze non-PILON parameters
    for param in model.source_model.parameters():
        param.requires_grad = False
    
    # Enable PILON parameters
    param_groups = []
    if config.train_primitives:
        for param in model.primitive_banks.parameters():
            param.requires_grad = True
        param_groups.append({
            'params': list(model.primitive_banks.parameters()),
            'lr': config.finetune_lr * 2.0,
            'name': 'primitives'
        })
    
    if config.train_compositions:
        for param in model.composition_weights.parameters():
            param.requires_grad = True
        param_groups.append({
            'params': list(model.composition_weights.parameters()),
            'lr': config.finetune_lr * 0.5,
            'weight_decay': 0.0,
            'name': 'compositions'
        })
    
    assert any(p.requires_grad for g in param_groups for p in g["params"]), (
        "No trainable PILON parameters found."
    )
    
    optimizer = AdamW(param_groups, weight_decay=0.01)
    
    # Phase training
    phase1_steps = int(config.finetune_steps * config.phase1_ratio) if config.use_phase_training else 0
    topk_anneal_start = int(config.finetune_steps * config.topk_anneal_start_ratio)
    
    # Precision
    amp_dtype = None
    if config.precision == "bf16" and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif config.precision == "fp16" and torch.cuda.is_available():
        amp_dtype = torch.float16
    
    autocast_ctx = autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
    
    model.source_model.eval()  # Keep non-PILON layers in eval mode (dropout off)
    model.primitive_banks.train()
    model.composition_weights.train()
    
    history = {"loss": [], "ppl": [], "top_k": [], "step": []}
    
    step = 0
    data_iter = iter(dataloader)
    start_time = time.time()
    
    if logger:
        logger.info(f"Starting LM-loss fine-tuning: {config.finetune_steps} steps")
        if config.use_phase_training:
            logger.info(f"  Phase 1 (primitives): 0-{phase1_steps}")
            logger.info(f"  Phase 2 (compositions): {phase1_steps}+")
        if config.topk_anneal:
            logger.info(f"  Top-k annealing: {config.n_primitives} -> {config.top_k}")
    
    while step < config.finetune_steps:
        step_start = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)
        
        # Phase training control
        if config.use_phase_training:
            if step < phase1_steps:
                for param in model.composition_weights.parameters():
                    param.requires_grad = False
                for param in model.primitive_banks.parameters():
                    param.requires_grad = config.train_primitives
            else:
                for param in model.composition_weights.parameters():
                    param.requires_grad = config.train_compositions
                for param in model.primitive_banks.parameters():
                    param.requires_grad = not config.freeze_primitives_phase2
        
        # Top-k annealing: LINEAR from n_primitives to top_k
        if config.topk_anneal and step >= topk_anneal_start:
            progress = (step - topk_anneal_start) / max(1, config.finetune_steps - topk_anneal_start)
            current_top_k = round(config.n_primitives - progress * (config.n_primitives - config.top_k))
            current_top_k = max(config.top_k, min(config.n_primitives, current_top_k))
            model.set_runtime_top_k(current_top_k)
        else:
            current_top_k = config.n_primitives
            model.set_runtime_top_k(current_top_k)
        
        optimizer.zero_grad()
        
        with autocast_ctx:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in param_groups for p in g["params"] if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        
        step += 1
        loss_val = loss.item()
        history["loss"].append(loss_val)
        history["ppl"].append(math.exp(min(loss_val, 100)))
        history["top_k"].append(current_top_k)
        history["step"].append(step)
        
        if step == 1 or step % 100 == 0:
            step_time = time.time() - step_start
            elapsed = time.time() - start_time
            phase_str = f"P{1 if step < phase1_steps else 2}" if config.use_phase_training else ""
            
            if logger:
                logger.info(
                    f"Step {step}/{config.finetune_steps} {phase_str} | "
                    f"Loss: {loss_val:.4f} | PPL: {math.exp(min(loss_val, 100)):.2f} | "
                    f"Top-k: {current_top_k} | {step/elapsed:.1f} steps/s | "
                    f"step={step_time:.2f}s"
                )
    
    # Set final top_k
    model.set_runtime_top_k(config.top_k)
    
    if logger:
        final_loss = sum(history["loss"][-100:]) / min(100, len(history["loss"]))
        logger.info(f"LM-loss fine-tuning complete! Final loss: {final_loss:.4f}, PPL: {math.exp(min(final_loss, 100)):.2f}")
    
    return history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_wikitext2_ppl(
    model,
    tokenizer,
    device: str = "cuda",
    max_seq_len: int = 512,
    stride: int = 256,
    logger: Optional[Logger] = None
) -> float:
    """
    Evaluate perplexity on WikiText-2 test set.
    
    Uses standard sliding window approach matching lm-eval-harness methodology.
    This is the primary quality metric for compression benchmarking.
    
    Args:
        model: HuggingFace model or CompressedModel
        tokenizer: Tokenizer
        device: Device string
        max_seq_len: Context window for evaluation
        stride: Sliding window stride
        logger: Optional logger
    
    Returns:
        WikiText-2 test set perplexity
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        if logger:
            logger.warning(f"Could not load WikiText-2: {e}. Skipping.")
        return -1.0
    
    # Concatenate all text
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    
    seq_len = input_ids.size(1)
    if logger:
        logger.info(f"WikiText-2 test: {seq_len} tokens, stride={stride}, max_seq_len={max_seq_len}")
    
    nlls = []
    n_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for begin in range(0, seq_len - 1, stride):
            end = min(begin + max_seq_len, seq_len)
            input_chunk = input_ids[:, begin:end]
            target_chunk = input_chunk.clone()
            
            # Only compute loss on the new tokens (not the context overlap)
            if begin > 0:
                overlap = end - begin - stride
                if overlap > 0:
                    target_chunk[:, :overlap] = -100
            
            outputs = model(input_ids=input_chunk, labels=target_chunk)
            
            # Handle both HF model output and dict output
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                continue
            
            # Count non-masked tokens
            n_valid = (target_chunk != -100).sum().item()
            if n_valid > 0:
                nlls.append(loss.item() * n_valid)
                n_tokens += n_valid
            
            if end >= seq_len:
                break
    
    if n_tokens == 0:
        if logger:
            logger.warning("No tokens evaluated for WikiText-2")
        return -1.0
    
    avg_nll = sum(nlls) / n_tokens
    ppl = math.exp(min(avg_nll, 100))
    
    if logger:
        logger.info(f"WikiText-2 PPL: {ppl:.2f} (avg NLL: {avg_nll:.4f}, {n_tokens} tokens)")
    
    return ppl


def evaluate_compression(
    original_model,
    compressed_model: CompressedModel,
    tokenizer,
    config: CompressionConfig,
    logger: Optional[Logger] = None
) -> Dict[str, float]:
    """Evaluate compression quality."""
    device = next(compressed_model.parameters()).device
    
    eval_dataset = load_text_dataset(
        config.finetune_dataset, tokenizer, max_seq_len=config.finetune_max_seq_len,
        max_tokens=50000, streaming=True
    )
    eval_loader = create_dataloader(eval_dataset, batch_size=config.finetune_batch_size, shuffle=False, num_workers=0)
    
    amp_dtype = None
    if config.precision == "bf16" and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    autocast_ctx = autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
    
    def compute_ppl(model, name: str) -> float:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                if n_batches >= 100:
                    break
                
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = attention_mask.to(device)
                
                with autocast_ctx:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                n_tokens = (labels != -100).sum().item()
                if n_tokens == 0:
                    n_tokens = labels.numel()
                
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens
                n_batches += 1
        
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 100))
        
        if logger:
            logger.info(f"{name}: loss={avg_loss:.4f}, ppl={ppl:.2f}")
        
        return ppl
    
    original_ppl = compute_ppl(original_model, "Original")
    compressed_ppl = compute_ppl(compressed_model, "Compressed")
    
    results = {
        "original_ppl": original_ppl,
        "compressed_ppl": compressed_ppl,
        "ppl_ratio": compressed_ppl / max(original_ppl, 1e-8),
        "ppl_increase_pct": (compressed_ppl / max(original_ppl, 1e-8) - 1) * 100,
    }
    
    # WikiText-2 evaluation (primary benchmark metric)
    if logger:
        logger.info("\nWikiText-2 Evaluation:")
    
    wt2_original = evaluate_wikitext2_ppl(
        original_model, tokenizer, device=str(device),
        max_seq_len=config.finetune_max_seq_len, logger=logger
    )
    wt2_compressed = evaluate_wikitext2_ppl(
        compressed_model, tokenizer, device=str(device),
        max_seq_len=config.finetune_max_seq_len, logger=logger
    )
    
    results["wikitext2_original_ppl"] = wt2_original
    results["wikitext2_compressed_ppl"] = wt2_compressed
    if wt2_original > 0 and wt2_compressed > 0:
        results["wikitext2_ppl_ratio"] = wt2_compressed / wt2_original
        results["wikitext2_ppl_increase_pct"] = (wt2_compressed / wt2_original - 1) * 100
        if logger:
            logger.info(f"WikiText-2 PPL ratio: {results['wikitext2_ppl_ratio']:.3f} "
                       f"({results['wikitext2_ppl_increase_pct']:+.1f}%)")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PILON-R Model Compression v3")
    
    parser.add_argument("model", type=str, nargs="?", help="HuggingFace model name")
    parser.add_argument("--load", type=str, help="Load compressed model from directory")
    parser.add_argument("--output", type=str, default="compressed_model")
    
    # Compression level (overrides individual settings)
    parser.add_argument("--compression-level", type=str, choices=get_all_compression_levels(),
                        help=f"Compression level preset: {', '.join(get_all_compression_levels())}")
    
    # Individual compression settings (used if --compression-level not specified)
    parser.add_argument("--n-primitives", type=int, default=32)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--n-bands", type=int, default=4)
    
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--finetune-steps", type=int, default=5000)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-dataset", type=str, default="Elriggs/openwebtext-100k")
    parser.add_argument("--finetune-mode", type=str, default="ffn-distill",
                        choices=["ffn-distill", "lm-loss"],
                        help="Fine-tune mode: ffn-distill (FFN-only distillation, fast) or lm-loss (end-to-end LM loss, better quality)")
    
    # Composition architecture (for pretrained conversion)
    parser.add_argument("--composition-mode", type=str, default="softmax",
                        choices=["softmax", "unconstrained"],
                        help="Composition weight mode: softmax (original) or unconstrained (signed weights for pretrained conversion)")
    parser.add_argument("--svd-non-overlapping", action="store_true",
                        help="Use non-overlapping orthogonal SVD slices (required for good pretrained conversion)")
    parser.add_argument("--neuron-alignment", action="store_true",
                        help="Align intermediate neurons across band layers before SVD (function-preserving reparameterization)")
    parser.add_argument("--neuron-alignment-signature", type=str, default="down",
                        choices=["down", "gate_up"],
                        help="Neuron signature mode: 'down' (residual write vector) or 'gate_up' (residual read vectors)")
    parser.add_argument("--compute-upper-bound", action="store_true",
                        help="Compute per-layer SVD upper bound (best possible error without sharing) and exit")
    
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16")
    
    # List available compression levels
    parser.add_argument("--list-levels", action="store_true", 
                        help="List available compression levels and exit")
    
    args = parser.parse_args()
    
    # Handle --list-levels
    if args.list_levels:
        print("\nAvailable compression levels:")
        print("-" * 50)
        for level in COMPRESSION_LEVELS:
            print(f"  {level['label']:12} n_primitives={level['n_primitives']:2}, "
                  f"rank={level['rank']:2}, top_k={level['top_k']:2}")
        print()
        return
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = Logger(Path(args.output) / "compress.log")
    
    logger.info("=" * 60)
    logger.info("PILON-R Model Compression v3")
    logger.info("=" * 60)
    
    # Handle --compute-upper-bound
    if getattr(args, 'compute_upper_bound', False):
        if not args.model:
            parser.error("Model name required for --compute-upper-bound")
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading model: {args.model}")
        source = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=None, trust_remote_code=True
        ).to(args.device)
        compute_per_layer_svd_upper_bound(source, args.rank, torch.device(args.device), logger)
        return
    
    if args.load:
        # Load existing compressed model
        compressed_model, config, result = load_compressed_model(
            Path(args.load), args.device, logger
        )
    else:
        if not args.model:
            parser.error("Model name required unless using --load")
        
        # Apply compression level if specified
        if args.compression_level:
            level_config = get_compression_config(args.compression_level)
            n_primitives = level_config["n_primitives"]
            rank = level_config["rank"]
            top_k = level_config["top_k"]
            logger.info(f"Using compression level: {args.compression_level}")
            logger.info(f"  n_primitives={n_primitives}, rank={rank}, top_k={top_k}")
        else:
            n_primitives = args.n_primitives
            rank = args.rank
            top_k = args.top_k
        
        config = CompressionConfig(
            source_model=args.model,
            compression_level=args.compression_level,
            n_primitives=n_primitives,
            rank=rank,
            top_k=top_k,
            n_bands=args.n_bands,
            finetune_steps=args.finetune_steps,
            finetune_lr=args.finetune_lr,
            finetune_dataset=args.finetune_dataset,
            finetune_mode=args.finetune_mode,
            composition_mode=args.composition_mode,
            svd_non_overlapping=args.svd_non_overlapping,
            neuron_alignment=args.neuron_alignment,
            neuron_alignment_signature=args.neuron_alignment_signature,
            output_dir=args.output,
            device=args.device,
            precision=args.precision,
        )
        
        logger.info(f"Source: {args.model}")
        logger.info(f"Config: n_primitives={config.n_primitives}, rank={config.rank}, top_k={config.top_k}")
        if config.composition_mode == "unconstrained":
            logger.info(f"  Composition: unconstrained, SVD: {'orthogonal' if config.svd_non_overlapping else 'overlapping'}")
        
        compressed_model, result = compress_model(args.model, config, logger)
    
    if args.finetune and not args.eval_only:
        logger.info("\n" + "=" * 60)
        logger.info("Fine-tuning")
        logger.info("=" * 60)
        finetune_compressed_model(compressed_model, config, logger)
    
    if args.eval_only or args.finetune:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation")
        logger.info("=" * 60)
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(config.source_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        original_model = AutoModelForCausalLM.from_pretrained(
            config.source_model, torch_dtype=torch.float16, device_map=None, trust_remote_code=True
        ).to(args.device)
        
        eval_results = evaluate_compression(original_model, compressed_model, tokenizer, config, logger)
        
        result.source_ppl = eval_results["original_ppl"]
        result.compressed_ppl = eval_results["compressed_ppl"]
        result.ppl_ratio = eval_results["ppl_ratio"]
        
        logger.info(f"\nResults (OpenWebText):")
        logger.info(f"  Original PPL: {result.source_ppl:.2f}")
        logger.info(f"  Compressed PPL: {result.compressed_ppl:.2f}")
        logger.info(f"  PPL Ratio: {result.ppl_ratio:.3f}")
        
        if "wikitext2_original_ppl" in eval_results and eval_results["wikitext2_original_ppl"] > 0:
            logger.info(f"\nResults (WikiText-2):")
            logger.info(f"  Original PPL: {eval_results['wikitext2_original_ppl']:.2f}")
            logger.info(f"  Compressed PPL: {eval_results['wikitext2_compressed_ppl']:.2f}")
            if "wikitext2_ppl_ratio" in eval_results:
                logger.info(f"  PPL Ratio: {eval_results['wikitext2_ppl_ratio']:.3f} "
                           f"({eval_results['wikitext2_ppl_increase_pct']:+.1f}%)")
    
    if not args.load:
        save_compressed_model(compressed_model, result, config, Path(args.output), logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
