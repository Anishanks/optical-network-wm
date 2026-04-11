"""
Training configuration for Optical Network World Model.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainConfig:
    """All hyperparameters in one place."""

    # ── Data ──
    data_dir: str = "data"
    context_length: int = 8          # sub-trajectory length T
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True

    # ── Encoder ──
    latent_dim: int = 128
    spectral_dim: int = 64
    node_hidden_dim: int = 128
    n_gnn_layers: int = 3
    n_gnn_heads: int = 4
    lp_summary_dim: int = 32
    use_spectral_conv: bool = True
    encoder_type: str = "gnn"        # 'gnn' or 'mlp'

    # ── Predictor ──
    pred_hidden_dim: int = 256
    n_pred_layers: int = 4
    n_pred_heads: int = 8
    action_emb_dim: int = 64
    pred_dropout: float = 0.1
    predictor_type: str = "transformer"  # 'transformer' or 'mlp'

    # ── Loss ──
    collapse_method: str = "variance"    # 'variance' or 'sigreg'
    collapse_weight: float = 0.1
    sigreg_projections: int = 1024

    # ── Optimizer ──
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0

    # ── Schedule ──
    n_epochs: int = 50
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    # ── Logging ──
    log_every: int = 20              # steps
    eval_every: int = 1              # epochs
    save_every: int = 5              # epochs
    output_dir: str = "checkpoints"
    run_name: str = "default"

    # ── Device ──
    device: str = "auto"             # 'auto', 'cuda', 'cpu'

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device