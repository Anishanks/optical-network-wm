"""
JEPA World Model for Optical Networks.

Assembles encoder + predictor with training loss.

Loss = prediction_loss + λ * anti_collapse_loss

Prediction loss:
  MSE between predicted next-embeddings and real next-embeddings
  (teacher forcing during training)

Anti-collapse (two options):
  - 'variance': simple variance regularization (default, simple)
    Penalizes embedding dimensions with variance < 1.0 across batch.
  - 'sigreg': SIGReg from LeWM (for paper results)
    Forces embeddings to match isotropic Gaussian via random projections.

Usage:
  model = OpticalWorldModel(latent_dim=128)
  loss, metrics = model.compute_loss(batch)
  loss.backward()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

try:
    from .encoder import OpticalNetworkEncoder, MLPEncoder
    from .predictor import TransformerPredictor, MLPPredictor
except ImportError:
    from encoder import OpticalNetworkEncoder, MLPEncoder
    from predictor import TransformerPredictor, MLPPredictor


# =====================================================================
# Anti-collapse regularizers
# =====================================================================

def variance_regularization(z: torch.Tensor, target_var: float = 1.0) -> torch.Tensor:
    """
    Simple variance regularization (from VICReg, simplified).

    Penalizes embedding dimensions whose variance across the batch
    falls below target_var. Prevents collapse to a constant vector.

    Args:
        z: [N, D] embeddings (flattened across batch and time)
        target_var: target variance per dimension
    Returns:
        penalty: scalar, 0 when all dims have var >= target_var
    """
    var = z.var(dim=0)  # [D] variance per dimension
    penalty = F.relu(target_var - var).mean()  # penalize only if var < target
    return penalty


def sigreg_regularization(z: torch.Tensor, n_projections: int = 1024) -> torch.Tensor:
    """
    SIGReg: Sketched Isotropic Gaussian Regularizer (from LeWM/LeJEPA).

    Projects embeddings onto random directions and tests normality
    using the Epps-Pulley statistic on each 1D projection.
    By Cramér-Wold theorem, matching all 1D marginals = matching joint.

    Args:
        z: [N, D] embeddings
        n_projections: number of random directions M
    Returns:
        loss: scalar, lower = closer to isotropic Gaussian
    """
    N, D = z.shape
    if N < 4:
        return torch.tensor(0.0, device=z.device)

    # Standardize
    z_centered = z - z.mean(dim=0, keepdim=True)
    z_std = z_centered.std(dim=0, keepdim=True).clamp(min=1e-6)
    z_norm = z_centered / z_std

    # Random projections: [D, M]
    directions = torch.randn(D, n_projections, device=z.device)
    directions = F.normalize(directions, dim=0)

    # Project: [N, M]
    projections = z_norm @ directions

    # Epps-Pulley test statistic per projection
    # EP(h) = (2/N) Σ_i exp(-h_i²/2) - (1/N²) Σ_{i,j} exp(-(h_i-h_j)²/2) - sqrt(2)
    # Simplified: use moment-based approximation
    # For a standard normal: E[h²] = 1, E[h⁴] = 3
    m2 = (projections ** 2).mean(dim=0)      # [M] should be ~1
    m4 = (projections ** 4).mean(dim=0)      # [M] should be ~3

    # Penalize deviation from Gaussian moments
    loss_m2 = (m2 - 1.0).pow(2).mean()
    loss_m4 = (m4 - 3.0).pow(2).mean()

    return loss_m2 + 0.1 * loss_m4


# =====================================================================
# World Model
# =====================================================================

class OpticalWorldModel(nn.Module):
    """
    JEPA World Model = Encoder + Predictor + Loss.

    Training loop:
      1. Sample sub-trajectory (obs_{1:T}, actions_{1:T-1})
      2. Encode all observations: z_{1:T} = encoder(obs_{1:T})
      3. Predict next embeddings: ẑ_{2:T} = predictor(z_{1:T-1}, a_{1:T-1})
      4. Loss = MSE(ẑ_{2:T}, z_{2:T}) + λ * anti_collapse(z_{1:T})
    """

    def __init__(
        self,
        # Encoder config
        latent_dim: int = 128,
        spectral_dim: int = 64,
        node_hidden_dim: int = 128,
        n_gnn_layers: int = 3,
        n_gnn_heads: int = 4,
        lp_summary_dim: int = 32,
        use_spectral_conv: bool = True,
        # Predictor config
        pred_hidden_dim: int = 256,
        n_pred_layers: int = 4,
        n_pred_heads: int = 8,
        action_emb_dim: int = 64,
        pred_dropout: float = 0.1,
        # Loss config
        collapse_method: str = 'variance',  # 'variance' or 'sigreg'
        collapse_weight: float = 0.1,
        sigreg_projections: int = 1024,
        # Encoder type
        encoder_type: str = 'gnn',  # 'gnn' or 'mlp'
        # Predictor type
        predictor_type: str = 'transformer',  # 'transformer' or 'mlp'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.collapse_method = collapse_method
        self.collapse_weight = collapse_weight
        self.sigreg_projections = sigreg_projections

        # ── Encoder ──
        if encoder_type == 'gnn':
            self.encoder = OpticalNetworkEncoder(
                latent_dim=latent_dim,
                spectral_dim=spectral_dim,
                node_hidden_dim=node_hidden_dim,
                n_gnn_layers=n_gnn_layers,
                n_heads=n_gnn_heads,
                lp_summary_dim=lp_summary_dim,
                use_spectral_conv=use_spectral_conv,
            )
        elif encoder_type == 'mlp':
            self.encoder = MLPEncoder(latent_dim=latent_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # ── Predictor ──
        if predictor_type == 'transformer':
            self.predictor = TransformerPredictor(
                latent_dim=latent_dim,
                hidden_dim=pred_hidden_dim,
                n_layers=n_pred_layers,
                n_heads=n_pred_heads,
                action_emb_dim=action_emb_dim,
                dropout=pred_dropout,
            )
        elif predictor_type == 'mlp':
            self.predictor = MLPPredictor(
                latent_dim=latent_dim,
            )
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a single timestep.

        Args:
            batch: state dict with all feature tensors [B, ...]
        Returns:
            z: [B, latent_dim]
        """
        return self.encoder(batch)

    def encode_sequence(self, batch: Dict[str, torch.Tensor],
                        seq_len: int) -> torch.Tensor:
        """
        Encode a sequence of timesteps.

        The batch contains tensors with shape [B, T, ...].
        We encode each timestep independently.

        Args:
            batch: state dict with [B, T, ...] tensors
            seq_len: T
        Returns:
            z_seq: [B, T, latent_dim]
        """
        B = next(iter(batch.values())).shape[0]
        T = seq_len

        z_list = []
        for t in range(T):
            # Extract single timestep
            step_batch = {}
            for key, val in batch.items():
                if val.dim() >= 2 and val.shape[1] == T:
                    step_batch[key] = val[:, t]
                else:
                    # Static features (topology) — same for all timesteps
                    step_batch[key] = val
            z_t = self.encoder(step_batch)  # [B, D]
            z_list.append(z_t)

        return torch.stack(z_list, dim=1)  # [B, T, D]

    def predict(self, z_seq: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """
        Predict next embeddings (teacher forcing).

        Args:
            z_seq:   [B, T, latent_dim]
            actions: [B, T, action_dim]
        Returns:
            z_pred: [B, T, latent_dim]
        """
        return self.predictor(z_seq, actions)

    def compute_loss(
        self,
        z_seq: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute JEPA training loss from pre-encoded sequence.

        This is the core training step. The caller is responsible for
        encoding the observations into z_seq first.

        Args:
            z_seq:   [B, T, latent_dim] — encoder outputs (all timesteps)
            actions: [B, T-1, action_dim] — actions between timesteps

        Returns:
            total_loss: scalar
            metrics: dict with loss components for logging
        """
        B, T, D = z_seq.shape

        # ── Prediction loss (teacher forcing) ──
        # Input to predictor: z_{1:T-1} and a_{1:T-1}
        # Target: z_{2:T}
        z_input = z_seq[:, :-1, :]     # [B, T-1, D]
        z_target = z_seq[:, 1:, :]     # [B, T-1, D]

        z_pred = self.predictor(z_input, actions)  # [B, T-1, D]

        pred_loss = F.mse_loss(z_pred, z_target)

        # ── Anti-collapse regularization ──
        # Applied on ALL embeddings in the batch (flatten B and T)
        z_flat = z_seq.reshape(B * T, D)  # [B*T, D]

        if self.collapse_method == 'variance':
            collapse_loss = variance_regularization(z_flat)
        elif self.collapse_method == 'sigreg':
            collapse_loss = sigreg_regularization(
                z_flat, n_projections=self.sigreg_projections
            )
        else:
            collapse_loss = torch.tensor(0.0, device=z_seq.device)

        # ── Total loss ──
        total_loss = pred_loss + self.collapse_weight * collapse_loss

        # ── Metrics for logging ──
        with torch.no_grad():
            z_var = z_flat.var(dim=0).mean()
            z_std = z_flat.std(dim=0).mean()
            pred_mse_per_step = F.mse_loss(
                z_pred, z_target, reduction='none'
            ).mean(dim=-1).mean(dim=0)  # would be [T-1] but we already mean'd

        metrics = {
            'loss/total': total_loss.item(),
            'loss/prediction': pred_loss.item(),
            'loss/collapse': collapse_loss.item(),
            'embedding/variance': z_var.item(),
            'embedding/std': z_std.item(),
            'embedding/mean_abs': z_flat.abs().mean().item(),
        }

        return total_loss, metrics

    def compute_loss_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        End-to-end loss computation from raw batch.

        Convenience method that encodes then computes loss.

        Args:
            batch: dict with [B, T, ...] state tensors + [B, T-1, action_dim] actions
            seq_len: T (number of timesteps in the sequence)
        Returns:
            total_loss, metrics
        """
        # Separate actions from state
        actions = batch.pop('actions')  # [B, T-1, action_dim]

        # Encode all timesteps
        z_seq = self.encode_sequence(batch, seq_len)  # [B, T, D]

        # Compute loss
        return self.compute_loss(z_seq, actions)

    def get_param_count(self) -> dict:
        """Parameter count breakdown."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        pred_params = sum(p.numel() for p in self.predictor.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            'encoder': enc_params,
            'predictor': pred_params,
            'total': total,
        }


# =====================================================================
# Quick test
# =====================================================================

if __name__ == "__main__":
    print("Testing world model...")

    # Use hardcoded constants for standalone test
    try:
        from encoder import MAX_NODES, MAX_LINKS, MAX_SLOTS, MAX_LIGHTPATHS
        from encoder import N_LINK_STATIC_FEATURES, N_NODE_FEATURES
        from encoder import N_LP_FEATURES, N_GLOBAL_FEATURES
    except ImportError:
        MAX_NODES, MAX_LINKS, MAX_SLOTS, MAX_LIGHTPATHS = 20, 40, 80, 160
        N_LINK_STATIC_FEATURES, N_NODE_FEATURES = 3, 5
        N_LP_FEATURES, N_GLOBAL_FEATURES = 20, 8

    from predictor import N_ACTION_FEATURES

    B = 4   # batch
    T = 6   # timesteps

    # Create fake sequential batch [B, T, ...]
    batch = {
        'spectral_occupancy': torch.zeros(B, T, MAX_LINKS, MAX_SLOTS),
        'channel_gsnr': torch.randn(B, T, MAX_LINKS, MAX_SLOTS) * 5 + 20,
        'channel_power': torch.randn(B, T, MAX_LINKS, MAX_SLOTS),
        'channel_ase': torch.randn(B, T, MAX_LINKS, MAX_SLOTS) * 0.1,
        'channel_nli': torch.randn(B, T, MAX_LINKS, MAX_SLOTS) * 0.01,
        'node_features': torch.randn(B, MAX_NODES, N_NODE_FEATURES),  # static
        'link_static': torch.randn(B, MAX_LINKS, N_LINK_STATIC_FEATURES),  # static
        'link_endpoints': torch.zeros(B, MAX_LINKS, 2, dtype=torch.long),  # static
        'link_mask': torch.zeros(B, MAX_LINKS, dtype=torch.bool),  # static
        'node_mask': torch.zeros(B, MAX_NODES, dtype=torch.bool),  # static
        'lp_features': torch.randn(B, T, MAX_LIGHTPATHS, N_LP_FEATURES),
        'lp_mask': torch.zeros(B, T, MAX_LIGHTPATHS, dtype=torch.bool),
        'global_features': torch.randn(B, T, N_GLOBAL_FEATURES),
        'actions': torch.randn(B, T - 1, N_ACTION_FEATURES),
    }

    # Set masks
    n_nodes, n_links, n_lps = 8, 11, 20
    batch['node_mask'][:, :n_nodes] = True
    batch['link_mask'][:, :n_links] = True
    batch['lp_mask'][:, :, :n_lps] = True
    for i in range(n_links):
        batch['link_endpoints'][:, i, 0] = i % n_nodes
        batch['link_endpoints'][:, i, 1] = (i + 1) % n_nodes

    # ── Test 1: Variance regularization ──
    print("\n=== World Model (variance reg) ===")
    model = OpticalWorldModel(
        latent_dim=128,
        collapse_method='variance',
        collapse_weight=0.1,
    )

    actions = batch.pop('actions')
    z_seq = model.encode_sequence(batch, T)
    print(f"  Encoded: {list(z_seq.shape)}")

    loss, metrics = model.compute_loss(z_seq, actions)
    print(f"  Loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # Verify backward works
    loss.backward()
    enc_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters() if p.requires_grad
    )
    pred_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.predictor.parameters() if p.requires_grad
    )
    print(f"  Encoder receives gradient: {'✓' if enc_grad else '✗'}")
    print(f"  Predictor receives gradient: {'✓' if pred_grad else '✗'}")

    # Param counts
    params = model.get_param_count()
    for k, v in params.items():
        print(f"  {k}: {v:,} params")

    # ── Test 2: SIGReg ──
    print("\n=== World Model (SIGReg) ===")
    model_sig = OpticalWorldModel(
        latent_dim=128,
        collapse_method='sigreg',
        collapse_weight=0.1,
    )
    model_sig.zero_grad()
    z_seq2 = model_sig.encode_sequence(batch, T)
    loss2, metrics2 = model_sig.compute_loss(z_seq2, actions)
    print(f"  Loss: {loss2.item():.4f}")
    print(f"  Collapse (SIGReg): {metrics2['loss/collapse']:.4f}")

    # ── Test 3: MLP baseline ──
    print("\n=== World Model (MLP encoder + MLP predictor) ===")
    model_mlp = OpticalWorldModel(
        latent_dim=128,
        encoder_type='mlp',
        predictor_type='mlp',
        collapse_method='variance',
    )
    z_seq3 = model_mlp.encode_sequence(batch, T)
    loss3, metrics3 = model_mlp.compute_loss(z_seq3, actions)
    print(f"  Loss: {loss3.item():.4f}")
    params3 = model_mlp.get_param_count()
    print(f"  Total: {params3['total']:,} params")

    # ── Test 4: Collapse detection ──
    print("\n=== Collapse detection ===")
    z_collapsed = torch.ones(B * T, 128) * 5.0  # constant embedding
    var_loss = variance_regularization(z_collapsed)
    print(f"  Collapsed embeddings → variance penalty: {var_loss.item():.4f}")
    assert var_loss.item() > 0.5, "Should detect collapse"

    z_healthy = torch.randn(B * T, 128)  # diverse embeddings
    var_loss_h = variance_regularization(z_healthy)
    print(f"  Healthy embeddings → variance penalty: {var_loss_h.item():.4f}")
    assert var_loss_h.item() < var_loss.item(), "Healthy should have less penalty than collapsed"
    print("  ✓ Collapse detection works")

    print("\n✓ All world model tests passed")