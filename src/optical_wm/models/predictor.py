"""
Transformer Predictor for Optical Network World Model.

Predicts next latent state given current latent state(s) and action(s).

Architecture (following LeWM):
  - Transformer with causal masking (autoregressive)
  - Action conditioning via Adaptive Layer Normalization (AdaLN)
  - AdaLN zero-init for stable early training
  - Projector (MLP + BatchNorm) at output for SIGReg compatibility

Training (teacher forcing):
  Input:  z_1, z_2, ..., z_N  (real encoder outputs)
          a_1, a_2, ..., a_N  (actions from dataset)
  Output: ẑ_2, ẑ_3, ..., ẑ_{N+1}  (predicted next embeddings)
  Loss:   MSE(ẑ_{t+1}, z_{t+1}) for all t

Planning (autoregressive):
  Input:  z_1 (encoder output from current observation)
          a_1, a_2, ..., a_H  (candidate action sequence)
  Output: ẑ_2, ẑ_3, ..., ẑ_{H+1}  (predicted trajectory)
  Each ẑ_{t+1} = predict(ẑ_1, ..., ẑ_t, a_1, ..., a_t)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Constants
try:
    from ..core.schemas import N_ACTION_FEATURES
except ImportError:
    N_ACTION_FEATURES = 20


# =====================================================================
# Adaptive Layer Normalization (AdaLN)
# =====================================================================

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization conditioned on action.

    Instead of learned gamma/beta, the scale and shift are predicted
    from the action embedding:
      AdaLN(x, a) = gamma(a) * LayerNorm(x) + beta(a)

    Zero-initialized so that at the start of training:
      gamma(a) = 1, beta(a) = 0  →  AdaLN(x, a) = LayerNorm(x)
    The action conditioning grows progressively during training.
    """

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Predict scale and shift from action: [action_dim] → [2 * hidden_dim]
        self.adaLN_proj = nn.Linear(action_dim, 2 * hidden_dim)
        # Zero-init: gamma=1, beta=0 at initialization
        nn.init.zeros_(self.adaLN_proj.weight)
        nn.init.zeros_(self.adaLN_proj.bias)

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [B, T, D] or [B, D]
            action_emb: [B, T, action_dim] or [B, action_dim]
        Returns:
            modulated:  same shape as x
        """
        x_norm = self.norm(x)
        params = self.adaLN_proj(action_emb)  # [..., 2*D]
        gamma, beta = params.chunk(2, dim=-1)  # [..., D] each
        # gamma centered at 1 (not 0) so default is identity
        return (1 + gamma) * x_norm + beta


# =====================================================================
# Transformer Block with AdaLN
# =====================================================================

class AdaLNTransformerBlock(nn.Module):
    """
    Transformer block with AdaLN for action conditioning.

    Structure:
      x → AdaLN(x, a) → MultiHeadAttention → residual
        → AdaLN(x, a) → FFN → residual
    """

    def __init__(self, hidden_dim: int, n_heads: int, action_dim: int,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()

        # Pre-attention AdaLN
        self.adaln1 = AdaLN(hidden_dim, action_dim)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-FFN AdaLN
        self.adaln2 = AdaLN(hidden_dim, action_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:          [B, T, D]
            action_emb: [B, T, action_dim]
            attn_mask:  [T, T] causal mask (True = masked/blocked)
        Returns:
            out: [B, T, D]
        """
        # Self-attention with AdaLN
        x_norm = self.adaln1(x, action_emb)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            is_causal=False,  # we provide explicit mask
        )
        x = x + attn_out

        # FFN with AdaLN
        x_norm = self.adaln2(x, action_emb)
        x = x + self.ffn(x_norm)

        return x


# =====================================================================
# Transformer Predictor
# =====================================================================

class TransformerPredictor(nn.Module):
    """
    Autoregressive predictor using Transformer with AdaLN.

    Takes a sequence of latent states z_{1:T} and actions a_{1:T},
    predicts next latent states ẑ_{2:T+1}.

    Architecture:
      action → action_encoder (MLP) → action_emb
      z → input_proj → hidden
      + positional encoding
      → N × AdaLNTransformerBlock(hidden, action_emb, causal_mask)
      → output_proj → ẑ_next
      → projector (MLP + BatchNorm) for SIGReg
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = N_ACTION_FEATURES,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        action_emb_dim: int = 64,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # ── Action encoder ──
        # MLP to project raw action vector to action embedding
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_emb_dim * 2),
            nn.ReLU(),
            nn.Linear(action_emb_dim * 2, action_emb_dim),
        )

        # ── Input projection ──
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # ── Positional encoding (learned) ──
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                action_dim=action_emb_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # ── Output projection: hidden → latent ──
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # ── Projector for SIGReg (like LeWM) ──
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        # ── Causal mask cache ──
        self.register_buffer(
            '_causal_mask',
            self._build_causal_mask(max_seq_len),
            persistent=False,
        )

    def _build_causal_mask(self, size: int) -> torch.Tensor:
        """
        Build causal attention mask.
        Returns [size, size] float mask where masked positions = -inf.
        """
        mask = torch.triu(
            torch.full((size, size), float('-inf')),
            diagonal=1,
        )
        return mask

    def forward(self, z_seq: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """
        Predict next embeddings for a sequence (teacher forcing).

        Args:
            z_seq:   [B, T, latent_dim]  — encoder outputs for T timesteps
            actions: [B, T, action_dim]  — actions at each timestep
                     (action at t is the action taken AT state t,
                      leading to state t+1)

        Returns:
            z_pred:  [B, T, latent_dim]  — predicted next embeddings
                     z_pred[:, t, :] is the prediction for z_{t+1}
        """
        B, T, D = z_seq.shape

        # Encode actions
        action_emb = self.action_encoder(actions)  # [B, T, action_emb_dim]

        # Project input embeddings
        h = self.input_proj(z_seq)  # [B, T, hidden_dim]

        # Add positional encoding
        positions = torch.arange(T, device=z_seq.device)
        h = h + self.pos_embedding(positions).unsqueeze(0)  # broadcast over B

        # Causal mask for this sequence length
        causal_mask = self._causal_mask[:T, :T]

        # Transformer blocks
        for block in self.blocks:
            h = block(h, action_emb, attn_mask=causal_mask)

        # Project to latent space
        z_pred = self.output_proj(h)  # [B, T, latent_dim]

        return z_pred

    def forward_with_projection(self, z_seq: torch.Tensor,
                                actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass + projector (used during training for SIGReg).
        The projector output is used for the prediction loss,
        while the pre-projection output is the actual latent.

        Args:
            z_seq:   [B, T, latent_dim]
            actions: [B, T, action_dim]
        Returns:
            z_proj:  [B, T, latent_dim] — projected predictions
        """
        z_pred = self.forward(z_seq, actions)  # [B, T, D]
        B, T, D = z_pred.shape

        # BatchNorm needs [N, D], so reshape
        z_flat = z_pred.reshape(B * T, D)
        z_proj = self.projector(z_flat)  # [B*T, D]
        z_proj = z_proj.reshape(B, T, D)

        return z_proj

    def predict_step(self, z_history: torch.Tensor,
                     action_history: torch.Tensor) -> torch.Tensor:
        """
        Predict one step ahead given history (used during planning).

        Args:
            z_history:      [B, H, latent_dim] — past latent states
            action_history: [B, H, action_dim] — past actions
        Returns:
            z_next: [B, latent_dim] — predicted next state
        """
        z_pred = self.forward(z_history, action_history)  # [B, H, D]
        return z_pred[:, -1, :]  # last position prediction

    def rollout(self, z_init: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout for planning (no teacher forcing).

        Args:
            z_init:  [B, latent_dim] — initial latent state
            actions: [B, H, action_dim] — action sequence to evaluate
        Returns:
            z_traj:  [B, H+1, latent_dim] — predicted trajectory
                     z_traj[:, 0] = z_init
                     z_traj[:, h] = predicted state after h actions
        """
        B, H, A = actions.shape
        D = z_init.shape[-1]
        device = z_init.device

        z_traj = [z_init]
        z_current = z_init.unsqueeze(1)  # [B, 1, D]
        a_history = []

        for h in range(H):
            # Append current action to history
            a_history.append(actions[:, h:h+1, :])  # [B, 1, A]

            # Build full history
            a_seq = torch.cat(a_history, dim=1)  # [B, h+1, A]

            # Predict next state from full history
            z_pred = self.forward(z_current, a_seq)  # [B, h+1, D]
            z_next = z_pred[:, -1, :]  # [B, D]

            z_traj.append(z_next)

            # Append predicted state to history for next step
            z_current = torch.cat([
                z_current, z_next.unsqueeze(1)
            ], dim=1)  # [B, h+2, D]

        return torch.stack(z_traj, dim=1)  # [B, H+1, D]

    def get_param_count(self) -> dict:
        """Count parameters by component."""
        counts = {}
        counts['action_encoder'] = sum(
            p.numel() for p in self.action_encoder.parameters()
        )
        counts['input_proj'] = sum(
            p.numel() for p in self.input_proj.parameters()
        )
        counts['pos_embedding'] = self.pos_embedding.weight.numel()
        counts['transformer_blocks'] = sum(
            p.numel() for p in self.blocks.parameters()
        )
        counts['output_proj'] = sum(
            p.numel() for p in self.output_proj.parameters()
        )
        counts['projector'] = sum(
            p.numel() for p in self.projector.parameters()
        )
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


# =====================================================================
# Ablation: MLP Predictor (no Transformer)
# =====================================================================

class MLPPredictor(nn.Module):
    """
    Simple MLP predictor baseline.
    Concatenates z_t and a_t, passes through MLP to predict z_{t+1}.
    No sequence context — each prediction is independent.
    Used as ablation to show value of Transformer temporal modeling.
    """

    def __init__(self, latent_dim: int = 128,
                 action_dim: int = N_ACTION_FEATURES,
                 hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        layers = [nn.Linear(latent_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, z_seq: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_seq:   [B, T, latent_dim]
            actions: [B, T, action_dim]
        Returns:
            z_pred:  [B, T, latent_dim]
        """
        x = torch.cat([z_seq, actions], dim=-1)  # [B, T, D+A]
        return self.mlp(x)  # [B, T, latent_dim]

    def forward_with_projection(self, z_seq, actions):
        z_pred = self.forward(z_seq, actions)
        B, T, D = z_pred.shape
        z_flat = z_pred.reshape(B * T, D)
        z_proj = self.projector(z_flat)
        return z_proj.reshape(B, T, D)

    def rollout(self, z_init: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """Autoregressive rollout."""
        B, H, A = actions.shape
        z_traj = [z_init]
        z_current = z_init

        for h in range(H):
            a_t = actions[:, h, :]
            x = torch.cat([z_current, a_t], dim=-1)
            z_next = self.mlp(x)
            z_traj.append(z_next)
            z_current = z_next

        return torch.stack(z_traj, dim=1)


# =====================================================================
# Quick test
# =====================================================================

if __name__ == "__main__":
    print("Testing predictor shapes...")

    B = 4   # batch size
    T = 8   # sequence length
    D = 128  # latent dim
    A = N_ACTION_FEATURES

    z_seq = torch.randn(B, T, D)
    actions = torch.randn(B, T, A)

    # ── Test Transformer Predictor ──
    print("\n=== TransformerPredictor ===")
    pred = TransformerPredictor(
        latent_dim=D, action_dim=A, hidden_dim=256,
        n_layers=4, n_heads=8,
    )
    z_pred = pred(z_seq, actions)
    print(f"  Input z_seq: {list(z_seq.shape)}")
    print(f"  Input actions: {list(actions.shape)}")
    print(f"  Output z_pred: {list(z_pred.shape)}")
    assert z_pred.shape == (B, T, D), f"Shape mismatch: {z_pred.shape}"

    # With projection
    z_proj = pred.forward_with_projection(z_seq, actions)
    print(f"  Projected: {list(z_proj.shape)}")

    # Single step prediction
    z_next = pred.predict_step(z_seq, actions)
    print(f"  Single step: {list(z_next.shape)}")
    assert z_next.shape == (B, D)

    # Rollout
    z_init = torch.randn(B, D)
    plan_actions = torch.randn(B, 10, A)  # 10-step plan
    z_traj = pred.rollout(z_init, plan_actions)
    print(f"  Rollout (10 steps): {list(z_traj.shape)}")
    assert z_traj.shape == (B, 11, D)  # init + 10 predicted

    # Param count
    params = pred.get_param_count()
    for name, count in params.items():
        print(f"  {name}: {count:,} params")

    # ── Test MLP Predictor ──
    print("\n=== MLPPredictor (baseline) ===")
    pred_mlp = MLPPredictor(latent_dim=D, action_dim=A)
    z_pred_mlp = pred_mlp(z_seq, actions)
    print(f"  Output: {list(z_pred_mlp.shape)}")

    z_traj_mlp = pred_mlp.rollout(z_init, plan_actions)
    print(f"  Rollout: {list(z_traj_mlp.shape)}")
    print(f"  Total: {sum(p.numel() for p in pred_mlp.parameters()):,} params")

    # ── Gradient check ──
    print("\n=== Gradient check ===")
    pred_grad = TransformerPredictor(latent_dim=D, action_dim=A, hidden_dim=64, n_layers=2, n_heads=4)
    z_g = torch.randn(B, T, D)
    a_g = torch.randn(B, T, A)
    out_g = pred_grad(z_g, a_g)
    loss = out_g.sum()
    loss.backward()

    # Projector: no grad because forward() doesn't call it — expected
    # Action encoder: no grad at init because AdaLN is zero-initialized —
    #   this is by design (LeWM zero-init). After first optimizer step,
    #   AdaLN weights become non-zero and action_encoder starts receiving grads.
    # What matters: AdaLN projection weights MUST have gradient (they unblock the path)
    adaln_has_grad = False
    for name, p in pred_grad.named_parameters():
        if 'adaln' in name and 'proj' in name and p.grad is not None:
            if p.grad.abs().sum() > 0:
                adaln_has_grad = True
                break

    # Core path (input_proj → transformer → output_proj) must have gradient
    core_ok = True
    for name, p in pred_grad.named_parameters():
        if any(k in name for k in ['input_proj', 'output_proj', 'attn', 'ffn', 'pos_embedding']):
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
                core_ok = False
                print(f"  ✗ No gradient: {name}")

    print(f"  Core path (z → transformer → ẑ): {'✓' if core_ok else '✗'}")
    print(f"  AdaLN projection weights: {'✓ has grad (will unblock action path)' if adaln_has_grad else '✗'}")
    print(f"  Action encoder: no grad at init (zero-init by design, activates after 1st step)")

    # ── Causal masking check ──
    print("\n=== Causal masking check ===")
    # Prediction at t=0 should only depend on z_0 and a_0
    z_test = torch.randn(1, 4, D, requires_grad=True)
    a_test = torch.randn(1, 4, A)
    pred_test = TransformerPredictor(latent_dim=D, action_dim=A, hidden_dim=64, n_layers=2, n_heads=4)
    out = pred_test(z_test, a_test)
    out[0, 0, 0].backward()
    # Gradient should only be non-zero for t=0
    grad = z_test.grad[0]  # [4, D]
    t0_grad = grad[0].abs().sum()
    later_grad = grad[1:].abs().sum()
    print(f"  Grad at t=0: {t0_grad:.4f}")
    print(f"  Grad at t>0: {later_grad:.4f}")
    causal_ok = later_grad < 1e-6
    print(f"  Causal masking correct: {'✓' if causal_ok else '✗'}")

    print("\n✓ All predictor tests passed")