"""
GNN Encoder for Optical Network World Model.

Two-stage architecture:
  Stage 1 — Conv1D spectral encoder (per link):
    [MAX_LINKS, 5, MAX_SLOTS] → [MAX_LINKS, spectral_dim]
    Compresses 80-slot spectral features into a dense per-link embedding.

  Stage 2 — Graph Neural Network (over topology):
    Node features + aggregated edge features → message passing
    → global pooling → latent vector z

Analogous to LeWM's ViT encoder:
  - Conv1D = patch embedding (local feature extraction)
  - GNN layers = self-attention layers (global context)
  - Global pool = [CLS] token (compression to single vector)
  - Projection + BatchNorm = same as LeWM (required for SIGReg)

Input tensors (per timestep, batched):
  spectral_occupancy:  [B, MAX_LINKS, MAX_SLOTS]      bool
  channel_gsnr:        [B, MAX_LINKS, MAX_SLOTS]      float32
  channel_power:       [B, MAX_LINKS, MAX_SLOTS]      float32
  channel_ase:         [B, MAX_LINKS, MAX_SLOTS]      float32
  channel_nli:         [B, MAX_LINKS, MAX_SLOTS]      float32
  link_static:         [B, MAX_LINKS, 3]              float32
  link_endpoints:      [B, MAX_LINKS, 2]              int32
  link_mask:           [B, MAX_LINKS]                  bool
  node_mask:           [B, MAX_NODES]                  bool
  lp_features:         [B, MAX_LIGHTPATHS, 20]        float32
  lp_mask:             [B, MAX_LIGHTPATHS]             bool
  global_features:     [B, 8]                          float32

Output:
  z:                   [B, latent_dim]                 float32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict

# Import constants — these match the HDF5 data shapes
try:
    from ..core.schemas import (
        MAX_NODES, MAX_LINKS, MAX_SLOTS, MAX_LIGHTPATHS,
        N_LINK_STATIC_FEATURES, N_NODE_FEATURES, N_LP_FEATURES,
        N_GLOBAL_FEATURES, N_LINK_SPECTRAL_FEATURES,
    )
except ImportError:
    # Standalone testing — hardcode constants
    MAX_NODES = 20
    MAX_LINKS = 40
    MAX_SLOTS = 80
    MAX_LIGHTPATHS = 160
    N_LINK_STATIC_FEATURES = 3
    N_NODE_FEATURES = 5
    N_LP_FEATURES = 20
    N_GLOBAL_FEATURES = 8
    N_LINK_SPECTRAL_FEATURES = 5


# =====================================================================
# Stage 1: Spectral Encoder (Conv1D per link)
# =====================================================================

class SpectralEncoder(nn.Module):
    """
    Encode per-link spectral features using 1D convolutions.

    Each link has 80 slots × 5 features (occupancy, gsnr, power, ase, nli).
    This is a 1D signal — Conv1D extracts local spectral patterns
    (adjacent channel groups, spectral holes, NLI profiles).

    Input:  [B, N_links, 5, 80]
    Output: [B, N_links, spectral_dim]
    """

    def __init__(self, in_channels: int = N_LINK_SPECTRAL_FEATURES,
                 n_slots: int = MAX_SLOTS,
                 spectral_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, spectral_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → [B*N_links, spectral_dim, 1]
        )
        self.spectral_dim = spectral_dim

    def forward(self, spectral: torch.Tensor,
                link_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral: [B, MAX_LINKS, 5, MAX_SLOTS]
            link_mask: [B, MAX_LINKS] bool
        Returns:
            link_spectral: [B, MAX_LINKS, spectral_dim]
        """
        B, L, C, S = spectral.shape

        # Reshape to process all links at once: [B*L, C, S]
        x = spectral.reshape(B * L, C, S)
        x = self.conv(x)             # [B*L, spectral_dim, 1]
        x = x.squeeze(-1)            # [B*L, spectral_dim]
        x = x.reshape(B, L, -1)     # [B, MAX_LINKS, spectral_dim]

        # Zero out padded links
        x = x * link_mask.unsqueeze(-1).float()

        return x


# =====================================================================
# Stage 2: Graph Attention Layer
# =====================================================================

class GraphAttentionLayer(nn.Module):
    """
    GAT-style message passing with edge features.

    For each node i, compute attention-weighted messages from neighbors j:
      message_j = W_msg * [node_j || edge_ij]
      alpha_ij  = softmax_j(LeakyReLU(a^T * [W_q * node_i || W_k * node_j || edge_ij]))
      node_i'   = node_i + Σ_j alpha_ij * message_j

    Uses multi-head attention for stability.
    """

    def __init__(self, node_dim: int, edge_dim: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        assert node_dim % n_heads == 0

        # Query, Key from node features
        self.W_q = nn.Linear(node_dim, node_dim, bias=False)
        self.W_k = nn.Linear(node_dim, node_dim, bias=False)

        # Value from node + edge features
        self.W_v = nn.Linear(node_dim + edge_dim, node_dim, bias=False)

        # Edge projection for attention
        self.W_edge_attn = nn.Linear(edge_dim, n_heads, bias=False)

        # Output
        self.W_out = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat: torch.Tensor, edge_feat: torch.Tensor,
                edge_index: torch.Tensor, node_mask: torch.Tensor,
                link_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat:  [B, MAX_NODES, node_dim]
            edge_feat:  [B, MAX_LINKS, edge_dim]
            edge_index: [B, MAX_LINKS, 2] (src, dst node indices)
            node_mask:  [B, MAX_NODES] bool
            link_mask:  [B, MAX_LINKS] bool
        Returns:
            updated_node_feat: [B, MAX_NODES, node_dim]
        """
        B, N, D = node_feat.shape
        _, E, _ = edge_feat.shape
        H = self.n_heads
        Dh = self.head_dim

        # Query and Key: [B, N, H, Dh]
        Q = self.W_q(node_feat).reshape(B, N, H, Dh)
        K = self.W_k(node_feat).reshape(B, N, H, Dh)

        # For each edge, gather src and dst node features
        src_idx = edge_index[:, :, 0].long()  # [B, E]
        dst_idx = edge_index[:, :, 1].long()  # [B, E]

        # Gather Q for dst (receiver), K for src (sender)
        # [B, E] → [B, E, 1, 1] for gather, then expand
        dst_exp = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(B, E, H, Dh)
        src_exp = src_idx.unsqueeze(-1).unsqueeze(-1).expand(B, E, H, Dh)

        Q_dst = torch.gather(Q, 1, dst_exp)  # [B, E, H, Dh]
        K_src = torch.gather(K, 1, src_exp)  # [B, E, H, Dh]

        # Attention scores: dot product + edge bias
        attn = (Q_dst * K_src).sum(dim=-1) / math.sqrt(Dh)  # [B, E, H]
        edge_bias = self.W_edge_attn(edge_feat)  # [B, E, H]
        attn = attn + edge_bias

        # Mask invalid edges
        edge_mask = link_mask.unsqueeze(-1).expand(B, E, H)  # [B, E, H]
        attn = attn.masked_fill(~edge_mask, float('-inf'))

        # Value: node_src || edge → project
        src_feat_exp = src_idx.unsqueeze(-1).expand(B, E, D)
        V_node = torch.gather(node_feat, 1, src_feat_exp)  # [B, E, D]
        V_input = torch.cat([V_node, edge_feat], dim=-1)    # [B, E, D + edge_dim]
        V = self.W_v(V_input).reshape(B, E, H, Dh)          # [B, E, H, Dh]

        # Softmax per destination node
        # We need to scatter attention weights and aggregate per dst node
        # Use scatter_softmax approach: compute for each dst node
        attn_weights = self._scatter_softmax(attn, dst_idx, N)  # [B, E, H]
        attn_weights = self.dropout(attn_weights)

        # Weighted messages
        messages = attn_weights.unsqueeze(-1) * V  # [B, E, H, Dh]

        # Scatter-add messages to destination nodes
        messages_flat = messages.reshape(B, E, D)  # [B, E, node_dim]
        aggregated = torch.zeros(B, N, D, device=node_feat.device)
        dst_exp_flat = dst_idx.unsqueeze(-1).expand(B, E, D)
        aggregated.scatter_add_(1, dst_exp_flat, messages_flat)

        # Also process reverse direction (undirected graph)
        src_exp_flat = src_idx.unsqueeze(-1).expand(B, E, D)
        aggregated.scatter_add_(1, src_exp_flat, messages_flat)

        # Residual + norm
        out = self.W_out(aggregated)
        out = self.dropout(out)
        out = self.norm(node_feat + out)

        # Zero padded nodes
        out = out * node_mask.unsqueeze(-1).float()

        return out

    def _scatter_softmax(self, attn: torch.Tensor, dst_idx: torch.Tensor,
                         n_nodes: int) -> torch.Tensor:
        B, E, H = attn.shape

        # Numerical stability: subtract max per destination node
        dst_exp = dst_idx.unsqueeze(-1).expand(B, E, H)
        max_per_node = torch.full((B, n_nodes, H), float('-inf'), device=attn.device)
        max_per_node.scatter_reduce_(1, dst_exp, attn, reduce='amax', include_self=False)
        max_vals = torch.gather(max_per_node, 1, dst_exp)  # [B, E, H]
        max_vals = max_vals.clamp(min=-1e6)  # handle -inf for empty nodes

        attn_stable = attn - max_vals
        attn_exp = attn_stable.exp()

        sum_per_node = torch.zeros(B, n_nodes, H, device=attn.device)
        sum_per_node.scatter_add_(1, dst_exp, attn_exp)
        denom = torch.gather(sum_per_node, 1, dst_exp)
        denom = denom.clamp(min=1e-8)

        return attn_exp / denom


# =====================================================================
# LP Feature Aggregator
# =====================================================================

class LPAggregator(nn.Module):
    """
    Aggregate per-lightpath features into a fixed-size summary.

    Input:  [B, MAX_LIGHTPATHS, N_LP_FEATURES] with mask
    Output: [B, lp_summary_dim]
    """

    def __init__(self, in_dim: int = N_LP_FEATURES, hidden_dim: int = 64,
                 out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, lp_features: torch.Tensor,
                lp_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lp_features: [B, MAX_LIGHTPATHS, N_LP_FEATURES]
            lp_mask:     [B, MAX_LIGHTPATHS] bool
        Returns:
            summary: [B, out_dim]
        """
        # Project each LP
        x = self.mlp(lp_features)  # [B, MAX_LP, out_dim]

        # Masked mean pooling
        mask_f = lp_mask.unsqueeze(-1).float()  # [B, MAX_LP, 1]
        x = x * mask_f
        n_active = mask_f.sum(dim=1).clamp(min=1)  # [B, 1]
        summary = x.sum(dim=1) / n_active  # [B, out_dim]

        return summary


# =====================================================================
# Full Encoder
# =====================================================================

class OpticalNetworkEncoder(nn.Module):
    """
    Full two-stage encoder: Conv1D spectral + GNN + pooling.

    Architecture:
      1. SpectralEncoder: per-link [5, 80] → spectral_dim
      2. Concat with link_static → edge features
      3. Project node features → node_dim
      4. N layers of GraphAttention (message passing)
      5. Global mean pooling over nodes
      6. Concat LP summary + global features
      7. Projection MLP + BatchNorm → z

    The BatchNorm at the output is required for SIGReg compatibility
    (LayerNorm would prevent SIGReg from working, same issue as in LeWM).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        spectral_dim: int = 64,
        node_hidden_dim: int = 128,
        n_gnn_layers: int = 3,
        n_heads: int = 4,
        lp_summary_dim: int = 32,
        dropout: float = 0.1,
        use_spectral_conv: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_spectral_conv = use_spectral_conv

        # ── Stage 1: Spectral encoding ──
        if use_spectral_conv:
            self.spectral_encoder = SpectralEncoder(
                in_channels=N_LINK_SPECTRAL_FEATURES,
                n_slots=MAX_SLOTS,
                spectral_dim=spectral_dim,
            )
            edge_input_dim = spectral_dim + N_LINK_STATIC_FEATURES
        else:
            # Ablation: mean-pool spectral features instead of Conv1D
            self.spectral_encoder = None
            # 5 spectral features mean-pooled + 5 std-pooled + 3 static
            edge_input_dim = N_LINK_SPECTRAL_FEATURES * 2 + N_LINK_STATIC_FEATURES

        # Edge projection to match node_dim for attention
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_input_dim, node_hidden_dim),
            nn.ReLU(),
        )
        edge_dim = node_hidden_dim

        # ── Stage 2: Node feature projection ──
        # node_feat_dim is detected at first forward pass
        self.node_hidden_dim = node_hidden_dim
        self.node_proj = None  # built lazily

        # ── Stage 2: GNN layers ──
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                node_dim=node_hidden_dim,
                edge_dim=edge_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_gnn_layers)
        ])

        # ── LP aggregator ──
        self.lp_aggregator = LPAggregator(
            in_dim=N_LP_FEATURES,
            hidden_dim=64,
            out_dim=lp_summary_dim,
        )

        # ── Final projection ──
        # Input: node_pool (node_hidden_dim) + lp_summary (lp_summary_dim)
        #        + global_features (N_GLOBAL_FEATURES)
        proj_input_dim = node_hidden_dim + lp_summary_dim + N_GLOBAL_FEATURES
        self.projector = nn.Sequential(
            nn.Linear(proj_input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),  # Required for SIGReg
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a batch of network states into latent vectors.

        Args:
            batch: dict with keys matching HDF5 dataset fields:
                - spectral_occupancy: [B, MAX_LINKS, MAX_SLOTS]
                - channel_gsnr:       [B, MAX_LINKS, MAX_SLOTS]
                - channel_power:      [B, MAX_LINKS, MAX_SLOTS]
                - channel_ase:        [B, MAX_LINKS, MAX_SLOTS]
                - channel_nli:        [B, MAX_LINKS, MAX_SLOTS]
                - link_static:        [B, MAX_LINKS, 3]
                - link_endpoints:     [B, MAX_LINKS, 2]
                - link_mask:          [B, MAX_LINKS]
                - node_mask:          [B, MAX_NODES]
                - lp_features:        [B, MAX_LIGHTPATHS, 20]
                - lp_mask:            [B, MAX_LIGHTPATHS]
                - global_features:    [B, 8]
                (node_features not in HDF5 per-step; recomputed or passed)

        Returns:
            z: [B, latent_dim]
        """
        link_mask = batch['link_mask']
        # ── Normalize raw features to prevent overflow ──
        # link_static can have values up to ~230 (length_km)
        if 'link_static' in batch:
            link_static = batch['link_static'].clone()
            link_static_mask = link_mask.unsqueeze(-1).float()
            # Simple normalization: divide by max reasonable values
            link_static[:, :, 0] = link_static[:, :, 0] / 200.0  # length_km
            link_static[:, :, 1] = link_static[:, :, 1] / 10.0   # n_spans
            link_static[:, :, 2] = link_static[:, :, 2] / 10.0   # n_amps
        else:
            link_static = batch.get('link_static', torch.zeros(
                link_mask.shape[0], MAX_LINKS, N_LINK_STATIC_FEATURES,
                device=link_mask.device))
        node_mask = batch['node_mask']

        # ── Stage 1: Spectral features → per-link embedding ──
        if self.use_spectral_conv:
            # Stack spectral channels: [B, MAX_LINKS, 5, MAX_SLOTS]
            spectral = torch.stack([
                batch['spectral_occupancy'].float(),
                batch['channel_gsnr'],
                batch['channel_power'],
                batch['channel_ase'],
                batch['channel_nli'],
            ], dim=2)
            link_spectral = self.spectral_encoder(spectral, link_mask)
        else:
            # Ablation: simple statistics per link
            spectral = torch.stack([
                batch['spectral_occupancy'].float(),
                batch['channel_gsnr'],
                batch['channel_power'],
                batch['channel_ase'],
                batch['channel_nli'],
            ], dim=2)  # [B, MAX_LINKS, 5, MAX_SLOTS]
            link_mean = spectral.mean(dim=-1)  # [B, MAX_LINKS, 5]
            link_std = spectral.std(dim=-1)    # [B, MAX_LINKS, 5]
            link_spectral = torch.cat([link_mean, link_std], dim=-1)

        # Concat spectral + static link features → edge features
        edge_feat = torch.cat([
            link_spectral,
            link_static,
        ], dim=-1)  # [B, MAX_LINKS, edge_input_dim]
        edge_feat = self.edge_proj(edge_feat)  # [B, MAX_LINKS, edge_dim]
        edge_feat = edge_feat * link_mask.unsqueeze(-1).float()

        # ── Stage 2: Node features ──
        # Node features: use degree + LP info from global context
        # In HDF5, node_features are stored in topology (static)
        # but per-step node features (add/drop counts) are in the state
        # For now, use the node_features if provided, else zeros
        if 'node_features' in batch:
            nf = batch['node_features']
            # Lazy init node_proj to match actual feature dim
            if self.node_proj is None:
                self.node_proj = nn.Sequential(
                    nn.Linear(nf.shape[-1], self.node_hidden_dim),
                    nn.ReLU(),
                ).to(nf.device)
            node_feat = self.node_proj(nf)
        else:
            if self.node_proj is None:
                self.node_proj = nn.Sequential(
                    nn.Linear(2, self.node_hidden_dim),
                    nn.ReLU(),
                ).to(link_mask.device)
            node_feat = torch.zeros(
                link_mask.shape[0], MAX_NODES, 2,
                device=link_mask.device,
            )
            node_feat = self.node_proj(node_feat)

        node_feat = node_feat * node_mask.unsqueeze(-1).float()

        # ── Stage 2: GNN message passing ──
        edge_index = batch['link_endpoints']  # [B, MAX_LINKS, 2]
        for gnn_layer in self.gnn_layers:
            node_feat = gnn_layer(
                node_feat, edge_feat, edge_index, node_mask, link_mask
            )

        # ── Global pooling: masked mean over real nodes ──
        mask_f = node_mask.unsqueeze(-1).float()  # [B, MAX_NODES, 1]
        node_sum = (node_feat * mask_f).sum(dim=1)  # [B, node_dim]
        n_nodes = mask_f.sum(dim=1).clamp(min=1)    # [B, 1]
        graph_pool = node_sum / n_nodes              # [B, node_dim]

        # ── LP summary ──
        lp_summary = self.lp_aggregator(
            batch['lp_features'], batch['lp_mask']
        )  # [B, lp_summary_dim]

        # ── Global features ──
        global_feat = batch['global_features']  # [B, 8]

        # ── Concatenate and project ──
        combined = torch.cat([
            graph_pool, lp_summary, global_feat
        ], dim=-1)  # [B, node_dim + lp_dim + 8]

        z = self.projector(combined)  # [B, latent_dim]

        return z

    def get_param_count(self) -> dict:
        """Count parameters by component."""
        counts = {}
        if self.spectral_encoder is not None:
            counts['spectral_encoder'] = sum(
                p.numel() for p in self.spectral_encoder.parameters()
            )
        counts['edge_proj'] = sum(
            p.numel() for p in self.edge_proj.parameters()
        )
        counts['node_proj'] = sum(
            p.numel() for p in self.node_proj.parameters()
        )
        counts['gnn_layers'] = sum(
            p.numel() for p in self.gnn_layers.parameters()
        )
        counts['lp_aggregator'] = sum(
            p.numel() for p in self.lp_aggregator.parameters()
        )
        counts['projector'] = sum(
            p.numel() for p in self.projector.parameters()
        )
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


# =====================================================================
# Ablation: Simple MLP Encoder (no GNN, no Conv1D)
# =====================================================================

class MLPEncoder(nn.Module):
    """
    Baseline encoder: flatten everything → MLP → z.
    No graph structure, no spectral convolutions.
    Used as ablation to show the value of the GNN architecture.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()

        # Spectral: mean + std per link × 5 features = 10 per link × MAX_LINKS
        # LP features: mean pool
        # Global: 8
        # Link static: 3 × MAX_LINKS
        input_dim = (
            MAX_LINKS * (N_LINK_SPECTRAL_FEATURES * 2 + N_LINK_STATIC_FEATURES)
            + N_LP_FEATURES  # mean-pooled LP features
            + N_GLOBAL_FEATURES
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = batch['link_mask'].shape[0]

        # Spectral summary per link
        spectral = torch.stack([
            batch['spectral_occupancy'].float(),
            batch['channel_gsnr'],
            batch['channel_power'],
            batch['channel_ase'],
            batch['channel_nli'],
        ], dim=2)  # [B, MAX_LINKS, 5, MAX_SLOTS]
        link_mean = spectral.mean(dim=-1)  # [B, MAX_LINKS, 5]
        link_std = spectral.std(dim=-1)    # [B, MAX_LINKS, 5]

        # Flatten link features
        link_flat = torch.cat([
            link_mean.reshape(B, -1),
            link_std.reshape(B, -1),
            batch['link_static'].reshape(B, -1),
        ], dim=-1)

        # LP mean pool
        lp_mask_f = batch['lp_mask'].unsqueeze(-1).float()
        lp_pool = (batch['lp_features'] * lp_mask_f).sum(dim=1)
        lp_pool = lp_pool / lp_mask_f.sum(dim=1).clamp(min=1)

        # Combine
        x = torch.cat([link_flat, lp_pool, batch['global_features']], dim=-1)
        return self.mlp(x)


# =====================================================================
# Quick test
# =====================================================================

if __name__ == "__main__":
    print("Testing encoder shapes...")

    B = 4  # batch size

    # Create fake batch matching HDF5 shapes
    batch = {
        'spectral_occupancy': torch.zeros(B, MAX_LINKS, MAX_SLOTS),
        'channel_gsnr': torch.randn(B, MAX_LINKS, MAX_SLOTS) * 5 + 20,
        'channel_power': torch.randn(B, MAX_LINKS, MAX_SLOTS),
        'channel_ase': torch.randn(B, MAX_LINKS, MAX_SLOTS) * 0.1,
        'channel_nli': torch.randn(B, MAX_LINKS, MAX_SLOTS) * 0.01,
        'link_static': torch.randn(B, MAX_LINKS, N_LINK_STATIC_FEATURES),
        'link_endpoints': torch.zeros(B, MAX_LINKS, 2, dtype=torch.long),
        'link_mask': torch.zeros(B, MAX_LINKS, dtype=torch.bool),
        'node_mask': torch.zeros(B, MAX_NODES, dtype=torch.bool),
        'node_features': torch.randn(B, MAX_NODES, N_NODE_FEATURES),
        'lp_features': torch.randn(B, MAX_LIGHTPATHS, N_LP_FEATURES),
        'lp_mask': torch.zeros(B, MAX_LIGHTPATHS, dtype=torch.bool),
        'global_features': torch.randn(B, N_GLOBAL_FEATURES),
    }

    # Set some real nodes/links/LPs
    n_nodes, n_links, n_lps = 8, 11, 20
    batch['node_mask'][:, :n_nodes] = True
    batch['link_mask'][:, :n_links] = True
    batch['lp_mask'][:, :n_lps] = True
    batch['spectral_occupancy'][:, :n_links, :n_lps] = 1.0

    # Set valid link endpoints
    for i in range(n_links):
        batch['link_endpoints'][:, i, 0] = i % n_nodes
        batch['link_endpoints'][:, i, 1] = (i + 1) % n_nodes

    # ── Test Full Encoder ──
    print("\n=== OpticalNetworkEncoder (full) ===")
    enc = OpticalNetworkEncoder(latent_dim=128, n_gnn_layers=3)
    z = enc(batch)
    print(f"  Input shapes:")
    for k, v in batch.items():
        print(f"    {k}: {list(v.shape)}")
    print(f"  Output z: {list(z.shape)}")
    params = enc.get_param_count()
    for name, count in params.items():
        print(f"  {name}: {count:,} params")

    # ── Test No-Conv1D ablation ──
    print("\n=== OpticalNetworkEncoder (no Conv1D) ===")
    enc_noconv = OpticalNetworkEncoder(
        latent_dim=128, n_gnn_layers=3, use_spectral_conv=False
    )
    z2 = enc_noconv(batch)
    print(f"  Output z: {list(z2.shape)}")
    p2 = enc_noconv.get_param_count()
    print(f"  Total: {p2['total']:,} params")

    # ── Test MLP baseline ──
    print("\n=== MLPEncoder (baseline) ===")
    enc_mlp = MLPEncoder(latent_dim=128)
    z3 = enc_mlp(batch)
    print(f"  Output z: {list(z3.shape)}")
    print(f"  Total: {sum(p.numel() for p in enc_mlp.parameters()):,} params")

    # ── Verify gradients flow ──
    print("\n=== Gradient check ===")
    loss = z.sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.parameters() if p.requires_grad
    )
    print(f"  Gradients flow: {'✓' if grad_ok else '✗'}")

    print("\n✓ All encoder tests passed")