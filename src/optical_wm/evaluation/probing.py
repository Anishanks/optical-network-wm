"""
Probing Evaluation (Éval 1) for Optical Network World Model.

Evaluates whether the encoder's latent space captures physical quantities.

Protocol:
  1. Load trained JEPA checkpoint
  2. Freeze encoder (no gradients)
  3. Encode all states in dataset → z vectors
  4. Train small probes (linear / MLP) to predict physical quantities from z
  5. Report MSE ↓ and Pearson r ↑

Probing targets (from global_features):
  - n_active_lightpaths    [0]
  - total_capacity_tbps    [1]
  - n_infeasible           [3]
  - worst_margin_db        [4]
  - avg_margin_db          [5]
  - spectral_utilization   [6]

Baselines:
  - JEPA encoder (GNN)
  - MLP encoder (ablation)
  - Raw features (PCA to latent_dim)
  - Random projection (sanity check)

Usage:
  python -m optical_wm.evaluation.probing \
    --checkpoint checkpoints/smoke/best.pt \
    --data data_100 \
    --output figures/probing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# =====================================================================
# Probing targets
# =====================================================================

PROBE_TARGETS = {
    'n_active_lps':        {'idx': 0, 'name': 'Active Lightpaths'},
    'total_capacity':      {'idx': 1, 'name': 'Total Capacity (Tbps)'},
    'n_infeasible':        {'idx': 3, 'name': 'Infeasible LPs'},
    'worst_margin':        {'idx': 4, 'name': 'Worst Margin (dB)'},
    'avg_margin':          {'idx': 5, 'name': 'Avg Margin (dB)'},
    'spectral_util':       {'idx': 6, 'name': 'Spectral Utilization'},
}


# =====================================================================
# Probe models
# =====================================================================

class LinearProbe(nn.Module):
    """Single linear layer: z → scalar."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z).squeeze(-1)


class MLPProbe(nn.Module):
    """2-layer MLP: z → hidden → scalar."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).squeeze(-1)


# =====================================================================
# Extract embeddings from dataset
# =====================================================================

@torch.no_grad()
def extract_embeddings(
    model,
    data_loader,
    device: str = 'cpu',
    max_batches: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract encoder embeddings and corresponding global features.

    Returns:
        embeddings: [N, latent_dim]
        global_features: [N, n_global_features]
    """
    model.eval()
    all_z = []
    all_gf = []

    for i, batch in enumerate(data_loader):
        if max_batches and i >= max_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Remove actions (not needed for encoding)
        actions = batch.pop('actions', None)

        T = batch['global_features'].shape[1]

        # Encode each timestep
        for t in range(T):
            step_batch = {}
            for key, val in batch.items():
                if val.dim() >= 2 and val.shape[1] == T:
                    step_batch[key] = val[:, t]
                else:
                    step_batch[key] = val

            z_t = model.encoder(step_batch)  # [B, D]
            gf_t = step_batch['global_features']  # [B, 8]

            all_z.append(z_t.cpu())
            all_gf.append(gf_t.cpu())

    embeddings = torch.cat(all_z, dim=0)       # [N, D]
    global_feats = torch.cat(all_gf, dim=0)    # [N, 8]

    return embeddings, global_feats


# =====================================================================
# Train and evaluate a probe
# =====================================================================

def train_probe(
    probe: nn.Module,
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 1e-3,
    n_epochs: int = 100,
    batch_size: int = 256,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Train a probe and return metrics.

    Returns:
        dict with 'mse', 'rmse', 'pearson_r', 'r_squared'
    """
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    z_train = z_train.to(device)
    y_train = y_train.to(device)
    z_val = z_val.to(device)
    y_val = y_val.to(device)

    # Normalize targets for training stability
    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-6)
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    N = len(z_train)

    for epoch in range(n_epochs):
        probe.train()
        # Mini-batch training
        perm = torch.randperm(N, device=device)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            z_batch = z_train[idx]
            y_batch = y_train_norm[idx]

            pred = probe(z_batch)
            loss = F.mse_loss(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    # Evaluate on validation set
    probe.eval()
    with torch.no_grad():
        pred_val = probe(z_val)
        # Denormalize
        pred_val_denorm = pred_val * y_std + y_mean
        y_val_denorm = y_val

        mse = F.mse_loss(pred_val_denorm, y_val_denorm).item()
        rmse = mse ** 0.5

        # Pearson correlation
        pred_np = pred_val_denorm.cpu().numpy()
        y_np = y_val_denorm.cpu().numpy()

        if np.std(pred_np) > 1e-8 and np.std(y_np) > 1e-8:
            pearson_r = float(np.corrcoef(pred_np, y_np)[0, 1])
        else:
            pearson_r = 0.0

        # R²
        ss_res = ((pred_val_denorm - y_val_denorm) ** 2).sum().item()
        ss_tot = ((y_val_denorm - y_val_denorm.mean()) ** 2).sum().item()
        r_squared = 1 - ss_res / max(ss_tot, 1e-8)

    return {
        'mse': mse,
        'rmse': rmse,
        'pearson_r': pearson_r,
        'r_squared': r_squared,
    }


# =====================================================================
# Run full probing evaluation
# =====================================================================

def run_probing(
    model,
    train_loader,
    val_loader,
    device: str = 'cpu',
    n_probe_epochs: int = 100,
    max_extract_batches: int = None,
) -> Dict:
    """
    Run complete probing evaluation.

    Returns dict with results per target per probe type.
    """
    print("\n  Extracting embeddings...")

    # Extract train embeddings
    z_train, gf_train = extract_embeddings(
        model, train_loader, device, max_batches=max_extract_batches
    )
    print(f"    Train: {z_train.shape[0]} samples, z dim={z_train.shape[1]}")

    # Extract val embeddings
    z_val, gf_val = extract_embeddings(
        model, val_loader, device, max_batches=max_extract_batches
    )
    print(f"    Val:   {z_val.shape[0]} samples, z dim={z_val.shape[1]}")

    latent_dim = z_train.shape[1]
    results = {}

    print(f"\n  Training probes (latent_dim={latent_dim})...")
    print(f"  {'Target':<22} {'Probe':<8} {'MSE':>8} {'RMSE':>8} {'r':>8} {'R²':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for target_key, target_info in PROBE_TARGETS.items():
        idx = target_info['idx']
        y_train = gf_train[:, idx]
        y_val = gf_val[:, idx]

        # Skip targets with no variance
        if y_train.std() < 1e-6:
            print(f"  {target_key:<22} SKIPPED (no variance)")
            continue

        results[target_key] = {}

        for probe_name, probe_cls in [('linear', LinearProbe),
                                       ('mlp', MLPProbe)]:
            probe = probe_cls(latent_dim)
            metrics = train_probe(
                probe, z_train, y_train, z_val, y_val,
                n_epochs=n_probe_epochs,
                device=device,
            )
            results[target_key][probe_name] = metrics

            print(
                f"  {target_key:<22} {probe_name:<8} "
                f"{metrics['mse']:>8.4f} {metrics['rmse']:>8.4f} "
                f"{metrics['pearson_r']:>8.4f} {metrics['r_squared']:>8.4f}"
            )

    # ── Summary ──
    print(f"\n  ── Summary ──")
    avg_r_linear = np.mean([
        results[t]['linear']['pearson_r']
        for t in results if 'linear' in results[t]
    ])
    avg_r_mlp = np.mean([
        results[t]['mlp']['pearson_r']
        for t in results if 'mlp' in results[t]
    ])
    print(f"  Avg Pearson r (linear): {avg_r_linear:.4f}")
    print(f"  Avg Pearson r (MLP):    {avg_r_mlp:.4f}")

    if avg_r_linear > 0.9:
        print(f"  ✅ Linear probes work well — physics is linearly accessible in z")
    elif avg_r_mlp > 0.9:
        print(f"  ⚠  Only MLP probes work — physics is encoded but non-linearly")
    else:
        print(f"  ✗  Probing fails — encoder may not capture physics")

    return results


# =====================================================================
# Baseline: probe on raw features (PCA compressed)
# =====================================================================

def run_raw_feature_baseline(
    train_loader,
    val_loader,
    latent_dim: int = 128,
    device: str = 'cpu',
    n_probe_epochs: int = 100,
    max_extract_batches: int = None,
) -> Dict:
    """
    Baseline: apply PCA to raw features, then probe.
    Shows whether the encoder adds value beyond dimensionality reduction.
    """
    print("\n  ── Raw Feature Baseline (PCA) ──")
    print("  Extracting raw features...")

    raw_train, gf_train = [], []
    raw_val, gf_val = [], []

    for loader, raw_list, gf_list in [
        (train_loader, raw_train, gf_train),
        (val_loader, raw_val, gf_val),
    ]:
        for i, batch in enumerate(loader):
            if max_extract_batches and i >= max_extract_batches:
                break
            T = batch['global_features'].shape[1]
            for t in range(T):
                # Extract simple per-step features
                gsnr_mean = batch['channel_gsnr'][:, t].mean(dim=-1).mean(dim=-1)  # [B]
                gsnr_std = batch['channel_gsnr'][:, t].std(dim=-1).mean(dim=-1)
                occ_mean = batch['spectral_occupancy'][:, t].float().mean(dim=-1).mean(dim=-1)
                power_mean = batch['channel_power'][:, t].mean(dim=-1).mean(dim=-1)
                gf = batch['global_features'][:, t]  # [B, 8]

                # Stack simple features
                feats = torch.stack([gsnr_mean, gsnr_std, occ_mean, power_mean], dim=-1)
                feats = torch.cat([feats, gf], dim=-1)  # [B, 12]

                raw_list.append(feats)
                gf_list.append(gf)

    raw_train = torch.cat(raw_train, dim=0)  # [N, 12]
    gf_train = torch.cat(gf_train, dim=0)
    raw_val = torch.cat(raw_val, dim=0)
    gf_val = torch.cat(gf_val, dim=0)

    print(f"    Train: {raw_train.shape[0]} samples, feat dim={raw_train.shape[1]}")

    results = {}
    feat_dim = raw_train.shape[1]

    print(f"  {'Target':<22} {'Probe':<8} {'MSE':>8} {'RMSE':>8} {'r':>8} {'R²':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for target_key, target_info in PROBE_TARGETS.items():
        idx = target_info['idx']
        y_train = gf_train[:, idx]
        y_val = gf_val[:, idx]

        if y_train.std() < 1e-6:
            continue

        results[target_key] = {}

        for probe_name, probe_cls in [('linear', LinearProbe),
                                       ('mlp', MLPProbe)]:
            probe = probe_cls(feat_dim)
            metrics = train_probe(
                probe, raw_train, y_train, raw_val, y_val,
                n_epochs=n_probe_epochs,
                device=device,
            )
            results[target_key][probe_name] = metrics

            print(
                f"  {target_key:<22} {probe_name:<8} "
                f"{metrics['mse']:>8.4f} {metrics['rmse']:>8.4f} "
                f"{metrics['pearson_r']:>8.4f} {metrics['r_squared']:>8.4f}"
            )

    return results


# =====================================================================
# Generate comparison table and figures
# =====================================================================

def generate_report(
    jepa_results: Dict,
    baseline_results: Dict = None,
    output_dir: str = "figures/probing",
):
    """Save results as JSON + optionally generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    report = {'jepa': jepa_results}
    if baseline_results:
        report['raw_features'] = baseline_results

    # Save JSON
    with open(f"{output_dir}/probing_results.json", 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {output_dir}/probing_results.json")

    # Try to generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        targets = list(jepa_results.keys())
        if not targets:
            return

        # Bar chart: Pearson r comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(targets))
        width = 0.2

        # JEPA linear
        r_jepa_lin = [jepa_results[t]['linear']['pearson_r'] for t in targets]
        ax.bar(x - 1.5*width, r_jepa_lin, width, label='JEPA (linear)',
               color='#065A82')

        # JEPA MLP
        r_jepa_mlp = [jepa_results[t]['mlp']['pearson_r'] for t in targets]
        ax.bar(x - 0.5*width, r_jepa_mlp, width, label='JEPA (MLP)',
               color='#059669')

        if baseline_results:
            # Raw linear
            r_raw_lin = [baseline_results.get(t, {}).get('linear', {}).get(
                'pearson_r', 0) for t in targets]
            ax.bar(x + 0.5*width, r_raw_lin, width,
                   label='Raw features (linear)', color='#D97706')

            # Raw MLP
            r_raw_mlp = [baseline_results.get(t, {}).get('mlp', {}).get(
                'pearson_r', 0) for t in targets]
            ax.bar(x + 1.5*width, r_raw_mlp, width,
                   label='Raw features (MLP)', color='#DC2626')

        ax.set_ylabel('Pearson r')
        ax.set_title('Probing: Physical Quantity Recovery from Latent Space')
        ax.set_xticks(x)
        ax.set_xticklabels([PROBE_TARGETS[t]['name'] for t in targets],
                           rotation=25, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5,
                   label='r=0.9 threshold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/probing_comparison.png", dpi=150)
        fig.savefig(f"{output_dir}/probing_comparison.pdf")
        plt.close(fig)
        print(f"  Saved: {output_dir}/probing_comparison.png")

    except ImportError:
        print("  matplotlib not available — skipping plots")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Probing evaluation for JEPA encoder"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='data_100',
                        help='Data directory')
    parser.add_argument('--output', type=str, default='figures/probing',
                        help='Output directory for results')
    parser.add_argument('--probe-epochs', type=int, default=100,
                        help='Epochs to train each probe')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding extraction')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Limit batches for quick test')
    parser.add_argument('--baseline', action='store_true',
                        help='Also run raw feature baseline')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # ── Load model ──
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device,
                      weights_only=False)
    config = ckpt.get('config', {})

    # Add parent directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    try:
        from ..models.world_model import OpticalWorldModel
        from ..training.dataset import create_dataloaders
    except ImportError:
        from models.world_model import OpticalWorldModel
        from training.dataset import create_dataloaders

    # Reconstruct model from config
    model = OpticalWorldModel(
        latent_dim=config.get('latent_dim', 128),
        spectral_dim=config.get('spectral_dim', 64),
        node_hidden_dim=config.get('node_hidden_dim', 128),
        n_gnn_layers=config.get('n_gnn_layers', 3),
        n_gnn_heads=config.get('n_gnn_heads', 4),
        lp_summary_dim=config.get('lp_summary_dim', 32),
        use_spectral_conv=config.get('use_spectral_conv', True),
        encoder_type=config.get('encoder_type', 'gnn'),
        pred_hidden_dim=config.get('pred_hidden_dim', 256),
        n_pred_layers=config.get('n_pred_layers', 4),
        n_pred_heads=config.get('n_pred_heads', 8),
        action_emb_dim=config.get('action_emb_dim', 64),
        predictor_type=config.get('predictor_type', 'transformer'),
        collapse_method=config.get('collapse_method', 'variance'),
    )

    # Load weights (handle lazy-init node_proj)
    state_dict = ckpt['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    model.eval()

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Create data loaders ──
    print(f"\n  Loading data from {args.data}...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data,
        context_length=config.get('context_length', 8),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    # ── Run probing ──
    print(f"\n{'='*60}")
    print(f"  Probing Evaluation")
    print(f"{'='*60}")

    jepa_results = run_probing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        n_probe_epochs=args.probe_epochs,
        max_extract_batches=args.max_batches,
    )

    # ── Raw feature baseline ──
    baseline_results = None
    if args.baseline:
        baseline_results = run_raw_feature_baseline(
            train_loader=train_loader,
            val_loader=val_loader,
            latent_dim=config.get('latent_dim', 128),
            device=args.device,
            n_probe_epochs=args.probe_epochs,
            max_extract_batches=args.max_batches,
        )

    # ── Generate report ──
    generate_report(jepa_results, baseline_results, args.output)

    print(f"\n{'='*60}")
    print(f"  Probing complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()