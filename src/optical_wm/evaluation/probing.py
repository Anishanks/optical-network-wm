"""
Predictive Probing Evaluation for Optical Network World Model.

Tests whether the encoder captures *dynamics*, not just information present
in the current state.

Protocol:
  For each anchor timestep t and horizon k in {1, 3, 5, 7}:
    target = global_features[t + k]
    Compare how well different inputs at time t predict that target.

Inputs compared (all evaluated at time t):
  - jepa:   z_t = encoder(state_t)                 [D]
  - gf_ar:  global_features_t                      [8]   (linear = AR baseline)
  - pooled: raw spectral aggregates (no gf)        [4]   (mean/std GSNR, occ, power)

Reference baseline:
  - persistence: predict gf[t+k][idx] = gf[t][idx]  (analytic, no training)

Why this differs from the old protocol:
  The previous version probed gf[t] from z_t, but gf[t] is part of the input
  passed to the encoder -- a trivial encoder that preserves information gets
  perfect scores. Predicting gf[t+k] forces the representation to encode
  dynamics, not just copy the state.

Usage:
  PYTHONPATH=src py -m optical_wm.evaluation.probing \
    --checkpoint checkpoints/run/best.pt \
    --data data_500 \
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
from typing import Dict, List, Tuple


PROBE_TARGETS = {
    'n_active_lps':   {'idx': 0, 'name': 'Active Lightpaths'},
    'total_capacity': {'idx': 1, 'name': 'Total Capacity (Tbps)'},
    'n_infeasible':   {'idx': 3, 'name': 'Infeasible LPs'},
    'worst_margin':   {'idx': 4, 'name': 'Worst Margin (dB)'},
    'avg_margin':     {'idx': 5, 'name': 'Avg Margin (dB)'},
    'spectral_util':  {'idx': 6, 'name': 'Spectral Utilization'},
}

DEFAULT_HORIZONS = [1, 3, 5, 7]


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


# ---------------------------------------------------------------
# Extract (anchor, future) pairs
# ---------------------------------------------------------------

@torch.no_grad()
def extract_pairs(
    model,
    data_loader,
    horizons: List[int],
    device: str = 'cpu',
    max_batches: int = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    For each horizon k, collect aligned samples.

    Returns:
        {k: {'z': [N,D], 'gf': [N,8], 'pooled': [N,4], 'y': [N,8]}}
        where y = global_features at (anchor + k).
    """
    model.eval()
    per_k = {k: {'z': [], 'gf': [], 'pooled': [], 'y': []} for k in horizons}
    max_k = max(horizons)

    for i, batch in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch.pop('actions', None)

        T = batch['global_features'].shape[1]
        if T <= max_k:
            continue

        # Encode every timestep in the window once
        z_per_t: List[torch.Tensor] = []
        pooled_per_t: List[torch.Tensor] = []
        for t in range(T):
            step_batch = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.shape[1] == T:
                    step_batch[key] = val[:, t]
                else:
                    step_batch[key] = val

            z_t = model.encoder(step_batch)
            z_per_t.append(z_t.cpu())

            # Raw spectral aggregates -- deliberately EXCLUDE global_features
            gsnr = step_batch['channel_gsnr']   # [B, n_ch, ...]
            occ = step_batch['spectral_occupancy'].float()
            power = step_batch['channel_power']
            dims = tuple(range(1, gsnr.dim()))
            pooled_t = torch.stack([
                gsnr.mean(dim=dims),
                gsnr.std(dim=dims),
                occ.mean(dim=dims),
                power.mean(dim=dims),
            ], dim=-1)
            pooled_per_t.append(pooled_t.cpu())

        gf_all = batch['global_features'].cpu()  # [B, T, 8]

        # Build per-horizon pairs
        for t in range(T):
            for k in horizons:
                if t + k >= T:
                    continue
                per_k[k]['z'].append(z_per_t[t])
                per_k[k]['gf'].append(gf_all[:, t])
                per_k[k]['pooled'].append(pooled_per_t[t])
                per_k[k]['y'].append(gf_all[:, t + k])

    # Cat
    out = {}
    for k in horizons:
        if not per_k[k]['z']:
            continue
        out[k] = {name: torch.cat(lst, dim=0) for name, lst in per_k[k].items()}
    return out


# ---------------------------------------------------------------
# Train a single probe
# ---------------------------------------------------------------

def train_probe(
    probe: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 1e-3,
    n_epochs: int = 100,
    batch_size: int = 256,
    device: str = 'cpu',
) -> Dict[str, float]:
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    # Standardize inputs per-feature for numerical stability across probes
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train_n = (x_train - x_mean) / x_std
    x_val_n = (x_val - x_mean) / x_std

    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-6)
    y_train_n = (y_train - y_mean) / y_std

    N = len(x_train_n)
    for _ in range(n_epochs):
        probe.train()
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            pred = probe(x_train_n[idx])
            loss = F.mse_loss(pred, y_train_n[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(x_val_n) * y_std + y_mean
        mse = F.mse_loss(pred, y_val).item()
        pred_np = pred.cpu().numpy()
        y_np = y_val.cpu().numpy()
        if np.std(pred_np) > 1e-8 and np.std(y_np) > 1e-8:
            r = float(np.corrcoef(pred_np, y_np)[0, 1])
        else:
            r = 0.0
        ss_res = ((pred - y_val) ** 2).sum().item()
        ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / max(ss_tot, 1e-8)

    return {'mse': mse, 'rmse': mse ** 0.5, 'pearson_r': r, 'r_squared': r2}


# ---------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------

INPUT_KEYS = [
    ('jepa',   'z'),       # learned encoder
    ('gf_ar',  'gf'),      # linear probe on this = linear AR baseline
    ('pooled', 'pooled'),  # raw spectral aggregates, no gf
]


def run_predictive_probing(
    model,
    train_loader,
    val_loader,
    horizons: List[int] = DEFAULT_HORIZONS,
    device: str = 'cpu',
    n_probe_epochs: int = 100,
    max_extract_batches: int = None,
) -> Dict:
    print(f"\n  Extracting (z_t, gf_{{t+k}}) pairs, horizons={horizons}...")
    train_data = extract_pairs(model, train_loader, horizons, device, max_extract_batches)
    val_data = extract_pairs(model, val_loader, horizons, device, max_extract_batches)

    results = {}
    for k in horizons:
        if k not in train_data or k not in val_data:
            print(f"  horizon k={k}: no samples (context too short), skipping")
            continue

        td = train_data[k]
        vd = val_data[k]
        print(
            f"\n  === Horizon k={k}  "
            f"(train N={td['z'].shape[0]}, val N={vd['z'].shape[0]}, "
            f"z_dim={td['z'].shape[1]}) ==="
        )
        print(
            f"  {'Target':<20} {'Model':<12} {'Probe':<6} "
            f"{'MSE':>9} {'r':>7} {'R2':>7}"
        )
        print(f"  {'-'*20} {'-'*12} {'-'*6} {'-'*9} {'-'*7} {'-'*7}")

        results[k] = {}
        for target_key, info in PROBE_TARGETS.items():
            idx = info['idx']
            y_train = td['y'][:, idx]
            y_val = vd['y'][:, idx]
            if y_train.std() < 1e-6 or y_val.std() < 1e-6:
                print(f"  {target_key:<20} SKIPPED (no variance in y)")
                continue

            results[k][target_key] = {}

            # Persistence baseline: predict gf[t+k][idx] = gf[t][idx]
            persist_pred = vd['gf'][:, idx]
            persist_mse = ((persist_pred - y_val) ** 2).mean().item()
            if persist_pred.std() > 1e-8 and y_val.std() > 1e-8:
                persist_r = float(np.corrcoef(
                    persist_pred.numpy(), y_val.numpy()
                )[0, 1])
            else:
                persist_r = 0.0
            ss_res = ((persist_pred - y_val) ** 2).sum().item()
            ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
            persist_r2 = 1 - ss_res / max(ss_tot, 1e-8)
            results[k][target_key]['persistence'] = {
                'mse': persist_mse, 'pearson_r': persist_r, 'r_squared': persist_r2,
            }
            print(
                f"  {target_key:<20} {'persist':<12} {'-':<6} "
                f"{persist_mse:>9.4f} {persist_r:>7.3f} {persist_r2:>7.3f}"
            )

            # Trained probes
            for input_name, input_key in INPUT_KEYS:
                X_train = td[input_key]
                X_val = vd[input_key]
                for probe_name, probe_cls in [('linear', LinearProbe),
                                               ('mlp', MLPProbe)]:
                    probe = probe_cls(X_train.shape[1])
                    m = train_probe(
                        probe, X_train, y_train, X_val, y_val,
                        n_epochs=n_probe_epochs, device=device,
                    )
                    results[k][target_key][f'{input_name}_{probe_name}'] = m
                    print(
                        f"  {target_key:<20} {input_name:<12} {probe_name:<6} "
                        f"{m['mse']:>9.4f} {m['pearson_r']:>7.3f} "
                        f"{m['r_squared']:>7.3f}"
                    )

    # -- Summary: avg Pearson r for JEPA vs baselines --
    print(f"\n  == Summary: avg Pearson r across targets ==")
    print(f"  {'k':>3} {'persist':>9} {'gf_ar_lin':>11} {'pooled_lin':>11} "
          f"{'jepa_lin':>11} {'jepa_mlp':>11}")
    for k in sorted(results.keys()):
        r_persist = _mean_over_targets(results[k], 'persistence', 'pearson_r')
        r_gf_lin = _mean_over_targets(results[k], 'gf_ar_linear', 'pearson_r')
        r_pooled_lin = _mean_over_targets(results[k], 'pooled_linear', 'pearson_r')
        r_jepa_lin = _mean_over_targets(results[k], 'jepa_linear', 'pearson_r')
        r_jepa_mlp = _mean_over_targets(results[k], 'jepa_mlp', 'pearson_r')
        print(
            f"  {k:>3} {r_persist:>9.3f} {r_gf_lin:>11.3f} "
            f"{r_pooled_lin:>11.3f} {r_jepa_lin:>11.3f} {r_jepa_mlp:>11.3f}"
        )

    print(
        "\n  Interpretation:\n"
        "    jepa > gf_ar  -> z_t encodes dynamics beyond the current aggregate state.\n"
        "    jepa ~ gf_ar  -> z_t preserves info but adds no predictive value.\n"
        "    jepa < gf_ar  -> encoder is losing information relevant to the target."
    )
    return results


def _mean_over_targets(k_results: Dict, key: str, metric: str) -> float:
    vals = [v[key][metric] for v in k_results.values() if key in v]
    return float(np.mean(vals)) if vals else float('nan')


# ---------------------------------------------------------------
# Report / plots
# ---------------------------------------------------------------

def generate_report(results: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/probing_results.json", 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\n  Saved: {output_dir}/probing_results.json")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping plots")
        return

    horizons = sorted(results.keys())
    if not horizons:
        return
    targets = list(next(iter(results.values())).keys())

    # Plot 1: avg Pearson r vs horizon for each model
    fig, ax = plt.subplots(figsize=(8, 5))
    models = [
        ('persistence',    'persistence',   '#6B7280', 'o', '--'),
        ('gf_ar_linear',   'gf AR (lin)',   '#D97706', 's', '-'),
        ('pooled_linear',  'pooled (lin)',  '#DC2626', '^', '-'),
        ('jepa_linear',    'JEPA (lin)',    '#065A82', 'o', '-'),
        ('jepa_mlp',       'JEPA (MLP)',    '#059669', 'D', '-'),
    ]
    for key, label, color, marker, ls in models:
        ys = [_mean_over_targets(results[k], key, 'pearson_r') for k in horizons]
        ax.plot(horizons, ys, marker=marker, linestyle=ls, color=color, label=label)
    ax.set_xlabel('Horizon k (steps)')
    ax.set_ylabel('Avg Pearson r (across targets)')
    ax.set_title('Predictive probing: gf[t+k] from inputs at t')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/probing_vs_horizon.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/probing_vs_horizon.png")

    # Plot 2: per-target Pearson r at k=1 and k=max
    for k in [horizons[0], horizons[-1]]:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(targets))
        width = 0.15
        offsets = [-2, -1, 0, 1, 2]
        for (key, label, color, _, _), off in zip(models, offsets):
            ys = [results[k].get(t, {}).get(key, {}).get('pearson_r', 0)
                  for t in targets]
            ax.bar(x + off * width, ys, width, label=label, color=color)
        ax.set_ylabel('Pearson r')
        ax.set_title(f'Predictive probing at horizon k={k}')
        ax.set_xticks(x)
        ax.set_xticklabels([PROBE_TARGETS[t]['name'] for t in targets],
                           rotation=25, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/probing_k{k}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {output_dir}/probing_k{k}.png")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predictive probing for JEPA encoder"
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='data_500')
    parser.add_argument('--output', type=str, default='figures/probing')
    parser.add_argument('--probe-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--horizons', type=int, nargs='+',
                        default=DEFAULT_HORIZONS)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print(f"\n  Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device,
                      weights_only=False)
    config = ckpt.get('config', {})

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    try:
        from ..models.world_model import OpticalWorldModel
        from ..training.dataset import create_dataloaders
    except ImportError:
        from models.world_model import OpticalWorldModel
        from training.dataset import create_dataloaders

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
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(args.device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    print(f"\n  Loading data from {args.data}...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data,
        context_length=config.get('context_length', 8),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    print(f"\n{'='*66}")
    print(f"  Predictive Probing Evaluation")
    print(f"{'='*66}")

    results = run_predictive_probing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        horizons=args.horizons,
        device=args.device,
        n_probe_epochs=args.probe_epochs,
        max_extract_batches=args.max_batches,
    )

    generate_report(results, args.output)

    print(f"\n{'='*66}")
    print(f"  Probing complete")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
