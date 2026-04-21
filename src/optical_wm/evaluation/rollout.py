"""
Multi-Step Rollout Evaluation (Éval 2) for Optical Network World Model.

Evaluates whether the predictor maintains faithful latent dynamics
over multiple autoregressive steps.

Protocol:
  1. Load trained JEPA checkpoint
  2. For each test episode, take a starting state
  3. Encode the real trajectory: z_1, z_2, ..., z_T (ground truth)
  4. Autoregressive rollout: ẑ_1=z_1, ẑ_{t+1} = predict(ẑ_{1:t}, a_{1:t})
  5. Compare ẑ_t vs z_t at each horizon h = t - t_start

Metrics:
  - MSE per horizon: MSE(ẑ_{t+h}, z_{t+h}) for h = 1, 2, ..., H
  - Rollout stability ratio: MSE(h=H) / MSE(h=1)
    LeWM reports 5.4× for JEPA vs 49× for LSTM
  - Probing on predictions: can we still recover GSNR from ẑ_{t+h}?
  - Per-action-type breakdown: which actions are hardest to predict?

Usage:
  python -m optical_wm.evaluation.rollout \
    --checkpoint checkpoints/v1/best.pt \
    --data data_100 \
    --max-horizon 25 \
    --output figures/rollout
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

try:
    from .probing import LinearProbe, MLPProbe, train_probe, PROBE_TARGETS
except ImportError:
    from probing import LinearProbe, MLPProbe, train_probe, PROBE_TARGETS


# =====================================================================
# Extract real embeddings for full episodes
# =====================================================================

@torch.no_grad()
def extract_episode_embeddings(
    model,
    data_loader,
    device: str = 'cpu',
    max_batches: int = None,
) -> List[Dict]:
    """
    Extract per-episode embedding sequences and actions.

    Returns list of dicts, each with:
      'z_real': [T, D]  — real encoder embeddings
      'actions': [T-1, A] — actions
      'global_features': [T, 8] — for probing targets
    """
    model.eval()
    episodes = []

    for i, batch in enumerate(data_loader):
        if max_batches and i >= max_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        actions = batch.pop('actions')  # [B, T-1, A]
        T = batch['global_features'].shape[1]
        B = actions.shape[0]

        # Encode each timestep
        z_seq = []
        for t in range(T):
            step_batch = {}
            for key, val in batch.items():
                if val.dim() >= 2 and val.shape[1] == T:
                    step_batch[key] = val[:, t]
                else:
                    step_batch[key] = val
            z_t = model.encoder(step_batch)
            z_seq.append(z_t)

        z_real = torch.stack(z_seq, dim=1)  # [B, T, D]

        # Store each sample in batch as separate episode
        for b in range(B):
            episodes.append({
                'z_real': z_real[b].cpu(),                    # [T, D]
                'actions': actions[b].cpu(),                   # [T-1, A]
                'global_features': batch['global_features'][b].cpu(),  # [T, 8]
            })

    return episodes


# =====================================================================
# Autoregressive rollout
# =====================================================================

@torch.no_grad()
def rollout_episode(
    model,
    z_init: torch.Tensor,
    actions: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Autoregressive rollout from z_init using actions.

    Args:
        model: world model with predictor
        z_init: [D] initial embedding
        actions: [H, A] action sequence
    Returns:
        z_pred: [H+1, D] predicted trajectory (z_pred[0] = z_init)
    """
    H = actions.shape[0]
    D = z_init.shape[0]

    # Use predictor's rollout method
    z_init_batch = z_init.unsqueeze(0).to(device)      # [1, D]
    actions_batch = actions.unsqueeze(0).to(device)     # [1, H, A]

    z_traj = model.predictor.rollout(z_init_batch, actions_batch)  # [1, H+1, D]

    return z_traj.squeeze(0).cpu()  # [H+1, D]


# =====================================================================
# Compute rollout metrics
# =====================================================================

def compute_rollout_metrics(
    model,
    episodes: List[Dict],
    max_horizon: int = 25,
    device: str = 'cpu',
) -> Dict:
    """
    Compute MSE per horizon across all episodes.

    Returns:
        dict with:
          'mse_per_horizon': {h: mean_mse}
          'stability_ratio': MSE(h_max) / MSE(h=1)
          'per_episode_mse': [[mse_h1, mse_h2, ...], ...]
    """
    mse_per_horizon = defaultdict(list)
    per_episode_mse = []

    n_episodes = len(episodes)
    print(f"\n  Rolling out {n_episodes} episodes (max horizon={max_horizon})...")

    for idx, ep in enumerate(episodes):
        z_real = ep['z_real']     # [T, D]
        actions = ep['actions']   # [T-1, A]
        T = z_real.shape[0]

        # Maximum rollout horizon for this episode
        H = min(max_horizon, T - 1)
        if H < 1:
            continue

        # Rollout from first state
        z_pred = rollout_episode(
            model, z_real[0], actions[:H], device
        )  # [H+1, D]

        # MSE at each horizon
        ep_mse = []
        for h in range(1, H + 1):
            mse = F.mse_loss(z_pred[h], z_real[h]).item()
            mse_per_horizon[h].append(mse)
            ep_mse.append(mse)

        per_episode_mse.append(ep_mse)

        if (idx + 1) % 100 == 0 or idx == n_episodes - 1:
            print(f"    [{idx+1}/{n_episodes}] h=1 MSE={mse_per_horizon[1][-1]:.4f}")

    # Average MSE per horizon
    avg_mse = {h: float(np.mean(vals)) for h, vals in sorted(mse_per_horizon.items())}
    std_mse = {h: float(np.std(vals)) for h, vals in sorted(mse_per_horizon.items())}

    # Stability ratio
    if 1 in avg_mse and max(avg_mse.keys()) > 1:
        h_max = max(avg_mse.keys())
        stability_ratio = avg_mse[h_max] / max(avg_mse[1], 1e-8)
    else:
        stability_ratio = 1.0

    return {
        'mse_per_horizon': avg_mse,
        'std_per_horizon': std_mse,
        'stability_ratio': stability_ratio,
        'h_max': max(avg_mse.keys()) if avg_mse else 0,
        'n_episodes': n_episodes,
        'per_episode_mse': per_episode_mse,
    }


# =====================================================================
# Baselines: linear-AR and persistence in latent space
# =====================================================================

def fit_linear_direct(episodes: List[Dict], max_horizon: int,
                      ridge: float = 1e-3) -> Dict[int, torch.Tensor]:
    """
    Direct multi-step linear fit. For each horizon k, train a separate
    predictor z_{t+k} = [z_t | a_{t:t+k-1} | 1] @ W_k.

    This avoids the autoregressive explosion of a compounded 1-step model:
    the fit is a ceiling for what a linear map can do at horizon k given
    the initial embedding and the full action sequence.

    Returns:
        {k: W_k of shape [D + k*A + 1, D]}
    """
    Ws: Dict[int, torch.Tensor] = {}
    for k in range(1, max_horizon + 1):
        X_list, Y_list = [], []
        for ep in episodes:
            z = ep['z_real']      # [T, D]
            a = ep['actions']     # [T-1, A]
            T = z.shape[0]
            if T <= k:
                continue
            for t in range(T - k):
                X_list.append(torch.cat([z[t], a[t:t + k].reshape(-1)], dim=-1))
                Y_list.append(z[t + k])
        if not X_list:
            continue
        X = torch.stack(X_list, dim=0)
        Y = torch.stack(Y_list, dim=0)
        ones = torch.ones(X.shape[0], 1, dtype=X.dtype)
        X = torch.cat([X, ones], dim=-1)
        XTX = X.T @ X
        reg = ridge * torch.eye(XTX.shape[0], dtype=XTX.dtype)
        Ws[k] = torch.linalg.solve(XTX + reg, X.T @ Y)
    return Ws


def predict_linear_direct(W_k: torch.Tensor, z_0: torch.Tensor,
                          actions_k: torch.Tensor) -> torch.Tensor:
    """
    Args:
        W_k:       [D + k*A + 1, D]
        z_0:       [D]
        actions_k: [k, A]
    Returns:
        z_hat_k:   [D]
    """
    x = torch.cat([z_0, actions_k.reshape(-1), torch.ones(1, dtype=z_0.dtype)], dim=-1)
    return x @ W_k


def compute_baseline_rollouts(
    episodes: List[Dict], Ws: Dict[int, torch.Tensor], max_horizon: int
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate direct-linear and persistence baselines on val windows.

    Returns:
        {
            'linear_direct': {k: mean_mse},
            'persistence':   {k: mean_mse},
        }
    """
    per_h_lin = defaultdict(list)
    per_h_pers = defaultdict(list)

    for ep in episodes:
        z_real = ep['z_real']
        actions = ep['actions']
        T = z_real.shape[0]
        H = min(max_horizon, T - 1)
        if H < 1:
            continue
        z_0 = z_real[0]
        for k in range(1, H + 1):
            if k not in Ws:
                continue
            z_hat = predict_linear_direct(Ws[k], z_0, actions[:k])
            per_h_lin[k].append(F.mse_loss(z_hat, z_real[k]).item())
            per_h_pers[k].append(F.mse_loss(z_0, z_real[k]).item())

    return {
        'linear_direct': {h: float(np.mean(v)) for h, v in sorted(per_h_lin.items())},
        'persistence':   {h: float(np.mean(v)) for h, v in sorted(per_h_pers.items())},
    }


# =====================================================================
# Probing on predicted embeddings
# =====================================================================

def probe_at_horizons(
    model,
    episodes: List[Dict],
    horizons: List[int] = [1, 5, 10, 15, 25],
    device: str = 'cpu',
    n_probe_epochs: int = 50,
) -> Dict:
    """
    Train probes on real embeddings, evaluate on predicted embeddings
    at various horizons.

    This shows whether physical quantities remain recoverable
    from autoregressive predictions.
    """
    print(f"\n  Probing on predicted embeddings at horizons {horizons}...")

    # Collect real embeddings for probe training
    z_all_real = []
    gf_all_real = []
    for ep in episodes:
        for t in range(ep['z_real'].shape[0]):
            z_all_real.append(ep['z_real'][t])
            gf_all_real.append(ep['global_features'][t])

    z_train = torch.stack(z_all_real)      # [N, D]
    gf_train = torch.stack(gf_all_real)    # [N, 8]
    latent_dim = z_train.shape[1]

    # For each horizon, collect predicted embeddings + real targets
    results = {}

    for h in horizons:
        # Collect predicted z at horizon h and corresponding real global_features
        z_pred_h = []
        gf_real_h = []

        for ep in episodes:
            T = ep['z_real'].shape[0]
            if h >= T:
                continue

            # Rollout from step 0
            z_pred = rollout_episode(
                model, ep['z_real'][0], ep['actions'][:h], device
            )  # [h+1, D]

            z_pred_h.append(z_pred[h])            # predicted z at horizon h
            gf_real_h.append(ep['global_features'][h])  # real target at horizon h

        if len(z_pred_h) < 10:
            print(f"    h={h}: too few samples ({len(z_pred_h)}), skipping")
            continue

        z_pred_tensor = torch.stack(z_pred_h)   # [M, D]
        gf_tensor = torch.stack(gf_real_h)       # [M, 8]

        # Split into train/val for the probe
        n_val = max(1, len(z_pred_tensor) // 5)
        z_p_val = z_pred_tensor[:n_val]
        gf_p_val = gf_tensor[:n_val]

        results[h] = {}

        for target_key, target_info in PROBE_TARGETS.items():
            idx = target_info['idx']
            y_train = gf_train[:, idx]
            y_val = gf_p_val[:, idx]

            if y_train.std() < 1e-6:
                continue

            # Train probe on REAL embeddings
            probe = LinearProbe(latent_dim)
            metrics = train_probe(
                probe,
                z_train, y_train,           # train on real
                z_p_val, y_val,             # evaluate on PREDICTED
                n_epochs=n_probe_epochs,
                device=device,
            )
            results[h][target_key] = metrics['pearson_r']

    # Print results
    targets_to_show = [t for t in PROBE_TARGETS if any(
        t in results.get(h, {}) for h in horizons
    )]

    if targets_to_show:
        header = f"  {'Target':<22}" + "".join(f"  h={h:>2}" for h in horizons if h in results)
        print(f"\n{header}")
        print(f"  {'-'*22}" + "".join(f"  {'-'*5}" for h in horizons if h in results))

        for target in targets_to_show:
            line = f"  {target:<22}"
            for h in horizons:
                if h in results and target in results[h]:
                    r = results[h][target]
                    line += f"  {r:5.3f}"
                else:
                    line += f"    n/a"
            print(line)

    return results


# =====================================================================
# Generate report and figures
# =====================================================================

def generate_report(
    rollout_metrics: Dict,
    probing_results: Dict = None,
    output_dir: str = "figures/rollout",
    baseline_metrics: Dict = None,
):
    """Save results and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    report = {
        'rollout': {
            'mse_per_horizon': rollout_metrics['mse_per_horizon'],
            'std_per_horizon': rollout_metrics['std_per_horizon'],
            'stability_ratio': rollout_metrics['stability_ratio'],
            'n_episodes': rollout_metrics['n_episodes'],
        },
    }
    if baseline_metrics:
        report['baselines'] = baseline_metrics
    if probing_results:
        report['probing_vs_horizon'] = {
            str(h): v for h, v in probing_results.items()
        }

    with open(f"{output_dir}/rollout_results.json", 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {output_dir}/rollout_results.json")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: MSE vs Horizon (the key curve)
        horizons = sorted(rollout_metrics['mse_per_horizon'].keys())
        mse_vals = [rollout_metrics['mse_per_horizon'][h] for h in horizons]
        std_vals = [rollout_metrics['std_per_horizon'][h] for h in horizons]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(horizons, mse_vals, 'o-', color='#065A82', linewidth=2,
                markersize=4, label='JEPA')
        ax.fill_between(
            horizons,
            [m - s for m, s in zip(mse_vals, std_vals)],
            [m + s for m, s in zip(mse_vals, std_vals)],
            alpha=0.15, color='#065A82',
        )
        if baseline_metrics:
            lin = baseline_metrics.get('linear_direct', {})
            pers = baseline_metrics.get('persistence', {})
            if lin:
                hs = sorted(lin.keys())
                ax.plot(hs, [lin[h] for h in hs], 's-', color='#D97706',
                        linewidth=2, markersize=4, label='Linear (direct k)')
            if pers:
                hs = sorted(pers.keys())
                ax.plot(hs, [pers[h] for h in hs], '^--', color='#6B7280',
                        linewidth=2, markersize=4, label='Persistence')
        ax.set_xlabel('Prediction Horizon (steps)')
        ax.set_ylabel('MSE (latent space)')
        ax.set_title('Multi-Step Prediction Error vs Horizon')
        ax.legend()
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/rollout_mse_vs_horizon.png", dpi=150)
        fig.savefig(f"{output_dir}/rollout_mse_vs_horizon.pdf")
        plt.close(fig)
        print(f"  Saved: rollout_mse_vs_horizon.png")

        # Figure 2: Probing accuracy vs Horizon
        if probing_results:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            probe_horizons = sorted(probing_results.keys())

            colors = {
                'n_active_lps': '#065A82',
                'avg_margin': '#059669',
                'worst_margin': '#D97706',
                'spectral_util': '#DC2626',
                'total_capacity': '#7C3AED',
            }

            for target in colors:
                if any(target in probing_results.get(h, {}) for h in probe_horizons):
                    vals = [probing_results[h].get(target, float('nan'))
                            for h in probe_horizons]
                    name = PROBE_TARGETS[target]['name']
                    ax.plot(probe_horizons, vals, 'o-', color=colors[target],
                            linewidth=2, markersize=4, label=name)

            ax.set_xlabel('Prediction Horizon (steps)')
            ax.set_ylabel('Probing Pearson r')
            ax.set_title('Physical Quantity Recovery from Predicted Embeddings')
            ax.set_ylim(0, 1.05)
            ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
            ax.legend(fontsize=8, loc='lower left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{output_dir}/probing_vs_horizon.png", dpi=150)
            fig.savefig(f"{output_dir}/probing_vs_horizon.pdf")
            plt.close(fig)
            print(f"  Saved: probing_vs_horizon.png")

    except ImportError:
        print("  matplotlib not available — skipping plots")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-step rollout evaluation for JEPA world model"
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='data_100')
    parser.add_argument('--output', type=str, default='figures/rollout')
    parser.add_argument('--max-horizon', type=int, default=25,
                        help='Maximum rollout horizon')
    parser.add_argument('--probe-horizons', type=str, default='1,3,5,7',
                        help='Comma-separated horizons for probing')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Limit val episodes for quick test')
    parser.add_argument('--max-train-batches', type=int, default=None,
                        help='Limit train batches used to fit linear-AR baseline')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-probing', action='store_true',
                        help='Skip probing on predictions')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    probe_horizons = [int(h) for h in args.probe_horizons.split(',')]

    # ── Load model ──
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

    # ── Load data ──
    print(f"\n  Loading data from {args.data}...")
    context_len = min(config.get('context_length', 8), args.max_horizon + 1)
    # Use longer context for rollout evaluation
    rollout_context = min(args.max_horizon + 1, 32)

    train_loader, val_loader = create_dataloaders(
        data_dir=args.data,
        context_length=rollout_context,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    # ── Extract episodes ──
    max_batches = args.max_episodes // args.batch_size + 1 if args.max_episodes else None
    episodes = extract_episode_embeddings(
        model, val_loader, args.device, max_batches=max_batches
    )
    print(f"  Extracted {len(episodes)} val episode windows")

    # Train episodes for linear-AR fit (separate pass, smaller cap)
    train_cap = args.max_train_batches if hasattr(args, 'max_train_batches') else None
    train_episodes = extract_episode_embeddings(
        model, train_loader, args.device, max_batches=train_cap
    )
    print(f"  Extracted {len(train_episodes)} train episode windows for baseline fit")

    # ── Rollout metrics ──
    print(f"\n{'='*60}")
    print(f"  Multi-Step Rollout Evaluation")
    print(f"{'='*60}")

    rollout_metrics = compute_rollout_metrics(
        model, episodes, max_horizon=args.max_horizon, device=args.device
    )

    # Fit direct-linear on train embeddings, evaluate baselines on val windows
    print(f"\n  Fitting direct-linear baselines (per horizon) on "
          f"{len(train_episodes)} train windows...")
    Ws = fit_linear_direct(train_episodes, max_horizon=args.max_horizon)
    baseline_metrics = compute_baseline_rollouts(
        episodes, Ws, max_horizon=args.max_horizon
    )

    # Print summary
    print(f"\n  ── MSE per Horizon ──")
    print(f"  {'h':>4} {'JEPA':>10} {'linear-dir':>11} {'persist':>10} {'ratio J/L':>10}")
    print(f"  {'-'*4} {'-'*10} {'-'*11} {'-'*10} {'-'*10}")
    lin = baseline_metrics['linear_direct']
    pers = baseline_metrics['persistence']
    for h in sorted(rollout_metrics['mse_per_horizon'].keys()):
        mse = rollout_metrics['mse_per_horizon'][h]
        l = lin.get(h, float('nan'))
        p = pers.get(h, float('nan'))
        ratio = mse / max(l, 1e-12) if l == l else float('nan')
        print(f"  {h:>4} {mse:>10.5f} {l:>10.5f} {p:>10.5f} {ratio:>10.3f}")

    ratio = rollout_metrics['stability_ratio']
    h_max = rollout_metrics['h_max']
    print(f"\n  Stability ratio (h={h_max}/h=1): {ratio:.1f}×")

    if ratio < 10:
        print(f"  ✅ Stable rollout (LeWM JEPA: 5.4×)")
    elif ratio < 50:
        print(f"  ⚠  Moderate degradation (LeWM LSTM: 49×)")
    else:
        print(f"  ✗  Unstable rollout (LeWM Transformer: 147×)")

    # ── Probing on predictions ──
    probing_results = None
    if not args.no_probing:
        # Filter horizons to what's available
        valid_horizons = [h for h in probe_horizons
                          if h <= args.max_horizon]
        if valid_horizons:
            probing_results = probe_at_horizons(
                model, episodes, horizons=valid_horizons,
                device=args.device, n_probe_epochs=50,
            )

    # ── Generate report ──
    generate_report(rollout_metrics, probing_results, args.output,
                    baseline_metrics=baseline_metrics)

    print(f"\n{'='*60}")
    print(f"  Rollout evaluation complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()