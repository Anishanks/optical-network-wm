"""
Counterfactual Action Ranking Evaluation (simple planning PoC).

Tests whether the world model can discriminate the *real* action sequence
recorded in the dataset from K *random* action sequences drawn from the
empirical pool, using a trained probe as the scoring oracle.

Protocol:
  1. Extract val starting states + their real H-step action sequences
  2. Build a pool of H-length action windows from train episodes
  3. For each val state:
       - Rollout real actions via predictor -> z_H_real
       - Rollout K random action sequences    -> z_H_alt[1..K]
       - Score every terminal latent with probe(z) -> predicted target
       - Rank the real score among the K alternatives
  4. Report:
       - Distribution of rank percentiles (lower = real ranks higher)
       - Mean predicted score: real vs random
       - Correlation between predicted and true target at horizon H

Interpretation:
  If real action sequences (produced by margin-aware / provisioning-aware
  policies) score higher than random alternatives on average, the model
  has planning-useful discriminative ability -- a prerequisite for any
  MPC / random-shooting planner -- without needing a real simulator in
  the loop.

Usage:
  python -m optical_wm.evaluation.planning \
    --checkpoint checkpoints/v2k_gnn_ctx16/best.pt \
    --data data_2k \
    --horizon 5 --n-alternatives 50 \
    --target avg_margin \
    --output figures/planning
"""
import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import os
import sys
from typing import Dict, List, Tuple

try:
    from .rollout import extract_episode_embeddings, rollout_episode
    from .probing import LinearProbe, MLPProbe, train_probe, PROBE_TARGETS
except ImportError:
    from rollout import extract_episode_embeddings, rollout_episode
    from probing import LinearProbe, MLPProbe, train_probe, PROBE_TARGETS


# =====================================================================
# Action pool from train episodes
# =====================================================================

def build_action_pool(
    episodes: List[Dict], horizon: int, max_windows: int = 20000,
) -> torch.Tensor:
    """
    Build a pool of H-length action windows from episodes.
    Returns [N, H, A] tensor.
    """
    windows = []
    for ep in episodes:
        actions = ep['actions']           # [T-1, A]
        n = actions.shape[0]
        if n < horizon:
            continue
        for t in range(n - horizon + 1):
            windows.append(actions[t:t + horizon])
            if len(windows) >= max_windows:
                break
        if len(windows) >= max_windows:
            break
    if not windows:
        raise RuntimeError("Empty action pool: episodes too short for horizon")
    return torch.stack(windows)           # [N, H, A]


# =====================================================================
# Probe training on real (z_t, y_t) pairs
# =====================================================================

def train_target_probe(
    train_episodes: List[Dict],
    val_episodes: List[Dict],
    target: str = 'avg_margin',
    probe_type: str = 'mlp',
    n_epochs: int = 100,
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict]:
    """Train a probe z -> global_features[target] on real encoder outputs."""
    idx = PROBE_TARGETS[target]['idx']

    def collect(eps):
        zs, ys = [], []
        for ep in eps:
            z = ep['z_real']                # [T, D]
            gf = ep['global_features']      # [T, 8]
            for t in range(z.shape[0]):
                zs.append(z[t])
                ys.append(gf[t, idx])
        return torch.stack(zs), torch.stack(ys)

    z_train, y_train = collect(train_episodes)
    z_val, y_val = collect(val_episodes)

    D = z_train.shape[1]
    probe_cls = MLPProbe if probe_type == 'mlp' else LinearProbe
    probe = probe_cls(D)
    metrics = train_probe(
        probe, z_train, y_train, z_val, y_val,
        n_epochs=n_epochs, device=device,
    )
    return probe, metrics


# =====================================================================
# Rank real-vs-random
# =====================================================================

@torch.no_grad()
def rank_real_vs_random(
    model,
    probe: nn.Module,
    val_episodes: List[Dict],
    action_pool: torch.Tensor,
    horizon: int = 5,
    n_alternatives: int = 50,
    target: str = 'avg_margin',
    maximize: bool = True,
    device: str = 'cpu',
    seed: int = 0,
) -> Dict:
    """
    For each val episode:
      - Rollout real actions for `horizon` steps, score terminal z
      - Rollout K random action sequences from pool, score terminal z
      - Compute rank percentile of real among alternatives

    Rank percentile convention:
       0.0 -> real ranks best, 1.0 -> real ranks worst among K+1.
    """
    idx_target = PROBE_TARGETS[target]['idx']
    rng = np.random.default_rng(seed)
    N_pool = action_pool.shape[0]

    model.eval()
    probe = probe.to(device).eval()

    real_scores: List[float] = []
    rand_mean_scores: List[float] = []
    rand_best_scores: List[float] = []
    rand_dispersion: List[float] = []
    rank_percentiles: List[float] = []
    real_true: List[float] = []

    n_skipped = 0
    for ep in val_episodes:
        z_real = ep['z_real']             # [T, D]
        actions = ep['actions']           # [T-1, A]
        T = z_real.shape[0]
        if T - 1 < horizon:
            n_skipped += 1
            continue

        z_0 = z_real[0]
        real_actions = actions[:horizon]

        # Rollout real
        z_pred_real = rollout_episode(model, z_0, real_actions, device)  # [H+1, D]
        score_real = probe(
            z_pred_real[-1].unsqueeze(0).to(device)
        ).item()

        # Rollout K random alternatives in a batched call
        sample_idx = rng.choice(N_pool, size=n_alternatives, replace=False)
        alt_actions = action_pool[sample_idx].to(device)                   # [K, H, A]
        z_init_batch = z_0.unsqueeze(0).expand(n_alternatives, -1).to(device)
        z_traj_batch = model.predictor.rollout(z_init_batch, alt_actions)  # [K, H+1, D]
        z_final_batch = z_traj_batch[:, -1]                                # [K, D]
        scores_alt = probe(z_final_batch).cpu().numpy()                    # [K]

        # Rank percentile
        if maximize:
            n_better = (scores_alt > score_real).sum()
        else:
            n_better = (scores_alt < score_real).sum()
        rank_pct = n_better / n_alternatives

        real_scores.append(score_real)
        rand_mean_scores.append(float(scores_alt.mean()))
        rand_best_scores.append(
            float(scores_alt.max() if maximize else scores_alt.min())
        )
        rand_dispersion.append(float(scores_alt.std()))
        rank_percentiles.append(float(rank_pct))

        if horizon < T:
            real_true.append(float(ep['global_features'][horizon, idx_target]))

    return {
        'real_scores': real_scores,
        'rand_mean_scores': rand_mean_scores,
        'rand_best_scores': rand_best_scores,
        'rand_dispersion': rand_dispersion,
        'rank_percentiles': rank_percentiles,
        'real_true_values': real_true,
        'n_evaluated': len(real_scores),
        'n_skipped': n_skipped,
    }


# =====================================================================
# Reporting
# =====================================================================

def print_report(
    results: Dict, probe_metrics: Dict,
    target: str, horizon: int, n_alt: int, maximize: bool,
):
    real = np.asarray(results['real_scores'])
    rand_mean = np.asarray(results['rand_mean_scores'])
    rand_best = np.asarray(results['rand_best_scores'])
    ranks = np.asarray(results['rank_percentiles'])
    real_true = np.asarray(results['real_true_values'])

    direction = 'maximize' if maximize else 'minimize'
    name = PROBE_TARGETS[target]['name']

    print(f"\n  ── Probe quality (on real (z, y) pairs) ──")
    print(f"    target:         {name}")
    print(f"    probe val MSE:  {probe_metrics['mse']:.4f}")
    print(f"    probe val r:    {probe_metrics['pearson_r']:.3f}")
    print(f"    probe val R2:   {probe_metrics['r_squared']:.3f}")

    print(f"\n  ── Action Ranking (h={horizon}, K={n_alt}, {direction}) ──")
    print(f"    n_evaluated:    {results['n_evaluated']}")
    print(f"    n_skipped:      {results['n_skipped']} (episodes too short)")

    if results['n_evaluated'] == 0:
        print("    No usable episodes; aborting report")
        return

    print(f"\n    Predicted {name} (via probe on terminal latent):")
    print(f"      real actions:     mean={real.mean():.4f}  "
          f"median={np.median(real):.4f}")
    print(f"      random actions:   mean={rand_mean.mean():.4f}  "
          f"median={np.median(rand_mean):.4f}")
    print(f"      random best-of-K: mean={rand_best.mean():.4f}  "
          f"median={np.median(rand_best):.4f}")

    gap = (real.mean() - rand_mean.mean())
    gap_sign = '+' if gap >= 0 else ''
    print(f"      real - random mean gap: {gap_sign}{gap:.4f}")

    print(f"\n    Rank percentile of real actions (0 = best of K+1):")
    print(f"      mean:           {ranks.mean():.3f}")
    print(f"      median:         {np.median(ranks):.3f}")
    print(f"      top-10%  rate:  {(ranks < 0.10).mean():.0%}")
    print(f"      top-25%  rate:  {(ranks < 0.25).mean():.0%}")
    print(f"      below-median:   {(ranks < 0.50).mean():.0%}")

    # Ground-truth correlation: is the probe's terminal score aligned
    # with the *actual* observed target at horizon?
    if len(real_true) == len(real) and len(real_true) > 2:
        if real.std() > 1e-8 and real_true.std() > 1e-8:
            r = float(np.corrcoef(real, real_true)[0, 1])
            print(f"\n    Pearson r (predicted vs real ground-truth "
                  f"{name} at h={horizon}):")
            print(f"      r = {r:.3f}")

    # Narrative verdict
    mean_rank = ranks.mean()
    print(f"\n  ── Verdict ──")
    if maximize and mean_rank < 0.25 and gap > 0:
        print("    ✅ Model ranks real actions in top 25% on average -- "
              "planner-ready discriminative ability")
    elif (not maximize) and mean_rank < 0.25 and gap < 0:
        print("    ✅ Model ranks real actions in top 25% on average -- "
              "planner-ready discriminative ability")
    elif mean_rank < 0.50:
        print("    ⚠  Model weakly discriminates real from random actions")
    else:
        print("    ✗  Model does not distinguish real actions from random "
              "alternatives under this objective")


def save_report(
    results: Dict, probe_metrics: Dict,
    target: str, horizon: int, n_alt: int, maximize: bool,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    real = np.asarray(results['real_scores'])
    rand_mean = np.asarray(results['rand_mean_scores'])
    rand_best = np.asarray(results['rand_best_scores'])
    ranks = np.asarray(results['rank_percentiles'])
    real_true = np.asarray(results['real_true_values'])

    summary = {
        'target': target,
        'horizon': horizon,
        'n_alternatives': n_alt,
        'maximize': maximize,
        'n_evaluated': results['n_evaluated'],
        'n_skipped': results['n_skipped'],
        'probe_val_metrics': probe_metrics,
        'predicted_real_mean': float(real.mean()) if real.size else None,
        'predicted_random_mean': float(rand_mean.mean()) if rand_mean.size else None,
        'predicted_random_best_mean': float(rand_best.mean()) if rand_best.size else None,
        'rank_percentile_mean': float(ranks.mean()) if ranks.size else None,
        'rank_percentile_median': float(np.median(ranks)) if ranks.size else None,
        'top10_rate': float((ranks < 0.10).mean()) if ranks.size else None,
        'top25_rate': float((ranks < 0.25).mean()) if ranks.size else None,
        'below_median_rate': float((ranks < 0.50).mean()) if ranks.size else None,
    }

    if len(real_true) == len(real) and len(real_true) > 2 and \
            real.std() > 1e-8 and real_true.std() > 1e-8:
        summary['predicted_vs_truth_pearson_r'] = float(
            np.corrcoef(real, real_true)[0, 1]
        )

    with open(f"{output_dir}/planning_results.json", 'w') as f:
        json.dump({
            'summary': summary,
            'per_episode': {
                'real_scores': results['real_scores'],
                'rand_mean_scores': results['rand_mean_scores'],
                'rand_best_scores': results['rand_best_scores'],
                'rank_percentiles': results['rank_percentiles'],
                'real_true_values': results['real_true_values'],
            },
        }, f, indent=2)
    print(f"\n  Saved: {output_dir}/planning_results.json")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: predicted score distributions (real vs random-mean vs random-best)
        ax = axes[0]
        name = PROBE_TARGETS[target]['name']
        ax.hist(rand_mean, bins=25, color='#9CA3AF', alpha=0.6,
                label='random (mean of K)', edgecolor='white')
        ax.hist(rand_best, bins=25, color='#FBBF24', alpha=0.5,
                label=f'random (best of K={n_alt})', edgecolor='white')
        ax.hist(real, bins=25, color='#065A82', alpha=0.8,
                label='real actions', edgecolor='white')
        ax.set_xlabel(f'Predicted {name} at h={horizon}')
        ax.set_ylabel('Episodes')
        ax.set_title('Score distribution: real vs random action sequences')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Right: rank percentile CDF
        ax = axes[1]
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        ax.plot(sorted_ranks, cdf, '-', color='#065A82', linewidth=2,
                label='model')
        # Baseline: uniform distribution over ranks
        ax.plot([0, 1], [0, 1], '--', color='#6B7280',
                alpha=0.6, label='uniform (no discrimination)')
        ax.axvline(0.10, color='#DC2626', linestyle=':', alpha=0.5,
                   label='top 10%')
        ax.axvline(0.25, color='#D97706', linestyle=':', alpha=0.5,
                   label='top 25%')
        ax.set_xlabel('Rank percentile of real (0 = best)')
        ax.set_ylabel('Fraction of episodes ≤ x')
        ax.set_title('CDF of real-action rank among K random alternatives')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        fig.savefig(f"{output_dir}/planning_action_ranking.png", dpi=150)
        fig.savefig(f"{output_dir}/planning_action_ranking.pdf")
        plt.close(fig)
        print(f"  Saved: {output_dir}/planning_action_ranking.png")

    except ImportError:
        print("  matplotlib not available -- skipping plots")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual action ranking evaluation"
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='figures/planning')
    parser.add_argument('--target', type=str, default='avg_margin',
                        choices=list(PROBE_TARGETS.keys()))
    parser.add_argument('--horizon', type=int, default=5,
                        help='Planning horizon (action sequence length)')
    parser.add_argument('--n-alternatives', type=int, default=50,
                        help='Random action sequences sampled per episode')
    parser.add_argument('--max-val-episodes', type=int, default=200)
    parser.add_argument('--max-train-batches', type=int, default=None,
                        help='Cap train batches used for probe + pool')
    parser.add_argument('--max-pool-windows', type=int, default=20000)
    parser.add_argument('--probe-type', type=str, default='mlp',
                        choices=['linear', 'mlp'])
    parser.add_argument('--probe-epochs', type=int, default=100)
    parser.add_argument('--minimize', action='store_true',
                        help='Minimize target instead of maximize '
                             '(e.g. for n_infeasible)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

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

    # ── Data ──
    print(f"\n  Loading data from {args.data}...")
    rollout_context = max(args.horizon + 1, 8)
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data,
        context_length=rollout_context,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    # ── Extract episodes ──
    val_max_batches = (args.max_val_episodes // args.batch_size + 1
                       if args.max_val_episodes else None)
    val_episodes = extract_episode_embeddings(
        model, val_loader, args.device, max_batches=val_max_batches,
    )
    print(f"  Extracted {len(val_episodes)} val episode windows")

    train_episodes = extract_episode_embeddings(
        model, train_loader, args.device, max_batches=args.max_train_batches,
    )
    print(f"  Extracted {len(train_episodes)} train episode windows")

    # ── Train probe ──
    print(f"\n  Training {args.probe_type} probe on '{args.target}'...")
    probe, probe_metrics = train_target_probe(
        train_episodes, val_episodes,
        target=args.target, probe_type=args.probe_type,
        n_epochs=args.probe_epochs, device=args.device,
    )
    print(f"    val MSE={probe_metrics['mse']:.4f} "
          f"r={probe_metrics['pearson_r']:.3f} "
          f"R2={probe_metrics['r_squared']:.3f}")

    # ── Action pool ──
    print(f"\n  Building action pool (horizon={args.horizon}, "
          f"max_windows={args.max_pool_windows})...")
    action_pool = build_action_pool(
        train_episodes, args.horizon, max_windows=args.max_pool_windows,
    )
    print(f"    pool size: {action_pool.shape[0]} action windows, "
          f"A={action_pool.shape[-1]}")

    # ── Run ranking ──
    print(f"\n{'='*60}")
    print(f"  Counterfactual Action Ranking")
    print(f"{'='*60}")

    results = rank_real_vs_random(
        model, probe, val_episodes, action_pool,
        horizon=args.horizon,
        n_alternatives=args.n_alternatives,
        target=args.target,
        maximize=not args.minimize,
        device=args.device,
        seed=args.seed,
    )

    print_report(
        results, probe_metrics,
        target=args.target, horizon=args.horizon,
        n_alt=args.n_alternatives, maximize=not args.minimize,
    )
    save_report(
        results, probe_metrics,
        target=args.target, horizon=args.horizon,
        n_alt=args.n_alternatives, maximize=not args.minimize,
        output_dir=args.output,
    )

    print(f"\n{'='*60}")
    print(f"  Action ranking evaluation complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
