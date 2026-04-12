"""
Goal-Conditioned Planning Evaluation (Éval 3) for Optical Network World Model.

Evaluates whether the world model can find action sequences that reach
a goal state, using Cross-Entropy Method (CEM) in latent space.

Protocol:
  1. Sample (start_state, goal_state) pairs from dataset (separated by K steps)
  2. Encode both: z_start = encoder(start), z_goal = encoder(goal)
  3. CEM searches for action sequence minimizing distance(ẑ_final, z_goal)
  4. Evaluate success rate at different thresholds

Metrics:
  - Latent success rate: distance(ẑ_final, z_goal) < threshold
  - Physical success rate: probed quantities within ±10% of goal
  - Planning speedup: wall-clock CEM time vs horizon steps × GNPy time

Usage:
  python -m optical_wm.evaluation.planning \
    --checkpoint checkpoints/v1/best.pt \
    --data data_100 \
    --output figures/planning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

try:
    from .probing import LinearProbe, train_probe, PROBE_TARGETS
except ImportError:
    from probing import LinearProbe, train_probe, PROBE_TARGETS


# =====================================================================
# CEM Planner
# =====================================================================

class CEMPlanner:
    """
    Cross-Entropy Method planner in latent space.

    Searches for an action sequence that drives the predicted
    latent state from z_start toward z_goal.
    """

    def __init__(
        self,
        action_dim: int = 20,
        horizon: int = 15,
        population_size: int = 64,
        elite_frac: float = 0.2,
        n_iterations: int = 5,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.population_size = population_size
        self.n_elite = max(1, int(population_size * elite_frac))
        self.n_iterations = n_iterations

        # Prior distribution over actions
        self.action_mean = action_mean if action_mean is not None else torch.zeros(action_dim)
        self.action_std = action_std if action_std is not None else torch.ones(action_dim)

    @torch.no_grad()
    def plan(
        self,
        model,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        device: str = 'cpu',
    ) -> Tuple[torch.Tensor, float]:
        """
        Find action sequence to reach z_goal from z_start.

        Args:
            model: world model with predictor.rollout()
            z_start: [D] initial latent state
            z_goal: [D] goal latent state
            device: computation device
        Returns:
            best_actions: [H, A] best action sequence found
            best_distance: float, final latent distance to goal
        """
        H = self.horizon
        A = self.action_dim
        N = self.population_size
        D = z_start.shape[0]

        z_start = z_start.to(device)
        z_goal = z_goal.to(device)

        # Initialize distribution: mean and std for each (horizon, action_dim)
        mu = self.action_mean.unsqueeze(0).expand(H, A).clone().to(device)   # [H, A]
        sigma = self.action_std.unsqueeze(0).expand(H, A).clone().to(device) # [H, A]

        best_actions = None
        best_distance = float('inf')

        for iteration in range(self.n_iterations):
            # Sample action sequences from current distribution
            # [N, H, A]
            noise = torch.randn(N, H, A, device=device)
            action_seqs = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

            # Rollout each sequence
            z_init_batch = z_start.unsqueeze(0).expand(N, D)  # [N, D]
            z_traj = model.predictor.rollout(z_init_batch, action_seqs)  # [N, H+1, D]

            # Evaluate: distance of final predicted state to goal
            z_final = z_traj[:, -1, :]  # [N, D]
            distances = torch.norm(z_final - z_goal.unsqueeze(0), dim=-1)  # [N]

            # Select elite
            elite_idx = distances.argsort()[:self.n_elite]
            elite_actions = action_seqs[elite_idx]  # [K, H, A]
            elite_dist = distances[elite_idx]

            # Update best
            if elite_dist[0].item() < best_distance:
                best_distance = elite_dist[0].item()
                best_actions = elite_actions[0].cpu()

            # Update distribution from elite
            mu = elite_actions.mean(dim=0)      # [H, A]
            sigma = elite_actions.std(dim=0).clamp(min=0.01)  # [H, A]

        return best_actions, best_distance


# =====================================================================
# Compute action statistics from dataset
# =====================================================================

def compute_action_stats(data_loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of actions across the dataset.
    Used as prior for CEM sampling.
    """
    all_actions = []
    for batch in data_loader:
        actions = batch['actions']  # [B, T-1, A]
        all_actions.append(actions.reshape(-1, actions.shape[-1]))

    all_actions = torch.cat(all_actions, dim=0)  # [N, A]
    return all_actions.mean(dim=0), all_actions.std(dim=0).clamp(min=0.1)


# =====================================================================
# Sample goal pairs from dataset
# =====================================================================

@torch.no_grad()
def sample_goal_pairs(
    model,
    data_loader,
    n_pairs: int = 50,
    min_gap: int = 10,
    max_gap: int = 25,
    device: str = 'cpu',
) -> List[Dict]:
    """
    Sample (start, goal) pairs from the dataset.

    For each pair:
      - start and goal are from the same episode
      - separated by min_gap to max_gap steps
      - we store z_start, z_goal, real actions, and real global_features

    Returns list of dicts.
    """
    model.eval()
    pairs = []

    for batch in data_loader:
        if len(pairs) >= n_pairs:
            break

        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        actions = batch_dev.pop('actions')  # [B, T-1, A]
        T = batch_dev['global_features'].shape[1]
        B = actions.shape[0]

        # Encode all timesteps
        z_seq = []
        for t in range(T):
            step_batch = {}
            for key, val in batch_dev.items():
                if val.dim() >= 2 and val.shape[1] == T:
                    step_batch[key] = val[:, t]
                else:
                    step_batch[key] = val
            z_t = model.encoder(step_batch)
            z_seq.append(z_t)

        z_all = torch.stack(z_seq, dim=1)  # [B, T, D]

        for b in range(B):
            if len(pairs) >= n_pairs:
                break

            # Sample a gap
            gap = np.random.randint(min_gap, min(max_gap + 1, T))
            t_start = 0
            t_goal = t_start + gap

            if t_goal >= T:
                continue

            pairs.append({
                'z_start': z_all[b, t_start].cpu(),
                'z_goal': z_all[b, t_goal].cpu(),
                'real_actions': actions[b, t_start:t_goal].cpu(),  # [gap, A]
                'gf_start': batch['global_features'][b, t_start].cpu(),
                'gf_goal': batch['global_features'][b, t_goal].cpu(),
                'gap': gap,
            })

    return pairs


# =====================================================================
# Train probes for physical success evaluation
# =====================================================================

def train_probes_for_planning(
    model, data_loader, device: str = 'cpu',
) -> Dict[str, nn.Module]:
    """Train linear probes on real embeddings for each physical quantity."""
    # Extract embeddings
    all_z, all_gf = [], []
    with torch.no_grad():                        
        for batch in data_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            batch_dev.pop('actions', None)
            T = batch_dev['global_features'].shape[1]

            for t in range(T):
                step_batch = {}
                for key, val in batch_dev.items():
                    if val.dim() >= 2 and val.shape[1] == T:
                        step_batch[key] = val[:, t]
                    else:
                        step_batch[key] = val
                z_t = model.encoder(step_batch)
                all_z.append(z_t.cpu())
                all_gf.append(step_batch['global_features'].cpu())

    z_all = torch.cat(all_z, dim=0)
    gf_all = torch.cat(all_gf, dim=0)

    D = z_all.shape[1]
    n_val = max(1, len(z_all) // 5)

    probes = {}
    for target_key, target_info in PROBE_TARGETS.items():
        idx = target_info['idx']
        y = gf_all[:, idx]
        if y.std() < 1e-6:
            continue

        probe = LinearProbe(D)
        train_probe(
            probe,
            z_all[n_val:], y[n_val:],
            z_all[:n_val], y[:n_val],
            n_epochs=100, device=device,
        )
        probes[target_key] = probe

    return probes


# =====================================================================
# Evaluate planning
# =====================================================================

def evaluate_planning(
    model,
    pairs: List[Dict],
    planner: CEMPlanner,
    probes: Dict[str, nn.Module] = None,
    device: str = 'cpu',
) -> Dict:
    """
    Run CEM planning on all pairs and evaluate.
    """
    results = {
        'latent_distances': [],
        'planning_times': [],
        'physical_success': defaultdict(list),
        'per_pair': [],
    }

    n_pairs = len(pairs)
    print(f"\n  Planning {n_pairs} goal pairs (CEM: pop={planner.population_size}, "
          f"iter={planner.n_iterations}, horizon={planner.horizon})...")

    for idx, pair in enumerate(pairs):
        z_start = pair['z_start']
        z_goal = pair['z_goal']
        gap = pair['gap']

        # Plan
        t0 = time.time()
        planned_actions, latent_dist = planner.plan(
            model, z_start, z_goal, device
        )
        plan_time = time.time() - t0

        results['latent_distances'].append(latent_dist)
        results['planning_times'].append(plan_time)

        # Physical success: probe the predicted final state
        pair_result = {
            'gap': gap,
            'latent_distance': latent_dist,
            'plan_time': plan_time,
        }

        if probes:
            # Get predicted final state
            z_init = z_start.unsqueeze(0).to(device)
            a_seq = planned_actions.unsqueeze(0).to(device)
            z_traj = model.predictor.rollout(z_init, a_seq)
            z_final = z_traj[0, -1].cpu()

            for target_key, probe in probes.items():
                probe.eval()
                with torch.no_grad():
                    pred_val = probe(z_final.unsqueeze(0)).item()
                    goal_val = pair['gf_goal'][PROBE_TARGETS[target_key]['idx']].item()

                    if abs(goal_val) > 1e-6:
                        rel_error = abs(pred_val - goal_val) / abs(goal_val)
                    else:
                        rel_error = abs(pred_val - goal_val)

                    success = rel_error < 0.10  # within 10%
                    results['physical_success'][target_key].append(success)
                    pair_result[f'{target_key}_pred'] = pred_val
                    pair_result[f'{target_key}_goal'] = goal_val
                    pair_result[f'{target_key}_error'] = rel_error

        results['per_pair'].append(pair_result)

        if (idx + 1) % 10 == 0 or idx == n_pairs - 1:
            avg_dist = np.mean(results['latent_distances'])
            avg_time = np.mean(results['planning_times'])
            print(f"    [{idx+1}/{n_pairs}] avg dist={avg_dist:.4f}, "
                  f"avg time={avg_time:.2f}s")

    return results


# =====================================================================
# Report
# =====================================================================

def print_report(results: Dict, planner: CEMPlanner):
    """Print planning evaluation summary."""
    dists = np.array(results['latent_distances'])
    times = np.array(results['planning_times'])

    print(f"\n  ── Latent Distance ──")
    print(f"    Mean:   {dists.mean():.4f}")
    print(f"    Median: {np.median(dists):.4f}")
    print(f"    Min:    {dists.min():.4f}")
    print(f"    Max:    {dists.max():.4f}")

    # Success rates at various thresholds
    print(f"\n  ── Latent Success Rate ──")
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        rate = (dists < threshold).mean()
        print(f"    dist < {threshold:.1f}: {rate:.0%}")

    # Physical success
    if results['physical_success']:
        print(f"\n  ── Physical Success Rate (within ±10%) ──")
        for target, successes in sorted(results['physical_success'].items()):
            rate = np.mean(successes)
            name = PROBE_TARGETS[target]['name']
            print(f"    {name:<28} {rate:.0%}")

    # Planning speed
    print(f"\n  ── Planning Speed ──")
    print(f"    Avg time per plan:  {times.mean():.2f}s")
    print(f"    Horizon:            {planner.horizon} steps")
    print(f"    CEM iterations:     {planner.n_iterations}")
    print(f"    Population size:    {planner.population_size}")

    # Speedup estimate vs GNPy
    # GNPy takes ~60ms per step on average (from dataset generation)
    gnpy_time = planner.horizon * 0.06  # seconds
    speedup = gnpy_time * planner.population_size * planner.n_iterations / max(times.mean(), 0.001)
    print(f"    Est. GNPy equivalent: {gnpy_time * planner.population_size * planner.n_iterations:.1f}s")
    print(f"    Planning speedup:     ~{speedup:.0f}×")


def save_report(results: Dict, output_dir: str):
    """Save results and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Clean results for JSON (remove non-serializable)
    export = {
        'latent_distances': results['latent_distances'],
        'planning_times': results['planning_times'],
        'physical_success': {
            k: [bool(s) for s in v]
            for k, v in results['physical_success'].items()
        },
        'summary': {
            'mean_distance': float(np.mean(results['latent_distances'])),
            'median_distance': float(np.median(results['latent_distances'])),
            'mean_plan_time': float(np.mean(results['planning_times'])),
        },
    }

    # Add physical success rates
    if results['physical_success']:
        export['summary']['physical_success_rates'] = {
            k: float(np.mean(v))
            for k, v in results['physical_success'].items()
        }

    with open(f"{output_dir}/planning_results.json", 'w') as f:
        json.dump(export, f, indent=2)
    print(f"\n  Saved: {output_dir}/planning_results.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: latent distance histogram
        ax = axes[0]
        ax.hist(results['latent_distances'], bins=20, color='#065A82',
                alpha=0.8, edgecolor='white')
        ax.axvline(np.median(results['latent_distances']), color='#DC2626',
                   linestyle='--', label=f"Median: {np.median(results['latent_distances']):.2f}")
        ax.set_xlabel('Latent Distance to Goal')
        ax.set_ylabel('Count')
        ax.set_title('CEM Planning: Distance to Goal')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Right: physical success rates
        if results['physical_success']:
            ax = axes[1]
            targets = sorted(results['physical_success'].keys())
            rates = [np.mean(results['physical_success'][t]) for t in targets]
            names = [PROBE_TARGETS[t]['name'] for t in targets]
            colors_map = {
                'n_active_lps': '#065A82', 'total_capacity': '#7C3AED',
                'worst_margin': '#D97706', 'avg_margin': '#059669',
                'spectral_util': '#DC2626', 'n_infeasible': '#888888',
            }
            cols = [colors_map.get(t, '#888888') for t in targets]
            bars = ax.bar(range(len(targets)), rates, color=cols)
            ax.set_xticks(range(len(targets)))
            ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
            ax.set_ylabel('Success Rate (±10%)')
            ax.set_title('Physical Quantity Match at Goal')
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.tight_layout()
        fig.savefig(f"{output_dir}/planning_results.png", dpi=150)
        fig.savefig(f"{output_dir}/planning_results.pdf")
        plt.close(fig)
        print(f"  Saved: planning_results.png")

    except ImportError:
        print("  matplotlib not available — skipping plots")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Goal-conditioned planning evaluation"
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='data_100')
    parser.add_argument('--output', type=str, default='figures/planning')
    parser.add_argument('--n-pairs', type=int, default=50,
                        help='Number of (start, goal) pairs to test')
    parser.add_argument('--horizon', type=int, default=15,
                        help='Planning horizon (CEM action sequence length)')
    parser.add_argument('--min-gap', type=int, default=10,
                        help='Minimum steps between start and goal')
    parser.add_argument('--max-gap', type=int, default=25,
                        help='Maximum steps between start and goal')
    parser.add_argument('--population', type=int, default=64,
                        help='CEM population size')
    parser.add_argument('--cem-iterations', type=int, default=5,
                        help='CEM refinement iterations')
    parser.add_argument('--no-probes', action='store_true',
                        help='Skip physical quantity evaluation')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')

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

    # ── Load data ──
    print(f"\n  Loading data from {args.data}...")
    # Use longer context for goal sampling
    context_len = min(args.max_gap + 1, 32)

    train_loader, val_loader = create_dataloaders(
        data_dir=args.data,
        context_length=context_len,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    # ── Compute action statistics ──
    print(f"  Computing action statistics...")
    action_mean, action_std = compute_action_stats(train_loader)
    print(f"    Action mean range: [{action_mean.min():.2f}, {action_mean.max():.2f}]")
    print(f"    Action std range:  [{action_std.min():.2f}, {action_std.max():.2f}]")

    # ── Sample goal pairs ──
    print(f"  Sampling {args.n_pairs} goal pairs (gap {args.min_gap}-{args.max_gap})...")
    pairs = sample_goal_pairs(
        model, val_loader,
        n_pairs=args.n_pairs,
        min_gap=args.min_gap,
        max_gap=min(args.max_gap, context_len - 1),
        device=args.device,
    )
    print(f"  Got {len(pairs)} pairs")

    if not pairs:
        print("  ERROR: No valid pairs found. Try reducing --min-gap")
        return

    # ── Train probes ──
    probes = None
    if not args.no_probes:
        print(f"\n  Training probes for physical evaluation...")
        probes = train_probes_for_planning(model, train_loader, args.device)
        print(f"    Trained {len(probes)} probes")

    # ── Create planner ──
    planner = CEMPlanner(
        action_dim=action_mean.shape[0],
        horizon=args.horizon,
        population_size=args.population,
        n_iterations=args.cem_iterations,
        action_mean=action_mean,
        action_std=action_std,
    )

    # ── Run planning ──
    print(f"\n{'='*60}")
    print(f"  Goal-Conditioned Planning Evaluation")
    print(f"{'='*60}")

    results = evaluate_planning(
        model, pairs, planner, probes, args.device
    )

    # ── Report ──
    print_report(results, planner)
    save_report(results, args.output)

    print(f"\n{'='*60}")
    print(f"  Planning evaluation complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()