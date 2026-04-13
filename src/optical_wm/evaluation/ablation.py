"""
Ablation Study (Éval 5) for Optical Network World Model.

Trains multiple model variants and compares probing + rollout metrics.

Variants:
  A — Full GNN (baseline):      Conv1D + GAT + Transformer predictor
  B — MLP encoder:              No graph structure, no Conv1D
  C — MLP predictor:            No temporal Transformer, independent predictions
  D — No Conv1D:                Mean/std spectral features instead of Conv1D
  E — Latent dim 64:            Smaller latent space
  F — Latent dim 256:           Larger latent space (optional)
  G — GNN 1 layer:              Shallow GNN (optional)

Usage:
  # Run all tier 1+2 ablations (~12h on GPU)
  python -m optical_wm.evaluation.ablation \
    --data data_full --device cuda --tiers 1,2

  # Run specific variants
  python -m optical_wm.evaluation.ablation \
    --data data_full --device cuda --variants A,B,C

  # Quick test on small dataset
  python -m optical_wm.evaluation.ablation \
    --data data_100 --device cpu --variants A,B --epochs 3
"""
import torch
import json
import argparse
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ..models.world_model import OpticalWorldModel
    from ..training.dataset import create_dataloaders
    from ..training.trainer import Trainer
    from ..training.config import TrainConfig
    from .probing import run_probing
    from .rollout import (
        extract_episode_embeddings, compute_rollout_metrics, probe_at_horizons
    )
except ImportError:
    from models.world_model import OpticalWorldModel
    from training.dataset import create_dataloaders
    from training.trainer import Trainer
    from training.config import TrainConfig
    from evaluation.probing import run_probing
    from evaluation.rollout import (
        extract_episode_embeddings, compute_rollout_metrics, probe_at_horizons
    )


# =====================================================================
# Ablation Variants
# =====================================================================

@dataclass
class AblationVariant:
    """One ablation configuration."""
    name: str
    description: str
    tier: int
    # Overrides from default config
    encoder_type: str = "gnn"
    predictor_type: str = "transformer"
    use_spectral_conv: bool = True
    latent_dim: int = 128
    n_gnn_layers: int = 3


VARIANTS = OrderedDict([
    ("A", AblationVariant(
        name="A_full_gnn",
        description="Full model: Conv1D + GAT-3 + Transformer",
        tier=1,
    )),
    ("B", AblationVariant(
        name="B_mlp_encoder",
        description="MLP encoder (no graph structure)",
        tier=1,
        encoder_type="mlp",
    )),
    ("C", AblationVariant(
        name="C_mlp_predictor",
        description="MLP predictor (no temporal context)",
        tier=1,
        predictor_type="mlp",
    )),
    ("D", AblationVariant(
        name="D_no_conv1d",
        description="No Conv1D spectral encoder (mean/std only)",
        tier=2,
        use_spectral_conv=False,
    )),
    ("E", AblationVariant(
        name="E_latent_64",
        description="Latent dimension 64 (vs 128)",
        tier=2,
        latent_dim=64,
    )),
    ("F", AblationVariant(
        name="F_latent_256",
        description="Latent dimension 256 (vs 128)",
        tier=3,
        latent_dim=256,
    )),
    ("G", AblationVariant(
        name="G_gnn_1layer",
        description="Shallow GNN (1 layer vs 3)",
        tier=3,
        n_gnn_layers=1,
    )),
])


# =====================================================================
# Train one variant
# =====================================================================

def train_variant(
    variant: AblationVariant,
    data_dir: str,
    output_dir: str,
    epochs: int = 15,
    batch_size: int = 32,
    context_length: int = 32,
    device: str = "cuda",
) -> str:
    """
    Train one ablation variant.
    Returns path to best checkpoint.
    """
    run_name = f"ablation_{variant.name}"
    ckpt_dir = os.path.join(output_dir, run_name)

    # Check if already trained
    best_pt = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_pt):
        print(f"    Checkpoint exists: {best_pt} — skipping training")
        return best_pt

    print(f"\n    Training {variant.name}: {variant.description}")

    # Config
    config = TrainConfig(
        data_dir=data_dir,
        context_length=context_length,
        batch_size=batch_size,
        latent_dim=variant.latent_dim,
        n_gnn_layers=variant.n_gnn_layers,
        use_spectral_conv=variant.use_spectral_conv,
        encoder_type=variant.encoder_type,
        predictor_type=variant.predictor_type,
        n_epochs=epochs,
        output_dir=output_dir,
        run_name=run_name,
        device=device,
        num_workers=0,
    )

    # Data
    train_loader, val_loader = create_dataloaders(
        data_dir=config.data_dir,
        context_length=config.context_length,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    # Model
    model = OpticalWorldModel(
        latent_dim=config.latent_dim,
        spectral_dim=config.spectral_dim,
        node_hidden_dim=config.node_hidden_dim,
        n_gnn_layers=config.n_gnn_layers,
        n_gnn_heads=config.n_gnn_heads,
        lp_summary_dim=config.lp_summary_dim,
        use_spectral_conv=config.use_spectral_conv,
        encoder_type=config.encoder_type,
        pred_hidden_dim=config.pred_hidden_dim,
        n_pred_layers=config.n_pred_layers,
        n_pred_heads=config.n_pred_heads,
        action_emb_dim=config.action_emb_dim,
        pred_dropout=config.pred_dropout,
        predictor_type=config.predictor_type,
        collapse_method=config.collapse_method,
        collapse_weight=config.collapse_weight,
    )

    params = model.get_param_count()
    print(f"    Params: encoder={params['encoder']:,}, "
          f"predictor={params['predictor']:,}, total={params['total']:,}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

    return best_pt


# =====================================================================
# Evaluate one variant
# =====================================================================

def evaluate_variant(
    variant: AblationVariant,
    checkpoint_path: str,
    data_dir: str,
    device: str = "cuda",
    max_horizon: int = 15,
    probe_horizons: List[int] = [1, 5, 10, 15],
    context_length: int = 32,
) -> Dict:
    """
    Run probing + rollout on a trained variant.
    Returns dict with all metrics.
    """
    print(f"\n    Evaluating {variant.name}...")

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    model = OpticalWorldModel(
        latent_dim=config.get('latent_dim', variant.latent_dim),
        spectral_dim=config.get('spectral_dim', 64),
        node_hidden_dim=config.get('node_hidden_dim', 128),
        n_gnn_layers=config.get('n_gnn_layers', variant.n_gnn_layers),
        n_gnn_heads=config.get('n_gnn_heads', 4),
        lp_summary_dim=config.get('lp_summary_dim', 32),
        use_spectral_conv=config.get('use_spectral_conv', variant.use_spectral_conv),
        encoder_type=config.get('encoder_type', variant.encoder_type),
        pred_hidden_dim=config.get('pred_hidden_dim', 256),
        n_pred_layers=config.get('n_pred_layers', 4),
        n_pred_heads=config.get('n_pred_heads', 8),
        action_emb_dim=config.get('action_emb_dim', 64),
        predictor_type=config.get('predictor_type', variant.predictor_type),
        collapse_method=config.get('collapse_method', 'variance'),
    )

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    best_val_loss = ckpt.get('best_val_loss', None)

    # Data loaders
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        context_length=context_length,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )

    results = {
        'variant': variant.name,
        'description': variant.description,
        'params': model.get_param_count(),
        'best_val_loss': best_val_loss,
    }

    # ── Probing ──
    print(f"    Running probing...")
    probing_results = run_probing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_probe_epochs=100,
        max_extract_batches=100,
    )
    results['probing'] = probing_results

    # Compute average probing r
    linear_rs = [
        probing_results[t]['linear']['pearson_r']
        for t in probing_results if 'linear' in probing_results[t]
    ]
    mlp_rs = [
        probing_results[t]['mlp']['pearson_r']
        for t in probing_results if 'mlp' in probing_results[t]
    ]
    results['avg_probing_r_linear'] = sum(linear_rs) / len(linear_rs) if linear_rs else 0
    results['avg_probing_r_mlp'] = sum(mlp_rs) / len(mlp_rs) if mlp_rs else 0

    # ── Rollout ──
    print(f"    Running rollout (max_horizon={max_horizon})...")

    # Use longer context for rollout
    _, val_loader_long = create_dataloaders(
        data_dir=data_dir,
        context_length=min(max_horizon + 1, context_length),
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )

    episodes = extract_episode_embeddings(
        model, val_loader_long, device, max_batches=50
    )

    rollout_metrics = compute_rollout_metrics(
        model, episodes, max_horizon=max_horizon, device=device
    )

    results['rollout'] = {
        'stability_ratio': rollout_metrics['stability_ratio'],
        'mse_h1': rollout_metrics['mse_per_horizon'].get(1, None),
        'mse_h5': rollout_metrics['mse_per_horizon'].get(5, None),
        'mse_h10': rollout_metrics['mse_per_horizon'].get(10, None),
        'mse_h15': rollout_metrics['mse_per_horizon'].get(
            max(rollout_metrics['mse_per_horizon'].keys()), None
        ),
    }

    return results


# =====================================================================
# Comparison Table
# =====================================================================

def print_comparison_table(all_results: List[Dict]):
    """Print a nice comparison table."""
    print(f"\n{'='*90}")
    print(f"  ABLATION COMPARISON TABLE")
    print(f"{'='*90}")

    # Header
    print(f"\n  {'Variant':<25} {'Params':>8} {'Val Loss':>9} "
          f"{'Probe(L)':>9} {'Probe(M)':>9} {'Stab.':>7} {'MSE h=1':>9}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*9}")

    for r in all_results:
        params = r['params']['total']
        val_loss = r.get('best_val_loss', 0) or 0
        probe_l = r.get('avg_probing_r_linear', 0)
        probe_m = r.get('avg_probing_r_mlp', 0)
        stability = r.get('rollout', {}).get('stability_ratio', 0)
        mse_h1 = r.get('rollout', {}).get('mse_h1', 0) or 0

        print(f"  {r['description'][:25]:<25} {params:>8,} {val_loss:>9.4f} "
              f"{probe_l:>9.4f} {probe_m:>9.4f} {stability:>6.1f}× {mse_h1:>9.4f}")

    # Find best
    if len(all_results) >= 2:
        best_probe = max(all_results, key=lambda r: r.get('avg_probing_r_linear', 0))
        best_rollout = min(all_results,
                           key=lambda r: r.get('rollout', {}).get('stability_ratio', 999))

        print(f"\n  Best probing:  {best_probe['variant']} "
              f"(r={best_probe['avg_probing_r_linear']:.4f})")
        print(f"  Best rollout:  {best_rollout['variant']} "
              f"(stability={best_rollout['rollout']['stability_ratio']:.1f}×)")


def save_results(all_results: List[Dict], output_dir: str):
    """Save results and generate comparison plot."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    # Clean non-serializable items
    clean_results = []
    for r in all_results:
        cr = {k: v for k, v in r.items()}
        cr['params'] = {k: int(v) for k, v in r['params'].items()}
        clean_results.append(cr)

    with open(f"{output_dir}/ablation_results.json", 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\n  Saved: {output_dir}/ablation_results.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        n = len(all_results)
        if n < 2:
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        names = [r['variant'] for r in all_results]
        x = np.arange(n)

        # Plot 1: Probing r (linear vs MLP)
        ax = axes[0]
        linear_r = [r.get('avg_probing_r_linear', 0) for r in all_results]
        mlp_r = [r.get('avg_probing_r_mlp', 0) for r in all_results]
        w = 0.35
        ax.bar(x - w/2, linear_r, w, label='Linear', color='#065A82')
        ax.bar(x + w/2, mlp_r, w, label='MLP', color='#059669')
        ax.set_ylabel('Avg Pearson r')
        ax.set_title('Probing Quality')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, color='gray', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Plot 2: Rollout stability
        ax = axes[1]
        stability = [r.get('rollout', {}).get('stability_ratio', 0) for r in all_results]
        colors = ['#059669' if s < 10 else '#D97706' if s < 50 else '#DC2626'
                  for s in stability]
        ax.bar(x, stability, color=colors)
        ax.set_ylabel('Stability Ratio (lower = better)')
        ax.set_title('Rollout Stability')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.axhline(5.4, color='#065A82', linestyle='--', alpha=0.5,
                   label='LeWM JEPA (5.4×)')
        ax.axhline(49, color='#DC2626', linestyle='--', alpha=0.5,
                   label='LeWM LSTM (49×)')
        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Plot 3: Parameter count vs probing quality
        ax = axes[2]
        params = [r['params']['total'] / 1e6 for r in all_results]
        ax.scatter(params, linear_r, s=100, c='#065A82', zorder=5)
        for i, name in enumerate(names):
            ax.annotate(name, (params[i], linear_r[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('Avg Probing r (linear)')
        ax.set_title('Efficiency: Quality vs Size')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        fig.savefig(f"{output_dir}/ablation_comparison.png", dpi=150)
        fig.savefig(f"{output_dir}/ablation_comparison.pdf")
        plt.close(fig)
        print(f"  Saved: {output_dir}/ablation_comparison.png")

    except ImportError:
        print("  matplotlib not available — skipping plots")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for JEPA world model"
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory')
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--figures', type=str, default='figures/ablation',
                        help='Output directory for figures')
    parser.add_argument('--variants', type=str, default=None,
                        help='Comma-separated variant keys (e.g., A,B,C)')
    parser.add_argument('--tiers', type=str, default='1,2',
                        help='Comma-separated tiers to run (e.g., 1,2,3)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Training epochs per variant')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--context-length', type=int, default=32)
    parser.add_argument('--max-horizon', type=int, default=15,
                        help='Max rollout horizon for evaluation')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, only evaluate existing checkpoints')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Select variants
    if args.variants:
        selected_keys = args.variants.split(',')
    else:
        tiers = [int(t) for t in args.tiers.split(',')]
        selected_keys = [k for k, v in VARIANTS.items() if v.tier in tiers]

    selected = [(k, VARIANTS[k]) for k in selected_keys if k in VARIANTS]

    print(f"\n{'='*70}")
    print(f"  Ablation Study")
    print(f"{'='*70}")
    print(f"  Data:       {args.data}")
    print(f"  Variants:   {', '.join(k for k, _ in selected)}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Context:    {args.context_length}")
    print(f"  Device:     {args.device}")
    print()

    all_results = []
    total_start = time.time()

    for key, variant in selected:
        print(f"\n{'─'*70}")
        print(f"  [{key}] {variant.name}: {variant.description}")
        print(f"{'─'*70}")

        t0 = time.time()

        # Train
        if not args.eval_only:
            checkpoint_path = train_variant(
                variant=variant,
                data_dir=args.data,
                output_dir=args.output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
            )
        else:
            checkpoint_path = os.path.join(
                args.output, f"ablation_{variant.name}", "best.pt"
            )
            if not os.path.exists(checkpoint_path):
                print(f"    Checkpoint not found: {checkpoint_path} — skipping")
                continue

        # Evaluate
        results = evaluate_variant(
            variant=variant,
            checkpoint_path=checkpoint_path,
            data_dir=args.data,
            device=args.device,
            max_horizon=args.max_horizon,
            context_length=args.context_length,
        )

        elapsed = time.time() - t0
        results['time_minutes'] = elapsed / 60
        all_results.append(results)

        print(f"\n    Done [{key}] in {elapsed/60:.1f} min")

    # Comparison
    print_comparison_table(all_results)
    save_results(all_results, args.figures)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  Ablation complete: {total_time/60:.1f} min total")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()