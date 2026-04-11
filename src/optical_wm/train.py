"""
Training CLI for Optical Network World Model.

Usage:
  # Quick smoke test (verify loss decreases)
  python -m optical_wm.train --data data_100 --epochs 3 --batch-size 4 --name smoke

  # Real training
  python -m optical_wm.train --data data_full --epochs 50 --name v1

  # Ablation: MLP encoder
  python -m optical_wm.train --data data_100 --encoder mlp --name ablation_mlp

  # Resume interrupted training
  python -m optical_wm.train --data data_full --resume checkpoints/v1/final.pt
"""
import argparse
import sys
import os
import json

# Add src to path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from .models.world_model import OpticalWorldModel
    from .training.dataset import create_dataloaders
    from .training.trainer import Trainer
    from .training.config import TrainConfig
except ImportError:
    from models.world_model import OpticalWorldModel
    from training.dataset import create_dataloaders
    from training.trainer import Trainer
    from training.config import TrainConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train JEPA world model for optical networks"
    )

    # Data
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory with HDF5 files')

    # Architecture
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--encoder', type=str, default='gnn',
                        choices=['gnn', 'mlp'],
                        help='Encoder architecture')
    parser.add_argument('--predictor', type=str, default='transformer',
                        choices=['transformer', 'mlp'],
                        help='Predictor architecture')
    parser.add_argument('--n-gnn-layers', type=int, default=3)
    parser.add_argument('--n-pred-layers', type=int, default=4)
    parser.add_argument('--no-spectral-conv', action='store_true',
                        help='Disable Conv1D spectral encoder (ablation)')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0,
                    help='DataLoader workers (0 for Windows)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--context-length', type=int, default=8)
    parser.add_argument('--collapse', type=str, default='variance',
                        choices=['variance', 'sigreg'])
    parser.add_argument('--collapse-weight', type=float, default=0.1)

    # Logging
    parser.add_argument('--name', type=str, default='default',
                        help='Run name for checkpoints')
    parser.add_argument('--output', type=str, default='checkpoints')
    parser.add_argument('--log-every', type=int, default=20)

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])

    args = parser.parse_args()

    # ── Build config ──
    config = TrainConfig(
        data_dir=args.data,
        context_length=args.context_length,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        n_gnn_layers=args.n_gnn_layers,
        n_pred_layers=args.n_pred_layers,
        use_spectral_conv=not args.no_spectral_conv,
        encoder_type=args.encoder,
        predictor_type=args.predictor,
        collapse_method=args.collapse,
        collapse_weight=args.collapse_weight,
        lr=args.lr,
        n_epochs=args.epochs,
        output_dir=args.output,
        run_name=args.name,
        log_every=args.log_every,
        device=args.device,
    )

    # ── Create data loaders ──
    print(f"\nLoading data from {args.data}...")
    train_loader, val_loader = create_dataloaders(
        data_dir=config.data_dir,
        context_length=config.context_length,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        pin_memory=config.pin_memory and config.resolve_device() != 'cpu',
    )

    # ── Create model ──
    print(f"\nBuilding model...")
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
    print(f"  Encoder:   {params['encoder']:>10,} params")
    print(f"  Predictor: {params['predictor']:>10,} params")
    print(f"  Total:     {params['total']:>10,} params")

    # ── Create trainer ──
    trainer = Trainer(model, train_loader, val_loader, config)

    # ── Resume if specified ──
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Train ──
    trainer.train()


if __name__ == "__main__":
    main()