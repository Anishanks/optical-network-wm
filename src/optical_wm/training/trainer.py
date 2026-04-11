"""
Training loop for Optical Network World Model.

Features:
  - Cosine LR schedule with warmup
  - Gradient clipping
  - Periodic validation
  - Checkpointing (save/resume)
  - Metric logging (to console + JSON)
  - Collapse monitoring

Usage:
  trainer = Trainer(model, train_loader, val_loader, config)
  trainer.train()
"""
import torch
import torch.nn as nn
import json
import time
import math
import os
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

try:
    from .config import TrainConfig
except ImportError:
    from config import TrainConfig


class Trainer:
    """Training loop for JEPA world model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: TrainConfig,
    ):
        self.config = config
        self.device = config.resolve_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # LR Schedule: cosine with warmup
        self.total_steps = len(train_loader) * config.n_epochs
        self.warmup_steps = len(train_loader) * config.warmup_epochs

        # Logging
        self.output_dir = Path(config.output_dir) / config.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = defaultdict(list)
        self.global_step = 0
        self.best_val_loss = float('inf')

    def get_lr(self, step: int) -> float:
        """Cosine schedule with linear warmup."""
        if step < self.warmup_steps:
            return self.config.lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.config.min_lr + 0.5 * (
            self.config.lr - self.config.min_lr
        ) * (1 + math.cos(math.pi * progress))

    def update_lr(self):
        """Apply current LR to optimizer."""
        lr = self.get_lr(self.global_step)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    # ─────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────

    def train(self):
        """Full training loop."""
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"  Training: {cfg.run_name}")
        print(f"{'='*60}")
        print(f"  Device:     {self.device}")
        print(f"  Epochs:     {cfg.n_epochs}")
        print(f"  Batch size: {cfg.batch_size}")
        print(f"  Context:    {cfg.context_length} steps")
        print(f"  LR:         {cfg.lr}")
        print(f"  Params:     {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Output:     {self.output_dir}")
        print()

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(cfg), f, indent=2, default=str)

        start_time = time.time()

        for epoch in range(1, cfg.n_epochs + 1):
            # Train one epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            if epoch % cfg.eval_every == 0:
                val_metrics = self.validate()
                val_loss = val_metrics['loss/total']

                # Best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt", epoch, val_loss)

                self._log_epoch(epoch, train_metrics, val_metrics, start_time)
            else:
                self._log_epoch(epoch, train_metrics, None, start_time)

            # Periodic save
            if epoch % cfg.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:03d}.pt", epoch)

            # Collapse detection
            if train_metrics.get('embedding/variance', 1.0) < 0.01:
                print(f"\n  ⚠ COLLAPSE DETECTED at epoch {epoch}!")
                print(f"    Embedding variance: {train_metrics['embedding/variance']:.6f}")
                print(f"    Consider increasing collapse_weight or checking architecture")

        # Final save
        self.save_checkpoint("final.pt", cfg.n_epochs)

        # Save history
        history_path = self.output_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Training complete: {total_time/60:.1f} min")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Saved: {self.output_dir}")
        print(f"{'='*60}\n")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch. Returns averaged metrics."""
        self.model.train()
        cfg = self.config
        epoch_metrics = defaultdict(float)
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.global_step += 1
            lr = self.update_lr()

            # Move to device
            batch = self._to_device(batch)

            # Forward pass
            loss, metrics = self._forward_step(batch)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if cfg.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )
                metrics['grad_norm'] = grad_norm.item()

            self.optimizer.step()

            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
            epoch_metrics['lr'] = lr
            n_batches += 1

            # Log
            if self.global_step % cfg.log_every == 0:
                print(
                    f"    step {self.global_step:>5} | "
                    f"loss {metrics['loss/total']:.4f} | "
                    f"pred {metrics['loss/prediction']:.4f} | "
                    f"var {metrics['embedding/variance']:.3f} | "
                    f"lr {lr:.2e}",
                    flush=True,
                )

        # Average
        for k in epoch_metrics:
            if k != 'lr':
                epoch_metrics[k] /= max(1, n_batches)

        return dict(epoch_metrics)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation. Returns averaged metrics."""
        self.model.eval()
        val_metrics = defaultdict(float)
        n_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            _, metrics = self._forward_step(batch)

            for k, v in metrics.items():
                val_metrics[k] += v
            n_batches += 1

        for k in val_metrics:
            val_metrics[k] /= max(1, n_batches)

        return dict(val_metrics)

    def _forward_step(self, batch: Dict[str, torch.Tensor]):
        """Single forward pass + loss computation."""
        T = self.config.context_length
        actions = batch.pop('actions')  # [B, T-1, A]

        # Encode sequence
        z_seq = self.model.encode_sequence(batch, T)  # [B, T, D]

        # Put actions back for next iteration
        batch['actions'] = actions

        # Compute loss
        return self.model.compute_loss(z_seq, actions)

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _log_epoch(self, epoch, train_metrics, val_metrics, start_time):
        """Log epoch summary."""
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (self.config.n_epochs - epoch)

        line = (
            f"  Epoch {epoch:>3}/{self.config.n_epochs} | "
            f"train {train_metrics['loss/total']:.4f} | "
            f"pred {train_metrics['loss/prediction']:.4f} | "
            f"var {train_metrics.get('embedding/variance', 0):.3f}"
        )
        if val_metrics:
            line += f" | val {val_metrics['loss/total']:.4f}"
            is_best = val_metrics['loss/total'] <= self.best_val_loss
            if is_best:
                line += " ★"
        line += f" | {elapsed/60:.1f}m (ETA {eta/60:.1f}m)"
        print(line)

        # Store history
        self.history['epoch'].append(epoch)
        for k, v in train_metrics.items():
            self.history[f'train/{k}'].append(v)
        if val_metrics:
            for k, v in val_metrics.items():
                self.history[f'val/{k}'].append(v)

    # ─────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────

    def save_checkpoint(self, filename: str, epoch: int,
                        val_loss: float = None):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
        }, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint to resume training."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.global_step = ckpt.get('global_step', 0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        epoch = ckpt.get('epoch', 0)
        print(f"  Resumed from {path} (epoch {epoch}, step {self.global_step})")
        return epoch