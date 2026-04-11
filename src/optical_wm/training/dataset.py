"""
PyTorch Dataset for Optical Network World Model.

Wraps HDF5Reader to sample sub-trajectories for JEPA training.

Each sample is a sub-trajectory of length T:
  - State tensors: [T, ...] for dynamic features
  - Static tensors: [...] for topology (same across timesteps)
  - Actions: [T-1, action_dim]

The dataset builds an index of (episode_id, start_step) pairs
and samples from them randomly via the DataLoader.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional

try:
    from ..core.hdf5_io import HDF5Reader
    from ..core.schemas import (
        MAX_NODES, MAX_LINKS, MAX_SLOTS, MAX_LIGHTPATHS,
        N_ACTION_FEATURES, N_GLOBAL_FEATURES, N_LP_FEATURES,
        N_LINK_STATIC_FEATURES, N_NODE_FEATURES,
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.hdf5_io import HDF5Reader
    from core.schemas import (
        MAX_NODES, MAX_LINKS, MAX_SLOTS, MAX_LIGHTPATHS,
        N_ACTION_FEATURES, N_GLOBAL_FEATURES, N_LP_FEATURES,
        N_LINK_STATIC_FEATURES, N_NODE_FEATURES,
    )


class OpticalWMDataset(Dataset):
    """
    PyTorch Dataset for world model training.

    Samples sub-trajectories of length `context_length` from episodes
    stored in HDF5 files. Each sample contains:
      - Dynamic state tensors [T, ...]
      - Static topology tensors [...]
      - Actions [T-1, action_dim]

    Multiple HDF5 files can be loaded (one per topology).
    """

    def __init__(
        self,
        h5_paths: List[str],
        context_length: int = 8,
        split: str = "train",
        normalize: bool = True,
        norm_stats: Optional[dict] = None,
    ):
        """
        Args:
            h5_paths: list of HDF5 file paths
            context_length: sub-trajectory length T
            split: 'train' or 'val'
            normalize: whether to normalize features
            norm_stats: dict with mean/std (from normalization.json)
        """
        self.context_length = context_length
        self.split = split
        self.normalize = normalize
        self.norm_stats = norm_stats

        # Open all files and build index
        self.readers: List[HDF5Reader] = []
        self.topologies: List[dict] = []
        self.index: List[tuple] = []  # (reader_idx, episode_id, start_step)

        for path in h5_paths:
            reader = HDF5Reader(path)
            reader.open()
            reader_idx = len(self.readers)
            self.readers.append(reader)

            # Load static topology
            topo = reader.get_topology()
            self.topologies.append(topo)

            # Get episode list for this split
            ep_list = self._get_split_episodes(reader, split)

            # Build index: all valid (episode, start_step) pairs
            for eid in ep_list:
                info = reader.get_episode_info(eid)
                n_steps = info['n_steps']
                # Sub-trajectories: need context_length states + (context_length-1) actions
                max_start = n_steps - context_length
                for t in range(max(1, max_start)):
                    self.index.append((reader_idx, eid, t))

        print(f"  Dataset [{split}]: {len(self.index)} sub-trajectories "
              f"from {sum(len(self._get_split_episodes(r, split)) for r in self.readers)} episodes "
              f"across {len(self.readers)} topologies")

    def _get_split_episodes(self, reader: HDF5Reader, split: str) -> List[str]:
        """Get episode IDs for the given split."""
        try:
            f = reader._file
            if 'split' in f and split in f['split']:
                eps = [e.decode() if isinstance(e, bytes) else e
                       for e in f['split'][split][()]]
                return eps
        except Exception:
            pass
        # Fallback: all episodes
        return reader.list_episodes()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        reader_idx, eid, t_start = self.index[idx]
        reader = self.readers[reader_idx]
        topo = self.topologies[reader_idx]
        T = self.context_length
        t_end = t_start + T

        # Load sub-trajectory
        sub = reader.load_subsequence(eid, t_start, t_end)

        # ── Dynamic state tensors [T, ...] ──
        sample = {
            'spectral_occupancy': torch.from_numpy(
                sub['spectral_occupancy'].astype(np.float32)
            ),  # [T, MAX_LINKS, MAX_SLOTS]
            'channel_gsnr': torch.from_numpy(sub['channel_gsnr']),
            'channel_power': torch.from_numpy(sub['channel_power']),
            'channel_ase': torch.from_numpy(sub['channel_ase']),
            'channel_nli': torch.from_numpy(sub['channel_nli']),
            'lp_features': torch.from_numpy(sub['lp_features']),
            'lp_mask': torch.from_numpy(sub['lp_mask'].astype(np.float32)),
            'global_features': torch.from_numpy(sub['global_features']),
        }

        # ── Static topology tensors [...] (same for all timesteps) ──
        sample['link_static'] = torch.from_numpy(topo['link_static'])
        sample['link_endpoints'] = torch.from_numpy(
            topo['link_endpoints'].astype(np.int64)
        )
        sample['link_mask'] = torch.from_numpy(topo['link_mask'])
        sample['node_mask'] = torch.from_numpy(topo['node_mask'])
        sample['node_features'] = torch.from_numpy(topo['node_features'])

        # ── Actions [T-1, action_dim] ──
        if 'actions' in sub and len(sub['actions']) >= T - 1:
            sample['actions'] = torch.from_numpy(sub['actions'][:T-1])
        else:
            sample['actions'] = torch.zeros(T - 1, N_ACTION_FEATURES)

        # ── Normalize ──
        if self.normalize and self.norm_stats:
            sample = self._apply_normalization(sample)

        return sample

    def _apply_normalization(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply z-score normalization to spectral features."""
        if 'channel_gsnr' in self.norm_stats:
            stats = self.norm_stats['channel_gsnr']
            mean, std = stats.get('mean', 0), stats.get('std', 1)
            if std > 0:
                mask = sample['channel_gsnr'] != 0  # only normalize active
                sample['channel_gsnr'][mask] = (
                    (sample['channel_gsnr'][mask] - mean) / std
                )
        if 'channel_power' in self.norm_stats:
            stats = self.norm_stats['channel_power']
            mean, std = stats.get('mean', 0), stats.get('std', 1)
            if std > 0:
                mask = sample['channel_power'] != 0
                sample['channel_power'][mask] = (
                    (sample['channel_power'][mask] - mean) / std
                )
        return sample

    def close(self):
        """Close all HDF5 readers."""
        for reader in self.readers:
            reader.close()

    def __del__(self):
        self.close()


def create_dataloaders(
    data_dir: str,
    context_length: int = 8,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    norm_stats: Optional[dict] = None,
) -> tuple:
    """
    Create train and val DataLoaders from a data directory.

    Args:
        data_dir: directory containing HDF5 files + normalization.json
    Returns:
        (train_loader, val_loader)
    """
    data_path = Path(data_dir)
    h5_files = sorted(data_path.glob("*.h5"))
    h5_paths = [str(f) for f in h5_files]

    if not h5_paths:
        raise FileNotFoundError(f"No HDF5 files in {data_dir}")

    # Load normalization stats if available
    if norm_stats is None:
        norm_path = data_path / "normalization.json"
        if norm_path.exists():
            import json
            with open(norm_path) as f:
                norm_stats = json.load(f)
            print(f"  Loaded normalization stats from {norm_path}")

    print(f"  Found {len(h5_paths)} HDF5 files")

    train_dataset = OpticalWMDataset(
        h5_paths=h5_paths,
        context_length=context_length,
        split="train",
        norm_stats=norm_stats,
    )

    val_dataset = OpticalWMDataset(
        h5_paths=h5_paths,
        context_length=context_length,
        split="val",
        norm_stats=norm_stats,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


# =====================================================================
# Quick test
# =====================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_dir>")
        print("Example: python dataset.py data_100")
        sys.exit(1)

    data_dir = sys.argv[1]
    print(f"Testing dataset from {data_dir}...\n")

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        context_length=8,
        batch_size=4,
        num_workers=0,
    )

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Load one batch
    batch = next(iter(train_loader))
    print(f"\n  Batch shapes:")
    for k, v in batch.items():
        print(f"    {k}: {list(v.shape)} {v.dtype}")

    # Verify shapes
    B = 4
    T = 8
    assert batch['spectral_occupancy'].shape == (B, T, MAX_LINKS, MAX_SLOTS)
    assert batch['channel_gsnr'].shape == (B, T, MAX_LINKS, MAX_SLOTS)
    assert batch['lp_features'].shape == (B, T, MAX_LIGHTPATHS, N_LP_FEATURES)
    assert batch['actions'].shape == (B, T - 1, N_ACTION_FEATURES)
    assert batch['link_mask'].shape == (B, MAX_LINKS)
    assert batch['global_features'].shape == (B, T, N_GLOBAL_FEATURES)

    print(f"\n✓ Dataset test passed")