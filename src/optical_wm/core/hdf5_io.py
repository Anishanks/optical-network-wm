"""
HDF5 Writer for Optical Network World Model Dataset.

Storage design:
  - One HDF5 file per topology family
  - Topology stored once (static)
  - Episodes stored as groups with pre-allocated arrays
  - Compression on spectral arrays (high redundancy)
  - Sub-trajectory sampling via DataLoader (not here)
"""
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import (
    MAX_NODES, MAX_LINKS, MAX_SLOTS, MAX_LIGHTPATHS, MAX_HOPS,
    N_LINK_STATIC_FEATURES, N_NODE_FEATURES, N_LP_FEATURES,
    N_GLOBAL_FEATURES, N_ACTION_FEATURES,
)


class HDF5Writer:
    """Writes episodes to HDF5 format."""

    def __init__(self, filepath: str, compression: str = "gzip",
                 compression_level: int = 4):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.comp_level = compression_level
        self._file = None

    def open(self):
        self._file = h5py.File(self.filepath, "a")  # append mode
        return self

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    # -----------------------------------------------------------------
    # Topology (written once per file)
    # -----------------------------------------------------------------

    def write_topology(self, adjacency: np.ndarray,
                       node_features: np.ndarray,
                       link_static: np.ndarray,
                       link_endpoints: np.ndarray,
                       node_mask: np.ndarray,
                       link_mask: np.ndarray,
                       metadata: Optional[dict] = None):
        """Write static topology data. Called once per HDF5 file."""
        f = self._file
        if "topology" in f:
            del f["topology"]

        topo = f.create_group("topology")
        topo.create_dataset("adjacency", data=adjacency, dtype=np.float32)
        topo.create_dataset("node_features", data=node_features, dtype=np.float32)
        topo.create_dataset("link_static", data=link_static, dtype=np.float32)
        topo.create_dataset("link_endpoints", data=link_endpoints, dtype=np.int32)
        topo.create_dataset("node_mask", data=node_mask, dtype=bool)
        topo.create_dataset("link_mask", data=link_mask, dtype=bool)

        if metadata:
            topo.attrs["metadata"] = json.dumps(metadata)

    # -----------------------------------------------------------------
    # Episodes
    # -----------------------------------------------------------------

    def write_episode(self, episode_id: str, n_steps: int,
                      states: List[Dict[str, np.ndarray]],
                      actions: List[np.ndarray],
                      metadata: Optional[dict] = None):
        """
        Write a complete episode to HDF5.

        Args:
            episode_id: unique episode name (e.g., "ep_0042")
            n_steps: number of timesteps T
            states: list of T state dicts (from encode_state)
            actions: list of T-1 action vectors
            metadata: episode metadata (policy, seed, etc.)
        """
        assert len(states) == n_steps, \
            f"Expected {n_steps} states, got {len(states)}"
        assert len(actions) == n_steps - 1, \
            f"Expected {n_steps-1} actions, got {len(actions)}"

        f = self._file
        ep_path = f"episodes/{episode_id}"

        # Remove if exists (overwrite)
        if ep_path in f:
            del f[ep_path]

        ep = f.create_group(ep_path)
        ep.attrs["n_steps"] = n_steps
        if metadata:
            ep.attrs["metadata"] = json.dumps(metadata)

        # --- States ---
        st = ep.create_group("states")
        T = n_steps
        n_links = states[0]["link_mask"].sum()

        # Spectral arrays (highly compressible)
        compress = dict(compression=self.compression,
                        compression_opts=self.comp_level)

        st.create_dataset(
            "spectral_occupancy",
            data=np.stack([s["spectral_occupancy"] for s in states]),
            dtype=bool, **compress,
        )
        st.create_dataset(
            "channel_gsnr",
            data=np.stack([s["channel_gsnr"] for s in states]),
            dtype=np.float32, **compress,
        )
        st.create_dataset(
            "channel_power",
            data=np.stack([s["channel_power"] for s in states]),
            dtype=np.float32, **compress,
        )
        st.create_dataset(
            "channel_ase",
            data=np.stack([s["channel_ase"] for s in states]),
            dtype=np.float32, **compress,
        )
        st.create_dataset(
            "channel_nli",
            data=np.stack([s["channel_nli"] for s in states]),
            dtype=np.float32, **compress,
        )

        # Per-lightpath
        st.create_dataset(
            "lp_features",
            data=np.stack([s["lp_features"] for s in states]),
            dtype=np.float32, **compress,
        )
        st.create_dataset(
            "lp_mask",
            data=np.stack([s["lp_mask"] for s in states]),
            dtype=bool, **compress,
        )

        # Global
        st.create_dataset(
            "global_features",
            data=np.stack([s["global_features"] for s in states]),
            dtype=np.float32,
        )

        # --- Actions ---
        ep.create_dataset(
            "actions",
            data=np.stack(actions),
            dtype=np.float32,
        )

        # --- Outcomes (separated for probing/evaluation) ---
        oc = ep.create_group("outcomes")

        # Extract per-LP GSNR, margin, feasible from lp_features
        # lp_features layout: [..., 17=gsnr, 18=margin, 19=feasible]
        lp_feat_stack = np.stack([s["lp_features"] for s in states])
        lp_mask_stack = np.stack([s["lp_mask"] for s in states])

        oc.create_dataset(
            "per_lp_gsnr",
            data=lp_feat_stack[:, :, 17],  # [T, max_lp]
            dtype=np.float32, **compress,
        )
        oc.create_dataset(
            "per_lp_margin",
            data=lp_feat_stack[:, :, 18],  # [T, max_lp]
            dtype=np.float32, **compress,
        )
        oc.create_dataset(
            "per_lp_feasible",
            data=lp_feat_stack[:, :, 19] > 0.5,  # [T, max_lp]
            dtype=bool, **compress,
        )

        # Total capacity per step (from global_features[1] = Tbps)
        global_stack = np.stack([s["global_features"] for s in states])
        oc.create_dataset(
            "total_capacity_tbps",
            data=global_stack[:, 1],  # [T]
            dtype=np.float32,
        )

    # -----------------------------------------------------------------
    # Split info
    # -----------------------------------------------------------------

    def write_split(self, train_episodes: List[str],
                    val_episodes: List[str]):
        """Write train/val split."""
        f = self._file
        if "split" in f:
            del f["split"]
        split = f.create_group("split")
        # Store as variable-length strings
        dt = h5py.string_dtype()
        split.create_dataset("train", data=train_episodes, dtype=dt)
        split.create_dataset("val", data=val_episodes, dtype=dt)

    # -----------------------------------------------------------------
    # Info
    # -----------------------------------------------------------------

    def list_episodes(self) -> List[str]:
        """List all episode IDs in the file."""
        if "episodes" not in self._file:
            return []
        return list(self._file["episodes"].keys())

    def get_episode_info(self, episode_id: str) -> dict:
        """Get metadata for an episode."""
        ep = self._file[f"episodes/{episode_id}"]
        info = {"n_steps": int(ep.attrs["n_steps"])}
        if "metadata" in ep.attrs:
            info["metadata"] = json.loads(ep.attrs["metadata"])
        return info


# =========================================================================
# Reader (for DataLoader)
# =========================================================================

class HDF5Reader:
    """Reads episodes from HDF5 for training."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._file = None

    def open(self):
        self._file = h5py.File(self.filepath, "r")
        return self

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def get_topology(self) -> Dict[str, np.ndarray]:
        """Load static topology."""
        topo = self._file["topology"]
        return {
            "adjacency": topo["adjacency"][()],
            "node_features": topo["node_features"][()],
            "link_static": topo["link_static"][()],
            "link_endpoints": topo["link_endpoints"][()],
            "node_mask": topo["node_mask"][()],
            "link_mask": topo["link_mask"][()],
        }

    def get_episode_count(self) -> int:
        return len(self._file["episodes"])

    def list_episodes(self) -> List[str]:
        return list(self._file["episodes"].keys())

    def get_episode_info(self, episode_id: str) -> dict:
        """Get metadata for an episode."""
        ep = self._file[f"episodes/{episode_id}"]
        info = {"n_steps": int(ep.attrs["n_steps"])}
        if "metadata" in ep.attrs:
            info["metadata"] = json.loads(ep.attrs["metadata"])
        return info

    def load_subsequence(self, episode_id: str, t_start: int,
                         t_end: int) -> Dict[str, np.ndarray]:
        """
        Load a sub-trajectory [t_start:t_end] from an episode.
        Used by the DataLoader to sample training windows.
        """
        ep = self._file[f"episodes/{episode_id}"]
        st = ep["states"]

        result = {
            "spectral_occupancy": st["spectral_occupancy"][t_start:t_end],
            "channel_gsnr": st["channel_gsnr"][t_start:t_end],
            "channel_power": st["channel_power"][t_start:t_end],
            "channel_ase": st["channel_ase"][t_start:t_end],
            "channel_nli": st["channel_nli"][t_start:t_end],
            "lp_features": st["lp_features"][t_start:t_end],
            "lp_mask": st["lp_mask"][t_start:t_end],
            "global_features": st["global_features"][t_start:t_end],
        }

        # Actions: one fewer than states
        if t_end - t_start > 1:
            a_start = max(0, t_start)
            a_end = t_end - 1
            result["actions"] = ep["actions"][a_start:a_end]

        # Outcomes (if available)
        if "outcomes" in ep:
            oc = ep["outcomes"]
            result["per_lp_gsnr"] = oc["per_lp_gsnr"][t_start:t_end]
            result["per_lp_margin"] = oc["per_lp_margin"][t_start:t_end]
            result["per_lp_feasible"] = oc["per_lp_feasible"][t_start:t_end]
            result["total_capacity_tbps"] = oc["total_capacity_tbps"][t_start:t_end]

        return result