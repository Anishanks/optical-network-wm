"""
Subsample a larger HDF5 dataset into a smaller one with the same structure.

Keeps per-topology policy mix proportional (largest-remainder method) and
uses deterministic ordering so smaller subsets nest inside larger ones:
data_500 is a subset of data_2k is a subset of data_full.

Usage (from project root):
  PYTHONPATH=src py scripts/subsample.py --dst data_2k  --total 2000
  PYTHONPATH=src py scripts/subsample.py --dst data_500 --total 500
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from optical_wm.generate import compute_normalization_stats


TOPOLOGIES = [
    ("topo_A_small_mesh", "train"),
    ("topo_B_medium_sparse", "train"),
    ("topo_C_medium_mesh", "train"),
    ("topo_D_test", "test"),
]


def group_by_policy(episode_ids):
    buckets = defaultdict(list)
    for eid in episode_ids:
        policy = eid.split("_", 2)[-1]
        buckets[policy].append(eid)
    for policy in buckets:
        buckets[policy].sort()
    return buckets


def stratified_pick(buckets, n_target):
    """Largest-remainder apportionment across policies."""
    total = sum(len(v) for v in buckets.values())
    if n_target > total:
        raise ValueError(f"Requested {n_target} > available {total}")

    picks = []
    remainders = []
    for policy, eids in buckets.items():
        exact = len(eids) * n_target / total
        n = int(exact)
        picks.extend(eids[:n])
        remainders.append((exact - n, policy, eids[n:]))

    deficit = n_target - len(picks)
    remainders.sort(key=lambda x: -x[0])
    i = 0
    while deficit > 0 and i < len(remainders):
        _, _, rest = remainders[i]
        if rest:
            picks.append(rest[0])
            deficit -= 1
        i += 1
    return picks


def subsample_topology(src_path, dst_path, n_episodes, val_frac):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        eps = list(src["episodes"].keys())
        buckets = group_by_policy(eps)
        picked = stratified_pick(buckets, n_episodes)

        src.copy(src["topology"], dst, name="topology")
        dst.create_group("episodes")
        for eid in picked:
            src.copy(src[f"episodes/{eid}"], dst["episodes"], name=eid)

        rng = np.random.default_rng(42)
        shuffled = list(picked)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_frac))
        split = dst.create_group("split")
        split.create_dataset("train", data=[e.encode() for e in shuffled[n_val:]])
        split.create_dataset("val", data=[e.encode() for e in shuffled[:n_val]])

    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data_full")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--total", type=int, required=True,
                        help="Target total episodes across all topologies")
    parser.add_argument("--val-frac", type=float, default=0.10)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    n_topos = len(TOPOLOGIES)
    per_topo = args.total // n_topos
    if per_topo * n_topos != args.total:
        print(f"Warning: {args.total} not divisible by {n_topos}, using {per_topo}/topo "
              f"→ {per_topo * n_topos} total")

    print(f"\nSubsampling {src}/ -> {dst}/  ({per_topo} eps x {n_topos} topos = "
          f"{per_topo * n_topos} total)\n")

    summary = {"source": str(src), "target_total": args.total,
               "per_topo_target": per_topo, "per_topology": {}, "total_picked": 0}

    for name, split in TOPOLOGIES:
        src_h5 = src / f"{name}.h5"
        dst_h5 = dst / f"{name}.h5"
        picked = subsample_topology(str(src_h5), str(dst_h5), per_topo, args.val_frac)
        mix = Counter(e.split("_", 2)[-1] for e in picked)
        size_mb = os.path.getsize(dst_h5) / 1024 / 1024
        print(f"  {name:<24} {split:<5} {len(picked):>4} eps  {size_mb:>6.1f} MB  "
              f"{dict(mix)}")
        summary["per_topology"][name] = {
            "split": split, "n_episodes": len(picked),
            "policy_mix": dict(mix), "size_mb": round(size_mb, 1),
        }
        summary["total_picked"] += len(picked)

    print("\nComputing normalization stats (train topos)...")
    norm_stats = {}
    for name, split in TOPOLOGIES:
        if split != "train":
            continue
        try:
            stats = compute_normalization_stats(str(dst / f"{name}.h5"))
            norm_stats[name] = stats
            print(f"  {name}: GSNR mean={stats['channel_gsnr']['mean']:.2f} "
                  f"std={stats['channel_gsnr']['std']:.2f}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    if norm_stats:
        gsnr_means = [s["channel_gsnr"]["mean"] for s in norm_stats.values()]
        gsnr_stds = [s["channel_gsnr"]["std"] for s in norm_stats.values()]
        agg_norm = {
            "channel_gsnr": {"mean": float(np.mean(gsnr_means)),
                             "std": float(np.mean(gsnr_stds))},
            "per_topology": norm_stats,
        }
        norm_path = dst / "normalization.json"
        with open(norm_path, "w") as f:
            json.dump(agg_norm, f, indent=2, default=str)
        print(f"  Wrote {norm_path}")

    meta_path = dst / f"dataset_{dst.name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {meta_path}")

    print(f"\nDone. {summary['total_picked']} episodes in {dst}/\n")


if __name__ == "__main__":
    main()
