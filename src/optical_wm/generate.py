"""
Dataset Generation Script for Optical Network World Model.

Aligned with LeWM dataset design principles:
  - Multi-topology (3 train + 1 test) for generalization
  - 5,000 episodes × 80-100 steps ≈ 450K transitions
  - 5 collection policies with diverse action types
  - Goal-conditioned evaluation support (episodes long enough for 25-50 step horizons)
  - Normalization statistics for training
  - Train/val split per topology

Usage (inside GNPy Docker):
  python -m optical_wm.generate --smoke     # 15 eps, ~2 min — sanity check
  python -m optical_wm.generate --small     # 100 eps, ~10 min — validate stats
  python -m optical_wm.generate --full      # 5000 eps, ~4-8 hours — real dataset
  python -m optical_wm.generate --full --resume  # resume interrupted run
"""
import argparse
import json
import time
import sys
import os
import io
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter

from .core.schemas import (
    MAX_NODES, MAX_LINKS, MAX_SLOTS, ActionType, MOD_THRESHOLDS, Modulation,
)
from .core.gnpy_wrapper import (
    build_test_topology, create_evaluator, TopologySpec,
    NetworkEvaluator, GNPY_AVAILABLE,
)
from .core.hdf5_io import HDF5Writer, HDF5Reader
from .policies.provisioning import ProvisioningPolicy, ProvisioningConfig
from .policies.margin_optimization import MarginOptPolicy, MarginOptConfig
from .policies.load_balancing import LoadBalancePolicy, LoadBalanceConfig
from .policies.recovery import RecoveryPolicy, RecoveryConfig
from .policies.mixed_ops import MixedOpsPolicy, MixedOpsConfig


# =====================================================================
# Topology definitions (3 train + 1 test)
# =====================================================================

@dataclass
class TopologyDef:
    """Definition of a topology family."""
    name: str
    n_nodes: int
    connectivity: float
    seed: int
    split: str  # 'train' or 'test'


TOPOLOGIES = [
    TopologyDef("topo_A_small_mesh",   n_nodes=8,  connectivity=0.40, seed=100, split="train"),
    TopologyDef("topo_B_medium_sparse", n_nodes=12, connectivity=0.25, seed=200, split="train"),
    TopologyDef("topo_C_medium_mesh",  n_nodes=14, connectivity=0.30, seed=300, split="train"),
    TopologyDef("topo_D_test",         n_nodes=10, connectivity=0.35, seed=400, split="test"),
]


# =====================================================================
# Policy mix (per topology)
# =====================================================================

@dataclass
class PolicyMix:
    """Number of episodes per policy for one topology."""
    provisioning: int
    margin_opt: int
    load_balance: int
    recovery: int
    mixed_ops: int

    @property
    def total(self) -> int:
        return (self.provisioning + self.margin_opt + self.load_balance +
                self.recovery + self.mixed_ops)

    def items(self):
        return [
            ("provisioning", self.provisioning),
            ("margin_opt", self.margin_opt),
            ("load_balance", self.load_balance),
            ("recovery", self.recovery),
            ("mixed_ops", self.mixed_ops),
        ]


# =====================================================================
# Run configurations
# =====================================================================

@dataclass
class RunConfig:
    """Full generation configuration."""
    name: str
    output_dir: str = "data"
    max_steps: int = 80
    val_frac: float = 0.10  # 10% validation split

    # Per-topology episode counts
    # Total = 4 topologies × per_topo.total
    per_topo: PolicyMix = None

    # Topology subset (None = all)
    topologies: List[TopologyDef] = None

    def __post_init__(self):
        if self.topologies is None:
            self.topologies = TOPOLOGIES

    @property
    def total_episodes(self) -> int:
        return self.per_topo.total * len(self.topologies)


SMOKE_CONFIG = RunConfig(
    name="smoke",
    max_steps=20,
    per_topo=PolicyMix(
        provisioning=1, margin_opt=1, load_balance=1,
        recovery=1, mixed_ops=1,
    ),
    topologies=[TOPOLOGIES[0]],  # only topo A for smoke
)

SMALL_CONFIG = RunConfig(
    name="small",
    max_steps=60,
    per_topo=PolicyMix(
        provisioning=15, margin_opt=10, load_balance=8,
        recovery=10, mixed_ops=7,
    ),
    topologies=[TOPOLOGIES[0], TOPOLOGIES[3]],  # topo A + D
)

FULL_CONFIG = RunConfig(
    name="full",
    max_steps=80,
    per_topo=PolicyMix(
        provisioning=375, margin_opt=250, load_balance=190,
        recovery=250, mixed_ops=185,
    ),
    # 1250 eps/topo × 4 topos = 5000 total
)


# =====================================================================
# Policy factory
# =====================================================================

def make_policy(name: str, evaluator: NetworkEvaluator,
                spec: TopologySpec, seed: int, max_steps: int):
    if name == "provisioning":
        return ProvisioningPolicy(evaluator, spec, ProvisioningConfig(
            seed=seed, max_steps=max_steps,
            initial_load_frac=0.15, target_load_frac=0.9,
            n_demands=max_steps + 30,
        ))
    elif name == "margin_opt":
        return MarginOptPolicy(evaluator, spec, MarginOptConfig(
            seed=seed, max_steps=max_steps, initial_load_frac=0.50,
        ))
    elif name == "load_balance":
        return LoadBalancePolicy(evaluator, spec, LoadBalanceConfig(
            seed=seed, max_steps=max_steps, initial_load_frac=0.45,
        ))
    elif name == "recovery":
        return RecoveryPolicy(evaluator, spec, RecoveryConfig(
            seed=seed, max_steps=max_steps,
            initial_load_frac=0.60, disruption_frac=0.4,
        ))
    elif name == "mixed_ops":
        return MixedOpsPolicy(evaluator, spec, MixedOpsConfig(
            seed=seed, max_steps=max_steps, epsilon=0.10,
        ))
    else:
        raise ValueError(f"Unknown policy: {name}")


# =====================================================================
# Validation functions
# =====================================================================

def validate_step(state_t: dict, action: np.ndarray, state_t1: dict,
                  step_idx: int, episode_id: str) -> List[str]:
    """Per-step physical validation. Returns list of warnings."""
    warns = []
    action_type = int(action[0])

    # V1: GSNR in physical range [0, 50] dB for active channels
    lp_mask_t1 = state_t1['lp_mask']
    lp_feat_t1 = state_t1['lp_features']
    for i in range(int(lp_mask_t1.sum())):
        if lp_mask_t1[i]:
            gsnr = lp_feat_t1[i, 17]  # index 17 = gsnr_db
            if gsnr < 0 or gsnr > 50:
                warns.append(
                    f"{episode_id} step {step_idx}: GSNR={gsnr:.1f} dB out of [0,50]"
                )
                break  # one warning per step is enough

    # V2: Margin consistency (margin ≈ gsnr - threshold)
    for i in range(int(lp_mask_t1.sum())):
        if lp_mask_t1[i]:
            gsnr = lp_feat_t1[i, 17]
            margin = lp_feat_t1[i, 18]
            mod_idx = int(lp_feat_t1[i, 13])
            mod = Modulation(mod_idx) if 0 <= mod_idx <= 2 else Modulation.QPSK
            threshold = MOD_THRESHOLDS.get(mod, 8.5)
            expected_margin = gsnr - threshold
            if abs(margin - expected_margin) > 0.5 and gsnr > 0:
                warns.append(
                    f"{episode_id} step {step_idx}: margin={margin:.2f} != "
                    f"gsnr({gsnr:.2f})-threshold({threshold:.1f})={expected_margin:.2f}"
                )
                break

    # V3: Action-state coherence
    n_lps_before = float(state_t['global_features'][0])
    n_lps_after = float(state_t1['global_features'][0])

    if action_type == ActionType.ADD and n_lps_after < n_lps_before - 0.5:
        warns.append(
            f"{episode_id} step {step_idx}: ADD but LPs decreased "
            f"{n_lps_before:.0f}→{n_lps_after:.0f}"
        )
    if action_type == ActionType.REMOVE and n_lps_after > n_lps_before + 0.5:
        warns.append(
            f"{episode_id} step {step_idx}: REMOVE but LPs increased "
            f"{n_lps_before:.0f}→{n_lps_after:.0f}"
        )

    # V4: Spectrum changed for ADD/REMOVE/REROUTE (not required for POWER/MOD)
    if action_type in (ActionType.ADD, ActionType.REMOVE, ActionType.REROUTE):
        occ_t = state_t['spectral_occupancy']
        occ_t1 = state_t1['spectral_occupancy']
        if np.array_equal(occ_t, occ_t1):
            warns.append(
                f"{episode_id} step {step_idx}: spectrum unchanged after "
                f"{ActionType(action_type).name}"
            )

    return warns


def validate_episode(episode: dict, episode_id: str) -> List[str]:
    """Per-episode validation with per-step checks. Returns list of warnings."""
    warns = []
    states = episode.get('states', [])
    actions = episode.get('actions', [])

    if len(states) == 0:
        warns.append(f"{episode_id}: no states")
        return warns
    if len(actions) == 0:
        warns.append(f"{episode_id}: no actions")
    if len(states) != len(actions) + 1:
        warns.append(f"{episode_id}: states({len(states)})/actions({len(actions)}) mismatch")

    # Check for zero-LP states (GNPy total failure)
    n_zero = sum(1 for s in states if s['global_features'][0] == 0)
    if n_zero > len(states) * 0.3:
        warns.append(f"{episode_id}: {n_zero}/{len(states)} zero-LP states")

    # Check GSNR range is plausible
    gsnrs = [s['global_features'][5] for s in states]  # avg margin
    if max(gsnrs) - min(gsnrs) < 0.01:
        warns.append(f"{episode_id}: flat GSNR trajectory")

    # Per-step validation (sample up to 10 steps to avoid slowdown)
    n_check = min(len(actions), 10)
    step_indices = np.linspace(0, len(actions) - 1, n_check, dtype=int)
    step_warns = 0
    for t in step_indices:
        sw = validate_step(states[t], actions[t], states[t + 1], t, episode_id)
        if sw:
            step_warns += 1
            if step_warns <= 3:  # limit verbose output
                warns.extend(sw)

    if step_warns > 3:
        warns.append(f"{episode_id}: {step_warns} steps with physics warnings (showing 3)")

    return warns


def validate_dataset(h5_path: str, topo_name: str) -> dict:
    """Dataset-level validation on a single HDF5 file."""
    reader = HDF5Reader(h5_path)
    reader.open()

    n_eps = reader.get_episode_count()
    ep_list = reader.list_episodes()

    all_gsnr = []
    all_action_types = []
    coupling_count = 0
    coupling_tested = 0
    policy_counts = Counter()

    for eid in ep_list:
        info = reader.get_episode_info(eid)
        n_steps = info['n_steps']
        meta = info.get('metadata', {})
        policy = meta.get('policy', 'unknown')
        policy_counts[policy] += 1

        # Load a subsequence to check stats
        try:
            sub = reader.load_subsequence(eid, 0, min(n_steps, 10))
            gsnr_data = sub['channel_gsnr']
            active = gsnr_data[gsnr_data > 0]
            if len(active) > 0:
                all_gsnr.extend(active.flatten().tolist())

            # Check inter-channel coupling: did adding a channel affect others?
            if n_steps >= 3 and 'actions' in sub:
                for t in range(min(len(sub['actions']), 5)):
                    action_type = int(sub['actions'][t][0])
                    all_action_types.append(action_type)

                    if action_type == ActionType.ADD:
                        coupling_tested += 1
                        occ_t = sub['spectral_occupancy'][t]
                        occ_t1 = sub['spectral_occupancy'][t + 1]
                        gsnr_t = sub['channel_gsnr'][t]
                        gsnr_t1 = sub['channel_gsnr'][t + 1]

                        # Check if existing channels' GSNR changed
                        mask = occ_t & occ_t1  # channels present in both
                        if mask.any():
                            delta = np.abs(gsnr_t[mask] - gsnr_t1[mask])
                            if delta.max() > 0.001:
                                coupling_count += 1
        except Exception:
            pass

    reader.close()

    # Compute stats
    all_gsnr = np.array(all_gsnr) if all_gsnr else np.array([0.0])
    action_dist = Counter(all_action_types)
    action_names = {ActionType(t).name: c for t, c in action_dist.items()}

    coupling_pct = (coupling_count / max(1, coupling_tested)) * 100

    result = {
        'topology': topo_name,
        'n_episodes': n_eps,
        'policy_distribution': dict(policy_counts),
        'gsnr_min': float(all_gsnr.min()),
        'gsnr_max': float(all_gsnr.max()),
        'gsnr_mean': float(all_gsnr.mean()),
        'gsnr_std': float(all_gsnr.std()),
        'action_distribution': action_names,
        'coupling_pct': coupling_pct,
        'coupling_tested': coupling_tested,
    }

    return result


def compute_normalization_stats(h5_path: str, n_sample: int = 50) -> dict:
    """Compute per-feature mean/std for training normalization."""
    reader = HDF5Reader(h5_path)
    reader.open()
    ep_list = reader.list_episodes()

    # Sample episodes
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(ep_list, size=min(n_sample, len(ep_list)),
                            replace=False)

    gsnr_vals, power_vals, global_vals = [], [], []

    for eid in sample_ids:
        info = reader.get_episode_info(eid)
        n_steps = info['n_steps']
        try:
            sub = reader.load_subsequence(eid, 0, n_steps)
            gsnr = sub['channel_gsnr']
            power = sub['channel_power']
            glob = sub['global_features']

            active_gsnr = gsnr[gsnr > 0]
            active_power = power[power != 0]

            if len(active_gsnr) > 0:
                gsnr_vals.extend(active_gsnr.flatten().tolist())
            if len(active_power) > 0:
                power_vals.extend(active_power.flatten().tolist())
            global_vals.append(glob)
        except Exception:
            pass

    reader.close()

    gsnr_arr = np.array(gsnr_vals) if gsnr_vals else np.array([0.0])
    power_arr = np.array(power_vals) if power_vals else np.array([0.0])
    global_arr = np.concatenate(global_vals, axis=0) if global_vals else np.zeros((1, 8))

    stats = {
        'channel_gsnr': {
            'mean': float(gsnr_arr.mean()),
            'std': float(gsnr_arr.std()),
            'min': float(gsnr_arr.min()),
            'max': float(gsnr_arr.max()),
        },
        'channel_power': {
            'mean': float(power_arr.mean()),
            'std': float(power_arr.std()),
            'min': float(power_arr.min()),
            'max': float(power_arr.max()),
        },
        'global_features': {
            'mean': global_arr.mean(axis=0).tolist(),
            'std': global_arr.std(axis=0).tolist(),
        },
        'probing_targets': [
            'global_features[0]: n_active_lightpaths',
            'global_features[1]: total_capacity_tbps',
            'global_features[4]: worst_margin_db',
            'global_features[5]: avg_margin_db',
            'global_features[6]: spectral_utilization',
        ],
    }
    return stats


# =====================================================================
# Topology → HDF5
# =====================================================================

def write_topology_to_h5(writer: HDF5Writer, spec: TopologySpec):
    """Write static topology data to HDF5."""
    n_nodes = len(spec.node_ids)
    n_links = len(spec.links)

    adjacency = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    for link in spec.links:
        i = spec.node_ids.index(link['src'])
        j = spec.node_ids.index(link['dst'])
        adjacency[i, j] = adjacency[j, i] = 1.0

    node_features = np.zeros((MAX_NODES, 2), dtype=np.float32)
    for i, nid in enumerate(spec.node_ids):
        loc = spec.node_locations.get(nid, (0.0, 0.0))
        node_features[i] = [loc[0], loc[1]]

    link_static = np.zeros((MAX_LINKS, 3), dtype=np.float32)
    link_endpoints = np.zeros((MAX_LINKS, 2), dtype=np.int32)
    for idx, link in enumerate(spec.links):
        link_static[idx] = [
            link['length_km'], link.get('n_spans', 1), link.get('n_amps', 1),
        ]
        link_endpoints[idx] = [
            spec.node_ids.index(link['src']),
            spec.node_ids.index(link['dst']),
        ]

    node_mask = np.zeros(MAX_NODES, dtype=bool)
    node_mask[:n_nodes] = True
    link_mask = np.zeros(MAX_LINKS, dtype=bool)
    link_mask[:n_links] = True

    writer.write_topology(
        adjacency=adjacency, node_features=node_features,
        link_static=link_static, link_endpoints=link_endpoints,
        node_mask=node_mask, link_mask=link_mask,
        metadata={'node_ids': spec.node_ids,
                  'n_nodes': n_nodes, 'n_links': n_links},
    )


# =====================================================================
# Episode plan
# =====================================================================

def build_episode_plan(
    policy_mix: PolicyMix, base_seed: int,
) -> List[Tuple[str, int]]:
    """Build (policy_name, seed) list for one topology."""
    plan = []
    seed = base_seed
    for policy_name, count in policy_mix.items():
        for i in range(count):
            plan.append((policy_name, seed + i))
        seed += count
    return plan


# =====================================================================
# Generate one topology
# =====================================================================

def generate_topology(
    topo_def: TopologyDef,
    policy_mix: PolicyMix,
    max_steps: int,
    output_dir: Path,
    val_frac: float,
    resume: bool = False,
) -> dict:
    """Generate all episodes for one topology. Returns stats dict."""
    h5_path = str(output_dir / f"{topo_def.name}.h5")
    plan = build_episode_plan(policy_mix, base_seed=topo_def.seed)
    total = len(plan)

    # Resume
    start_idx = 0
    if resume and os.path.exists(h5_path):
        try:
            r = HDF5Reader(h5_path)
            r.open()
            start_idx = r.get_episode_count()
            r.close()
            print(f"    Resuming {topo_def.name} from episode {start_idx}/{total}")
        except Exception:
            start_idx = 0
    elif os.path.exists(h5_path) and not resume:
        os.remove(h5_path)

    # Build network (suppress GNPy noise)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        spec = build_test_topology(
            n_nodes=topo_def.n_nodes,
            connectivity=topo_def.connectivity,
            seed=topo_def.seed,
        )
        evaluator, spec = create_evaluator(spec)
    finally:
        sys.stdout, sys.stderr = old_out, old_err

    print(f"    Network: {len(spec.node_ids)} nodes, {len(spec.links)} links")

    # Open writer
    writer = HDF5Writer(h5_path)
    writer.open()
    if start_idx == 0:
        write_topology_to_h5(writer, spec)

    # Generate
    policy_stats = {}
    all_warnings = []
    total_transitions = 0
    gen_start = time.time()

    for idx in range(start_idx, total):
        policy_name, seed = plan[idx]
        episode_id = f"ep_{idx:05d}_{policy_name}"

        # Suppress GNPy output
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            policy = make_policy(policy_name, evaluator, spec, seed, max_steps)
            episode = policy.generate_episode()
        except Exception as e:
            sys.stdout, sys.stderr = old_out, old_err
            print(f"    ERROR {episode_id}: {e}")
            all_warnings.append(f"{episode_id}: {e}")
            continue
        finally:
            sys.stdout, sys.stderr = old_out, old_err

        # Validate
        ep_warns = validate_episode(episode, episode_id)
        all_warnings.extend(ep_warns)

        # Write
        n_actions = len(episode['actions'])
        n_states = len(episode['states'])
        if n_actions > 0:
            try:
                writer.write_episode(
                    episode_id=episode_id,
                    n_steps=n_states,
                    states=episode['states'],
                    actions=episode['actions'],
                    metadata=episode['metadata'],
                )
            except Exception as e:
                print(f"    WRITE ERROR {episode_id}: {e}")
                all_warnings.append(f"{episode_id}: WRITE {e}")
                continue

        # Stats
        if policy_name not in policy_stats:
            policy_stats[policy_name] = {
                'count': 0, 'total_steps': 0, 'action_types': {},
            }
        ps = policy_stats[policy_name]
        ps['count'] += 1
        ps['total_steps'] += n_actions
        total_transitions += n_actions

        for a in episode['actions']:
            atype = ActionType(int(a[0])).name
            ps['action_types'][atype] = ps['action_types'].get(atype, 0) + 1

        # Progress
        done = idx - start_idx + 1
        elapsed = time.time() - gen_start
        avg_per_ep = elapsed / done
        eta_min = (total - idx - 1) * avg_per_ep / 60

        if done <= 2 or done % max(1, total // 10) == 0 or idx == total - 1:
            print(
                f"    [{done:>5}/{total}] {policy_name:>15} | "
                f"{n_actions:>3} steps | ETA {eta_min:.1f}m"
                f"{' ⚠' if ep_warns else ''}"
            )

    writer.close()
    gen_time = time.time() - gen_start

    # Write train/val split
    _write_split(h5_path, val_frac)

    actual = sum(ps['count'] for ps in policy_stats.values())

    return {
        'topology': topo_def.name,
        'split': topo_def.split,
        'n_nodes': topo_def.n_nodes,
        'n_links': len(spec.links),
        'episodes': actual,
        'transitions': total_transitions,
        'time_s': gen_time,
        'policy_stats': policy_stats,
        'n_warnings': len(all_warnings),
        'warnings': all_warnings[:20],
        'h5_path': h5_path,
    }


def _write_split(h5_path: str, val_frac: float):
    """Add train/val split to HDF5 file."""
    import h5py
    with h5py.File(h5_path, 'a') as f:
        if 'split' in f:
            del f['split']

        ep_names = list(f['episodes'].keys())
        n_val = max(1, int(len(ep_names) * val_frac))

        rng = np.random.default_rng(42)
        rng.shuffle(ep_names)

        val_eps = ep_names[:n_val]
        train_eps = ep_names[n_val:]

        split = f.create_group('split')
        split.create_dataset('train', data=[e.encode() for e in train_eps])
        split.create_dataset('val', data=[e.encode() for e in val_eps])


# =====================================================================
# Main orchestrator
# =====================================================================

def run_generation(config: RunConfig, resume: bool = False):
    if not GNPY_AVAILABLE:
        print("ERROR: GNPy not available. Run inside Docker.")
        sys.exit(1)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_all = config.per_topo.total * len(config.topologies)

    print(f"\n{'='*65}")
    print(f"  Optical Network World Model — Dataset Generation")
    print(f"{'='*65}")
    print(f"  Mode:        {config.name}")
    print(f"  Topologies:  {len(config.topologies)} "
          f"({sum(1 for t in config.topologies if t.split=='train')} train, "
          f"{sum(1 for t in config.topologies if t.split=='test')} test)")
    print(f"  Eps/topo:    {config.per_topo.total}")
    print(f"  Total eps:   {total_all}")
    print(f"  Steps/ep:    {config.max_steps}")
    print(f"  Est. trans:  ~{total_all * config.max_steps * 0.7:,.0f}")
    print(f"  Output:      {output_dir}/")
    print(f"  Val split:   {config.val_frac:.0%}")
    print()

    all_results = []
    grand_start = time.time()

    for topo_def in config.topologies:
        print(f"  ── Topology: {topo_def.name} ({topo_def.n_nodes}n, "
              f"conn={topo_def.connectivity}, {topo_def.split}) ──")

        result = generate_topology(
            topo_def=topo_def,
            policy_mix=config.per_topo,
            max_steps=config.max_steps,
            output_dir=output_dir,
            val_frac=config.val_frac,
            resume=resume,
        )
        all_results.append(result)

        print(f"    Done: {result['episodes']} eps, "
              f"{result['transitions']:,} transitions, "
              f"{result['time_s']/60:.1f} min")
        if result['n_warnings'] > 0:
            print(f"    ⚠ {result['n_warnings']} warnings")
        print()

    grand_time = time.time() - grand_start

    # ================================================================
    # Grand summary
    # ================================================================
    total_eps = sum(r['episodes'] for r in all_results)
    total_trans = sum(r['transitions'] for r in all_results)
    total_warns = sum(r['n_warnings'] for r in all_results)

    print(f"{'='*65}")
    print(f"  GENERATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Total episodes:    {total_eps}")
    print(f"  Total transitions: {total_trans:,}")
    print(f"  Total time:        {grand_time/60:.1f} min")
    print(f"  Warnings:          {total_warns}")
    print()

    # Per-topology table
    print(f"  {'Topology':<28} {'Split':<6} {'Eps':>5} {'Trans':>8} "
          f"{'Time':>6}  {'File':>8}")
    print(f"  {'-'*28} {'-'*6} {'-'*5} {'-'*8} {'-'*6}  {'-'*8}")
    for r in all_results:
        fsize = os.path.getsize(r['h5_path']) / 1024 / 1024
        print(f"  {r['topology']:<28} {r['split']:<6} {r['episodes']:>5} "
              f"{r['transitions']:>8,} {r['time_s']/60:>5.1f}m  {fsize:>6.1f}MB")
    print()

    # Aggregate action distribution
    agg_actions = Counter()
    for r in all_results:
        for ps in r['policy_stats'].values():
            for atype, count in ps['action_types'].items():
                agg_actions[atype] += count
    total_actions = sum(agg_actions.values())
    print("  Action distribution (all topologies):")
    for atype, count in sorted(agg_actions.items()):
        print(f"    {atype:<16} {count:>7}  ({count/total_actions:>5.1%})")
    print()

    # ================================================================
    # Dataset-level validation
    # ================================================================
    print(f"  Validating datasets...")
    validation_results = []
    for r in all_results:
        try:
            v = validate_dataset(r['h5_path'], r['topology'])
            validation_results.append(v)
            print(f"    {r['topology']}: {v['n_episodes']} eps, "
                  f"GSNR=[{v['gsnr_min']:.1f}, {v['gsnr_max']:.1f}], "
                  f"coupling={v['coupling_pct']:.0f}% ✓")
        except Exception as e:
            print(f"    {r['topology']}: FAILED — {e}")

    # ================================================================
    # Normalization stats (train topologies only)
    # ================================================================
    print(f"\n  Computing normalization stats...")
    norm_stats = {}
    for r in all_results:
        if r.get('split') == 'train' or any(
            t.split == 'train' and t.name == r['topology']
            for t in config.topologies
        ):
            try:
                stats = compute_normalization_stats(r['h5_path'])
                norm_stats[r['topology']] = stats
                print(f"    {r['topology']}: GSNR μ={stats['channel_gsnr']['mean']:.1f} "
                      f"σ={stats['channel_gsnr']['std']:.1f}")
            except Exception as e:
                print(f"    {r['topology']}: FAILED — {e}")

    # Aggregate normalization across train topos
    if norm_stats:
        all_gsnr_means = [s['channel_gsnr']['mean'] for s in norm_stats.values()]
        all_gsnr_stds = [s['channel_gsnr']['std'] for s in norm_stats.values()]
        agg_norm = {
            'channel_gsnr': {
                'mean': float(np.mean(all_gsnr_means)),
                'std': float(np.mean(all_gsnr_stds)),
            },
            'per_topology': norm_stats,
        }
        norm_path = str(output_dir / "normalization.json")
        with open(norm_path, 'w') as f:
            json.dump(agg_norm, f, indent=2, default=str)
        print(f"    Saved: {norm_path}")

    # ================================================================
    # Save metadata
    # ================================================================
    meta = {
        'config': {
            'name': config.name,
            'max_steps': config.max_steps,
            'val_frac': config.val_frac,
            'topologies': [
                {'name': t.name, 'n_nodes': t.n_nodes,
                 'connectivity': t.connectivity, 'split': t.split}
                for t in config.topologies
            ],
            'per_topo_policy_mix': {
                k: v for k, v in config.per_topo.items()
            },
        },
        'results': {
            'total_episodes': total_eps,
            'total_transitions': total_trans,
            'total_time_min': grand_time / 60,
            'total_warnings': total_warns,
        },
        'per_topology': all_results,
        'validation': validation_results,
        'action_distribution': dict(agg_actions),
        'evaluation_spec': {
            'probing_targets': [
                'n_active_lightpaths', 'total_capacity_tbps',
                'worst_margin_db', 'avg_margin_db',
                'spectral_utilization',
            ],
            'planning_horizons': [10, 25, 50],
            'planning_budget': 50,
            'goal_sampling': 'same episode, t+k steps ahead',
        },
    }

    meta_path = str(output_dir / f"dataset_{config.name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  Metadata: {meta_path}")

    print(f"\n{'='*65}")
    print(f"  Done. {total_eps} episodes, {total_trans:,} transitions, "
          f"{grand_time/60:.1f} min.")
    print(f"{'='*65}")

    return all_results


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate optical network world model dataset"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke', action='store_true',
                       help='Smoke test: ~5 episodes, ~2 min')
    group.add_argument('--small', action='store_true',
                       help='Small: ~40 episodes, ~10 min')
    group.add_argument('--full', action='store_true',
                       help='Full: 5000 episodes, ~4-8 hours')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted run')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory')

    args = parser.parse_args()

    if args.smoke:
        config = SMOKE_CONFIG
    elif args.small:
        config = SMALL_CONFIG
    else:
        config = FULL_CONFIG

    config.output_dir = args.output
    run_generation(config, resume=args.resume)


if __name__ == "__main__":
    main()