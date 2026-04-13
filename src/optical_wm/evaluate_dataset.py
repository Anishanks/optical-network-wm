"""
Dataset Evaluation for Optical Network World Model.

Three-level evaluation:
  Level 1 — Integrity: sanity checks, coverage, physical consistency
  Level 2 — Dynamics: trajectory richness, action sensitivity, coupling
  Level 3 — Paper figures: publication-ready plots and tables

Usage:
  python -m optical_wm.evaluate_dataset --data data/ --report
  python -m optical_wm.evaluate_dataset --data data/ --figures --output figures/
  python -m optical_wm.evaluate_dataset --data data/ --all
"""
import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ─── Try importing plotting (optional) ───
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─── Try importing scipy for statistical tests (optional) ───
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─── Project imports ───
from .core.schemas import (
    ActionType, Modulation, MOD_THRESHOLDS,
    MAX_SLOTS, MAX_LINKS, MAX_LIGHTPATHS,
    N_ACTION_FEATURES,
)
from .core.hdf5_io import HDF5Reader


# =====================================================================
# Named indices for global_features array
# =====================================================================
# Avoids silent breakage if the HDF5 writer reorders columns.

GF_N_ACTIVE_LPS = 0
GF_TOTAL_CAPACITY_TBPS = 1
GF_N_FEASIBLE = 2
GF_N_INFEASIBLE = 3
GF_WORST_MARGIN_DB = 4
GF_AVG_MARGIN_DB = 5
GF_SPECTRAL_UTILIZATION = 6
GF_MAX_LINK_UTILIZATION = 7
GF_EXPECTED_DIM = 8  # total number of global features


# =====================================================================
# Data loading helpers
# =====================================================================

def find_h5_files(data_dir: str) -> List[str]:
    """Find all HDF5 dataset files."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.h5"))
    return [str(f) for f in files]


def load_all_transitions(h5_path: str, max_episodes: int = None,
                         max_steps_per_ep: int = None) -> dict:
    """
    Load transitions from one HDF5 file.
    Returns dict with arrays for analysis.
    """
    reader = HDF5Reader(h5_path)
    reader.open()

    ep_list = reader.list_episodes()
    if max_episodes:
        ep_list = ep_list[:max_episodes]

    data = {
        "gsnr_values": [],              # all active channel GSNRs
        "margin_values": [],            # all per-LP margins
        "load_per_step": [],            # spectral utilization per step
        "n_lps_per_step": [],           # LP count per step
        "action_types": [],             # action type per transition
        "action_load_pairs": [],        # (action_type, load_bucket) pairs
        "delta_gsnr_per_action": defaultdict(list),   # action_type → [delta_avg_gsnr]
        "delta_margin_per_action": defaultdict(list),  # action_type → [delta_avg_margin]
        "gsnr_trajectories": [],        # per-episode avg GSNR sequences
        "margin_trajectories": [],      # per-episode avg margin sequences
        "coupling_results": [],         # (n_existing, n_changed, max_delta) per ADD
        "policies": [],                 # policy name per episode
        "n_steps_per_ep": [],           # episode lengths
        "infeasible_counts": [],        # n_infeasible per step
        "episode_gsnr_ranges": [],      # max-min avg GSNR per episode
        "episode_margin_ranges": [],    # max-min avg margin per episode
        "episode_monotonic": [],        # True if GSNR is monotonic
        "episode_has_rise": [],         # GSNR rises at some point
        "episode_has_fall": [],         # GSNR falls at some point
    }

    for eid in ep_list:
        try:
            info = reader.get_episode_info(eid)
            n_steps = info["n_steps"]
            meta = info.get("metadata", {})
            policy = meta.get("policy", "unknown")
            data["policies"].append(policy)
            data["n_steps_per_ep"].append(n_steps)

            limit = min(n_steps, max_steps_per_ep) if max_steps_per_ep else n_steps
            sub = reader.load_subsequence(eid, 0, limit)

            # ── Validate global_features shape ──
            gf_shape = sub["global_features"].shape
            assert gf_shape[-1] == GF_EXPECTED_DIM, (
                f"global_features has {gf_shape[-1]} columns, "
                f"expected {GF_EXPECTED_DIM}. Schema mismatch — "
                f"check HDF5 writer column order."
            )

            # ── Per-step metrics ──
            gsnr_seq = []    # track avg GSNR (from active channel GSNRs)
            margin_seq = []  # track avg margin (from global features)
            for t in range(limit):
                gf = sub["global_features"][t]
                n_active = gf[GF_N_ACTIVE_LPS]
                util = gf[GF_SPECTRAL_UTILIZATION]
                n_infeasible = gf[GF_N_INFEASIBLE]
                avg_margin = gf[GF_AVG_MARGIN_DB]

                data["load_per_step"].append(float(util))
                data["n_lps_per_step"].append(float(n_active))
                data["infeasible_counts"].append(float(n_infeasible))
                margin_seq.append(float(avg_margin))

                # Active channel GSNRs
                ch_gsnr = sub["channel_gsnr"][t]
                occ = sub["spectral_occupancy"][t]
                active_gsnr = ch_gsnr[occ]
                if len(active_gsnr) > 0:
                    data["gsnr_values"].extend(active_gsnr.tolist())
                    gsnr_seq.append(float(active_gsnr.mean()))
                else:
                    gsnr_seq.append(0.0)

                # Per-LP margins — use boolean mask directly
                lp_mask = sub["lp_mask"][t]
                lp_feat = sub["lp_features"][t]
                if lp_mask.any():
                    margins = lp_feat[lp_mask, 18]
                    data["margin_values"].extend(margins.tolist())

            # ── Per-transition metrics ──
            if "actions" in sub and limit > 1:
                n_actions = min(len(sub["actions"]), limit - 1)
                for t in range(n_actions):
                    action_type = int(sub["actions"][t][0])
                    data["action_types"].append(action_type)

                    # Load bucket (by 10%)
                    load = sub["global_features"][t][GF_SPECTRAL_UTILIZATION]
                    load_bucket = int(load * 10) * 10  # 0, 10, 20, ...
                    data["action_load_pairs"].append((action_type, load_bucket))

                    # Delta avg GSNR (from per-channel means)
                    delta_gsnr = gsnr_seq[t + 1] - gsnr_seq[t]
                    data["delta_gsnr_per_action"][action_type].append(delta_gsnr)

                    # Delta avg margin (from global features)
                    margin_before = sub["global_features"][t][GF_AVG_MARGIN_DB]
                    margin_after = sub["global_features"][t + 1][GF_AVG_MARGIN_DB]
                    delta_margin = float(margin_after - margin_before)
                    data["delta_margin_per_action"][action_type].append(delta_margin)

                    # Inter-channel coupling for ADD actions
                    if action_type == ActionType.ADD:
                        occ_t = sub["spectral_occupancy"][t]
                        occ_t1 = sub["spectral_occupancy"][t + 1]
                        gsnr_t = sub["channel_gsnr"][t]
                        gsnr_t1 = sub["channel_gsnr"][t + 1]

                        # Channels present in BOTH steps (existing, not the new one)
                        both = occ_t & occ_t1
                        n_existing = int(both.sum())
                        if n_existing > 0:
                            deltas = np.abs(gsnr_t[both] - gsnr_t1[both])
                            n_changed = int((deltas > 0.001).sum())
                            max_delta = float(deltas.max())
                            data["coupling_results"].append(
                                (n_existing, n_changed, max_delta)
                            )

            # ── Per-episode metrics ──
            data["gsnr_trajectories"].append(gsnr_seq)
            data["margin_trajectories"].append(margin_seq)

            if len(gsnr_seq) >= 3:
                gsnr_range = max(gsnr_seq) - min(gsnr_seq)
                data["episode_gsnr_ranges"].append(gsnr_range)

                margin_range = max(margin_seq) - min(margin_seq)
                data["episode_margin_ranges"].append(margin_range)

                # Non-monotonicity based on GSNR (physical variable)
                diffs = [gsnr_seq[i + 1] - gsnr_seq[i]
                         for i in range(len(gsnr_seq) - 1)]
                has_rise = any(d > 0.01 for d in diffs)
                has_fall = any(d < -0.01 for d in diffs)
                data["episode_has_rise"].append(has_rise)
                data["episode_has_fall"].append(has_fall)
                data["episode_monotonic"].append(
                    not (has_rise and has_fall)
                )

        except Exception as e:
            print(f"  WARNING: Error loading {eid}: {e}")
            continue

    reader.close()
    return data


def merge_data(datasets: List[dict]) -> dict:
    """Merge data from multiple HDF5 files."""
    merged = {
        "gsnr_values": [],
        "margin_values": [],
        "load_per_step": [],
        "n_lps_per_step": [],
        "action_types": [],
        "action_load_pairs": [],
        "delta_gsnr_per_action": defaultdict(list),
        "delta_margin_per_action": defaultdict(list),
        "gsnr_trajectories": [],
        "margin_trajectories": [],
        "coupling_results": [],
        "policies": [],
        "n_steps_per_ep": [],
        "infeasible_counts": [],
        "episode_gsnr_ranges": [],
        "episode_margin_ranges": [],
        "episode_monotonic": [],
        "episode_has_rise": [],
        "episode_has_fall": [],
    }

    for d in datasets:
        for key in merged:
            if key in ("delta_gsnr_per_action", "delta_margin_per_action"):
                for atype, vals in d[key].items():
                    merged[key][atype].extend(vals)
            elif isinstance(merged[key], list):
                merged[key].extend(d[key])

    return merged


# =====================================================================
# Level 1 — Integrity Checks
# =====================================================================

def evaluate_integrity(data: dict) -> dict:
    """
    Sanity checks. Returns dict of {check_name: (passed, message)}.
    """
    results = {}

    # ── 1.1 GSNR range ──
    gsnr = np.array(data["gsnr_values"])
    if len(gsnr) > 0:
        gmin, gmax = gsnr.min(), gsnr.max()
        gmean, gstd = gsnr.mean(), gsnr.std()
        passed = gmin > -5 and gmax < 55 and gstd > 1.0
        results["gsnr_range"] = (
            passed,
            f"[{gmin:.1f}, {gmax:.1f}] dB, μ={gmean:.1f}, σ={gstd:.1f}"
        )
    else:
        results["gsnr_range"] = (False, "No GSNR values found")

    # ── 1.2 Load coverage ──
    n_lps = np.array(data["n_lps_per_step"])
    if len(n_lps) > 0:
        lmin, lmax = n_lps.min(), n_lps.max()
        lp_range = lmax - lmin
        passed = lp_range > 30
        results["load_coverage"] = (
            passed,
            f"LP count range: [{lmin:.0f}, {lmax:.0f}], "
            f"spread={lp_range:.0f} LPs"
        )
    else:
        results["load_coverage"] = (False, "No load data")

    # ── 1.3 Action type coverage ──
    action_counts = Counter(data["action_types"])
    total_actions = len(data["action_types"])
    if total_actions > 0:
        fracs = {ActionType(t).name: c / total_actions
                 for t, c in action_counts.items()}
        min_frac = min(fracs.values()) if fracs else 0
        n_types = len(fracs)
        passed = n_types >= 4 and min_frac > 0.03
        detail = ", ".join(f"{n}={f:.1%}" for n, f in sorted(fracs.items()))
        results["action_coverage"] = (passed, f"{n_types} types: {detail}")
    else:
        results["action_coverage"] = (False, "No actions found")

    # ── 1.4 Policy coverage ──
    policy_counts = Counter(data["policies"])
    n_policies = len(policy_counts)
    passed = n_policies >= 4
    detail = ", ".join(f"{p}={c}" for p, c in sorted(policy_counts.items()))
    results["policy_coverage"] = (passed, f"{n_policies} policies: {detail}")

    # ── 1.5 Inter-channel coupling ──
    coupling = data["coupling_results"]
    if len(coupling) >= 10:
        n_with_coupling = sum(1 for _, nc, _ in coupling if nc > 0)
        coupling_rate = n_with_coupling / len(coupling)
        avg_changed = np.mean([nc for _, nc, _ in coupling])
        deltas_positive = [md for _, _, md in coupling if md > 0]
        avg_delta = np.mean(deltas_positive) if deltas_positive else 0
        passed = coupling_rate > 0.5
        results["coupling_present"] = (
            passed,
            f"{coupling_rate:.0%} of ADD transitions show coupling "
            f"(avg {avg_changed:.1f} channels affected, "
            f"avg Δ={avg_delta:.3f} dB)"
        )
    else:
        results["coupling_present"] = (
            False, f"Only {len(coupling)} ADD transitions sampled"
        )

    # ── 1.6 No excessive infeasible ──
    infeas = np.array(data["infeasible_counts"])
    n_lps_arr = np.array(data["n_lps_per_step"])
    if len(infeas) > 0 and len(n_lps_arr) > 0:
        mask = n_lps_arr > 0
        if mask.any():
            ratios = infeas[mask] / n_lps_arr[mask]
            avg_ratio = ratios.mean()
            passed = avg_ratio < 0.2
            results["infeasible_rate"] = (
                passed,
                f"Avg infeasible ratio: {avg_ratio:.1%}"
            )
        else:
            results["infeasible_rate"] = (True, "No active LPs to check")
    else:
        results["infeasible_rate"] = (False, "No data")

    # ── 1.7 Episode lengths ──
    steps = np.array(data["n_steps_per_ep"])
    if len(steps) > 0:
        passed = steps.mean() > 10 and steps.min() >= 3
        results["episode_lengths"] = (
            passed,
            f"μ={steps.mean():.0f}, min={steps.min()}, "
            f"max={steps.max()}, σ={steps.std():.1f}"
        )
    else:
        results["episode_lengths"] = (False, "No episodes")

    # ── 1.8 Load distribution uniformity ──
    loads = np.array(data["load_per_step"])
    if len(loads) > 100:
        # Check that we have sufficient coverage across load range
        # Bin into 10% buckets and check no bucket is empty
        bins = np.arange(0, 1.1, 0.1)
        hist, _ = np.histogram(loads, bins=bins)
        occupied_bins = (hist > 0).sum()
        total_bins = len(hist)
        # Also check no single bucket dominates (>50% of all steps)
        max_bin_frac = hist.max() / hist.sum() if hist.sum() > 0 else 1.0
        passed = occupied_bins >= 5 and max_bin_frac < 0.5
        results["load_distribution"] = (
            passed,
            f"{occupied_bins}/{total_bins} load buckets occupied, "
            f"max bucket has {max_bin_frac:.0%} of steps"
        )
    else:
        results["load_distribution"] = (False, "Not enough steps to evaluate")

    return results


# =====================================================================
# Level 2 — Dynamics Analysis
# =====================================================================

def evaluate_dynamics(data: dict) -> dict:
    """
    Assess trajectory richness and action sensitivity.
    Returns dict of {metric_name: (value, interpretation)}.
    """
    results = {}

    # ── 2.1 GSNR trajectory diversity (inter-episode) ──
    ranges = np.array(data["episode_gsnr_ranges"])
    if len(ranges) > 0:
        results["gsnr_range_per_episode"] = (
            f"μ={ranges.mean():.2f} dB, σ={ranges.std():.2f}",
            "Good" if ranges.mean() > 0.5
            else "Low — trajectories may be too flat"
        )

    margin_ranges = np.array(data["episode_margin_ranges"])
    if len(margin_ranges) > 0:
        results["margin_range_per_episode"] = (
            f"μ={margin_ranges.mean():.2f} dB, σ={margin_ranges.std():.2f}",
            "Good" if margin_ranges.mean() > 0.5
            else "Low — margin trajectories may be too flat"
        )

    # ── 2.2 Non-monotonicity (based on GSNR) ──
    mono = np.array(data["episode_monotonic"])
    rises = np.array(data["episode_has_rise"])
    falls = np.array(data["episode_has_fall"])
    if len(mono) > 0:
        non_mono_rate = 1 - mono.mean()
        rise_rate = rises.mean()
        fall_rate = falls.mean()
        results["non_monotonic_rate"] = (
            f"{non_mono_rate:.0%} of episodes are non-monotonic",
            "Good" if non_mono_rate > 0.2
            else "Low — need more recovery/mixed_ops"
        )
        results["rise_fall_rates"] = (
            f"Episodes with GSNR rise: {rise_rate:.0%}, "
            f"with GSNR fall: {fall_rate:.0%}",
            "Good" if rise_rate > 0.3 and fall_rate > 0.3
            else "Imbalanced — one direction dominates"
        )

    # ── 2.3 Action→GSNR sensitivity ──
    deltas = data["delta_gsnr_per_action"]
    if len(deltas) >= 2:
        delta_stats = {}
        for atype, vals in sorted(deltas.items()):
            arr = np.array(vals)
            if len(arr) > 0:
                delta_stats[ActionType(atype).name] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "n": len(arr),
                }

        # Spread between action types
        means = [s["mean"] for s in delta_stats.values()]
        mean_spread = max(means) - min(means) if means else 0
        results["action_delta_gsnr"] = (
            json.dumps(
                {k: f"μ={v['mean']:+.3f} σ={v['std']:.3f} (n={v['n']})"
                 for k, v in delta_stats.items()},
                indent=2,
            ),
            "Good" if mean_spread > 0.1
            else "Weak — actions produce similar GSNR effects"
        )
        results["action_effect_spread"] = (
            f"{mean_spread:.3f} dB spread between action types",
            "Good" if mean_spread > 0.1 else "Low"
        )

    # ── 2.4 Action discriminability (Kruskal-Wallis test) ──
    if HAS_SCIPY and len(deltas) >= 2:
        groups = []
        group_names = []
        for atype in sorted(deltas.keys()):
            vals = deltas[atype]
            if len(vals) >= 5:
                groups.append(vals)
                group_names.append(ActionType(atype).name)

        if len(groups) >= 2:
            stat, pvalue = scipy_stats.kruskal(*groups)
            passed = pvalue < 0.05
            results["action_discriminability"] = (
                f"Kruskal-Wallis H={stat:.1f}, p={pvalue:.2e} "
                f"across {len(groups)} action types ({', '.join(group_names)})",
                "Good — actions produce statistically distinguishable ΔGSNR"
                if passed
                else "FAIL — actions are NOT distinguishable (p≥0.05). "
                     "World model cannot learn action conditioning."
            )
    elif not HAS_SCIPY:
        results["action_discriminability"] = (
            "scipy not available",
            "Install scipy for Kruskal-Wallis test"
        )

    # ── 2.5 Conditional action diversity ──
    pairs = data["action_load_pairs"]
    if len(pairs) > 0:
        buckets = defaultdict(set)
        bucket_counts = defaultdict(lambda: defaultdict(int))
        for atype, load_bucket in pairs:
            buckets[load_bucket].add(atype)
            bucket_counts[load_bucket][atype] += 1

        diverse_buckets = sum(1 for v in buckets.values() if len(v) >= 2)
        total_buckets = len(buckets)
        max_diversity = max(len(v) for v in buckets.values()) if buckets else 0

        results["conditional_diversity"] = (
            f"{diverse_buckets}/{total_buckets} load buckets have ≥2 action types "
            f"(max {max_diversity} types in one bucket)",
            "Good" if diverse_buckets >= total_buckets * 0.5
            else "Low — state→action correlation too strong"
        )

    # ── 2.6 State-action coverage gaps ──
    if len(pairs) > 100:
        all_action_types = sorted(set(t for t, _ in pairs))
        all_load_buckets = sorted(set(b for _, b in pairs))

        # Build occupancy matrix
        min_transitions_per_cell = 5
        empty_cells = []
        sparse_cells = []
        for lb in all_load_buckets:
            for at in all_action_types:
                count = bucket_counts[lb][at]
                if count == 0:
                    empty_cells.append(
                        (f"{lb}%", ActionType(at).name)
                    )
                elif count < min_transitions_per_cell:
                    sparse_cells.append(
                        (f"{lb}%", ActionType(at).name, count)
                    )

        total_cells = len(all_action_types) * len(all_load_buckets)
        filled_cells = total_cells - len(empty_cells)
        coverage_pct = filled_cells / total_cells if total_cells > 0 else 0

        detail = f"{filled_cells}/{total_cells} cells filled ({coverage_pct:.0%})"
        if sparse_cells:
            n_sparse = len(sparse_cells)
            detail += f", {n_sparse} cells have <{min_transitions_per_cell} transitions"

        results["state_action_coverage"] = (
            detail,
            "Good" if coverage_pct > 0.5 and len(sparse_cells) < total_cells * 0.3
            else "Gaps detected — some (load, action) regions are blind"
        )

    # ── 2.7 Coupling strength vs load ──
    coupling = data["coupling_results"]
    if len(coupling) >= 10:
        by_load = defaultdict(list)
        for n_existing, n_changed, max_delta in coupling:
            bucket = (n_existing // 5) * 5
            by_load[bucket].append(max_delta)

        # Does coupling increase with load?
        load_buckets = sorted(by_load.keys())
        if len(load_buckets) >= 2:
            low_load_delta = np.mean(by_load[load_buckets[0]])
            high_load_delta = np.mean(by_load[load_buckets[-1]])
            results["coupling_vs_load"] = (
                f"Low load ({load_buckets[0]} ch): "
                f"avg Δ={low_load_delta:.3f} dB, "
                f"High load ({load_buckets[-1]} ch): "
                f"avg Δ={high_load_delta:.3f} dB",
                "Good — coupling scales with load"
                if high_load_delta > low_load_delta
                else "Unexpected — coupling doesn't increase with load"
            )

    # ── 2.8 Multi-lag temporal autocorrelation ──
    trajs = data["gsnr_trajectories"]
    if len(trajs) > 10:
        lags_to_test = [1, 5, 10]
        lag_results = {}

        for lag in lags_to_test:
            autocorrs = []
            for traj in trajs:
                if len(traj) >= lag + 5:
                    arr = np.array(traj)
                    arr = arr - arr.mean()
                    norm = np.dot(arr, arr)
                    if norm > 1e-8:
                        ac = np.dot(arr[:-lag], arr[lag:]) / norm
                        autocorrs.append(float(ac))

            if autocorrs:
                lag_results[lag] = {
                    "mean": float(np.mean(autocorrs)),
                    "std": float(np.std(autocorrs)),
                    "n": len(autocorrs),
                }

        if lag_results:
            detail_parts = []
            for lag in sorted(lag_results.keys()):
                r = lag_results[lag]
                detail_parts.append(f"lag-{lag}: {r['mean']:.3f}±{r['std']:.3f}")

            ac1 = lag_results.get(1, {}).get("mean", 0)
            ac5 = lag_results.get(5, {}).get("mean", 0)
            ac10 = lag_results.get(10, {}).get("mean", 0)

            results["temporal_autocorrelation"] = (
                " | ".join(detail_parts),
                _interpret_autocorrelation(ac1, ac5, ac10)
            )

    return results


def _interpret_autocorrelation(ac1: float, ac5: float, ac10: float) -> str:
    """Interpret multi-lag autocorrelation for world model planning."""
    parts = []

    if ac1 > 0.95:
        parts.append("lag-1 very high — trajectories may be too smooth/flat")
    elif ac1 < 0.3:
        parts.append("lag-1 low — trajectories may be too noisy")
    else:
        parts.append("lag-1 OK")

    if ac5 > 0.5:
        parts.append("lag-5 indicates 5-step planning is feasible")
    else:
        parts.append("lag-5 low — planning horizon >5 steps will be difficult")

    if ac10 > 0.3:
        parts.append("lag-10 suggests long-horizon rollouts are viable")
    elif ac10 > 0:
        parts.append("lag-10 weak — long-horizon rollouts will accumulate error")
    else:
        parts.append("lag-10 near zero — model is predictive only short-term")

    return "; ".join(parts)


# =====================================================================
# Level 3 — Paper Figures
# =====================================================================

# IEEE single-column: 3.5in, double-column: 7.0in
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.0, 3.5)
FIG_DOUBLE_TALL = (7.0, 4.5)
FIG_DPI_PDF = 300   # irrelevant for vector PDF but sets raster fallback
FIG_DPI_DEBUG = 72


def generate_paper_figures(data: dict, output_dir: str):
    """Generate publication-ready figures."""
    if not HAS_MPL:
        print("  matplotlib not available — skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": FIG_DPI_PDF,
    })

    # Color palette
    colors = {
        "ADD": "#059669", "REMOVE": "#DC2626", "REROUTE": "#065A82",
        "POWER_ADJUST": "#D97706", "MOD_CHANGE": "#7C3AED",
    }
    policy_colors = {
        "provisioning": "#059669", "margin_optimization": "#D97706",
        "load_balancing": "#065A82", "recovery": "#DC2626",
        "mixed_ops": "#7C3AED",
    }

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(f"{output_dir}/{name}.pdf", bbox_inches="tight")
        fig.savefig(f"{output_dir}/{name}.png", dpi=FIG_DPI_DEBUG,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {name}")

    # ── Fig 1: GSNR Distribution ──
    gsnr = np.array(data["gsnr_values"])
    if len(gsnr) > 0:
        fig, ax = plt.subplots(figsize=FIG_DOUBLE)
        ax.hist(gsnr, bins=80, color="#065A82", alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.set_xlabel("GSNR (dB)")
        ax.set_ylabel("Count")
        ax.set_title("GSNR Distribution Across All Transitions")
        ax.axvline(gsnr.mean(), color="#DC2626", linestyle="--",
                   alpha=0.7, label=f"Mean: {gsnr.mean():.1f} dB")
        ax.legend()
        _save(fig, "fig1_gsnr_distribution")

    # ── Fig 2: Action Type Distribution ──
    action_counts = Counter(data["action_types"])
    if action_counts:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        names = [ActionType(t).name for t in sorted(action_counts.keys())]
        counts = [action_counts[t] for t in sorted(action_counts.keys())]
        cols = [colors.get(n, "#888888") for n in names]
        bars = ax.bar(names, counts, color=cols, edgecolor="white",
                      linewidth=0.5)
        ax.set_ylabel("Count")
        ax.set_title("Action Type Distribution")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    f"{count:,}", ha="center", va="bottom", fontsize=6)
        ax.tick_params(axis="x", rotation=30)
        _save(fig, "fig2_action_distribution")

    # ── Fig 3: Action Effect on GSNR (box plot) ──
    deltas_gsnr = data["delta_gsnr_per_action"]
    if len(deltas_gsnr) >= 2:
        fig, ax = plt.subplots(figsize=FIG_DOUBLE)
        sorted_types = sorted(deltas_gsnr.keys())
        labels = [ActionType(t).name for t in sorted_types]
        box_data = [np.array(deltas_gsnr[t]) for t in sorted_types]
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, name in zip(bp["boxes"], labels):
            patch.set_facecolor(colors.get(name, "#888888"))
            patch.set_alpha(0.75)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("ΔGSNR (dB)")
        ax.set_title("Effect of Each Action Type on Average GSNR")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, "fig3_action_effect_boxplot")

    # ── Fig 4: Conditional Diversity Heatmap ──
    pairs = data["action_load_pairs"]
    if len(pairs) > 100:
        all_action_types = sorted(set(t for t, _ in pairs))
        all_load_buckets = sorted(set(b for _, b in pairs))
        matrix = np.zeros((len(all_action_types), len(all_load_buckets)))

        for atype, load_bucket in pairs:
            ai = all_action_types.index(atype)
            li = all_load_buckets.index(load_bucket)
            matrix[ai, li] += 1

        # Normalize per column (load bucket)
        col_sums = matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        matrix_norm = matrix / col_sums

        fig, ax = plt.subplots(figsize=FIG_DOUBLE)
        im = ax.imshow(matrix_norm, cmap="YlOrRd", aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(len(all_load_buckets)))
        ax.set_xticklabels([f"{b}%" for b in all_load_buckets])
        ax.set_yticks(range(len(all_action_types)))
        ax.set_yticklabels([ActionType(t).name for t in all_action_types])
        ax.set_xlabel("Load Bucket (spectral utilization)")
        ax.set_ylabel("Action Type")
        ax.set_title("Action Distribution Conditioned on Network Load")
        plt.colorbar(im, ax=ax, label="P(action | load)")
        _save(fig, "fig4_conditional_diversity")

    # ── Fig 5: Average GSNR Trajectory by Policy ──
    trajs = data["gsnr_trajectories"]
    policies = data["policies"]
    if len(trajs) > 0 and len(policies) == len(trajs):
        fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_TALL)

        # 5a: GSNR trajectories
        _plot_trajectory_by_policy(
            axes[0], trajs, policies, policy_colors,
            ylabel="Avg GSNR (dB)",
            title="GSNR Trajectory by Policy"
        )

        # 5b: Margin trajectories
        margin_trajs = data["margin_trajectories"]
        if len(margin_trajs) == len(policies):
            _plot_trajectory_by_policy(
                axes[1], margin_trajs, policies, policy_colors,
                ylabel="Avg Margin (dB)",
                title="Margin Trajectory by Policy"
            )

        _save(fig, "fig5_trajectories_by_policy")

    # ── Fig 6: Coupling Evidence ──
    coupling = data["coupling_results"]
    if len(coupling) >= 20:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        n_existing = [c[0] for c in coupling]
        max_delta = [c[2] for c in coupling]
        ax.scatter(n_existing, max_delta, alpha=0.3, s=8, color="#065A82")
        ax.set_xlabel("Number of Existing Channels")
        ax.set_ylabel("Max GSNR Change (dB)")
        ax.set_title("Inter-Channel NLI Coupling")

        # Trend line
        if len(n_existing) > 20:
            z = np.polyfit(n_existing, max_delta, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(min(n_existing), max(n_existing), 100)
            ax.plot(x_fit, p(x_fit), "--", color="#DC2626", alpha=0.7,
                    label=f"Trend: {z[0]:.4f} dB/channel")
            ax.legend()

        _save(fig, "fig6_coupling_evidence")

    # ── Fig 7: Autocorrelation decay ──
    trajs = data["gsnr_trajectories"]
    if len(trajs) > 10:
        max_lag = 15
        mean_ac = []
        std_ac = []
        valid_lags = []

        for lag in range(1, max_lag + 1):
            acs = []
            for traj in trajs:
                if len(traj) >= lag + 5:
                    arr = np.array(traj)
                    arr = arr - arr.mean()
                    norm = np.dot(arr, arr)
                    if norm > 1e-8:
                        ac = np.dot(arr[:-lag], arr[lag:]) / norm
                        acs.append(ac)
            if len(acs) >= 5:
                valid_lags.append(lag)
                mean_ac.append(np.mean(acs))
                std_ac.append(np.std(acs))

        if valid_lags:
            fig, ax = plt.subplots(figsize=FIG_SINGLE)
            mean_ac = np.array(mean_ac)
            std_ac = np.array(std_ac)
            ax.plot(valid_lags, mean_ac, "o-", color="#065A82",
                    markersize=3, linewidth=1.5)
            ax.fill_between(valid_lags, mean_ac - std_ac, mean_ac + std_ac,
                            alpha=0.15, color="#065A82")
            ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5,
                       label="r=0.5 (predictability threshold)")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
            ax.set_xlabel("Lag (steps)")
            ax.set_ylabel("Autocorrelation")
            ax.set_title("GSNR Autocorrelation Decay")
            ax.legend()
            ax.set_ylim(-0.1, 1.05)
            _save(fig, "fig7_autocorrelation_decay")

    # ── Fig 8: Load distribution histogram ──
    loads = np.array(data["load_per_step"])
    if len(loads) > 100:
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        ax.hist(loads, bins=20, color="#065A82", alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Spectral Utilization")
        ax.set_ylabel("Count (steps)")
        ax.set_title("Load Distribution Across All Steps")
        _save(fig, "fig8_load_distribution")

    # ── Save raw data for custom plotting ──
    coupling_rate = 0
    if coupling:
        coupling_rate = (
            sum(1 for _, nc, _ in coupling if nc > 0) /
            max(1, len(coupling))
        )

    stats_export = {
        "dataset_summary": {
            "n_episodes": len(data["policies"]),
            "n_transitions": len(data["action_types"]),
            "n_gsnr_samples": len(data["gsnr_values"]),
            "n_margin_samples": len(data["margin_values"]),
        },
        "gsnr": {
            "mean": float(np.mean(gsnr)) if len(gsnr) > 0 else 0,
            "std": float(np.std(gsnr)) if len(gsnr) > 0 else 0,
            "min": float(np.min(gsnr)) if len(gsnr) > 0 else 0,
            "max": float(np.max(gsnr)) if len(gsnr) > 0 else 0,
        },
        "margin": {
            "mean": float(np.mean(data["margin_values"]))
                    if data["margin_values"] else 0,
            "std": float(np.std(data["margin_values"]))
                   if data["margin_values"] else 0,
        },
        "action_distribution": {
            ActionType(t).name: c
            for t, c in Counter(data["action_types"]).items()
        },
        "delta_gsnr_per_action": {
            ActionType(t).name: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "n": len(v),
            }
            for t, v in data["delta_gsnr_per_action"].items() if len(v) > 0
        },
        "delta_margin_per_action": {
            ActionType(t).name: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "n": len(v),
            }
            for t, v in data["delta_margin_per_action"].items() if len(v) > 0
        },
        "policy_distribution": dict(Counter(data["policies"])),
        "coupling": {
            "rate": coupling_rate,
            "n_samples": len(coupling),
            "avg_max_delta_db": float(
                np.mean([md for _, _, md in coupling if md > 0])
            ) if any(md > 0 for _, _, md in coupling) else 0,
            "trend_slope_db_per_ch": _compute_coupling_slope(coupling),
        },
        "episode_stats": {
            "avg_length": float(np.mean(data["n_steps_per_ep"]))
                         if data["n_steps_per_ep"] else 0,
            "non_monotonic_rate": float(
                1 - np.mean(data["episode_monotonic"])
            ) if data["episode_monotonic"] else 0,
            "avg_gsnr_range_db": float(np.mean(data["episode_gsnr_ranges"]))
                                if data["episode_gsnr_ranges"] else 0,
        },
    }

    with open(f"{output_dir}/dataset_stats.json", "w") as f:
        json.dump(stats_export, f, indent=2)
    print(f"  Saved: dataset_stats.json")


def _plot_trajectory_by_policy(ax, trajs, policies, policy_colors,
                               ylabel="", title=""):
    """Plot mean±std trajectory grouped by policy."""
    by_policy = defaultdict(list)
    for traj, pol in zip(trajs, policies):
        by_policy[pol].append(traj)

    for pol, pol_trajs in sorted(by_policy.items()):
        n_interp = 50
        interped = []
        for t in pol_trajs:
            if len(t) >= 3:
                x_old = np.linspace(0, 1, len(t))
                x_new = np.linspace(0, 1, n_interp)
                interped.append(np.interp(x_new, x_old, t))

        if len(interped) >= 3:
            arr = np.array(interped)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.arange(n_interp)
            col = policy_colors.get(pol, "#888888")
            ax.plot(x, mean, label=pol, color=col, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std,
                            alpha=0.15, color=col)

    ax.set_xlabel("Episode Progress (normalized)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=6, loc="best")


def _compute_coupling_slope(coupling: list) -> float:
    """Linear trend slope of max_delta vs n_existing channels."""
    if len(coupling) < 10:
        return 0.0
    n_existing = np.array([c[0] for c in coupling], dtype=float)
    max_delta = np.array([c[2] for c in coupling], dtype=float)
    if n_existing.std() < 1e-8:
        return 0.0
    z = np.polyfit(n_existing, max_delta, 1)
    return float(z[0])


# =====================================================================
# Report Printer
# =====================================================================

def print_report(integrity: dict, dynamics: dict, data: dict):
    """Print formatted evaluation report."""
    n_eps = len(data["policies"])
    n_trans = len(data["action_types"])

    print(f"\n{'=' * 65}")
    print(f"  DATASET EVALUATION REPORT")
    print(f"{'=' * 65}")
    print(f"  Episodes: {n_eps:,}")
    print(f"  Transitions: {n_trans:,}")
    print(f"  GSNR samples: {len(data['gsnr_values']):,}")
    print(f"  Margin samples: {len(data['margin_values']):,}")
    print()

    # Level 1
    print(f"  ── Level 1: Integrity Checks ──")
    all_passed = True
    for name, (passed, msg) in integrity.items():
        icon = "✓" if passed else "✗"
        print(f"    {icon}  {name}: {msg}")
        if not passed:
            all_passed = False
    print()
    if all_passed:
        print(f"    ✅ All integrity checks passed")
    else:
        print(f"    ⚠  Some integrity checks failed — review before training")
    print()

    # Level 2
    print(f"  ── Level 2: Dynamics Analysis ──")
    for name, (value, interp) in dynamics.items():
        if "\n" in str(value):
            print(f"    {name}:")
            for line in str(value).split("\n"):
                print(f"      {line}")
            print(f"      → {interp}")
        else:
            print(f"    {name}: {value}")
            print(f"      → {interp}")
    print()

    # Summary verdict
    print(f"  ── Verdict ──")
    if not all_passed:
        print(f"    ⚠  Dataset has integrity issues — fix before training")
    elif len(data["coupling_results"]) < 10:
        print(f"    ⚠  Too few transitions to evaluate coupling "
              f"— generate more data")
    else:
        coupling_rate = (
            sum(1 for _, nc, _ in data["coupling_results"] if nc > 0) /
            max(1, len(data["coupling_results"]))
        )
        non_mono = (
            1 - np.mean(data["episode_monotonic"])
            if data["episode_monotonic"] else 0
        )

        # Check discriminability if available
        discrim_ok = True
        if "action_discriminability" in dynamics:
            val_str = dynamics["action_discriminability"][0]
            if "p=" in val_str:
                # Extract p-value
                try:
                    p_str = val_str.split("p=")[1].split(" ")[0]
                    p_val = float(p_str)
                    discrim_ok = p_val < 0.05
                except (ValueError, IndexError):
                    pass

        if coupling_rate > 0.5 and non_mono > 0.15 and discrim_ok:
            print(f"    ✅ Dataset is ready for JEPA training")
            print(f"       Coupling: {coupling_rate:.0%} | "
                  f"Non-monotonic: {non_mono:.0%} | "
                  f"Actions discriminable: {'Yes' if discrim_ok else 'No'}")
        else:
            print(f"    ⚠  Dataset may not be ready for JEPA training")
            reasons = []
            if coupling_rate <= 0.5:
                reasons.append(f"Coupling: {coupling_rate:.0%} (want >50%)")
            if non_mono <= 0.15:
                reasons.append(
                    f"Non-monotonic: {non_mono:.0%} (want >15%)"
                )
            if not discrim_ok:
                reasons.append(
                    "Actions are NOT statistically discriminable"
                )
            for r in reasons:
                print(f"       {r}")

    print(f"\n{'=' * 65}\n")


# =====================================================================
# Main
# =====================================================================

def run_evaluation(data_dir: str, report: bool = True,
                   figures: bool = False, output_dir: str = "figures",
                   max_episodes: int = None):
    """Run full evaluation pipeline."""

    h5_files = find_h5_files(data_dir)
    if not h5_files:
        print(f"No HDF5 files found in {data_dir}/")
        return

    print(f"Found {len(h5_files)} HDF5 files:")
    for f in h5_files:
        print(f"  {f}")
    print()

    # Load data
    print("Loading data...")
    datasets = []
    for h5_path in h5_files:
        topo = Path(h5_path).stem
        print(f"  Loading {topo}...")
        d = load_all_transitions(h5_path, max_episodes=max_episodes)
        print(f"    {len(d['policies'])} episodes, "
              f"{len(d['action_types'])} transitions")
        datasets.append(d)

    data = merge_data(datasets)
    print(f"\nTotal: {len(data['policies'])} episodes, "
          f"{len(data['action_types'])} transitions\n")

    # Evaluate
    integrity = evaluate_integrity(data)
    dynamics = evaluate_dynamics(data)

    if report:
        print_report(integrity, dynamics, data)

    if figures:
        print("Generating figures...")
        generate_paper_figures(data, output_dir)
        print()

    return {
        "integrity": integrity,
        "dynamics": dynamics,
        "data_summary": {
            "n_episodes": len(data["policies"]),
            "n_transitions": len(data["action_types"]),
            "n_gsnr_samples": len(data["gsnr_values"]),
            "n_margin_samples": len(data["margin_values"]),
        },
    }


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate optical network world model dataset"
    )
    parser.add_argument("--data", type=str, default="data",
                        help="Directory containing HDF5 files")
    parser.add_argument("--report", action="store_true",
                        help="Print evaluation report")
    parser.add_argument("--figures", action="store_true",
                        help="Generate paper figures")
    parser.add_argument("--output", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--all", action="store_true",
                        help="Run everything (report + figures)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit episodes per file (for quick checks)")

    args = parser.parse_args()

    if args.all:
        args.report = True
        args.figures = True

    if not args.report and not args.figures:
        args.report = True  # default to report

    run_evaluation(
        data_dir=args.data,
        report=args.report,
        figures=args.figures,
        output_dir=args.output,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()