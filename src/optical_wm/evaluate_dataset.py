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

# ─── Project imports ───
from .core.schemas import (
    ActionType, Modulation, MOD_THRESHOLDS,
    MAX_SLOTS, MAX_LINKS, MAX_LIGHTPATHS,
    N_ACTION_FEATURES,
)
from .core.hdf5_io import HDF5Reader


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
        "gsnr_values": [],          # all active channel GSNRs
        "margin_values": [],        # all per-LP margins
        "load_per_step": [],        # spectral utilization per step
        "n_lps_per_step": [],       # LP count per step
        "action_types": [],         # action type per transition
        "action_load_pairs": [],    # (action_type, load_bucket) pairs
        "delta_gsnr_per_action": defaultdict(list),  # action_type → [delta_gsnr]
        "gsnr_trajectories": [],    # per-episode GSNR sequences
        "coupling_results": [],     # (n_existing, n_changed, max_delta) per ADD
        "policies": [],             # policy name per episode
        "n_steps_per_ep": [],       # episode lengths
        "infeasible_counts": [],    # n_infeasible per step
        "episode_gsnr_ranges": [],  # max-min GSNR per episode
        "episode_monotonic": [],    # True if GSNR is monotonic
        "episode_has_rise": [],     # GSNR rises at some point
        "episode_has_fall": [],     # GSNR falls at some point
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

            # ── Per-step metrics ──
            gsnr_seq = []
            for t in range(limit):
                # Global features
                gf = sub["global_features"][t]
                n_active = gf[0]
                util = gf[6]
                n_infeasible = gf[3]
                avg_margin = gf[5]

                data["load_per_step"].append(float(util))
                data["n_lps_per_step"].append(float(n_active))
                data["infeasible_counts"].append(float(n_infeasible))
                gsnr_seq.append(float(avg_margin))

                # Active channel GSNRs
                ch_gsnr = sub["channel_gsnr"][t]
                occ = sub["spectral_occupancy"][t]
                active_gsnr = ch_gsnr[occ]
                if len(active_gsnr) > 0:
                    data["gsnr_values"].extend(active_gsnr.tolist())

                # Per-LP margins
                lp_mask = sub["lp_mask"][t]
                lp_feat = sub["lp_features"][t]
                for i in range(int(lp_mask.sum())):
                    if lp_mask[i]:
                        data["margin_values"].append(float(lp_feat[i, 18]))

            # ── Per-transition metrics ──
            if "actions" in sub and limit > 1:
                n_actions = min(len(sub["actions"]), limit - 1)
                for t in range(n_actions):
                    action_type = int(sub["actions"][t][0])
                    data["action_types"].append(action_type)

                    # Load bucket (by 10%)
                    load = sub["global_features"][t][6]
                    load_bucket = int(load * 10) * 10  # 0, 10, 20, ...
                    data["action_load_pairs"].append((action_type, load_bucket))

                    # Delta GSNR (avg margin change)
                    gsnr_before = sub["global_features"][t][5]
                    gsnr_after = sub["global_features"][t + 1][5]
                    delta = float(gsnr_after - gsnr_before)
                    data["delta_gsnr_per_action"][action_type].append(delta)

                    # Inter-channel coupling for ADD actions
                    if action_type == ActionType.ADD:
                        occ_t = sub["spectral_occupancy"][t]
                        occ_t1 = sub["spectral_occupancy"][t + 1]
                        gsnr_t = sub["channel_gsnr"][t]
                        gsnr_t1 = sub["channel_gsnr"][t + 1]

                        # Channels present in both steps
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

            if len(gsnr_seq) >= 3:
                gsnr_range = max(gsnr_seq) - min(gsnr_seq)
                data["episode_gsnr_ranges"].append(gsnr_range)

                diffs = [gsnr_seq[i+1] - gsnr_seq[i]
                         for i in range(len(gsnr_seq)-1)]
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
        "gsnr_trajectories": [],
        "coupling_results": [],
        "policies": [],
        "n_steps_per_ep": [],
        "infeasible_counts": [],
        "episode_gsnr_ranges": [],
        "episode_monotonic": [],
        "episode_has_rise": [],
        "episode_has_fall": [],
    }

    for d in datasets:
        for key in merged:
            if key == "delta_gsnr_per_action":
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
        avg_delta = np.mean([md for _, _, md in coupling if md > 0]) if any(md > 0 for _, _, md in coupling) else 0
        passed = coupling_rate > 0.5
        results["coupling_present"] = (
            passed,
            f"{coupling_rate:.0%} of ADD transitions show coupling "
            f"(avg {avg_changed:.1f} channels affected, "
            f"avg Δ={avg_delta:.3f} dB)"
        )
    else:
        results["coupling_present"] = (False, f"Only {len(coupling)} ADD transitions sampled")

    # ── 1.6 No excessive infeasible ──
    infeas = np.array(data["infeasible_counts"])
    n_lps = np.array(data["n_lps_per_step"])
    if len(infeas) > 0 and len(n_lps) > 0:
        mask = n_lps > 0
        if mask.any():
            ratios = infeas[mask] / n_lps[mask]
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
            "Good" if ranges.mean() > 0.5 else "Low — trajectories may be too flat"
        )

    # ── 2.2 Non-monotonicity ──
    mono = np.array(data["episode_monotonic"])
    rises = np.array(data["episode_has_rise"])
    falls = np.array(data["episode_has_fall"])
    if len(mono) > 0:
        non_mono_rate = 1 - mono.mean()
        rise_rate = rises.mean()
        fall_rate = falls.mean()
        results["non_monotonic_rate"] = (
            f"{non_mono_rate:.0%} of episodes are non-monotonic",
            "Good" if non_mono_rate > 0.2 else "Low — need more recovery/mixed_ops"
        )
        results["rise_fall_rates"] = (
            f"Episodes with GSNR rise: {rise_rate:.0%}, "
            f"with GSNR fall: {fall_rate:.0%}",
            "Good" if rise_rate > 0.3 and fall_rate > 0.3 else
            "Imbalanced — one direction dominates"
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

        # Check if action types produce distinguishable effects
        means = [s["mean"] for s in delta_stats.values()]
        mean_spread = max(means) - min(means)
        results["action_delta_gsnr"] = (
            json.dumps(
                {k: f"μ={v['mean']:+.3f} σ={v['std']:.3f} (n={v['n']})"
                 for k, v in delta_stats.items()},
                indent=2,
            ),
            "Good" if mean_spread > 0.1 else
            "Weak — actions produce similar GSNR effects"
        )
        results["action_effect_spread"] = (
            f"{mean_spread:.3f} dB spread between action types",
            "Good" if mean_spread > 0.1 else "Low"
        )

    # ── 2.4 Conditional action diversity ──
    pairs = data["action_load_pairs"]
    if len(pairs) > 0:
        buckets = defaultdict(set)
        for atype, load_bucket in pairs:
            buckets[load_bucket].add(atype)

        diverse_buckets = sum(1 for v in buckets.values() if len(v) >= 2)
        total_buckets = len(buckets)
        max_diversity = max(len(v) for v in buckets.values()) if buckets else 0

        bucket_detail = {
            f"{b}%": [ActionType(t).name for t in sorted(types)]
            for b, types in sorted(buckets.items())
        }
        results["conditional_diversity"] = (
            f"{diverse_buckets}/{total_buckets} load buckets have ≥2 action types "
            f"(max {max_diversity} types in one bucket)",
            "Good" if diverse_buckets >= total_buckets * 0.5 else
            "Low — state→action correlation too strong"
        )

    # ── 2.5 Coupling strength ──
    coupling = data["coupling_results"]
    if len(coupling) >= 10:
        # Group by number of existing channels
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
                f"Low load ({load_buckets[0]} ch): avg Δ={low_load_delta:.3f} dB, "
                f"High load ({load_buckets[-1]} ch): avg Δ={high_load_delta:.3f} dB",
                "Good — coupling scales with load" if high_load_delta > low_load_delta
                else "Unexpected — coupling doesn't increase with load"
            )

    # ── 2.6 Autocorrelation of GSNR ──
    trajs = data["gsnr_trajectories"]
    if len(trajs) > 10:
        autocorrs = []
        for traj in trajs:
            if len(traj) >= 10:
                arr = np.array(traj)
                arr = arr - arr.mean()
                norm = np.dot(arr, arr)
                if norm > 1e-8:
                    # Lag-1 autocorrelation
                    ac1 = np.dot(arr[:-1], arr[1:]) / norm
                    autocorrs.append(float(ac1))

        if autocorrs:
            avg_ac = np.mean(autocorrs)
            results["temporal_autocorrelation"] = (
                f"Avg lag-1 autocorrelation: {avg_ac:.3f}",
                "Good — smooth but not constant" if 0.3 < avg_ac < 0.95 else
                "Check — may be too flat or too noisy"
            )

    return results


# =====================================================================
# Level 3 — Paper Figures
# =====================================================================

def generate_paper_figures(data: dict, output_dir: str):
    """Generate publication-ready figures."""
    if not HAS_MPL:
        print("  matplotlib not available — skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
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

    # ── Fig 1: GSNR Distribution ──
    gsnr = np.array(data["gsnr_values"])
    if len(gsnr) > 0:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(gsnr, bins=80, color="#065A82", alpha=0.85, edgecolor="white",
                linewidth=0.3)
        ax.set_xlabel("GSNR (dB)")
        ax.set_ylabel("Count")
        ax.set_title("GSNR Distribution Across All Transitions")
        ax.axvline(gsnr.mean(), color="#DC2626", linestyle="--", alpha=0.7,
                   label=f"Mean: {gsnr.mean():.1f} dB")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{output_dir}/fig1_gsnr_distribution.png")
        fig.savefig(f"{output_dir}/fig1_gsnr_distribution.pdf")
        plt.close(fig)
        print(f"  Saved: fig1_gsnr_distribution")

    # ── Fig 2: Action Type Distribution ──
    action_counts = Counter(data["action_types"])
    if action_counts:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        names = [ActionType(t).name for t in sorted(action_counts.keys())]
        counts = [action_counts[t] for t in sorted(action_counts.keys())]
        cols = [colors.get(n, "#888888") for n in names]
        bars = ax.bar(names, counts, color=cols, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Count")
        ax.set_title("Action Type Distribution")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f"{count:,}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/fig2_action_distribution.png")
        fig.savefig(f"{output_dir}/fig2_action_distribution.pdf")
        plt.close(fig)
        print(f"  Saved: fig2_action_distribution")

    # ── Fig 3: Action Effect on GSNR (box plot) ──
    deltas = data["delta_gsnr_per_action"]
    if len(deltas) >= 2:
        fig, ax = plt.subplots(figsize=(7, 4))
        sorted_types = sorted(deltas.keys())
        labels = [ActionType(t).name for t in sorted_types]
        box_data = [np.array(deltas[t]) for t in sorted_types]
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, name in zip(bp["boxes"], labels):
            patch.set_facecolor(colors.get(name, "#888888"))
            patch.set_alpha(0.75)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("ΔGSNR (dB)")
        ax.set_title("Effect of Each Action Type on Average GSNR")
        fig.tight_layout()
        fig.savefig(f"{output_dir}/fig3_action_effect_boxplot.png")
        fig.savefig(f"{output_dir}/fig3_action_effect_boxplot.pdf")
        plt.close(fig)
        print(f"  Saved: fig3_action_effect_boxplot")

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

        fig, ax = plt.subplots(figsize=(8, 3.5))
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
        fig.tight_layout()
        fig.savefig(f"{output_dir}/fig4_conditional_diversity.png")
        fig.savefig(f"{output_dir}/fig4_conditional_diversity.pdf")
        plt.close(fig)
        print(f"  Saved: fig4_conditional_diversity")

    # ── Fig 5: Average GSNR Trajectory by Policy ──
    trajs = data["gsnr_trajectories"]
    policies = data["policies"]
    if len(trajs) > 0 and len(policies) == len(trajs):
        fig, ax = plt.subplots(figsize=(8, 4))
        by_policy = defaultdict(list)
        for traj, pol in zip(trajs, policies):
            by_policy[pol].append(traj)

        for pol, pol_trajs in sorted(by_policy.items()):
            # Normalize to same length (interpolate to 50 steps)
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
                ax.plot(x, mean, label=pol, color=col, linewidth=2)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=col)

        ax.set_xlabel("Episode Progress (normalized)")
        ax.set_ylabel("Avg Margin (dB)")
        ax.set_title("GSNR Trajectory by Policy (mean ± std)")
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(f"{output_dir}/fig5_gsnr_by_policy.png")
        fig.savefig(f"{output_dir}/fig5_gsnr_by_policy.pdf")
        plt.close(fig)
        print(f"  Saved: fig5_gsnr_by_policy")

    # ── Fig 6: Coupling Evidence ──
    coupling = data["coupling_results"]
    if len(coupling) >= 20:
        fig, ax = plt.subplots(figsize=(6, 4))
        n_existing = [c[0] for c in coupling]
        max_delta = [c[2] for c in coupling]
        ax.scatter(n_existing, max_delta, alpha=0.3, s=10, color="#065A82")
        ax.set_xlabel("Number of Existing Channels")
        ax.set_ylabel("Max GSNR Change (dB)")
        ax.set_title("Inter-Channel Coupling: Impact of ADD on Existing Channels")

        # Trend line
        if len(n_existing) > 20:
            z = np.polyfit(n_existing, max_delta, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(min(n_existing), max(n_existing), 100)
            ax.plot(x_fit, p(x_fit), "--", color="#DC2626", alpha=0.7,
                    label=f"Trend: {z[0]:.4f} dB/channel")
            ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(f"{output_dir}/fig6_coupling_evidence.png")
        fig.savefig(f"{output_dir}/fig6_coupling_evidence.pdf")
        plt.close(fig)
        print(f"  Saved: fig6_coupling_evidence")

    # ── Save raw data for custom plotting ──
    stats_export = {
        "gsnr": {
            "mean": float(np.mean(data["gsnr_values"])) if data["gsnr_values"] else 0,
            "std": float(np.std(data["gsnr_values"])) if data["gsnr_values"] else 0,
            "min": float(np.min(data["gsnr_values"])) if data["gsnr_values"] else 0,
            "max": float(np.max(data["gsnr_values"])) if data["gsnr_values"] else 0,
        },
        "action_distribution": {
            ActionType(t).name: c for t, c in Counter(data["action_types"]).items()
        },
        "delta_gsnr_per_action": {
            ActionType(t).name: {
                "mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)
            }
            for t, v in data["delta_gsnr_per_action"].items() if len(v) > 0
        },
        "policy_distribution": dict(Counter(data["policies"])),
        "n_episodes": len(data["policies"]),
        "n_transitions": len(data["action_types"]),
        "coupling_rate": (
            sum(1 for _, nc, _ in coupling if nc > 0) / max(1, len(coupling))
            if coupling else 0
        ),
    }

    with open(f"{output_dir}/dataset_stats.json", "w") as f:
        json.dump(stats_export, f, indent=2)
    print(f"  Saved: dataset_stats.json")


# =====================================================================
# Report Printer
# =====================================================================

def print_report(integrity: dict, dynamics: dict, data: dict):
    """Print formatted evaluation report."""
    n_eps = len(data["policies"])
    n_trans = len(data["action_types"])

    print(f"\n{'='*65}")
    print(f"  DATASET EVALUATION REPORT")
    print(f"{'='*65}")
    print(f"  Episodes: {n_eps:,}")
    print(f"  Transitions: {n_trans:,}")
    print(f"  GSNR samples: {len(data['gsnr_values']):,}")
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
        print(f"    ⚠  Too few transitions to evaluate coupling — generate more data")
    else:
        coupling_rate = sum(1 for _, nc, _ in data["coupling_results"] if nc > 0) / max(1, len(data["coupling_results"]))
        non_mono = 1 - np.mean(data["episode_monotonic"]) if data["episode_monotonic"] else 0

        if coupling_rate > 0.5 and non_mono > 0.15:
            print(f"    ✅ Dataset is ready for JEPA training")
            print(f"       Coupling: {coupling_rate:.0%} | Non-monotonic: {non_mono:.0%}")
        else:
            print(f"    ⚠  Dataset may be too simple for JEPA")
            print(f"       Coupling: {coupling_rate:.0%} (want >50%)")
            print(f"       Non-monotonic: {non_mono:.0%} (want >15%)")

    print(f"\n{'='*65}\n")


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