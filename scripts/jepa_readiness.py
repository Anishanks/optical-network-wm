"""
JEPA-specific dataset readiness diagnostics.

Metrics (computed on a 24-d normalized per-step summary vector):
  1. Rank diagnostics   - rank(state) vs rank(delta state)
  2. Baselines          - zero / identity / linear-AR (1-step MSE)
  3. AR divergence      - multi-step rollout of linear AR (k=1,2,5,10)
  4. Action-cond SNR    - F-ratio per feature
  5. Train/test shift   - KS statistic magnitude (not p-value)
  6. Window diversity   - unique action sequences AND windows with multiple
                          action types, at context_length=8

Single dataset:
  PYTHONPATH=src py scripts/jepa_readiness.py --data data_500
Scaling comparison:
  PYTHONPATH=src py scripts/jepa_readiness.py --data data_500 data_2k data_full
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as sst
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from optical_wm.core.hdf5_io import HDF5Reader
from optical_wm.core.schemas import ActionType


TRAIN_TOPOS = {"topo_A_small_mesh", "topo_B_medium_sparse", "topo_C_medium_mesh"}
TEST_TOPOS = {"topo_D_test"}

FEATURE_NAMES = [
    *[f"gf_{i}" for i in range(8)],
    "gsnr_p10", "gsnr_p25", "gsnr_p50", "gsnr_p75", "gsnr_p90", "gsnr_mean",
    "pw_p25", "pw_p50", "pw_p75", "pw_mean",
    "nli_mean", "nli_std", "nli_max",
    "margin_mean", "margin_std", "margin_min",
]
D = len(FEATURE_NAMES)


# =====================================================================
# Feature construction
# =====================================================================

def aggregated_norm(norm_json):
    agg = {}
    if "channel_gsnr" in norm_json:
        agg["channel_gsnr"] = norm_json["channel_gsnr"]
    per = norm_json.get("per_topology", {})
    if per:
        cp_m = [v["channel_power"]["mean"] for v in per.values()]
        cp_s = [v["channel_power"]["std"] for v in per.values()]
        agg["channel_power"] = {"mean": float(np.mean(cp_m)),
                                "std": float(np.mean(cp_s))}
        gf_m = np.array([v["global_features"]["mean"] for v in per.values()])
        gf_s = np.array([v["global_features"]["std"] for v in per.values()])
        agg["global_features"] = {"mean": gf_m.mean(axis=0).tolist(),
                                  "std": gf_s.mean(axis=0).tolist()}
        if "channel_gsnr" not in agg:
            g_m = [v["channel_gsnr"]["mean"] for v in per.values()]
            g_s = [v["channel_gsnr"]["std"] for v in per.values()]
            agg["channel_gsnr"] = {"mean": float(np.mean(g_m)),
                                   "std": float(np.mean(g_s))}
    return agg


def step_summary(s, norm):
    gf_m = np.array(norm["global_features"]["mean"])
    gf_s = np.maximum(np.array(norm["global_features"]["std"]), 1e-6)
    gf_z = (np.asarray(s["global_features"], dtype=np.float64) - gf_m) / gf_s

    gm = norm["channel_gsnr"]["mean"]; gs = max(norm["channel_gsnr"]["std"], 1e-6)
    pm = norm["channel_power"]["mean"]; ps = max(norm["channel_power"]["std"], 1e-6)

    occ = s["spectral_occupancy"]
    if occ.any():
        ag = (s["channel_gsnr"][occ] - gm) / gs
        ap = (s["channel_power"][occ] - pm) / ps
        an = s["channel_nli"][occ]
    else:
        ag = np.zeros(1); ap = np.zeros(1); an = np.zeros(1)

    gp = np.percentile(ag, [10, 25, 50, 75, 90])
    pp = np.percentile(ap, [25, 50, 75])

    lp_mask = s["lp_mask"]
    if lp_mask.any():
        margins = s["lp_features"][lp_mask, 18]
        m_stats = [float(margins.mean()), float(margins.std()),
                   float(margins.min())]
    else:
        m_stats = [0.0, 0.0, 0.0]

    return np.concatenate([
        gf_z, gp, [ag.mean()], pp, [ap.mean()],
        [float(an.mean()), float(an.std()), float(an.max())],
        m_stats,
    ])


def build_topo_vectors(h5_path, norm, max_eps=None):
    reader = HDF5Reader(h5_path); reader.open()
    ep_list = reader.list_episodes()
    if max_eps:
        ep_list = ep_list[:max_eps]

    ep_vecs, ep_actions = [], []
    for eid in ep_list:
        info = reader.get_episode_info(eid)
        n = info["n_steps"]
        sub = reader.load_subsequence(eid, 0, n)
        V = np.zeros((n, D))
        for t in range(n):
            V[t] = step_summary({
                "channel_gsnr": sub["channel_gsnr"][t],
                "channel_power": sub["channel_power"][t],
                "channel_nli": sub["channel_nli"][t],
                "spectral_occupancy": sub["spectral_occupancy"][t],
                "lp_mask": sub["lp_mask"][t],
                "lp_features": sub["lp_features"][t],
                "global_features": sub["global_features"][t],
            }, norm)
        ep_vecs.append(V)
        ep_actions.append(sub["actions"] if "actions" in sub else np.zeros((0, 20)))
    reader.close()
    return ep_vecs, ep_actions


# =====================================================================
# Metrics
# =====================================================================

def effective_rank(X, threshold=0.95):
    if X.shape[0] == 0:
        return 0
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s ** 2
    if var.sum() == 0:
        return 0
    cum = np.cumsum(var) / var.sum()
    return int((cum < threshold).sum()) + 1


def stack_deltas(ep_vecs):
    ds = []
    for V in ep_vecs:
        if len(V) >= 2:
            ds.append(V[1:] - V[:-1])
    return np.concatenate(ds) if ds else np.zeros((0, D))


def zero_baseline(ep_vecs):
    tot, n = 0.0, 0
    for V in ep_vecs:
        tot += float((V ** 2).sum()); n += V.size
    return tot / max(n, 1)


def identity_baseline(ep_vecs):
    tot, n = 0.0, 0
    for V in ep_vecs:
        if len(V) < 2: continue
        d = V[1:] - V[:-1]
        tot += float((d ** 2).sum()); n += d.size
    return tot / max(n, 1)


def fit_linear_ar(ep_vecs, actions, split_idx=None):
    """Fit closed-form ridge. Returns W and feature layout."""
    Xs, Xa, Y = [], [], []
    ep_ids = []
    for eid, (V, A) in enumerate(zip(ep_vecs, actions)):
        m = min(len(V) - 1, len(A))
        if m <= 0: continue
        Xs.append(V[:m]); Xa.append(A[:m]); Y.append(V[1:m + 1])
        ep_ids.extend([eid] * m)
    if not Xs:
        return None, None
    Xs = np.concatenate(Xs); Xa = np.concatenate(Xa); Y = np.concatenate(Y)
    ep_ids = np.array(ep_ids)
    X = np.concatenate([Xs, Xa, np.ones((len(Xs), 1))], axis=1)

    if split_idx is None:
        rng = np.random.default_rng(42)
        unique_eps = np.unique(ep_ids)
        rng.shuffle(unique_eps)
        n_tr_ep = int(0.8 * len(unique_eps))
        tr_eps = set(unique_eps[:n_tr_ep].tolist())
        tr_mask = np.array([e in tr_eps for e in ep_ids])
        te_mask = ~tr_mask
    else:
        tr_mask, te_mask = split_idx

    lam = 1e-3
    A_mat = X[tr_mask].T @ X[tr_mask] + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A_mat, X[tr_mask].T @ Y[tr_mask])
    pred = X[te_mask] @ W
    mse_1step = float(((pred - Y[te_mask]) ** 2).mean())
    return W, (mse_1step, ep_ids, tr_mask, te_mask)


def linear_ar_rollout(W, ep_vecs, actions, horizons=(1, 2, 5, 10),
                      n_sample=500, seed=42):
    """Roll out linear AR K steps ahead, measure MSE per horizon."""
    rng = np.random.default_rng(seed)
    rollouts = {h: [] for h in horizons}
    max_h = max(horizons)

    # Sample episodes
    valid = [i for i, (V, A) in enumerate(zip(ep_vecs, actions))
             if len(V) > max_h + 1 and len(A) >= max_h]
    if not valid:
        return {h: float("nan") for h in horizons}
    rng.shuffle(valid)
    sampled = valid[:n_sample]

    for i in sampled:
        V, A = ep_vecs[i], actions[i]
        max_start = min(len(V) - max_h - 1, len(A) - max_h)
        if max_start <= 0: continue
        start = rng.integers(0, max_start)
        state = V[start].copy()
        for step in range(1, max_h + 1):
            a = A[start + step - 1]
            x = np.concatenate([state, a, [1.0]])
            state = x @ W
            if step in horizons:
                true_state = V[start + step]
                rollouts[step].append(float(((state - true_state) ** 2).mean()))

    return {h: float(np.mean(v)) if v else float("nan")
            for h, v in rollouts.items()}


def action_snr(ep_vecs, actions):
    deltas = defaultdict(list)
    for V, A in zip(ep_vecs, actions):
        m = min(len(V) - 1, len(A))
        for t in range(m):
            deltas[int(A[t][0])].append(V[t + 1] - V[t])
    if len(deltas) < 2:
        return np.zeros(D), []
    types = sorted(deltas.keys())
    means = np.zeros((len(types), D))
    within = np.zeros(D); total_n = 0
    for i, at in enumerate(types):
        arr = np.array(deltas[at])
        means[i] = arr.mean(axis=0)
        within += arr.var(axis=0) * len(arr); total_n += len(arr)
    within /= max(total_n, 1)
    between = means.var(axis=0)
    snr = between / (within + 1e-10)
    return snr, types


def ks_magnitudes(train_X, test_X):
    if not HAS_SCIPY or test_X.shape[0] == 0:
        return None
    return [sst.ks_2samp(train_X[:, j], test_X[:, j]) for j in range(D)]


def window_diversity(ep_vecs, actions, L=8):
    total, uniq_seqs, multi_action = 0, set(), 0
    for V, A in zip(ep_vecs, actions):
        if len(V) < L or len(A) < L - 1: continue
        types = np.array(A)[:, 0].astype(int)
        for s in range(len(V) - L + 1):
            if s + L - 1 > len(types): break
            seq = tuple(types[s:s + L - 1].tolist())
            uniq_seqs.add(seq); total += 1
            if len(set(seq)) >= 2:
                multi_action += 1
    return {
        "n_windows": total,
        "n_unique": len(uniq_seqs),
        "uniqueness": len(uniq_seqs) / max(total, 1),
        "multi_action_rate": multi_action / max(total, 1),
    }


# =====================================================================
# Analyze one dataset
# =====================================================================

def analyze(data_dir, max_episodes=None):
    data_dir = Path(data_dir)
    with open(data_dir / "normalization.json") as f:
        norm = aggregated_norm(json.load(f))

    train_V, train_A, test_V, test_A = [], [], [], []
    per_topo = {}
    for h5 in sorted(data_dir.glob("*.h5")):
        topo = h5.stem
        V, A = build_topo_vectors(str(h5), norm, max_episodes)
        steps = sum(len(v) for v in V)
        per_topo[topo] = (len(V), steps)
        if topo in TRAIN_TOPOS:
            train_V += V; train_A += A
        elif topo in TEST_TOPOS:
            test_V += V; test_A += A

    train_X = np.concatenate(train_V) if train_V else np.zeros((0, D))
    test_X = np.concatenate(test_V) if test_V else np.zeros((0, D))

    state_rank_95 = effective_rank(train_X, 0.95)
    state_rank_99 = effective_rank(train_X, 0.99)
    delta_X = stack_deltas(train_V)
    delta_rank_95 = effective_rank(delta_X, 0.95)
    delta_rank_99 = effective_rank(delta_X, 0.99)
    var = train_X.var(axis=0)
    dead = np.where(var < 1e-4)[0]

    zero = zero_baseline(train_V)
    ident = identity_baseline(train_V)
    W, fit_info = fit_linear_ar(train_V, train_A)
    ar_1step = fit_info[0] if fit_info else float("nan")
    rollout = linear_ar_rollout(W, train_V, train_A) if W is not None else {}

    snr, types = action_snr(train_V, train_A)

    ks = ks_magnitudes(train_X, test_X)
    if ks:
        ks_stats = np.array([s for s, _ in ks])
        ks_strong = int((ks_stats > 0.2).sum())
        ks_top5 = sorted(enumerate(ks), key=lambda x: -x[1][0])[:5]
    else:
        ks_stats = np.zeros(D); ks_strong = 0; ks_top5 = []

    wd = window_diversity(train_V, train_A, L=8)

    return {
        "data_dir": str(data_dir),
        "per_topo": per_topo,
        "n_train_steps": len(train_X), "n_test_steps": len(test_X),
        "state_rank_95": state_rank_95, "state_rank_99": state_rank_99,
        "delta_rank_95": delta_rank_95, "delta_rank_99": delta_rank_99,
        "dead_features": [FEATURE_NAMES[i] for i in dead],
        "zero_mse": zero, "ident_mse": ident, "ar_1step_mse": ar_1step,
        "rollout_mse": rollout,
        "action_types": [ActionType(t).name for t in types],
        "snr": snr, "snr_mean": float(snr.mean()),
        "snr_strong": int((snr > 0.5).sum()),
        "snr_top5": np.argsort(-snr)[:5],
        "ks_stats": ks_stats, "ks_strong": ks_strong, "ks_top5": ks_top5,
        "window": wd,
    }


# =====================================================================
# Reports
# =====================================================================

def print_single_report(r):
    D_ = D
    print("\n" + "=" * 70)
    print(f"  JEPA READINESS REPORT  -  {r['data_dir']}")
    print("=" * 70)
    for topo, (neps, nst) in r["per_topo"].items():
        print(f"  {topo}: {neps} episodes, {nst:,} steps")
    print(f"  Train: {r['n_train_steps']:,} steps | Test: {r['n_test_steps']:,} steps | dim={D_}")

    print("\n  [1] Rank Diagnostics (what has content after normalization)")
    print(f"    rank(state)  @ 95%: {r['state_rank_95']}/{D_}  |  @ 99%: {r['state_rank_99']}/{D_}")
    print(f"    rank(delta)  @ 95%: {r['delta_rank_95']}/{D_}  |  @ 99%: {r['delta_rank_99']}/{D_}"
          "   <- what JEPA must model")
    print(f"    Dead features (var<1e-4): {len(r['dead_features'])}/{D_}  {r['dead_features']}")
    print(f"    -> {'DYNAMICS RANK LOW' if r['delta_rank_95'] < 4 else 'OK'}")

    print("\n  [2] One-step Prediction Baselines")
    print(f"    Zero   (scale):     {r['zero_mse']:.4f}")
    print(f"    Identity (copy):    {r['ident_mse']:.4f}  <- JEPA must beat")
    print(f"    Linear AR s,a->s':  {r['ar_1step_mse']:.4f}  <- JEPA should beat")
    ratio = r['ident_mse'] / max(r['ar_1step_mse'], 1e-10)
    print(f"    Identity / AR ratio: {ratio:.2f}  "
          "(>1.2 = linear model finds signal; >1.5 = more room for JEPA)")

    print("\n  [3] Linear AR Multi-step Rollout (MSE per horizon)")
    for h, mse in sorted(r["rollout_mse"].items()):
        print(f"    k={h:>2}:  MSE={mse:.4f}")

    print("\n  [4] Action-Conditional Signal (F-ratio per feature)")
    print(f"    Action types: {r['action_types']}")
    print(f"    Mean SNR: {r['snr_mean']:.3f} | Strong (>0.5): {r['snr_strong']}/{D_}")
    print(f"    Top-5 responsive features:")
    for i in r["snr_top5"]:
        print(f"      {FEATURE_NAMES[i]:<15} SNR={r['snr'][i]:.3f}")

    print("\n  [5] Train/Test Shift (KS statistic magnitude)")
    if r["ks_top5"]:
        print(f"    Features with KS>0.2: {r['ks_strong']}/{D_}")
        print(f"    Top-5 shifted:")
        for i, (s, p) in r["ks_top5"]:
            print(f"      {FEATURE_NAMES[i]:<15} KS={s:.3f}")
    else:
        print("    No test set (skipped)")

    print("\n  [6] Window Diversity at context_length=8")
    w = r["window"]
    print(f"    Windows: {w['n_windows']:,} | Unique sequences: {w['n_unique']:,} "
          f"({w['uniqueness']:.1%})")
    print(f"    Multi-action windows: {w['multi_action_rate']:.1%} "
          "(fraction with >=2 distinct action types)")

    # Verdict (relaxed)
    flags = []
    if r["delta_rank_95"] < 4:           flags.append("dynamics span too low")
    if len(r["dead_features"]) > 2:      flags.append("multiple dead features")
    if ratio < 1.1:                      flags.append("AR barely beats identity")
    if r["snr_strong"] < 3:              flags.append("fewer than 3 action-responsive features")
    if w["multi_action_rate"] < 0.15:    flags.append("most windows are single-action")

    print("\n  VERDICT")
    if flags:
        print("    NOT READY:")
        for f in flags: print(f"      - {f}")
    else:
        print("    READY FOR JEPA:")
        print("      - dynamics span is non-trivial")
        print("      - linear AR has room to be improved by JEPA")
        print("      - actions move multiple features meaningfully")
        print("      - windows contain action transitions, not just one type")
    print()


def print_comparison(results):
    print("\n" + "=" * 100)
    print("  JEPA READINESS - SCALING COMPARISON")
    print("=" * 100)
    cols = [
        ("Dataset", 12),
        ("Train eps", 10),
        ("Train steps", 11),
        ("Dyn rank", 9),
        ("Ident MSE", 10),
        ("AR MSE", 9),
        ("AR k=10", 9),
        ("Ident/AR", 9),
        ("Strong SNR", 11),
        ("Unique win%", 11),
        ("Multi-act%", 10),
        ("KS>0.2", 7),
    ]
    header = "  " + " ".join(f"{n:<{w}}" for n, w in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        train_eps = sum(neps for topo, (neps, _) in r["per_topo"].items()
                        if topo in TRAIN_TOPOS)
        ratio = r["ident_mse"] / max(r["ar_1step_mse"], 1e-10)
        ar10 = r["rollout_mse"].get(10, float("nan"))
        name = Path(r["data_dir"]).name
        vals = [
            name, f"{train_eps}", f"{r['n_train_steps']:,}",
            f"{r['delta_rank_95']}/{D}", f"{r['ident_mse']:.4f}",
            f"{r['ar_1step_mse']:.4f}", f"{ar10:.4f}",
            f"{ratio:.2f}", f"{r['snr_strong']}/{D}",
            f"{r['window']['uniqueness']:.1%}",
            f"{r['window']['multi_action_rate']:.1%}",
            f"{r['ks_strong']}/{D}",
        ]
        print("  " + " ".join(f"{v:<{w}}" for v, (_, w) in zip(vals, cols)))
    print()
    print("  Notes:")
    print("    - Dyn rank: dims needed to explain 95% of delta variance (higher = richer dynamics)")
    print("    - Ident/AR: >1.2 means linear AR finds structure over naive copy (more = better signal)")
    print("    - AR k=10: how quickly does a linear rollout diverge (lower is easier to model)")
    print("    - Strong SNR: count of features the action reliably moves")
    print("    - Unique win%: distinct 8-step action sequences / total windows")
    print("    - Multi-act%: fraction of 8-step windows with >=2 action types")
    print("    - KS>0.2: count of features with real distribution shift on topo_D")
    print()


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True,
                        help="One or more dataset directories")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    results = []
    for d in args.data:
        print(f"Analyzing {d}...", flush=True)
        r = analyze(d, args.max_episodes)
        results.append(r)

    if len(results) == 1:
        print_single_report(results[0])
    else:
        for r in results:
            print_single_report(r)
        print_comparison(results)


if __name__ == "__main__":
    main()
