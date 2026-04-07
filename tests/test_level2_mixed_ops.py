"""
Level 2 Test: Mixed Operations Policy.
REQUIRES GNPy — run inside Docker.

Run:
  cd /work && pip install h5py networkx -q && pip install -e . -q
  python tests/test_level2_mixed_ops.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import time
from collections import Counter

from optical_wm.core.schemas import ActionType
from optical_wm.core.gnpy_wrapper import (
    build_test_topology, create_evaluator, GNPY_AVAILABLE,
)
from optical_wm.policies.mixed_ops import MixedOpsPolicy, MixedOpsConfig


def setup(seed=42, max_steps=30, epsilon=0.10):
    spec = build_test_topology(n_nodes=6, connectivity=0.4, seed=seed)
    evaluator, spec = create_evaluator(spec)
    config = MixedOpsConfig(
        max_steps=max_steps, epsilon=epsilon, seed=seed,
    )
    return MixedOpsPolicy(evaluator, spec, config), spec


# =====================================================================
# Test 2X.1 — Episode completes
# =====================================================================

def test_episode_completes():
    print("\n--- Test 2X.1: Episode completes ---")
    policy, _ = setup(seed=42, max_steps=30)
    t0 = time.time()
    ep = policy.generate_episode()
    elapsed = time.time() - t0

    meta = ep['metadata']
    assert meta['n_steps'] > 0
    assert len(ep['states']) == meta['n_steps'] + 1
    assert len(ep['actions']) == meta['n_steps']

    print(f"  Steps: {meta['n_steps']}")
    print(f"  Action types: {meta['action_type_counts']}")
    print(f"  Phase counts: {meta['phase_counts']}")
    print(f"  Time: {elapsed:.1f}s")
    print("✓ Episode completes OK")
    return ep


# =====================================================================
# Test 2X.2 — All 4 phases present
# =====================================================================

def test_four_phases(ep=None):
    print("\n--- Test 2X.2: Four phases present ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    phases = set()
    for s in ep['step_info']:
        main_phase = s['phase'].split('_')[0]
        phases.add(main_phase)

    expected = {'provision', 'optimize', 'perturb', 'reprovision'}
    missing = expected - phases

    assert len(missing) == 0, (
        f"Missing phases: {missing}. Found: {phases}"
    )

    print(f"  Phases found: {sorted(phases)}")
    print("✓ Four phases OK")


# =====================================================================
# Test 2X.3 — Multiple action types (≥3)
# =====================================================================

def test_action_diversity(ep=None):
    print("\n--- Test 2X.3: Action type diversity ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    type_counts = Counter(int(a[0]) for a in ep['actions'])
    type_names = {ActionType(t).name: c for t, c in type_counts.items()}

    assert len(type_counts) >= 3, (
        f"Only {len(type_counts)} action types: {type_names}. "
        f"Expected ≥3 (ADD, REMOVE, POWER_ADJUST at minimum)."
    )

    print(f"  Action distribution: {type_names}")
    print(f"  Distinct types: {len(type_counts)}")
    print("✓ Action diversity OK")


# =====================================================================
# Test 2X.4 — LP count is non-monotonic (dip from perturbation)
# =====================================================================

def test_non_monotonic_lp_count(ep=None):
    print("\n--- Test 2X.4: Non-monotonic LP count ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    n_lps = [s['global_features'][0] for s in ep['states']]

    # Should increase, then decrease, then increase again
    max_lp = max(n_lps)
    min_after_peak = min(n_lps[n_lps.index(max_lp):])

    has_increase = any(n_lps[i+1] > n_lps[i] for i in range(len(n_lps)-1))
    has_decrease = any(n_lps[i+1] < n_lps[i] for i in range(len(n_lps)-1))

    assert has_increase and has_decrease, (
        f"LP trajectory is monotonic. "
        f"Range: {min(n_lps):.0f} → {max(n_lps):.0f}. "
        f"Increases: {has_increase}, Decreases: {has_decrease}"
    )

    print(f"  LP range: {min(n_lps):.0f} → {max(n_lps):.0f}")
    print(f"  Has increases: {has_increase}")
    print(f"  Has decreases: {has_decrease}")
    print(f"  Trajectory: {' → '.join(str(int(n)) for n in n_lps[::5])}")
    print("✓ Non-monotonic LP count OK")


# =====================================================================
# Test 2X.5 — Conditional action diversity
#   (same load range → multiple action types)
# =====================================================================

def test_conditional_diversity(ep=None):
    print("\n--- Test 2X.5: Conditional action diversity (P(a|s)) ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    # Bucket states by LP count (proxy for load)
    buckets = {}  # lp_count_bucket → set of action types
    for i, action in enumerate(ep['actions']):
        n_lps = int(ep['states'][i]['global_features'][0])
        bucket = (n_lps // 5) * 5  # group by 5s
        action_type = int(action[0])
        if bucket not in buckets:
            buckets[bucket] = set()
        buckets[bucket].add(action_type)

    # At least one bucket should have ≥2 action types
    max_diversity = max(len(v) for v in buckets.values())
    diverse_buckets = sum(1 for v in buckets.values() if len(v) >= 2)

    for bucket, types in sorted(buckets.items()):
        type_names = [ActionType(t).name for t in types]
        print(f"  Load ~{bucket} LPs: {type_names}")

    assert max_diversity >= 2, (
        f"No load bucket has ≥2 action types. "
        f"State→action correlation is too strong."
    )

    print(f"  Buckets with ≥2 types: {diverse_buckets}/{len(buckets)}")
    print("✓ Conditional diversity OK")


# =====================================================================
# Test 2X.6 — Seed diversity
# =====================================================================

def test_seed_diversity():
    print("\n--- Test 2X.6: Seed diversity ---")
    p1, _ = setup(seed=42, max_steps=20)
    p2, _ = setup(seed=88, max_steps=20)
    ep1 = p1.generate_episode()
    ep2 = p2.generate_episode()

    s1 = ep1['states'][-1]['global_features']
    s2 = ep2['states'][-1]['global_features']
    assert not np.array_equal(s1, s2)

    print(f"  Seed 42: {ep1['metadata']['action_type_counts']}")
    print(f"  Seed 88: {ep2['metadata']['action_type_counts']}")
    print("✓ Seed diversity OK")


# =====================================================================
# Run all
# =====================================================================

if __name__ == "__main__":
    if not GNPY_AVAILABLE:
        print("GNPy NOT AVAILABLE. Run inside Docker.")
        sys.exit(1)

    print("=" * 60)
    print("Level 2: Mixed Operations Policy Tests")
    print("=" * 60)

    passed = failed = 0
    ep = None

    for name, fn, use_ep in [
        ("2X.1 Episode completes", test_episode_completes, False),
        ("2X.2 Four phases", test_four_phases, True),
        ("2X.3 Action diversity", test_action_diversity, True),
        ("2X.4 Non-monotonic LP", test_non_monotonic_lp_count, True),
        ("2X.5 Conditional diversity", test_conditional_diversity, True),
        ("2X.6 Seed diversity", test_seed_diversity, False),
    ]:
        try:
            if use_ep:
                result = fn(ep)
            else:
                result = fn()
            if name.endswith("completes"):
                ep = result
            passed += 1
        except Exception as e:
            print(f"\n✗ {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Mixed Operations: {passed} passed, {failed} failed")
