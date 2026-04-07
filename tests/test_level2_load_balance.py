"""
Level 2 Test: Load Balancing Policy.
REQUIRES GNPy — run inside Docker.

Run:
  cd /work && pip install h5py networkx -q && pip install -e . -q
  python tests/test_level2_load_balance.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import time

from optical_wm.core.schemas import ActionType
from optical_wm.core.gnpy_wrapper import (
    build_test_topology, create_evaluator, GNPY_AVAILABLE,
)
from optical_wm.policies.load_balancing import (
    LoadBalancePolicy, LoadBalanceConfig,
)


def setup(seed=42, max_steps=15):
    spec = build_test_topology(n_nodes=6, connectivity=0.4, seed=seed)
    evaluator, spec = create_evaluator(spec)
    config = LoadBalanceConfig(
        initial_load_frac=0.3, max_steps=max_steps, seed=seed,
    )
    return LoadBalancePolicy(evaluator, spec, config), spec


def test_episode_completes():
    print("\n--- Test 2L.1: Episode completes ---")
    policy, _ = setup(seed=42, max_steps=15)
    t0 = time.time()
    ep = policy.generate_episode()
    elapsed = time.time() - t0

    meta = ep['metadata']
    assert meta['n_steps'] > 0
    assert len(ep['states']) == meta['n_steps'] + 1

    print(f"  Steps: {meta['n_steps']}")
    print(f"  Initial util std: {meta['initial_util_std']:.4f}")
    print(f"  Final util std: {meta['final_util_std']:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    print("✓ Episode completes OK")
    return ep


def test_reroute_actions(ep=None):
    print("\n--- Test 2L.2: Actions are REROUTE ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    types = [int(a[0]) for a in ep['actions']]
    n_reroute = sum(1 for t in types if t == ActionType.REROUTE)
    reroute_frac = n_reroute / max(1, len(types))

    assert reroute_frac >= 0.8, (
        f"Only {reroute_frac:.0%} REROUTE actions. "
        f"Load balancing should primarily reroute."
    )
    print(f"  REROUTE: {n_reroute}/{len(types)} ({reroute_frac:.0%})")
    print("✓ REROUTE actions OK")


def test_lp_count_stable(ep=None):
    print("\n--- Test 2L.3: LP count stable ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    n_lps = [s['global_features'][0] for s in ep['states']]
    assert n_lps[0] == n_lps[-1], (
        f"LP count changed: {n_lps[0]:.0f} → {n_lps[-1]:.0f}. "
        f"Load balancing should only reroute, not add/remove."
    )
    print(f"  LP count: {n_lps[0]:.0f} (stable throughout)")
    print("✓ LP count stable OK")


def test_seed_diversity():
    print("\n--- Test 2L.4: Seed diversity ---")
    p1, _ = setup(seed=42, max_steps=10)
    p2, _ = setup(seed=77, max_steps=10)
    ep1 = p1.generate_episode()
    ep2 = p2.generate_episode()

    s1 = ep1['states'][-1]['global_features']
    s2 = ep2['states'][-1]['global_features']
    assert not np.array_equal(s1, s2)

    print(f"  Seed 42:  util_std={ep1['metadata']['final_util_std']:.4f}")
    print(f"  Seed 77:  util_std={ep2['metadata']['final_util_std']:.4f}")
    print("✓ Seed diversity OK")


if __name__ == "__main__":
    if not GNPY_AVAILABLE:
        print("GNPy NOT AVAILABLE. Run inside Docker.")
        sys.exit(1)

    print("=" * 60)
    print("Level 2: Load Balancing Policy Tests")
    print("=" * 60)

    passed = failed = 0
    ep = None

    for name, fn, use_ep in [
        ("2L.1 Episode completes", test_episode_completes, False),
        ("2L.2 REROUTE actions", test_reroute_actions, True),
        ("2L.3 LP count stable", test_lp_count_stable, True),
        ("2L.4 Seed diversity", test_seed_diversity, False),
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
    print(f"Load Balancing: {passed} passed, {failed} failed")
