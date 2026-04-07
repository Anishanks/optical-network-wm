"""
Level 2 Test: Margin Optimization Policy.
REQUIRES GNPy — run inside Docker.

Run:
  cd /work && pip install h5py networkx -q && pip install -e . -q
  python tests/test_level2_margin_opt.py
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
from optical_wm.policies.margin_optimization import (
    MarginOptPolicy, MarginOptConfig,
)


def setup(seed=42, max_steps=15):
    spec = build_test_topology(n_nodes=6, connectivity=0.4, seed=seed)
    evaluator, spec = create_evaluator(spec)
    config = MarginOptConfig(
        initial_load_frac=0.3, max_steps=max_steps, seed=seed,
    )
    return MarginOptPolicy(evaluator, spec, config), spec


def test_episode_completes():
    print("\n--- Test 2M.1: Episode completes ---")
    policy, _ = setup(seed=42, max_steps=15)
    t0 = time.time()
    ep = policy.generate_episode()
    elapsed = time.time() - t0

    meta = ep['metadata']
    assert meta['n_steps'] > 0
    assert len(ep['states']) == meta['n_steps'] + 1
    assert len(ep['actions']) == meta['n_steps']

    print(f"  Steps: {meta['n_steps']}")
    print(f"  Initial worst margin: {meta['initial_worst_margin']:.2f} dB")
    print(f"  Final worst margin: {meta['final_worst_margin']:.2f} dB")
    print(f"  Time: {elapsed:.1f}s")
    print("✓ Episode completes OK")
    return ep


def test_mixed_action_types(ep=None):
    print("\n--- Test 2M.2: Mixed action types ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    types = set(int(a[0]) for a in ep['actions'])
    print(f"  Action types used: {[ActionType(t).name for t in types]}")

    # Should use at least 2 different action types
    assert len(types) >= 2, (
        f"Only {len(types)} action type(s) used: {types}. "
        f"Expected POWER_ADJUST + MOD_CHANGE or REROUTE."
    )
    print("✓ Mixed action types OK")


def test_gsnr_variation(ep=None):
    print("\n--- Test 2M.3: GSNR variation ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    margins = [s['global_features'][4] for s in ep['states']]
    margin_range = max(margins) - min(margins)

    assert margin_range > 0.05, (
        f"Margin range too small: {margin_range:.3f} dB"
    )
    print(f"  Worst margin range: {margin_range:.3f} dB")
    print(f"  Worst margin: {min(margins):.2f} → {max(margins):.2f} dB")
    print("✓ GSNR variation OK")


def test_seed_diversity():
    print("\n--- Test 2M.4: Seed diversity ---")
    p1, _ = setup(seed=42, max_steps=10)
    p2, _ = setup(seed=99, max_steps=10)
    ep1 = p1.generate_episode()
    ep2 = p2.generate_episode()

    s1 = ep1['states'][-1]['global_features']
    s2 = ep2['states'][-1]['global_features']
    assert not np.array_equal(s1, s2)

    print(f"  Seed 42:  {ep1['metadata']['n_steps']} steps, "
          f"final margin={s1[4]:.2f} dB")
    print(f"  Seed 99:  {ep2['metadata']['n_steps']} steps, "
          f"final margin={s2[4]:.2f} dB")
    print("✓ Seed diversity OK")


if __name__ == "__main__":
    if not GNPY_AVAILABLE:
        print("GNPy NOT AVAILABLE. Run inside Docker.")
        sys.exit(1)

    print("=" * 60)
    print("Level 2: Margin Optimization Policy Tests")
    print("=" * 60)

    passed = failed = 0
    ep = None

    for name, fn, use_ep in [
        ("2M.1 Episode completes", test_episode_completes, False),
        ("2M.2 Mixed action types", test_mixed_action_types, True),
        ("2M.3 GSNR variation", test_gsnr_variation, True),
        ("2M.4 Seed diversity", test_seed_diversity, False),
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
    print(f"Margin Optimization: {passed} passed, {failed} failed")
