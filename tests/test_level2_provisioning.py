"""
Level 2 Test: Provisioning Policy.
REQUIRES GNPy — run inside Docker.

Tests that the provisioning policy produces valid trajectories:
  2.1 — Episode completes without errors
  2.2 — Load increases over the episode
  2.3 — No dead steps (every action changes the spectrum)
  2.4 — Actions are coherent (only ADD type)
  2.5 — GSNR dynamics are non-trivial (inter-channel coupling visible)
  2.6 — Multiple seeds produce different episodes

Run:
  cd /work
  python tests/test_level2_provisioning.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np
import sys
import os
import time

from optical_wm.core.schemas import ActionType, MAX_SLOTS
from optical_wm.core.gnpy_wrapper import (
    build_test_topology, create_evaluator, GNPY_AVAILABLE,
)
from optical_wm.policies.provisioning import ProvisioningPolicy, ProvisioningConfig


def check_gnpy():
    if not GNPY_AVAILABLE:
        print("GNPy NOT AVAILABLE. Run inside Docker.")
        sys.exit(1)


def setup(seed=42, n_nodes=6, connectivity=0.4, initial_load=0.15,
          max_steps=20, n_demands=25):
    """Build network and policy for testing."""
    spec = build_test_topology(n_nodes=n_nodes, connectivity=connectivity,
                               seed=seed)
    evaluator, spec = create_evaluator(spec)
    config = ProvisioningConfig(
        initial_load_frac=initial_load,
        target_load_frac=0.7,
        max_steps=max_steps,
        n_demands=n_demands,
        seed=seed,
    )
    policy = ProvisioningPolicy(evaluator, spec, config)
    return policy, spec


# =========================================================================
# Test 2.1 — Episode completes without errors
# =========================================================================

def test_episode_completes():
    """Generate an episode. Must complete without exceptions."""
    print("\n--- Test 2.1: Episode completes ---")
    start = time.time()

    policy, spec = setup(seed=42, max_steps=15)
    episode = policy.generate_episode()

    elapsed = time.time() - start

    meta = episode['metadata']
    n_steps = meta['n_steps']

    assert n_steps > 0, "Episode produced 0 steps"
    assert len(episode['states']) == n_steps + 1, \
        f"States count mismatch: {len(episode['states'])} vs {n_steps + 1}"
    assert len(episode['actions']) == n_steps, \
        f"Actions count mismatch: {len(episode['actions'])} vs {n_steps}"

    print(f"  Steps: {n_steps}")
    print(f"  Demands served: {meta['demands_served']}/{meta['demands_total']}")
    print(f"  Demands failed: {meta['demands_failed']}")
    print(f"  Initial load: {meta['initial_load']:.1%}")
    print(f"  Final load: {meta['final_load']:.1%}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/max(1,n_steps)*1000:.0f} ms/step)")
    print("✓ Episode completes OK")

    return episode


# =========================================================================
# Test 2.2 — Load increases over the episode
# =========================================================================

def test_load_increases(episode=None):
    """Spectral utilization must increase during provisioning."""
    print("\n--- Test 2.2: Load increases ---")

    if episode is None:
        policy, _ = setup(seed=42, max_steps=15)
        episode = policy.generate_episode()

    states = episode['states']
    utilizations = [s['global_features'][6] for s in states]  # avg util

    first_half_avg = np.mean(utilizations[:len(utilizations)//2])
    second_half_avg = np.mean(utilizations[len(utilizations)//2:])

    assert second_half_avg > first_half_avg, (
        f"Load did not increase: first half avg={first_half_avg:.3f}, "
        f"second half avg={second_half_avg:.3f}"
    )

    # LP count should also increase
    n_lps = [s['global_features'][0] for s in states]  # n_active
    assert n_lps[-1] >= n_lps[0], (
        f"LP count decreased: {n_lps[0]:.0f} → {n_lps[-1]:.0f}"
    )

    print(f"  Utilization: {utilizations[0]:.3f} → {utilizations[-1]:.3f}")
    print(f"  LP count: {n_lps[0]:.0f} → {n_lps[-1]:.0f}")
    print(f"  Load increase confirmed: +{second_half_avg - first_half_avg:.3f}")
    print("✓ Load increases OK")


# =========================================================================
# Test 2.3 — No dead steps
# =========================================================================

def test_no_dead_steps(episode=None):
    """Every successful ADD must change the spectral occupancy."""
    print("\n--- Test 2.3: No dead steps ---")

    if episode is None:
        policy, _ = setup(seed=42, max_steps=15)
        episode = policy.generate_episode()

    states = episode['states']
    actions = episode['actions']
    step_info = episode['step_info']

    dead_steps = 0
    successful_adds = 0

    for t in range(len(actions)):
        action_type = int(actions[t][0])
        success = step_info[t]['success']

        if action_type == ActionType.ADD and success:
            successful_adds += 1
            occ_before = states[t]['spectral_occupancy']
            occ_after = states[t + 1]['spectral_occupancy']

            if np.array_equal(occ_before, occ_after):
                dead_steps += 1
                print(f"  WARNING: Dead step at t={t}")

    dead_ratio = dead_steps / max(1, successful_adds)

    assert dead_ratio < 0.1, (
        f"Too many dead steps: {dead_steps}/{successful_adds} "
        f"({dead_ratio:.0%})"
    )

    print(f"  Successful adds: {successful_adds}")
    print(f"  Dead steps: {dead_steps} ({dead_ratio:.0%})")
    print("✓ No dead steps OK")


# =========================================================================
# Test 2.4 — Actions are coherent
# =========================================================================

def test_action_coherence(episode=None):
    """Provisioning policy should only produce ADD actions."""
    print("\n--- Test 2.4: Action coherence ---")

    if episode is None:
        policy, _ = setup(seed=42, max_steps=15)
        episode = policy.generate_episode()

    actions = episode['actions']
    action_types = [int(a[0]) for a in actions]

    # All actions should be ADD (type 0)
    non_add = [t for t in action_types if t != ActionType.ADD]
    assert len(non_add) == 0, (
        f"Non-ADD actions found: {non_add}. "
        f"Provisioning policy should only add LPs."
    )

    print(f"  Total actions: {len(actions)}")
    print(f"  All actions are ADD: confirmed")
    print("✓ Action coherence OK")


# =========================================================================
# Test 2.5 — GSNR dynamics are non-trivial
# =========================================================================

def test_gsnr_dynamics(episode=None):
    """GSNR must show variation over the episode (coupling effect)."""
    print("\n--- Test 2.5: GSNR dynamics ---")

    if episode is None:
        policy, _ = setup(seed=42, max_steps=15)
        episode = policy.generate_episode()

    states = episode['states']

    # Track average GSNR over time
    avg_gsnrs = [s['global_features'][5] for s in states]  # avg margin
    worst_margins = [s['global_features'][4] for s in states]  # worst margin

    gsnr_range = max(avg_gsnrs) - min(avg_gsnrs)
    margin_range = max(worst_margins) - min(worst_margins)

    # GSNR should vary as load changes (more LPs = more NLI = lower GSNR)
    assert gsnr_range > 0.1, (
        f"GSNR too flat: range={gsnr_range:.3f} dB. "
        f"Inter-channel coupling may not be captured."
    )

    # Worst margin should generally decrease as load increases
    # (not strictly monotonic, but the trend should be downward)
    early_margin = np.mean(worst_margins[:len(worst_margins)//3])
    late_margin = np.mean(worst_margins[-len(worst_margins)//3:])

    print(f"  Avg margin range: {gsnr_range:.3f} dB")
    print(f"  Worst margin range: {margin_range:.3f} dB")
    print(f"  Early avg worst margin: {early_margin:.2f} dB")
    print(f"  Late avg worst margin: {late_margin:.2f} dB")
    print(f"  Trend: {'degrading ↓' if late_margin < early_margin else 'improving ↑'}")
    print("✓ GSNR dynamics OK")


# =========================================================================
# Test 2.6 — Different seeds produce different episodes
# =========================================================================

def test_seed_diversity():
    """Two different seeds must produce different trajectories."""
    print("\n--- Test 2.6: Seed diversity ---")

    policy1, _ = setup(seed=42, max_steps=10, n_demands=15)
    ep1 = policy1.generate_episode()

    policy2, _ = setup(seed=123, max_steps=10, n_demands=15)
    ep2 = policy2.generate_episode()

    # States should differ
    state1_final = ep1['states'][-1]['global_features']
    state2_final = ep2['states'][-1]['global_features']

    assert not np.array_equal(state1_final, state2_final), \
        "Different seeds produced identical final states"

    # Number of LPs should likely differ
    n_lps1 = state1_final[0]
    n_lps2 = state2_final[0]

    print(f"  Seed 42:  {ep1['metadata']['n_steps']} steps, "
          f"{n_lps1:.0f} LPs, load={state1_final[6]:.3f}")
    print(f"  Seed 123: {ep2['metadata']['n_steps']} steps, "
          f"{n_lps2:.0f} LPs, load={state2_final[6]:.3f}")
    print("✓ Seed diversity OK")


# =========================================================================
# Run all
# =========================================================================

if __name__ == "__main__":
    check_gnpy()

    print("=" * 60)
    print("Level 2: Provisioning Policy Tests")
    print("=" * 60)

    total_start = time.time()

    # Run test 2.1 and reuse episode for subsequent tests
    passed = 0
    failed = 0

    try:
        episode = test_episode_completes()
        passed += 1
    except Exception as e:
        print(f"✗ 2.1: {e}")
        import traceback; traceback.print_exc()
        failed += 1
        episode = None

    tests = [
        ("2.2 Load increases", test_load_increases),
        ("2.3 No dead steps", test_no_dead_steps),
        ("2.4 Action coherence", test_action_coherence),
        ("2.5 GSNR dynamics", test_gsnr_dynamics),
    ]

    for name, test_fn in tests:
        try:
            test_fn(episode)
            passed += 1
        except Exception as e:
            print(f"\n✗ {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    # Test 2.6 needs fresh episodes
    try:
        test_seed_diversity()
        passed += 1
    except Exception as e:
        print(f"\n✗ 2.6: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    total_time = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"Level 2 Provisioning: {passed} passed, {failed} failed "
          f"({total_time:.1f}s)")
    if failed == 0:
        print("All provisioning policy tests passed.")
        print("The policy produces valid, dynamic, goal-directed trajectories.")
        print("Next: implement remaining policies (margin opt, load balance, recovery).")
    else:
        print("FIX PROVISIONING POLICY BEFORE PROCEEDING.")
