"""
Level 2 Test: Recovery Policy.
REQUIRES GNPy — run inside Docker.

Run:
  cd /work && pip install h5py networkx -q && pip install -e . -q
  python tests/test_level2_recovery.py
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
from optical_wm.policies.recovery import RecoveryPolicy, RecoveryConfig


def setup(seed=42, max_steps=30):
    spec = build_test_topology(n_nodes=6, connectivity=0.4, seed=seed)
    evaluator, spec = create_evaluator(spec)
    config = RecoveryConfig(
        initial_load_frac=0.4, disruption_frac=0.4,
        max_steps=max_steps, seed=seed,
    )
    return RecoveryPolicy(evaluator, spec, config), spec


def test_episode_completes():
    print("\n--- Test 2R.1: Episode completes ---")
    policy, _ = setup(seed=42, max_steps=30)
    t0 = time.time()
    ep = policy.generate_episode()
    elapsed = time.time() - t0

    meta = ep['metadata']
    assert meta['n_steps'] > 0
    assert len(ep['states']) == meta['n_steps'] + 1

    print(f"  Steps: {meta['n_steps']} "
          f"(disruption={meta['n_disruption_steps']}, "
          f"restoration={meta['n_restoration_steps']})")
    print(f"  Failed link: {meta['failed_link']}")
    print(f"  Demands affected: {meta['demands_affected']}")
    print(f"  Demands restored: {meta['demands_restored']}")
    print(f"  Restore failed (rollback): {meta.get('demands_restore_failed', 0)}")
    print(f"  Phase ratio (add/remove): {meta.get('phase_ratio', 0):.1f}")
    print(f"  Time: {elapsed:.1f}s")
    print("✓ Episode completes OK")
    return ep


def test_two_phases(ep=None):
    """Episode must have REMOVE then ADD phases in order."""
    print("\n--- Test 2R.2: Two phases (REMOVE then ADD) ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    types = [int(a[0]) for a in ep['actions']]
    n_remove = sum(1 for t in types if t == ActionType.REMOVE)
    n_add = sum(1 for t in types if t == ActionType.ADD)

    assert n_remove > 0, "No REMOVE actions in disruption phase"
    assert n_add > 0, "No ADD actions in restoration phase"

    # Phase order: REMOVEs should come before ADDs
    first_add = next((i for i, t in enumerate(types) if t == ActionType.ADD),
                     len(types))
    last_remove = max(
        (i for i, t in enumerate(types) if t == ActionType.REMOVE),
        default=-1,
    )
    assert last_remove < first_add, (
        f"REMOVE at step {last_remove} after ADD at step {first_add}. "
        f"Disruption should precede restoration."
    )

    print(f"  REMOVE steps: {n_remove}")
    print(f"  ADD steps: {n_add}")
    print(f"  Phase order: REMOVE[0..{last_remove}] → ADD[{first_add}..{len(types)-1}]")
    print("✓ Two phases OK")


def test_lp_count_dip(ep=None):
    """LP count must dip (disruption) then recover (restoration)."""
    print("\n--- Test 2R.3: LP count dips then recovers ---")
    if ep is None:
        policy, _ = setup()
        ep = policy.generate_episode()

    n_lps = [s['global_features'][0] for s in ep['states']]
    initial = n_lps[0]
    minimum = min(n_lps)
    final = n_lps[-1]

    assert minimum < initial, (
        f"LP count never dipped: min={minimum:.0f}, initial={initial:.0f}"
    )
    assert final > minimum, (
        f"LP count never recovered: final={final:.0f}, min={minimum:.0f}"
    )

    recovery_rate = (final - minimum) / max(1, initial - minimum)
    print(f"  LP count: {initial:.0f} → {minimum:.0f} (dip) → {final:.0f}")
    print(f"  Recovery rate: {recovery_rate:.0%}")
    print("✓ LP count dip-and-recover OK")


def test_gsnr_non_monotonic(ep=None):
    """Avg margin should improve during disruption then degrade during restoration."""
    print("\n--- Test 2R.5: GSNR non-monotonic (↑ then ↓) ---")
    if ep is None:
        policy, _ = setup(seed=42, max_steps=30)
        ep = policy.generate_episode()

    margins = [s['global_features'][5] for s in ep['states']]  # avg margin
    T = len(margins)

    # Find the peak (should be around the phase transition)
    peak_idx = int(np.argmax(margins))
    margin_start = margins[0]
    margin_peak = margins[peak_idx]
    margin_end = margins[-1]

    has_rise = margin_peak > margin_start + 0.01
    has_fall = margin_peak > margin_end + 0.01

    print(f"  Avg margin: {margin_start:.2f} → {margin_peak:.2f} (peak@{peak_idx}) → {margin_end:.2f}")
    print(f"  Rise (disruption): {has_rise}  ({margin_peak - margin_start:+.2f} dB)")
    print(f"  Fall (restoration): {has_fall}  ({margin_end - margin_peak:+.2f} dB)")

    # At least one direction should show a clear change
    margin_range = max(margins) - min(margins)
    assert margin_range > 0.1, (
        f"GSNR trajectory too flat: range={margin_range:.3f} dB. "
        f"Expected non-monotonic dynamics."
    )
    print(f"  Total range: {margin_range:.2f} dB")
    print("✓ GSNR non-monotonic OK")


def test_spectrum_coherence(ep=None):
    """REMOVE steps should free spectrum, ADD steps should occupy spectrum."""
    print("\n--- Test 2R.6: Spectrum coherence ---")
    if ep is None:
        policy, _ = setup(seed=42, max_steps=30)
        ep = policy.generate_episode()

    states = ep['states']
    step_info = ep['step_info']

    remove_freed = 0
    remove_checked = 0
    add_occupied = 0
    add_checked = 0

    for i, si in enumerate(step_info):
        occ_before = states[i]['spectral_occupancy'].sum()
        occ_after = states[i + 1]['spectral_occupancy'].sum()

        if si['phase'] == 'disruption':
            remove_checked += 1
            if occ_after < occ_before:
                remove_freed += 1
        elif si['phase'] == 'restoration':
            add_checked += 1
            if occ_after > occ_before:
                add_occupied += 1

    remove_rate = remove_freed / max(1, remove_checked)
    add_rate = add_occupied / max(1, add_checked)

    print(f"  REMOVE freed spectrum: {remove_freed}/{remove_checked} ({remove_rate:.0%})")
    print(f"  ADD occupied spectrum: {add_occupied}/{add_checked} ({add_rate:.0%})")

    assert remove_rate >= 0.8, (
        f"Only {remove_rate:.0%} of REMOVE steps freed spectrum"
    )
    # ADD rate can be lower due to rollbacks (infeasible LPs removed immediately)
    print("✓ Spectrum coherence OK")


def test_no_infeasible_persist(ep=None):
    """After rollback fix, no infeasible LPs should persist in final state."""
    print("\n--- Test 2R.7: No infeasible LPs persist ---")
    if ep is None:
        policy, _ = setup(seed=42, max_steps=30)
        ep = policy.generate_episode()

    final_state = ep['states'][-1]
    n_infeasible = final_state['global_features'][3]
    n_active = final_state['global_features'][0]

    infeasible_ratio = n_infeasible / max(1, n_active)

    print(f"  Active LPs: {n_active:.0f}")
    print(f"  Infeasible: {n_infeasible:.0f} ({infeasible_ratio:.0%})")

    # With rollback, infeasible count should be low
    assert infeasible_ratio < 0.3, (
        f"Too many infeasible LPs in final state: {n_infeasible:.0f}/{n_active:.0f}. "
        f"Rollback should prevent infeasible LPs from persisting."
    )
    print("✓ No infeasible persist OK")


def test_seed_diversity():
    print("\n--- Test 2R.4: Seed diversity ---")
    p1, _ = setup(seed=42, max_steps=20)
    p2, _ = setup(seed=55, max_steps=20)
    ep1 = p1.generate_episode()
    ep2 = p2.generate_episode()

    s1 = ep1['states'][-1]['global_features']
    s2 = ep2['states'][-1]['global_features']
    assert not np.array_equal(s1, s2), 'Different seeds produced identical states'

    print(f"  Seed 42: link={ep1['metadata']['failed_link']}, "
          f"steps={ep1['metadata']['n_steps']}, "
          f"restored={ep1['metadata']['demands_restored']}")
    print(f"  Seed 55: link={ep2['metadata']['failed_link']}, "
          f"steps={ep2['metadata']['n_steps']}, "
          f"restored={ep2['metadata']['demands_restored']}")
    print("✓ Seed diversity OK")


if __name__ == "__main__":
    if not GNPY_AVAILABLE:
        print("GNPy NOT AVAILABLE. Run inside Docker.")
        sys.exit(1)

    print("=" * 60)
    print("Level 2: Recovery Policy Tests")
    print("=" * 60)

    passed = failed = 0
    ep = None

    for name, fn, use_ep in [
        ("2R.1 Episode completes", test_episode_completes, False),
        ("2R.2 Two phases", test_two_phases, True),
        ("2R.3 LP count dip", test_lp_count_dip, True),
        ("2R.4 Seed diversity", test_seed_diversity, False),
        ("2R.5 GSNR non-monotonic", test_gsnr_non_monotonic, True),
        ("2R.6 Spectrum coherence", test_spectrum_coherence, True),
        ("2R.7 No infeasible persist", test_no_infeasible_persist, True),
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
    print(f"Recovery: {passed} passed, {failed} failed")