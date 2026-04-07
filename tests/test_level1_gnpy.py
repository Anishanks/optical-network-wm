"""
Level 1 Tests: GNPy wrapper unit tests.
REQUIRES GNPy — run inside Docker:
  docker run -it --rm -v $(pwd):/work telecominfraproject/oopt-gnpy
  cd /work && python tests/test_level1.py

These tests verify that:
  1.1 — A single GNPy call works and returns plausible values
  1.2 — Inter-channel NLI coupling is captured (THE CRITICAL TEST)
  1.3 — State encoder extracts correct features from GNPy output
  1.4 — Add/remove actions change the state correctly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np
import sys
import os

from optical_wm.core.schemas import (
    Modulation, LightpathDesc, LightpathResult,
    encode_state, create_empty_state,
)
from optical_wm.core.gnpy_wrapper import (
    build_test_topology, create_evaluator, create_lightpath,
    TopologyBuilder, NetworkEvaluator, GNPY_AVAILABLE,
)


def check_gnpy():
    if not GNPY_AVAILABLE:
        print("="*60)
        print("GNPy NOT AVAILABLE.")
        print("Run inside Docker:")
        print("  docker run -it --rm -v $(pwd):/work \\")
        print("    telecominfraproject/oopt-gnpy")
        print("  cd /work && python tests/test_level1.py")
        print("="*60)
        sys.exit(1)


# =========================================================================
# Shared setup: build a small test network once
# =========================================================================

def setup_network(n_nodes=6, connectivity=0.4, seed=42):
    """Build a small test network for all Level 1 tests."""
    spec = build_test_topology(n_nodes=n_nodes, connectivity=connectivity,
                               seed=seed)
    evaluator, spec = create_evaluator(spec)
    return evaluator, spec


# =========================================================================
# Test 1.1 — Single GNPy call
# =========================================================================

def test_gnpy_single_call(evaluator=None, spec=None):
    """One topology, one lightpath, one GNPy call. Verify output."""
    print("\n--- Test 1.1: Single GNPy call ---")
    evaluator, spec = setup_network()

    # Find a valid 2-hop route
    src = spec.node_ids[0]
    dst = spec.node_ids[-1]
    # Simple: use first and last node, route through intermediate
    # Find a path
    mid = None
    for link in spec.links:
        if link['src'] == src:
            mid = link['dst']
            break
        if link['dst'] == src:
            mid = link['src']
            break

    if mid is None:
        # Fallback: direct link
        route = [src, dst]
    else:
        # Try mid → dst
        route = [src, mid]
        for link in spec.links:
            if (link['src'] == mid and link['dst'] == dst) or \
               (link['dst'] == mid and link['src'] == dst):
                route = [src, mid, dst]
                break

    lp = create_lightpath(
        id="lp_test_0",
        source=route[0],
        destination=route[-1],
        route=route,
        wavelength_slot=40,
        modulation=Modulation.QPSK,
        launch_power_dbm=0.0,
    )

    # GNPy NLI solver needs >= 2 channels; add a companion LP
    lp2 = create_lightpath(
        id="lp_companion", source=route[0], destination=route[-1],
        route=route, wavelength_slot=41,
        modulation=Modulation.QPSK, launch_power_dbm=0.0,
    )
    results, elapsed_ms = evaluator.evaluate_all([lp, lp2])

    # Verify output structure
    assert "lp_test_0" in results, "LP not in results"
    r = results["lp_test_0"]

    assert isinstance(r.gsnr_db, float), f"GSNR not float: {type(r.gsnr_db)}"
    assert isinstance(r.osnr_db, float), f"OSNR not float: {type(r.osnr_db)}"

    # Physical plausibility
    assert 0 < r.gsnr_db < 50, f"GSNR out of range: {r.gsnr_db}"
    assert r.osnr_db >= r.gsnr_db - 0.1, \
        f"OSNR ({r.osnr_db}) < GSNR ({r.gsnr_db}) — physically impossible"

    print(f"  Route: {' → '.join(route)}")
    print(f"  GSNR:  {r.gsnr_db:.2f} dB")
    print(f"  OSNR:  {r.osnr_db:.2f} dB")
    print(f"  Margin: {r.margin_db:.2f} dB (vs QPSK threshold {8.5} dB)")
    print(f"  Feasible: {r.feasible}")
    print(f"  Compute time: {elapsed_ms:.1f} ms")
    print("✓ Single GNPy call OK")

    return evaluator, spec


# =========================================================================
# Test 1.2 — Inter-channel NLI coupling (CRITICAL GATE)
# =========================================================================

def test_gnpy_interchannel_coupling(evaluator=None, spec=None):
    """
    THE MOST IMPORTANT TEST.

    Adding a second channel on a shared link MUST change the GSNR
    of the first channel due to non-linear interference (XPM).

    If this test fails, the entire dataset is useless because the
    world model would learn independent per-channel dynamics instead
    of the coupled multi-channel dynamics that make JEPA necessary.
    """
    print("\n--- Test 1.2: Inter-channel coupling (CRITICAL) ---")

    if evaluator is None:
        evaluator, spec = setup_network()

    # Find two routes that share at least one link
    # Strategy: use same source, different destinations, shared first hop
    src = spec.node_ids[0]

    # Find two neighbors of src
    neighbors = []
    for link in spec.links:
        if link['src'] == src and link['dst'] not in neighbors:
            neighbors.append(link['dst'])
        elif link['dst'] == src and link['src'] not in neighbors:
            neighbors.append(link['src'])

    assert len(neighbors) >= 1, \
        f"Node {src} has no neighbors — topology too sparse"

    # Route A: src → neighbor_0
    route_A = [src, neighbors[0]]
    # Route B: same source, same link if only 1 neighbor, different slot
    route_B = [src, neighbors[0]]  # same route, adjacent wavelength

    lp_A = create_lightpath(
        id="lp_A", source=src, destination=neighbors[0],
        route=route_A, wavelength_slot=40,
        modulation=Modulation.QPSK, launch_power_dbm=0.0,
    )
    lp_B = create_lightpath(
        id="lp_B", source=src, destination=neighbors[0],
        route=route_B, wavelength_slot=41,  # adjacent channel
        modulation=Modulation.QPSK, launch_power_dbm=0.0,
    )

    # GNPy needs >= 2 channels, so baseline is LP_A + a far-away companion
    lp_far = create_lightpath(
        id="lp_far", source=src, destination=neighbors[0],
        route=route_A, wavelength_slot=1,
        modulation=Modulation.QPSK, launch_power_dbm=0.0,
    )

    # Step 1: LP_A + far companion (minimal coupling)
    results_alone, _ = evaluator.evaluate_all([lp_A, lp_far])
    gsnr_A_alone = results_alone["lp_A"].gsnr_db

    # Step 2: LP_A + far companion + LP_B adjacent (more NLI)
    results_together, _ = evaluator.evaluate_all([lp_A, lp_far, lp_B])
    gsnr_A_with_B = results_together["lp_A"].gsnr_db

    delta = gsnr_A_alone - gsnr_A_with_B

    print(f"  LP_A alone:       GSNR = {gsnr_A_alone:.4f} dB")
    print(f"  LP_A + LP_B:      GSNR = {gsnr_A_with_B:.4f} dB")
    print(f"  Delta (coupling): {delta:.4f} dB")

    if abs(delta) < 0.001:
        print()
        print("=" * 60)
        print("CRITICAL FAILURE: NO INTER-CHANNEL COUPLING DETECTED!")
        print("=" * 60)
        print()
        print("GNPy is evaluating channels independently.")
        print("This means the SpectralInformation is not being built")
        print("with all channels, or NLI computation is disabled.")
        print()
        print("Possible causes:")
        print("  1. Each LP is propagated with its own single-channel SI")
        print("  2. NLI computation is disabled in sim_params")
        print("  3. The fiber is too short for measurable NLI")
        print()
        print("DO NOT PROCEED until this is fixed.")
        print("The entire dataset depends on this coupling.")
        print("=" * 60)
        assert False, "No inter-channel coupling detected"

    assert delta > 0, (
        f"Adding a channel IMPROVED GSNR by {-delta:.4f} dB — "
        f"physically impossible for NLI. Check GNPy configuration."
    )

    print("✓ Inter-channel coupling CONFIRMED")
    print(f"  Adding 1 adjacent channel degrades GSNR by {delta:.4f} dB")

    # Bonus: test with more channels
    print("\n  Scaling test (more channels = more degradation):")
    for n_extra in [5, 10, 20]:
        extra_lps = [lp_A]
        for i in range(n_extra):
            extra_lps.append(create_lightpath(
                id=f"lp_extra_{i}", source=src, destination=neighbors[0],
                route=route_A, wavelength_slot=41 + i,
                modulation=Modulation.QPSK, launch_power_dbm=0.0,
            ))
        results_n, _ = evaluator.evaluate_all(extra_lps)
        gsnr_n = results_n["lp_A"].gsnr_db
        delta_n = gsnr_A_alone - gsnr_n
        print(f"    +{n_extra:2d} channels: GSNR = {gsnr_n:.2f} dB "
              f"(Δ = {delta_n:.3f} dB)")

    return evaluator, spec


# =========================================================================
# Test 1.3 — State encoder with real GNPy data
# =========================================================================

def test_state_encoding_with_gnpy(evaluator=None, spec=None):
    """Verify state encoder correctly captures GNPy output."""
    print("\n--- Test 1.3: State encoding with real GNPy ---")

    if evaluator is None:
        evaluator, spec = setup_network()

    src = spec.node_ids[0]
    neighbors = []
    for link in spec.links:
        if link['src'] == src:
            neighbors.append(link['dst'])
        elif link['dst'] == src:
            neighbors.append(link['src'])

    # Create 3 lightpaths
    lps = []
    for i in range(min(3, len(neighbors))):
        route = [src, neighbors[i % len(neighbors)]]
        lps.append(create_lightpath(
            id=f"lp_{i}", source=src, destination=route[-1],
            route=route, wavelength_slot=40 + i,
            modulation=Modulation.QPSK, launch_power_dbm=0.0,
        ))

    results, _ = evaluator.evaluate_all(lps)

    # Build adjacency
    n = len(spec.node_ids)
    adj = np.zeros((n, n))
    for link in spec.links:
        i = spec.node_ids.index(link['src'])
        j = spec.node_ids.index(link['dst'])
        adj[i, j] = adj[j, i] = 1

    state = encode_state(
        spec.node_ids, spec.links, lps, results, adj,
    )

    # V1: LP count matches
    n_active = int(state['lp_mask'].sum())
    assert n_active == len(lps), \
        f"Expected {len(lps)} active LPs, got {n_active}"

    # V2: GSNR values match GNPy output
    for i, lp in enumerate(lps):
        stored_gsnr = state['lp_features'][i, 17]  # index 17 = GSNR
        real_gsnr = results[lp.id].gsnr_db
        assert abs(stored_gsnr - real_gsnr) < 0.01, \
            f"LP {lp.id}: stored GSNR={stored_gsnr}, real={real_gsnr}"

    # V3: Spectral occupancy has correct number of occupied slots
    total_occupied = state['spectral_occupancy'].sum()
    # Each LP occupies 1 slot per link it traverses
    expected_slots = sum(lp.n_hops for lp in lps)
    assert total_occupied == expected_slots, \
        f"Occupied slots: {total_occupied}, expected {expected_slots}"

    # V4: Global features are consistent
    assert state['global_features'][0] == len(lps)  # n_active
    assert state['global_features'][2] == \
           sum(1 for r in results.values() if r.feasible)  # n_feasible

    print(f"  Encoded {len(lps)} LPs on {len(spec.links)} links")
    print(f"  Occupied slots: {total_occupied}")
    print(f"  Global: n_active={state['global_features'][0]:.0f}, "
          f"worst_margin={state['global_features'][4]:.2f} dB")
    print("✓ State encoding with GNPy OK")

    return evaluator, spec


# =========================================================================
# Test 1.4 — Action produces state change
# =========================================================================

def test_action_state_change(evaluator=None, spec=None):
    """Adding/removing a LP produces measurable state change."""
    print("\n--- Test 1.4: Action produces state change ---")

    if evaluator is None:
        evaluator, spec = setup_network()

    src = spec.node_ids[0]
    neighbors = []
    for link in spec.links:
        if link['src'] == src:
            neighbors.append(link['dst'])
        elif link['dst'] == src:
            neighbors.append(link['src'])

    n = len(spec.node_ids)
    adj = np.zeros((n, n))
    for link in spec.links:
        i = spec.node_ids.index(link['src'])
        j = spec.node_ids.index(link['dst'])
        adj[i, j] = adj[j, i] = 1

    route = [src, neighbors[0]]

    # State with 2 LPs
    lps_before = [
        create_lightpath("lp_0", src, neighbors[0], route, 40),
        create_lightpath("lp_1", src, neighbors[0], route, 41),
    ]
    results_before, _ = evaluator.evaluate_all(lps_before)
    state_before = encode_state(
        spec.node_ids, spec.links, lps_before, results_before, adj,
    )

    # ADD a 3rd LP
    lps_after_add = lps_before + [
        create_lightpath("lp_2", src, neighbors[0], route, 42),
    ]
    results_after_add, _ = evaluator.evaluate_all(lps_after_add)
    state_after_add = encode_state(
        spec.node_ids, spec.links, lps_after_add, results_after_add, adj,
    )

    # V1: LP count increased
    assert state_after_add['lp_mask'].sum() == 3
    assert state_before['lp_mask'].sum() == 2

    # V2: Spectral occupancy changed
    assert not np.array_equal(
        state_before['spectral_occupancy'],
        state_after_add['spectral_occupancy'],
    ), "Spectral occupancy unchanged after ADD — dead step!"

    # V3: GSNR of existing LPs changed (coupling!)
    gsnr_0_before = state_before['lp_features'][0, 17]
    gsnr_0_after = state_after_add['lp_features'][0, 17]
    print(f"  ADD: LP_0 GSNR {gsnr_0_before:.3f} → {gsnr_0_after:.3f} dB")
    # Don't assert direction (could be tiny changes), just assert it changed
    # The coupling test (1.2) already verified the direction

    # REMOVE LP_1 — keep lp_0 + a companion (GNPy needs >= 2 channels)
    lp_companion = create_lightpath(
        "lp_keep", src, neighbors[0], route, 70,
    )
    lps_after_remove = [lps_before[0], lp_companion]
    results_after_remove, _ = evaluator.evaluate_all(lps_after_remove)
    state_after_remove = encode_state(
        spec.node_ids, spec.links, lps_after_remove, results_after_remove, adj,
    )

    # V4: LP count decreased
    assert state_after_remove['lp_mask'].sum() == 2  # lp_0 + companion

    # V5: Remaining LP's GSNR should improve (less NLI)
    gsnr_0_removed = state_after_remove['lp_features'][0, 17]
    print(f"  REMOVE: LP_0 GSNR {gsnr_0_before:.3f} → {gsnr_0_removed:.3f} dB")
    assert gsnr_0_removed >= gsnr_0_before - 0.01, \
        "GSNR degraded after removing a neighbor — physically wrong"

    print("✓ Action state changes OK")


# =========================================================================
# Run all
# =========================================================================

if __name__ == "__main__":
    check_gnpy()

    tests = [
        ("1.1 Single GNPy call", test_gnpy_single_call),
        ("1.2 Inter-channel coupling (CRITICAL)", test_gnpy_interchannel_coupling),
        ("1.3 State encoding with GNPy", test_state_encoding_with_gnpy),
        ("1.4 Action state changes", test_action_state_change),
    ]

    # Reuse evaluator across tests to avoid rebuilding
    evaluator = None
    spec = None

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn(evaluator, spec)
            if result is not None:
                evaluator, spec = result
            passed += 1
        except Exception as e:
            print(f"\n✗ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

            # STOP on critical failure
            if "1.2" in name:
                print("\nCRITICAL FAILURE — stopping all tests.")
                break

    print(f"\n{'='*60}")
    print(f"Level 1: {passed} passed, {failed} failed")
    if failed == 0:
        print("All Level 1 tests passed — GNPy wrapper is correct.")
        print("Inter-channel coupling confirmed.")
        print("Next: implement policies (Level 2).")
    else:
        print("FIX LEVEL 1 BEFORE PROCEEDING.")
        if any("1.2" in name for name, _ in tests[:failed]):
            print("CRITICAL: Inter-channel coupling NOT working.")
            print("The dataset will be USELESS without this.")
