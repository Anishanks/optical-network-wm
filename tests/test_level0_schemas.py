"""
Level 0 Tests: Schema validation.
NO GNPy required — run these anywhere to verify data structures.

Run: python -m pytest tests/test_level0.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np
import tempfile
import os

from optical_wm.core.schemas import (
    MAX_NODES, MAX_LINKS, MAX_SLOTS, MAX_LIGHTPATHS, MAX_HOPS,
    N_LINK_STATIC_FEATURES, N_NODE_FEATURES, N_LP_FEATURES,
    N_GLOBAL_FEATURES, N_ACTION_FEATURES,
    ActionType, Modulation, MOD_THRESHOLDS, MOD_CAPACITY,
    LightpathDesc, LightpathResult,
    create_empty_state, encode_state, encode_action, decode_action,
)
from optical_wm.core.hdf5_io import HDF5Writer, HDF5Reader


# =========================================================================
# Test 0.1: Constants are consistent
# =========================================================================

def test_constants():
    """Verify all dimension constants are positive and consistent."""
    assert MAX_NODES > 0
    assert MAX_LINKS > 0
    assert MAX_SLOTS == 80  # C-band standard
    assert MAX_LIGHTPATHS >= MAX_SLOTS  # can't have more LPs than slots
    assert MAX_HOPS > 2  # at least src-intermediate-dst
    assert N_LP_FEATURES == 20
    assert N_ACTION_FEATURES == 20
    assert N_GLOBAL_FEATURES == 8
    print("✓ Constants OK")


# =========================================================================
# Test 0.2: Empty state has correct shapes
# =========================================================================

def test_empty_state_shapes():
    """Verify create_empty_state produces correct tensor shapes."""
    state = create_empty_state(n_nodes=12, n_links=18)

    assert state['adjacency'].shape == (MAX_NODES, MAX_NODES)
    assert state['adjacency'].dtype == np.float32

    assert state['node_features'].shape == (MAX_NODES, N_NODE_FEATURES)
    assert state['node_features'].dtype == np.float32

    assert state['link_static'].shape == (MAX_LINKS, N_LINK_STATIC_FEATURES)
    assert state['link_endpoints'].shape == (MAX_LINKS, 2)
    assert state['link_endpoints'].dtype == np.int32

    assert state['node_mask'].shape == (MAX_NODES,)
    assert state['node_mask'].dtype == bool

    assert state['link_mask'].shape == (MAX_LINKS,)
    assert state['link_mask'].dtype == bool

    assert state['spectral_occupancy'].shape == (MAX_LINKS, MAX_SLOTS)
    assert state['spectral_occupancy'].dtype == bool

    assert state['channel_gsnr'].shape == (MAX_LINKS, MAX_SLOTS)
    assert state['channel_gsnr'].dtype == np.float32

    assert state['channel_power'].shape == (MAX_LINKS, MAX_SLOTS)
    assert state['channel_ase'].shape == (MAX_LINKS, MAX_SLOTS)
    assert state['channel_nli'].shape == (MAX_LINKS, MAX_SLOTS)

    assert state['lp_features'].shape == (MAX_LIGHTPATHS, N_LP_FEATURES)
    assert state['lp_mask'].shape == (MAX_LIGHTPATHS,)

    assert state['global_features'].shape == (N_GLOBAL_FEATURES,)

    # All should be zeros initially
    assert np.all(state['spectral_occupancy'] == False)
    assert np.all(state['channel_gsnr'] == 0)
    assert np.all(state['lp_mask'] == False)

    print("✓ Empty state shapes OK")


# =========================================================================
# Test 0.3: Action encode/decode roundtrip
# =========================================================================

def test_action_roundtrip():
    """Encode an action, decode it, verify identity."""
    # ADD action
    action = encode_action(
        action_type=ActionType.ADD,
        src_node=3, dst_node=7,
        route=[3, 5, 7],
        wavelength_slot=42,
        modulation=int(Modulation.QPSK),
        power_delta_db=0.0,
        rate_gbps=128.0,
    )

    assert action.shape == (N_ACTION_FEATURES,)
    assert action.dtype == np.float32

    decoded = decode_action(action)
    assert decoded['type'] == ActionType.ADD
    assert decoded['src_node'] == 3
    assert decoded['dst_node'] == 7
    assert decoded['route'] == [3, 5, 7]
    assert decoded['wavelength_slot'] == 42
    assert decoded['modulation'] == Modulation.QPSK
    assert decoded['rate_gbps'] == 128.0

    # REROUTE action
    action2 = encode_action(
        action_type=ActionType.REROUTE,
        target_lp_idx=5,
        route=[1, 4, 6, 9],
    )
    decoded2 = decode_action(action2)
    assert decoded2['type'] == ActionType.REROUTE
    assert decoded2['target_lp_idx'] == 5
    assert decoded2['route'] == [1, 4, 6, 9]

    # POWER_ADJUST action
    action3 = encode_action(
        action_type=ActionType.POWER_ADJUST,
        target_lp_idx=3,
        power_delta_db=1.5,
    )
    decoded3 = decode_action(action3)
    assert decoded3['type'] == ActionType.POWER_ADJUST
    assert decoded3['target_lp_idx'] == 3
    assert abs(decoded3['power_delta_db'] - 1.5) < 1e-5

    print("✓ Action roundtrip OK")


# =========================================================================
# Test 0.4: State encoding with fake data
# =========================================================================

def test_state_encoding():
    """Encode a state with fake lightpaths and results, verify structure."""
    node_ids = [f"n{i}" for i in range(6)]
    link_data = [
        {'id': 'l0', 'src': 'n0', 'dst': 'n1', 'length_km': 80.0,
         'n_spans': 1, 'n_amps': 1},
        {'id': 'l1', 'src': 'n1', 'dst': 'n2', 'length_km': 120.0,
         'n_spans': 2, 'n_amps': 2},
        {'id': 'l2', 'src': 'n0', 'dst': 'n3', 'length_km': 100.0,
         'n_spans': 2, 'n_amps': 2},
        {'id': 'l3', 'src': 'n3', 'dst': 'n2', 'length_km': 90.0,
         'n_spans': 1, 'n_amps': 1},
    ]

    adj = np.zeros((6, 6))
    adj[0, 1] = adj[1, 0] = 1
    adj[1, 2] = adj[2, 1] = 1
    adj[0, 3] = adj[3, 0] = 1
    adj[3, 2] = adj[2, 3] = 1

    # Two lightpaths
    lp1 = LightpathDesc(
        id='lp_0', source='n0', destination='n2',
        route=['n0', 'n1', 'n2'], wavelength_slot=40,
        frequency_thz=193.35, modulation=Modulation.QPSK,
        launch_power_dbm=0.0,
    )
    lp2 = LightpathDesc(
        id='lp_1', source='n0', destination='n2',
        route=['n0', 'n3', 'n2'], wavelength_slot=41,
        frequency_thz=193.40, modulation=Modulation.QAM16,
        launch_power_dbm=1.0,
    )

    # Fake GNPy results
    results = {
        'lp_0': LightpathResult(
            lightpath_id='lp_0', gsnr_db=20.0, osnr_db=25.0,
            signal_dbm=-2.0, ase_dbm=-27.0, nli_dbm=-35.0,
            margin_db=11.5, feasible=True,
        ),
        'lp_1': LightpathResult(
            lightpath_id='lp_1', gsnr_db=16.0, osnr_db=22.0,
            signal_dbm=-1.0, ase_dbm=-23.0, nli_dbm=-30.0,
            margin_db=0.5, feasible=True,
        ),
    }

    state = encode_state(node_ids, link_data, [lp1, lp2], results, adj)

    # Check masks
    assert state['node_mask'][:6].all()
    assert not state['node_mask'][6:].any()
    assert state['link_mask'][:4].all()
    assert not state['link_mask'][4:].any()

    # Check LP features
    assert state['lp_mask'][0] == True
    assert state['lp_mask'][1] == True
    assert state['lp_mask'][2] == False
    assert state['lp_features'][0, 17] == 20.0  # GSNR of lp_0
    assert state['lp_features'][1, 17] == 16.0  # GSNR of lp_1

    # Check spectral occupancy — lp_0 on links l0, l1 at slot 40
    # link l0 is at index 0, l1 at index 1
    assert state['spectral_occupancy'][0, 40] == True  # l0, slot 40
    assert state['spectral_occupancy'][1, 40] == True  # l1, slot 40
    assert state['spectral_occupancy'][0, 41] == False  # l0, slot 41 (lp_1 not here)
    # lp_1 on links l2, l3 at slot 41
    assert state['spectral_occupancy'][2, 41] == True  # l2, slot 41
    assert state['spectral_occupancy'][3, 41] == True  # l3, slot 41

    # Check global features
    assert state['global_features'][0] == 2  # n_active
    assert state['global_features'][2] == 2  # n_feasible
    assert state['global_features'][3] == 0  # n_infeasible
    assert state['global_features'][4] == 0.5  # worst margin

    print("✓ State encoding OK")


# =========================================================================
# Test 0.5: HDF5 write/read roundtrip
# =========================================================================

def test_hdf5_roundtrip():
    """Write a fake episode to HDF5, read it back, verify identity."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create fake data
        n_steps = 5
        states = [create_empty_state(6, 4) for _ in range(n_steps)]
        actions = [encode_action(ActionType.ADD, src_node=0, dst_node=3)
                   for _ in range(n_steps - 1)]

        # Write some non-zero values to verify roundtrip
        states[0]['spectral_occupancy'][0, 10] = True
        states[0]['channel_gsnr'][0, 10] = 22.5
        states[0]['lp_mask'][0] = True
        states[0]['lp_features'][0, 17] = 22.5
        states[0]['global_features'][0] = 1.0

        # Write topology
        topo_adj = np.eye(MAX_NODES, dtype=np.float32)
        topo_nodes = np.zeros((MAX_NODES, N_NODE_FEATURES), dtype=np.float32)
        topo_links = np.zeros((MAX_LINKS, N_LINK_STATIC_FEATURES), dtype=np.float32)
        topo_endpoints = np.zeros((MAX_LINKS, 2), dtype=np.int32)
        node_mask = np.zeros(MAX_NODES, dtype=bool)
        node_mask[:6] = True
        link_mask = np.zeros(MAX_LINKS, dtype=bool)
        link_mask[:4] = True

        # Write
        with HDF5Writer(tmp_path) as writer:
            writer.write_topology(
                topo_adj, topo_nodes, topo_links, topo_endpoints,
                node_mask, link_mask,
                metadata={"n_nodes": 6, "n_links": 4},
            )
            writer.write_episode(
                "ep_test", n_steps, states, actions,
                metadata={"policy": "provisioning", "seed": 42},
            )
            episodes = writer.list_episodes()
            assert "ep_test" in episodes

        # Read back
        with HDF5Reader(tmp_path) as reader:
            topo = reader.get_topology()
            assert topo['adjacency'].shape == (MAX_NODES, MAX_NODES)
            assert topo['node_mask'][:6].all()

            assert reader.get_episode_count() == 1

            # Load full episode
            seq = reader.load_subsequence("ep_test", 0, n_steps)
            assert seq['spectral_occupancy'].shape == (n_steps, MAX_LINKS, MAX_SLOTS)
            assert seq['channel_gsnr'].shape == (n_steps, MAX_LINKS, MAX_SLOTS)
            assert seq['actions'].shape == (n_steps - 1, N_ACTION_FEATURES)

            # Verify values survived roundtrip
            assert seq['spectral_occupancy'][0, 0, 10] == True
            assert seq['spectral_occupancy'][0, 0, 11] == False
            np.testing.assert_almost_equal(
                seq['channel_gsnr'][0, 0, 10], 22.5
            )
            assert seq['lp_mask'][0, 0] == True
            np.testing.assert_almost_equal(
                seq['lp_features'][0, 0, 17], 22.5
            )

            # Load sub-trajectory
            sub = reader.load_subsequence("ep_test", 1, 4)
            assert sub['spectral_occupancy'].shape[0] == 3  # steps 1,2,3
            assert sub['actions'].shape[0] == 2  # 2 actions between 3 states

            info = reader.get_episode_info("ep_test")
            assert info['n_steps'] == n_steps
            assert info['metadata']['policy'] == "provisioning"

        print("✓ HDF5 roundtrip OK")

    finally:
        os.unlink(tmp_path)


# =========================================================================
# Test 0.6: Modulation/capacity constants
# =========================================================================

def test_modulation_constants():
    """Verify modulation thresholds and capacities are consistent."""
    # QPSK needs less GSNR than 16QAM needs less than 64QAM
    assert MOD_THRESHOLDS[Modulation.QPSK] < MOD_THRESHOLDS[Modulation.QAM16]
    assert MOD_THRESHOLDS[Modulation.QAM16] < MOD_THRESHOLDS[Modulation.QAM64]

    # Higher modulation = more capacity
    assert MOD_CAPACITY[Modulation.QPSK] < MOD_CAPACITY[Modulation.QAM16]
    assert MOD_CAPACITY[Modulation.QAM16] < MOD_CAPACITY[Modulation.QAM64]

    # LightpathResult margin computation
    lr = LightpathResult(
        lightpath_id="test", gsnr_db=20.0, osnr_db=25.0,
        signal_dbm=0, ase_dbm=-25, nli_dbm=-30,
    )
    lr.compute_margin(Modulation.QPSK)
    assert lr.margin_db == 20.0 - MOD_THRESHOLDS[Modulation.QPSK]
    assert lr.feasible == True

    lr2 = LightpathResult(
        lightpath_id="test2", gsnr_db=5.0, osnr_db=10.0,
        signal_dbm=0, ase_dbm=-10, nli_dbm=-15,
    )
    lr2.compute_margin(Modulation.QPSK)
    assert lr2.feasible == False  # 5.0 < 8.5

    print("✓ Modulation constants OK")


# =========================================================================
# Test 0.7: LightpathDesc properties
# =========================================================================

def test_lightpath_desc():
    """Verify LightpathDesc computed properties."""
    lp = LightpathDesc(
        id="lp_test", source="n0", destination="n5",
        route=["n0", "n2", "n4", "n5"],
        wavelength_slot=20, frequency_thz=192.35,
        modulation=Modulation.QAM16,
        launch_power_dbm=1.0,
    )
    assert lp.n_hops == 3
    assert lp.capacity_gbps == 256.0
    assert lp.min_gsnr_db == 15.5

    print("✓ LightpathDesc OK")


# =========================================================================
# Run all
# =========================================================================

if __name__ == "__main__":
    tests = [
        test_constants,
        test_empty_state_shapes,
        test_action_roundtrip,
        test_state_encoding,
        test_hdf5_roundtrip,
        test_modulation_constants,
        test_lightpath_desc,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Level 0: {passed} passed, {failed} failed")
    if failed == 0:
        print("All Level 0 tests passed — schemas are correct.")
        print("Next: run Level 1 tests inside GNPy Docker.")
    else:
        print("FIX LEVEL 0 BEFORE PROCEEDING.")
