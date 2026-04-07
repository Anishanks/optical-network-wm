"""
Data schemas for Optical Network World Model dataset.

Design principles:
  - Fixed-size tensors everywhere (no ragged arrays)
  - All features are float32 or bool for GPU compatibility
  - Padding with zeros + mask arrays for variable-length data
  - Dimensions chosen to match real GNPy output
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum


# =========================================================================
# Constants
# =========================================================================

MAX_NODES = 20          # max nodes in any topology
MAX_LINKS = 40          # max links (edges) in any topology
MAX_SLOTS = 80          # wavelength slots per link (C-band, 50 GHz spacing)
MAX_LIGHTPATHS = 80     # max simultaneous lightpaths
MAX_HOPS = 8            # max hops in a lightpath route

# Feature dimensions
N_LINK_STATIC_FEATURES = 3    # length_km, n_spans, n_amplifiers
N_LINK_SPECTRAL_FEATURES = 5  # per slot: occupancy, gsnr, power, ase, nli
N_NODE_FEATURES = 5            # degree, n_add_drop, n_through, is_src, is_dst
N_LP_FEATURES = 20             # see encode_lightpath_features()
N_GLOBAL_FEATURES = 8          # see encode_global_features()
N_ACTION_FEATURES = 20         # see encode_action()


# =========================================================================
# Enums
# =========================================================================

class ActionType(IntEnum):
    ADD = 0
    REMOVE = 1
    REROUTE = 2
    POWER_ADJUST = 3
    MOD_CHANGE = 4


class Modulation(IntEnum):
    QPSK = 0
    QAM16 = 1
    QAM64 = 2


# Minimum GSNR required for each modulation (dB)
MOD_THRESHOLDS = {
    Modulation.QPSK: 8.5,
    Modulation.QAM16: 15.5,
    Modulation.QAM64: 22.5,
}

# Capacity per modulation at 32 Gbaud (Gbps)
MOD_CAPACITY = {
    Modulation.QPSK: 128.0,    # 32 * 2 * 2
    Modulation.QAM16: 256.0,   # 32 * 4 * 2
    Modulation.QAM64: 384.0,   # 32 * 6 * 2
}


# =========================================================================
# Lightpath descriptor (used during generation, not stored directly)
# =========================================================================

@dataclass
class LightpathDesc:
    """Describes a lightpath for GNPy provisioning."""
    id: str
    source: str               # node uid
    destination: str          # node uid
    route: List[str]          # list of node uids
    wavelength_slot: int      # 0-79
    frequency_thz: float      # center frequency
    modulation: Modulation
    launch_power_dbm: float
    baud_rate_gbaud: float = 32.0
    slot_width_ghz: float = 37.5
    roll_off: float = 0.15

    @property
    def capacity_gbps(self) -> float:
        return MOD_CAPACITY[self.modulation]

    @property
    def n_hops(self) -> int:
        return len(self.route) - 1

    @property
    def min_gsnr_db(self) -> float:
        return MOD_THRESHOLDS[self.modulation]


# =========================================================================
# GNPy result (per lightpath)
# =========================================================================

@dataclass
class LightpathResult:
    """GNPy output for one lightpath."""
    lightpath_id: str
    gsnr_db: float
    osnr_db: float
    signal_dbm: float
    ase_dbm: float
    nli_dbm: float
    chromatic_dispersion_ps_nm: float = 0.0
    pmd_ps: float = 0.0
    margin_db: float = 0.0
    feasible: bool = True

    def compute_margin(self, modulation: Modulation):
        self.margin_db = self.gsnr_db - MOD_THRESHOLDS[modulation]
        self.feasible = self.margin_db > 0
        return self


# =========================================================================
# State encoding functions
# =========================================================================

def create_empty_state(n_nodes: int, n_links: int) -> Dict[str, np.ndarray]:
    """Create a zeroed state dict with correct shapes."""
    return {
        # Topology (static within episode)
        'adjacency': np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32),
        'node_features': np.zeros((MAX_NODES, N_NODE_FEATURES), dtype=np.float32),
        'link_static': np.zeros((MAX_LINKS, N_LINK_STATIC_FEATURES), dtype=np.float32),
        'link_endpoints': np.zeros((MAX_LINKS, 2), dtype=np.int32),

        # Masks (which nodes/links are real vs padding)
        'node_mask': np.zeros(MAX_NODES, dtype=bool),
        'link_mask': np.zeros(MAX_LINKS, dtype=bool),

        # Spectral state (dynamic — changes every step)
        'spectral_occupancy': np.zeros((MAX_LINKS, MAX_SLOTS), dtype=bool),
        'channel_gsnr': np.zeros((MAX_LINKS, MAX_SLOTS), dtype=np.float32),
        'channel_power': np.zeros((MAX_LINKS, MAX_SLOTS), dtype=np.float32),
        'channel_ase': np.zeros((MAX_LINKS, MAX_SLOTS), dtype=np.float32),
        'channel_nli': np.zeros((MAX_LINKS, MAX_SLOTS), dtype=np.float32),

        # Per-lightpath features
        'lp_features': np.zeros((MAX_LIGHTPATHS, N_LP_FEATURES), dtype=np.float32),
        'lp_mask': np.zeros(MAX_LIGHTPATHS, dtype=bool),

        # Global summary
        'global_features': np.zeros(N_GLOBAL_FEATURES, dtype=np.float32),
    }


def encode_state(
    node_ids: List[str],
    link_data: List[dict],  # [{src, dst, length_km, n_spans, n_amps}]
    lightpaths: List[LightpathDesc],
    gnpy_results: Dict[str, LightpathResult],
    adjacency: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Encode the full network state into fixed-size tensors.

    Args:
        node_ids: list of node UIDs
        link_data: list of dicts with link properties
        lightpaths: active lightpaths
        gnpy_results: GNPy output per lightpath ID
        adjacency: [n_nodes, n_nodes] adjacency matrix
    """
    state = create_empty_state(len(node_ids), len(link_data))
    n_nodes = len(node_ids)
    n_links = len(link_data)
    node_idx = {uid: i for i, uid in enumerate(node_ids)}

    # --- Topology (static) ---
    state['adjacency'][:n_nodes, :n_nodes] = adjacency
    state['node_mask'][:n_nodes] = True
    state['link_mask'][:n_links] = True

    for i, link in enumerate(link_data):
        src_i = node_idx[link['src']]
        dst_i = node_idx[link['dst']]
        state['link_endpoints'][i] = [src_i, dst_i]
        state['link_static'][i] = [
            link['length_km'],
            link['n_spans'],
            link['n_amps'],
        ]

    # --- Node features ---
    # Compute degree, add/drop counts
    for i, nid in enumerate(node_ids):
        degree = int(adjacency[i].sum())
        n_add = sum(1 for lp in lightpaths if lp.source == nid)
        n_drop = sum(1 for lp in lightpaths if lp.destination == nid)
        n_through = sum(
            1 for lp in lightpaths
            if nid in lp.route[1:-1]  # intermediate nodes only
        )
        state['node_features'][i] = [
            degree,
            n_add + n_drop,
            n_through,
            float(n_add > 0),
            float(n_drop > 0),
        ]

    # --- Spectral state (per link, per slot) ---
    # Build link lookup: (src, dst) → link_index
    link_lookup = {}
    for i, link in enumerate(link_data):
        link_lookup[(link['src'], link['dst'])] = i
        link_lookup[(link['dst'], link['src'])] = i  # bidirectional

    for lp in lightpaths:
        result = gnpy_results.get(lp.id)
        if result is None:
            continue

        slot = lp.wavelength_slot
        # Mark this LP's slot as occupied on every link it traverses
        for hop in range(lp.n_hops):
            link_key = (lp.route[hop], lp.route[hop + 1])
            link_idx = link_lookup.get(link_key)
            if link_idx is not None and 0 <= slot < MAX_SLOTS:
                state['spectral_occupancy'][link_idx, slot] = True
                state['channel_gsnr'][link_idx, slot] = result.gsnr_db
                state['channel_power'][link_idx, slot] = result.signal_dbm
                state['channel_ase'][link_idx, slot] = result.ase_dbm
                state['channel_nli'][link_idx, slot] = result.nli_dbm

    # --- Per-lightpath features ---
    for i, lp in enumerate(lightpaths):
        if i >= MAX_LIGHTPATHS:
            break
        result = gnpy_results.get(lp.id)
        if result is None:
            continue

        state['lp_mask'][i] = True

        # Encode route as node indices, padded
        route_indices = [node_idx.get(n, 0) for n in lp.route[:MAX_HOPS]]
        route_padded = route_indices + [0] * (MAX_HOPS - len(route_indices))

        state['lp_features'][i] = [
            node_idx.get(lp.source, 0),       # 0: src node index
            node_idx.get(lp.destination, 0),   # 1: dst node index
            lp.n_hops,                         # 2: route length
            *route_padded,                     # 3-10: route node indices
            lp.wavelength_slot,                # 11: wavelength slot
            lp.frequency_thz,                  # 12: center frequency
            float(lp.modulation),              # 13: modulation enum
            lp.launch_power_dbm,               # 14: launch power
            lp.baud_rate_gbaud,                # 15: baud rate
            lp.capacity_gbps,                  # 16: capacity
            result.gsnr_db,                    # 17: GSNR
            result.margin_db,                  # 18: margin
            float(result.feasible),            # 19: feasible
        ]

    # --- Global features ---
    n_active = sum(state['lp_mask'])
    margins = [
        gnpy_results[lp.id].margin_db
        for lp in lightpaths
        if lp.id in gnpy_results
    ]
    gsnrs = [
        gnpy_results[lp.id].gsnr_db
        for lp in lightpaths
        if lp.id in gnpy_results
    ]

    state['global_features'] = np.array([
        n_active,                                              # 0
        sum(lp.capacity_gbps for lp in lightpaths) / 1000,    # 1: total Tbps
        sum(1 for m in margins if m > 0),                      # 2: n feasible
        sum(1 for m in margins if m <= 0),                     # 3: n infeasible
        min(margins) if margins else 0.0,                      # 4: worst margin
        float(np.mean(margins)) if margins else 0.0,           # 5: avg margin
        float(np.mean(state['spectral_occupancy'][
            state['link_mask']
        ].mean())),                                            # 6: avg utilization
        float(state['spectral_occupancy'][
            state['link_mask']
        ].sum(axis=1).max() / MAX_SLOTS) if n_active > 0 else 0.0,
                                                               # 7: max link util
    ], dtype=np.float32)

    return state


# =========================================================================
# Action encoding
# =========================================================================

def encode_action(
    action_type: ActionType,
    target_lp_idx: int = -1,
    src_node: int = -1,
    dst_node: int = -1,
    route: Optional[List[int]] = None,
    wavelength_slot: int = -1,
    modulation: int = 0,
    power_delta_db: float = 0.0,
    rate_gbps: float = 0.0,
) -> np.ndarray:
    """Encode an action as a fixed-size float32 vector."""
    route_padded = [0] * MAX_HOPS
    if route is not None:
        for i, r in enumerate(route[:MAX_HOPS]):
            route_padded[i] = r

    action = np.zeros(N_ACTION_FEATURES, dtype=np.float32)
    action[0] = float(action_type)
    action[1] = float(target_lp_idx)
    action[2] = float(src_node)
    action[3] = float(dst_node)
    action[4:4 + MAX_HOPS] = route_padded
    action[12] = float(wavelength_slot)
    action[13] = float(modulation)
    action[14] = power_delta_db
    action[15] = rate_gbps
    # 16-19: reserved for future use
    return action


def decode_action(action: np.ndarray) -> dict:
    """Decode a fixed-size action vector back to a dict."""
    return {
        'type': ActionType(int(action[0])),
        'target_lp_idx': int(action[1]),
        'src_node': int(action[2]),
        'dst_node': int(action[3]),
        'route': [int(action[4 + i]) for i in range(MAX_HOPS)
                  if action[4 + i] > 0],
        'wavelength_slot': int(action[12]),
        'modulation': Modulation(int(action[13])),
        'power_delta_db': float(action[14]),
        'rate_gbps': float(action[15]),
    }
