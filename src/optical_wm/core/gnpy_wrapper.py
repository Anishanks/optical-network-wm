"""
GNPy Wrapper for World Model Dataset Generation.

CRITICAL DESIGN REQUIREMENT:
  All lightpaths must be evaluated TOGETHER so that inter-channel
  NLI (XPM, FWM) is captured. Evaluating each LP independently
  produces physically incorrect results and a useless dataset.

Usage:
  This module requires GNPy installed. Run inside the GNPy Docker:
    docker run -it --rm -v $(pwd):/work telecominfraproject/oopt-gnpy

Architecture:
  1. TopologyBuilder: converts our topology description → GNPy network
  2. NetworkEvaluator: manages spectrum, propagates, extracts results
"""
import time
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .schemas import LightpathDesc, LightpathResult, Modulation, MOD_THRESHOLDS

# GNPy imports — will fail outside Docker
try:
    from gnpy.tools.json_io import load_equipment, load_network
    from gnpy.core.network import build_network
    from gnpy.core.info import (
        SpectralInformation,
        ReferenceCarrier,
        Carrier,
        carriers_to_spectral_information,
        create_input_spectral_information,
    )
    from gnpy.core.utils import watt2dbm, dbm2watt
    import gnpy.core.elements as elements
    import networkx as nx
    GNPY_AVAILABLE = True
except ImportError:
    GNPY_AVAILABLE = False
    # Stub for type hints when GNPy not installed
    def dbm2watt(x):
        return 10 ** (x / 10) * 1e-3
    def watt2dbm(x):
        return 10 * np.log10(x * 1e3) if x > 0 else -99.0


# =========================================================================
# Topology Builder
# =========================================================================

@dataclass
class TopologySpec:
    """Our internal topology representation."""
    node_ids: List[str]
    node_locations: Dict[str, Tuple[float, float]]  # uid → (lat, lon)
    links: List[dict]  # [{id, src, dst, length_km, n_spans, n_amps, fiber_type}]
    equipment_file: Optional[str] = None


class TopologyBuilder:
    """Converts TopologySpec → GNPy network + equipment."""

    def __init__(self, spec: TopologySpec):
        self.spec = spec

    def build(self) -> Tuple:
        """Build GNPy network from our topology spec."""
        if not GNPY_AVAILABLE:
            raise RuntimeError("GNPy not available. Run inside Docker.")

        equipment = self._load_equipment()
        topo_json = self._build_topology_json()

        # Write temp file for GNPy loader
        topo_path = Path("/tmp/wm_gnpy_topology.json")
        with open(topo_path, "w") as f:
            json.dump(topo_json, f, indent=2)

        network = load_network(topo_path, equipment)

        # build_network expects an object with .power, .baud_rate,
        # .slot_width, .nb_channel attributes
        class _RefChannel:
            def __init__(self):
                self.power = dbm2watt(0.0)
                self.baud_rate = 32e9
                self.slot_width = 50e9
                self.nb_channel = 96
        ref_ch = _RefChannel()
        build_network(network, equipment, ref_ch)

        return network, equipment

    def _load_equipment(self):
        """Load GNPy equipment library."""
        import gnpy
        if self.spec.equipment_file:
            eqpt_path = Path(self.spec.equipment_file)
        else:
            eqpt_path = Path(gnpy.__file__).parent / "example-data" / "eqpt_config.json"
        return load_equipment(eqpt_path)

    def _build_topology_json(self) -> dict:
        """Build GNPy-format topology JSON."""
        elems = []
        conns = []

        # Transceivers (one per node)
        for nid in self.spec.node_ids:
            loc = self.spec.node_locations.get(nid, (0.0, 0.0))
            elems.append({
                "uid": nid,
                "type": "Transceiver",
                "metadata": {"location": {
                    "city": nid,
                    "region": "",
                    "latitude": float(loc[0]),
                    "longitude": float(loc[1]),
                }}
            })

        # Fibers + inline amplifiers for each link
        for link in self.spec.links:
            n_spans = link.get('n_spans', max(1, int(math.ceil(
                link['length_km'] / 80
            ))))
            span_length = link['length_km'] / n_spans
            fiber_type = link.get('fiber_type', 'SSMF')

            prev_uid = link['src']

            for si in range(n_spans):
                # Fiber element
                fid = f"fiber_{link['id']}_{si:02d}"
                elems.append({
                    "uid": fid,
                    "type": "Fiber",
                    "type_variety": fiber_type,
                    "params": {
                        "length": span_length,
                        "length_units": "km",
                        "loss_coef": 0.2,
                        "con_in": 0.5 if si == 0 else 0.0,
                        "con_out": 0.5 if si == n_spans - 1 else 0.0,
                    },
                    "metadata": {"location": {
                        "latitude": 0.0, "longitude": 0.0,
                    }}
                })
                conns.append({"from_node": prev_uid, "to_node": fid})

                # Inline amplifier after each span
                aid = f"amp_{link['id']}_{si:02d}"
                elems.append({
                    "uid": aid,
                    "type": "Edfa",
                    "type_variety": "std_medium_gain",
                    "operational": {
                        "gain_target": span_length * 0.2 + 0.5,
                        "tilt_target": 0,
                        "out_voa": 0,
                    },
                    "metadata": {"location": {
                        "latitude": 0.0, "longitude": 0.0,
                    }}
                })
                conns.append({"from_node": fid, "to_node": aid})
                prev_uid = aid

            # Connect last amp to destination
            conns.append({"from_node": prev_uid, "to_node": link['dst']})

        return {"elements": elems, "connections": conns}


# =========================================================================
# Network Evaluator
# =========================================================================

class NetworkEvaluator:
    """
    Evaluates the full network state using GNPy.

    KEY: All lightpaths are evaluated together to capture
    inter-channel NLI (XPM, FWM via Gaussian Noise model).
    """

    def __init__(self, network, equipment, topology_spec: TopologySpec):
        if not GNPY_AVAILABLE:
            raise RuntimeError("GNPy not available.")

        self.network = network
        self.equipment = equipment
        self.spec = topology_spec

        # Build UID → node lookup
        self.uid_to_node = {n.uid: n for n in network.nodes()}

        # Verify transceivers exist
        self.transceivers = {
            n.uid: n for n in network.nodes()
            if isinstance(n, elements.Transceiver)
        }

        # Reference channel params
        self.ref_power_dbm = 0.0
        self.ref_baud_rate = 32e9
        self.ref_slot_width = 50e9
        self.ref_roll_off = 0.15
        self.ref_f_min = 191.35e12
        self.ref_spacing = 50e9

    def evaluate_all(
        self,
        lightpaths: List[LightpathDesc],
    ) -> Tuple[Dict[str, LightpathResult], float]:
        """
        Evaluate ALL lightpaths together.
        Returns per-LP results dict + compute time in ms.
        """
        start = time.time()

        if not lightpaths:
            return {}, 0.0

        # Build the spectrum with ALL active channels
        spectrum = self._build_full_spectrum(lightpaths)

        results = {}
        for lp in lightpaths:
            try:
                result = self._propagate_lightpath(lp, spectrum)
                result.compute_margin(lp.modulation)
                results[lp.id] = result
            except Exception as e:
                print(f"[GNPy] Propagation failed for {lp.id}: {e}")
                results[lp.id] = LightpathResult(
                    lightpath_id=lp.id,
                    gsnr_db=0.0, osnr_db=0.0,
                    signal_dbm=-30.0, ase_dbm=-30.0, nli_dbm=-30.0,
                    margin_db=-99.0, feasible=False,
                )

        elapsed = (time.time() - start) * 1000
        return results, elapsed

    def _build_full_spectrum(self, lightpaths: List[LightpathDesc]) -> dict:
        """Build spectrum dict: frequency_hz → Carrier for ALL active channels."""
        spectrum = {}
        for lp in lightpaths:
            freq_hz = lp.frequency_thz * 1e12
            spectrum[freq_hz] = Carrier(
                delta_pdb=lp.launch_power_dbm - self.ref_power_dbm,
                baud_rate=lp.baud_rate_gbaud * 1e9,
                slot_width=lp.slot_width_ghz * 1e9,
                roll_off=lp.roll_off,
                tx_osnr=40.0,
                tx_power=dbm2watt(lp.launch_power_dbm),
                label=lp.id,
            )
        return spectrum

    def _propagate_lightpath(
        self, lp: LightpathDesc, full_spectrum: dict,
    ) -> LightpathResult:
        """Propagate full spectrum along this LP's route, extract its metrics."""
        path = self._build_element_path(lp.route)
        if path is None or len(path) < 2:
            raise ValueError(f"Could not build path for route {lp.route}")

        # Create SpectralInformation with ALL channels
        si = carriers_to_spectral_information(
            full_spectrum,
            power=dbm2watt(self.ref_power_dbm),
        )

        # Propagate through each element
        for element in path:
            si = element(si)

        # Find this LP's channel index
        target_freq = lp.frequency_thz * 1e12
        ch_idx = np.argmin(np.abs(si.frequency - target_freq))

        freq_error = abs(si.frequency[ch_idx] - target_freq)
        if freq_error > 1e9:
            raise ValueError(
                f"Channel freq mismatch: target={target_freq/1e12:.4f} THz, "
                f"found={si.frequency[ch_idx]/1e12:.4f} THz"
            )

        # Extract per-channel metrics
        signal_w = float(si.signal[ch_idx])
        ase_w = float(si.ase[ch_idx])
        nli_w = float(si.nli[ch_idx])

        signal_dbm = watt2dbm(signal_w) if signal_w > 0 else -99.0
        ase_dbm = watt2dbm(ase_w) if ase_w > 0 else -99.0
        nli_dbm = watt2dbm(nli_w) if nli_w > 1e-30 else -99.0

        osnr_db = 10 * np.log10(signal_w / max(ase_w, 1e-30))
        gsnr_db = 10 * np.log10(signal_w / max(ase_w + nli_w, 1e-30))

        cd = float(si.chromatic_dispersion[ch_idx])
        pmd = float(si.pmd[ch_idx])

        return LightpathResult(
            lightpath_id=lp.id,
            gsnr_db=float(gsnr_db),
            osnr_db=float(osnr_db),
            signal_dbm=float(signal_dbm),
            ase_dbm=float(ase_dbm),
            nli_dbm=float(nli_dbm),
            chromatic_dispersion_ps_nm=cd,
            pmd_ps=pmd,
        )

    def _build_element_path(self, route: List[str]) -> Optional[list]:
        """Convert node route [A, B, C] into GNPy element path."""
        if len(route) < 2:
            return None

        path = []
        src = self.uid_to_node.get(route[0])
        if src is None:
            return None
        path.append(src)

        for i in range(len(route) - 1):
            hop_src = route[i]
            hop_dst = route[i + 1]

            link = self._find_link(hop_src, hop_dst)
            if link is None:
                return None

            reversed_link = (link['src'] != hop_src)
            n_spans = link.get('n_spans', max(1, int(math.ceil(
                link['length_km'] / 80
            ))))

            span_indices = list(range(n_spans))
            if reversed_link:
                span_indices = list(reversed(span_indices))

            for si in span_indices:
                fid = f"fiber_{link['id']}_{si:02d}"
                aid = f"amp_{link['id']}_{si:02d}"

                fiber_node = self.uid_to_node.get(fid)
                amp_node = self.uid_to_node.get(aid)

                if fiber_node:
                    path.append(fiber_node)
                if amp_node:
                    path.append(amp_node)

            dst = self.uid_to_node.get(hop_dst)
            if dst:
                path.append(dst)

        return path if len(path) >= 2 else None

    def _find_link(self, src: str, dst: str) -> Optional[dict]:
        """Find link connecting src and dst (either direction)."""
        for link in self.spec.links:
            if (link['src'] == src and link['dst'] == dst) or \
               (link['src'] == dst and link['dst'] == src):
                return link
        return None


# =========================================================================
# Helpers
# =========================================================================

def build_test_topology(n_nodes: int, connectivity: float = 0.3,
                        seed: int = 42) -> TopologySpec:
    """Create a random mesh topology for testing."""
    rng = np.random.default_rng(seed)

    cols = int(math.ceil(math.sqrt(n_nodes)))
    spacing = 100.0

    node_ids = [f"node_{i:03d}" for i in range(n_nodes)]
    locations = {}
    for i in range(n_nodes):
        row, col = divmod(i, cols)
        lat = row * spacing + rng.uniform(-20, 20)
        lon = col * spacing + rng.uniform(-20, 20)
        locations[node_ids[i]] = (lat, lon)

    # Build MST + extra edges
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            li, lo = locations[node_ids[i]]
            lj, lo2 = locations[node_ids[j]]
            d = math.sqrt((li - lj)**2 + (lo - lo2)**2)
            dist[i, j] = dist[j, i] = d

    edges_sorted = sorted(
        [(dist[i, j], i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
    )
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            return True
        return False

    selected_edges = []
    for d, i, j in edges_sorted:
        if union(i, j):
            selected_edges.append((i, j, d))

    target_edges = int(n_nodes * (n_nodes - 1) / 2 * connectivity)
    remaining = [(d, i, j) for d, i, j in edges_sorted
                 if (i, j, d) not in [(e[0], e[1], e[2]) for e in selected_edges]]
    rng.shuffle(remaining)
    for d, i, j in remaining:
        if len(selected_edges) >= target_edges:
            break
        selected_edges.append((i, j, d))

    links = []
    for idx, (i, j, d) in enumerate(selected_edges):
        n_spans = max(1, int(math.ceil(d / 80)))
        links.append({
            'id': f"link_{idx:03d}",
            'src': node_ids[i],
            'dst': node_ids[j],
            'length_km': float(d),
            'n_spans': n_spans,
            'n_amps': n_spans,
            'fiber_type': 'SSMF',
        })

    return TopologySpec(
        node_ids=node_ids,
        node_locations=locations,
        links=links,
    )


def create_evaluator(spec: TopologySpec) -> Tuple[NetworkEvaluator, TopologySpec]:
    """Build everything needed for evaluation."""
    builder = TopologyBuilder(spec)
    network, equipment = builder.build()
    evaluator = NetworkEvaluator(network, equipment, spec)
    return evaluator, spec


def create_lightpath(
    id: str,
    source: str,
    destination: str,
    route: List[str],
    wavelength_slot: int,
    modulation: Modulation = Modulation.QPSK,
    launch_power_dbm: float = 0.0,
    ref_frequency_thz: float = 191.35,
    spacing_ghz: float = 50.0,
) -> LightpathDesc:
    """Helper to create a LightpathDesc with correct frequency."""
    frequency_thz = ref_frequency_thz + wavelength_slot * spacing_ghz / 1000
    return LightpathDesc(
        id=id,
        source=source,
        destination=destination,
        route=route,
        wavelength_slot=wavelength_slot,
        frequency_thz=frequency_thz,
        modulation=modulation,
        launch_power_dbm=launch_power_dbm,
    )
