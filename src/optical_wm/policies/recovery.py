"""
Recovery Policy for World Model Dataset Generation.

Simulates a network disruption followed by restoration.
Two-phase episode:
  Phase 1 (Disruption): Remove LPs to simulate failures (fiber cut, node down)
  Phase 2 (Restoration): Re-provision removed services on alternative paths

Design principles:
  - Starts from a loaded network
  - Phase 1: REMOVE actions (simulates cascading failure on a link/node)
  - Phase 2: ADD actions (re-provision on surviving infrastructure)
  - Shows GSNR improvement during phase 1 (less load), then degradation in phase 2
  - Creates interesting non-monotonic GSNR dynamics
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..core.schemas import (
    LightpathDesc, LightpathResult, Modulation, ActionType,
    encode_action, encode_state, MAX_SLOTS,
)
from ..core.gnpy_wrapper import (
    NetworkEvaluator, TopologySpec, create_lightpath,
)


@dataclass
class RecoveryConfig:
    """Configuration for recovery policy."""
    initial_load_frac: float = 0.5     # start well loaded
    disruption_frac: float = 0.4       # remove 40% of LPs
    max_steps: int = 50
    k_shortest_paths: int = 3
    seed: int = 42


class RecoveryPolicy:
    """
    Expert policy that handles disruption and recovery.

    Trajectory structure:
      Phase 1 — Disruption (~40% of steps):
        - Simulate a link failure: remove all LPs using that link
        - Each REMOVE step drops one LP
        - GSNR improves as load decreases

      Phase 2 — Restoration (~60% of steps):
        - Re-provision dropped services on alternative paths
        - Each ADD step re-adds one LP with a new route
        - GSNR degrades as load increases again
    """

    def __init__(self, evaluator: NetworkEvaluator, spec: TopologySpec,
                 config: RecoveryConfig = None):
        self.evaluator = evaluator
        self.spec = spec
        self.config = config or RecoveryConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.graph = nx.Graph()
        for link in spec.links:
            self.graph.add_edge(
                link['src'], link['dst'],
                weight=link['length_km'],
                link_id=link['id'],
            )

        self.link_lookup = {}
        for link in spec.links:
            self.link_lookup[(link['src'], link['dst'])] = link['id']
            self.link_lookup[(link['dst'], link['src'])] = link['id']

        self.wl_usage: Dict[str, set] = {
            link['id']: set() for link in spec.links
        }

        n = len(spec.node_ids)
        self.adjacency = np.zeros((n, n))
        for link in spec.links:
            i = spec.node_ids.index(link['src'])
            j = spec.node_ids.index(link['dst'])
            self.adjacency[i, j] = self.adjacency[j, i] = 1

    def generate_episode(self) -> dict:
        """Generate a disruption + recovery episode."""
        # Phase 0: Build loaded network
        lightpaths = self._build_loaded_network()

        results, _ = self.evaluator.evaluate_all(lightpaths)
        state = self._encode_current_state(lightpaths, results)

        states = [state]
        actions = []
        step_info = []

        # Phase 1: Disruption — pick a link and remove all LPs on it
        failed_link = self._pick_failure_link(lightpaths)
        affected_lps = self._find_lps_on_link(failed_link, lightpaths)

        # Also randomly remove additional LPs to reach disruption_frac
        n_to_remove = max(
            len(affected_lps),
            int(len(lightpaths) * self.config.disruption_frac),
        )
        # Add random LPs beyond the affected ones
        remaining = [lp for lp in lightpaths if lp not in affected_lps]
        self.rng.shuffle(remaining)
        extra_remove = remaining[:n_to_remove - len(affected_lps)]
        lps_to_remove = affected_lps + extra_remove

        # Record demands for restoration
        demands_to_restore = [
            (lp.source, lp.destination, lp.modulation)
            for lp in lps_to_remove
        ]

        # Execute removals one by one
        for i, lp in enumerate(lps_to_remove):
            if len(lightpaths) <= 2:
                break  # keep minimum for GNPy

            # Remove LP
            lightpaths.remove(lp)
            self._release_wavelength(lp)

            lp_idx = i  # original index doesn't matter much
            src_idx = self.spec.node_ids.index(lp.source)
            dst_idx = self.spec.node_ids.index(lp.destination)

            action = encode_action(
                action_type=ActionType.REMOVE,
                target_lp_idx=lp_idx,
                src_node=src_idx,
                dst_node=dst_idx,
                wavelength_slot=lp.wavelength_slot,
            )

            results, gnpy_ms = self.evaluator.evaluate_all(lightpaths)
            state = self._encode_current_state(lightpaths, results)
            states.append(state)
            actions.append(action)

            step_info.append({
                'step': len(actions) - 1,
                'phase': 'disruption',
                'action_type': 'REMOVE',
                'target_lp': lp.id,
                'failed_link': failed_link,
                'n_lps': len(lightpaths),
                'gnpy_ms': gnpy_ms,
            })

        # Phase 2: Restoration — re-provision on alternative paths
        # Remove failed link from graph for routing
        recovery_graph = self.graph.copy()
        link_data = next(
            (l for l in self.spec.links if l['id'] == failed_link), None
        )
        if link_data:
            try:
                recovery_graph.remove_edge(link_data['src'], link_data['dst'])
            except nx.NetworkXError:
                pass

        restore_counter = 0
        for src, dst, modulation in demands_to_restore:
            if len(actions) >= self.config.max_steps:
                break

            # Find route avoiding failed link
            try:
                route = nx.shortest_path(
                    recovery_graph, src, dst, weight='weight'
                )
            except nx.NetworkXNoPath:
                # Try with k-shortest on full graph
                try:
                    paths = list(nx.shortest_simple_paths(
                        self.graph, src, dst, weight='weight'
                    ))[:self.config.k_shortest_paths]
                    # Filter out paths using failed link
                    route = None
                    for p in paths:
                        if not self._path_uses_link(p, failed_link):
                            route = p
                            break
                    if route is None:
                        route = paths[0]  # fallback to any path
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

            slot = self._first_fit_wavelength(route)
            if slot is None:
                continue

            lp = create_lightpath(
                id=f"lp_restore_{restore_counter:04d}",
                source=src, destination=dst,
                route=route, wavelength_slot=slot,
                modulation=modulation,
                launch_power_dbm=0.0,
            )

            lightpaths.append(lp)
            self._mark_wavelength(lp)
            restore_counter += 1

            src_idx = self.spec.node_ids.index(src)
            dst_idx = self.spec.node_ids.index(dst)
            route_indices = [self.spec.node_ids.index(n) for n in route]

            action = encode_action(
                action_type=ActionType.ADD,
                src_node=src_idx,
                dst_node=dst_idx,
                route=route_indices,
                wavelength_slot=slot,
                modulation=int(modulation),
            )

            results, gnpy_ms = self.evaluator.evaluate_all(lightpaths)
            state = self._encode_current_state(lightpaths, results)
            states.append(state)
            actions.append(action)

            step_info.append({
                'step': len(actions) - 1,
                'phase': 'restoration',
                'action_type': 'ADD',
                'target_lp': lp.id,
                'n_lps': len(lightpaths),
                'gnpy_ms': gnpy_ms,
            })

        # Count phases
        n_remove = sum(1 for s in step_info if s['phase'] == 'disruption')
        n_add = sum(1 for s in step_info if s['phase'] == 'restoration')

        metadata = {
            'policy': 'recovery',
            'seed': self.config.seed,
            'n_steps': len(actions),
            'failed_link': failed_link,
            'n_disruption_steps': n_remove,
            'n_restoration_steps': n_add,
            'demands_affected': len(demands_to_restore),
            'demands_restored': restore_counter,
            'n_nodes': len(self.spec.node_ids),
            'n_links': len(self.spec.links),
        }

        return {
            'states': states,
            'actions': actions,
            'metadata': metadata,
            'step_info': step_info,
        }

    def _build_loaded_network(self) -> List[LightpathDesc]:
        """Create a well-loaded network."""
        target_lps = int(MAX_SLOTS * self.config.initial_load_frac)
        target_lps = max(target_lps, 8)

        lightpaths = []
        lp_counter = 0
        attempts = 0

        while len(lightpaths) < target_lps and attempts < target_lps * 5:
            attempts += 1
            src, dst = self.rng.choice(
                self.spec.node_ids, size=2, replace=False
            )
            try:
                route = nx.shortest_path(
                    self.graph, src, dst, weight='weight'
                )
            except nx.NetworkXNoPath:
                continue

            slot = self._first_fit_wavelength(route)
            if slot is None:
                continue

            lp = create_lightpath(
                id=f"lp_pre_{lp_counter:04d}",
                source=src, destination=dst,
                route=route, wavelength_slot=slot,
                modulation=Modulation.QPSK,
                launch_power_dbm=0.0,
            )
            lightpaths.append(lp)
            self._mark_wavelength(lp)
            lp_counter += 1

        return lightpaths

    def _pick_failure_link(self, lightpaths: List[LightpathDesc]) -> str:
        """Pick a link to fail — prefer links with moderate traffic."""
        link_counts = {}
        for lp in lightpaths:
            for i in range(lp.n_hops):
                lid = self.link_lookup.get((lp.route[i], lp.route[i + 1]))
                if lid:
                    link_counts[lid] = link_counts.get(lid, 0) + 1

        if not link_counts:
            return self.spec.links[0]['id']

        # Pick link with moderate load (not min, not max — interesting scenario)
        sorted_links = sorted(link_counts.items(), key=lambda x: x[1])
        mid_idx = len(sorted_links) // 2
        return sorted_links[mid_idx][0]

    def _find_lps_on_link(
        self, link_id: str, lightpaths: List[LightpathDesc],
    ) -> List[LightpathDesc]:
        """Find all LPs traversing a given link."""
        affected = []
        for lp in lightpaths:
            for i in range(lp.n_hops):
                lid = self.link_lookup.get((lp.route[i], lp.route[i + 1]))
                if lid == link_id:
                    affected.append(lp)
                    break
        return affected

    def _path_uses_link(self, path: List[str], link_id: str) -> bool:
        """Check if a path uses a specific link."""
        for i in range(len(path) - 1):
            lid = self.link_lookup.get((path[i], path[i + 1]))
            if lid == link_id:
                return True
        return False

    # ---- shared helpers ----

    def _first_fit_wavelength(self, route):
        link_ids = []
        for i in range(len(route) - 1):
            lid = self.link_lookup.get((route[i], route[i + 1]))
            if lid is None:
                return None
            link_ids.append(lid)
        for slot in range(MAX_SLOTS):
            if all(slot not in self.wl_usage.get(lid, set())
                   for lid in link_ids):
                return slot
        return None

    def _mark_wavelength(self, lp):
        for i in range(lp.n_hops):
            lid = self.link_lookup.get((lp.route[i], lp.route[i + 1]))
            if lid:
                self.wl_usage.setdefault(lid, set()).add(lp.wavelength_slot)

    def _release_wavelength(self, lp):
        for i in range(lp.n_hops):
            lid = self.link_lookup.get((lp.route[i], lp.route[i + 1]))
            if lid:
                self.wl_usage.get(lid, set()).discard(lp.wavelength_slot)

    def _encode_current_state(self, lightpaths, results):
        return encode_state(
            node_ids=self.spec.node_ids,
            link_data=self.spec.links,
            lightpaths=lightpaths,
            gnpy_results=results,
            adjacency=self.adjacency,
        )
