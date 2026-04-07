"""
Load Balancing Policy for World Model Dataset Generation.

Simulates an operator rebalancing traffic across the network to equalize
link utilization. Starts from an intentionally unbalanced loaded network.
Each step picks an LP on the most congested link and reroutes it
to a less congested path.

Design principles:
  - Starts with unbalanced load (some links heavily loaded, others empty)
  - Every step targets the most congested link
  - Actions are REROUTE type (same src-dst, different path)
  - Trajectory shows utilization variance decreasing over time
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from ..core.schemas import (
    LightpathDesc, LightpathResult, Modulation, ActionType,
    encode_action, encode_state, MAX_SLOTS,
)
from ..core.gnpy_wrapper import (
    NetworkEvaluator, TopologySpec, create_lightpath,
)


@dataclass
class LoadBalanceConfig:
    """Configuration for load balancing policy."""
    initial_load_frac: float = 0.35
    max_steps: int = 40
    k_shortest_paths: int = 4
    seed: int = 42


class LoadBalancePolicy:
    """
    Expert policy that balances load across network links.

    Trajectory structure:
      1. Start with intentionally unbalanced load
      2. Each step: find most congested link, pick an LP on it
      3. Reroute that LP to a path using less congested links
      4. Stop when utilization variance is low or max_steps reached
    """

    def __init__(self, evaluator: NetworkEvaluator, spec: TopologySpec,
                 config: LoadBalanceConfig = None):
        self.evaluator = evaluator
        self.spec = spec
        self.config = config or LoadBalanceConfig()
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
        """Generate a load balancing episode."""
        lightpaths = self._build_unbalanced_network()

        results, _ = self.evaluator.evaluate_all(lightpaths)
        state = self._encode_current_state(lightpaths, results)

        states = [state]
        actions = []
        step_info = []

        for step in range(self.config.max_steps):
            if not lightpaths:
                break

            # Compute per-link utilization
            link_util = self._compute_link_utilization(lightpaths)
            util_values = list(link_util.values())
            util_std = float(np.std(util_values)) if util_values else 0

            # Find most congested link
            most_congested = max(link_util, key=link_util.get)
            congestion = link_util[most_congested]

            # Find an LP on that link to reroute
            target_lp = self._find_lp_on_link(most_congested, lightpaths)
            if target_lp is None:
                break

            # Try to reroute to less congested path
            action, success = self._try_reroute_for_balance(
                target_lp, lightpaths, link_util, step
            )

            results, gnpy_ms = self.evaluator.evaluate_all(lightpaths)
            state = self._encode_current_state(lightpaths, results)
            states.append(state)
            actions.append(action)

            new_util = self._compute_link_utilization(lightpaths)
            new_std = float(np.std(list(new_util.values()))) if new_util else 0

            step_info.append({
                'step': step,
                'target_lp': target_lp.id,
                'congested_link': most_congested,
                'congestion': congestion,
                'util_std_before': util_std,
                'util_std_after': new_std,
                'success': success,
                'gnpy_ms': gnpy_ms,
            })

        # Compute utilization variance trajectory
        util_stds = []
        for s in states:
            occ = s['spectral_occupancy']
            per_link = occ.mean(axis=1) if occ.ndim > 1 else occ
            util_stds.append(float(np.std(per_link)))

        metadata = {
            'policy': 'load_balancing',
            'seed': self.config.seed,
            'n_steps': len(actions),
            'initial_util_std': util_stds[0] if util_stds else 0,
            'final_util_std': util_stds[-1] if util_stds else 0,
            'n_nodes': len(self.spec.node_ids),
            'n_links': len(self.spec.links),
        }

        return {
            'states': states,
            'actions': actions,
            'metadata': metadata,
            'step_info': step_info,
        }

    def _build_unbalanced_network(self) -> List[LightpathDesc]:
        """Create intentionally unbalanced load — concentrate on few links."""
        target_lps = int(MAX_SLOTS * self.config.initial_load_frac)
        target_lps = max(target_lps, 5)

        # Pick 2-3 "hot" src-dst pairs to create imbalance
        nodes = self.spec.node_ids
        n_hot_pairs = min(3, len(nodes) // 2)
        hot_pairs = []
        for _ in range(n_hot_pairs):
            src, dst = self.rng.choice(nodes, size=2, replace=False)
            hot_pairs.append((src, dst))

        lightpaths = []
        lp_counter = 0
        attempts = 0

        while len(lightpaths) < target_lps and attempts < target_lps * 5:
            attempts += 1

            # 70% hot pairs, 30% random — creates imbalance
            if self.rng.random() < 0.7 and hot_pairs:
                src, dst = hot_pairs[self.rng.integers(len(hot_pairs))]
            else:
                src, dst = self.rng.choice(nodes, size=2, replace=False)

            try:
                # Always use shortest path → concentrates on same links
                route = nx.shortest_path(
                    self.graph, src, dst, weight='weight'
                )
            except nx.NetworkXNoPath:
                continue

            slot = self._first_fit_wavelength(route)
            if slot is None:
                continue

            lp = create_lightpath(
                id=f"lp_lb_{lp_counter:04d}",
                source=src, destination=dst,
                route=route, wavelength_slot=slot,
                modulation=Modulation.QPSK,
                launch_power_dbm=0.0,
            )
            lightpaths.append(lp)
            self._mark_wavelength(lp)
            lp_counter += 1

        return lightpaths

    def _compute_link_utilization(
        self, lightpaths: List[LightpathDesc],
    ) -> Dict[str, float]:
        """Compute utilization (fraction of used slots) per link."""
        util = {}
        for link in self.spec.links:
            lid = link['id']
            n_used = len(self.wl_usage.get(lid, set()))
            util[lid] = n_used / MAX_SLOTS
        return util

    def _find_lp_on_link(
        self, link_id: str, lightpaths: List[LightpathDesc],
    ) -> Optional[LightpathDesc]:
        """Find an LP traversing the given link, prefer multi-hop LPs."""
        candidates = []
        for lp in lightpaths:
            for i in range(lp.n_hops):
                lid = self.link_lookup.get((lp.route[i], lp.route[i + 1]))
                if lid == link_id:
                    candidates.append(lp)
                    break

        if not candidates:
            return None

        # Prefer LPs with more hops (more benefit from rerouting)
        candidates.sort(key=lambda lp: -lp.n_hops)
        return candidates[0]

    def _try_reroute_for_balance(
        self, lp: LightpathDesc, lightpaths: List[LightpathDesc],
        link_util: Dict[str, float], step: int,
    ) -> Tuple[np.ndarray, bool]:
        """Reroute LP to a path with lower max-link utilization."""
        try:
            paths = list(nx.shortest_simple_paths(
                self.graph, lp.source, lp.destination, weight='weight'
            ))[:self.config.k_shortest_paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self._make_noop_action(lp, lightpaths), False

        best_path = None
        best_max_util = float('inf')
        best_slot = None

        for path in paths:
            if path == lp.route:
                continue

            # Temporarily release old wavelength to check availability
            self._release_wavelength(lp)
            slot = self._first_fit_wavelength(path)
            self._mark_wavelength(lp)  # restore

            if slot is None:
                continue

            # Compute max utilization along this candidate path
            max_util = 0
            for i in range(len(path) - 1):
                lid = self.link_lookup.get((path[i], path[i + 1]))
                if lid:
                    max_util = max(max_util, link_util.get(lid, 0))

            if max_util < best_max_util:
                best_max_util = max_util
                best_path = path
                best_slot = slot

        if best_path is not None:
            # Apply the reroute
            self._release_wavelength(lp)
            lp.route = best_path
            lp.wavelength_slot = best_slot
            lp.frequency_thz = 191.35 + best_slot * 50.0 / 1000
            self._mark_wavelength(lp)

            lp_idx = next(
                (i for i, l in enumerate(lightpaths) if l.id == lp.id), -1
            )
            src_idx = self.spec.node_ids.index(lp.source)
            dst_idx = self.spec.node_ids.index(lp.destination)
            route_indices = [self.spec.node_ids.index(n) for n in best_path]

            action = encode_action(
                action_type=ActionType.REROUTE,
                target_lp_idx=lp_idx,
                src_node=src_idx,
                dst_node=dst_idx,
                route=route_indices,
                wavelength_slot=best_slot,
            )
            return action, True

        return self._make_noop_action(lp, lightpaths), False

    def _make_noop_action(
        self, lp: LightpathDesc, lightpaths: List[LightpathDesc],
    ) -> np.ndarray:
        """Create a reroute action that keeps the same route (no-op)."""
        lp_idx = next(
            (i for i, l in enumerate(lightpaths) if l.id == lp.id), -1
        )
        src_idx = self.spec.node_ids.index(lp.source)
        dst_idx = self.spec.node_ids.index(lp.destination)
        route_indices = [self.spec.node_ids.index(n) for n in lp.route]

        return encode_action(
            action_type=ActionType.REROUTE,
            target_lp_idx=lp_idx,
            src_node=src_idx,
            dst_node=dst_idx,
            route=route_indices,
            wavelength_slot=lp.wavelength_slot,
        )

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
