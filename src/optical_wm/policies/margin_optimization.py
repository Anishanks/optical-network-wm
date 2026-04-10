"""
Margin Optimization Policy for World Model Dataset Generation.

Simulates an operator improving network margins on an already-loaded network.
Each step picks the LP with worst margin and tries to improve it via:
  1. Power adjustment (+/- 0.5 dB increments)
  2. Modulation downgrade (e.g. QAM16 → QPSK for more margin)
  3. Reroute to a shorter/less congested path

Design principles:
  - Starts from a loaded network (output of provisioning)
  - Every step targets a specific LP with a clear optimization goal
  - Actions mix POWER_ADJUST, MOD_CHANGE, REROUTE types
  - Trajectory shows margins improving over time
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..core.schemas import (
    LightpathDesc, LightpathResult, Modulation, ActionType,
    encode_action, encode_state, MAX_SLOTS, MOD_THRESHOLDS,
)
from ..core.gnpy_wrapper import (
    NetworkEvaluator, TopologySpec, create_lightpath,
)


@dataclass
class MarginOptConfig:
    """Configuration for margin optimization policy."""
    initial_load_frac: float = 0.4     # start with 40% load
    max_steps: int = 40
    power_step_db: float = 0.5         # power adjustment granularity
    power_min_db: float = -3.0
    power_max_db: float = 3.0
    k_shortest_paths: int = 3
    seed: int = 42


class MarginOptPolicy:
    """
    Expert policy that optimizes margins on a loaded network.

    Trajectory structure:
      1. Start with ~40% network load (random provisioning)
      2. Each step: pick worst-margin LP, try to improve it
      3. Strategy priority: power adjust → modulation change → reroute
      4. Stop when max_steps reached or all margins > threshold
    """

    def __init__(self, evaluator: NetworkEvaluator, spec: TopologySpec,
                 config: MarginOptConfig = None):
        self.evaluator = evaluator
        self.spec = spec
        self.config = config or MarginOptConfig()
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
        """Generate a margin optimization episode."""
        # Phase 0: Build a loaded network
        lightpaths = self._build_loaded_network()

        results, _ = self.evaluator.evaluate_all(lightpaths)
        state = self._encode_current_state(lightpaths, results)

        states = [state]
        actions = []
        step_info = []

        for step in range(self.config.max_steps):
            if not lightpaths:
                break

            # Find LP with worst margin
            worst_lp, worst_result = self._find_worst_margin_lp(
                lightpaths, results
            )
            if worst_lp is None:
                break

            # Try optimization strategies in priority order
            action, success, strategy = self._optimize_lp(
                worst_lp, worst_result, lightpaths, step
            )

            # Re-evaluate entire network
            results, gnpy_ms = self.evaluator.evaluate_all(lightpaths)

            state = self._encode_current_state(lightpaths, results)
            states.append(state)
            actions.append(action)

            step_info.append({
                'step': step,
                'target_lp': worst_lp.id,
                'strategy': strategy,
                'success': success,
                'n_lps': len(lightpaths),
                'gnpy_ms': gnpy_ms,
            })

        margins = [s['global_features'][4] for s in states]

        metadata = {
            'policy': 'margin_optimization',
            'seed': self.config.seed,
            'n_steps': len(actions),
            'initial_worst_margin': float(margins[0]) if margins else 0,
            'final_worst_margin': float(margins[-1]) if margins else 0,
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
        """Create a loaded network to optimize.

        Load is jittered ±50% to diversify starting conditions.
        Uses suboptimal power/modulation to create margin issues.
        """
        base = self.config.initial_load_frac
        jittered = self.rng.uniform(base * 0.5, base * 1.5)
        target_lps = int(MAX_SLOTS * jittered)
        target_lps = max(target_lps, 5)

        lightpaths = []
        lp_counter = 0
        attempts = 0

        while len(lightpaths) < target_lps and attempts < target_lps * 5:
            attempts += 1
            src, dst = self.rng.choice(
                self.spec.node_ids, size=2, replace=False
            )
            try:
                route = nx.shortest_path(self.graph, src, dst, weight='weight')
            except nx.NetworkXNoPath:
                continue

            slot = self._first_fit_wavelength(route)
            if slot is None:
                continue

            # Intentionally use suboptimal power to create margin issues
            power = self.rng.uniform(-2.0, 2.0)
            modulation = self.rng.choice(
                [Modulation.QPSK, Modulation.QAM16, Modulation.QAM16]
            )

            lp = create_lightpath(
                id=f"lp_load_{lp_counter:04d}",
                source=src, destination=dst,
                route=route, wavelength_slot=slot,
                modulation=modulation,
                launch_power_dbm=float(power),
            )
            lightpaths.append(lp)
            self._mark_wavelength(lp)
            lp_counter += 1

        return lightpaths

    def _find_worst_margin_lp(
        self, lightpaths: List[LightpathDesc],
        results: Dict[str, LightpathResult],
        n_candidates: int = 3,
    ) -> Tuple[Optional[LightpathDesc], Optional[LightpathResult]]:
        """Pick a target LP randomly among the n worst-margin LPs.

        This avoids getting stuck on a single incorrigible LP and
        diversifies which LPs get optimized across steps.
        """
        scored = []
        for lp in lightpaths:
            r = results.get(lp.id)
            if r is None or r.gsnr_db <= 0:
                continue
            threshold = MOD_THRESHOLDS.get(lp.modulation, 8.5)
            margin = r.gsnr_db - threshold
            scored.append((margin, lp, r))

        if not scored:
            return None, None

        # Sort by margin ascending (worst first), pick among top-n
        scored.sort(key=lambda x: x[0])
        pick = min(n_candidates, len(scored))
        idx = int(self.rng.integers(pick))
        _, chosen_lp, chosen_result = scored[idx]
        return chosen_lp, chosen_result

    def _optimize_lp(
        self, lp: LightpathDesc, result: LightpathResult,
        lightpaths: List[LightpathDesc], step: int,
    ) -> Tuple[np.ndarray, bool, str]:
        """Try to optimize a specific LP. Strategy chosen by margin severity.

        Expert logic:
          margin < -3 dB  → big fix needed: reroute (50%) or mod downgrade (40%)
          -3 ≤ margin < 0 → moderate fix: mod downgrade (40%) or power (40%)
          margin ≥ 0       → fine-tuning: power adjust (70%)

        A 10% random noise always applies to inject diversity.
        """
        threshold = MOD_THRESHOLDS.get(lp.modulation, 8.5)
        margin = result.gsnr_db - threshold

        roll = self.rng.random()

        if margin < -3.0:
            # Severe — need structural change
            if roll < 0.10:
                # noise: power adjust (unlikely to help but adds diversity)
                action, success = self._try_power_adjust(lp, lightpaths, step)
                return action, success, 'power_adjust'
            elif roll < 0.50:
                return self._try_reroute(lp, lightpaths, step)
            else:
                return self._try_mod_change(lp, lightpaths, step)

        elif margin < 0.0:
            # Moderate — could go either way
            if roll < 0.10:
                return self._try_reroute(lp, lightpaths, step)
            elif roll < 0.50:
                return self._try_mod_change(lp, lightpaths, step)
            else:
                action, success = self._try_power_adjust(lp, lightpaths, step)
                return action, success, 'power_adjust'

        else:
            # Positive margin — fine-tuning
            if roll < 0.70:
                action, success = self._try_power_adjust(lp, lightpaths, step)
                return action, success, 'power_adjust'
            elif roll < 0.90:
                return self._try_mod_change(lp, lightpaths, step)
            else:
                return self._try_reroute(lp, lightpaths, step)

    def _try_power_adjust(
        self, lp: LightpathDesc, lightpaths: List[LightpathDesc], step: int,
    ) -> Tuple[np.ndarray, bool]:
        """Adjust LP launch power.

        Biased toward increasing power when margin is negative (more signal),
        and toward decreasing when margin is positive (reduce NLI on neighbors).
        10% random inversion for diversity.
        """
        threshold = MOD_THRESHOLDS.get(lp.modulation, 8.5)
        # Use last known GSNR approximation from launch power context
        if self.rng.random() < 0.10:
            # Random direction (noise)
            delta = self.rng.choice([-1.0, -0.5, 0.5, 1.0])
        else:
            # Expert direction: positive delta when we need more signal
            magnitude = self.rng.choice([0.5, 1.0])
            delta = magnitude  # default: increase power
            # If power is already high, try decreasing (reduce NLI on neighbors)
            if lp.launch_power_dbm >= 1.5:
                delta = -magnitude

        new_power = np.clip(
            lp.launch_power_dbm + delta,
            self.config.power_min_db,
            self.config.power_max_db,
        )
        actual_delta = new_power - lp.launch_power_dbm

        lp.launch_power_dbm = float(new_power)

        lp_idx = next(
            (i for i, l in enumerate(lightpaths) if l.id == lp.id), -1
        )
        src_idx = self.spec.node_ids.index(lp.source)
        dst_idx = self.spec.node_ids.index(lp.destination)

        action = encode_action(
            action_type=ActionType.POWER_ADJUST,
            target_lp_idx=lp_idx,
            src_node=src_idx,
            dst_node=dst_idx,
            wavelength_slot=lp.wavelength_slot,
            power_delta_db=float(actual_delta),
        )
        return action, True

    def _try_mod_change(
        self, lp: LightpathDesc, lightpaths: List[LightpathDesc], step: int,
    ) -> Tuple[np.ndarray, bool, str]:
        """Downgrade modulation for more margin. Returns (action, success, actual_strategy)."""
        old_mod = lp.modulation
        if old_mod == Modulation.QAM64:
            new_mod = Modulation.QAM16
        elif old_mod == Modulation.QAM16:
            new_mod = Modulation.QPSK
        else:
            # Already QPSK, fallback to power adjust
            action, success = self._try_power_adjust(lp, lightpaths, step)
            return action, success, 'power_adjust_fallback'

        lp.modulation = new_mod

        lp_idx = next(
            (i for i, l in enumerate(lightpaths) if l.id == lp.id), -1
        )
        src_idx = self.spec.node_ids.index(lp.source)
        dst_idx = self.spec.node_ids.index(lp.destination)

        action = encode_action(
            action_type=ActionType.MOD_CHANGE,
            target_lp_idx=lp_idx,
            src_node=src_idx,
            dst_node=dst_idx,
            wavelength_slot=lp.wavelength_slot,
            modulation=int(new_mod),
        )
        return action, True, 'mod_change'

    def _try_reroute(
        self, lp: LightpathDesc, lightpaths: List[LightpathDesc], step: int,
    ) -> Tuple[np.ndarray, bool, str]:
        """Try an alternative route. Returns (action, success, actual_strategy)."""
        try:
            paths = list(nx.shortest_simple_paths(
                self.graph, lp.source, lp.destination, weight='weight'
            ))[:self.config.k_shortest_paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            action, success = self._try_power_adjust(lp, lightpaths, step)
            return action, success, 'power_adjust_fallback'

        # Find a different route with available wavelength
        for path in paths:
            if path == lp.route:
                continue

            # Release old wavelength
            self._release_wavelength(lp)
            slot = self._first_fit_wavelength(path)

            if slot is not None:
                old_slot = lp.wavelength_slot
                lp.route = path
                lp.wavelength_slot = slot
                lp.frequency_thz = 191.35 + slot * 50.0 / 1000
                self._mark_wavelength(lp)

                lp_idx = next(
                    (i for i, l in enumerate(lightpaths) if l.id == lp.id), -1
                )
                src_idx = self.spec.node_ids.index(lp.source)
                dst_idx = self.spec.node_ids.index(lp.destination)
                route_indices = [
                    self.spec.node_ids.index(n) for n in path
                ]

                action = encode_action(
                    action_type=ActionType.REROUTE,
                    target_lp_idx=lp_idx,
                    src_node=src_idx,
                    dst_node=dst_idx,
                    route=route_indices,
                    wavelength_slot=slot,
                )
                return action, True, 'reroute'
            else:
                # Restore old wavelength
                self._mark_wavelength(lp)

        # All routes failed, fallback to power adjust
        action, success = self._try_power_adjust(lp, lightpaths, step)
        return action, success, 'power_adjust_fallback'

    # ---- shared helpers (same as provisioning) ----

    def _first_fit_wavelength(self, route: List[str],
                              top_k: int = 5) -> Optional[int]:
        """Find an available slot, random among the first top_k free."""
        link_ids = []
        for i in range(len(route) - 1):
            lid = self.link_lookup.get((route[i], route[i + 1]))
            if lid is None:
                return None
            link_ids.append(lid)
        free_slots = []
        for slot in range(MAX_SLOTS):
            if all(slot not in self.wl_usage.get(lid, set())
                   for lid in link_ids):
                free_slots.append(slot)
                if len(free_slots) >= top_k:
                    break
        if not free_slots:
            return None
        return int(self.rng.choice(free_slots))

    def _mark_wavelength(self, lp: LightpathDesc):
        for i in range(lp.n_hops):
            lid = self.link_lookup.get((lp.route[i], lp.route[i + 1]))
            if lid:
                self.wl_usage.setdefault(lid, set()).add(lp.wavelength_slot)

    def _release_wavelength(self, lp: LightpathDesc):
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