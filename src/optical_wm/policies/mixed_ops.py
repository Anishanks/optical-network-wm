"""
Mixed Operations Policy for World Model Dataset Generation.

Simulates a realistic operator workflow that combines multiple operation types
within a single episode. This breaks the state→action correlation that exists
when each policy is run independently.

Episode structure (4 phases):
  Phase 1 — Provisioning:  ADD new LPs to build load
  Phase 2 — Optimization:  POWER_ADJUST / MOD_CHANGE / REROUTE to improve margins
  Phase 3 — Perturbation:  REMOVE some LPs (simulate churn, failures, decommission)
  Phase 4 — Re-provision:  ADD replacement LPs on different routes

Additionally, ε-greedy noise injects random off-policy actions:
  - ADD during optimization phase
  - REMOVE during provisioning phase
  - Random power adjustments at any point

This ensures that for similar states, the dataset contains diverse actions,
forcing the world model to learn P(s'|s,a) rather than P(a|s).
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
class MixedOpsConfig:
    """Configuration for mixed operations policy."""
    # Phase allocation (fractions of max_steps)
    provision_frac: float = 0.30
    optimize_frac: float = 0.25
    perturb_frac: float = 0.20
    reprovision_frac: float = 0.25

    max_steps: int = 50
    initial_load_frac: float = 0.15    # start light, build up
    epsilon: float = 0.10              # probability of off-policy action
    k_shortest_paths: int = 3
    power_step_db: float = 0.5
    power_min_db: float = -3.0
    power_max_db: float = 3.0
    seed: int = 42


class MixedOpsPolicy:
    """
    Expert policy combining all operation types in a single episode.

    Key property for world model training:
      Same state region → multiple action types observed
      (breaks P(a|s) correlation, forces learning P(s'|s,a))
    """

    def __init__(self, evaluator: NetworkEvaluator, spec: TopologySpec,
                 config: MixedOpsConfig = None):
        self.evaluator = evaluator
        self.spec = spec
        self.config = config or MixedOpsConfig()
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
        """Generate a multi-phase mixed operations episode."""
        cfg = self.config

        # Compute phase boundaries
        n_provision = int(cfg.max_steps * cfg.provision_frac)
        n_optimize = int(cfg.max_steps * cfg.optimize_frac)
        n_perturb = int(cfg.max_steps * cfg.perturb_frac)
        n_reprovision = cfg.max_steps - n_provision - n_optimize - n_perturb

        # Phase 0: Build initial light load
        lightpaths = self._build_initial_load()
        results, _ = self.evaluator.evaluate_all(lightpaths)
        state = self._encode_current_state(lightpaths, results)

        states = [state]
        actions = []
        step_info = []
        removed_demands = []  # for re-provisioning phase
        lp_counter = len(lightpaths)

        # ============================================================
        # Phase 1: Provisioning (ADD, with ε-noise)
        # ============================================================
        for step in range(n_provision):
            if self.rng.random() < cfg.epsilon and len(lightpaths) > 3:
                # Off-policy: random power adjust or remove
                action, phase = self._random_off_policy(
                    lightpaths, lp_counter, 'provision'
                )
                if phase == 'remove':
                    demand = self._record_and_remove_lp(lightpaths)
                    removed_demands.append(demand)
                    src_idx = self.spec.node_ids.index(demand[0])
                    dst_idx = self.spec.node_ids.index(demand[1])
                    action = encode_action(
                        action_type=ActionType.REMOVE,
                        src_node=src_idx,
                        dst_node=dst_idx,
                    )
                    phase = 'provision_noise_remove'
            else:
                # On-policy: ADD a new LP
                lp, action = self._try_add(lightpaths, lp_counter)
                if lp is not None:
                    lightpaths.append(lp)
                    self._mark_wavelength(lp)
                    lp_counter += 1
                    phase = 'provision_add'
                else:
                    action = self._do_power_adjust(lightpaths)
                    phase = 'provision_fallback_power'

            results, gnpy_ms = self.evaluator.evaluate_all(lightpaths)
            state = self._encode_current_state(lightpaths, results)
            states.append(state)
            actions.append(action)
            step_info.append({
                'step': len(actions) - 1, 'phase': phase,
                'n_lps': len(lightpaths), 'gnpy_ms': gnpy_ms,
            })

        # ============================================================
        # Phase 2: Optimization (POWER/MOD/REROUTE, with ε-noise)
        # ============================================================
        for step in range(n_optimize):
            if self.rng.random() < cfg.epsilon:
                # Off-policy: ADD or REMOVE during optimization
                if self.rng.random() < 0.5 and len(lightpaths) > 3:
                    demand = self._record_and_remove_lp(lightpaths)
                    removed_demands.append(demand)
                    src_idx = self.spec.node_ids.index(demand[0])
                    dst_idx = self.spec.node_ids.index(demand[1])
                    action = encode_action(
                        action_type=ActionType.REMOVE,
                        src_node=src_idx,
                        dst_node=dst_idx,
                    )
                    phase = 'optimize_noise_remove'
                else:
                    lp, action = self._try_add(lightpaths, lp_counter)
                    if lp is not None:
                        lightpaths.append(lp)
                        self._mark_wavelength(lp)
                        lp_counter += 1
                    phase = 'optimize_noise_add'
            else:
                # On-policy: optimize worst-margin LP
                strategy = self.rng.choice(
                    ['power', 'power', 'modulation', 'reroute'],
                )
                action, phase = self._optimize_step(
                    lightpaths, strategy, lp_counter
                )

            results, gnpy_ms = self.evaluator.evaluate_all(lightpaths)
            state = self._encode_current_state(lightpaths, results)
            states.append(state)
            actions.append(action)
            step_info.append({
                'step': len(actions) - 1, 'phase': phase,
                'n_lps': len(lightpaths), 'gnpy_ms': gnpy_ms,
            })

        # ============================================================
        # Phase 3: Perturbation (REMOVE, with ε-noise ADD)
        # ============================================================
        for step in range(n_perturb):
            if self.rng.random() < cfg.epsilon:
                # Off-policy: ADD during perturbation
                lp, action = self._try_add(lightpaths, lp_counter)
                if lp is not None:
                    lightpaths.append(lp)
                    self._mark_wavelength(lp)
                    lp_counter += 1
                phase = 'perturb_noise_add'
            else:
                # On-policy: REMOVE a random LP
                if len(lightpaths) > 3:
                    demand = self._record_and_remove_lp(lightpaths)
                    removed_demands.append(demand)
                    phase = 'perturb_remove'
                    src_idx = self.spec.node_ids.index(demand[0])
                    dst_idx = self.spec.node_ids.index(demand[1])
                    action = encode_action(
                        action_type=ActionType.REMOVE,
                        src_node=src_idx,
                        dst_node=dst_idx,
                    )
                else:
                    action = self._do_power_adjust(lightpaths)
                    phase = 'perturb_fallback_power'

            results, gnpy_ms = self.evaluator.evaluate_all(lightpaths)
            state = self._encode_current_state(lightpaths, results)
            states.append(state)
            actions.append(action)
            step_info.append({
                'step': len(actions) - 1, 'phase': phase,
                'n_lps': len(lightpaths), 'gnpy_ms': gnpy_ms,
            })

        # ============================================================
        # Phase 4: Re-provisioning (ADD on new routes, with ε-noise)
        # ============================================================
        demand_idx = 0
        for step in range(n_reprovision):
            if self.rng.random() < cfg.epsilon and len(lightpaths) > 3:
                # Off-policy: power adjust or remove
                action = self._do_power_adjust(lightpaths)
                phase = 'reprovision_noise_power'
            else:
                # On-policy: re-add a removed demand
                added = False
                while demand_idx < len(removed_demands):
                    src, dst, mod = removed_demands[demand_idx]
                    demand_idx += 1

                    lp, action = self._try_add_specific(
                        lightpaths, lp_counter, src, dst, mod
                    )
                    if lp is not None:
                        lightpaths.append(lp)
                        self._mark_wavelength(lp)
                        lp_counter += 1
                        added = True
                        phase = 'reprovision_add'
                        break

                if not added:
                    # No more demands to restore, add random
                    lp, action = self._try_add(lightpaths, lp_counter)
                    if lp is not None:
                        lightpaths.append(lp)
                        self._mark_wavelength(lp)
                        lp_counter += 1
                    phase = 'reprovision_random_add'

            results, gnpy_ms = self.evaluator.evaluate_all(lightpaths)
            state = self._encode_current_state(lightpaths, results)
            states.append(state)
            actions.append(action)
            step_info.append({
                'step': len(actions) - 1, 'phase': phase,
                'n_lps': len(lightpaths), 'gnpy_ms': gnpy_ms,
            })

        # ============================================================
        # Metadata
        # ============================================================
        action_types = [int(a[0]) for a in actions]
        type_counts = {}
        for t in action_types:
            name = ActionType(t).name
            type_counts[name] = type_counts.get(name, 0) + 1

        phase_counts = {}
        for s in step_info:
            p = s['phase'].split('_')[0]  # main phase name
            phase_counts[p] = phase_counts.get(p, 0) + 1

        metadata = {
            'policy': 'mixed_ops',
            'seed': cfg.seed,
            'n_steps': len(actions),
            'action_type_counts': type_counts,
            'phase_counts': phase_counts,
            'epsilon': cfg.epsilon,
            'n_nodes': len(self.spec.node_ids),
            'n_links': len(self.spec.links),
        }

        return {
            'states': states,
            'actions': actions,
            'metadata': metadata,
            'step_info': step_info,
        }

    # =================================================================
    # Initial load
    # =================================================================

    def _build_initial_load(self) -> List[LightpathDesc]:
        target = int(MAX_SLOTS * self.config.initial_load_frac)
        target = max(target, 3)

        lightpaths = []
        counter = 0
        attempts = 0

        while len(lightpaths) < target and attempts < target * 5:
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

            power = float(np.clip(
                self.rng.normal(0, 0.5), -2.0, 2.0
            ))
            lp = create_lightpath(
                id=f"lp_mix_{counter:04d}",
                source=src, destination=dst,
                route=route, wavelength_slot=slot,
                modulation=Modulation.QPSK,
                launch_power_dbm=power,
            )
            lightpaths.append(lp)
            self._mark_wavelength(lp)
            counter += 1

        return lightpaths

    # =================================================================
    # Action primitives
    # =================================================================

    def _try_add(
        self, lightpaths: List[LightpathDesc], counter: int,
    ) -> Tuple[Optional[LightpathDesc], np.ndarray]:
        """Try to add a random LP."""
        for _ in range(10):
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

            mod = self.rng.choice([Modulation.QPSK, Modulation.QAM16])
            power = float(np.clip(self.rng.normal(0, 0.5), -2.0, 2.0))

            lp = create_lightpath(
                id=f"lp_mix_{counter:04d}",
                source=src, destination=dst,
                route=route, wavelength_slot=slot,
                modulation=mod,
                launch_power_dbm=power,
            )

            src_idx = self.spec.node_ids.index(src)
            dst_idx = self.spec.node_ids.index(dst)
            route_indices = [self.spec.node_ids.index(n) for n in route]

            action = encode_action(
                action_type=ActionType.ADD,
                src_node=src_idx,
                dst_node=dst_idx,
                route=route_indices,
                wavelength_slot=slot,
                modulation=int(mod),
                power_delta_db=power,
            )
            return lp, action

        # Fallback: power adjust
        return None, self._do_power_adjust(lightpaths)

    def _try_add_specific(
        self, lightpaths: List[LightpathDesc], counter: int,
        src: str, dst: str, modulation: Modulation,
    ) -> Tuple[Optional[LightpathDesc], np.ndarray]:
        """Try to add an LP for a specific src-dst pair."""
        try:
            paths = list(nx.shortest_simple_paths(
                self.graph, src, dst, weight='weight'
            ))[:self.config.k_shortest_paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, self._do_power_adjust(lightpaths)

        for route in paths:
            slot = self._first_fit_wavelength(route)
            if slot is None:
                continue

            lp = create_lightpath(
                id=f"lp_mix_{counter:04d}",
                source=src, destination=dst,
                route=route, wavelength_slot=slot,
                modulation=modulation,
                launch_power_dbm=0.0,
            )

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
            return lp, action

        return None, self._do_power_adjust(lightpaths)

    def _do_power_adjust(self, lightpaths: List[LightpathDesc]) -> np.ndarray:
        """Adjust power on a random LP."""
        if not lightpaths:
            return encode_action(action_type=ActionType.POWER_ADJUST)

        lp = self.rng.choice(lightpaths)
        delta = self.rng.choice([-1.0, -0.5, 0.5, 1.0])
        new_power = float(np.clip(
            lp.launch_power_dbm + delta,
            self.config.power_min_db,
            self.config.power_max_db,
        ))
        actual_delta = new_power - lp.launch_power_dbm
        lp.launch_power_dbm = new_power

        lp_idx = next(
            (i for i, l in enumerate(lightpaths) if l.id == lp.id), -1
        )
        src_idx = self.spec.node_ids.index(lp.source)
        dst_idx = self.spec.node_ids.index(lp.destination)

        return encode_action(
            action_type=ActionType.POWER_ADJUST,
            target_lp_idx=lp_idx,
            src_node=src_idx,
            dst_node=dst_idx,
            wavelength_slot=lp.wavelength_slot,
            power_delta_db=float(actual_delta),
        )

    def _record_and_remove_lp(
        self, lightpaths: List[LightpathDesc],
    ) -> Tuple[str, str, Modulation]:
        """Remove a random LP and return (src, dst, mod) for potential re-add."""
        idx = self.rng.integers(len(lightpaths))
        lp = lightpaths.pop(idx)
        self._release_wavelength(lp)
        return (lp.source, lp.destination, lp.modulation)

    def _random_off_policy(
        self, lightpaths: List[LightpathDesc], counter: int, current_phase: str,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Generate a random off-policy action."""
        roll = self.rng.random()

        if roll < 0.4:
            # Power adjust (always safe)
            return self._do_power_adjust(lightpaths), 'power'
        elif roll < 0.7 and len(lightpaths) > 3:
            # Signal remove — caller handles the actual removal
            return None, 'remove'
        else:
            # Add
            lp, action = self._try_add(lightpaths, counter)
            if lp is not None:
                lightpaths.append(lp)
                self._mark_wavelength(lp)
            return action, 'add'

    def _optimize_step(
        self, lightpaths: List[LightpathDesc], strategy: str, counter: int,
    ) -> Tuple[np.ndarray, str]:
        """Execute one optimization step."""
        if not lightpaths:
            return encode_action(action_type=ActionType.POWER_ADJUST), 'optimize_empty'

        if strategy == 'power':
            return self._do_power_adjust(lightpaths), 'optimize_power'

        elif strategy == 'modulation':
            # Find LP with highest modulation and downgrade
            candidates = [
                lp for lp in lightpaths if lp.modulation != Modulation.QPSK
            ]
            if not candidates:
                return self._do_power_adjust(lightpaths), 'optimize_power_fallback'

            lp = self.rng.choice(candidates)
            old_mod = lp.modulation
            new_mod = Modulation(max(0, int(old_mod) - 1))
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
            return action, 'optimize_modulation'

        elif strategy == 'reroute':
            # Pick random LP and try alternative route
            lp = self.rng.choice(lightpaths)
            try:
                paths = list(nx.shortest_simple_paths(
                    self.graph, lp.source, lp.destination, weight='weight'
                ))[:self.config.k_shortest_paths]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return self._do_power_adjust(lightpaths), 'optimize_power_fallback'

            for path in paths:
                if path == lp.route:
                    continue
                self._release_wavelength(lp)
                slot = self._first_fit_wavelength(path)
                if slot is not None:
                    lp.route = path
                    lp.wavelength_slot = slot
                    lp.frequency_thz = 191.35 + slot * 50.0 / 1000
                    self._mark_wavelength(lp)

                    lp_idx = next(
                        (i for i, l in enumerate(lightpaths) if l.id == lp.id),
                        -1,
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
                    return action, 'optimize_reroute'
                else:
                    self._mark_wavelength(lp)

            return self._do_power_adjust(lightpaths), 'optimize_power_fallback'

        return self._do_power_adjust(lightpaths), 'optimize_unknown'

    # =================================================================
    # Shared helpers
    # =================================================================

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