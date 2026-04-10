"""
Provisioning Policy for World Model Dataset Generation.

Simulates an admin deploying new services on the network.
Each step provisions one lightpath from a demand list using:
  1. k-shortest-path routing
  2. First-fit wavelength assignment
  3. Conservative modulation selection (QPSK default, higher for short hops)
  4. GNPy validation after each addition

Design principles:
  - Every step changes the network state (no dead steps)
  - Actions are intentional and goal-directed (satisfy demands)
  - Failed provisions are recorded as actions (with infeasible outcome)
  - The trajectory shows load increasing from initial to target
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..core.schemas import (
    LightpathDesc, LightpathResult, Modulation, ActionType,
    encode_action, encode_state, MAX_SLOTS,
)
from ..core.gnpy_wrapper import (
    NetworkEvaluator, TopologySpec, create_lightpath,
)


@dataclass
class Demand:
    """A service request to provision."""
    src: str
    dst: str
    rate_gbps: float = 128.0  # default QPSK capacity
    priority: int = 1         # higher = more important


@dataclass
class ProvisioningConfig:
    """Configuration for the provisioning policy."""
    initial_load_frac: float = 0.2     # start at 20% load
    target_load_frac: float = 0.9      # aim for 90% load
    max_steps: int = 60                # maximum episode length
    n_demands: int = 50                # number of demands to generate
    k_shortest_paths: int = 3          # routing alternatives
    default_power_dbm: float = 0.0     # launch power
    power_noise_std: float = 0.5       # per-LP power variation
    seed: int = 42


class ProvisioningPolicy:
    """
    Expert-like policy that provisions lightpaths greedily.

    Trajectory structure:
      1. Start with initial_load_frac of network loaded
      2. Generate a demand list (random src-dst pairs)
      3. For each demand: find route, assign wavelength, provision
      4. Each step = one provisioning attempt (success or fail)
      5. Stop when all demands served or max_steps reached
    """

    def __init__(self, evaluator: NetworkEvaluator, spec: TopologySpec,
                 config: ProvisioningConfig = None):
        self.evaluator = evaluator
        self.spec = spec
        self.config = config or ProvisioningConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Build networkx graph for routing
        self.graph = nx.Graph()
        for link in spec.links:
            self.graph.add_edge(
                link['src'], link['dst'],
                weight=link['length_km'],
                link_id=link['id'],
            )

        # Track wavelength usage per link
        self.wl_usage: Dict[str, set] = {
            link['id']: set() for link in spec.links
        }

        # Link lookup for bidirectional access
        self.link_lookup = {}
        for link in spec.links:
            self.link_lookup[(link['src'], link['dst'])] = link['id']
            self.link_lookup[(link['dst'], link['src'])] = link['id']

        # Adjacency matrix for state encoding
        n = len(spec.node_ids)
        self.adjacency = np.zeros((n, n))
        for link in spec.links:
            i = spec.node_ids.index(link['src'])
            j = spec.node_ids.index(link['dst'])
            self.adjacency[i, j] = self.adjacency[j, i] = 1

    def generate_episode(self) -> dict:
        """
        Generate a complete provisioning episode.

        Returns dict with:
          - states: list of encoded states
          - actions: list of encoded actions
          - lightpaths_per_step: list of LP lists (for debugging)
          - metadata: episode info
          - demands_served: number of demands successfully provisioned
        """
        # Phase 0: Build initial load
        lightpaths = self._build_initial_load()

        # Evaluate initial state
        results, _ = self.evaluator.evaluate_all(lightpaths)
        state = self._encode_current_state(lightpaths, results)

        states = [state]
        actions = []
        lps_per_step = [list(lightpaths)]
        step_info = []

        # Generate demand list
        demands = self._generate_demands()

        # Phase 1: Provision demands one by one
        demand_idx = 0
        demands_served = 0
        demands_failed = 0

        for step in range(self.config.max_steps):
            if demand_idx >= len(demands):
                break  # all demands attempted

            demand = demands[demand_idx]
            demand_idx += 1

            # Try to provision this demand
            new_lp, action, success = self._try_provision(
                demand, lightpaths, step
            )

            if success and new_lp is not None:
                lightpaths.append(new_lp)
                demands_served += 1
            else:
                demands_failed += 1

            # Always evaluate full network after action
            results, gnpy_ms = self.evaluator.evaluate_all(lightpaths)

            # Check if any existing LP became infeasible after addition
            if success and new_lp is not None:
                new_result = results.get(new_lp.id)
                if new_result and not new_result.feasible:
                    # New LP is infeasible — rollback
                    lightpaths.remove(new_lp)
                    self._release_wavelength(new_lp)
                    results, _ = self.evaluator.evaluate_all(lightpaths)
                    success = False
                    demands_served -= 1
                    demands_failed += 1

            # Encode state and action
            state = self._encode_current_state(lightpaths, results)
            states.append(state)
            actions.append(action)
            lps_per_step.append(list(lightpaths))

            step_info.append({
                'step': step,
                'demand': f"{demand.src}→{demand.dst}",
                'success': success,
                'n_lps': len(lightpaths),
                'gnpy_ms': gnpy_ms,
            })

        # Compute utilization trajectory
        utilizations = [
            s['global_features'][6]  # avg spectral utilization
            for s in states
        ]

        metadata = {
            'policy': 'provisioning',
            'seed': self.config.seed,
            'n_steps': len(actions),
            'initial_load': float(utilizations[0]) if utilizations else 0,
            'final_load': float(utilizations[-1]) if utilizations else 0,
            'demands_total': len(demands),
            'demands_served': demands_served,
            'demands_failed': demands_failed,
            'n_nodes': len(self.spec.node_ids),
            'n_links': len(self.spec.links),
        }

        return {
            'states': states,
            'actions': actions,
            'lightpaths_per_step': lps_per_step,
            'metadata': metadata,
            'step_info': step_info,
        }

    # -----------------------------------------------------------------
    # Initial load
    # -----------------------------------------------------------------

    def _build_initial_load(self) -> List[LightpathDesc]:
        """Create initial lightpaths to reach initial_load_frac.

        The actual load fraction is jittered ±50% around the configured
        value to diversify the starting conditions across episodes.
        """
        # Jitter initial load: e.g. 0.2 → uniform [0.10, 0.30]
        base = self.config.initial_load_frac
        jittered = self.rng.uniform(base * 0.5, base * 1.5)
        target_lps = int(MAX_SLOTS * jittered)
        target_lps = max(target_lps, 3)  # minimum for GNPy

        lightpaths = []
        lp_counter = 0
        attempts = 0
        max_attempts = target_lps * 5

        while len(lightpaths) < target_lps and attempts < max_attempts:
            attempts += 1
            # Random src-dst pair
            src, dst = self.rng.choice(
                self.spec.node_ids, size=2, replace=False
            )

            route = self._find_route(src, dst)
            if route is None:
                continue

            slot = self._first_fit_wavelength(route)
            if slot is None:
                continue

            modulation = self._select_modulation(route)
            power = self.config.default_power_dbm + \
                    self.rng.normal(0, self.config.power_noise_std)
            power = float(np.clip(power, -3.0, 3.0))

            lp = create_lightpath(
                id=f"lp_init_{lp_counter:04d}",
                source=src, destination=dst,
                route=route, wavelength_slot=slot,
                modulation=modulation,
                launch_power_dbm=power,
            )
            lightpaths.append(lp)
            self._mark_wavelength(lp)
            lp_counter += 1

        return lightpaths

    # -----------------------------------------------------------------
    # Demand generation
    # -----------------------------------------------------------------

    def _generate_demands(self) -> List[Demand]:
        """Generate a shuffled list of service demands."""
        demands = []
        nodes = self.spec.node_ids

        for i in range(self.config.n_demands):
            src, dst = self.rng.choice(nodes, size=2, replace=False)
            demand = Demand(
                src=src, dst=dst,
                rate_gbps=float(self.rng.choice([128, 256])),  # QPSK or 16QAM
                priority=int(self.rng.integers(1, 4)),
            )
            demands.append(demand)

        # Sort by priority (highest first)
        demands.sort(key=lambda d: -d.priority)
        return demands

    # -----------------------------------------------------------------
    # Provisioning logic
    # -----------------------------------------------------------------

    def _try_provision(self, demand: Demand, existing_lps: List[LightpathDesc],
                       step: int) -> Tuple[Optional[LightpathDesc], np.ndarray, bool]:
        """
        Try to provision a single demand.
        Returns (new_lp_or_None, action_vector, success_bool).
        """
        # Try k shortest paths
        routes = self._find_k_routes(demand.src, demand.dst,
                                      self.config.k_shortest_paths)

        for route in routes:
            slot = self._first_fit_wavelength(route)
            if slot is None:
                continue  # no available wavelength on this route

            modulation = self._select_modulation(route)
            power = self.config.default_power_dbm + \
                    self.rng.normal(0, self.config.power_noise_std)
            power = float(np.clip(power, -3.0, 3.0))

            lp = create_lightpath(
                id=f"lp_{step:04d}",
                source=demand.src,
                destination=demand.dst,
                route=route,
                wavelength_slot=slot,
                modulation=modulation,
                launch_power_dbm=power,
            )

            # Build action vector
            src_idx = self.spec.node_ids.index(demand.src)
            dst_idx = self.spec.node_ids.index(demand.dst)
            route_indices = [self.spec.node_ids.index(n) for n in route]

            action = encode_action(
                action_type=ActionType.ADD,
                src_node=src_idx,
                dst_node=dst_idx,
                route=route_indices,
                wavelength_slot=slot,
                modulation=int(modulation),
                power_delta_db=power,
                rate_gbps=lp.capacity_gbps,
            )

            self._mark_wavelength(lp)
            return lp, action, True

        # All routes failed — record as failed ADD action
        src_idx = self.spec.node_ids.index(demand.src)
        dst_idx = self.spec.node_ids.index(demand.dst)
        action = encode_action(
            action_type=ActionType.ADD,
            src_node=src_idx,
            dst_node=dst_idx,
            wavelength_slot=-1,  # indicates failure
            rate_gbps=demand.rate_gbps,
        )
        return None, action, False

    # -----------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------

    def _find_route(self, src: str, dst: str) -> Optional[List[str]]:
        """Find shortest path."""
        try:
            return nx.shortest_path(self.graph, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            return None

    def _find_k_routes(self, src: str, dst: str, k: int) -> List[List[str]]:
        """Find up to k shortest simple paths."""
        try:
            paths = list(nx.shortest_simple_paths(
                self.graph, src, dst, weight='weight'
            ))
            return paths[:k]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # -----------------------------------------------------------------
    # Wavelength assignment
    # -----------------------------------------------------------------

    def _first_fit_wavelength(self, route: List[str],
                              top_k: int = 5) -> Optional[int]:
        """Find an available wavelength slot on all links of the route.

        Instead of strict first-fit (always slot 0, 1, 2, ...),
        pick randomly among the first ``top_k`` free slots.
        This diversifies spectral occupation patterns across episodes,
        giving the world model richer input for learning NLI dynamics.
        """
        link_ids = []
        for i in range(len(route) - 1):
            lid = self.link_lookup.get((route[i], route[i + 1]))
            if lid is None:
                return None
            link_ids.append(lid)

        # Collect free slots (up to top_k)
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
        """Mark wavelength as used on all links of the LP's route."""
        for i in range(lp.n_hops):
            lid = self.link_lookup.get((lp.route[i], lp.route[i + 1]))
            if lid:
                self.wl_usage.setdefault(lid, set()).add(lp.wavelength_slot)

    def _release_wavelength(self, lp: LightpathDesc):
        """Release wavelength on all links of the LP's route."""
        for i in range(lp.n_hops):
            lid = self.link_lookup.get((lp.route[i], lp.route[i + 1]))
            if lid:
                self.wl_usage.get(lid, set()).discard(lp.wavelength_slot)

    # -----------------------------------------------------------------
    # Modulation selection
    # -----------------------------------------------------------------

    def _select_modulation(self, route: List[str]) -> Modulation:
        """Conservative modulation: QPSK unless very short path."""
        hops = len(route) - 1
        if hops <= 1:
            return Modulation.QAM16  # short path → higher capacity
        return Modulation.QPSK  # default conservative

    # -----------------------------------------------------------------
    # State encoding
    # -----------------------------------------------------------------

    def _encode_current_state(self, lightpaths: List[LightpathDesc],
                               results: Dict[str, LightpathResult]) -> dict:
        """Encode the current network state."""
        return encode_state(
            node_ids=self.spec.node_ids,
            link_data=self.spec.links,
            lightpaths=lightpaths,
            gnpy_results=results,
            adjacency=self.adjacency,
        )