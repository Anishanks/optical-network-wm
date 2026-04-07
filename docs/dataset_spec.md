# JEPA World Model Dataset — Implementation Specification

## 1. Dataset Overview

### Dimensioning

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Total episodes | 2,000 | ~120K transitions, balanced compute vs coverage |
| Steps per episode | 40–60 (varies by policy) | Long enough for planning horizons of 15–25 steps |
| Total transitions | ~100,000–120,000 | 10–30× less than LeWM in count, but ~10× richer per transition |
| Topologies | 3 families (train) + 1 (test) | Generalization test across unseen topology |
| Policies | 4 intentional policies | Expert-like, goal-directed trajectories |
| Storage format | HDF5 (single file per topology family) | Fast data loading, ~100× smaller than JSON |
| GNPy mode | Real propagation only | No mock — scientifically mandatory |

### Episode Distribution

| Policy | Episodes | Steps/ep | Total steps | Purpose |
|--------|----------|----------|-------------|---------|
| Provisioning | 600 (30%) | 40–60 | ~30,000 | Learn load-building dynamics |
| Margin Optimization | 500 (25%) | 40–50 | ~22,500 | Learn corrective actions |
| Load Balancing | 400 (20%) | 40–50 | ~18,000 | Learn structural rerouting |
| Recovery | 500 (25%) | 50–60 | ~27,500 | Learn restoration dynamics |

### Topology Families

| Family | Nodes | Links | Connectivity | Split | Episodes |
|--------|-------|-------|-------------|-------|----------|
| A — Small mesh | 8 | ~12 | 0.40 | Train | 500 |
| B — Medium sparse | 12 | ~18 | 0.25 | Train | 500 |
| C — Medium mesh | 14 | ~25 | 0.30 | Train | 500 |
| D — Test topology | 10 | ~16 | 0.35 | Test | 500 |

---

## 2. State Representation (~2,400 dimensions)

### 2.1 Per-link features (edge features)

For each link in the topology, store a spectral snapshot:

```
link_features[link_id]:
    # Static (constant within episode)
    length_km:              float32         # physical length
    n_spans:                int32           # number of fiber spans
    n_amplifiers:           int32           # number of EDFAs

    # Dynamic (changes every step — this is where GNPy dynamics live)
    spectral_occupancy:     bool[80]        # which slots are occupied
    channel_gsnr_db:        float32[80]     # GSNR per slot (0 if empty)
    channel_power_dbm:      float32[80]     # power per slot (0 if empty)
    channel_ase_dbm:        float32[80]     # ASE noise per slot (0 if empty)
    channel_nli_dbm:        float32[80]     # NLI per slot (0 if empty)
    total_utilization:      float32         # fraction of slots occupied
    worst_gsnr_db:          float32         # worst GSNR on this link
    avg_gsnr_db:            float32         # average GSNR of active channels
```

Dimensions: ~325 per link × 18 links (avg) ≈ 5,850
After Conv1D encoding: 64 per link × 18 = 1,152

### 2.2 Per-node features (node features)

```
node_features[node_id]:
    degree:                 int32           # number of connected links
    n_add_drop_channels:    int32           # channels added/dropped here
    total_through_channels: int32           # express channels through this ROADM
    is_source:              bool            # has active transmitters
    is_destination:         bool            # has active receivers
```

Dimensions: 5 per node × 12 nodes (avg) = 60

### 2.3 Per-lightpath features (service features)

```
lightpath_features[lp_id]:
    source_node:            int32           # source node index
    dest_node:              int32           # destination node index
    route_length_hops:      int32           # number of hops
    route_node_indices:     int32[max_hops] # padded route (max_hops=8)
    wavelength_slot:        int32           # frequency slot index
    frequency_thz:          float32         # center frequency
    modulation:             int32           # enum: 0=QPSK, 1=16QAM, 2=64QAM
    launch_power_dbm:       float32         # launch power
    baud_rate_gbaud:        float32         # symbol rate
    capacity_gbps:          float32         # line rate
    # GNPy outputs for this LP
    gsnr_db:                float32         # end-to-end GSNR
    osnr_db:                float32         # end-to-end OSNR
    margin_db:              float32         # margin above modulation threshold
    feasible:               bool            # margin > 0
    cd_ps_nm:               float32         # accumulated chromatic dispersion
    pmd_ps:                 float32         # accumulated PMD
```

Dimensions: ~20 per LP × 40 LPs (avg) = 800

### 2.4 Global state (summary)

```
global_features:
    n_active_lightpaths:    int32
    total_capacity_tbps:    float32
    n_feasible:             int32
    n_infeasible:           int32
    worst_margin_db:        float32
    avg_margin_db:          float32
    spectral_utilization:   float32         # global average
    max_link_utilization:   float32         # most loaded link
```

Dimensions: 8

### Total raw dimensions: ~6,718 (before encoding)
### After GNN encoding target: 128–256

---

## 3. Action Representation

### 3.1 Action space (fixed-size vector)

```
action:
    type:                   int32           # 0=add, 1=remove, 2=reroute,
                                            # 3=power_adjust, 4=mod_change
    target_lp_id:           int32           # index of target LP (-1 for add)
    src_node:               int32           # source (for add)
    dst_node:               int32           # destination (for add)
    route:                  int32[max_hops] # padded route
    wavelength_slot:        int32           # frequency slot
    modulation:             int32           # target modulation (for mod_change)
    power_delta_db:         float32         # power adjustment (for power_adjust)
    rate_gbps:              float32         # requested rate (for add)
```

Fixed-size vector: ~20 dimensions (padded)

### 3.2 Action types by policy

| Policy | Primary actions | Secondary actions |
|--------|----------------|-------------------|
| Provisioning | add (80%) | power_adjust (10%), mod_change (10%) |
| Margin Optim. | reroute (40%), power_adjust (30%) | mod_change (20%), remove (10%) |
| Load Balance | reroute (70%) | remove+add (20%), power_adjust (10%) |
| Recovery | add (50%), reroute (30%) | power_adjust (15%), mod_change (5%) |

---

## 4. Policy Implementations

### 4.1 Provisioning Policy

```
Input:  demand_list = [(src, dst, rate_gbps)] — shuffled per episode
State:  network with initial_load ∈ [10%, 40%]

For each demand in demand_list:
    1. routes = k_shortest_paths(src, dst, k=3)
    2. For each route in routes:
        a. slot = first_fit_wavelength(route)
        b. If slot is None: continue to next route
        c. modulation = best_feasible_modulation(route_length)
        d. power = optimal_launch_power(route)  # or default
        e. Provision LP, run GNPy on FULL network
        f. If new LP feasible AND no existing LP became infeasible:
             → ACCEPT, record transition, next demand
        g. Else:
             → REJECT this route, try next
    3. If all routes fail:
        a. Try power_adjust on congested links
        b. If still fails → skip demand, record as failed action

Termination: all demands attempted OR 60 steps reached
```

**Verification checklist:**
- [ ] Each step changes at least 1 channel in the spectrum
- [ ] GSNR of ALL channels is recomputed by GNPy after each action
- [ ] Failed provisions are recorded as actions (type=add) with outcome showing infeasibility
- [ ] Demand list is different per episode (shuffled with episode seed)
- [ ] Initial load varies: 10–40% (uniform random per episode)

### 4.2 Margin Optimization Policy

```
Input:  network at 50–80% load, some LPs with margin < 3 dB
Goal:   all LPs with margin > 3 dB

At each step:
    1. Identify worst_lp = LP with lowest margin
    2. If worst_lp.margin > 3.0 dB → DONE (early termination)
    3. Strategy selection (priority order):
        a. Try reroute to shorter path:
           - Find alternate route with fewer hops
           - Assign new wavelength on that route
           - Run GNPy, check if margin improved
        b. Try power optimization:
           - Increase power by 0.5–2.0 dB
           - Run GNPy, check GSNR tradeoff (more power → more NLI)
        c. Try modulation downgrade:
           - 64QAM → 16QAM → QPSK
           - Reduces required GSNR threshold → increases margin
           - But reduces capacity
        d. Last resort — remove lowest-priority LP:
           - Reduces NLI on shared links
           - Other LPs margins should improve
    4. Record transition regardless of success

Termination: all margins > 3 dB OR 50 steps reached
```

**Verification checklist:**
- [ ] Worst margin improves (on average) over the episode
- [ ] Each action targets the most critical LP (not random)
- [ ] Reroute actually changes the path (not same route)
- [ ] Power adjustments respect GNPy equipment limits
- [ ] Modulation downgrades correctly update capacity_gbps

### 4.3 Load Balancing Policy

```
Input:  network with uneven link utilization (some links >70%, others <30%)
Goal:   max_link_utilization - min_link_utilization < threshold

At each step:
    1. Find most_loaded_link = link with highest utilization
    2. Find a LP traversing most_loaded_link
    3. Find alternate route that avoids most_loaded_link
    4. If alternate route exists AND has available wavelength:
        a. Remove LP from current route
        b. Re-provision on alternate route
        c. Run GNPy, verify feasibility
        d. If feasible → ACCEPT
        e. If not → ROLLBACK to previous state
    5. If no alternate route:
        a. Try splitting the demand across two shorter LPs
        b. Or skip and try next LP on the loaded link

Termination: utilization spread < threshold OR 50 steps reached
```

**Verification checklist:**
- [ ] Load balancing metric (std of utilization) decreases over episode
- [ ] Rerouted LPs actually avoid the congested link
- [ ] Rollback is clean (wavelength properly released on failure)
- [ ] Each step results in spectrum change on at least 2 links (old route, new route)

### 4.4 Recovery Policy

```
Phase 1 — Disruption (steps 0–15):
    1. Start with healthy network at 50–70% load
    2. At step 0: record healthy state
    3. Steps 1–15: remove 5–15 LPs (simulating churn/maintenance)
       - Random selection with bias toward high-capacity LPs
       - Each removal improves GSNR of remaining LPs on shared links
       - Record each removal as an action

Phase 2 — Restoration (steps 16–55):
    4. Goal: re-provision all removed services
    5. For each removed service (in priority order):
        a. Try original route first
        b. If not feasible (spectrum occupied) → try alternate routes
        c. If feasible → provision, run GNPy
        d. Possibly adjust power or modulation to fit
    6. Optimize remaining margins if time allows

Termination: all services restored OR 60 steps reached
```

**Verification checklist:**
- [ ] Phase 1 shows improving GSNR (fewer channels = less NLI)
- [ ] Phase 2 shows degrading then stabilizing GSNR (adding channels back)
- [ ] Removed services list is recorded in episode metadata
- [ ] Restoration may use different routes than original (path diversity)
- [ ] Episode metadata records restoration success rate

---

## 5. GNPy Integration Requirements

### 5.1 What GNPy must compute at EVERY step

After each action, a FULL network evaluation:

```python
def evaluate_full_network(network, equipment, all_lightpaths):
    """
    Returns SpectralInformation at destination transceiver
    for EVERY active lightpath.

    CRITICAL: All lightpaths must be propagated together
    to capture inter-channel NLI correctly.
    """
    results = {}
    for lp in all_lightpaths:
        path = build_path(network, lp.route)
        si = create_spectral_information(
            all_active_frequencies,  # ALL channels, not just this LP
            all_active_powers,
            equipment
        )
        propagated_si = propagate(path, si)
        # Extract metrics for THIS specific channel
        ch_idx = find_channel_index(propagated_si, lp.frequency)
        results[lp.id] = {
            'gsnr_db': propagated_si.gsnr[ch_idx],
            'osnr_db': propagated_si.osnr_db[ch_idx],
            'signal_dbm': propagated_si.signal_dbm[ch_idx],
            'ase_dbm': propagated_si.ase_dbm[ch_idx],
            'nli': propagated_si.nli[ch_idx],
            'cd': propagated_si.chromatic_dispersion[ch_idx],
            'pmd': propagated_si.pmd[ch_idx],
        }
    return results
```

### 5.2 GNPy call optimization

Each GNPy call is expensive (~100ms–1s). Optimize:

- **Batch per path:** Group LPs sharing the same route, propagate once.
- **Incremental where possible:** If only one LP was added, only paths
  sharing links with the new LP need full recomputation.
- **Cache topology/equipment:** Build the network graph once per episode,
  modify only the SpectralInformation between steps.

### 5.3 Estimated compute time

| Component | Time per step | Steps/episode | Episodes | Total |
|-----------|--------------|---------------|----------|-------|
| GNPy evaluation | ~300ms (avg) | 50 (avg) | 2,000 | ~8.3 hours |
| State encoding | ~1ms | 50 | 2,000 | ~100 seconds |
| Action selection | ~1ms | 50 | 2,000 | ~100 seconds |
| **Total** | | | | **~9 hours** |

Parallelizable across topology families (4 independent jobs = ~2.5 hours on 4 cores).

---

## 6. HDF5 Storage Format

### 6.1 File structure

```
optical_wm_dataset/
├── topology_A.h5          # all episodes for topology family A
├── topology_B.h5          # all episodes for topology family B
├── topology_C.h5          # all episodes for topology family C
├── topology_D.h5          # test topology
├── metadata.json          # dataset-level metadata
└── normalization.json     # per-feature mean/std for training
```

### 6.2 HDF5 internal structure

```
topology_A.h5
│
├── topology/                          # STATIC — stored once
│   ├── adjacency:         [N, N]      float32   (adjacency matrix)
│   ├── node_features:     [N, 5]      float32
│   ├── link_features:     [E, 3]      float32   (length, n_spans, n_amps)
│   └── link_endpoints:    [E, 2]      int32     (src, dst node indices)
│
├── episodes/
│   ├── ep_0000/
│   │   ├── metadata:                  JSON string (policy, seed, etc.)
│   │   ├── n_steps:                   int32 scalar
│   │   │
│   │   ├── states/                    # One dataset per feature group
│   │   │   ├── spectral_occupancy:    [T, E, 80]    bool
│   │   │   ├── channel_gsnr:          [T, E, 80]    float32
│   │   │   ├── channel_power:         [T, E, 80]    float32
│   │   │   ├── channel_ase:           [T, E, 80]    float32
│   │   │   ├── channel_nli:           [T, E, 80]    float32
│   │   │   ├── lp_features:           [T, max_lp, 20] float32 (padded)
│   │   │   ├── lp_mask:              [T, max_lp]    bool (which LPs active)
│   │   │   └── global_features:       [T, 8]        float32
│   │   │
│   │   ├── actions/                   [T-1, 20]     float32 (fixed-size)
│   │   │
│   │   └── outcomes/                  # GNPy results per step
│   │       ├── per_lp_gsnr:          [T, max_lp]    float32
│   │       ├── per_lp_margin:         [T, max_lp]    float32
│   │       ├── per_lp_feasible:       [T, max_lp]    bool
│   │       ├── total_capacity:        [T]            float32
│   │       └── gnpy_time_ms:          [T]            float32
│   │
│   ├── ep_0001/
│   │   └── ...
│   └── ...
│
└── split/
    ├── train_episodes:     list of episode names
    └── val_episodes:       list of episode names (10% held out)
```

### 6.3 Key design choices

**max_lp = 80:** Maximum lightpaths per step. Padded with zeros,
masked with lp_mask. Covers up to 100% utilization of 80 wavelengths.

**T varies per episode:** Each episode has its own n_steps.
The dataloader samples sub-trajectories of length N (context window)
from within episodes.

**Compression:** Use gzip compression (level=4) on spectral arrays.
Binary occupancy arrays compress ~10×.

### 6.4 Estimated storage

| Component | Per step | Per episode (50 steps) | 2,000 episodes |
|-----------|----------|----------------------|----------------|
| Spectral (5 × E × 80 × 4B) | ~28 KB | 1.4 MB | 2.8 GB |
| LP features (max_lp × 20 × 4B) | ~6 KB | 300 KB | 600 MB |
| Actions + outcomes | ~1 KB | 50 KB | 100 MB |
| **Total (uncompressed)** | | | **~3.5 GB** |
| **Total (gzip compressed)** | | | **~500 MB–1 GB** |

---

## 7. Validation & Verification Checklist

### 7.1 Per-step validation (run during generation)

```python
def validate_step(state_t, action, state_t_plus_1, outcome):
    """Run after every GNPy call. Fail fast on errors."""

    # V1: State changed — no dead steps
    assert not np.array_equal(
        state_t.spectral_occupancy,
        state_t_plus_1.spectral_occupancy
    ) or action.type in [POWER_ADJUST, MOD_CHANGE], \
        "Spectrum unchanged after add/remove/reroute"

    # V2: GNPy consistency — GSNR physically plausible
    active_gsnr = outcome.per_lp_gsnr[outcome.per_lp_feasible]
    assert np.all(active_gsnr > 0) and np.all(active_gsnr < 50), \
        f"GSNR out of physical range: {active_gsnr}"

    # V3: Channel count matches
    n_occupied = state_t_plus_1.spectral_occupancy.sum()
    n_lps = state_t_plus_1.lp_mask.sum()
    # Note: one LP may span multiple links, so n_occupied >= n_lps
    assert n_lps > 0 or action.type == REMOVE, \
        "No active lightpaths but action was not remove"

    # V4: Margin consistency
    for lp_idx in range(n_lps):
        gsnr = outcome.per_lp_gsnr[lp_idx]
        mod_threshold = MOD_THRESHOLDS[state_t_plus_1.lp_modulation[lp_idx]]
        expected_margin = gsnr - mod_threshold
        assert abs(outcome.per_lp_margin[lp_idx] - expected_margin) < 0.01

    # V5: Action coherence — action type matches state change
    if action.type == ADD:
        assert state_t_plus_1.n_active_lps >= state_t.n_active_lps
    elif action.type == REMOVE:
        assert state_t_plus_1.n_active_lps <= state_t.n_active_lps
```

### 7.2 Per-episode validation (run after episode generation)

```python
def validate_episode(episode):
    """Verify episode-level properties."""

    T = episode.n_steps
    assert T >= 30, f"Episode too short: {T} steps"

    # E1: Trajectory has clear dynamics (not flat)
    gsnr_trajectory = [step.global_features.avg_gsnr for step in episode]
    gsnr_range = max(gsnr_trajectory) - min(gsnr_trajectory)
    assert gsnr_range > 1.0, f"GSNR range too small: {gsnr_range} dB"

    # E2: Actions are diverse (not all same type)
    action_types = [step.action.type for step in episode]
    unique_types = set(action_types)
    assert len(unique_types) >= 2, "Only one action type in episode"

    # E3: Policy-specific validation
    if episode.policy == "provisioning":
        # Load should increase over episode
        load_start = episode.steps[0].global_features.spectral_utilization
        load_end = episode.steps[-1].global_features.spectral_utilization
        assert load_end > load_start, "Provisioning didn't increase load"

    elif episode.policy == "margin_optimization":
        # Worst margin should improve (on average, not monotonically)
        margin_first_half = np.mean([s.global_features.worst_margin
                                     for s in episode.steps[:T//2]])
        margin_second_half = np.mean([s.global_features.worst_margin
                                      for s in episode.steps[T//2:]])
        assert margin_second_half > margin_first_half, \
            "Margin optimization didn't improve margins"

    elif episode.policy == "load_balancing":
        util_std_start = np.std(episode.steps[0].link_utilizations)
        util_std_end = np.std(episode.steps[-1].link_utilizations)
        assert util_std_end < util_std_start, \
            "Load balancing didn't reduce utilization spread"

    elif episode.policy == "recovery":
        # Phase 1: LP count decreases, Phase 2: LP count increases
        mid = T // 3
        assert episode.steps[mid].n_active_lps < episode.steps[0].n_active_lps
        assert episode.steps[-1].n_active_lps > episode.steps[mid].n_active_lps
```

### 7.3 Dataset-level validation (run after all episodes)

```python
def validate_dataset(dataset):
    """Verify dataset-level statistical properties."""

    # D1: Policy distribution
    policy_counts = Counter(ep.policy for ep in dataset.episodes)
    for policy, count in policy_counts.items():
        assert count >= 300, f"Policy {policy} underrepresented: {count}"

    # D2: GSNR distribution covers the range
    all_gsnr = np.concatenate([ep.all_gsnr_values for ep in dataset.episodes])
    assert all_gsnr.min() < 5.0, "No near-failure states in dataset"
    assert all_gsnr.max() > 30.0, "No high-quality states in dataset"
    assert np.std(all_gsnr) > 3.0, "GSNR distribution too narrow"

    # D3: Action type distribution
    all_actions = [a.type for ep in dataset.episodes for a in ep.actions]
    action_dist = Counter(all_actions)
    for atype in [ADD, REMOVE, REROUTE, POWER_ADJUST]:
        frac = action_dist[atype] / len(all_actions)
        assert frac > 0.05, f"Action {atype} < 5% of dataset: {frac:.2%}"

    # D4: Inter-channel coupling is present
    # Sample 100 add-actions and verify that OTHER channels' GSNR changed
    sample = random.sample(add_transitions, min(100, len(add_transitions)))
    coupling_present = 0
    for t in sample:
        other_gsnr_before = t.state_t.per_lp_gsnr[t.state_t.lp_mask]
        other_gsnr_after = t.state_t1.per_lp_gsnr[t.state_t.lp_mask]  # same LPs
        if not np.allclose(other_gsnr_before, other_gsnr_after, atol=0.01):
            coupling_present += 1
    assert coupling_present > 80, \
        f"Inter-channel coupling rare: {coupling_present}/100 transitions"

    # D5: No duplicate episodes
    signatures = set()
    for ep in dataset.episodes:
        sig = (ep.policy, ep.seed, ep.topology_family)
        assert sig not in signatures, f"Duplicate episode: {sig}"
        signatures.add(sig)

    print(f"Dataset validated: {len(dataset.episodes)} episodes")
    print(f"  GSNR range: [{all_gsnr.min():.1f}, {all_gsnr.max():.1f}] dB")
    print(f"  Action distribution: {dict(action_dist)}")
    print(f"  Inter-channel coupling: {coupling_present}%")
```

---

## 8. DataLoader for Training

### 8.1 Sub-trajectory sampling

```python
class OpticalWMDataset(torch.utils.data.Dataset):
    """
    Samples sub-trajectories of length N from episodes.
    Analogous to LeWM's HDF5Dataset with num_steps parameter.
    """
    def __init__(self, hdf5_path, context_length=4, keys_to_load=None):
        self.h5 = h5py.File(hdf5_path, 'r')
        self.context_length = context_length

        # Build index: list of (episode_name, start_step)
        self.index = []
        for ep_name in self.h5['episodes']:
            T = self.h5[f'episodes/{ep_name}/n_steps'][()]
            for t in range(T - context_length):
                self.index.append((ep_name, t))

    def __getitem__(self, idx):
        ep_name, t0 = self.index[idx]
        t1 = t0 + self.context_length
        ep = self.h5[f'episodes/{ep_name}']

        return {
            'spectral_occupancy': ep['states/spectral_occupancy'][t0:t1],
            'channel_gsnr':       ep['states/channel_gsnr'][t0:t1],
            'channel_power':      ep['states/channel_power'][t0:t1],
            'lp_features':        ep['states/lp_features'][t0:t1],
            'lp_mask':            ep['states/lp_mask'][t0:t1],
            'global_features':    ep['states/global_features'][t0:t1],
            'actions':            ep['actions'][t0:t1-1],
            'adjacency':          self.h5['topology/adjacency'][()],
            'link_endpoints':     self.h5['topology/link_endpoints'][()],
        }
```

### 8.2 Normalization

Store per-feature statistics computed over the training set:

```json
{
    "channel_gsnr": {"mean": 18.5, "std": 6.2, "min": -2.0, "max": 42.0},
    "channel_power": {"mean": -1.0, "std": 3.5, "min": -10.0, "max": 6.0},
    "global_features": {
        "worst_margin": {"mean": 3.2, "std": 4.1},
        "avg_margin": {"mean": 8.5, "std": 3.8},
        "spectral_utilization": {"mean": 0.45, "std": 0.22}
    }
}
```

---

## 9. Implementation Order

### Phase 1 — Core infrastructure (week 1)
1. HDF5 writer class with the schema above
2. GNPy wrapper: topology builder + full network evaluator
3. State encoder: extract all features from GNPy output
4. Per-step validation function

### Phase 2 — Policies (week 2)
5. Provisioning policy (simplest — pure addition)
6. Recovery policy (add + remove, two phases)
7. Margin optimization policy (requires targeting worst LP)
8. Load balancing policy (requires utilization tracking)

### Phase 3 — Generation & validation (week 3)
9. Generate 100 episodes per policy, validate
10. Fix any issues found by validation
11. Full generation (2,000 episodes)
12. Dataset-level validation
13. Compute and save normalization statistics

### Phase 4 — DataLoader & smoke test (week 3–4)
14. PyTorch DataLoader with sub-trajectory sampling
15. Smoke test: train a tiny model for 1 epoch, verify loss decreases
16. Verify inter-channel coupling in loaded batches
