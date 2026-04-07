# Optical Network World Model

JEPA-based world model dataset generator for optical network digital twins.

## Project Structure

```
optical-network-wm/
├── src/optical_wm/
│   ├── core/
│   │   ├── schemas.py           # Data structures, state/action encoding
│   │   ├── gnpy_wrapper.py      # GNPy interface with inter-channel coupling
│   │   └── hdf5_io.py           # HDF5 read/write for efficient storage
│   └── policies/
│       └── provisioning.py      # Expert-like LP provisioning policy
├── tests/
│   ├── conftest.py              # Path setup for tests
│   ├── test_level0_schemas.py   # Schema tests (no GNPy needed)
│   ├── test_level1_gnpy.py      # GNPy wrapper tests (Docker required)
│   └── test_level2_provisioning.py  # Provisioning policy tests
├── docs/
│   └── dataset_spec.md          # Full dataset specification
└── data/                        # Generated datasets (.gitignore'd)
```

## Testing Pyramid

### Level 0 — Run anywhere (no GNPy)

```bash
pip install -e .
python tests/test_level0_schemas.py
```

### Level 1 — Run inside GNPy Docker

```bash
docker run -it --rm -v "C:\optical-network-wm:/work" telecominfraproject/oopt-gnpy
cd /work
pip install h5py networkx
pip install -e .
python tests/test_level1_gnpy.py
```

### Level 2 — Policy tests (GNPy Docker)

```bash
python tests/test_level2_provisioning.py
```

## Test Results

| Level | Tests | Status |
|-------|-------|--------|
| 0 — Schemas | 7/7 | ✅ Passed |
| 1 — GNPy wrapper | 4/4 | ✅ Passed (coupling confirmed: 0.095 dB) |
| 2 — Provisioning | 6 tests | ⏳ To verify |

## Key Design Decisions

- **Real GNPy only** — no mock physics. Every transition uses full GNPy propagation.
- **Inter-channel coupling** — all channels evaluated together (XPM/FWM via GN model).
- **Full graph state** — ~2,400 dimensions per step (not 14 scalar summaries).
- **Intentional policies** — expert-like, goal-directed trajectories (not random actions).
- **HDF5 storage** — compressed, fast loading for training.
