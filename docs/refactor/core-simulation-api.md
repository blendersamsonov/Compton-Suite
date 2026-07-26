# ComptonSuite Core Refactor Plan

**Status**: Planned — not yet implemented  
**Location**: `docs/refactor/core-simulation-api.md`  
**Target**: Extract model-agnostic simulation core from GUI into `compton_suite.core`

---

## Goal

Create a reusable `compton_suite.core` package with a straight-line data flow:

```
SimulationConfig → MacroBunch → CollisionParams → ModelProtocol.run() → CommonResults
```

GUI becomes a thin consumer. Models keep internal caching (XIGMA-I table) via optional protocol methods.

---

## New Module Structure

```
src/compton_suite/
├── core/
│   ├── __init__.py              # Public API: run_simulation, SimulationConfig, etc.
│   ├── protocol.py              # ModelProtocol, ModelCapabilities, ModelParameter
│   ├── collision.py             # CollisionParams (SI pint quantities only)
│   ├── simulation.py            # run_simulation(), SimulationConfig, SimulationResult
│   └── adapters/
│       ├── __init__.py          # discover_adapters(), get_adapter()
│       ├── kascade_adapter.py
│       ├── xigma_adapter.py
│       ├── xigma_direct_adapter.py
│       └── analytical_adapter.py
├── gui/                         # THIN consumer only
│   ├── app.py                   # Uses compton_suite.core.* ONLY
│   └── (model_api.py, adapters/ DELETED)
├── models/                      # PHYSICS ENGINES ONLY (no adapters)
│   ├── kaskade/kascade.py
│   ├── xigma_i/...              # config.py loses build_params()
│   ├── xigma_direct/...
│   └── analytical/analytical.py
└── io/                          # UNCHANGED (compton_io)
```

---

## Key Types

### `CollisionParams` (core/collision.py)
- **SI pint Quantities only** — no duplicate SI/CGS storage
- Models convert to their internal units on their end:
  - kascade/analytical: use SI magnitudes directly
  - xigma_i/xigma_direct: call `.to_cgs()` for CGS-Gaussian view
- XIGMA's a0 formula computed once in `__post_init__` (known ~49% diff vs `laser.a0_focus`)

### `ModelProtocol` (core/protocol.py)
```python
def capabilities() -> ModelCapabilities
def available() -> tuple[bool, str]
def model_parameters() -> list[ModelParameter]
def params_to_config(shared_fields, model_params, quantum) -> (config, extra)
def run(config, collision_params, *, electrons, model_params) -> CommonResults
def load_ele_file(path) -> MacroBunch
def ele_file_summary(bunch) -> dict
def spectrum_in_angular_range(result, theta_x_range, theta_y_range, **kwargs)

# Optional caching (XIGMA-I):
def prepare_cached_state(collision_params, model_params) -> Any
def run_with_cache(config, collision_params, *, electrons, model_params, cached_state)
def invalidate_cache(cached_state, changed_fields) -> bool
```

### `SimulationConfig` (core/simulation.py)
```python
model_name: str                    # "kascade" | "xigma" | "xigma_direct" | "analytical"
beam: GaussianElectronBeam | None
laser: GaussianParaxialLaser | None
ele_file_path: str | None
shared_fields: dict | None         # flat SI dict
model_params: dict | None          # model-specific numerical params
quantum: bool = False
n_particles: int | None            # override model default
seed: int = 0
device: str | None                 # 'auto' | 'gpu' | 'cpu'
beta_ff: float = 0.0
ellipticity: float = 0.0
phi_pol: float = 0.0
a0_max: float = 0.5
crossing_angle_rad: float = 0.0
geometry_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
```

---

## Migration Checklist

| Step | Task |
|------|------|
| 1 | Create `core/protocol.py`, `core/collision.py`, `core/simulation.py` |
| 2 | Create `core/adapters/` with 4 adapters (move from `models/*/gui_adapter.py`) |
| 3 | Update `core/__init__.py` — re-export public API |
| 4 | Update `gui/app.py` — use `core.*` only, remove `model_api`, `adapters/` |
| 5 | **DELETE** `gui/model_api.py` and `gui/adapters/` |
| 6 | Update `validation/runners.py` — use `core.run_simulation` |
| 7 | Update `validation/scenarios.py` — use `core.collision.build_collision_params` |
| 8 | Update `models/xigma_i/config.py` — remove `build_params` (moved to core) |
| 9 | Update `src/compton_suite/__init__.py` — re-export core API |
| 10 | Update `scripts/headless_test.py`, `scripts/physics_params_demo.py` |
| 11 | Run tests: `pytest tests/`, `python validation/run_cross_validation.py` |

---

## Backward Compatibility

- `compton_suite.core` = **new public API**
- `compton_suite.gui.model_api` **removed** — GUI updated
- `compton_io.collision.build_params` **kept** (deprecated, delegates to core)
- `xigma_i.config.build_params` **removed** — use `core.collision.build_collision_params`
- Validation suite uses new API, produces same `CommonResults`

---

## Example Usage (Post-Refactor)

```python
from compton_suite.core import run_simulation, SimulationConfig
from compton_suite.io.bunch import GaussianElectronBeam
from compton_suite.io.laser import GaussianParaxialLaser

beam = GaussianElectronBeam(...)
laser = GaussianParaxialLaser(...)
config = SimulationConfig(model_name="xigma", beam=beam, laser=laser,
                          model_params={"n_particles_01": 100_000, "device_preference": "gpu"})
result = run_simulation(config)
print(f"Total yield: {result.results.total_yield:.3e}")
```