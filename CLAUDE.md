# xigma_i (XIGMA-I)

Semi-analytic, GPU-only Compton-scattering calculation: computes photon
yield, energy spectrum, and angular/spectral-angular distributions by
integrating a differential cross section over the electron-laser beam
overlap (Monte-Carlo-sampled trajectories, not an event generator — no
per-photon/per-electron discrete emission tracking). See `passport.md` for
the full physics "passport" (trust level **C** for the linear/classical
regime, **D** for the nonlinear a0-emulation regime; no unit tests, no
cross-code validation as of this writing).

This is one of two independent physics engines plugged into the
`compton-gui` project (sibling repo) via `src/xigma_i/gui_adapter.py`.

## Files

- `core.py` — the `Compton` class + CUDA raw kernels (`cupyx.jit.rawkernel`).
  - `set_electron_parameters`/`set_laser_parameters`/`set_foci_displacement` — plain numpy setup, **CGS-like units (cm, erg)**, unlike `dfe5_compton_mc`'s SI convention.
  - `calculate_intersection(theta_num, particles_amount)` — the main GPU call. Launches `particle_kernel` over a `(particles_amount, theta_num²)` grid × `N_STEPS=128` threads; cost is `O(particles_amount · theta_num² · N_STEPS)` — **do not** pass a large `particles_amount` naively (dfe5's GUI default `n_mc=200,000` would be a catastrophic kernel launch here; `gui_adapter.py` clamps it to `[512, 8192]`).
  - `calculate_spectrum` / `calculate_angular_spectrum` — angle-integrated / angle-resolved spectral density; `calculate_angular_spectrum` already accepts **arbitrary** `theta_x`/`theta_y` device arrays, not just a precomputed grid — useful for on-demand angular-range queries without new kernel work.
  - `particle_kernel` — per (particle, angle-cell, time-step) thread. Computes real position `x, y, z` in **dimensionless, k0-scaled units** (`x_dimensionless = k0_las · x_cm`; `k0_las = 2π/lambda_l`) and a local weight `f_cur`. Deposits `f_cur` into three outputs via atomic-add: `intersect` (angle grid), `envelope` (1D time histogram, `time_envelope`/`env_ts`), and — added 2026-07 — `spatial` (2D bilinear x/y histogram, `spatial_envelope`/`spatial_x_edges`/`spatial_y_edges`, `N_SPATIAL=64` per axis).
  - **Normalization gotcha**: `coef` (the physical-units prefactor applied to `intersection`/`time_envelope`) bakes in an angular-Gaussian-profile normalization term (`1/(2π·sigma_thx·sigma_thy)`) that does **not** carry over to a spatial density — reusing it directly for `spatial_envelope` caused a float32 overflow during development (a Python-float scalar division computed in float32 before ever touching the array). The fix: `spatial_envelope` is instead self-normalized by rescaling its raw accumulation so that `spatial_envelope.sum() * dx_cm * dy_cm == calculate_total()`'s yield, exactly — correct by construction since both are sums of the exact same `f_cur` accumulation, just partitioned differently (by angle vs. by position). If you add a fourth deposition axis, don't reuse `coef` for it without checking this reasoning applies.
  - `spectrum_kernel` — per-photon spectral sampling kernel (angle-resolved).
- `deposition.py` (Stage 1) — array-module-agnostic (`xp = cp.get_array_module(...)`, works with plain numpy *or* cupy): `deposit_nearest`/`deposit_cic`, `Grid4D`/`Table` (gamma, theta_x, theta_y, a0) — **not** real-space position, despite the name "deposition"; this bins emission-angle/intensity phase space, unrelated to the spatial-distribution feature above.
- `particles.py` (Stage 0) — pure numpy/CPU: `sample_bunch`, `push_and_sample`. No cupy dependency.
- `spectrum4d.py` (Stage 2) — GPU: `spectrum_kernel_4d`, consumes a `deposition.Table`.
- `reference.py` — pure numpy CPU validation/reference implementations (`angle_integrated_spectrum`, `interp4d`, `spectrum_from_table`, `direct_binning_spectrum`). **`direct_binning_spectrum` has a known, unresolved ~3000-4000x normalization bug**, documented in its own docstring — don't trust its absolute output without re-deriving.
- `gui_adapter.py` — the bridge to `compton-gui`'s `ModelAdapter` contract.
  - Never imports `cupy`/`core` at module scope — only inside `available()`/`run_simulation()`/`spectrum_in_angular_range()` — so `import xigma_i.gui_adapter` degrades gracefully when cupy/CUDA isn't installed (the GUI wraps that import in `try/except` and shows the model greyed-out instead of crashing).
  - Defines its **own local** `BinnedSpectrum`/`BinnedAngularSpectrum`/`BinnedTemporalEnvelope`/`BinnedSpatialDistribution`/`AngularRangeSpectrumResult` dataclasses — deliberately *not* imported from `compton_gui.model_api`, so this package doesn't have to depend on the GUI project. They're structurally identical, duck-type compatible, but **not the same Python class** — this tripped up an `isinstance`-based check on the GUI side once (see that repo's CLAUDE.md); don't "fix" it by importing `compton_gui` here, the decoupling is intentional.
  - `Config` mirrors `dfe5_compton_mc.Config`'s field names/SI units where a physical mapping exists (so the GUI's model-agnostic spread-estimate formula keeps working regardless of active model), but converts to CGS at the `set_*_parameters` call boundary. `crossing_angle` must be `0.0` (head-on only — `Compton` has no crossing-angle support at all). `quantum` is accepted for interface symmetry but has no effect; use `emulate_nonlinearity` for xigma-i's actual (unrelated) nonlinearity axis.
  - `XigmaAdapter` caches `self._last_results` (which itself carries a private `_compton`/`_gamma_0`/`_sigma_gamma_0`, set by `run_simulation`) so `spectrum_in_angular_range()` can reuse the built `Compton` instance for a fresh on-demand `calculate_angular_spectrum()` call over a user-picked window, without re-running the whole simulation.

## GPU requirement

`core.Compton`'s methods (`calculate_intersection`, `calculate_spectrum`,
`calculate_angular_spectrum`) and `spectrum4d.py` are **hard cupy/GPU
dependencies, no CPU fallback**. Only `deposition.py`, `particles.py`, and
`reference.py` are numpy-capable/CPU-only.

## Testing

No unit tests in this repo. Validated via the sibling `compton-gui` repo's
`scripts/headless_test.py` (calls `params_to_config → run → validate_results`
plus the temporal/spatial/angular fields through `XigmaAdapter`), and via
ad hoc GPU scripts during development.

On this dev machine: system Python has no pip/cupy/matplotlib. Use the
`miniforge3` conda env named `core` (has cupy 14.0.1, numpy, matplotlib,
tkinter, all working against the local GTX 1660 Ti):

```bash
conda run -n core --no-capture-output python3 /path/to/compton-gui/scripts/headless_test.py
```

`conda run` silently swallows stdout unless you pass `--no-capture-output`.
`cupyx.jit.rawkernel` needs to introspect real source — write test scripts to
an actual `.py` file and run that; it can't compile a kernel defined via
`python3 -c "..."`/stdin (`RuntimeError: JIT needs access to the Python
source code ... cannot be retrieved within the Python interactive
interpreter`).

## Relationship to sibling repos

- `../MC-Kost` (or wherever it's checked out) — `dfe5_compton_mc`, the other
  physics engine plugged into the same GUI. No dependency either direction.
- `../compton-gui` — the shared Tkinter GUI. Depends on this repo only
  through `gui_adapter.py`'s contract (never touches `core.py`/`deposition.py`/etc.
  directly). `bootstrap.py` there assumes this repo's `src/` sits at a sibling
  path unless overridden via `COMPTON_GUI_XIGMA_SRC`.

## pyproject.toml note

Currently pins `requires-python = ">=3.14"` and `cupy>=13.6`. Prebuilt cupy
wheels can lag brand-new CPython releases; if `pip install cupy-cudaXXx`
doesn't find a matching wheel, a conda/mamba env (conda-forge) is the more
reliable install path — confirmed working on this machine via the `core` env
above.
