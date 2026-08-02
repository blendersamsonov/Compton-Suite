# GUI Integration via ModelAdapter

The GUI (`src/gammaforge/gui/app.py`) is model-agnostic and plugs in physics engines through the `ModelAdapter` protocol. This decouples the GUI from any specific simulation engine, allowing new models to be added without touching GUI code.

## The ModelAdapter Protocol

Every model adapter must implement three methods:

```python
class ModelAdapter(Protocol):
    def model_params(self) -> list[tuple[str, float | str, str]]:
        """Model-specific parameters as (label, default, key) triples.
        
        Used by GUI to populate the Model Parameters panel. Default can be
        float (numeric) or str (choice/enum). For choice fields, also
        implement model_choices()."""
        ...
    
    def model_choices(self) -> dict[str, list[str]]:
        """Optional: dict mapping parameter keys to allowed string values.
        
        If a key appears here, GUI renders a dropdown instead of text entry.
        Example: {"device_preference": ["auto", "gpu", "cpu"]}"""
        return {}
    
    def run(self, job: Job) -> Photons:
        """Run the simulation with compiled config; return results.
        
        Job bundles:
        - interaction: InteractionParameters (shared beam+laser)
        - electrons: Bunch (pre-sampled macroparticles)
        - output: OutputSpec (resolution knobs)
        - seed: int (random seed)
        - extra: dict (this model's model_params() values, read live from GUI)"""
        ...
```

## Results Contract

Every model's `run()` returns `gammaforge.io.photons.Photons`. The spectrum can be one of two shapes:

- **`SampledPhotonSpectrum`** (unbinned, per-macroparticle): Has `weight` field; used by Monte Carlo models (kascade).
- **`BinnedPhotonSpectrum`** (smooth binned density): Has `dNdE_per_eV`/`rate`/`density` fields; used by semi-analytic models (xigma-i, delta, analytical).

**Duck-type these shapes, never use `isinstance()` against both boundaries simultaneously:**
```python
if hasattr(spectrum, "weight"):  # Sampled
    ...
elif hasattr(spectrum, "dNdE_per_eV"):  # Binned
    ...
```

This is critical because separate model packages define their own structurally-identical dataclasses to avoid coupling to this GUI package.

## Model Registration

The `discover_models()` function in `src/gammaforge/models/api.py` registers all available adapters:

- **kascade** (`models/kascade/kascade_adapter.py`) — CPU Monte Carlo, always available.
- **xigma-i** (`models/xigma_i/adapter.py::XigmaAdapter`) — GPU tabulated pipeline, greyed out if cupy/CUDA unavailable.
- **delta** (`models/xigma_i/adapter.py::DirectAdapter`) — brute-force per-particle binning (xigma-i's Stage 0 only), greyed out if cupy/CUDA unavailable.
- **analytical** (`models/analytical.py::Adapter`) — closed-form estimate, always available, runs as real-time preview.

## Adding a New Observable

Pattern for adding a new spectral visualization (e.g., polarization, temporal envelope):

1. **Define dataclass pair** in `src/gammaforge/io/photons.py`:
   - `SampledPolarizationSpectrum` (with `weight` field for Monte Carlo models)
   - `BinnedPolarizationSpectrum` (with binned arrays for semi-analytic models)
   - Add optional `Photons.polarization_spectrum: SampledPolarizationSpectrum | BinnedPolarizationSpectrum | None`

2. **Populate in adapter's run() method**:
   - kascade: usually cheap since raw arrays already computed.
   - xigma-i/delta: check `core.py` kernel functions before adding new computation.
   - analytical: derive from analytic formulas.

3. **Render in GUI** (`src/gammaforge/gui/app.py`):
   - Add tab in `_build_plot_area()`.
   - Add `_render_polarization_spectrum()` method using duck-typing (not isinstance).
   - Gate visibility in `_apply_model_capabilities()` if not all models support it.

4. **Test** (`scripts/headless_test.py`):
   - Extend `test_model()` to verify the new field is populated for each adapter.

5. **Verify**: `python3 scripts/headless_test.py` to confirm all models run without errors.
