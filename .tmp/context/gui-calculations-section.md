# Feature: GUI Calculations Section

## Description
Add a separate "Calculations" section to the GUI that allows users to:
- Select which models to run
- Configure model-specific parameters
- Run calculations across multiple models
- View results in a unified interface

## Requirements

### UI Structure
1. **Calculations Section**: Separate from current "Run" button
2. **Model Tabs**: One tab per available model (KASCADE, XIGMA-I, Delta, Analytical)
3. **Per-Model Controls**:
   - "Use for calculations" checkbox
   - Model-specific parameters (only shown when checkbox is checked)
4. **Calculate Button**: Runs all selected models
5. **Results Display**:
   - 1D plots: Common figure with legend and different colors per model
   - 2D plots: List to select and view results from a particular model

### Technical Details

#### Current GUI Architecture
- `src/gammaforge/gui/app.py` - Main GUI application (1174 lines)
- `src/gammaforge/gui/model_api.py` - ModelAdapter protocol
- `src/gammaforge/gui/runner.py` - Headless runner
- `src/gammaforge/gui/output.py` - Output/plot handling

#### Model Adapters
Each model has an adapter implementing:
- `capabilities()` → ModelCapabilities
- `extra_params()` → list of extra parameters
- `extra_choices()` → list of dropdown choices
- `params_to_config(fields, quantum)` → ModelConfig
- `run(cfg, n_mc, seed, electrons)` → CommonResults

#### CommonResults (from io/results.py)
- n_photons: int
- mean_energy_eV: float
- energy_spread_eV: float
- divergence_rad: float
- spectrum_1d: Optional[dict] with 'energies_eV', 'dNde_eV_inv', 'dNde_photon_inv'
- spectrum_2d: Optional[dict] with 'theta_x_rad', 'theta_y_rad', 'dNdOmega_sr_inv'

### User's Vision
- "A separate section for calculations"
- "Tabs for all available models with model-specific parameters"
- "Each tab has 'use for calculations' checkmark"
- "If it's on, the numerical parameters are shown and can be tweaked"
- "A 'calculate' button then runs all the selected models"
- "1d plots should be displayed in common figure with legend and different colors per model"
- "For 2d plots, the user should have a list from which they can select to view the results of a particular model"

## Out of Scope
- Modifying existing "Run" button behavior
- Changing model physics or algorithms
- Adding new physics models

## Dependencies
- Current GUI infrastructure
- ModelAdapter protocol
- CommonResults dataclass
