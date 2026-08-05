#!/usr/bin/env python3
r"""
app.py
======

Desktop GUI front-end for pluggable Compton-scattering physics models.

Layout (left | right, resizable split via PanedWindow):

  LEFT (IO panels):
  * "Electrons" panel (blue)  -- electron-bunch parameters, MC sample count,
    and random seed.  When a 6-D ``.ele`` file is loaded via
    ``File -> Load file *.ele...`` the physical beam entries turn read-only
    (n_mc/seed remain editable).
  * "Laser" panel (blue)      -- laser + relative-position parameters; the peak
    normalises vector potential a_0 is derived live from the laser parameters.
  * "Compton photons" panel (yellow) -- collimation angles, Calculate button,
    and post-run statistics (fluxes, photon multiplicities, recoil parameter).
  * "Model Parameters" panel (grey) -- model-specific numeric fields, rebuilt
    whenever the active model changes.
  * "Analytical Preview" panel (grey) -- always-on fast analytical estimate
    running alongside the selected model.
  RIGHT (results):
  * A tabbed plot area:
      1. Spectrum & Electron  -- full (4*pi) + collimated photon spectrum
         + initial-vs-final electron energy distribution.
      2. Temporal Envelope    -- photon-emission rate/count vs. time.
      3. Spatial Distribution -- transverse (x, y) distribution of photons
         at emission.
      4. Angular Distribution -- angle-only (theta_x, theta_y) photon density.

This GUI is model-agnostic: physics engines are plugged in through the
``model_api.ModelAdapter`` registry (see ``model_api.py``) instead of a
hardcoded import.``.

Run: ``python3 -m gammaforge.gui.app`` or ``scripts/run_gui.py``.
"""

from __future__ import annotations

import dataclasses
import os
import queue
import threading
import traceback

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from gammaforge.io.bunch import (
    GaussianElectronBeam,
    drift,
    fit_gaussian,
    sample_gaussian_bunch,
    sigma_from_emittance,
)
from gammaforge.io.interaction import InteractionParameters, recoil_parameter
from gammaforge.io.io_formats.sdds import load_elegant_ele
from gammaforge.io.io_formats.yaml_spec import (
    load_electron_beam,
    load_laser,
    save_electron_beam,
    save_laser,
)
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.units import (
    C_LIGHT,
    MEC2_EV,
    NoConvention,
    PhysicalMeaning,
    PhysicalQuantity,
    TimeConvention,
    WidthConvention,
    fwhm_to_sigma_intensity,
    sigma_intensity_to_fwhm,
    sigma_intensity_to_w0,
    ureg,
    w0_to_sigma_intensity,
)
from gammaforge.models.api import (
    AXIS_ENERGY,
    AXIS_THETA_X,
    AXIS_THETA_Y,
    AXIS_TIME,
    AXIS_X,
    AXIS_Y,
    Job,
    ModelAdapter,
    Results,
    discover_models,
    find_slice,
    total_yield,
    validate_results,
)

# panel colours
BLUE = "#d6e4f5"
RED = "#f5d6d6"
YELLOW = "#f7f0c0"
GREY = "#e4e4e4"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _float_or_none(var: tk.StringVar):
    try:
        return float(var.get())
    except (ValueError, tk.TclError):
        return None


def _pq(value: float, unit: str, meaning: PhysicalMeaning, convention=None) -> PhysicalQuantity:
    """Shortcut to build a PhysicalQuantity -- the GUI wraps its own raw
    floats directly now (no `*_from_shared_fields` factory indirection;
    same pattern `io/bunch.py`/`io/laser.py` use internally)."""
    return PhysicalQuantity(value, unit, meaning, convention)


def _native(value: float, native_unit: str, target_unit: str) -> float:
    """Convert a raw field value from the GUI's own display unit
    (``native_unit``, e.g. "nanocoulomb") to the unit a ``PhysicalQuantity``
    is about to be built with (``target_unit``) -- via real pint dimensional
    analysis, not a hand-typed scale factor."""
    return ureg.Quantity(value, native_unit).to(target_unit).magnitude

# ---------------------------------------------------------------------------
# Widget helper -- coloured field grid
# ---------------------------------------------------------------------------
def add_field_grid(parent, specs, fields, n_cols, bg, width=10, group_starts=(),
                   state="normal", choices=None):
    """Place (label, default, key) triples in an n_cols-wide grid inside a
    coloured (bg) panel, using classic tk widgets so the background shows.

    Returns a list of (widget, label key) for the fields that were placed,
    so the caller can flip their ``state`` (e.g. disabled when the panel is
    being used as an output display).

    If ``choices`` is provided (dict mapping key -> list of allowed strings),
    a ttk.Combobox (dropdown) is created instead of an Entry for those keys.
    """
    choices = choices or {}
    entries = []
    for idx, (label, default, key) in enumerate(specs):
        row, col = divmod(idx, n_cols)
        lpad = 18 if idx in group_starts else 8
        tk.Label(parent, text=label, bg=bg, anchor="w").grid(
            row=row, column=col * 2, sticky="w", padx=(lpad, 3), pady=3)
        var = tk.StringVar(value=str(default))
        if key in choices:
            # Choice field: use Combobox (dropdown)
            ent = ttk.Combobox(parent, textvariable=var, values=choices[key],
                               width=width, state="readonly" if state == "normal" else "disabled",
                               justify="right")
        else:
            # Numeric/text field: use Entry
            ent = tk.Entry(parent, textvariable=var, width=width, justify="right",
                           state=state)
        ent.grid(row=row, column=col * 2 + 1, padx=(0, 10), pady=3)
        fields[key] = var
        entries.append((ent, key))
    return entries


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class ComptonGUIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compton-GUI")
        self.geometry("1280x960+20+10")

        self.fields: dict[str, tk.StringVar] = {}
        self.res: Results | None = None
        self.preview_res: Results | None = None    # always-on analytical model's Results | None
        self.interaction_used: InteractionParameters | None = None   # (laser, electrons) used for self.res
        self.rep_rate_hz = 1.0
        self.q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None

        # Loaded .ele file state (None unless File -> Load file *.ele... was used).
        self.loaded_bunch = None
        self.loaded_path: str | None = None

        # Model registry: discover once, pick kascade as the startup default
        # (it's always available; xigma-i/delta simply aren't registered at
        # all -- absent from the Model menu -- if cupy/CUDA/numba isn't
        # usable on this machine).
        self.models = discover_models()
        self.model_var = tk.StringVar(value="kascade")
        self.active_adapter: ModelAdapter = self.models["kascade"]

        # The fast analytical model runs
        # automatically alongside whichever model is selected, on every
        # Calculate click -- a real-time preview and base sanity check, not
        # one of the models the Model menu switches between. None if it
        # isn't registered/available (e.g. gammaforge not discoverable).
        self.analytical_adapter = self.models["analytical"]

        # Cached list of (Entry widget, key) tuples for the Electron panel
        # input fields, so we can flip their state when the panel switches
        # between input-mode and display-mode.
        self._electron_entries: list[tuple[tk.Entry, str]] = []
        self._sample_entries: list[tuple[tk.Entry, str]] = []
        # Trace ids we attach to electron-panel StringVars so we can detach
        # them while the panel is in display mode (otherwise the
        # ``_update_derived`` callback would clobber our output values).
        self._electron_traces: list[tuple[tk.StringVar, str]] = []

        # --- build layout ---
        self._build_menu()

        # Resizable split (PanedWindow must own its pane frames)
        self._paned = tk.PanedWindow(self, orient="horizontal",
                                     sashwidth=6, sashrelief="groove")
        self._paned.pack(fill="both", expand=True)

        # Left pane: all IO panels (electrons, laser, compton, model params, preview)
        self._left_frame = tk.Frame(self._paned)
        self._build_electrons_panel()
        self._build_laser_panel()
        self._build_compton_panel()
        self._build_model_params_panel()
        self._build_output_panel()
        self._build_preview_panel()

        # Right pane: plot area
        self._plot_frame = tk.Frame(self._paned)
        self._build_plot_area()

        # Register panes with the PanedWindow
        self._paned.add(self._left_frame, minsize=300, width=580)
        self._paned.add(self._plot_frame, minsize=400)

        self._wire_live_updates()
        self._update_derived()

    # ---- menu -----------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load file *.ele...", command=self._load_ele)
        filemenu.add_command(label="Clear loaded file",
                              command=self._clear_loaded_ele)
        filemenu.add_separator()
        filemenu.add_command(label="Save plots...", command=self._save_fig)
        filemenu.add_separator()
        filemenu.add_command(label="Save electron beam config...", command=self._save_beam_config)
        filemenu.add_command(label="Load electron beam config...", command=self._load_beam_config)
        filemenu.add_command(label="Save laser config...", command=self._save_laser_config)
        filemenu.add_command(label="Load laser config...", command=self._load_laser_config)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.filemenu = filemenu
        self._ele_menu_indices = (0, 1)  # "Load file...", "Clear loaded file"

        modelmenu = tk.Menu(menubar, tearoff=0)
        for name, adapter in self.models.items():
            # Skip the analytical model -- it runs always in the background
            # as a preview, not as a user-selectable model.
            if name == "analytical":
                continue
            modelmenu.add_radiobutton(
                label=name, variable=self.model_var, value=name,
                command=self._on_model_selected)
        menubar.add_cascade(label="Model", menu=modelmenu)
        self.modelmenu = modelmenu

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.config(menu=menubar)

    # ---- model selection --------------------------------------------------
    def _on_model_selected(self):
        name = self.model_var.get()
        self.active_adapter = self.models[name]
        self._rebuild_model_params_panel()

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Compton-GUIde\n\n"
            "Model-agnostic GUI front-end for pluggable Compton-scattering "
            "physics engines (kascade, xigma-i, ...).")

    def _save_fig(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=150)

    # ---- YAML config save/load ------------------------------------------
    def _save_beam_config(self):
        """Save the current Electrons-panel fields as a
        ``gaussian_6d_waist`` YAML config (``io.io_formats.yaml_spec``)."""
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml", filetypes=[("YAML", "*.yaml *.yml")])
        if not path:
            return
        try:
            beam, _laser = self._build_beam_and_laser()
            save_electron_beam(beam, path)
        except Exception as e:
            messagebox.showerror("Cannot save electron beam config", str(e))

    def _load_beam_config(self):
        """Load a ``gaussian_6d_waist`` YAML config and populate the
        Electrons-panel input fields (never touches ``self.loaded_bunch`` --
        a YAML config is initial parameters for further sampling, not raw
        particle data like a loaded ``.ele`` file)."""
        path = filedialog.askopenfilename(
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if not path:
            return
        try:
            beam = load_electron_beam(path)
        except Exception as e:
            messagebox.showerror(
                "Cannot load electron beam config",
                f"Failed to parse {os.path.basename(path)}:\n\n{e}")
            return

        if self.loaded_bunch is not None:
            self._clear_loaded_ele()

        gamma = beam.gamma0
        emit_geom_x_m = beam.emit_geom_x_m.to_unit("meter").magnitude
        emit_geom_y_m = beam.emit_geom_y_m.to_unit("meter").magnitude
        sigma_x_m = beam.sigma_x_m.to_unit("meter").magnitude
        sigma_y_m = beam.sigma_y_m.to_unit("meter").magnitude
        # Invert _build_beam_and_laser's forward math exactly (GUI convention
        # is emit_norm = emit_geom * gamma, no beta0 factor; beta_x/y are a
        # GUI-only convenience input, not a GaussianElectronBeam field, so
        # they're reconstructed from sigma_from_emittance's own inverse:
        # beta = sigma^2 / emit_geom) so save->load->save is a fixed point.
        emit_x_norm_m = emit_geom_x_m * gamma
        emit_y_norm_m = emit_geom_y_m * gamma
        beta_x_m = (sigma_x_m ** 2 / emit_geom_x_m) if emit_geom_x_m > 0 else 0.0
        beta_y_m = (sigma_y_m ** 2 / emit_geom_y_m) if emit_geom_y_m > 0 else 0.0

        self._duration_convention_var.set("RMS")
        self.fields["mean_energy_MeV"].set(f"{beam.total_energy.to('megaelectron_volt').magnitude:.6g}")
        self.fields["rel_spread_pct"].set(f"{beam.rel_energy_spread_rms * 100:.6g}")
        self.fields["charge_nC"].set(f"{beam.bunch_charge_C.to_unit('nanocoulomb').magnitude:.6g}")
        self.fields["bunch_duration_ps"].set(f"{beam.sigma_t_s.to_unit('picosecond').magnitude:.6g}")
        self.fields["emit_x_mmmrad"].set(f"{_native(emit_x_norm_m, 'meter', 'millimeter * milliradian'):.6g}")
        self.fields["emit_y_mmmrad"].set(f"{_native(emit_y_norm_m, 'meter', 'millimeter * milliradian'):.6g}")
        self.fields["beta_x_m"].set(f"{beta_x_m:.6g}")
        self.fields["beta_y_m"].set(f"{beta_y_m:.6g}")
        self.fields["chirp_h"].set(f"{beam.chirp_h:.6g}")
        self.fields["dispersion_x"].set(f"{beam.dispersion_x:.6g}")
        self.fields["dispersion_y"].set(f"{beam.dispersion_y:.6g}")
        self.fields["alpha_x"].set(f"{beam.alpha_x:.6g}")
        self.fields["alpha_y"].set(f"{beam.alpha_y:.6g}")
        self._update_derived()

    def _save_laser_config(self):
        """Save the current Laser-panel fields as a ``gaussian_paraxial``
        YAML config (``io.io_formats.yaml_spec``)."""
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml", filetypes=[("YAML", "*.yaml *.yml")])
        if not path:
            return
        try:
            _beam, laser = self._build_beam_and_laser()
            save_laser(laser, path)
        except Exception as e:
            messagebox.showerror("Cannot save laser config", str(e))

    def _load_laser_config(self):
        """Load a ``gaussian_paraxial`` YAML config and populate the
        Laser-panel input fields."""
        path = filedialog.askopenfilename(
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if not path:
            return
        try:
            laser = load_laser(path)
        except Exception as e:
            messagebox.showerror(
                "Cannot load laser config",
                f"Failed to parse {os.path.basename(path)}:\n\n{e}")
            return

        self._pulse_convention_var.set("RMS")
        self._waist_convention_var.set("RMS")
        self._waist_input_mode.set("diameter")
        self.fields["laser_wavelength_nm"].set(f"{laser.wavelength_m.to_unit('nanometer').magnitude:.6g}")
        self.fields["laser_energy_mJ"].set(f"{laser.pulse_energy_J.to_unit('millijoule').magnitude:.6g}")
        self.fields["pulse_duration_ps"].set(f"{laser.duration_rms_s.to_unit('picosecond').magnitude:.6g}")
        self.fields["crossing_angle"].set(f"{laser.crossing_angle:.6g}")
        self.fields["z_mismatch_mm"].set(f"{laser.focus_z_m.to_unit('millimeter').magnitude:.6g}")
        self.fields["beta_ff"].set(f"{laser.beta_ff:.6g}")
        self.fields["phi_pol"].set(f"{laser.phi_pol:.6g}")
        self.fields["ellipticity"].set(f"{laser.ellipticity:.6g}")

        wx_um = laser.waist_rms_x_m.to_unit("micrometer").magnitude
        wy_um = laser.waist_rms_y_m.to_unit("micrometer").magnitude
        self.fields["waist_diameter_um"].set(f"{2.0 * wx_um:.6g}")
        elliptical = abs(wy_um - wx_um) > 1e-9 * max(abs(wx_um), abs(wy_um), 1.0)
        self._elliptical_waist_var.set(elliptical)
        self.fields["waist_diameter_y_um"].set(f"{2.0 * wy_um:.6g}")
        self._on_elliptical_waist_toggle()
        self._update_derived()

    # ---- SDDS .ele file load / clear -----------------------------------
    def _load_ele(self):
        """Load a 6-D electron bunch from an SDDS ``.ele`` file and turn the
        Electrons panel into an output display of the parameters that the
        simulation will actually use.

        ``load_elegant_ele`` alone leaves ``gaussian_fit`` unset (a plain
        ``.ele`` carries no bunch-charge/current info, so there's nothing
        to derive one from) -- every ``Bunch`` reaching a ``Job`` needs
        ``gaussian_fit`` populated (see ``io.interaction``'s module
        docstring), so this fits the loaded macroparticles with
        ``fit_gaussian`` and overrides its non-authoritative derived charge
        with the GUI's own ``charge_nC`` field before attaching it."""
        path = filedialog.askopenfilename(
            title="Load 6-D electron distribution",
            defaultextension=".ele",
            filetypes=[("SDDS electron distribution", "*.ele"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            bunch = load_elegant_ele(path)
        except Exception as e:
            messagebox.showerror(
                "Cannot load .ele file",
                f"Failed to parse {os.path.basename(path)}:\n\n{e}")
            return
        try:
            fit = fit_gaussian(bunch)
            charge_nC = _float_or_none(self.fields["charge_nC"])
            if charge_nC is not None:
                fit = dataclasses.replace(
                    fit,
                    bunch_charge_C=_pq(_native(charge_nC, "nanocoulomb", "coulomb"), "coulomb",
                                       PhysicalMeaning.BUNCH_CHARGE, NoConvention.PLAIN),
                )
            bunch.gaussian_fit = fit  # Bunch is a plain (non-frozen) dataclass
            summary = {
                "mean_energy_MeV": fit.total_energy.to("megaelectron_volt").magnitude,
                "rel_spread_pct": fit.rel_energy_spread_rms * 100,
                "bunch_duration_ps": fit.sigma_t_s.to_unit("picosecond").magnitude,
                "emit_x_mmmrad": fit.emit_norm_x.to("millimeter * milliradian").magnitude,
                "emit_y_mmmrad": fit.emit_norm_y.to("millimeter * milliradian").magnitude,
                "beta_x_m": fit.beta_star_x.to("meter").magnitude,
                "beta_y_m": fit.beta_star_y.to("meter").magnitude,
                "sigma_ex_um": fit.sigma_x_m.to_unit("micrometer").magnitude,
                "sigma_ey_um": fit.sigma_y_m.to_unit("micrometer").magnitude,
            }
        except Exception as e:
            messagebox.showerror(
                "Cannot compute parameters",
                f"Loaded {os.path.basename(path)} but failed to derive "
                f"parameters:\n\n{e}")
            return
        self.loaded_bunch = bunch
        self.loaded_path = path
        self._set_electron_panel_display(summary, os.path.basename(path))

    def _clear_loaded_ele(self):
        """Drop the loaded ``.ele`` bunch and restore the Electrons panel
        to its normal input mode (using whatever is currently in the
        entry fields)."""
        if self.loaded_bunch is None:
            return
        self.loaded_bunch = None
        self.loaded_path = None
        self._set_electron_panel_input()

    def _set_electron_panel_display(self, summary, basename):
        """Flip the Electron-panel entries to read-only and write the
        parameters derived from the loaded file."""
        # Detach live-update traces so they don't clobber our display values
        for var, tname in self._electron_traces:
            try:
                var.trace_remove("write", tname)
            except tk.TclError:
                pass
        self._electron_traces = []

        values = {
            "mean_energy_MeV": f"{summary['mean_energy_MeV']:.4g}",
            "rel_spread_pct":  f"{summary['rel_spread_pct']:.4g}",
            "bunch_duration_ps": f"{summary['bunch_duration_ps']:.4g}",
            "emit_x_mmmrad":   f"{summary['emit_x_mmmrad']:.4g}",
            "emit_y_mmmrad":   f"{summary['emit_y_mmmrad']:.4g}",
            "beta_x_m":        f"{summary['beta_x_m']:.4g}",
            "beta_y_m":        f"{summary['beta_y_m']:.4g}",
        }
        for key, text in values.items():
            self.fields[key].set(text)
        for ent, _key in self._electron_entries:
            ent.config(state="readonly", disabledforeground="#123",
                       readonlybackground=BLUE)

        self.sigma_ex_lbl.config(
            text=f"sigma_x = {summary['sigma_ex_um']:.3f} um")
        self.sigma_ey_lbl.config(
            text=f"sigma_y = {summary['sigma_ey_um']:.3f} um")
        self.electron_mode_lbl.config(
            text=f"(loaded from {basename} - Electrons panel is read-only)")

    def _set_electron_panel_input(self):
        """Restore the Electron-panel entries to editable input mode and
        re-attach the live-update traces."""
        for ent, _key in self._electron_entries:
            ent.config(state="normal")
        self.electron_mode_lbl.config(
            text="(input mode - edit the fields above to define the bunch)")
        # Re-attach the write traces used by _update_derived.
        for k in ("emit_x_mmmrad", "emit_y_mmmrad", "beta_x_m", "beta_y_m",
                  "mean_energy_MeV"):
            tname = self.fields[k].trace_add(
                "write", lambda *a: self._update_derived())
            self._electron_traces.append((self.fields[k], tname))
        self._update_derived()

    # ---- Electrons panel (blue) ----------------------------------------
    def _build_electrons_panel(self):
        p = tk.LabelFrame(self._left_frame, text="ELECTRONS", bg=BLUE, fg="#123",
                      font=("TkDefaultFont", 14, "bold"))
        p.pack(side="top", fill="x", padx=6, pady=(6, 3))

        specs = [
        ("Mean energy [MeV]", 2000, "mean_energy_MeV"),
        ("Rel. spread [%]", 0.1, "rel_spread_pct"),
        ("Charge [nC]", 1, "charge_nC"),
        ("Duration [ps]", 10, "bunch_duration_ps"),
        ("eps_x,n [mm*mrad]", 1.0, "emit_x_mmmrad"),
        ("eps_y,n [mm*mrad]", 1.0, "emit_y_mmmrad"),
        ("beta_x [m]", 0.2, "beta_x_m"),
        ("beta_y [m]", 0.2, "beta_y_m"),
        ("Chirp h [1/m]", 0.0, "chirp_h"),
        ("Dispersion Dx", 0.0, "dispersion_x"),
        ("Dispersion Dy", 0.0, "dispersion_y"),
        ("Twiss alpha_x", 0.0, "alpha_x"),
        ("Twiss alpha_y", 0.0, "alpha_y"),
        ("Propagate distance [m]", 0.0, "drift_L_m"),
        ]
        self._electron_entries = add_field_grid(
            p, specs, self.fields, n_cols=4, bg=BLUE, width=5)

        # Duration convention selector (RMS or FWHM)
        tk.Label(p, text="Duration convention:", bg=BLUE, anchor="w").grid(
            row=0, column=10, sticky="w", padx=(15, 3), pady=3)
        self._duration_convention_var = tk.StringVar(value="RMS")
        duration_convention_menu = tk.OptionMenu(p, self._duration_convention_var,
                                                  "RMS", "FWHM")
        duration_convention_menu.config(bg=BLUE, width=6)
        duration_convention_menu.grid(row=0, column=11, padx=(0, 10), pady=3)

        # derived transverse sizes - show multiple conventions
        self.sigma_ex_frame = tk.Frame(p, bg=BLUE)
        self.sigma_ex_frame.grid(row=1, column=10, columnspan=2, sticky="w", padx=(15, 5), pady=3)
        self.sigma_ex_lbl = tk.Label(self.sigma_ex_frame, text="sigma_x = --", bg=BLUE, anchor="w")
        self.sigma_ex_lbl.pack(side="left")
        self.sigma_ex_fwhm_lbl = tk.Label(self.sigma_ex_frame, text="(FWHM: --)", bg=BLUE, anchor="w", fg="#456")
        self.sigma_ex_fwhm_lbl.pack(side="left", padx=(5, 0))
        self.sigma_ex_1e2_lbl = tk.Label(self.sigma_ex_frame, text="(1/e²: --)", bg=BLUE, anchor="w", fg="#456")
        self.sigma_ex_1e2_lbl.pack(side="left", padx=(5, 0))

        self.sigma_ey_frame = tk.Frame(p, bg=BLUE)
        self.sigma_ey_frame.grid(row=2, column=10, columnspan=2, sticky="w", padx=(15, 5), pady=3)
        self.sigma_ey_lbl = tk.Label(self.sigma_ey_frame, text="sigma_y = --", bg=BLUE, anchor="w")
        self.sigma_ey_lbl.pack(side="left")
        self.sigma_ey_fwhm_lbl = tk.Label(self.sigma_ey_frame, text="(FWHM: --)", bg=BLUE, anchor="w", fg="#456")
        self.sigma_ey_fwhm_lbl.pack(side="left", padx=(5, 0))
        self.sigma_ey_1e2_lbl = tk.Label(self.sigma_ey_frame, text="(1/e²: --)", bg=BLUE, anchor="w", fg="#456")
        self.sigma_ey_1e2_lbl.pack(side="left", padx=(5, 0))

        # MC sample count & seed (row 4, below the field grid -- specs above
        # now spans rows 0-3 since Issue-3's 5 chirp/dispersion/Twiss fields
        # push it past its old 2-row/8-field size)
        for col, (label, default, key) in enumerate([
                ("# electrons (MC)", 200000, "n_mc"),
                ("Random seed", 1, "seed")]):
            tk.Label(p, text=label, bg=BLUE, anchor="w").grid(
                row=4, column=col * 2, sticky="w", padx=(8, 3), pady=3)
            var = tk.StringVar(value=str(default))
            ent = tk.Entry(p, textvariable=var, width=11, justify="right")
            ent.grid(row=4, column=col * 2 + 1, padx=(0, 10), pady=3)
            self.fields[key] = var
            self._sample_entries.append((ent, key))

        # hint that flips between "input" and "loaded from <file>" in display mode
        self.electron_mode_lbl = tk.Label(
            p, text="(input mode - edit the fields above to define the bunch)",
            bg=BLUE, anchor="w", fg="#234",
            font=("TkDefaultFont", 9, "italic"))
        self.electron_mode_lbl.grid(row=5, column=0, columnspan=12, sticky="w",
                                    padx=10, pady=(2, 2))

    # ---- Laser panel (red) --------------------------------------------
    def _build_laser_panel(self):
        p = tk.LabelFrame(self._left_frame, text="LASER", bg=RED, fg="#123",
                          font=("TkDefaultFont", 14, "bold"))
        p.pack(side="top", fill="x", padx=6, pady=3)

        # Primary laser parameters
        specs = [
            ("Laser wavelength [nm]", 1000, "laser_wavelength_nm"),
            ("Laser energy [mJ]", 300, "laser_energy_mJ"),
            ("Pulse duration [ps]", 3, "pulse_duration_ps"),
            ("Pulse frequency [Hz]", 100, "pulse_frequency_Hz"),
            ("Crossing angle [rad]", 0, "crossing_angle"),
        ]
        laser_entries = add_field_grid(p, specs, self.fields, n_cols=5, bg=RED, width=5)
        self._laser_entry_by_key = {key: ent for ent, key in laser_entries}

        # Duration convention selector for pulse duration
        tk.Label(p, text="Pulse convention:", bg=RED, anchor="w").grid(
            row=0, column=10, sticky="w", padx=(15, 3), pady=3)
        self._pulse_convention_var = tk.StringVar(value="RMS")
        pulse_convention_menu = tk.OptionMenu(p, self._pulse_convention_var,
                                               "RMS", "FWHM")
        pulse_convention_menu.config(bg=RED, width=6)
        pulse_convention_menu.grid(row=0, column=11, padx=(0, 10), pady=3)

        # Laser waist input section (row 1)
        waist_frame = tk.LabelFrame(p, text="Laser Waist", bg=RED, fg="#123",
                                     font=("TkDefaultFont", 10, "bold"))
        waist_frame.grid(row=1, column=0, columnspan=12, sticky="we", padx=4, pady=(6, 2))

        # Waist diameter input with convention selector
        tk.Label(waist_frame, text="Waist diameter [um]:", bg=RED, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 3), pady=3)
        self._waist_diameter_var = tk.StringVar(value="10")
        waist_entry = tk.Entry(waist_frame, textvariable=self._waist_diameter_var,
                               width=11, justify="right")
        waist_entry.grid(row=0, column=1, padx=(0, 6), pady=3)
        self.fields["waist_diameter_um"] = self._waist_diameter_var

        tk.Label(waist_frame, text="Convention:", bg=RED, anchor="w").grid(
            row=0, column=2, sticky="w", padx=(8, 3), pady=3)
        self._waist_convention_var = tk.StringVar(value="FWHM")
        waist_convention_menu = tk.OptionMenu(waist_frame, self._waist_convention_var,
                                               "RMS", "FWHM", "1/e²")
        waist_convention_menu.config(bg=RED, width=6)
        waist_convention_menu.grid(row=0, column=3, padx=(0, 6), pady=3)

        # Alternative: Rayleigh length input
        tk.Label(waist_frame, text="OR Rayleigh length [m]:", bg=RED, anchor="w").grid(
            row=0, column=4, sticky="w", padx=(15, 3), pady=3)
        self._rayleigh_var = tk.StringVar(value="0.00126")
        rayleigh_entry = tk.Entry(waist_frame, textvariable=self._rayleigh_var,
                                   width=11, justify="right")
        rayleigh_entry.grid(row=0, column=5, padx=(0, 6), pady=3)
        self.fields["rayleigh_length_m"] = self._rayleigh_var

        # Radio buttons to select waist input mode
        self._waist_input_mode = tk.StringVar(value="diameter")
        tk.Radiobutton(waist_frame, text="Use diameter", variable=self._waist_input_mode,
                       value="diameter", bg=RED, anchor="w").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=(8, 0), pady=2)
        tk.Radiobutton(waist_frame, text="Use Rayleigh length", variable=self._waist_input_mode,
                       value="rayleigh", bg=RED, anchor="w").grid(
            row=1, column=2, columnspan=2, sticky="w", padx=(8, 0), pady=2)

        # Elliptical waist: default is a round beam (Y waist = X waist,
        # unchanged behavior); checking this reveals an independent Y
        # diameter entry, sharing the same convention selector/input-mode
        # radio buttons above (no separate Y convention UI).
        self._elliptical_waist_var = tk.BooleanVar(value=False)
        elliptical_check = tk.Checkbutton(
            waist_frame, text="Elliptical waist", variable=self._elliptical_waist_var,
            bg=RED, anchor="w", command=self._on_elliptical_waist_toggle)
        elliptical_check.grid(row=2, column=0, columnspan=2, sticky="w", padx=(8, 0), pady=2)

        self._waist_diameter_y_lbl = tk.Label(
            waist_frame, text="Waist diameter Y [um]:", bg=RED, anchor="w")
        self._waist_diameter_y_lbl.grid(row=2, column=2, sticky="w", padx=(8, 3), pady=3)
        self._waist_diameter_y_var = tk.StringVar(value="10")
        self._waist_diameter_y_entry = tk.Entry(
            waist_frame, textvariable=self._waist_diameter_y_var, width=11, justify="right")
        self._waist_diameter_y_entry.grid(row=2, column=3, padx=(0, 6), pady=3)
        self.fields["waist_diameter_y_um"] = self._waist_diameter_y_var
        # Hidden until "Elliptical waist" is checked -- default round-beam
        # behavior is unaffected.
        self._waist_diameter_y_lbl.grid_remove()
        self._waist_diameter_y_entry.grid_remove()

        # Mismatch fields (row 2)
        mismatch_specs = [
            ("t-mismatch [ps]", 0, "time_mismatch_ps"),
            ("X-mismatch [mm]", 0, "x_mismatch_mm"),
            ("Y-mismatch [mm]", 0, "y_mismatch_mm"),
            ("Z-mismatch [mm]", 0, "z_mismatch_mm"),
        ]
        mismatch_frame = tk.Frame(p, bg=RED)
        mismatch_frame.grid(row=2, column=0, columnspan=12, sticky="we", padx=4, pady=2)
        for i, (label, default, key) in enumerate(mismatch_specs):
            tk.Label(mismatch_frame, text=label, bg=RED, anchor="w").grid(
                row=0, column=i*2, sticky="w", padx=(8, 3), pady=2)
            var = tk.StringVar(value=str(default))
            ent = tk.Entry(mismatch_frame, textvariable=var, width=8, justify="right")
            ent.grid(row=0, column=i*2+1, padx=(0, 6), pady=2)
            self.fields[key] = var

        # a0 and laser radius display (row 3)
        self.a0_lbl = tk.Label(p, text="a_0 : --", bg=RED, anchor="w",
                               font=("TkDefaultFont", 10, "bold"))
        self.a0_lbl.grid(row=3, column=0, columnspan=2, sticky="w", padx=8,
                         pady=(4, 2))

        self.laser_radius_lbls = []
        for column, text in ((2, "RMS radius = --"),
                             (4, "FWHM radius = --"),
                             (6, "e^{-1/2} radius = --")):
            label = tk.Label(p, text=text, bg=RED, anchor="w")
            label.grid(row=3, column=column, columnspan=2, sticky="w",
                       padx=8, pady=(4, 2))
            self.laser_radius_lbls.append(label)

        # Advanced laser optics (row 4)
        advanced_frame = tk.Frame(p, bg=RED)
        advanced_frame.grid(row=4, column=0, columnspan=12, sticky="we", padx=4, pady=(4, 2))
        adv_specs = [
            ("Flying-focus beta_ff (0=static, 1=co-moving)", 0.0, "beta_ff"),
            ("Polarization angle [rad]", 0.0, "phi_pol"),
            ("Polarization ellipticity", 0.0, "ellipticity"),
        ]
        for i, (label, default, key) in enumerate(adv_specs):
            tk.Label(advanced_frame, text=label, bg=RED, anchor="w").grid(
                row=0, column=i*2, sticky="w", padx=(8, 3), pady=2)
            var = tk.StringVar(value=str(default))
            ent = tk.Entry(advanced_frame, textvariable=var, width=8, justify="right")
            ent.grid(row=0, column=i*2+1, padx=(0, 6), pady=2)
            self.fields[key] = var

    def _on_elliptical_waist_toggle(self):
        """Show/hide the independent Y-waist-diameter entry -- unchecked
        (default) keeps the round-beam behavior of a single diameter field."""
        if self._elliptical_waist_var.get():
            self._waist_diameter_y_lbl.grid()
            self._waist_diameter_y_entry.grid()
        else:
            self._waist_diameter_y_lbl.grid_remove()
            self._waist_diameter_y_entry.grid_remove()

    # ---- Compton photons panel (yellow) --------------------------------
    def _build_compton_panel(self):
        p = tk.LabelFrame(self._left_frame, text="COMPTON PHOTONS", bg=YELLOW, fg="#432",
                          font=("TkDefaultFont", 14, "bold"))
        p.pack(side="top", fill="x", padx=6, pady=3)

        # inputs row (collimation angles + calculate button)
        inp = tk.Frame(p, bg=YELLOW)
        inp.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(2, 6))
        specs = [
            ("theta_x,col [mrad]", 0.05, "theta_x_col_mrad"),
            ("theta_y,col [mrad]", 0.05, "theta_y_col_mrad"),
        ]
        self._compton_entries: dict[str, tk.Entry] = {}
        for i, (label, default, key) in enumerate(specs):
            tk.Label(inp, text=label, bg=YELLOW).grid(row=0, column=2 * i, sticky="w", padx=(8, 3))
            var = tk.StringVar(value=str(default))
            ent = tk.Entry(inp, textvariable=var, width=11, justify="right")
            ent.grid(row=0, column=2 * i + 1, padx=(0, 6))
            self.fields[key] = var
            self._compton_entries[key] = ent
        self.calc_btn = tk.Button(inp, text="Calculate", command=self.on_start)
        self.calc_btn.grid(row=0, column=4, padx=12)
        self.status_lbl = tk.Label(inp, text="idle", bg=YELLOW, width=10)
        self.status_lbl.grid(row=0, column=5, padx=4)

        # outputs: fluxes / statistics (the redundant duck-typed spread-estimate
        # box that used to live here was removed -- the preview panel's real
        # `analytical` model run already covers this, see _build_preview_panel)
        bottom_frame = tk.Frame(p, bg=YELLOW)
        bottom_frame.grid(row=3, column=0, columnspan=3, sticky="we", padx=6, pady=4)
        bottom_frame.columnconfigure(0, weight=0, minsize=120)
        bottom_frame.columnconfigure(1, weight=0, minsize=120)
        bottom_frame.columnconfigure(2, weight=0, minsize=120)

        mono = ("TkFixedFont", 11)

        righta = tk.Frame(bottom_frame, bg=YELLOW, width=120, height=200)
        righta.propagate(False)
        righta.grid(row=0, column=1, sticky="nw", padx=50, pady=4)

        rightb = tk.Frame(bottom_frame, bg=YELLOW, width=120, height=200)
        rightb.propagate(False)
        rightb.grid(row=0, column=2, sticky="nw", padx=5, pady=4)

        self.stat_lbls: dict[str, tk.Label] = {}
        stat_specs = [
            ("total_flux", "Total flux:"),
            ("coll_flux", "Collimated flux:"),
            ("nph_ne", "Nph/Ne:"),
            ("ne0", "Ne0(0ph emit e-):"),
            ("ne1", "Ne1(1ph emit e-):"),
            ("ne2", "Ne2(2ph emit e-):"),
        ]
        for r, (key, name) in enumerate(stat_specs):
            tk.Label(righta, text=name, bg=YELLOW, font=mono, width=17, anchor="w").grid(row=r, column=0, sticky="w")
            lab = tk.Label(righta, text="--", bg=YELLOW, font=mono, width=19, anchor="w")
            lab.grid(row=r, column=1, sticky="w")
            self.stat_lbls[key] = lab

        tk.Label(rightb, text="Recoil parameter:", bg=YELLOW, font=mono, anchor="w").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.stat_lbls["recoil_q"] = tk.Label(rightb, text="--", bg=YELLOW, font=mono, anchor="w")
        self.stat_lbls["recoil_q"].grid(row=0, column=1, sticky="w", padx=5, pady=2)

    # ---- Model parameters panel (grey) -- model-specific numeric fields,
    # rebuilt whenever the active model changes (see _on_model_selected).
    def _build_model_params_panel(self):
        self.model_params_frame = tk.LabelFrame(
            self._left_frame, text="MODEL PARAMETERS", bg=GREY, fg="#123",
            font=("TkDefaultFont", 14, "bold"))
        self.model_params_frame.pack(side="top", fill="x", padx=6, pady=3)
        self._rebuild_model_params_panel()

    def _rebuild_model_params_panel(self):
        """Repopulate the Model Parameters panel from
        ``self.active_adapter.model_params()``. Never re-packs the frame
        itself (only its children) so switching models repeatedly doesn't
        reorder it relative to the plot area below."""
        for child in self.model_params_frame.winfo_children():
            child.destroy()

        specs = self.active_adapter.model_params()
        if not specs:
            tk.Label(self.model_params_frame, bg=GREY, fg="#567",
                     text="(this model has no model-specific parameters)",
                     font=("TkDefaultFont", 9, "italic")).grid(
                row=0, column=0, sticky="w", padx=8, pady=4)
            return

        # Preserve values already entered if the key survives a model
        # switch (e.g. switching away and back); otherwise use the
        # adapter's own default.
        seeded = [(label, self.fields[key].get() if key in self.fields else default, key)
                  for label, default, key in specs]
        choices = self.active_adapter.model_choices()
        add_field_grid(self.model_params_frame, seeded, self.fields,
                       n_cols=4, bg=GREY, width=8, choices=choices)

    # ---- output resolution panel (model-agnostic) -----------------------
    def _build_output_panel(self):
        """Panel for configuring output resolution (spectrum, temporal,
        spatial, angular bins). These values are passed to models via
        OutputSpec."""
        self._output_panel_frame = tk.LabelFrame(
            self._left_frame, text="OUTPUT RESOLUTION", bg=GREY, fg="#123",
            font=("TkDefaultFont", 14, "bold"))
        self._output_panel_frame.pack(side="top", fill="x", padx=6, pady=3)

        # Default values matching OutputSpec
        output_specs = [
            ("Energy bins", 256, "n_energy_bins"),
            ("Time bins", 128, "n_time_bins"),
            ("Spatial bins X", 64, "n_spatial_bins_x"),
            ("Spatial bins Y", 64, "n_spatial_bins_y"),
            ("Angular bins X", 64, "n_angular_bins_x"),
            ("Angular bins Y", 64, "n_angular_bins_y"),
        ]
        add_field_grid(self._output_panel_frame, output_specs, self.fields, n_cols=4, bg=GREY, width=8)

    def _get_output_spec(self):
        """Build an OutputSpec from the GUI fields."""
        from gammaforge.models.api import OutputSpec, SliceRequest

        n_energy_bins = max(1, int(float(self.fields["n_energy_bins"].get())))
        n_time_bins = max(1, int(float(self.fields["n_time_bins"].get())))
        n_spatial_bins_x = max(1, int(float(self.fields["n_spatial_bins_x"].get())))
        n_spatial_bins_y = max(1, int(float(self.fields["n_spatial_bins_y"].get())))
        n_angular_bins_x = max(1, int(float(self.fields["n_angular_bins_x"].get())))
        n_angular_bins_y = max(1, int(float(self.fields["n_angular_bins_y"].get())))
        return OutputSpec(slices=(
            SliceRequest((AXIS_ENERGY,), (n_energy_bins,)),
            SliceRequest((AXIS_TIME,), (n_time_bins,)),
            SliceRequest((AXIS_X, AXIS_Y), (n_spatial_bins_x, n_spatial_bins_y)),
            SliceRequest((AXIS_THETA_X, AXIS_THETA_Y), (n_angular_bins_x, n_angular_bins_y)),
            SliceRequest((AXIS_ENERGY, AXIS_THETA_X, AXIS_THETA_Y), (96, 33, 33)),
        ))

    # ---- analytical preview panel (always-on, independent of the
    # selected model) --------------------------------------------------
    def _build_preview_panel(self):
        """Small always-visible panel showing the fast analytical model's
        estimate, run automatically alongside whichever model is actually
        selected (see on_start/_poll_queue) -- a real-time preview and base
        sanity check"""
        p = tk.LabelFrame(self._left_frame, text="ANALYTICAL PREVIEW (always-on)", bg=GREY, fg="#123",
                          font=("TkDefaultFont", 11, "bold"))
        p.pack(side="top", fill="x", padx=6, pady=3)
        mono = ("TkFixedFont", 10)
        self.preview_lbls: dict[str, tk.Label] = {}
        specs = [("status", "Status:"), ("total_yield", "Total yield:"),
                 ("width", "Est. spectrum width (FWHM, s):")]
        for c, (key, name) in enumerate(specs):
            tk.Label(p, text=name, bg=GREY, font=mono).grid(row=0, column=2 * c, sticky="w", padx=(8, 3), pady=3)
            lab = tk.Label(p, text="--", bg=GREY, font=mono, width=16, anchor="w")
            lab.grid(row=0, column=2 * c + 1, sticky="w", padx=(0, 8))
            self.preview_lbls[key] = lab

    def _render_preview(self):
        """Update the preview panel from self.preview_res (set by
        _poll_queue). Never raises -- a malformed/missing preview result
        just shows as unavailable, it must never break the main model's
        own output rendering."""
        if self.preview_res is None:
            self.preview_lbls["status"].config(text="failed")
            self.preview_lbls["total_yield"].config(text="--")
            self.preview_lbls["width"].config(text="--")
            return
        try:
            self.preview_lbls["status"].config(text="ok")
            self.preview_lbls["total_yield"].config(text=f"{total_yield(self.preview_res):.4e}")
            width = self.preview_res.model_specific.get("estimated_spectrum_width_fwhm")
            self.preview_lbls["width"].config(text=f"{width:.4g}" if width is not None else "--")
        except Exception:
            traceback.print_exc()
            self.preview_lbls["status"].config(text="render error")

    # ---- plots (tabbed notebook) ----------------------------------------
    def _build_plot_area(self):
        self.notebook = ttk.Notebook(self._plot_frame)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        # Tab 1: Spectrum & Electron (the original two-figure view)
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Spectrum & Electron")
        self.fig = plt.Figure(figsize=(11, 3.6))
        self.ax_spec, self.ax_e = self.fig.subplots(1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab1)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Tab 2: Temporal Envelope
        self.tab_temporal = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_temporal, text="Temporal Envelope")
        self.fig_t = plt.Figure(figsize=(11, 3.6))
        self.ax_t = self.fig_t.add_subplot(111)
        self.canvas_t = FigureCanvasTkAgg(self.fig_t, master=self.tab_temporal)
        self.canvas_t.get_tk_widget().pack(fill="both", expand=True)

        # Tab 3: Spatial Distribution
        self.tab_spatial = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_spatial, text="Spatial Distribution")
        self.fig_s = plt.Figure(figsize=(11, 3.6))
        self.ax_s = self.fig_s.add_subplot(111)
        self.canvas_s = FigureCanvasTkAgg(self.fig_s, master=self.tab_spatial)
        self.canvas_s.get_tk_widget().pack(fill="both", expand=True)

        # Tab 4: Angular Distribution
        self.tab_angular = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_angular, text="Angular Distribution")
        self.fig_a = plt.Figure(figsize=(11, 3.6))
        self.ax_a = self.fig_a.add_subplot(111)
        self.canvas_a = FigureCanvasTkAgg(self.fig_a, master=self.tab_angular)
        self.canvas_a.get_tk_widget().pack(fill="both", expand=True)

        # Tab 5: Angle-Resolved Spectra -- four (energy, angle) slices/
        # projections of the same joint (AXIS_ENERGY, AXIS_THETA_X,
        # AXIS_THETA_Y) data _render_angular_distribution's ang3d fallback
        # already reads, see _render_angle_resolved_spectra.
        self.tab_angle_resolved = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_angle_resolved, text="Angle-Resolved Spectra")
        self.fig_ar = plt.Figure(figsize=(11, 7.2))
        ((self.ax_ar_tx0, self.ax_ar_ty0),
         (self.ax_ar_tx_sum, self.ax_ar_ty_sum)) = self.fig_ar.subplots(2, 2)
        self.canvas_ar = FigureCanvasTkAgg(self.fig_ar, master=self.tab_angle_resolved)
        self.canvas_ar.get_tk_widget().pack(fill="both", expand=True)

        self._render_placeholder()

    def _render_placeholder(self):
        for ax, t in ((self.ax_spec, "Collimated photon spectrum"),
                      (self.ax_e, "Electron energy: initial vs. final")):
            ax.clear()
            ax.text(0.5, 0.5, "Press Calculate to run", ha="center", va="center")
            ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw()
        for ax, canvas, fig, title in (
                (self.ax_t, self.canvas_t, self.fig_t, "Temporal envelope"),
                (self.ax_s, self.canvas_s, self.fig_s, "Spatial distribution"),
                (self.ax_a, self.canvas_a, self.fig_a, "Angular distribution")):
            ax.clear()
            ax.text(0.5, 0.5, "Press Calculate to run", ha="center", va="center")
            ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            canvas.draw()

        for ax, title in ((self.ax_ar_tx0, r"$s$-$\theta_x$ at $\theta_y=0$"),
                           (self.ax_ar_ty0, r"$s$-$\theta_y$ at $\theta_x=0$"),
                           (self.ax_ar_tx_sum, r"$s$-$\theta_x$, summed over $\theta_y$"),
                           (self.ax_ar_ty_sum, r"$s$-$\theta_y$, summed over $\theta_x$")):
            ax.clear()
            ax.text(0.5, 0.5, "Press Calculate to run", ha="center", va="center")
            ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        self.fig_ar.tight_layout()
        self.canvas_ar.draw()

    # ---- live (no-run) updates -----------------------------------------
    def _wire_live_updates(self):
        for k in ("emit_x_mmmrad", "emit_y_mmmrad", "beta_x_m", "beta_y_m",
                 "mean_energy_MeV"):
            tname = self.fields[k].trace_add(
                "write", lambda *a: self._update_derived())
            self._electron_traces.append((self.fields[k], tname))
        for k in ("laser_wavelength_nm", "laser_energy_mJ", "pulse_duration_ps",
                  "rayleigh_length_m", "waist_diameter_um"):
            self.fields[k].trace_add("write", lambda *a: self._update_derived())
        # Trace convention selectors
        self._duration_convention_var.trace_add("write", lambda *a: self._update_derived())
        self._pulse_convention_var.trace_add("write", lambda *a: self._update_derived())
        self._waist_convention_var.trace_add("write", lambda *a: self._update_derived())
        self._waist_input_mode.trace_add("write", lambda *a: self._update_derived())
        for k in ("theta_x_col_mrad", "theta_y_col_mrad"):
            self.fields[k].trace_add("write", lambda *a: self._update_outputs())
        for k in ("theta_x_col_mrad", "theta_y_col_mrad"):
            self.fields[k].trace_add("write", lambda *a: self._update_outputs())

    def _update_derived(self):
        """Refresh electron and laser values derived from the current fields."""
        energy_mev = _float_or_none(self.fields["mean_energy_MeV"])
        gamma = _native(energy_mev, "megaelectron_volt", "electron_volt") / MEC2_EV if energy_mev is not None else None
        emit_x_mmmrad = _float_or_none(self.fields["emit_x_mmmrad"])
        emit_y_mmmrad = _float_or_none(self.fields["emit_y_mmmrad"])
        emit_x_m = _native(emit_x_mmmrad, "millimeter * milliradian", "meter") if emit_x_mmmrad is not None else None
        emit_y_m = _native(emit_y_mmmrad, "millimeter * milliradian", "meter") if emit_y_mmmrad is not None else None
        sx = sigma_from_emittance(emit_x_m, _float_or_none(self.fields["beta_x_m"]), gamma)
        sy = sigma_from_emittance(emit_y_m, _float_or_none(self.fields["beta_y_m"]), gamma)

        # Show beam size in multiple conventions (RMS, FWHM, 1/e²)
        if sx:
            sx_fwhm = _native(sigma_intensity_to_fwhm(sx), "meter", "micrometer")
            sx_1e2 = _native(sigma_intensity_to_w0(sx), "meter", "micrometer")
            self.sigma_ex_lbl.config(text=f"sigma_x = {_native(sx, 'meter', 'micrometer'):.3f} um")
            self.sigma_ex_fwhm_lbl.config(text=f"(FWHM: {sx_fwhm:.3f} um)")
            self.sigma_ex_1e2_lbl.config(text=f"(1/e²: {sx_1e2:.3f} um)")
        else:
            self.sigma_ex_lbl.config(text="sigma_x = --")
            self.sigma_ex_fwhm_lbl.config(text="(FWHM: --)")
            self.sigma_ex_1e2_lbl.config(text="(1/e²: --)")

        if sy:
            sy_fwhm = _native(sigma_intensity_to_fwhm(sy), "meter", "micrometer")
            sy_1e2 = _native(sigma_intensity_to_w0(sy), "meter", "micrometer")
            self.sigma_ey_lbl.config(text=f"sigma_y = {_native(sy, 'meter', 'micrometer'):.3f} um")
            self.sigma_ey_fwhm_lbl.config(text=f"(FWHM: {sy_fwhm:.3f} um)")
            self.sigma_ey_1e2_lbl.config(text=f"(1/e²: {sy_1e2:.3f} um)")
        else:
            self.sigma_ey_lbl.config(text="sigma_y = --")
            self.sigma_ey_fwhm_lbl.config(text="(FWHM: --)")
            self.sigma_ey_1e2_lbl.config(text="(1/e²: --)")

        # Handle waist diameter / Rayleigh length conversion
        wavelength_nm = _float_or_none(self.fields["laser_wavelength_nm"])
        wavelength_m = _native(wavelength_nm, "nanometer", "meter") if wavelength_nm is not None else None

        # Diameter mode: keep the Rayleigh-length field's live display in sync.
        if self._waist_input_mode.get() == "diameter" and wavelength_m is not None:
            laser_si = self._laser_fields_si()
            if laser_si is not None:
                z_R = 4.0 * np.pi * laser_si["sigma0_l_m"] ** 2 / wavelength_m
                self._rayleigh_var.set(f"{z_R:.6g}")

        laser_si = self._laser_fields_si()
        if laser_si is not None:
            pulse = GaussianParaxialLaser(
                pulse_energy_J=_pq(laser_si["pulse_energy_J"], "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
                wavelength_m=_pq(laser_si["wavelength_m"], "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
                waist_rms_x_m=_pq(laser_si["sigma0_l_m"], "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
                waist_rms_y_m=_pq(laser_si["sigma0_l_m_y"], "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
                duration_rms_s=_pq(laser_si["sigma_par_L_m"] / C_LIGHT, "second", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
                focus_z_m=_pq(laser_si["focus_z_m"], "meter", PhysicalMeaning.DISPLACEMENT, NoConvention.PLAIN),
            )
            self.a0_lbl.config(text=f"a_0 : {pulse.a0_interaction:.4g}")
            radii = {
                "rms": laser_si["sigma0_l_m"],
                "fwhm": sigma_intensity_to_fwhm(laser_si["sigma0_l_m"]),
                "w0_1e2": sigma_intensity_to_w0(laser_si["sigma0_l_m"]),
            }
        else:
            self.a0_lbl.config(text="a_0 : --")
            radii = None

        for label, name, key in zip(
                self.laser_radius_lbls, ("RMS", "FWHM", "e^{-1/2}"), ("rms", "fwhm", "w0_1e2")):
            text = (f"{name} radius = {_native(radii[key], 'meter', 'micrometer'):.3f} um"
                    if radii is not None else f"{name} radius = --")
            label.config(text=text)

    def _laser_fields_si(self):
        """Compute (wavelength_m, sigma0_l_m, sigma_par_L_m, pulse_energy_J,
        focus_z_m) from the current Laser-panel fields, honouring the waist
        input mode (diameter vs. Rayleigh length) and pulse-duration
        convention (RMS vs. FWHM). Returns ``None`` if a required field is
        missing/unparseable."""
        wavelength_nm = _float_or_none(self.fields["laser_wavelength_nm"])
        energy_mJ = _float_or_none(self.fields["laser_energy_mJ"])
        duration_ps = _float_or_none(self.fields["pulse_duration_ps"])
        if None in (wavelength_nm, energy_mJ, duration_ps):
            return None
        wavelength_m = _native(wavelength_nm, "nanometer", "meter")

        duration_s = _native(duration_ps, "picosecond", "second")
        if self._pulse_convention_var.get() == "FWHM":
            duration_s = fwhm_to_sigma_intensity(duration_s)

        if self._waist_input_mode.get() == "diameter":
            waist_diameter_um = _float_or_none(self.fields["waist_diameter_um"])
            if waist_diameter_um is None:
                return None
            waist_radius_m = _native(waist_diameter_um, "micrometer", "meter") / 2.0
            waist_convention = self._waist_convention_var.get()
            if waist_convention == "RMS":
                sigma0_l_m = waist_radius_m
            elif waist_convention == "FWHM":
                sigma0_l_m = fwhm_to_sigma_intensity(waist_radius_m)
            else:  # 1/e^2
                sigma0_l_m = w0_to_sigma_intensity(waist_radius_m)
        else:
            rayleigh_m = _float_or_none(self.fields["rayleigh_length_m"])
            if rayleigh_m is None:
                return None
            sigma0_l_m = (rayleigh_m * wavelength_m / (4.0 * np.pi)) ** 0.5

        # Elliptical waist: independent Y diameter, same convention as X.
        # Only meaningful in "diameter" input mode -- a single Rayleigh
        # length can't imply two different waists, so an elliptical Y waist
        # while in "rayleigh" mode just falls back to the round-beam value.
        sigma0_l_m_y = sigma0_l_m
        if self._elliptical_waist_var.get() and self._waist_input_mode.get() == "diameter":
            waist_diameter_y_um = _float_or_none(self.fields["waist_diameter_y_um"])
            if waist_diameter_y_um is None:
                return None
            waist_radius_y_m = _native(waist_diameter_y_um, "micrometer", "meter") / 2.0
            waist_convention = self._waist_convention_var.get()
            if waist_convention == "RMS":
                sigma0_l_m_y = waist_radius_y_m
            elif waist_convention == "FWHM":
                sigma0_l_m_y = fwhm_to_sigma_intensity(waist_radius_y_m)
            else:  # 1/e^2
                sigma0_l_m_y = w0_to_sigma_intensity(waist_radius_y_m)

        z_mismatch_mm = _float_or_none(self.fields["z_mismatch_mm"])
        return dict(
            wavelength_m=wavelength_m,
            sigma0_l_m=sigma0_l_m,
            sigma0_l_m_y=sigma0_l_m_y,
            sigma_par_L_m=duration_s * C_LIGHT,
            pulse_energy_J=_native(energy_mJ, "millijoule", "joule"),
            focus_z_m=_native(z_mismatch_mm or 0.0, "millimeter", "meter"),
        )

    def _build_beam_and_laser(self) -> tuple[GaussianElectronBeam, GaussianParaxialLaser]:
        """Compile the current Electrons + Laser panel fields into the
        analytic beam/laser descriptions. Building the shared
        ``InteractionParameters`` needs a *sampled* ``Bunch`` too (it holds
        ``electrons``, not just ``beam``), so that happens one level up in
        ``on_start()`` once electrons are available. Raises ``ValueError``
        if a required field is missing."""
        energy_mev = _float_or_none(self.fields["mean_energy_MeV"])
        rel_spread_pct = _float_or_none(self.fields["rel_spread_pct"])
        charge_nC = _float_or_none(self.fields["charge_nC"])
        duration_ps = _float_or_none(self.fields["bunch_duration_ps"])
        emit_x_mmmrad = _float_or_none(self.fields["emit_x_mmmrad"])
        emit_y_mmmrad = _float_or_none(self.fields["emit_y_mmmrad"])
        beta_x_m = _float_or_none(self.fields["beta_x_m"])
        beta_y_m = _float_or_none(self.fields["beta_y_m"])
        if None in (energy_mev, rel_spread_pct, charge_nC, duration_ps,
                    emit_x_mmmrad, emit_y_mmmrad, beta_x_m, beta_y_m):
            raise ValueError("All Electrons-panel fields must be filled in with numbers")
        # Optional correlations -- default to 0.0 (no chirp/dispersion/tilt)
        # like beta_ff/phi_pol/ellipticity on the Laser panel, not required.
        chirp_h = _float_or_none(self.fields["chirp_h"]) or 0.0
        dispersion_x = _float_or_none(self.fields["dispersion_x"]) or 0.0
        dispersion_y = _float_or_none(self.fields["dispersion_y"]) or 0.0
        alpha_x = _float_or_none(self.fields["alpha_x"]) or 0.0
        alpha_y = _float_or_none(self.fields["alpha_y"]) or 0.0

        energy_eV = _native(energy_mev, "megaelectron_volt", "electron_volt")
        gamma = energy_eV / MEC2_EV
        kinetic_energy_eV = energy_eV - MEC2_EV
        emit_x_m = _native(emit_x_mmmrad, "millimeter * milliradian", "meter")
        emit_y_m = _native(emit_y_mmmrad, "millimeter * milliradian", "meter")
        sigma_x_m = sigma_from_emittance(emit_x_m, beta_x_m, gamma)
        sigma_y_m = sigma_from_emittance(emit_y_m, beta_y_m, gamma)

        bunch_duration_s = _native(duration_ps, "picosecond", "second")
        if self._duration_convention_var.get() == "FWHM":
            bunch_duration_s = fwhm_to_sigma_intensity(bunch_duration_s)

        beam = GaussianElectronBeam(
            bunch_charge_C=_pq(_native(charge_nC, "nanocoulomb", "coulomb"), "coulomb",
                               PhysicalMeaning.BUNCH_CHARGE, NoConvention.PLAIN),
            kinetic_energy_eV=_pq(kinetic_energy_eV, "electron_volt", PhysicalMeaning.BEAM_ENERGY, NoConvention.PLAIN),
            rel_energy_spread_rms=rel_spread_pct / 100.0,
            sigma_x_m=_pq(sigma_x_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
            sigma_y_m=_pq(sigma_y_m, "meter", PhysicalMeaning.ELECTRON_BEAM_SIZE, WidthConvention.SIGMA_INTENSITY_RMS),
            emit_geom_x_m=_pq(emit_x_m / gamma, "meter", PhysicalMeaning.EMITTANCE, NoConvention.PLAIN),
            emit_geom_y_m=_pq(emit_y_m / gamma, "meter", PhysicalMeaning.EMITTANCE, NoConvention.PLAIN),
            sigma_t_s=_pq(bunch_duration_s, "second", PhysicalMeaning.BUNCH_LENGTH, TimeConvention.SIGMA_INTENSITY_RMS),
            sigma_pz=rel_spread_pct / 100.0,
            chirp_h=chirp_h, dispersion_x=dispersion_x, dispersion_y=dispersion_y,
            alpha_x=alpha_x, alpha_y=alpha_y,
        )

        laser_si = self._laser_fields_si()
        if laser_si is None:
            raise ValueError("All Laser-panel fields must be filled in with numbers")
        beta_ff = _float_or_none(self.fields["beta_ff"]) or 0.0
        phi_pol = _float_or_none(self.fields["phi_pol"]) or 0.0
        ellipticity = _float_or_none(self.fields["ellipticity"]) or 0.0
        crossing_angle = _float_or_none(self.fields["crossing_angle"]) or 0.0
        laser = GaussianParaxialLaser(
            pulse_energy_J=_pq(laser_si["pulse_energy_J"], "joule", PhysicalMeaning.PULSE_ENERGY, NoConvention.PLAIN),
            wavelength_m=_pq(laser_si["wavelength_m"], "meter", PhysicalMeaning.WAVELENGTH, NoConvention.PLAIN),
            waist_rms_x_m=_pq(laser_si["sigma0_l_m"], "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
            waist_rms_y_m=_pq(laser_si["sigma0_l_m_y"], "meter", PhysicalMeaning.LASER_WIDTH, WidthConvention.SIGMA_INTENSITY_RMS),
            duration_rms_s=_pq(laser_si["sigma_par_L_m"] / C_LIGHT, "second", PhysicalMeaning.PULSE_DURATION, TimeConvention.SIGMA_INTENSITY_RMS),
            focus_z_m=_pq(laser_si["focus_z_m"], "meter", PhysicalMeaning.DISPLACEMENT, NoConvention.PLAIN),
            beta_ff=beta_ff, phi_pol=phi_pol, ellipticity=ellipticity,
            crossing_angle=crossing_angle,
        )
        return beam, laser

    def _model_extra_fields(self, adapter: ModelAdapter) -> dict:
        """Build the ``Job.extra`` dict for ``adapter`` from its own
        ``model_params()`` keys, reading each from ``self.fields`` if the
        GUI has a live entry for it (i.e. the Model Parameters panel is
        currently showing this adapter's fields), else falling back to the
        adapter's own declared default. ``beta_ff``/``phi_pol``/
        ``ellipticity``/``crossing_angle`` are NOT here -- they're
        laser-pulse properties that flow into ``GaussianParaxialLaser``
        directly via ``_build_beam_and_laser()``, not model-specific
        extras."""
        extra = {}
        for _label, default, key in adapter.model_params():
            extra[key] = self.fields[key].get() if key in self.fields else default
        return extra

    # ---- run / threading ------------------------------------------------
    def on_start(self):
        if self.worker is not None and self.worker.is_alive():
            return

        try:
            beam, laser = self._build_beam_and_laser()
        except Exception as e:
            messagebox.showerror("Invalid parameter", str(e))
            return

        if self.loaded_bunch is not None:
            # Invariant: every Bunch reaching a Job must have gaussian_fit
            # populated. _load_ele already fits and attaches gaussian_fit
            # to self.loaded_bunch before it's ever stored here.
            electrons = self.loaded_bunch
        else:
            try:
                n_mc = int(float(self.fields["n_mc"].get()))
                seed = int(float(self.fields["seed"].get()))
            except (ValueError, tk.TclError):
                messagebox.showerror("Invalid parameter", "# electrons (MC) and Random seed must be numbers")
                return
            electrons = sample_gaussian_bunch(
                beam, n_particles=n_mc, rng=np.random.default_rng(seed))

        drift_L_m = _float_or_none(self.fields["drift_L_m"]) or 0.0
        if drift_L_m != 0.0:
            electrons = drift(electrons, drift_L_m)

        interaction = InteractionParameters(laser=laser, electrons=electrons)

        try:
            seed = int(float(self.fields["seed"].get()))
        except (ValueError, tk.TclError):
            seed = 0
        rep_rate_hz = _float_or_none(self.fields["pulse_frequency_Hz"])
        self.rep_rate_hz = rep_rate_hz if rep_rate_hz else 1.0
        output_spec = self._get_output_spec()

        job = Job(interaction=interaction, output=output_spec,
                  seed=seed, extra=self._model_extra_fields(self.active_adapter))
        preview_job = Job(interaction=interaction, output=output_spec,
                          seed=seed, extra=self._model_extra_fields(self.analytical_adapter))

        self.interaction_used = interaction
        self.calc_btn.config(state="disabled")
        self.status_lbl.config(text="running...")

        adapter = self.active_adapter
        analytical_adapter = self.analytical_adapter

        def work():
            try:
                res = adapter.run(job)
                problems = validate_results(res)
                if problems:
                    raise RuntimeError(f"adapter returned malformed results: {'; '.join(problems)}")
            except Exception as e:
                self.q.put(("error", "".join(traceback.format_exception(e))))
                return

            try:
                preview_res = analytical_adapter.run(preview_job)
            except Exception:
                traceback.print_exc()
                preview_res = None

            self.q.put(("done", (res, preview_res)))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self.after(100, self._poll_queue)

    def _poll_queue(self):
        try:
            status, payload = self.q.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_queue)
            return
        self.calc_btn.config(state="normal")

        if status == "error":
            self.status_lbl.config(text="error")
            messagebox.showerror("Simulation failed", payload)
            return

        self.res, self.preview_res = payload
        self.status_lbl.config(text="done")
        self._update_outputs()
        self._render_preview()

    # ---- collimation-dependent outputs (no re-run) ---------------------
    def _collimation_rad(self):
        tx = _float_or_none(self.fields["theta_x_col_mrad"])
        ty = _float_or_none(self.fields["theta_y_col_mrad"])
        tx = _native(tx, "milliradian", "radian") if tx is not None else None
        ty = _native(ty, "milliradian", "radian") if ty is not None else None
        return tx, ty

    def _update_outputs(self):
        if self.res is None or self.interaction_used is None:
            return
        res, interaction = self.res, self.interaction_used
        tx, ty = self._collimation_rad()

        # --- fluxes and statistics (branch on result shape) ---
        total_flux, coll_flux, cmask = self._photon_fluxes(res, tx, ty)
        self.stat_lbls["total_flux"].config(text=f"{total_flux:.3e} ph/s")
        self.stat_lbls["coll_flux"].config(text=f"{coll_flux:.3e} ph/s")

        pm = res.model_specific.get("photon_multiplicity")
        if pm is not None:
            self.stat_lbls["nph_ne"].config(text=f"{pm['mean_n_phot']:.4e}")
            self.stat_lbls["ne0"].config(text=f"{pm['frac_n0']*100:.3f} %")
            self.stat_lbls["ne1"].config(text=f"{pm['frac_n1']*100:.3f} %")
            self.stat_lbls["ne2"].config(text=f"{pm['frac_n2']*100:.3f} %")
        else:
            for key in ("nph_ne", "ne0", "ne1", "ne2"):
                self.stat_lbls[key].config(text="N/A")

        recoil_q = recoil_parameter(
            interaction.electrons.gaussian_fit.gamma0,
            interaction.laser.wavelength_m.to_unit("meter").magnitude)
        self.stat_lbls["recoil_q"].config(text=f"{recoil_q:.6f}")

        self._render_plots(res, cmask, interaction)

    def _photon_fluxes(self, res, tx, ty):
        """Return (total_flux, collimated_flux, cmask_or_None) [ph/s].

        cmask is only meaningful (and only returned) when ``res.macrophoton``
        is present (kascade), where _render_plots uses it to mask the raw
        photon array directly (exact, no grid involved).

        Otherwise, collimated flux comes from an on-demand
        active_adapter.spectrum_in_angular_range() query -- a fresh grid
        sized for the ACTUAL requested window -- rather than re-integrating
        the cached, wide-range energy+angle joint slice (used only by the
        Angular Distribution tab's visualization). That used to double as
        the "collimated flux" source here too, but its grid is sized for a
        4*pi overview, not a tight collimation window: e.g. delta's
        default 9x9 cache captures just 1 grid point inside the GUI's own
        default 0.05 mrad window, and reusing that point's full (much wider)
        cache-grid cell as its effective solid angle overcounts badly --
        confirmed producing a "collimated" spectrum peaking ~260x above the
        true 4*pi spectrum near the Compton edge. spectrum_in_angular_range
        already exists precisely to avoid this (it was fixed earlier this
        session to evaluate the physics kernel fresh at the requested
        window, not mask/reuse a cache) -- this just routes through it here
        too, still no MC re-run (reuses the cached per-particle arrays).
        """
        if res.macrophoton is not None:
            mp = res.macrophoton
            n_tot = mp.x.size
            total_flux = n_tot * mp.weight * self.rep_rate_hz
            if tx is not None and ty is not None and n_tot:
                cmask = (np.abs(mp.thx) <= tx) & (np.abs(mp.thy) <= ty)
            else:
                cmask = np.zeros(n_tot, dtype=bool)
            coll_flux = int(cmask.sum()) * mp.weight * self.rep_rate_hz
            return total_flux, coll_flux, cmask

        total_flux = total_yield(res) * self.rep_rate_hz
        if tx is None or ty is None:
            return total_flux, 0.0, None
        n_energy_bins = max(1, int(float(self.fields["n_energy_bins"].get())))
        rng_result = self.active_adapter.spectrum_in_angular_range(
            (-tx, tx), (-ty, ty), n_energy=n_energy_bins)
        coll_flux = (rng_result.n_photons_in_range or 0.0) * self.rep_rate_hz
        return total_flux, coll_flux, None

    # ---- plotting -------------------------------------------------------
    def _render_plots(self, res, cmask, interaction):
        self._render_spectrum(res, cmask)

        if res.electrons is not None:
            self._render_electron_state(interaction.electrons, res.electrons)
        else:
            self.ax_e.clear()
            self.ax_e.text(0.5, 0.5,
                            "N/A for this model\n(no electron final-state tracking)",
                            ha="center", va="center")
            self.ax_e.set_title("Electron energy: initial vs. final")
            self.ax_e.set_xticks([]); self.ax_e.set_yticks([])

        self.fig.tight_layout()
        self.canvas.draw()

        self._render_temporal_envelope(res)
        self._render_spatial_distribution(res)
        self._render_angular_distribution(res)
        self._render_angle_resolved_spectra(res)

    def _render_spectrum(self, res, cmask):
        self.ax_spec.clear()
        if res.macrophoton is not None:
            mp = res.macrophoton
            E_keV = mp.gamma / 1e3  # kascade stores photon E_eV directly in Bunch.gamma
            Ec = E_keV[cmask] if cmask is not None and cmask.size else E_keV[:0]
            if E_keV.size:
                emax = E_keV.max() * 1.02
                self.ax_spec.hist(E_keV, bins=120, range=(0, emax),
                                  weights=np.full(E_keV.size, mp.weight),
                                  color="0.8", label="all (4*pi)")
            if Ec.size:
                self.ax_spec.hist(Ec, bins=120, range=(0, E_keV.max() * 1.02),
                                  weights=np.full(Ec.size, mp.weight),
                                  histtype="step", color="crimson", lw=1.5,
                                  label="collimated")
            self.ax_spec.set_ylabel("photons / bin")
            self.ax_spec.set_title("Collimated photon spectrum")
        else:
            spec = find_slice(res.photon_slices, AXIS_ENERGY)
            if spec is not None:
                E_keV = spec.axes[AXIS_ENERGY] / 1e3
                dNdE_per_keV = spec.distr * 1e3
                self.ax_spec.plot(E_keV, dNdE_per_keV, color="0.4", label="all (4*pi)")

            tx, ty = self._collimation_rad()
            if tx is not None and ty is not None:
                # On-demand query at a grid sized for THIS window, not the
                # wide-range energy+angle joint slice -- see _photon_fluxes'
                # docstring for why re-integrating that cache here used to
                # produce a "collimated" curve that could spike far above the
                # true 4*pi spectrum near the Compton edge. n_energy is
                # explicit here (not the adapter's own default of 96/65) so
                # the "Energy bins" field actually controls this curve's
                # resolution too, not just the "all (4*pi)" one above.
                n_energy_bins = max(1, int(float(self.fields["n_energy_bins"].get())))
                rng_result = self.active_adapter.spectrum_in_angular_range(
                    (-tx, tx), (-ty, ty), n_energy=n_energy_bins)
                coll_spec = rng_result.spectrum
                E_coll = coll_spec.axes.get(AXIS_ENERGY)
                if E_coll is not None and E_coll.size:
                    self.ax_spec.plot(E_coll / 1e3, coll_spec.distr * 1e3,
                                      color="crimson", label="collimated")
            self.ax_spec.set_ylabel(r"dN/dE [photons / keV]")
            self.ax_spec.set_title("Photon spectrum (semi-analytic, binned)")

        self.ax_spec.set_xlabel(r"photon energy $\hbar\omega_\gamma$ [keV]")
        self.ax_spec.legend(fontsize=8)

    def _render_electron_state(self, initial, final):
        self.ax_e.clear()
        E_i = initial.gamma * MEC2_EV / 1e6
        E_f = final.gamma * MEC2_EV / 1e6
        rng = (min(E_f.min(), E_i.min()), E_i.max())
        self.ax_e.hist(E_i, bins=120, range=rng, weights=np.full(E_i.size, initial.weight),
                       histtype="step", color="gray", lw=1.3, label="initial")
        self.ax_e.hist(E_f, bins=120, range=rng, weights=np.full(E_f.size, final.weight),
                       histtype="step", color="crimson", lw=1.3, label="final (total)")
        self.ax_e.set_xlabel("electron energy [MeV]")
        self.ax_e.set_ylabel("electrons / bin")
        self.ax_e.set_title("Electron energy: initial vs. final")
        self.ax_e.set_yscale("log")
        self.ax_e.legend(fontsize=8)

    def _render_unavailable(self, ax, canvas, fig, title, message):
        ax.clear()
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        canvas.draw()

    def _render_temporal_envelope(self, res):
        te = find_slice(res.photon_slices, AXIS_TIME)
        if te is None:
            self._render_unavailable(
                self.ax_t, self.canvas_t, self.fig_t, "Temporal envelope",
                "N/A for this model\n(no temporal data)")
            return
        self.ax_t.clear()
        self.ax_t.plot(te.axes[AXIS_TIME] * 1e12, te.distr, color="0.4")
        self.ax_t.set_ylabel("photons / s")
        self.ax_t.set_xlabel("emission time [ps]")
        self.ax_t.set_title("Photon temporal envelope")
        self.fig_t.tight_layout()
        self.canvas_t.draw()

    def _render_spatial_distribution(self, res):
        sd = find_slice(res.photon_slices, AXIS_X, AXIS_Y)
        if sd is None:
            self._render_unavailable(
                self.ax_s, self.canvas_s, self.fig_s, "Spatial distribution",
                "N/A for this model\n(no spatial-deposition kernel)")
            return
        self.ax_s.clear()
        self.ax_s.pcolormesh(sd.axes[AXIS_X] * 1e6, sd.axes[AXIS_Y] * 1e6,
                             sd.distr.T, shading="auto")
        self.ax_s.set_xlabel("x [um]")
        self.ax_s.set_ylabel("y [um]")
        self.ax_s.set_title("Photon transverse (spatial) distribution at emission")
        self.fig_s.tight_layout()
        self.canvas_s.draw()

    def _render_angular_distribution(self, res):
        self.ax_a.clear()
        ang2d = find_slice(res.photon_slices, AXIS_THETA_X, AXIS_THETA_Y)
        if ang2d is not None:
            self.ax_a.pcolormesh(ang2d.axes[AXIS_THETA_X] * 1e3, ang2d.axes[AXIS_THETA_Y] * 1e3,
                                 ang2d.distr.T, shading="auto")
        else:
            ang3d = find_slice(res.photon_slices, AXIS_ENERGY, AXIS_THETA_X, AXIS_THETA_Y)
            if ang3d is None:
                self._render_unavailable(
                    self.ax_a, self.canvas_a, self.fig_a,
                    "Angular distribution", "N/A (no angular data)")
                return
            # angle-only density: integrate the joint energy+angle slice
            # over the energy axis (a pure aggregation of already-present
            # data, no new adapter/engine call). ``axes``' iteration order
            # matches ``distr``'s dimension order, but different adapters
            # may order that dict differently -- reorder explicitly to
            # (theta_x, theta_y, energy) rather than assume a fixed layout.
            names = list(ang3d.axes.keys())
            order = [names.index(AXIS_THETA_X), names.index(AXIS_THETA_Y), names.index(AXIS_ENERGY)]
            arr = np.moveaxis(ang3d.distr, order, [0, 1, 2])
            dE = np.gradient(ang3d.axes[AXIS_ENERGY])
            d2N_dOmega = np.einsum("ijk,k->ij", arr, dE)
            self.ax_a.pcolormesh(ang3d.axes[AXIS_THETA_X] * 1e3, ang3d.axes[AXIS_THETA_Y] * 1e3,
                                 d2N_dOmega.T, shading="auto")
        self.ax_a.set_xlabel(r"$\theta_x$ [mrad]")
        self.ax_a.set_ylabel(r"$\theta_y$ [mrad]")
        self.ax_a.set_title("Photon angular distribution (energy-integrated)")
        self.fig_a.tight_layout()
        self.canvas_a.draw()

    def _render_angle_resolved_spectra(self, res):
        """Four (energy, angle) views: on-axis slices at the other
        transverse angle's nearest-to-zero grid point, and the two
        one-angle-summed (i.e. energy vs. one angle, integrated over the
        other) projections.

        Data source: if a collimation window is set (theta_x_col_mrad/
        theta_y_col_mrad on the Compton Photons tab) and the active model
        supports it, a fresh on-demand spectrum_in_angular_range() query
        sized for THAT window -- mirrors _render_spectrum's "collimated"
        curve and _photon_fluxes' collimated-flux stat, both of which
        already avoid reslicing the coarse generous grid run() precomputes
        (see their docstrings for why: a tight collimation window can
        capture just one or two points of that wide-range grid, badly
        overcounting). angle_energy_grid is the joint (theta_x, theta_y,
        energy) grid the same call already builds internally before
        collapsing it to spectrum_in_angular_range's own 1D angle-
        integrated spectrum -- exposing it costs nothing extra to compute.
        Falls back to the joint (AXIS_ENERGY, AXIS_THETA_X, AXIS_THETA_Y)
        slice from res.photon_slices (the coarse generous window) when no
        collimation window is set, or the model has no
        spectrum_in_angular_range (kascade, analytical)."""
        axes_ar = (self.ax_ar_tx0, self.ax_ar_ty0, self.ax_ar_tx_sum, self.ax_ar_ty_sum)

        ang3d = None
        tx, ty = self._collimation_rad()
        if tx is not None and ty is not None and hasattr(self.active_adapter, "spectrum_in_angular_range"):
            n_energy_bins = max(1, int(float(self.fields["n_energy_bins"].get())))
            rng_result = self.active_adapter.spectrum_in_angular_range(
                (-tx, tx), (-ty, ty), n_energy=n_energy_bins)
            ang3d = rng_result.angle_energy_grid
        if ang3d is None:
            ang3d = find_slice(res.photon_slices, AXIS_ENERGY, AXIS_THETA_X, AXIS_THETA_Y)
        if ang3d is None:
            for ax in axes_ar:
                ax.clear()
                ax.text(0.5, 0.5, "N/A (no angle-resolved spectrum data)",
                        ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
            self.ax_ar_tx0.set_title(r"$s$-$\theta_x$ at $\theta_y=0$")
            self.ax_ar_ty0.set_title(r"$s$-$\theta_y$ at $\theta_x=0$")
            self.ax_ar_tx_sum.set_title(r"$s$-$\theta_x$, summed over $\theta_y$")
            self.ax_ar_ty_sum.set_title(r"$s$-$\theta_y$, summed over $\theta_x$")
            self.fig_ar.tight_layout()
            self.canvas_ar.draw()
            return

        # ``axes``' iteration order matches ``distr``'s dimension order, but
        # different adapters may order that dict differently -- reorder
        # explicitly, same as _render_angular_distribution's ang3d branch.
        names = list(ang3d.axes.keys())
        order = [names.index(AXIS_THETA_X), names.index(AXIS_THETA_Y), names.index(AXIS_ENERGY)]
        arr = np.moveaxis(ang3d.distr, order, [0, 1, 2])  # (theta_x, theta_y, energy)
        theta_x = ang3d.axes[AXIS_THETA_X]
        theta_y = ang3d.axes[AXIS_THETA_Y]
        E_keV = ang3d.axes[AXIS_ENERGY] / 1e3
        ix0 = int(np.argmin(np.abs(theta_x)))
        iy0 = int(np.argmin(np.abs(theta_y)))
        dtx, dty = np.gradient(theta_x), np.gradient(theta_y)

        for ax in axes_ar:
            ax.clear()

        # Top row: on-axis slices at the other angle's nearest-to-zero point.
        self.ax_ar_tx0.pcolormesh(E_keV, theta_x * 1e3, arr[:, iy0, :], shading="auto")
        self.ax_ar_tx0.set_title(r"$s$-$\theta_x$ at $\theta_y \approx$"
                                  f" {theta_y[iy0] * 1e3:.3g} mrad")

        self.ax_ar_ty0.pcolormesh(E_keV, theta_y * 1e3, arr[ix0, :, :], shading="auto")
        self.ax_ar_ty0.set_title(r"$s$-$\theta_y$ at $\theta_x \approx$"
                                  f" {theta_x[ix0] * 1e3:.3g} mrad")

        # Bottom row: summed (quadrature-weighted) over the other angle.
        d2N_dE_dtx = np.einsum("ijk,j->ik", arr, dty)
        self.ax_ar_tx_sum.pcolormesh(E_keV, theta_x * 1e3, d2N_dE_dtx, shading="auto")
        self.ax_ar_tx_sum.set_title(r"$s$-$\theta_x$, summed over $\theta_y$")

        d2N_dE_dty = np.einsum("ijk,i->jk", arr, dtx)
        self.ax_ar_ty_sum.pcolormesh(E_keV, theta_y * 1e3, d2N_dE_dty, shading="auto")
        self.ax_ar_ty_sum.set_title(r"$s$-$\theta_y$, summed over $\theta_x$")

        for ax, ylabel in ((self.ax_ar_tx0, r"$\theta_x$ [mrad]"),
                            (self.ax_ar_ty0, r"$\theta_y$ [mrad]"),
                            (self.ax_ar_tx_sum, r"$\theta_x$ [mrad]"),
                            (self.ax_ar_ty_sum, r"$\theta_y$ [mrad]")):
            ax.set_xlabel(r"photon energy $\hbar\omega_\gamma$ [keV]")
            ax.set_ylabel(ylabel)

        self.fig_ar.tight_layout()
        self.canvas_ar.draw()


def main():
    app = ComptonGUIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
