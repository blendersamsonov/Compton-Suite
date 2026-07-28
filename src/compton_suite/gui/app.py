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
hardcoded import. Model-specific controls (crossing angle, quantum toggle,
.ele loading, new-observable tabs, ...) are greyed out per the active
model's ``capabilities()``; see ``_apply_model_capabilities``.

Run: ``python3 -m compton_suite.gui.app`` or ``scripts/run_gui.py``.
"""

from __future__ import annotations

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

from compton_suite.gui.physics_constants import C_LIGHT, MEC2_EV

from compton_suite.io.laser import a0_from_fields, focal_radii_m
from compton_suite.io.bunch import sigma_from_emittance
from compton_suite.io.interaction import recoil_parameter

from compton_suite.gui.model_api import ModelAdapter, SampledSpectrum, validate_results
from compton_suite.gui.models import discover_models

from compton_suite.io.bunch import beam_from_shared_fields, sample_gaussian_bunch

from compton_suite.gui.calculations import CalculationsSection

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


def peak_a0(fields: dict):
    """Peak normalised vector potential a_0 from the laser fields (or None).

    Delegates to :func:`compton_suite.io.laser.a0_from_fields` -- the
    single source of truth for a0 computation.
    """
    from compton_suite.io.converters import fwhm_to_sigma_intensity
    from compton_suite.io.enums import TimeConvention

    lam_nm = _float_or_none(fields["laser_wavelength_nm"])
    e_mJ = _float_or_none(fields["laser_energy_mJ"])
    dur_ps = _float_or_none(fields["pulse_duration_ps"])
    R_m = _float_or_none(fields["rayleigh_length_m"])
    if None in (lam_nm, e_mJ, dur_ps, R_m) or min(lam_nm, e_mJ, dur_ps, R_m) <= 0:
        return None
    sigma0_l = 0.5 * np.sqrt(R_m * lam_nm * 1e-9 / np.pi)

    # Convert duration based on convention (RMS or FWHM)
    # Check if we're in GUI context (has _pulse_convention_var)
    if hasattr(peak_a0, '_pulse_convention_var') and peak_a0._pulse_convention_var is not None:
        convention = peak_a0._pulse_convention_var.get()
    else:
        convention = "RMS"  # Default for non-GUI usage

    if convention == "FWHM":
        duration_rms_s = C_LIGHT * fwhm_to_sigma_intensity(dur_ps * 1e-12)
    else:  # RMS
        duration_rms_s = C_LIGHT * (dur_ps * 1e-12)

    return float(a0_from_fields(
        wavelength_m=lam_nm * 1e-9,
        pulse_energy_J=e_mJ * 1e-3,
        waist_rms_m=sigma0_l,
        duration_rms_s=duration_rms_s,
    ))


def sigma_e(emit_norm_mmmrad, beta_m, gamma):
    """Transverse rms bunch size [m] from normalized emittance, beta [m] and gamma.

    Delegates to :func:`compton_suite.io.bunch.sigma_from_emittance`.
    """
    if emit_norm_mmmrad is None or beta_m is None or gamma is None:
        return None
    val = sigma_from_emittance(emit_norm_mmmrad * 1e-6, beta_m, gamma)
    return val if val > 0 else None


def laser_focal_radii(wavelength_nm, rayleigh_length_m):
    """Return radial RMS, FWHM, and exp(-1/2) focal radii [m].

    Delegates to :func:`compton_suite.io.laser.focal_radii_m`.
    """
    if (wavelength_nm is None or rayleigh_length_m is None
            or wavelength_nm <= 0 or rayleigh_length_m <= 0):
        return None
    return focal_radii_m(
        wavelength_m=wavelength_nm * 1e-9,
        rayleigh_length_m=rayleigh_length_m,
    )


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
class ComptonGuideApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compton-GUIde")
        self.geometry("1280x960+20+10")

        self.fields: dict[str, tk.StringVar] = {}
        self.quantum_var = tk.BooleanVar(value=False)  # False = classical, True = quantum
        self.nonlin_var = tk.BooleanVar(value=True)    # xigma-i-only: emulate a0 downshift
        self.res = None                 # model_api.CommonResults | None
        self.preview_res = None         # always-on analytical model's CommonResults | None
        self.cfg_used = None            # active model's Config-shaped object | None
        self.rep_rate_hz = 1.0
        self.a0_used = 0.0
        self.q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None

        # Model registry: discover once, pick kascade as the startup default
        # (it's always available; xigma-i may be registered as an
        # UnavailableAdapter if cupy/CUDA isn't usable on this machine).
        self.models = discover_models()
        self.model_var = tk.StringVar(value="kascade")
        self.active_adapter: ModelAdapter = self.models["kascade"]

        # The fast analytical model (ModelCapabilities.is_fast_preview) runs
        # automatically alongside whichever model is selected, on every
        # Calculate click -- a real-time preview and base sanity check, not
        # one of the models the Model menu switches between. None if it
        # isn't registered/available (e.g. compton_suite not discoverable).
        self.preview_adapter = next(
            (a for a in self.models.values()
             if a.available()[0] and getattr(a.capabilities(), "is_fast_preview", False)), None)
        # The preview adapter's own extra_params() (e.g. a collimation
        # angle) need a value in self.fields even when it isn't the active
        # model -- _rebuild_model_params_panel only ever seeds the ACTIVE
        # model's extra fields, so without this the preview would KeyError
        # on startup unless the user happened to switch to it once first.
        # No widget is created for these (invisible until the user actually
        # selects this model) -- just a usable default for the background run.
        if self.preview_adapter is not None:
            for _label, default, key in self.preview_adapter.extra_params():
                self.fields.setdefault(key, tk.StringVar(value=str(default)))

        # SDDS-bunch loading state.  ``loaded_bunch`` is the MacroBunch
        # returned by the active adapter's ``load_ele_file`` (per-particle
        # arrays + header parameters in ``.meta``) and is fed to the
        # adapter's ``run`` as the ``electrons`` argument.  ``loaded_path``
        # is the source filename shown in the menu.
        self.loaded_bunch = None  # compton_suite.io.bunch.MacroBunch | None
        self.loaded_path: str | None = None
        # Cached list of (Entry widget, key) tuples for the Electron panel
        # input fields, so we can flip their state when the panel switches
        # between input-mode and display-mode.
        self._electron_entries: list[tuple[tk.Entry, str]] = []
        # n_mc / seed entries live in the Electrons panel but are tracked
        # separately -- they must NOT be set read-only when a .ele file
        # is loaded (they are sampling controls, not beam parameters).
        self._sample_entries: list[tuple[tk.Entry, str]] = []
        # Trace ids we attach to electron-panel StringVars so we can detach
        # them while the panel is in display mode (otherwise the
        # ``_update_derived`` callback would clobber our output values).
        self._electron_traces: list[tuple[tk.StringVar, str]] = []
        
        # Calculations section (separate window)
        self.calculations_window: tk.Toplevel | None = None
        self.calculations_section: CalculationsSection | None = None

        # --- build layout ---
        self._build_menu()

        # Resizable split (PanedWindow must own its pane frames)
        self._paned = tk.PanedWindow(self, orient="horizontal",
                                     sashwidth=6, sashrelief="groove")
        self._paned.pack(fill="both", expand=True)

        # Left pane: all IO panels (electrons, laser, compton, model params, preview)
        self._left_frame = tk.Frame(self._paned)
        self._build_trust_banner()
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
        self._apply_model_capabilities()

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
        filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.filemenu = filemenu
        self._ele_menu_indices = (0, 1)  # "Load file...", "Clear loaded file"

        modelmenu = tk.Menu(menubar, tearoff=0)
        for name, adapter in self.models.items():
            caps = adapter.capabilities()
            # Skip the analytical model -- it runs always in the background
            # as a preview, not as a user-selectable model.
            if caps.is_fast_preview:
                continue
            modelmenu.add_radiobutton(
                label=caps.display_name, variable=self.model_var, value=name,
                command=self._on_model_selected)
            available, reason = adapter.available()
            if not available:
                modelmenu.entryconfig(caps.display_name, state="disabled")
        menubar.add_cascade(label="Model", menu=modelmenu)
        self.modelmenu = modelmenu

        optmenu = tk.Menu(menubar, tearoff=0)
        optmenu.add_radiobutton(label="Classical",
                                 variable=self.quantum_var, value=False)
        optmenu.add_radiobutton(label="Quantum corrections",
                                 variable=self.quantum_var, value=True)
        optmenu.add_separator()
        optmenu.add_checkbutton(label="Emulate nonlinearity (a0 downshift)",
                                 variable=self.nonlin_var)
        menubar.add_cascade(label="Options", menu=optmenu)
        self.optmenu = optmenu
        self._quantum_menu_indices = (0, 1)
        self._nonlin_menu_index = 3

        calcmenu = tk.Menu(menubar, tearoff=0)
        calcmenu.add_command(label="Calculate", command=self.on_start)
        calcmenu.add_separator()
        calcmenu.add_command(label="Multi-Model Calculations...", command=self._open_calculations_section)
        menubar.add_cascade(label="Calculations", menu=calcmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.config(menu=menubar)

    def _build_trust_banner(self):
        self.trust_lbl = tk.Label(self._left_frame, text="", anchor="w",
                                   font=("TkDefaultFont", 9, "italic"))
        self.trust_lbl.pack(side="top", fill="x", padx=8, pady=(2, 0))

    # ---- model selection --------------------------------------------------
    def _on_model_selected(self):
        name = self.model_var.get()
        self.active_adapter = self.models[name]
        self._rebuild_model_params_panel()
        self._apply_model_capabilities()

    def _apply_model_capabilities(self):
        """Grey out / restore controls the active model doesn't support, and
        refresh the trust banner. Called on startup and on every model switch."""
        caps = self.active_adapter.capabilities()

        # crossing angle: force to 0 and disable when unsupported
        crossing_entry = self._laser_entry_by_key.get("crossing_angle")
        if crossing_entry is not None:
            if caps.supports_crossing_angle:
                crossing_entry.config(state="normal")
            else:
                self.fields["crossing_angle"].set("0")
                crossing_entry.config(state="disabled")

        # quantum toggle
        for idx in self._quantum_menu_indices:
            self.optmenu.entryconfig(
                idx, state="normal" if caps.supports_quantum_toggle else "disabled")
        if not caps.supports_quantum_toggle:
            self.quantum_var.set(False)

        # nonlinearity-emulation toggle (xigma-i-only axis)
        self.optmenu.entryconfig(
            self._nonlin_menu_index,
            state="normal" if caps.supports_nonlinearity_emulation else "disabled")

        # .ele file I/O
        for idx in self._ele_menu_indices:
            self.filemenu.entryconfig(
                idx, state="normal" if caps.supports_ele_file_io else "disabled")
        if not caps.supports_ele_file_io and self.loaded_bunch is not None:
            messagebox.showwarning(
                "Model note",
                f"{caps.display_name} does not support loaded .ele bunches; "
                "clearing the loaded file.")
            self._clear_loaded_ele()

        # n_mc is always editable -- it's a model-agnostic electron bunch
        # parameter, not model-specific. IO module uses it to sample the
        # MacroBunch that gets passed to every model's run().

        # new-observable tabs
        for tab, supported in (
                (self.tab_temporal, caps.supports_temporal_envelope),
                (self.tab_spatial, caps.supports_spatial_distribution),
                (self.tab_angular, caps.supports_angular_distribution)):
            self.notebook.tab(tab, state="normal" if supported else "disabled")

        # trust banner
        colour = "#175" if caps.trust_level == "production" else "#a52"
        self.trust_lbl.config(
            text=f"Model: {caps.display_name} - {caps.trust_level}"
                 + (f"  ({caps.trust_note})" if caps.trust_level != "production" else ""),
            fg=colour)

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Compton-GUIde\n\n"
            "Model-agnostic GUI front-end for pluggable Compton-scattering "
            "physics engines (kascade, xigma-i, ...).")
    
    def _open_calculations_section(self):
        """Open the multi-model calculations section in a separate window."""
        if self.calculations_window is not None and self.calculations_window.winfo_exists():
            # Bring existing window to front
            self.calculations_window.lift()
            self.calculations_window.focus_force()
            return
        
        # Create new window
        self.calculations_window = tk.Toplevel(self)
        self.calculations_window.title("Multi-Model Calculations")
        self.calculations_window.geometry("1200x800+100+100")
        
        # Create calculations section
        self.calculations_section = CalculationsSection(
            self.calculations_window, self.fields)
        self.calculations_section.pack(fill="both", expand=True)
        
        # Handle window close
        self.calculations_window.protocol("WM_DELETE_WINDOW", self._close_calculations_section)
    
    def _close_calculations_section(self):
        """Close the calculations section window."""
        if self.calculations_window is not None:
            self.calculations_window.destroy()
            self.calculations_window = None
            self.calculations_section = None

    def _save_fig(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=150)

    # ---- SDDS .ele file load / clear -----------------------------------
    def _load_ele(self):
        """Load a 6-D electron bunch from an SDDS ``.ele`` file and turn the
        Electrons panel into an output display of the parameters that the
        simulation will actually use."""
        path = filedialog.askopenfilename(
            title="Load 6-D electron distribution",
            defaultextension=".ele",
            filetypes=[("SDDS electron distribution", "*.ele"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            bunch = self.active_adapter.load_ele_file(path)
        except Exception as e:
            messagebox.showerror(
                "Cannot load .ele file",
                f"Failed to parse {os.path.basename(path)}:\n\n{e}")
            return
        try:
            summary = self.active_adapter.ele_file_summary(bunch)
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

        # MC sample count & seed (row 3, below the sigma labels)
        for col, (label, default, key) in enumerate([
                ("# electrons (MC)", 200000, "n_mc"),
                ("Random seed", 1, "seed")]):
            tk.Label(p, text=label, bg=BLUE, anchor="w").grid(
                row=3, column=col * 2, sticky="w", padx=(8, 3), pady=3)
            var = tk.StringVar(value=str(default))
            ent = tk.Entry(p, textvariable=var, width=11, justify="right")
            ent.grid(row=3, column=col * 2 + 1, padx=(0, 10), pady=3)
            self.fields[key] = var
            self._sample_entries.append((ent, key))

        # hint that flips between "input" and "loaded from <file>" in display mode
        self.electron_mode_lbl = tk.Label(
            p, text="(input mode - edit the fields above to define the bunch)",
            bg=BLUE, anchor="w", fg="#234",
            font=("TkDefaultFont", 9, "italic"))
        self.electron_mode_lbl.grid(row=4, column=0, columnspan=12, sticky="w",
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
        ``self.active_adapter.extra_params()``. Never re-packs the frame
        itself (only its children) so switching models repeatedly doesn't
        reorder it relative to the plot area below."""
        for child in self.model_params_frame.winfo_children():
            child.destroy()

        specs = self.active_adapter.extra_params()
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
        choices = self.active_adapter.extra_choices()
        add_field_grid(self.model_params_frame, seeded, self.fields,
                       n_cols=4, bg=GREY, width=8, choices=choices)

    # ---- output resolution panel (model-agnostic) -----------------------
    def _build_output_panel(self):
        """Panel for configuring output resolution (spectrum, temporal,
        spatial, angular bins). These values are passed to models via
        OutputSpec."""
        p = tk.LabelFrame(
            self._left_frame, text="OUTPUT RESOLUTION", bg=GREY, fg="#123",
            font=("TkDefaultFont", 14, "bold"))
        p.pack(side="top", fill="x", padx=6, pady=3)

        # Default values matching OutputSpec
        output_specs = [
            ("Energy bins", 256, "n_energy_bins"),
            ("Time bins", 128, "n_time_bins"),
            ("Spatial bins X", 64, "n_spatial_bins_x"),
            ("Spatial bins Y", 64, "n_spatial_bins_y"),
            ("Angular bins X", 64, "n_angular_bins_x"),
            ("Angular bins Y", 64, "n_angular_bins_y"),
        ]
        add_field_grid(p, output_specs, self.fields, n_cols=4, bg=GREY, width=8)

    def _get_output_spec(self):
        """Build an OutputSpec from the GUI fields."""
        from compton_suite.gui.model_api import OutputSpec
        return OutputSpec(
            n_energy_bins=max(1, int(float(self.fields["n_energy_bins"].get()))),
            n_time_bins=max(1, int(float(self.fields["n_time_bins"].get()))),
            n_spatial_bins_x=max(1, int(float(self.fields["n_spatial_bins_x"].get()))),
            n_spatial_bins_y=max(1, int(float(self.fields["n_spatial_bins_y"].get()))),
            n_angular_bins_x=max(1, int(float(self.fields["n_angular_bins_x"].get()))),
            n_angular_bins_y=max(1, int(float(self.fields["n_angular_bins_y"].get()))),
        )

    # ---- analytical preview panel (always-on, independent of the
    # selected model) --------------------------------------------------
    def _build_preview_panel(self):
        """Small always-visible panel showing the fast analytical model's
        estimate, run automatically alongside whichever model is actually
        selected (see on_start/_poll_queue) -- a real-time preview and base
        sanity check, not gated by the active model's tab-capabilities the
        way the plot-area tabs are (_apply_model_capabilities)."""
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
        if self.preview_adapter is None:
            self.preview_lbls["status"].config(text="unavailable")

    def _render_preview(self):
        """Update the preview panel from self.preview_res (set by
        _poll_queue). Never raises -- a malformed/missing preview result
        just shows as unavailable, it must never break the main model's
        own output rendering."""
        if self.preview_adapter is None:
            return
        if self.preview_res is None:
            self.preview_lbls["status"].config(text="failed")
            self.preview_lbls["total_yield"].config(text="--")
            self.preview_lbls["width"].config(text="--")
            return
        try:
            self.preview_lbls["status"].config(text="ok")
            self.preview_lbls["total_yield"].config(text=f"{self.preview_res.total_yield:.4e}")
            width = self.preview_res.summary.get("estimated_spectrum_width_fwhm")
            self.preview_lbls["width"].config(text=f"{width:.4g}" if width is not None else "--")
        except Exception:
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
        from compton_suite.io.converters import sigma_intensity_to_fwhm, sigma_intensity_to_w0
        from compton_suite.io.enums import WidthConvention
        from compton_suite.io.units import Q_

        energy_mev = _float_or_none(self.fields["mean_energy_MeV"])
        gamma = energy_mev * 1e6 / MEC2_EV if energy_mev is not None else None
        sx = sigma_e(_float_or_none(self.fields["emit_x_mmmrad"]),
                     _float_or_none(self.fields["beta_x_m"]), gamma)
        sy = sigma_e(_float_or_none(self.fields["emit_y_mmmrad"]),
                     _float_or_none(self.fields["beta_y_m"]), gamma)

        # Show beam size in multiple conventions (RMS, FWHM, 1/e²)
        if sx:
            sx_fwhm = sigma_intensity_to_fwhm(sx) * 1e6
            sx_1e2 = sigma_intensity_to_w0(sx) * 1e6
            self.sigma_ex_lbl.config(text=f"sigma_x = {sx*1e6:.3f} um")
            self.sigma_ex_fwhm_lbl.config(text=f"(FWHM: {sx_fwhm:.3f} um)")
            self.sigma_ex_1e2_lbl.config(text=f"(1/e²: {sx_1e2:.3f} um)")
        else:
            self.sigma_ex_lbl.config(text="sigma_x = --")
            self.sigma_ex_fwhm_lbl.config(text="(FWHM: --)")
            self.sigma_ex_1e2_lbl.config(text="(1/e²: --)")

        if sy:
            sy_fwhm = sigma_intensity_to_fwhm(sy) * 1e6
            sy_1e2 = sigma_intensity_to_w0(sy) * 1e6
            self.sigma_ey_lbl.config(text=f"sigma_y = {sy*1e6:.3f} um")
            self.sigma_ey_fwhm_lbl.config(text=f"(FWHM: {sy_fwhm:.3f} um)")
            self.sigma_ey_1e2_lbl.config(text=f"(1/e²: {sy_1e2:.3f} um)")
        else:
            self.sigma_ey_lbl.config(text="sigma_y = --")
            self.sigma_ey_fwhm_lbl.config(text="(FWHM: --)")
            self.sigma_ey_1e2_lbl.config(text="(1/e²: --)")

        # Handle waist diameter / Rayleigh length conversion
        wavelength_m = _float_or_none(self.fields["laser_wavelength_nm"])
        if wavelength_m is not None:
            wavelength_m *= 1e-9

        if self._waist_input_mode.get() == "diameter":
            # Convert diameter to Rayleigh length
            waist_diameter_um = _float_or_none(self.fields["waist_diameter_um"])
            if waist_diameter_um is not None and wavelength_m is not None:
                waist_convention = self._waist_convention_var.get()
                waist_diameter_m = waist_diameter_um * 1e-6
                # Convert diameter to sigma_intensity (RMS radius)
                if waist_convention == "RMS":
                    sigma_intensity = waist_diameter_m / 2.0
                elif waist_convention == "FWHM":
                    from compton_suite.io.converters import fwhm_to_sigma_intensity
                    sigma_intensity = fwhm_to_sigma_intensity(waist_diameter_m / 2.0)
                else:  # 1/e²
                    from compton_suite.io.converters import w0_to_sigma_intensity
                    sigma_intensity = w0_to_sigma_intensity(waist_diameter_m / 2.0)
                # Rayleigh length: z_R = pi * w0^2 / lambda, where w0 = 2 * sigma_intensity
                w0 = 2 * sigma_intensity
                z_R = np.pi * w0**2 / wavelength_m
                self._rayleigh_var.set(f"{z_R:.6g}")
            else:
                z_R = None
        else:
            # Use Rayleigh length directly
            z_R = _float_or_none(self.fields["rayleigh_length_m"])

        a0 = peak_a0(self.fields)
        self.a0_lbl.config(text=f"a_0 : {a0:.4g}" if a0 is not None else "a_0 : --")

        radii = laser_focal_radii(wavelength_m / 1e-9 if wavelength_m else None, z_R)
        radius_names = ("RMS", "FWHM", "e^{-1/2}")
        for label, name, radius in zip(
                self.laser_radius_lbls, radius_names,
                radii if radii is not None else (None,) * 3):
            text = f"{name} radius = {radius*1e6:.3f} um" if radius else f"{name} radius = --"
            label.config(text=text)

    # ---- run / threading ------------------------------------------------
    def on_start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        adapter = self.active_adapter
        try:
            # Add convention selectors to fields for adapters to use
            fields_with_conventions = dict(self.fields)
            fields_with_conventions["_duration_convention"] = self._duration_convention_var.get()
            fields_with_conventions["_pulse_convention"] = self._pulse_convention_var.get()
            cfg, extra = adapter.params_to_config(fields_with_conventions, self.quantum_var.get())
        except Exception as e:
            messagebox.showerror("Invalid parameter", str(e))
            return
        if extra["warnings"]:
            messagebox.showwarning("Model note", "\n\n".join(extra["warnings"]))

        # Electron sampling is the IO layer's job, not each model's own --
        # the GUI draws ONE canonical MacroBunch here (via compton_suite.io) and
        # passes it to every model uniformly. Every adapter's run() now
        # *requires* ``electrons`` (kascade's own sample_initial_electrons
        # and xigma_i/delta's own gui_adapter-level self-sampling
        # were deleted -- the "should not be responsible for that" cross-
        # repo cleanup), so this can no longer be skipped or left None.
        # If a 6-D .ele file was loaded, that takes precedence and IS the
        # bunch (no sampling needed); its length becomes the effective
        # ``n_mc``.
        n_mc = int(extra["n_mc"])
        if self.loaded_bunch is not None:
            electrons = self.loaded_bunch
            n_mc = self.loaded_bunch.n_particles
        else:
            # IO module samples n_mc particles into a MacroBunch that gets
            # passed to every model's run(). Models read electrons.n_particles
            # if they need the count.
            n_sample = n_mc
            beam = beam_from_shared_fields(
                eps0=cfg.eps0, sigma_eps_rel=cfg.sigma_eps_rel,
                emit_x=cfg.emit_x, emit_y=cfg.emit_y,
                sigma0_x=cfg.sigma0_x, sigma0_y=cfg.sigma0_y,
                sigma_par_e=cfg.sigma_par_e, N_e=cfg.N_e,
            )
            electrons = sample_gaussian_bunch(
                beam, n_particles=n_sample, rng=np.random.default_rng(int(extra["seed"])))

        self.cfg_used = cfg
        self.rep_rate_hz = extra["rep_rate_hz"]
        self.a0_used = peak_a0(self.fields) or 0.0
        self.calc_btn.config(state="disabled")
        self.status_lbl.config(text="running...")

        # The always-on analytical preview (see _build_preview_panel) runs
        # alongside the selected model on every Calculate click, using the
        # same shared fields. params_to_config is parsed synchronously here
        # (like the main model's cfg above) so a bad preview config doesn't
        # silently swallow a real parameter error; running it happens in
        # the worker thread below, wrapped so a preview failure can never
        # block or corrupt the main model's own result.
        preview_cfg = preview_extra = None
        if self.preview_adapter is not None:
            try:
                preview_cfg, preview_extra = self.preview_adapter.params_to_config(
                    fields_with_conventions, self.quantum_var.get())
            except Exception:
                preview_cfg = None

        # Build OutputSpec from GUI fields
        output_spec = self._get_output_spec()

        def work():
            try:
                res = adapter.run(cfg, n_mc=n_mc, seed=extra["seed"],
                                  electrons=electrons, output=output_spec)
                problems = validate_results(res)
                if problems:
                    raise RuntimeError(
                        f"{adapter.capabilities().display_name} adapter returned "
                        f"malformed results: {'; '.join(problems)}")
            except Exception as e:
                self.q.put(("error", "".join(traceback.format_exception(e))))
                return

            preview_res = None
            if preview_cfg is not None:
                try:
                    preview_res = self.preview_adapter.run(
                        preview_cfg, n_mc=int(preview_extra["n_mc"]),
                        seed=int(preview_extra["seed"]), electrons=electrons,
                        output=output_spec)
                except Exception:
                    preview_res = None  # preview is best-effort, never fatal

            self.q.put(("ok", (res, preview_res)))

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
        tx = tx * 1e-3 if tx is not None else None
        ty = ty * 1e-3 if ty is not None else None
        return tx, ty

    def _update_outputs(self):
        if self.res is None or self.cfg_used is None:
            return
        res, cfg = self.res, self.cfg_used
        tx, ty = self._collimation_rad()

        # gamma is still needed below for the recoil-parameter stat
        gamma = cfg.eps0

        # --- fluxes and statistics (branch on result shape) ---
        total_flux, coll_flux, cmask = self._photon_fluxes(res, tx, ty)
        self.stat_lbls["total_flux"].config(text=f"{total_flux:.3e} ph/s")
        self.stat_lbls["coll_flux"].config(text=f"{coll_flux:.3e} ph/s")

        if res.photon_multiplicity is not None:
            pm = res.photon_multiplicity
            self.stat_lbls["nph_ne"].config(text=f"{pm.mean_n_phot:.4e}")
            self.stat_lbls["ne0"].config(text=f"{pm.frac_n0*100:.3f} %")
            self.stat_lbls["ne1"].config(text=f"{pm.frac_n1*100:.3f} %")
            self.stat_lbls["ne2"].config(text=f"{pm.frac_n2*100:.3f} %")
        else:
            for key in ("nph_ne", "ne0", "ne1", "ne2"):
                self.stat_lbls[key].config(text="N/A")

        recoil_q = recoil_parameter(cfg.eps0, cfg.lambda_L)
        self.stat_lbls["recoil_q"].config(text=f"{recoil_q:.6f}")

        self._render_plots(res, cmask)

    def _photon_fluxes(self, res, tx, ty):
        """Return (total_flux, collimated_flux, cmask_or_None) [ph/s].

        cmask is only meaningful (and only returned) for SampledSpectrum
        results, where _render_plots uses it to mask the raw photon array
        directly (exact, no grid involved).

        For BinnedSpectrum results, collimated flux comes from an on-demand
        active_adapter.spectrum_in_angular_range() query -- a fresh grid
        sized for the ACTUAL requested window -- rather than re-integrating
        the cached, wide-range res.angular_spectrum grid (used only by the
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
        if isinstance(res.spectrum, SampledSpectrum):
            spec = res.spectrum
            n_tot = spec.E_eV.size
            total_flux = n_tot * spec.weight * self.rep_rate_hz
            photon_samples = res.photon_samples
            thx = getattr(photon_samples, "ph_thx_lab", None)
            thy = getattr(photon_samples, "ph_thy_lab", None)
            if tx is not None and ty is not None and n_tot and thx is not None:
                cmask = (np.abs(thx) <= tx) & (np.abs(thy) <= ty)
            else:
                cmask = np.zeros(n_tot, dtype=bool)
            coll_flux = int(cmask.sum()) * spec.weight * self.rep_rate_hz
            return total_flux, coll_flux, cmask

        total_flux = res.total_yield * self.rep_rate_hz
        caps = self.active_adapter.capabilities()
        if tx is None or ty is None or not caps.supports_angular_range_spectrum:
            return total_flux, 0.0, None
        rng_result = self.active_adapter.spectrum_in_angular_range((-tx, tx), (-ty, ty))
        coll_flux = (rng_result.n_photons_in_range or 0.0) * self.rep_rate_hz
        return total_flux, coll_flux, None

    # ---- plotting -------------------------------------------------------
    def _render_plots(self, res, cmask):
        if isinstance(res.spectrum, SampledSpectrum):
            self._render_spectrum_sampled(res, cmask)
        else:
            self._render_spectrum_binned(res)

        if res.electron_state is not None:
            self._render_electron_state(res.electron_state)
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

    def _render_spectrum_sampled(self, res, cmask):
        spec = res.spectrum
        self.ax_spec.clear()
        E_keV = spec.E_eV / 1e3
        Ec = E_keV[cmask] if cmask is not None and cmask.size else E_keV[:0]
        if E_keV.size:
            emax = E_keV.max() * 1.02
            self.ax_spec.hist(E_keV, bins=120, range=(0, emax),
                              weights=np.full(E_keV.size, spec.weight),
                              color="0.8", label="all (4*pi)")
        if Ec.size:
            self.ax_spec.hist(Ec, bins=120, range=(0, E_keV.max() * 1.02),
                              weights=np.full(Ec.size, spec.weight),
                              histtype="step", color="crimson", lw=1.5,
                              label="collimated")
        self.ax_spec.set_xlabel(r"photon energy $\hbar\omega_\gamma$ [keV]")
        self.ax_spec.set_ylabel("photons / bin")
        self.ax_spec.set_title("Collimated photon spectrum")
        self.ax_spec.legend(fontsize=8)

    def _render_spectrum_binned(self, res):
        spec = res.spectrum
        self.ax_spec.clear()
        E_keV = spec.E_eV / 1e3
        dNdE_per_keV = spec.dNdE_per_eV * 1e3
        self.ax_spec.plot(E_keV, dNdE_per_keV, color="0.4", label="all (4*pi)")

        tx, ty = self._collimation_rad()
        caps = self.active_adapter.capabilities()
        if tx is not None and ty is not None and caps.supports_angular_range_spectrum:
            # On-demand query at a grid sized for THIS window, not the
            # wide-range angular_spectrum cache -- see _photon_fluxes'
            # docstring for why re-integrating that cache here used to
            # produce a "collimated" curve that could spike far above the
            # true 4*pi spectrum near the Compton edge.
            rng_result = self.active_adapter.spectrum_in_angular_range((-tx, tx), (-ty, ty))
            coll_spec = rng_result.spectrum
            if hasattr(coll_spec, "dNdE_per_eV") and coll_spec.E_eV.size:
                self.ax_spec.plot(coll_spec.E_eV / 1e3, coll_spec.dNdE_per_eV * 1e3,
                                  color="crimson", label="collimated")

        self.ax_spec.set_xlabel(r"photon energy $\hbar\omega_\gamma$ [keV]")
        self.ax_spec.set_ylabel(r"dN/dE [photons / keV]")
        self.ax_spec.set_title("Photon spectrum (semi-analytic, binned)")
        self.ax_spec.legend(fontsize=8)

    def _render_electron_state(self, electron_state):
        self.ax_e.clear()
        E_i = electron_state.eps_i * MEC2_EV / 1e6
        E_f = electron_state.eps_f * MEC2_EV / 1e6
        w = np.full(E_i.size, electron_state.weight)
        rng = (min(E_f.min(), E_i.min()), E_i.max())
        self.ax_e.hist(E_i, bins=120, range=rng, weights=w, histtype="step",
                       color="gray", lw=1.3, label="initial")
        self.ax_e.hist(E_f, bins=120, range=rng, weights=w, histtype="step",
                       color="crimson", lw=1.3, label="final (total)")
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
        te = res.temporal_envelope
        if te is None:
            self._render_unavailable(
                self.ax_t, self.canvas_t, self.fig_t, "Temporal envelope",
                "N/A for this model\n(no temporal data)")
            return
        self.ax_t.clear()
        # Duck-typed on shape, not isinstance: adapters for models that
        # shouldn't depend on this GUI project (e.g. xigma_i.gui_adapter)
        # define their own structurally-identical local dataclasses rather
        # than importing SampledTemporalEnvelope/BinnedTemporalEnvelope.
        if hasattr(te, "weight"):
            t_ps = te.t_seconds * 1e12
            if t_ps.size:
                self.ax_t.hist(t_ps, bins=120,
                               weights=np.full(t_ps.size, te.weight),
                               color="0.4")
            self.ax_t.set_ylabel("photons / bin")
        else:
            self.ax_t.plot(te.t_seconds * 1e12, te.rate, color="0.4")
            self.ax_t.set_ylabel("emission rate [a.u.]")
        self.ax_t.set_xlabel("emission time [ps]")
        self.ax_t.set_title("Photon temporal envelope")
        self.fig_t.tight_layout()
        self.canvas_t.draw()

    def _render_spatial_distribution(self, res):
        sd = res.spatial_distribution
        if sd is None:
            self._render_unavailable(
                self.ax_s, self.canvas_s, self.fig_s, "Spatial distribution",
                "N/A for this model\n(no spatial-deposition kernel)")
            return
        self.ax_s.clear()
        # Duck-typed on shape, not isinstance -- see _render_temporal_envelope.
        if hasattr(sd, "weight"):
            x_um = sd.x * 1e6
            y_um = sd.y * 1e6
            if x_um.size:
                h, xedges, yedges = np.histogram2d(
                    x_um, y_um, bins=80,
                    weights=np.full(x_um.size, sd.weight))
                self.ax_s.pcolormesh(xedges, yedges, h.T, shading="auto")
        else:
            self.ax_s.pcolormesh(sd.x_centers * 1e6, sd.y_centers * 1e6,
                                 sd.density.T, shading="auto")
        self.ax_s.set_xlabel("x [um]")
        self.ax_s.set_ylabel("y [um]")
        self.ax_s.set_title("Photon transverse (spatial) distribution at emission")
        self.fig_s.tight_layout()
        self.canvas_s.draw()

    def _render_angular_distribution(self, res):
        self.ax_a.clear()
        if isinstance(res.spectrum, SampledSpectrum):
            photon_samples = res.photon_samples
            thx = getattr(photon_samples, "ph_thx_lab", None)
            thy = getattr(photon_samples, "ph_thy_lab", None)
            weight = getattr(photon_samples, "weight", 1.0)
            if thx is None or thy is None or thx.size == 0:
                self._render_unavailable(
                    self.ax_a, self.canvas_a, self.fig_a,
                    "Angular distribution", "N/A (no photons)")
                return
            thx_mrad = thx * 1e3
            thy_mrad = thy * 1e3
            h, xedges, yedges = np.histogram2d(
                thx_mrad, thy_mrad, bins=80,
                weights=np.full(thx_mrad.size, weight))
            self.ax_a.pcolormesh(xedges, yedges, h.T, shading="auto")
        else:
            ang = res.angular_spectrum
            if ang is None:
                self._render_unavailable(
                    self.ax_a, self.canvas_a, self.fig_a,
                    "Angular distribution", "N/A (no angular data)")
                return
            # angle-only density: integrate the cached d2N/dE dOmega grid
            # over the energy axis (a pure aggregation of already-present
            # data, no new adapter/engine call).
            dE = np.gradient(ang.E_eV)
            d2N_dOmega = np.einsum("ijk,k->ij", ang.d2NdEdOmega, dE)
            self.ax_a.pcolormesh(ang.theta_x * 1e3, ang.theta_y * 1e3,
                                 d2N_dOmega.T, shading="auto")
        self.ax_a.set_xlabel(r"$\theta_x$ [mrad]")
        self.ax_a.set_ylabel(r"$\theta_y$ [mrad]")
        self.ax_a.set_title("Photon angular distribution (energy-integrated)")
        self.fig_a.tight_layout()
        self.canvas_a.draw()


def main():
    app = ComptonGuideApp()
    app.mainloop()


if __name__ == "__main__":
    main()
