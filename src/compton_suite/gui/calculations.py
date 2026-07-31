"""
calculations.py
===============

Multi-model calculation support for the Compton-GUIde main window.

Provides:
- ``ModelSelectionPanel`` -- a frame with model checkboxes that can be
  embedded in the main GUI's left pane.
- ``MultiModelRunner`` -- runs selected models and returns results.
- ``render_multi_spectrum(ax, results)`` -- overlays multiple models on one
  axis with a colour-coded legend.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

from compton_suite.models.api import Photons, discover_models, validate_results


GREY = "#e4e4e4"


class ModelSelectionPanel(tk.Frame):
    """A compact frame showing model checkboxes and a Calculate button,
    designed to replace/integrate into the main GUI's photon panel.

    Each available model (except fast-preview) gets a checkbox + its
    model-specific parameters (shown/hidden via the checkbox).
    """

    def __init__(self, parent, fields: dict[str, tk.StringVar], bg: str = GREY, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)

        self.fields = fields
        self.models = discover_models()

        # Model selection state
        self.model_selections: dict[str, tk.BooleanVar] = {}
        """:obj:`~tk.BooleanVar` per model (True = selected for calculation)."""
        self.model_param_vars: dict[str, dict[str, tk.StringVar]] = {}
        """Model-specific parameter StringVars, keyed by (model_name, param_key)."""
        self.model_param_frames: dict[str, tk.Frame] = {}
        """Frames holding each model's parameter entries (shown/hidden with checkbox)."""

        # Results
        self.results: dict[str, Photons] = {}
        """Latest results keyed by model name."""

        self._build_ui(bg)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self, bg: str):
        title_row = tk.Frame(self, bg=bg)
        title_row.pack(fill="x", padx=4, pady=(2, 4))
        tk.Label(title_row, text="MODELS TO RUN", bg=bg, fg="#123",
                font=("TkDefaultFont", 12, "bold")).pack(side="left", padx=4)

        self._info_lbl = tk.Label(title_row, text="", bg=bg, fg="#456",
                                  font=("TkDefaultFont", 9, "italic"))
        self._info_lbl.pack(side="right", padx=8)

        # Create a notebook for model-specific parameters
        self._param_book = ttk.Notebook(self)
        self._param_book.pack(fill="x", padx=4, pady=2)

        for name, adapter in self.models.items():
            caps = adapter.capabilities()
            if caps.is_fast_preview:
                continue
            self._create_model_page(name, adapter, caps, bg)

        # Action row: Calculate button + status
        action_row = tk.Frame(self, bg=bg)
        action_row.pack(fill="x", padx=4, pady=4)

        self.calc_btn = tk.Button(action_row, text="Calculate",
                                  command=self._on_calculate,
                                  font=("TkDefaultFont", 11, "bold"))
        self.calc_btn.pack(side="left", padx=4)

        self.status_lbl = tk.Label(action_row, text="idle", bg=bg, width=12,
                                   font=("TkDefaultFont", 10))
        self.status_lbl.pack(side="left", padx=8)

    def _create_model_page(self, name: str, adapter, caps, bg: str):
        """Create a notebook page for one model with checkbox + params."""
        page = ttk.Frame(self._param_book)
        self._param_book.add(page, text=caps.display_name)

        # Checkbox row
        check_frame = tk.Frame(page, bg=GREY)
        check_frame.pack(fill="x", padx=4, pady=2)

        sel_var = tk.BooleanVar(value=False)
        self.model_selections[name] = sel_var
        cb = tk.Checkbutton(check_frame, text="Use for calculations",
                            variable=sel_var, bg=GREY,
                            font=("TkDefaultFont", 10, "bold"),
                            command=lambda: self._toggle_page(page, name))
        cb.pack(side="left", padx=4, pady=2)

        # Params frame (hidden until checkbox is on)
        param_frame = tk.Frame(page, bg=GREY)
        self.model_param_frames[name] = param_frame
        self.model_param_vars[name] = {}

        specs = adapter.model_params()
        if specs:
            choices = adapter.model_choices()
            for label_text, default, key in specs:
                row = tk.Frame(param_frame, bg=GREY)
                row.pack(fill="x", padx=8, pady=2)
                tk.Label(row, text=label_text, bg=GREY,
                        font=("TkDefaultFont", 9)).pack(side="left")
                var = tk.StringVar(value=str(default))
                self.model_param_vars[name][key] = var
                if key in choices:
                    w = ttk.Combobox(row, textvariable=var,
                                     values=choices[key], width=12, state="readonly")
                else:
                    w = tk.Entry(row, textvariable=var, width=12, justify="right")
                w.pack(side="right", padx=4)
        else:
            tk.Label(param_frame, text="(no model-specific parameters)",
                    bg=GREY, fg="#567",
                    font=("TkDefaultFont", 9, "italic")).pack(padx=8, pady=8)

    def _toggle_page(self, page, name: str):
        """Show/hide parameter frame when checkbox toggles."""
        frame = self.model_param_frames[name]
        if self.model_selections[name].get():
            frame.pack(fill="both", expand=True, padx=4, pady=2)
        else:
            frame.pack_forget()
        self._update_info()

    def _update_info(self):
        n = sum(1 for v in self.model_selections.values() if v.get())
        self._info_lbl.config(text=f"{n} model(s) selected" if n else "")

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def _on_calculate(self):
        """Called by the Calculate button -- override in the main app or
        connect to a callback.

        This default simply shows the selected models; the main app's
        ``on_start`` actually runs them (see app.py integration).
        """
        selected = self.get_selected_models()
        if not selected:
            # Tell the user but don't crash — the main app's on_start
            # should handle this case.
            self.status_lbl.config(text="no models")
            import tkinter.messagebox as mb
            mb.showwarning("No models", "Select at least one model.")
            return
        self.status_lbl.config(text="ready")
        if hasattr(self, "_on_run_callback") and self._on_run_callback:
            self._on_run_callback(selected)

    def get_selected_models(self) -> list[str]:
        """Return names of models whose checkbox is checked."""
        return [n for n, v in self.model_selections.items() if v.get()]

    # ------------------------------------------------------------------
    # Plotting helpers (called from app.py after results land)
    # ------------------------------------------------------------------
    @staticmethod
    def render_multi_spectrum(ax, results: dict[str, Photons]):
        """Overlay spectra from all models on *ax* with colour-coded legend."""
        ax.clear()
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for (name, res), color in zip(results.items(), colors):
            if res.spectrum is None:
                continue
            spec = res.spectrum
            E_keV = spec.E_eV / 1e3
            if hasattr(spec, "dNdE_per_eV"):
                ax.plot(E_keV, spec.dNdE_per_eV * 1e3, color=color, label=name)
            elif hasattr(spec, "weight"):
                ax.hist(E_keV, bins=120, range=(0, E_keV.max() * 1.02),
                        weights=np.full(E_keV.size, spec.weight),
                        color=color, label=name, histtype="step", lw=1.5)
        ax.set_xlabel(r"photon energy $\hbar\omega_\gamma$ [keV]")
        ax.set_ylabel("dN/dE [photons / keV]")
        ax.legend(fontsize=9)
        ax.set_title("Photon spectrum (all selected models)")

    @staticmethod
    def render_multi_angular(ax, results: dict[str, Photons], active_model: str):
        """Render 2D angular distribution for a specific model."""
        ax.clear()
        res = results.get(active_model)
        if res is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Angular distribution — {active_model}")
            return

        if res.angular_spectrum is not None:
            ang = res.angular_spectrum
            dE = np.gradient(ang.E_eV)
            d2N = np.einsum("ijk,k->ij", ang.d2NdEdOmega, dE)
            ax.pcolormesh(ang.theta_x * 1e3, ang.theta_y * 1e3,
                          d2N.T, shading="auto")
        elif (hasattr(res, "final_photons") and res.final_photons is not None):
            ps = res.final_photons
            tx = getattr(ps, "ph_thx_lab", None)
            ty = getattr(ps, "ph_thy_lab", None)
            w = getattr(ps, "weight", 1.0)
            if tx is not None and ty is not None and tx.size > 0:
                h, xe, ye = np.histogram2d(tx * 1e3, ty * 1e3, bins=80,
                                           weights=np.full(tx.size, w))
                ax.pcolormesh(xe, ye, h.T, shading="auto")
            else:
                ax.text(0.5, 0.5, "No angular data", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "No angular data", ha="center", va="center")

        ax.set_xlabel(r"$\theta_x$ [mrad]")
        ax.set_ylabel(r"$\theta_y$ [mrad]")
        ax.set_title(f"Angular distribution — {active_model}")

    @staticmethod
    def render_multi_temporal(ax, results: dict[str, Photons]):
        """Overlay temporal envelopes from all models."""
        ax.clear()
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for (name, res), color in zip(results.items(), colors):
            te = res.temporal_envelope
            if te is None:
                continue
            if hasattr(te, "weight"):
                t_ps = te.t_seconds * 1e12
                if t_ps.size:
                    ax.hist(t_ps, bins=120,
                            weights=np.full(t_ps.size, te.weight),
                            color=color, label=name, histtype="step", lw=1.5)
            else:
                ax.plot(te.t_seconds * 1e12, te.rate, color=color, label=name)
        ax.set_xlabel("emission time [ps]")
        ax.set_ylabel("photons / bin")
        ax.legend(fontsize=9)
        ax.set_title("Temporal envelope")

    @staticmethod
    def render_multi_spatial(ax, results: dict[str, Photons]):
        """Overlay spatial distributions from all models that have them."""
        ax.clear()
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for (name, res), color in zip(results.items(), colors):
            sd = res.spatial_distribution
            if sd is None:
                continue
            if hasattr(sd, "weight"):
                x_um = sd.x * 1e6
                y_um = sd.y * 1e6
                if x_um.size:
                    h, xe, ye = np.histogram2d(x_um, y_um, bins=80,
                                               weights=np.full(x_um.size, sd.weight))
                    ax.contour(xe[:-1], ye[:-1], h.T, levels=4, colors=[color], alpha=0.6)
            else:
                ax.contour(sd.x_centers * 1e6, sd.y_centers * 1e6,
                           sd.density.T, levels=4, colors=[color], alpha=0.6)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        ax.set_title("Spatial distribution")
