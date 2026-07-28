#!/usr/bin/env python3
"""
calculations.py
===============

Separate "Calculations" section for the Compton-GUIde that allows users to:
- Select which models to run via checkboxes
- Configure model-specific parameters per model
- Run calculations across multiple selected models
- View unified results:
  - 1D plots: Common figure with legend and different colors per model
  - 2D plots: List to select and view results from a particular model

This module is a tk.Frame subclass that can be embedded in the main app.
"""

from __future__ import annotations

import queue
import threading
import traceback
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, messagebox

from compton_suite.gui.model_api import ModelAdapter, CommonResults, validate_results
from compton_suite.gui.models import discover_models
from compton_suite.io.bunch import beam_from_shared_fields, sample_gaussian_bunch


# Colour scheme (matching main app)
BLUE = "#d6e4f5"
RED = "#f5d6d6"
YELLOW = "#f7f0c0"
GREY = "#e4e4e4"
GREEN = "#d6f5d6"


class CalculationsSection(tk.Frame):
    """A separate section for multi-model calculations with unified results display.
    
    This frame contains:
    - Model tabs with "Use for calculations" checkbox per model
    - Calculate button to run all selected models
    - Results display area with 1D plots (common figure) and 2D plot selection
    """
    
    def __init__(self, parent, fields: dict[str, tk.StringVar], **kwargs):
        super().__init__(parent, **kwargs)
        
        self.fields = fields
        self.models = discover_models()
        self.model_selections: dict[str, tk.BooleanVar] = {}
        self.model_frames: dict[str, tk.Frame] = {}
        self.model_params: dict[str, dict[str, tk.StringVar]] = {}
        self.results: dict[str, CommonResults] = {}
        
        # Threading
        self.worker: threading.Thread | None = None
        self.q: queue.Queue = queue.Queue()
        
        # Build the UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the main UI structure."""
        # Header
        header = tk.Frame(self, bg=YELLOW)
        header.pack(fill="x", padx=5, pady=(5, 2))
        tk.Label(header, text="MULTI-MODEL CALCULATIONS", bg=YELLOW, fg="#123",
                font=("TkDefaultFont", 14, "bold")).pack(side="left", padx=8)
        
        # Create a PanedWindow for left (model selection) and right (results)
        self._paned = tk.PanedWindow(self, orient="horizontal",
                                     sashwidth=6, sashrelief="groove")
        self._paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left pane: Model selection with tabs
        self._left_frame = tk.Frame(self._paned)
        self._build_model_selection_area()
        self._paned.add(self._left_frame, minsize=250, width=400)
        
        # Right pane: Results display
        self._right_frame = tk.Frame(self._paned)
        self._build_results_area()
        self._paned.add(self._right_frame, minsize=400)
    
    def _build_model_selection_area(self):
        """Build the model selection area with tabs."""
        # Model selection frame
        model_frame = tk.LabelFrame(self._left_frame, text="SELECT MODELS", bg=BLUE, fg="#123",
                                    font=("TkDefaultFont", 12, "bold"))
        model_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create notebook for model tabs
        self._notebook = ttk.Notebook(model_frame)
        self._notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create a tab for each model
        for name, adapter in self.models.items():
            caps = adapter.capabilities()
            if caps.is_fast_preview:
                continue  # Skip the analytical preview model
            
            self._create_model_tab(name, adapter, caps)
        
        # Calculate button at bottom
        btn_frame = tk.Frame(self._left_frame, bg=BLUE)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        self._calculate_btn = tk.Button(btn_frame, text="Calculate Selected Models",
                                        command=self._on_calculate, bg=GREEN,
                                        font=("TkDefaultFont", 12, "bold"))
        self._calculate_btn.pack(fill="x", padx=5, pady=5)
        
        # Status label
        self._status_lbl = tk.Label(btn_frame, text="Ready", bg=BLUE, fg="#234",
                                   font=("TkDefaultFont", 10, "italic"))
        self._status_lbl.pack(pady=2)
    
    def _create_model_tab(self, name: str, adapter: ModelAdapter, caps):
        """Create a tab for a specific model with checkbox and parameters."""
        # Create tab frame
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text=caps.display_name)
        
        # Checkbox frame at top
        check_frame = tk.Frame(tab, bg=GREY)
        check_frame.pack(fill="x", padx=5, pady=5)
        
        # "Use for calculations" checkbox
        var = tk.BooleanVar(value=False)
        self.model_selections[name] = var
        
        checkbox = tk.Checkbutton(check_frame, text="Use for calculations",
                                 variable=var, bg=GREY, fg="#123",
                                 font=("TkDefaultFont", 11, "bold"),
                                 command=lambda: self._toggle_model_params(name))
        checkbox.pack(side="left", padx=8)
        
        # Model parameters frame (initially hidden)
        params_frame = tk.Frame(tab, bg=GREY)
        self.model_frames[name] = params_frame
        
        # Store parameter StringVars
        self.model_params[name] = {}
        
        # Add model-specific parameters
        specs = adapter.extra_params()
        if specs:
            choices = adapter.extra_choices()
            for label_text, default, key in specs:
                # Create parameter entry row
                param_row = tk.Frame(params_frame, bg=GREY)
                param_row.pack(fill="x", padx=10, pady=3)
                
                # Label
                tk.Label(param_row, text=label_text, bg=GREY, anchor="w",
                        font=("TkDefaultFont", 10)).pack(side="left")
                
                # StringVar for this parameter
                var = tk.StringVar(value=str(default))
                self.model_params[name][key] = var
                
                # Input widget
                if key in choices:
                    # Dropdown for choices
                    combo = ttk.Combobox(param_row, textvariable=var,
                                        values=choices[key], width=12, state="readonly")
                    combo.pack(side="right", padx=5)
                else:
                    # Entry for numeric values
                    entry = tk.Entry(param_row, textvariable=var, width=12,
                                    justify="right", font=("TkDefaultFont", 10))
                    entry.pack(side="right", padx=5)
        else:
            # No parameters message
            tk.Label(params_frame, text="(no model-specific parameters)",
                    bg=GREY, fg="#567",
                    font=("TkDefaultFont", 9, "italic")).pack(padx=10, pady=10)
    
    def _toggle_model_params(self, name: str):
        """Show/hide model parameters based on checkbox state."""
        is_selected = self.model_selections[name].get()
        frame = self.model_frames[name]
        
        if is_selected:
            # Show parameters frame
            frame.pack(fill="both", expand=True, padx=5, pady=5)
        else:
            # Hide parameters frame
            frame.pack_forget()
    
    def _build_results_area(self):
        """Build the results display area."""
        # Results frame
        results_frame = tk.LabelFrame(self._right_frame, text="RESULTS", bg=YELLOW, fg="#432",
                                      font=("TkDefaultFont", 12, "bold"))
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create notebook for 1D and 2D plots
        self._results_notebook = ttk.Notebook(results_frame)
        self._results_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 1: 1D Plots (spectrum)
        self._tab_1d = ttk.Frame(self._results_notebook)
        self._results_notebook.add(self._tab_1d, text="1D Spectrum")
        
        # Tab 2: 2D Plots (angular distribution)
        self._tab_2d = ttk.Frame(self._results_notebook)
        self._results_notebook.add(self._tab_2d, text="2D Angular")
        
        # Tab 3: Summary
        self._tab_summary = ttk.Frame(self._results_notebook)
        self._results_notebook.add(self._tab_summary, text="Summary")
        
        # Setup 1D plot area
        self._setup_1d_plot_area()
        
        # Setup 2D plot area
        self._setup_2d_plot_area()
        
        # Setup summary area
        self._setup_summary_area()
    
    def _setup_1d_plot_area(self):
        """Setup the 1D plot area with common figure."""
        # Create figure for 1D plots
        self._fig_1d = plt.Figure(figsize=(10, 6))
        self._ax_spec = self._fig_1d.add_subplot(111)
        
        # Canvas
        self._canvas_1d = FigureCanvasTkAgg(self._fig_1d, master=self._tab_1d)
        self._canvas_1d.get_tk_widget().pack(fill="both", expand=True)
        
        # Initial placeholder
        self._ax_spec.text(0.5, 0.5, "Run calculations to see 1D results",
                          ha="center", va="center", fontsize=12)
        self._ax_spec.set_xlabel(r"photon energy $\hbar\omega_\gamma$ [keV]")
        self._ax_spec.set_ylabel("dN/dE [photons / keV]")
        self._ax_spec.set_title("Collimated photon spectra (all selected models)")
        self._fig_1d.tight_layout()
        self._canvas_1d.draw()
    
    def _setup_2d_plot_area(self):
        """Setup the 2D plot area with selection list."""
        # Split pane: list on left, plot on right
        paned = tk.PanedWindow(self._tab_2d, orient="horizontal",
                               sashwidth=4, sashrelief="groove")
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left: Model selection list
        list_frame = tk.Frame(paned, bg=GREY)
        paned.add(list_frame, minsize=120, width=150)
        
        tk.Label(list_frame, text="Select model:", bg=GREY, font=("TkDefaultFont", 10, "bold")).pack(pady=5)
        
        # Listbox for model selection
        self._model_listbox = tk.Listbox(list_frame, height=10, font=("TkDefaultFont", 10))
        self._model_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        self._model_listbox.bind("<<ListboxSelect>>", self._on_model_select_2d)
        
        # Right: 2D plot
        plot_frame = tk.Frame(paned)
        paned.add(plot_frame, minsize=300)
        
        # Create figure for 2D plots
        self._fig_2d = plt.Figure(figsize=(8, 6))
        self._ax_2d = self._fig_2d.add_subplot(111)
        
        # Canvas
        self._canvas_2d = FigureCanvasTkAgg(self._fig_2d, master=plot_frame)
        self._canvas_2d.get_tk_widget().pack(fill="both", expand=True)
        
        # Initial placeholder
        self._ax_2d.text(0.5, 0.5, "Run calculations to see 2D results",
                        ha="center", va="center", fontsize=12)
        self._ax_2d.set_xlabel(r"$\theta_x$ [mrad]")
        self._ax_2d.set_ylabel(r"$\theta_y$ [mrad]")
        self._ax_2d.set_title("Angular distribution")
        self._fig_2d.tight_layout()
        self._canvas_2d.draw()
    
    def _setup_summary_area(self):
        """Setup the summary area."""
        # Summary text widget
        self._summary_text = tk.Text(self._tab_summary, wrap="word", font=("TkFixedFont", 10))
        self._summary_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._summary_text.insert("1.0", "Run calculations to see summary...")
        self._summary_text.config(state="disabled")
    
    def _on_calculate(self):
        """Handle calculate button click."""
        if self.worker is not None and self.worker.is_alive():
            return
        
        # Get selected models
        selected = [name for name, var in self.model_selections.items() if var.get()]
        
        if not selected:
            messagebox.showwarning("No models selected",
                                 "Please select at least one model to calculate.")
            return
        
        # Update status
        self._status_lbl.config(text="Calculating...")
        self._calculate_btn.config(state="disabled")
        
        # Get common parameters from main fields
        common_params = {
            "fields": dict(self.fields),
            "selected_models": selected,
            "model_params": {name: dict(self.model_params[name]) for name in selected}
        }
        
        # Start worker thread
        def work():
            try:
                results = {}
                for name in selected:
                    adapter = self.models[name]
                    model_params = self.model_params[name]
                    
                    # Merge common fields with model-specific params
                    merged_fields = dict(self.fields)
                    for key, var in model_params.items():
                        merged_fields[key] = var.get()
                    
                    # Parse config
                    cfg, extra = adapter.params_to_config(merged_fields, quantum=False)
                    
                    # Sample electrons
                    beam = beam_from_shared_fields(
                        eps0=cfg.eps0, sigma_eps_rel=cfg.sigma_eps_rel,
                        emit_x=cfg.emit_x, emit_y=cfg.emit_y,
                        sigma0_x=cfg.sigma0_x, sigma0_y=cfg.sigma0_y,
                        sigma_par_e=cfg.sigma_par_e, N_e=cfg.N_e,
                    )
                    electrons = sample_gaussian_bunch(
                        beam, n_particles=int(extra["n_mc"]),
                        rng=np.random.default_rng(int(extra["seed"]))
                    )
                    
                    # Run model
                    res = adapter.run(cfg, n_mc=int(extra["n_mc"]),
                                     seed=extra["seed"], electrons=electrons)
                    
                    # Validate results
                    problems = validate_results(res)
                    if problems:
                        raise RuntimeError(
                            f"{adapter.capabilities().display_name} adapter returned "
                            f"malformed results: {'; '.join(problems)}")
                    
                    results[name] = res
                
                self.q.put(("ok", results))
            
            except Exception as e:
                self.q.put(("error", "".join(traceback.format_exception(e))))
        
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self.after(100, self._poll_queue)
    
    def _poll_queue(self):
        """Poll the queue for results."""
        try:
            status, payload = self.q.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_queue)
            return
        
        self._calculate_btn.config(state="normal")
        
        if status == "error":
            self._status_lbl.config(text="error")
            messagebox.showerror("Calculation failed", payload)
            return
        
        # Store results
        self.results = payload
        self._status_lbl.config(text="done")
        
        # Update all displays
        self._update_1d_plots()
        self._update_2d_plot_list()
        self._update_summary()
    
    def _update_1d_plots(self):
        """Update 1D plots with all selected model results."""
        self._ax_spec.clear()
        
        # Color cycle for different models
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for (name, res), color in zip(self.results.items(), colors):
            if res.spectrum is None:
                continue
            
            # Check if it's sampled or binned spectrum
            if hasattr(res.spectrum, "E_eV") and hasattr(res.spectrum, "dNdE_per_eV"):
                # Binned spectrum
                E_keV = res.spectrum.E_eV / 1e3
                dNdE_per_keV = res.spectrum.dNdE_per_eV * 1e3
                self._ax_spec.plot(E_keV, dNdE_per_keV, color=color,
                                  label=f"{name} (4π)")
            
            # Try to get collimated spectrum if available
            if hasattr(res, "spectrum_in_angular_range") and callable(res.spectrum_in_angular_range):
                # This would need adapter reference, skip for now
                pass
        
        self._ax_spec.set_xlabel(r"photon energy $\hbar\omega_\gamma$ [keV]")
        self._ax_spec.set_ylabel("dN/dE [photons / keV]")
        self._ax_spec.set_title("Collimated photon spectra (all selected models)")
        self._ax_spec.legend(fontsize=9)
        self._ax_spec.grid(True, alpha=0.3)
        self._fig_1d.tight_layout()
        self._canvas_1d.draw()
    
    def _update_2d_plot_list(self):
        """Update the 2D plot model selection list."""
        self._model_listbox.delete(0, tk.END)
        
        for name, res in self.results.items():
            # Check if model has angular distribution data
            has_2d = False
            if res.angular_spectrum is not None:
                has_2d = True
            elif hasattr(res, "photon_samples") and res.photon_samples is not None:
                if hasattr(res.photon_samples, "ph_thx_lab"):
                    has_2d = True
            
            if has_2d:
                self._model_listbox.insert(tk.END, name)
    
    def _on_model_select_2d(self, event):
        """Handle model selection in 2D plot list."""
        selection = self._model_listbox.curselection()
        if not selection:
            return
        
        name = self._model_listbox.get(selection[0])
        res = self.results.get(name)
        
        if res is None:
            return
        
        self._render_2d_plot(name, res)
    
    def _render_2d_plot(self, name: str, res: CommonResults):
        """Render 2D angular distribution for a specific model."""
        self._ax_2d.clear()
        
        # Check for angular spectrum data
        if res.angular_spectrum is not None:
            ang = res.angular_spectrum
            # Integrate over energy to get angle-only density
            dE = np.gradient(ang.E_eV)
            d2N_dOmega = np.einsum("ijk,k->ij", ang.d2NdEdOmega, dE)
            self._ax_2d.pcolormesh(ang.theta_x * 1e3, ang.theta_y * 1e3,
                                  d2N_dOmega.T, shading="auto")
        
        # Check for sampled photon data
        elif hasattr(res, "photon_samples") and res.photon_samples is not None:
            photon_samples = res.photon_samples
            thx = getattr(photon_samples, "ph_thx_lab", None)
            thy = getattr(photon_samples, "ph_thy_lab", None)
            weight = getattr(photon_samples, "weight", 1.0)
            
            if thx is not None and thy is not None and thx.size > 0:
                thx_mrad = thx * 1e3
                thy_mrad = thy * 1e3
                h, xedges, yedges = np.histogram2d(
                    thx_mrad, thy_mrad, bins=80,
                    weights=np.full(thx_mrad.size, weight))
                self._ax_2d.pcolormesh(xedges, yedges, h.T, shading="auto")
        
        self._ax_2d.set_xlabel(r"$\theta_x$ [mrad]")
        self._ax_2d.set_ylabel(r"$\theta_y$ [mrad]")
        self._ax_2d.set_title(f"Angular distribution - {name}")
        self._fig_2d.tight_layout()
        self._canvas_2d.draw()
    
    def _update_summary(self):
        """Update the summary text with results from all models."""
        self._summary_text.config(state="normal")
        self._summary_text.delete("1.0", tk.END)
        
        self._summary_text.insert("1.0", "MULTI-MODEL CALCULATION RESULTS\n")
        self._summary_text.insert("1.0", "=" * 50 + "\n\n")
        
        for name, res in self.results.items():
            self._summary_text.insert("1.0", f"MODEL: {name.upper()}\n")
            self._summary_text.insert("1.0", "-" * 30 + "\n")
            
            # Basic stats
            self._summary_text.insert("1.0", f"  Total yield: {res.total_yield:.4e}\n")
            self._summary_text.insert("1.0", f"  Mean energy: {res.mean_energy_eV:.4e} eV\n")
            self._summary_text.insert("1.0", f"  Energy spread: {res.energy_spread_eV:.4e} eV\n")
            self._summary_text.insert("1.0", f"  Divergence: {res.divergence_rad:.4e} rad\n")
            
            # Photon multiplicity
            if res.photon_multiplicity is not None:
                pm = res.photon_multiplicity
                self._summary_text.insert("1.0", f"  Mean n_phot: {pm.mean_n_phot:.4e}\n")
                self._summary_text.insert("1.0", f"  P(0): {pm.frac_n0*100:.2f}%\n")
                self._summary_text.insert("1.0", f"  P(1): {pm.frac_n1*100:.2f}%\n")
                self._summary_text.insert("1.0", f"  P(2): {pm.frac_n2*100:.2f}%\n")
            
            self._summary_text.insert("1.0", "\n\n")
        
        self._summary_text.config(state="disabled")
    
    def get_selected_models(self) -> list[str]:
        """Return list of selected model names."""
        return [name for name, var in self.model_selections.items() if var.get()]


def create_calculations_section(parent, fields: dict[str, tk.StringVar]) -> CalculationsSection:
    """Factory function to create and return a CalculationsSection."""
    return CalculationsSection(parent, fields)
