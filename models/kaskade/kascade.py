#!/usr/bin/env python3
r"""
kascade.py
===================

Generalisation of ``dfe4_compton_mc.py`` (sequential multi-photon inverse-Compton
Monte-Carlo) in two directions:

1. **Arbitrary collision (crossing) angle** between the electron-beam direction
   (+z) and the laser-pulse propagation direction.  ``crossing_angle`` is the
   angle by which the laser is tilted away from exact head-on
   counter-propagation (``crossing_angle = 0`` reproduces dfe4's head-on case).
   The laser propagates along ``n_L = (sin phi, 0, -cos phi)`` (tilted in the
   x-z plane).  This enters through

     * the 3-D overlap integral (a full 3-D Gaussian pulse propagating along
       ``n_L``, evaluated along the electron trajectory);
     * the Moeller flux factor ``c (1 - beta cos alpha) = c (1 + beta cos phi)``
       in the emission rate (dfe4's ``2c`` = the head-on value);
     * the Doppler up-shift factor ``D = (1 - beta cos alpha)/(1 - beta)``
       (dfe4's ``4 gamma^2`` = the head-on value) in the scattered-photon
       energy.

   The scattered radiation stays beamed within ~1/gamma of the *electron*
   direction regardless of the collision angle (correct for relativistic
   electrons), so the emission-cone geometry is unchanged; the crossing angle
   only rescales the energy (via ``D``) and the yield (via the flux factor and
   the overlap geometry).

2. **Optional quantum branch** (``quantum = true``) replacing the classical
   Thomson treatment by:

     * **Electron recoil** in the photon-energy formula: the denominator gains
       the recoil parameter ``X = D eps_L/eps`` (``= 2k`` with
       ``k = gamma eps_L (1 - beta cos alpha)`` the rest-frame photon energy in
       units of ``m c^2``), so the Compton edge is red-shifted to
       ``D eps_L/(1+X)`` instead of the Thomson ``D eps_L``.  The parent
       electron loses exactly the emitted photon energy (as already in dfe4),
       which is now the proper single-photon quantum recoil.
     * **Klein--Nishina cross section** instead of Thomson: (a) the total cross
       section is scaled by ``sigma_KN(k)/sigma_T`` (reducing the yield at
       larger ``k``), and (b) the per-photon emission is reweighted by the
       Klein--Nishina/Thomson differential ratio ``R_KN`` (which softens/hardens
       the spectrum consistently with recoil).

   With ``quantum = false`` the code is classical (Thomson) and, at
   ``crossing_angle = 0``, reduces to dfe4 to the level of the ``beta -> 1``
   approximation (``D = (1+beta)/(1-beta) ~ 4 gamma^2``).

Everything else (initial-bunch sampling, the sequential emission chain, the
``a_0^2/2`` non-linear pond. red-shift, the on-axis collimated spectrum) is
carried over unchanged from dfe4.

A pure physics library: ``Config``/``run_simulation``/``Results`` and the
``.ele`` I/O helpers below, no CLI, no plotting, no config-file loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from compton_io import constants as _io_constants
from compton_io.bunch import MacroBunch as _MacroBunch
from compton_io.io_formats.sdds import save_elegant_ele as _save_elegant_ele
from compton_io.laser import GaussianParaxialLaser as _GaussianParaxialLaser

# ---------------------------------------------------------------------------
# Physical constants (SI) -- from compton_io.constants (shared, pint-
# derived source of truth also used by compton_guide/xigma_i) rather than
# hand-typed literals. Zero numeric change here: this module's previous
# literals already agreed with compton_io's values to their quoted
# precision (unlike xigma_i's older-CODATA-vintage hbar/electron mass,
# which did need an actual, deliberate numeric update).
# ---------------------------------------------------------------------------
C_LIGHT = _io_constants.C_LIGHT           # speed of light            [m/s]
E_CHARGE = _io_constants.E_CHARGE         # elementary charge         [C]
HBAR = _io_constants.HBAR                 # reduced Planck constant   [J s]
EPS0 = _io_constants.EPS0                 # vacuum permittivity       [F/m]
SIGMA_T = _io_constants.SIGMA_T_M2        # Thomson cross section     [m^2]
MEC2_EV = _io_constants.MEC2_EV           # electron rest energy      [eV]
MEC2_J = MEC2_EV * E_CHARGE               # electron rest energy      [J]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Physical parameters for the bunch, the laser and the interaction."""

    # ---- electron bunch -------------------------------------------------
    # defaults mirror dfe4's current working point (NCPM ~2 GeV regime)
    eps0: float = 1000.0
    sigma_eps_rel: float = 0.0001
    emit_x: float = 1.0e-14
    emit_y: float = 1.0e-14
    sigma0_x: float = 10.0e-6
    sigma0_y: float = 10.0e-6
    sigma_par_e: float = 50.0e-6
    N_e: float = 1.0e9

    # ---- laser pulse ----------------------------------------------------
    lambda_L: float = 1.0e-6
    sigma0_l: float = 20.0e-6
    R_sf: float = 0.0
    sigma_par_L: float = 1.0e-3
    pulse_energy_J: float = 28e-3
    N_L: float = 0.0

    # ---- collision geometry (centres at maximum overlap, t = 0) ---------
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0
    crossing_angle: float = 0.0   # laser tilt from head-on [rad]; 0 = head-on
    #                               laser propagates along (sin phi, 0, -cos phi)

    # ---- quantum options ------------------------------------------------
    quantum: bool = False         # True -> Klein-Nishina cross section + recoil
    #                               False -> classical Thomson (dfe4)

    # ---- output angular window (goal 3 of dfe4.lyx) ----------------------
    Theta_x: float = 0.0
    Theta_y: float = 0.0

    # ---- on-axis collimation --------------------------------------------
    theta_onaxis_gamma: float = 0.2   # keep photons with gamma*theta_lab < this

    # ---- numerics -------------------------------------------------------
    n_time: int = 201
    n_sigma_time: float = 6.0
    max_photons: int = 30
    chunk: int = 5_000

    # ---- derived (filled in __post_init__) ------------------------------
    omega_L: float = field(init=False)
    hbar_omega_L_J: float = field(init=False)
    eps_L: float = field(init=False)
    beta_x: float = field(init=False)
    beta_y: float = field(init=False)

    def __post_init__(self) -> None:
        self.omega_L = 2.0 * np.pi * C_LIGHT / self.lambda_L
        self.hbar_omega_L_J = HBAR * self.omega_L
        self.eps_L = self.hbar_omega_L_J / MEC2_J
        if self.N_L <= 0.0 or self.R_sf <= 0.0:
            # N_L (pulse photon count) and R_sf (Rayleigh-range-like scale)
            # are exactly compton_io.laser.GaussianParaxialLaser's
            # n_photons/rayleigh_x_m -- read from there instead of
            # re-deriving the same formulas locally (verified to agree to
            # float precision). laser_density/laser_a0sq below stay local:
            # they evaluate the envelope at an arbitrary (x, y, z, t),
            # including a nonzero crossing_angle, which GaussianParaxialLaser
            # (head-on/on-axis only, see its own module docstring) doesn't
            # support.
            _laser = _GaussianParaxialLaser(
                pulse_energy_J=self.pulse_energy_J, wavelength_m=self.lambda_L,
                waist_rms_x_m=self.sigma0_l, waist_rms_y_m=self.sigma0_l,
                duration_rms_s=self.sigma_par_L / C_LIGHT,
            )
            if self.N_L <= 0.0:
                self.N_L = _laser.n_photons
            if self.R_sf <= 0.0:
                self.R_sf = _laser.rayleigh_x_m
        self.beta_x = self.sigma0_x ** 2 / self.emit_x
        self.beta_y = self.sigma0_y ** 2 / self.emit_y

    @property
    def sigma_eps(self) -> float:
        return self.sigma_eps_rel * self.eps0


# ---------------------------------------------------------------------------
# Collision geometry and relativistic / quantum helpers
# ---------------------------------------------------------------------------
def beta_of(eps):
    """v/c from the normalised energy eps (= Lorentz gamma), clipped physical."""
    e = np.clip(np.asarray(eps, dtype=float), 1.0 + 1e-12, None)
    return np.sqrt(np.clip(1.0 - 1.0 / e ** 2, 0.0, 1.0))


def laser_axis(cfg: Config) -> Tuple[float, float, float]:
    """Laser propagation unit vector n_L = (sin phi, 0, -cos phi).

    phi = crossing_angle; phi = 0 gives head-on counter-propagation (-z).
    """
    phi = cfg.crossing_angle
    return (np.sin(phi), 0.0, -np.cos(phi))


def cos_collision(cfg: Config) -> float:
    """cos(angle between +z electron direction and laser propagation n_L).

    = n_L . zhat = -cos(phi).  Head-on (phi=0) -> -1.
    """
    return -np.cos(cfg.crossing_angle)


def doppler_D(eps, cfg: Config):
    """Doppler up-shift factor D = (1 - beta cos alpha)/(1 - beta).

    On-axis Thomson scattered energy is ``eps_gamma = D eps_L`` (in m c^2
    units).  Head-on: D = (1+beta)/(1-beta) = gamma^2 (1+beta)^2 ~ 4 gamma^2.
    """
    b = beta_of(eps)
    ca = cos_collision(cfg)
    return (1.0 - b * ca) / (1.0 - b)


def recoil_X(eps, cfg: Config):
    """Recoil parameter X = D eps_L/eps = 2k (k = rest-frame photon energy).

    The recoil-shifted Compton edge is ``D eps_L/(1+X)``.  Head-on: X = 4 eps eps_L.
    """
    return doppler_D(eps, cfg) * cfg.eps_L / np.clip(np.asarray(eps, float), 1e-30, None)


def kn_sigma_ratio(k):
    """Total Klein--Nishina cross section relative to Thomson, sigma_KN(k)/sigma_T.

    k = rest-frame photon energy in units of m c^2.  -> 1 as k -> 0.
    """
    k = np.asarray(k, dtype=float)
    out = np.empty_like(k)
    small = k < 1e-3
    ks = np.where(small, 1e-3, k)  # avoid 0/0 in the closed form
    term = ((1.0 + ks) / ks ** 3 * (2.0 * ks * (1.0 + ks) / (1.0 + 2.0 * ks)
                                    - np.log1p(2.0 * ks))
            + np.log1p(2.0 * ks) / (2.0 * ks)
            - (1.0 + 3.0 * ks) / (1.0 + 2.0 * ks) ** 2)
    out = 0.75 * term
    # small-k series: 1 - 2k + 26 k^2/5 - ...
    out = np.where(small, 1.0 - 2.0 * k + 5.2 * k ** 2, out)
    return out


def kn_over_thomson(u2, X):
    """Klein--Nishina / Thomson differential-cross-section ratio R_KN in [0,1].

    ``u2 = (gamma theta)^2`` is the (squared) lab emission angle relative to the
    electron, ``X`` the recoil parameter.  Derived via the rest-frame Compton
    kinematics: the lab emission angle maps to rest-frame scattering angle
    ``theta'`` with ``cos theta' = 1 - (1/P - 1)/k``, ``P = (1+u2)/(1+u2+X)``
    the scattered/incident rest-frame energy ratio and ``k = X/2``.  Then
    ``R_KN = P^2 (P + 1/P - sin^2 theta')/(1 + cos^2 theta')``.  -> 1 as X -> 0.
    """
    P = (1.0 + u2) / (1.0 + u2 + X)
    k = 0.5 * X
    inv = (1.0 / P - 1.0) / np.clip(k, 1e-30, None)   # = 1 - cos theta'
    ct = 1.0 - inv                                     # cos theta'
    sin2 = np.clip(1.0 - ct * ct, 0.0, 1.0)
    R = P ** 2 * (P + 1.0 / P - sin2) / (1.0 + ct * ct)
    return np.clip(R, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Laser density (3-D Gaussian pulse propagating along n_L) and a_0^2
# ---------------------------------------------------------------------------
def laser_density(x, y, z, t, cfg: Config):
    """Gaussian laser photon density propagating along n_L, focus at delta.

    Reduces exactly to dfe4's counter-propagating pulse when crossing_angle = 0
    and delta = 0.  The waist is at the focus point ``delta`` and the pulse
    centre passes the focus at ``t = 0``.
    """
    nx, ny, nz = laser_axis(cfg)
    rx = x - cfg.delta_x
    ry = y - cfg.delta_y
    rz = z - cfg.delta_z
    u = rx * nx + ry * ny + rz * nz               # along propagation from focus
    perp2 = np.clip(rx * rx + ry * ry + rz * rz - u * u, 0.0, None)
    sp2 = cfg.sigma0_l ** 2 * (1.0 + (u / cfg.R_sf) ** 2)
    norm = 1.0 / ((2.0 * np.pi) ** 1.5 * sp2 * cfg.sigma_par_L)
    arg = -perp2 / (2.0 * sp2) - (u - C_LIGHT * t) ** 2 / (2.0 * cfg.sigma_par_L ** 2)
    return norm * np.exp(arg)


def laser_a0sq(x, y, z, t, cfg: Config):
    """Normalised laser vector-potential squared a_0^2(r,t) (as in dfe4)."""
    k_a0 = (2.0 * E_CHARGE ** 2 * HBAR * C_LIGHT ** 2 * cfg.N_L
            / (EPS0 * MEC2_J ** 2 * cfg.omega_L))
    return k_a0 * laser_density(x, y, z, t, cfg)


# ---------------------------------------------------------------------------
# Compton angular kernel L_theta and emission-angle sampler
# ---------------------------------------------------------------------------
def L_theta(thx, thy, eps):
    """Normalised (linearly-polarised) Thomson angular distribution."""
    u2 = eps ** 2 * (thx ** 2 + thy ** 2)
    ux2 = eps ** 2 * thx ** 2
    return (3.0 / (2.0 * np.pi)) * eps ** 2 / (1.0 + u2) ** 2 * (
        1.0 - 4.0 * ux2 / (1.0 + u2) ** 2
    )


def sample_emission_angle(eps: np.ndarray, rng: np.random.Generator, X=None):
    """Sample (thx, thy) from L_theta (classical) or L_theta * R_KN (quantum).

    Proposal q(theta) ~ 1/(1+eps^2 theta^2)^2 (invertible radial CDF), accepted
    with probability ``bracket`` (the L_theta polarisation factor) and, if
    ``X`` is given, additionally with the Klein--Nishina/Thomson weight
    ``R_KN`` -- so the returned angles follow the Klein--Nishina angular shape.
    Exactly one angle is produced per electron (count set elsewhere by the
    optical depth), so the KN normalisation does not bias the photon number.
    """
    n = eps.shape[0]
    thx = np.empty(n)
    thy = np.empty(n)
    remaining = np.arange(n)
    quantum = X is not None
    while remaining.size:
        m = remaining.size
        e = eps[remaining]
        u = rng.random(m)
        r = np.sqrt(u / (1.0 - u)) / e
        phi = rng.random(m) * (2.0 * np.pi)
        tx = r * np.cos(phi)
        ty = r * np.sin(phi)
        u2 = (e * r) ** 2
        weight = 1.0 - 4.0 * (e * tx) ** 2 / (1.0 + u2) ** 2   # L_theta bracket
        if quantum:
            weight = weight * kn_over_thomson(u2, X[remaining])
        accept = rng.random(m) < weight
        idx = remaining[accept]
        thx[idx] = tx[accept]
        thy[idx] = ty[accept]
        remaining = remaining[~accept]
    return thx, thy


# ---------------------------------------------------------------------------
# Cumulative optical depth along the electron trajectory (general geometry)
# ---------------------------------------------------------------------------
def build_lambda_grid(x_w, thx, y_w, thy, z0, cfg: Config):
    """Cumulative optical depth Lambda(tau) per electron, general collision angle.

    The interaction window is centred at the per-electron overlap time and the
    integrand uses the Moeller flux factor ``c (1 - beta0 cos alpha)`` and (in
    quantum mode) the Klein--Nishina total-cross-section scaling.
    """
    nx, ny, nz = laser_axis(cfg)

    # per-electron overlap time t* : (r_e(t)-delta).n_L = c t  (envelope centre)
    m = thx * nx + thy * ny + nz                       # d/dt of u along axis, /c
    one_minus_m = 1.0 - m                              # ~ (1 + cos phi)
    A = ((x_w + z0 * thx - cfg.delta_x) * nx
         + (y_w + z0 * thy - cfg.delta_y) * ny
         + (z0 - cfg.delta_z) * nz)
    tstar = A / (C_LIGHT * one_minus_m)

    # common relative-time grid; longitudinal envelope width in t
    sigma_t = cfg.sigma_par_L / (C_LIGHT * (1.0 - nz))   # 1 - nz = 1 + cos phi
    tau = np.linspace(-cfg.n_sigma_time * sigma_t,
                      cfg.n_sigma_time * sigma_t, cfg.n_time)
    dt = tau[1] - tau[0]

    t = tstar[:, None] + tau[None, :]
    z = z0[:, None] + C_LIGHT * t
    x = x_w[:, None] + z * thx[:, None]
    y = y_w[:, None] + z * thy[:, None]

    beta0 = float(beta_of(cfg.eps0))
    flux = 1.0 - beta0 * cos_collision(cfg)             # = 1 + beta0 cos phi
    sigma_scale = 1.0
    if cfg.quantum:
        k0 = 0.5 * float(recoil_X(cfg.eps0, cfg))       # rest-frame photon energy
        sigma_scale = float(kn_sigma_ratio(k0))

    w_T = (cfg.N_L * SIGMA_T * sigma_scale * C_LIGHT * flux
           * laser_density(x, y, z, t, cfg))
    trap = 0.5 * (w_T[:, 1:] + w_T[:, :-1]) * dt
    Lambda = np.concatenate([np.zeros((x_w.shape[0], 1)), np.cumsum(trap, axis=1)], axis=1)
    return tau, tstar, Lambda


def invert_lambda(Lambda: np.ndarray, tau: np.ndarray, threshold: np.ndarray):
    """Solve Lambda(tau_emit) = threshold row-by-row (Lambda monotonic in tau)."""
    total = Lambda[:, -1]
    valid = threshold <= total
    thr_c = np.clip(threshold, 0.0, total)
    idx = np.sum(Lambda <= thr_c[:, None], axis=1) - 1
    idx = np.clip(idx, 0, Lambda.shape[1] - 2)
    lo = np.take_along_axis(Lambda, idx[:, None], axis=1)[:, 0]
    hi = np.take_along_axis(Lambda, (idx + 1)[:, None], axis=1)[:, 0]
    tlo = tau[idx]
    thi = tau[idx + 1]
    frac = np.where(hi > lo, (thr_c - lo) / np.maximum(hi - lo, 1e-300), 0.0)
    tau_emit = tlo + frac * (thi - tlo)
    return tau_emit, valid


# ---------------------------------------------------------------------------
# Sequential multi-photon emission chain (general geometry + optional quantum)
# ---------------------------------------------------------------------------
def run_multiphoton_chain(x_w, thx, y_w, thy, z0, eps, cfg: Config,
                           rng: np.random.Generator):
    """Simulate the emission chain for a chunk of electrons.

    Records, per photon, the emission angle relative to the electron
    (``thx``/``thy``) and the lab-frame photon propagation angle
    (``thx_lab``/``thy_lab`` = electron angle at emission + emission angle).
    """
    n = x_w.shape[0]
    tau, tstar, Lambda = build_lambda_grid(x_w, thx, y_w, thy, z0, cfg)

    eps_cur = eps.copy()
    thx_cur = thx.copy()
    thy_cur = thy.copy()
    Lambda_prev = np.zeros(n)
    active = np.ones(n, dtype=bool)

    parents: List[np.ndarray] = []
    gens: List[np.ndarray] = []
    energies_eV: List[np.ndarray] = []
    ph_thx: List[np.ndarray] = []
    ph_thy: List[np.ndarray] = []
    ph_thx_lab: List[np.ndarray] = []
    ph_thy_lab: List[np.ndarray] = []
    t_emits: List[np.ndarray] = []
    x_emits: List[np.ndarray] = []
    y_emits: List[np.ndarray] = []

    for gen in range(1, cfg.max_photons + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        # step 2: emission time from p = 1 - exp[-tau(t_{i-1}, t_i)]
        p = rng.random(idx.size)
        threshold = Lambda_prev[idx] - np.log1p(-p)
        tau_emit, valid = invert_lambda(Lambda[idx], tau, threshold)

        active[idx[~valid]] = False
        emit_idx = idx[valid]
        if emit_idx.size == 0:
            continue

        t_emit = tstar[emit_idx] + tau_emit[valid]
        z_e = z0[emit_idx] + C_LIGHT * t_emit
        x_e = x_w[emit_idx] + z_e * thx[emit_idx]
        y_e = y_w[emit_idx] + z_e * thy[emit_idx]
        a0sq = laser_a0sq(x_e, y_e, z_e, t_emit, cfg)

        # Doppler factor and recoil parameter at the electron's current energy
        eps_e = eps_cur[emit_idx]
        Dfac = doppler_D(eps_e, cfg)
        Xrec = Dfac * cfg.eps_L / eps_e

        # step 3: photon angle from L_theta (x Klein-Nishina weight if quantum)
        if cfg.quantum:
            thxg, thyg = sample_emission_angle(eps_e, rng, X=Xrec)
        else:
            thxg, thyg = sample_emission_angle(eps_e, rng)
        theta2 = thxg ** 2 + thyg ** 2

        # lab-frame photon direction = electron angle before recoil + emission angle
        thx_lab = thx_cur[emit_idx] + thxg
        thy_lab = thy_cur[emit_idx] + thyg

        # step 3/4: photon energy (Doppler D, non-linear a0^2/2, quantum recoil X)
        X_on = Xrec if cfg.quantum else 0.0
        k_n = Dfac * cfg.eps_L / (1.0 + eps_e ** 2 * theta2 + 0.5 * a0sq + X_on)
        eps_next = np.clip(eps_e - k_n, 1e-6, None)

        # step 4: recoil kick to the electron angle
        thx_new = thx_cur[emit_idx] - (k_n / eps_next) * np.sin(thxg)
        thy_new = thy_cur[emit_idx] - (k_n / eps_next) * np.sin(thyg)

        eps_cur[emit_idx] = eps_next
        thx_cur[emit_idx] = thx_new
        thy_cur[emit_idx] = thy_new
        Lambda_prev[emit_idx] = threshold[valid]

        parents.append(emit_idx.copy())
        gens.append(np.full(emit_idx.size, gen, dtype=int))
        energies_eV.append(k_n * MEC2_EV)
        ph_thx.append(thxg)
        ph_thy.append(thyg)
        ph_thx_lab.append(thx_lab)
        ph_thy_lab.append(thy_lab)
        t_emits.append(t_emit)
        x_emits.append(x_e)
        y_emits.append(y_e)

    truncated = int(active.sum())

    photons = dict(
        parent=np.concatenate(parents) if parents else np.empty(0, dtype=int),
        gen=np.concatenate(gens) if gens else np.empty(0, dtype=int),
        E_eV=np.concatenate(energies_eV) if energies_eV else np.empty(0),
        thx=np.concatenate(ph_thx) if ph_thx else np.empty(0),
        thy=np.concatenate(ph_thy) if ph_thy else np.empty(0),
        thx_lab=np.concatenate(ph_thx_lab) if ph_thx_lab else np.empty(0),
        thy_lab=np.concatenate(ph_thy_lab) if ph_thy_lab else np.empty(0),
        t=np.concatenate(t_emits) if t_emits else np.empty(0),
        x=np.concatenate(x_emits) if x_emits else np.empty(0),
        y=np.concatenate(y_emits) if y_emits else np.empty(0),
    )
    # Per-electron last emission time: the time of the most recent photon
    # emitted by that electron.  Electrons that emitted zero photons
    # (n=0) keep the value 0 -- they "stay at the focus" by convention.
    n_e = x_w.shape[0]
    t_last_emit = np.zeros(n_e)
    if photons["parent"].size:
        np.maximum.at(t_last_emit, photons["parent"], photons["t"])
    return eps_cur, thx_cur, thy_cur, photons, truncated, Lambda[:, -1], t_last_emit


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------
@dataclass
class Results:
    cfg: Config
    n_mc: int
    weight: float
    eps_i: np.ndarray
    thx_i: np.ndarray
    thy_i: np.ndarray
    z0: np.ndarray
    lambda_total: np.ndarray
    n_phot: np.ndarray
    eps_f: np.ndarray
    thx_f: np.ndarray
    thy_f: np.ndarray
    # final bunch geometry at t = t_last_emit (the moment of the most recent
    # photon emission per electron; equals z0/t0 for n=0 electrons)
    z_f: np.ndarray
    x_f: np.ndarray
    y_f: np.ndarray
    t_last_emit: np.ndarray
    ph_E_eV: np.ndarray
    ph_thx: np.ndarray                 # emission angle relative to electron
    ph_thy: np.ndarray
    ph_thx_lab: np.ndarray            # lab-frame photon angle
    ph_thy_lab: np.ndarray
    ph_t: np.ndarray                   # per-photon emission time [s]
    ph_x: np.ndarray                   # per-photon transverse position at emission [m]
    ph_y: np.ndarray
    ph_gen: np.ndarray
    ph_parent: np.ndarray
    onaxis_mask: np.ndarray
    truncated: int
    summary: Dict[str, float]


# ---------------------------------------------------------------------------
# Main Monte-Carlo driver
# ---------------------------------------------------------------------------
def run_simulation(cfg: Config, n_mc: int = 20_000, seed: int = 0,
                  *, electrons: Dict[str, np.ndarray]) -> Results:
    """Run the multi-photon inverse-Compton Monte-Carlo.

    Parameters
    ----------
    cfg : Config
        Laser, geometry, quantum and bunch-magnitude parameters.  The
        per-particle spread/divergence/position values come entirely from
        ``electrons``; the remaining magnitude-style fields (e.g. ``N_e``)
        are still used for weight/yield normalisations.
    n_mc : int
        Nominal macro-electron count -- overridden by the length of
        ``electrons``'s arrays if they differ (see below).
    seed : int
        Seed for the random number generator (used for the emission chain
        itself -- photon sampling, not electron sampling).
    electrons : dict
        Pre-sampled initial bunch with keys ``eps``, ``z0``, ``x_w``,
        ``y_w``, ``thx``, ``thy`` (per-particle arrays, all in SI units
        apart from ``eps`` which is Lorentz gamma). Required: electron
        sampling is the caller's job, not this engine's -- build a
        ``compton_io.bunch.GaussianElectronBeam`` and draw one via
        ``compton_io.bunch.sample_gaussian_bunch``, then convert the
        resulting ``MacroBunch`` to this dict shape (see
        ``compton_guide.adapters.kascade_adapter._macrobunch_to_kascade_electrons``
        or ``validation/runners.py`` for the conversion). Passing
        ``electrons=None`` raises a plain ``TypeError`` below (indexing
        ``None``), not a silent internal resample: there is exactly one
        place electrons get drawn from a beam description, not one per
        model.
    """
    rng = np.random.default_rng(seed)
    weight = cfg.N_e / n_mc

    # Honour the size of the supplied bunch; clamp n_mc to its length.
    n_loaded = int(np.asarray(electrons["eps"]).shape[0])
    if n_loaded <= 0:
        raise ValueError("run_simulation: 'electrons' must be non-empty")
    if n_loaded != n_mc:
        n_mc = n_loaded
        weight = cfg.N_e / n_mc
    el = dict(
        z0=np.asarray(electrons["z0"], dtype=float),
        eps=np.asarray(electrons["eps"], dtype=float),
        thx=np.asarray(electrons["thx"], dtype=float),
        thy=np.asarray(electrons["thy"], dtype=float),
        x_w=np.asarray(electrons["x_w"], dtype=float),
        y_w=np.asarray(electrons["y_w"], dtype=float),
    )

    eps_f = np.empty(n_mc)
    thx_f = np.empty(n_mc)
    thy_f = np.empty(n_mc)
    lambda_total = np.empty(n_mc)
    n_phot = np.zeros(n_mc, dtype=int)
    t_last_emit_total = np.zeros(n_mc)
    truncated_total = 0

    parents: List[np.ndarray] = []
    gens: List[np.ndarray] = []
    energies_eV: List[np.ndarray] = []
    ph_thx: List[np.ndarray] = []
    ph_thy: List[np.ndarray] = []
    ph_thx_lab: List[np.ndarray] = []
    ph_thy_lab: List[np.ndarray] = []
    ph_t: List[np.ndarray] = []
    ph_x: List[np.ndarray] = []
    ph_y: List[np.ndarray] = []

    for s in range(0, n_mc, cfg.chunk):
        e = slice(s, min(s + cfg.chunk, n_mc))
        eps_c, thx_c, thy_c, photons, trunc, lam, tle = run_multiphoton_chain(
            el["x_w"][e], el["thx"][e], el["y_w"][e], el["thy"][e], el["z0"][e],
            el["eps"][e], cfg, rng)
        eps_f[e] = eps_c
        thx_f[e] = thx_c
        thy_f[e] = thy_c
        lambda_total[e] = lam
        t_last_emit_total[e] = tle
        truncated_total += trunc
        n_phot[e] = np.bincount(photons["parent"], minlength=eps_c.shape[0]) \
            if photons["parent"].size else 0

        if photons["parent"].size:
            parents.append(photons["parent"] + s)
            gens.append(photons["gen"])
            energies_eV.append(photons["E_eV"])
            ph_thx.append(photons["thx"])
            ph_thy.append(photons["thy"])
            ph_thx_lab.append(photons["thx_lab"])
            ph_thy_lab.append(photons["thy_lab"])
            ph_t.append(photons["t"])
            ph_x.append(photons["x"])
            ph_y.append(photons["y"])

    ph_parent = np.concatenate(parents) if parents else np.empty(0, dtype=int)
    ph_gen = np.concatenate(gens) if gens else np.empty(0, dtype=int)
    ph_E_eV = np.concatenate(energies_eV) if energies_eV else np.empty(0)
    ph_thx_a = np.concatenate(ph_thx) if ph_thx else np.empty(0)
    ph_thy_a = np.concatenate(ph_thy) if ph_thy else np.empty(0)
    ph_thx_lab_a = np.concatenate(ph_thx_lab) if ph_thx_lab else np.empty(0)
    ph_thy_lab_a = np.concatenate(ph_thy_lab) if ph_thy_lab else np.empty(0)
    ph_t_a = np.concatenate(ph_t) if ph_t else np.empty(0)
    ph_x_a = np.concatenate(ph_x) if ph_x else np.empty(0)
    ph_y_a = np.concatenate(ph_y) if ph_y else np.empty(0)
    total_photons = ph_E_eV.size

    # on-axis collimation (lab-frame angle)
    gamma_theta_lab = cfg.eps0 * np.hypot(ph_thx_lab_a, ph_thy_lab_a)
    onaxis_mask = gamma_theta_lab < cfg.theta_onaxis_gamma

    # angular window filter (goal 3)
    if cfg.Theta_x > 0.0 or cfg.Theta_y > 0.0:
        tx_lim = cfg.Theta_x if cfg.Theta_x > 0.0 else np.inf
        ty_lim = cfg.Theta_y if cfg.Theta_y > 0.0 else np.inf
        in_window = (np.abs(ph_thx_lab_a) <= tx_lim) & (np.abs(ph_thy_lab_a) <= ty_lim)
    else:
        in_window = np.ones(total_photons, dtype=bool)

    # ---- summary quantities --------------------------------------------
    mean_lambda = float(lambda_total.mean())
    N_gamma_total = cfg.N_e * mean_lambda
    eloss = np.bincount(ph_parent, weights=ph_E_eV / MEC2_EV, minlength=n_mc) \
        if total_photons else np.zeros(n_mc)
    frac0_measured = float(np.mean(n_phot == 0))
    frac0_poisson = float(np.mean(np.exp(-lambda_total)))

    D0 = float(doppler_D(cfg.eps0, cfg))
    X0 = float(recoil_X(cfg.eps0, cfg))
    edge_factor = D0 / (1.0 + X0) if cfg.quantum else D0
    E_gamma_max_eV = edge_factor * cfg.eps_L * MEC2_EV

    n_onaxis = int(onaxis_mask.sum())
    E_onaxis = ph_E_eV[onaxis_mask]
    onaxis_mean_eV = float(E_onaxis.mean()) if n_onaxis else 0.0
    onaxis_rms_rel = float(E_onaxis.std() / onaxis_mean_eV) if n_onaxis and onaxis_mean_eV else 0.0

    summary = {
        "n_mc": float(n_mc),
        "N_gamma_total": N_gamma_total,
        "mean_photons_per_electron": mean_lambda,
        "measured_mean_n_phot": float(n_phot.mean()),
        "total_macrophotons": float(total_photons),
        "total_photons_weighted": float(total_photons) * weight,
        "total_photons_in_window": float(in_window.sum()),
        "total_photons_in_window_weighted": float(in_window.sum()) * weight,
        "frac_n0_measured": frac0_measured,
        "frac_n0_poisson_closedform": frac0_poisson,
        "frac_n1": float(np.mean(n_phot == 1)),
        "frac_n2": float(np.mean(n_phot == 2)),
        "frac_n3plus": float(np.mean(n_phot >= 3)),
        "E_gamma_eV_mean": float(ph_E_eV.mean()) if total_photons else 0.0,
        "E_gamma_eV_max_observed": float(ph_E_eV.max()) if total_photons else 0.0,
        "E_gamma_eV_onaxis_max": E_gamma_max_eV,
        "mean_electron_dgamma": float(eloss.mean()),
        "energy_balance_eV": float(ph_E_eV.sum() - eloss.sum() * MEC2_EV),
        "truncated_electrons": float(truncated_total),
        "theta_onaxis_gamma": cfg.theta_onaxis_gamma,
        "onaxis_macrophotons": float(n_onaxis),
        "onaxis_photons_weighted": float(n_onaxis) * weight,
        "onaxis_frac_of_total": float(n_onaxis) / total_photons if total_photons else 0.0,
        "onaxis_E_gamma_eV_mean": onaxis_mean_eV,
        "onaxis_E_gamma_rms_rel": onaxis_rms_rel,
        # geometry / quantum diagnostics
        "crossing_angle_rad": cfg.crossing_angle,
        "quantum": float(bool(cfg.quantum)),
        "doppler_D0": D0,
        "recoil_X0": X0,
        "kn_sigma_ratio0": float(kn_sigma_ratio(0.5 * X0)) if cfg.quantum else 1.0,
    }

    # final per-electron positions at the last emission time (t = t_last_emit)
    z_f = el["z0"] + C_LIGHT * t_last_emit_total
    x_f = el["x_w"] + z_f * thx_f
    y_f = el["y_w"] + z_f * thy_f

    # Always write the final 6-D distribution in SDDS ``.ele`` format, whether
    # the input bunch was loaded from a ``.ele`` file or sampled by the GUI
    # from the Electrons-panel inputs.  The reference energy used in the
    # ``dP = (p - p0) / p0`` column is
    #   1) the input file's ``Energy`` header (when the bunch was loaded
    #      from a ``.ele`` file), so the column is consistent with the input
    #      definition; or, when no file was loaded,
    #   2) the mean of the *initial* simulated Lorentz factors
    #      (``mean(eps_i)``), so the column tracks the user-set mean energy
    #      in the Electrons panel.
    eps_i_mean = float(np.mean(el["eps"])) if el["eps"].size else float(cfg.eps0)
    ref_gamma = eps_i_mean
    if electrons.get("params"):
        params = electrons["params"]
        if params.get("Energy"):
            ref_gamma = float(params["Energy"]) * 1000.0 * 1e6 / MEC2_EV
    if not (np.isfinite(ref_gamma) and ref_gamma > 0):
        ref_gamma = float(cfg.eps0) if cfg.eps0 > 0 else 1.0
    # save_elegant_ele derives the dP column from this reference gamma
    # (dP = gamma/reference_gamma - 1) -- ultra-relativistic limit
    # p ~ E, so (p - p0) / p0 ~= (eps - eps0) / eps0.
    # Default output filename; can be overridden via the
    # ``final_ele_path`` entry of the ``electrons`` dict (e.g. when
    # ``_load_ele`` wants a per-input-file path) or by an environment
    # variable for headless runs.
    final_ele_path = "final_distribution.ele"
    if electrons.get("final_ele_path"):
        final_ele_path = electrons["final_ele_path"]
    elif os.environ.get("KASCADE_FINAL_ELE_PATH"):
        final_ele_path = os.environ["KASCADE_FINAL_ELE_PATH"]
    _save_elegant_ele(
        _MacroBunch(x=x_f, y=y_f, z=z_f, thx=thx_f, thy=thy_f, gamma=eps_f, weight=weight),
        final_ele_path,
        reference_gamma=ref_gamma,
        description=("Final 6D electron distribution after "
                     "interaction with the laser pulse "
                     "(kascade)"))
    summary = dict(summary)
    summary["final_distribution_path"] = final_ele_path

    return Results(
        cfg=cfg, n_mc=n_mc, weight=weight,
        eps_i=el["eps"], thx_i=el["thx"], thy_i=el["thy"], z0=el["z0"],
        lambda_total=lambda_total, n_phot=n_phot,
        eps_f=eps_f, thx_f=thx_f, thy_f=thy_f,
        z_f=z_f, x_f=x_f, y_f=y_f, t_last_emit=t_last_emit_total,
        ph_E_eV=ph_E_eV, ph_thx=ph_thx_a, ph_thy=ph_thy_a,
        ph_thx_lab=ph_thx_lab_a, ph_thy_lab=ph_thy_lab_a,
        ph_t=ph_t_a, ph_x=ph_x_a, ph_y=ph_y_a,
        ph_gen=ph_gen, ph_parent=ph_parent, onaxis_mask=onaxis_mask,
        truncated=truncated_total, summary=summary,
    )

