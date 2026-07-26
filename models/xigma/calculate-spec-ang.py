"""Example: total yield, angle-integrated spectrum, and angular spectrum
for a laser-electron collision, from a TOML config (see example-config.toml).

Uses the tabulated-energy pipeline (particles.py/deposition.py/
spectrum4d.py/reference.py via TabulatedEngine) -- the only compute path
in this package. Runs on a CUDA GPU if one is available, else falls back
to a CPU/numba implementation of the same kernels automatically (see
config._detect_device).
"""
import argparse
import os
import sys
import tomllib

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from xigma_i.config import Compton, _detect_device
from xigma_i.tabulated_engine import TabulatedEngine

c = 2.99792458e+10


def main():
    parser = argparse.ArgumentParser(description="Calculate angular and spectral distribution of Compton photons")
    parser.add_argument(
        "-c", "--config",
        default="example-config.toml",
        help="Path to the TOML configuration file (default: example-config.toml)"
    )
    parser.add_argument("--n-particles", type=int, default=100_000,
                         help="Stage 0/1 particle count (table statistics)")
    parser.add_argument("--n-steps", type=int, default=64,
                         help="trajectory-integration resolution per particle")
    parser.add_argument("--n-bins", type=int, nargs=4, default=(64, 64, 64, 16),
                         metavar=("N_GAMMA", "N_THETA_X", "N_THETA_Y", "N_A0"))
    parser.add_argument("--samples-per-point", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    e = config["electron"]
    l = config["laser"]

    # Electrons
    chargeNC = e["charge"]
    gamma_0 = e["gamma"]
    sigma_gamma_p = e["sigma_gamma_rel"]
    sigma_ex = e["sigma_x"]
    sigma_ey = e["sigma_y"]
    electron_duration = e["duration"]
    norm_emit_x = e["norm_emit_x"]
    norm_emit_y = e["norm_emit_y"]
    focus_x = e["focus_x"]
    focus_y = e["focus_y"]
    focus_z = e["focus_z"]

    emit_x = norm_emit_x / gamma_0
    emit_y = norm_emit_y / gamma_0
    sigma_ez = electron_duration * c

    # Laser
    w_l = l["energy"]
    lambda_l = l["lambda"]
    sigma_lr0 = l["sigma_r"]
    laser_duration = l["duration"]
    pol = l["polarization_angle"]
    beta_ff = l.get("beta_ff", 0.0)
    sigma_lz = laser_duration * c

    compton = Compton()
    compton.set_electron_parameters(chargeNC=chargeNC, emit_x=emit_x, emit_y=emit_y, sigma_ex=sigma_ex, sigma_ey=sigma_ey, sigma_ez=sigma_ez)
    compton.set_laser_parameters(WL=w_l, lambda_l=lambda_l, sigma_lr0=sigma_lr0, sigma_lz=sigma_lz, beta_ff=beta_ff)
    compton.set_foci_displacement(focus_x, focus_y, focus_z)

    sigma_gamma = sigma_gamma_p * gamma_0
    phi_pol = pol * np.pi / 180.0

    device = _detect_device()
    print(f"backend: {device}")
    xp = compton.xp
    backend = 'cupy' if device == 'gpu' else 'numpy'

    engine = TabulatedEngine(compton)
    engine.run(args.n_particles, gamma_0, sigma_gamma, n_steps=args.n_steps,
               n_bins=tuple(args.n_bins), backend=backend,
               rng=np.random.default_rng(args.seed))
    print(f"Total yield: {engine.total_yield:.2e} photons")

    s_scale = 4.0 * compton.Wph  # E [MeV] = s_scale * s

    thx = xp.linspace(0.0, 1.0 / gamma_0, 127, dtype=xp.float32)
    th_zero = xp.array([0.0], dtype=xp.float32)
    ss = (xp.linspace(0.8, 1.1, 256) * gamma_0 ** 2).astype(xp.float32)
    spec_kwargs = dict(phi_pol=phi_pol, samples_per_point=args.samples_per_point, device=device)
    spec_x, _, _ = engine.angular_spectrum(ss, thx, th_zero, **spec_kwargs)
    spec_y, _, _ = engine.angular_spectrum(ss, th_zero, thx, **spec_kwargs)

    ss_tot = (xp.linspace(0.0, 1.1, 512) * gamma_0 ** 2).astype(xp.float32)
    dNds_tot = compton.asnumpy(engine.spectrum(ss_tot))
    dNdE_tot = dNds_tot / s_scale

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")

    ax = axes[0]
    ax.plot(compton.asnumpy(ss_tot) * s_scale, dNdE_tot)
    ax.set_xlabel("E, MeV")
    ax.set_ylabel(r"dN/dE, MeV$^{-1}$")

    # spec_x/spec_y (d2N/ds dOmega, dimensionless-s) -> d2N/dE dOmega, per MeV
    data_y = (spec_y / s_scale).sum(axis=0) * 1e-6
    data_x = (spec_x / s_scale).sum(axis=1) * 1e-6

    vmin = min(data_y.min(), data_x.min())
    vmax = max(data_y.max(), data_x.max())
    ss_host = compton.asnumpy(ss)
    thx_host = compton.asnumpy(thx)
    for i, spec in enumerate([data_y, data_x]):
        ax = axes[1 + i]
        im = ax.imshow(spec, origin='lower',
                        extent=[ss_host[0] * s_scale, ss_host[-1] * s_scale, thx_host[0] * 1e3, thx_host[-1] * 1e3],
                        vmin=vmin, vmax=vmax)
        ax.set_aspect('auto')
        ax.set_xlabel("E, MeV")
        ax.set_ylabel(f"θ{'yx'[i]}, mrad")

    fig.colorbar(im, ax=[axes[1], axes[2]], label=r'dN/dEdΩ, MeV$^{-1}$μsr$^{-1}$')
    plt.show()


if __name__ == "__main__":
    main()
