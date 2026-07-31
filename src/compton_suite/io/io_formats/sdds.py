"""Elegant / SDDS ``.ele`` file I/O for :class:`compton_suite.io.bunch.Bunch`.

Relocated from ``kascade.load_ele_file``/``save_ele_file`` (moved, not
duplicated -- that repo's own CLI never used these; their only caller was
the GUI's ``.ele``-load path). Same minimal hand-rolled SDDS ASCII parser:
reads the header parameters (``Energy``, ``SigmaX``, ``SigmaXp``, ...) and
the per-particle columns (``x``, ``xp``, ``y``, ``yp``, ``z``, ``dP``).

Caveat carried over unchanged: a plain elegant ``.ele`` file as parsed here
carries no bunch-charge/current information, so a loaded
:class:`Bunch`'s ``weight`` field is not authoritative -- every
   existing consumer (e.g. kascade's ``run_simulation``) recomputes the real
   per-macroparticle weight from its own separately-entered charge/``N_e``
   config field, ignoring whatever ``weight`` a loaded bunch carries. Only the
   particle-array *shape* (position, angle, energy distribution) is
   authoritative from a loaded file; downstream code should not rely on
``Bunch.weight``/``N_e`` from a loaded file without an explicit
charge override.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np

from ..bunch import Bunch
from ..units import MEC2_EV

__all__ = ["load_elegant_ele", "save_elegant_ele"]

_NAME_RE = re.compile(r'name\s*=\s*(?:"([^"]+)"|(\S+))', re.IGNORECASE)
_VAL_RE = re.compile(r'fix\s*=\s*("([^"]*)"|[+\-]?\d+\.?\d*(?:[eE][+\-]?\d+)?)', re.IGNORECASE)


def load_elegant_ele(path: str) -> Bunch:
    """Parse a 6-D electron-bunch ``.ele`` file in SDDS ASCII format.

    Required columns: ``x``, ``xp``, ``y``, ``yp``, ``z``, ``dP``. Mean
    energy is read from the header ``Energy`` [GeV] parameter; per-particle
    energy is ``eps0 * (1 + dP)`` (ultra-relativistic, ``p ~ E``).
    """
    with open(path, "r") as fh:
        text = fh.read()

    lines = text.splitlines()
    if not lines or not lines[0].strip().startswith("SDDS"):
        raise ValueError(f"{path}: not an SDDS file (missing SDDS1 header)")

    params: dict[str, float] = {}
    columns: List[str] = []
    in_data = False
    data_lines: List[str] = []

    i = 1
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line or line.startswith("!"):
            continue
        low = line.lower()
        if low.startswith("&description"):
            continue
        if low.startswith("&end"):
            break
        if low.startswith("&parameter"):
            m_n = _NAME_RE.search(line)
            m_v = _VAL_RE.search(line)
            if m_n and m_v:
                name = m_n.group(1) or m_n.group(2)
                v = m_v.group(2) if m_v.group(2) is not None else m_v.group(1)
                try:
                    params[name] = float(v)
                except (TypeError, ValueError):
                    pass
            continue
        if low.startswith("&column"):
            m_n = _NAME_RE.search(line)
            if m_n:
                columns.append(m_n.group(1) or m_n.group(2))
            continue
        if low.startswith("&data"):
            in_data = True
            continue
        if in_data:
            data_lines.append(raw)

    if not columns:
        raise ValueError(f"{path}: no &column definitions found")
    if not data_lines:
        raise ValueError(f"{path}: no data rows found before &end")

    rows: List[List[float]] = []
    expected = len(columns)
    for raw in data_lines:
        s = raw.strip()
        if not s:
            continue
        if "=" in s.split()[0]:
            continue
        vals = s.split()
        if len(vals) < expected:
            raise ValueError(f"{path}: row has {len(vals)} fields, expected {expected}")
        try:
            row = [float(v) for v in vals[:expected]]
        except ValueError as e:
            raise ValueError(f"{path}: cannot parse row: {s!r}") from e
        rows.append(row)

    arr = np.asarray(rows, dtype=float)
    col = {c: arr[:, j] for j, c in enumerate(columns)}

    for required in ("x", "xp", "y", "yp", "z", "dP"):
        if required not in col:
            raise ValueError(f"{path}: required column '{required}' missing")

    if "Energy" in params:
        energy_MeV = float(params["Energy"]) * 1000.0
    else:
        energy_MeV = None
    eps0_mean = energy_MeV * 1e6 / MEC2_EV if energy_MeV is not None else None
    if eps0_mean is None or not np.isfinite(eps0_mean) or eps0_mean <= 0:
        raise ValueError(f"{path}: cannot determine mean beam energy")
    gamma = eps0_mean * (1.0 + col["dP"])
    gamma = np.clip(gamma, 1.0 + 1e-9, None)

    n_mc = col["x"].size
    return Bunch(
        x=col["x"], y=col["y"], z=col["z"], thx=col["xp"], thy=col["yp"], gamma=gamma,
        weight=1.0,  # not authoritative -- see module docstring
        meta={"source_path": path, "sdds_params": params, "n_particles_loaded": int(n_mc)},
    )


def save_elegant_ele(bunch: Bunch, path: str, *,
                      description: str = "6D electron bunch",
                      reference_gamma: float | None = None) -> None:
    """Write a :class:`Bunch` in SDDS ASCII ``.ele`` format.

    Mirrors :func:`load_elegant_ele`'s layout: one ``&parameter`` per
    header scalar (``TotalParticles``, ``Energy``, ``NormEmittance``,
    ``BetaFunction``, ``SigmaX``, ``SigmaXp``, ``SigmaY``, ``SigmaYp``,
    ``SigmaZ``, ``SigmaDp``) and one ``&column`` per particle coordinate
    (``x``, ``xp``, ``y``, ``yp``, ``z``, ``dP``).

    ``reference_gamma``: the gamma the ``dP``/``Energy`` header columns are
    computed relative to. Defaults to the bunch's own mean gamma (natural
    for a freshly-sampled bunch), but a caller that already has its own
    reference (e.g. a fixed reference energy carried over from an earlier
    loaded ``.ele`` file's header, rather than the current bunch's actual
    mean) should pass it explicitly to preserve that reference exactly.
    """
    x = np.asarray(bunch.x, dtype=float).ravel()
    xp = np.asarray(bunch.thx, dtype=float).ravel()
    y = np.asarray(bunch.y, dtype=float).ravel()
    yp = np.asarray(bunch.thy, dtype=float).ravel()
    z = np.asarray(bunch.z, dtype=float).ravel()
    gamma = np.asarray(bunch.gamma, dtype=float).ravel()
    n = x.size
    if not (xp.size == y.size == yp.size == z.size == gamma.size == n):
        raise ValueError("save_elegant_ele: all bunch arrays must be the same length")

    gamma0 = float(reference_gamma) if reference_gamma is not None else float(np.mean(gamma))
    dP = gamma / gamma0 - 1.0
    mean_energy_MeV = gamma0 * MEC2_EV / 1e6

    sx = float(np.std(x, ddof=0))
    sxp = float(np.std(xp, ddof=0))
    sy = float(np.std(y, ddof=0))
    syp = float(np.std(yp, ddof=0))
    sz = float(np.std(z, ddof=0))
    sdp = float(np.std(dP, ddof=0))

    if sxp > 0 and sx > 0:
        emit_x_geom = sx * sxp
        beta_x = sx * sx / emit_x_geom
    else:
        emit_x_geom = 0.0
        beta_x = 0.0
    if syp > 0 and sy > 0:
        emit_y_geom = sy * syp
        beta_y = sy * sy / emit_y_geom
    else:
        emit_y_geom = 0.0
        beta_y = 0.0

    energy_GeV = mean_energy_MeV / 1000.0
    emit_x_norm_mm_mrad = emit_x_geom * gamma0 * 1e6
    emit_y_norm_mm_mrad = emit_y_geom * gamma0 * 1e6
    if emit_y_norm_mm_mrad <= 0 and emit_x_norm_mm_mrad > 0:
        emit_y_norm_mm_mrad = emit_x_norm_mm_mrad
        beta_y = beta_x

    lines: List[str] = []
    lines.append("SDDS1")
    lines.append(f'&description text="{description}" &')
    lines.append(f"&parameter name=TotalParticles type=long description=\"Number of macroparticles\" fix={n} &")
    lines.append(f"&parameter name=Energy type=double units=GeV description=\"Mean beam energy\" fix={energy_GeV:.6e} &")
    lines.append(f"&parameter name=NormEmittance type=double units=mm*mrad description=\"Normalized emittance\" fix={emit_x_norm_mm_mrad:.6e} &")
    lines.append(f"&parameter name=BetaFunction type=double units=m description=\"Beta function at location\" fix={beta_x:.6e} &")
    lines.append(f"&parameter name=SigmaX type=double units=m description=\"RMS horizontal beam size\" fix={sx:.6e} &")
    lines.append(f"&parameter name=SigmaXp type=double units=rad description=\"RMS horizontal divergence\" fix={sxp:.6e} &")
    lines.append(f"&parameter name=SigmaY type=double units=m description=\"RMS vertical beam size\" fix={sy:.6e} &")
    lines.append(f"&parameter name=SigmaYp type=double units=rad description=\"RMS vertical divergence\" fix={syp:.6e} &")
    lines.append(f"&parameter name=SigmaZ type=double units=m description=\"RMS bunch length\" fix={sz:.6e} &")
    lines.append(f"&parameter name=SigmaDp type=double description=\"RMS momentum spread\" fix={sdp:.6e} &")
    lines.append("&column name=x type=double units=m description=\"Horizontal position\" &")
    lines.append("&column name=xp type=double units=rad description=\"Horizontal angle (px/pz)\" &")
    lines.append("&column name=y type=double units=m description=\"Vertical position\" &")
    lines.append("&column name=yp type=double units=rad description=\"Vertical angle (py/pz)\" &")
    lines.append("&column name=z type=double units=m description=\"Longitudinal position (head-tail, positive = ahead)\" &")
    lines.append("&column name=dP type=double description=\"Relative momentum deviation (p-p0)/p0\" &")
    lines.append("&data mode=ascii")
    for xi, xpi, yi, ypi, zi, di in zip(x, xp, y, yp, z, dP):
        lines.append(f" {xi:.6e}  {xpi:.6e}  {yi:.6e}  {ypi:.6e}  {zi:.6e}  {di:.6e}")
    lines.append("&end")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
