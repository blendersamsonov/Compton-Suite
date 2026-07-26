"""Cross-checks for constants.py's pint-derived values.

No cupy/GPU/tkinter needed. Run with `python3 -m pytest tests/` or
`python3 tests/test_constants.py` directly (plain asserts, no pytest
dependency required).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compton_suite.io import constants  # noqa: E402


def test_si_values_are_sane():
    # Exact-defined SI constants -- must match to full precision.
    assert constants.C_LIGHT == 299_792_458.0
    assert constants.E_CHARGE == 1.602176634e-19

    # CODATA-measured constants -- sanity range, not exact-match (pint's
    # own vintage may tick forward over time; these bounds are generous).
    assert 1.0545e-34 < constants.HBAR < 1.0546e-34
    assert 9.1093e-31 < constants.ME < 9.1094e-31
    assert 510_998.0 < constants.MEC2_EV < 510_999.5
    assert 8.854e-12 < constants.EPS0 < 8.855e-12
    assert 0.007297 < constants.ALPHA < 0.007298
    assert 2.817e-15 < constants.R_E_M < 2.819e-15
    assert 6.652e-29 < constants.SIGMA_T_M2 < 6.653e-29


def test_mec2_consistent_with_me():
    # MEC2_EV should equal ME*C_LIGHT**2 converted to eV via E_CHARGE
    # (E[eV] = E[J] / e[C]).
    mec2_from_me = constants.ME * constants.C_LIGHT**2 / constants.E_CHARGE
    assert abs(mec2_from_me - constants.MEC2_EV) / constants.MEC2_EV < 1e-12


def test_sigma_t_consistent_with_r_e():
    import math

    sigma_t_from_r_e = (8.0 * math.pi / 3.0) * constants.R_E_M**2
    assert abs(sigma_t_from_r_e - constants.SIGMA_T_M2) / constants.SIGMA_T_M2 < 1e-3


def test_cgs_views_match_si():
    assert constants.HBAR_ERG_S == constants.HBAR * 1e7
    assert constants.ME_G == constants.ME * 1e3
    assert constants.C_CM_S == constants.C_LIGHT * 1e2

    # 1 C = c[cm/s]/10 statC -- cross-check against xigma_i's own
    # previously-correct `el` literal (4.803204673e-10 statC), which had
    # no CODATA-vintage mismatch (only hbar/electron mass did).
    assert abs(constants.EL_STATC - 4.803204673e-10) / 4.803204673e-10 < 1e-8


def test_hbar_within_known_xigma_i_legacy_delta():
    # xigma_i's pre-migration hbar (2014-CODATA-vintage CGS literal,
    # 1.054571800e-27 erg*s) is off from this module's canonical value by
    # ~1.6e-8 relative -- documents the exact magnitude of the physics
    # change xigma_i's config.py migration applies.
    legacy_hbar_erg_s = 1.054571800e-27
    rel_delta = abs(constants.HBAR_ERG_S - legacy_hbar_erg_s) / legacy_hbar_erg_s
    assert 1e-9 < rel_delta < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll constants tests passed.")
