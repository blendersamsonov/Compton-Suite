# compton_suite.io: Shared Physics Layer

Unified physical constants, unit registry, electron-bunch/laser-pulse representations, and output dataclasses for all physics engines and the GUI.

## Why this exists

**Multiple copies of physical constants cause silent divergence.** Every physics engine and the GUI must import constants and parameter-semantics frameworks from the same single source. If each module maintained its own copies, a `PhysicalQuantity` built with one copy's enums would fail validation against another's spec. `compton_suite.io` is the single, audited source; every consumer depends on it directly.

## Organization

All framework, constants, and parameter semantics live in unified `src/compton_suite/io/` with no split files (`constants.py`, `enums.py`, `quantities.py` merged into single files where appropriate per refactor). See root `CLAUDE.md`'s "Layout" section for the complete file map and dependency flow.

## Specs

For data format/contract documentation:

- `docs/io/specs/gaussian_paraxial_laser_io_v0.1.md` — laser-pulse analytic model (gaussian_paraxial v0.1)
- `docs/io/specs/electron_beam_io_v0.1_full.md` / `_short.md` — electron-bunch analytic model (gaussian_6d_waist v0.1)
