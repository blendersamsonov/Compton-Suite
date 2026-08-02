# compton_suite.io: Shared Physics Layer

Unified physical constants, unit registry, electron-bunch/laser-pulse representations, and output dataclasses for all physics engines and the GUI.

## Why this exists

Each of the six component repositories (before this merge) maintained its own copy of physical constants—with a real ~1.6e-8 relative numeric disagreement between xigma-i's older-CODATA copy and the others. When separate copies of parameter-semantics frameworks coexisted, a `PhysicalQuantity` built with one copy's enums failed validation against another's `ModelSpec`.

`compton_suite.io` consolidates this into a single, audited, shared layer that every consumer imports directly—eliminating silent unit/constant divergence.

## Organization

All framework, constants, and parameter semantics live in unified `src/compton_suite/io/` with no split files (`constants.py`, `enums.py`, `quantities.py` merged into single files where appropriate per refactor). See root `CLAUDE.md`'s "Layout" section for the complete file map and dependency flow.

## Specs

For data format/contract documentation:

- `docs/io/specs/gaussian_paraxial_laser_io_v0.1.md` — laser-pulse analytic model (gaussian_paraxial v0.1)
- `docs/io/specs/electron_beam_io_v0.1_full.md` / `_short.md` — electron-bunch analytic model (gaussian_6d_waist v0.1)
