# Model tasks

## Xigma-i

- [x] Make sure to properly use streaming capabilities when ran from gui. Get VRAM amount and make sure that the program never allocates gpu arrays larger than, say 70%, of full amount -- `particles.push_and_sample(..., chunk=...)` + `particles.estimate_chunk_size()`, wired into both `XigmaAdapter`/`DirectAdapter` as a `model_params()` field (`0` = auto-size from `gammaforge.misc.available_vram_bytes()`)
- [ ] Implement crossing angle. Should only change the polarization factor. Geometric overlap uses photon density so angle between electron's momentum and polarization is irrelevant
- [ ] Remove '-i' from naming, should be just "XIGMA"
- [ ] Consider if gamma-axis rescaling can be achieved similarly to the a0 rescaling, which would allow to change particles mean energy without needed to recompute stages 0-1

## Analytical model

- [ ] Include foci displacement in the analytical model
- [ ] Consider whether a closed form analytical expression for total yield exists for non-round beams
- [ ] Collimated spectrum construction from total yields, collimation angle and spectrum width. Should just give an approximately correct number of collimated photons and spectrum constructed as convolution of single electron spectrum with energy distribution function (and possibly a0)

## All models

- [ ] Implement jitter and averaging over shots
