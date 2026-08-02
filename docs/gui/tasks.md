# GUI tasks

In order of priority:
- [ ] Abandon angular-range tab.
- [ ] Remove hard-coded seed and number of macroelectrons from compton photons tab into model specific parameters
- [ ] Grey out the inputs after a simulation is done, except for total number of reals electrons, i.e. charge (it's just a global scalar). For XIGMA leave pulse energy and gamma active, since spectra can be quickly replotted if geometry is unchanged. Add a "release" button to allow user to change parameters again.
- [ ] Dropdown menu for each input with units selection, i.e. cm, m, mm, um, etc. Should be still possible when the values are read only when e.g. they are extracted from a macrobunch. Physical quantity should be conserved when changing the units, so whenever units are change a proper conversion is performed
- [ ] Different self-consistent ways of defining spatial scales: either with beam waist or rayleigh range/beta function
- [ ] 2d/3d sketches showing interaction geometry. Axes, 3D ellipses for electrons and laser, arrows showing laser  polarization. "Ghosts" with reduced alpha at different time delays from the focus (defined by the electron beam)
- [ ] Sliders for inputs
- [ ] Parameter scans/ranges
- [ ] Saving to file (graphs, photons representation)
