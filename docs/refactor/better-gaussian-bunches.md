# Gaussian Electron Beam Modeling — Summary & Implementation Notes

## 1. Goal

Design a **physically consistent, simulation-ready representation** of an electron bunch that:

- Works with **macroparticle ensembles**
- Provides a **Gaussian parametric model**
- Supports **sampling ↔ fitting ↔ validation**
- Remains consistent with **relativistic dynamics in vacuum**

---

## 2. Representation

### Particle-level (simulation)

We use a **slice-based representation**:

\[
(x, x', y, y', z, \gamma)
\]

where:

- \(x, y, z\): positions
- \(x' = p_x / p_z\), \(y' = p_y / p_z\): angles
- \(\gamma\): relativistic factor

Important:
- This is **not a snapshot in time**
- It is a **phase-space slice**

---

### Canonical (internal for sampling)

Sampling must be done in:

\[
(x, p_x, y, p_y, z, \gamma)
\]

to ensure:

\[
\gamma^2 = 1 + p_x^2 + p_y^2 + p_z^2
\]

---

## 3. Sampling (Gaussian at waist)

### Assumptions

- Beam defined at **waist** (\(\alpha = 0\))
- No correlations at generation stage
- Vacuum propagation → correlations emerge naturally

---

### Algorithm

1. Sample:
\[
x \sim \mathcal{N}(0, \sigma_x), \quad
p_x \sim \mathcal{N}(0, \gamma_0 \sigma_{x'})
\]

Same for \(y\).

2. Sample longitudinal:
\[
z \sim \mathcal{N}(0, \sigma_z), \quad
\gamma \sim \mathcal{N}(\gamma_0, \sigma_\gamma)
\]

3. Enforce mass-shell:
\[
p_z = \sqrt{\gamma^2 - 1 - p_x^2 - p_y^2}
\]

4. Convert:
\[
x' = \frac{p_x}{p_z}, \quad y' = \frac{p_y}{p_z}
\]

---

### Key properties

- Relativistically consistent
- Correct coupling between transverse momentum and energy
- Avoids unphysical states

---

## 4. Vacuum propagation

Drift over distance \(L\):

\[
x \rightarrow x + x' L
\]
\[
y \rightarrow y + y' L
\]

- Produces **Twiss tilt (\(\alpha \neq 0\)) automatically**
- No need to encode \(\alpha\) in sampling

---

## 5. Fitting model

We fit a **structured Gaussian model** with limited correlations:

### Included correlations

- Transverse:
  \[
  \langle x x' \rangle, \quad \langle y y' \rangle
  \]
- Chirp:
  \[
  \langle z \gamma \rangle
  \]
- Dispersion:
  \[
  \langle x \gamma \rangle, \quad \langle y \gamma \rangle
  \]

---

### Procedure

1. Build data matrix:
\[
X = (x, x', y, y', z, \gamma)
\]

2. Compute:
\[
\mu = \mathbb{E}[X], \quad \Sigma = \mathrm{Cov}(X)
\]

3. Use structured covariance (optional projection)

---

## 6. Extract physical parameters

### Emittance

\[
\epsilon_x = \sqrt{\sigma_x^2 \sigma_{x'}^2 - \langle x x' \rangle^2}
\]

### Twiss

\[
\beta_x = \frac{\sigma_x^2}{\epsilon_x}, \quad
\alpha_x = -\frac{\langle x x' \rangle}{\epsilon_x}
\]

(same for \(y\))

---

### Dispersion

\[
D_x = \frac{\langle x \gamma \rangle}{\sigma_\gamma^2}
\]

---

### Chirp

\[
h = \frac{\langle z \gamma \rangle}{\sigma_z^2}
\]

---

## 7. Fit accuracy estimation

### Problem

We must distinguish:

- **sampling noise** (\(\sim 1/\sqrt{N}\))
- **true model mismatch**

---

### Solution: synthetic baseline

1. Compute fit \((\mu, \Sigma)\)
2. Generate synthetic Gaussian samples
3. Compare metrics

---

### Metrics

#### (A) Mahalanobis distance

\[
d^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)
\]

Expected:
\[
d^2 \sim \chi^2(6)
\]

---

#### (B) KS test

Compare empirical \(d^2\) to \(\chi^2\)

---

#### (C) Log-likelihood

\[
\log p(x) = -\frac{1}{2} \left( k \log(2\pi) + \log|\Sigma| + d^2 \right)
\]

---

### Final diagnostics

Compare:

- real vs synthetic KS
- real vs synthetic mean \(d^2\)
- log-likelihood difference

---

### Interpretation

| Result | Meaning |
|------|--------|
| real ≈ synthetic | fit is noise-limited |
| real > synthetic | model mismatch |
| large deviation | non-Gaussian structure |

---

## 8. Important implementation details

### (1) Always sample in canonical variables

Never sample directly in:
\[
(x, x', y, y', z, \gamma)
\]

---

### (2) Enforce mass-shell

Mandatory:
\[
\gamma^2 > 1 + p_x^2 + p_y^2
\]

Use rejection sampling.

---

### (3) Center data before fitting

Always subtract mean:
\[
X_c = X - \mu
\]

---

### (4) Use unbiased or biased covariance consistently

- For physics → biased (population)
- For statistics → unbiased (sample)

---

### (5) Slice vs time

- Data is a **slice**, not a time snapshot
- Do NOT attempt time synchronization

---

### (6) Correlation structure

Allowed:
- \(x\)-\(x'\), \(y\)-\(y'\)
- \(z\)-\(\gamma\)
- \(x\)-\(\gamma\), \(y\)-\(\gamma\)

Avoid:
- arbitrary full 6×6 correlations
- unless physically justified

---

### (7) Twiss emergence

- Sampling at waist + drift ⇒ correct Twiss
- No need to fit \(\alpha\) at generation

---

### (8) Numerical stability

- Use `slogdet` for log-det
- Regularize covariance if needed:
  \[
  \Sigma \rightarrow \Sigma + \epsilon I
  \]

---

## 9. Conceptual takeaway

The correct pipeline is:

\[
\text{Gaussian model (waist)} \rightarrow \text{canonical sampling} \rightarrow \text{drift} \rightarrow \text{fit} \rightarrow \text{validate}
\]

This guarantees:

- physical consistency
- minimal bias
- correct interpretation of deviations

---

## 10. Future extensions

- Add chirp/dispersion in sampling
- Full canonical covariance transport:
  \[
  \Sigma \rightarrow R \Sigma R^T
  \]
- Non-Gaussian models (tails, mixtures)
- Integration with C++/GPU simulation

---

## 11. Key insight

> A "Gaussian in beam variables" is not automatically physical.  
> A "Gaussian in canonical variables with mass-shell constraint" is.

---

```python
"""
Electron beam Gaussian modeling, canonical sampling, fitting, and validation.

Overview
--------
This module provides a **physically consistent workflow** for representing,
sampling, fitting, and validating relativistic electron bunches in vacuum.

Key design principles:
- Particles are represented as a **beam slice** in variables:
    (x, x', y, y', z, gamma)
- Sampling is done in **canonical variables**:
    (x, p_x, y, p_y, z, gamma)
  with an enforced **relativistic mass-shell constraint**
- The Gaussian model is defined at the **beam waist**
  (alpha_x = alpha_y = 0), and Twiss tilt emerges automatically
  via ballistic propagation
- The fitting extracts only **physically meaningful correlations**:
    * transverse Twiss (alpha, beta, emittance)
    * longitudinal chirp
    * dispersion
- Accuracy is evaluated against a **sampling-noise baseline**
  using Mahalanobis distance, KS statistics, and log-likelihood

This module is intended for:
- Particle simulations (PIC / tracking / QED pipelines)
- Diagnostics and beam reconstruction
- Interfacing particle clouds with analytical beam models

Assumptions:
- Vacuum propagation (no collective fields)
- Gaussian core beam (non-Gaussian tails not modeled)
- Linear correlations only

Units:
- SI for positions (meters, seconds)
- gamma is dimensionless
- momenta normalized to mc

------------------------------------------------------------
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import chi2, kstest


# ============================================================
# Basic container
# ============================================================

@dataclass
class MacroBunch:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    thx: np.ndarray
    thy: np.ndarray
    gamma: np.ndarray
    weight: float

    @property
    def n_particles(self):
        return len(self.x)

    @property
    def N_e(self):
        return self.weight * self.n_particles


# ============================================================
# Canonical Gaussian sampling (waist)
# ============================================================

def sample_gaussian_waist(
    N,
    sigma_x,
    sigma_y,
    sigma_xp,
    sigma_yp,
    sigma_z,
    gamma0,
    sigma_gamma,
    rng=None,
):
    """
    Sample particles from a Gaussian beam defined at waist.

    Sampling is done in canonical variables:
        (x, p_x, y, p_y, z, gamma)

    and then mapped to:
        (x, x'=p_x/p_z, y, y'=p_y/p_z, z, gamma)

    Ensures:
    - exact relativistic consistency
    - correct coupling between divergence and energy
    """

    if rng is None:
        rng = np.random.default_rng()

    x = rng.normal(0, sigma_x, N)
    y = rng.normal(0, sigma_y, N)
    z = rng.normal(0, sigma_z, N)

    px = rng.normal(0, gamma0 * sigma_xp, N)
    py = rng.normal(0, gamma0 * sigma_yp, N)
    gamma = rng.normal(gamma0, sigma_gamma, N)

    # enforce mass-shell
    pz2 = gamma**2 - 1 - px**2 - py**2
    mask = pz2 > 0

    while not np.all(mask):
        n_bad = np.sum(~mask)
        px[~mask] = rng.normal(0, gamma0 * sigma_xp, n_bad)
        py[~mask] = rng.normal(0, gamma0 * sigma_yp, n_bad)
        gamma[~mask] = rng.normal(gamma0, sigma_gamma, n_bad)
        pz2 = gamma**2 - 1 - px**2 - py**2
        mask = pz2 > 0

    pz = np.sqrt(pz2)

    thx = px / pz
    thy = py / pz

    return MacroBunch(x, y, z, thx, thy, gamma, weight=1.0)


# ============================================================
# Ballistic propagation (vacuum drift)
# ============================================================

def drift(bunch: MacroBunch, L: float):
    """
    Propagate beam in vacuum over distance L.
    """
    x = bunch.x + bunch.thx * L
    y = bunch.y + bunch.thy * L

    return MacroBunch(
        x=x,
        y=y,
        z=bunch.z,
        thx=bunch.thx,
        thy=bunch.thy,
        gamma=bunch.gamma,
        weight=bunch.weight,
    )


# ============================================================
# Fit with physical correlations
# ============================================================

def fit_beam(bunch: MacroBunch):
    """
    Fit structured Gaussian model including:
    - Twiss (x-x', y-y')
    - Chirp (z-gamma)
    - Dispersion (x-gamma, y-gamma)
    """

    X = np.stack(
        [bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma],
        axis=1,
    )

    mu = np.mean(X, axis=0)
    Xc = X - mu

    Sigma = np.cov(Xc, rowvar=False, bias=True)

    return mu, Sigma


# ============================================================
# Extract physical parameters
# ============================================================

def extract_parameters(mu, Sigma):
    """
    Extract physically meaningful beam parameters.
    """

    # indices
    ix, ixp, iy, iyp, iz, ig = range(6)

    # transverse emittance
    def emit(i, ip):
        return np.sqrt(
            Sigma[i, i] * Sigma[ip, ip] - Sigma[i, ip] ** 2
        )

    emit_x = emit(ix, ixp)
    emit_y = emit(iy, iyp)

    # Twiss
    beta_x = Sigma[ix, ix] / emit_x
    alpha_x = -Sigma[ix, ixp] / emit_x

    beta_y = Sigma[iy, iy] / emit_y
    alpha_y = -Sigma[iy, iyp] / emit_y

    # dispersion
    sigma_gamma2 = Sigma[ig, ig]
    D_x = Sigma[ix, ig] / sigma_gamma2
    D_y = Sigma[iy, ig] / sigma_gamma2

    # chirp
    sigma_z2 = Sigma[iz, iz]
    chirp = Sigma[iz, ig] / sigma_z2

    return {
        "emit_x": emit_x,
        "emit_y": emit_y,
        "beta_x": beta_x,
        "alpha_x": alpha_x,
        "beta_y": beta_y,
        "alpha_y": alpha_y,
        "D_x": D_x,
        "D_y": D_y,
        "chirp": chirp,
    }


# ============================================================
# Fit quality metrics
# ============================================================

def evaluate_fit(bunch, mu, Sigma, n_synthetic=3):
    """
    Evaluate Gaussian fit quality with sampling-noise baseline.
    """

    X = np.stack(
        [bunch.x, bunch.thx, bunch.y, bunch.thy, bunch.z, bunch.gamma],
        axis=1,
    )

    Xc = X - mu
    inv = np.linalg.inv(Sigma)

    d2 = np.einsum("ni,ij,nj->n", Xc, inv, Xc)

    # real stats
    ks_real, _ = kstest(d2, chi2(df=6).cdf)
    mean_real = np.mean(d2)

    # synthetic baseline
    rng = np.random.default_rng()
    ks_syn = []
    mean_syn = []

    for _ in range(n_synthetic):
        Xs = rng.multivariate_normal(mu, Sigma, size=len(X))
        d2s = np.einsum("ni,ij,nj->n", Xs - mu, inv, Xs - mu)

        ks, _ = kstest(d2s, chi2(df=6).cdf)
        ks_syn.append(ks)
        mean_syn.append(np.mean(d2s))

    return {
        "ks_real": ks_real,
        "ks_syn": np.mean(ks_syn),
        "ks_excess": ks_real - np.mean(ks_syn),
        "mean_d2_real": mean_real,
        "mean_d2_syn": np.mean(mean_syn),
    }


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # sample beam at waist
    bunch = sample_gaussian_waist(
        N=100000,
        sigma_x=1e-6,
        sigma_y=1e-6,
        sigma_xp=1e-3,
        sigma_yp=1e-3,
        sigma_z=3e-6,
        gamma0=1000,
        sigma_gamma=10,
    )

    # propagate
    bunch = drift(bunch, L=0.1)

    # fit
    mu, Sigma = fit_beam(bunch)

    # extract parameters
    params = extract_parameters(mu, Sigma)

    # evaluate fit
    stats = evaluate_fit(bunch, mu, Sigma)

    print("Beam parameters:")
    for k, v in params.items():
        print(f"{k}: {v:.3e}")

    print("\nFit quality:")
    for k, v in stats.items():
        print(f"{k}: {v:.3e}")
```