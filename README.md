# Boltzmann Prime Transitions

**The Prime Column Transition Matrix Is a Boltzmann Distribution at Temperature ln(N)**

Code and data for the paper by Antonio P. Matos (Fancyland LLC / Lattice OS).

---

## The Result

The transition matrix governing consecutive primes between residue classes modulo _m_ is predicted by a zero-parameter Boltzmann distribution:

$$T(a \to b) = \frac{\exp(-d(a,b) / \ln N)}{\sum_{j} \exp(-d(a,j) / \ln N)}$$

where _d(a,b)_ is the forward cyclic distance between admissible residue classes, _d(a,a) = m_ (self-return costs a full modular cycle), and _ln N_ is the Boltzmann temperature set by the Prime Number Theorem.

**No free parameters.** The modulus _m_ and the scale _N_ are inputs; the decay rate λ = 1/ln(N) is fixed by the PNT.

### Key Numbers

| Modulus | Columns | R² (10⁸–10⁹) | λ·ln(N) |
| ------- | ------- | ------------ | ------- |
| 30      | 8       | 0.970        | 0.989   |
| 210     | 48      | 0.988        | 1.055   |

The Lemke Oliver–Soundararajan (2016) diagonal suppression follows as a one-line corollary: self-transitions have the highest energy cost (_m_), making them the least probable transition at any finite temperature.

---

## Repository Structure

```
paper/
  BOLTZMANN_PRIME_TRANSITIONS.md   — Full paper (Markdown source)
  boltzmann_prime_transitions.tex  — LaTeX source for arXiv submission
  boltzmann_prime_transitions.pdf  — Compiled PDF (9 pages)

scripts/
  prime_drum_wave22c_boltzmann.py              — The zero-parameter model (core result)
  prime_drum_wave22d_10e9.py                   — 10⁹ extension (segmented sieve)
  prime_drum_wave22b_boltzmann.py              — Two-parameter intermediate model
  prime_drum_wave22_boltzmann.py               — Initial column-index model (failed, R²=0.08)
  prime_drum_wave22e_10e10.py                  — 10¹⁰ extension
  prime_drum_wave22f_scaling.py                — Scaling law analysis
  prime_drum_wave22g_hardy_littlewood.py       — Singular series test (negative result)
  prime_drum_wave22h_residual_archaeology.py   — Residual decomposition (SVD, chirality)
  prime_drum_wave22h_verify.py                 — Character hypothesis verification
  prime_drum_wave23_heisenberg.py              — Entropic uncertainty bound test

data/
  wave22c_boltzmann_*.json       — Zero-parameter model results
  wave22d_10e9_*.json            — 10⁹ results
  wave22e_10e10_*.json           — 10¹⁰ results
  wave22h_archaeology_*.json     — Residual decomposition data
  wave23_heisenberg_*.json       — Entropic uncertainty results
```

---

## Quick Start

Requirements: Python 3.8+, NumPy, SciPy.

```bash
pip install numpy scipy
```

**Run the core model** (takes ~30 seconds to sieve primes to 10⁹):

```bash
python scripts/prime_drum_wave22c_boltzmann.py
```

This will:

1. Sieve all primes to 10⁹ (50,847,534 primes)
2. Compute empirical transition matrices in six octave windows
3. Compute the zero-parameter Boltzmann predictions
4. Report R² and λ convergence at each scale

**Run the residual analysis:**

```bash
python scripts/prime_drum_wave22h_residual_archaeology.py
```

**Run the Hardy-Littlewood test (negative result):**

```bash
python scripts/prime_drum_wave22g_hardy_littlewood.py
```

---

## How to Read the Scripts

The scripts are numbered by "wave" — each wave represents one step in a 23-wave adversarial falsification protocol. The path to the result was:

- **Wave 22a** (`wave22_boltzmann.py`): Column-index distance. **R² = 0.08. Failed catastrophically.** The Boltzmann idea was right; the distance metric was wrong.
- **Wave 22b** (`wave22b_boltzmann.py`): Corrected to forward residue distance, two-parameter fit. R² = 0.97. Discovered that the self-penalty converges to _m_.
- **Wave 22c** (`wave22c_boltzmann.py`): Set _d(a,a) = m_ and _λ = 1/ln(N)_. **Zero parameters. R² = 0.970.** This is the paper's core result.
- **Wave 22d–f**: Extended to 10⁹, 10¹⁰, and scaling analysis.
- **Wave 22g** (`wave22g_hardy_littlewood.py`): Tested whether the Hardy-Littlewood singular series improves the model. **It does not.** The optimal weight is exactly zero.
- **Wave 22h** (`wave22h_residual_archaeology.py`): Decomposed the 3% residual into circulant and non-circulant components. Found chirality (columns 7,11 enhanced; 23,19 suppressed).
- **Wave 23** (`wave23_heisenberg.py`): Tested the entropic uncertainty bound. Boltzmann rows are anomalously close to the bound (2.6th percentile vs. random) but this is a property of smooth exponential distributions, not specific to primes.

---

## Citation

If you use this code or data, please cite:

```
Antonio P. Matos (2026). "The Prime Column Transition Matrix Is a Boltzmann Distribution
at Temperature ln(N)." Preprint. https://github.com/fancyland-llc/boltzmann-prime-transitions
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

_Fancyland LLC — Lattice OS research infrastructure._
*https://www.fancyland.net*
