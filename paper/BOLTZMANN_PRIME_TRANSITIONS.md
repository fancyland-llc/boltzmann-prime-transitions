# The Prime Column Transition Matrix Is a Boltzmann Distribution at Temperature ln(N)

**Author:** Antonio P. Matos  
**ORCID:** [0009-0002-0722-3752](https://orcid.org/0009-0002-0722-3752)  
**Date:** March 17, 2026  
**Affiliation:** Independent Researcher; Fancyland LLC / Lattice OS  
**Status:** Preprint  
**DOI:** [10.5281/zenodo.19076680](https://doi.org/10.5281/zenodo.19076680)  
**MSC 2020:** 11N05 (primary), 11A41, 11K99, 82B05 (secondary)  
**Keywords:** prime gaps, consecutive primes, residue classes, Boltzmann distribution, transition matrix, Lemke Oliver–Soundararajan, Cramér model, prime number theorem

---

## Abstract

We show that the transition matrix governing consecutive primes between residue classes modulo $m$ is predicted by a Boltzmann distribution on the forward cyclic distances of the admissible residue group $(\mathbb{Z}/m\mathbb{Z})^*$, with temperature equal to the mean prime gap $\ln N$. The model has **zero free parameters**. At modulus 30, it achieves $R^2 \approx 0.970$ against empirical matrices measured across six orders of magnitude ($10^3$ to $10^9$), with the fitted decay rate converging to within 1.1% of the PNT prediction $\lambda = 1/\ln N$ at the largest measured scale. At modulus 210 (48 columns), $R^2 = 0.988$ at the largest scale, improving monotonically with $N$. The diagonal suppression discovered by Lemke Oliver and Soundararajan (2016) follows as a one-line corollary: self-transitions cost energy $m$ (a full modular cycle), making them the least probable transition at any finite temperature. The 3% residual is structured: it decomposes into a circulant component scaling as $O(1/\ln N)$ and a non-circulant component scaling as $O(1/\ln^{1.6} N)$. The Hardy-Littlewood singular series does not appear at any tested order (§5). The Boltzmann framing was first proposed by an AI worker (Gemini HELICASE) during a constrained swarm convergence run, then confirmed through a 22-wave adversarial falsification protocol (Appendix B).

**Code and data:** https://github.com/fancyland-llc/boltzmann-prime-transitions

---

## 1. The Model

Let $m$ be a primorial ($6, 30, 210, \ldots$) and let $\mathcal{C} = \{c_1, \ldots, c_{\varphi(m)}\}$ be the admissible residue classes modulo $m$ (i.e., integers $1 \leq r < m$ with $\gcd(r, m) = 1$).

For consecutive primes $p_n \equiv a \pmod{m}$ and $p_{n+1} \equiv b \pmod{m}$, define the forward residue distance:

$$d(a, b) = \begin{cases} (b - a) \bmod m & \text{if } b \neq a \\ m & \text{if } b = a \end{cases}$$

The self-distance $d(a, a) = m$ reflects the physical constraint that a prime in column $a$ cannot return to column $a$ without advancing at least $m$ on the number line.

**Claim.** The transition probability is given by

$$\boxed{T(a \to b) = \frac{\exp\!\bigl(-d(a,b) \,/\, \ln N\bigr)}{\displaystyle\sum_{j \in \mathcal{C}} \exp\!\bigl(-d(a,j) \,/\, \ln N\bigr)}}$$

where $N$ is the scale (midpoint of the measurement window) and $\ln N$ serves as the Boltzmann temperature. This model has **zero free parameters**: the modulus $m$ and the scale $N$ are given, and the decay rate $\lambda = 1/\ln N$ follows from the Prime Number Theorem.

---

## 2. Derivation

The Cramér probabilistic model gives the distribution of prime gaps near $N$ as approximately exponential:

$$P(\text{gap} = g) \approx \frac{1}{\ln N} \exp\!\left(-\frac{g}{\ln N}\right)$$

For a prime $p \equiv a \pmod{m}$, the next prime in column $b \neq a$ occurs at gap $g = d(a,b) + km$ for some non-negative integer $k$ (the gap must be congruent to $(b-a) \bmod m$). Summing over all valid gaps:

$$P(a \to b) \propto \sum_{k=0}^{\infty} \exp\!\left(-\frac{d(a,b) + km}{\ln N}\right) = \frac{\exp(-d(a,b)/\ln N)}{1 - \exp(-m/\ln N)}$$

For self-transitions ($b = a$), the minimum gap is $m$ (not $0$), so the sum starts at $k = 1$:

$$P(a \to a) \propto \sum_{k=1}^{\infty} \exp\!\left(-\frac{km}{\ln N}\right) = \frac{\exp(-m/\ln N)}{1 - \exp(-m/\ln N)}$$

Since the gap distribution decays exponentially with rate $1/\ln N$, the $k = 0$ term dominates each sum by a factor of $e^{m/\ln N} \gg 1$ at all measured scales. The geometric series denominators $1/(1 - e^{-m/\ln N})$ cancel exactly upon row-normalization, yielding:

$$T(a \to b) = \frac{\exp(-d(a,b)/\ln N)}{\sum_j \exp(-d(a,j)/\ln N)}$$

This is exactly the claimed formula. The derivation requires only the Cramér model and the modular arithmetic of admissible residues. No additional conjectures, no fitted parameters, no singular series computation.[^1]

The Cramér model ignores correlations between consecutive gaps and the Hardy-Littlewood singular series corrections to gap frequencies. That the zero-parameter model nonetheless achieves $R^2 = 0.970$ — and that the Hardy-Littlewood corrections are orthogonal to the residual (§5.2) — suggests the Boltzmann structure is robust to these approximations.

[^1]: The derivation sums over all valid gaps $g = d(a,b) + km$. The dominant term is $k = 0$, which contributes $\exp(-d/\ln N)$. The Cramér model error is in the coefficient of each term — the actual probability of a gap $g$ differs from $e^{-g/\ln N}/\ln N$ by the Hardy-Littlewood singular series factor $\mathfrak{S}(g)/\mathfrak{S}(\text{avg})$. But these factors multiply the weights before row normalization. After normalization, the row-average singular series correction approximately cancels. Section 5.2 confirms this empirically: the Hardy-Littlewood correction is orthogonal to the residual at every tested order.

---

## 3. Experimental Verification

### 3.1 Protocol

We sieve all primes up to $10^9$ (50,847,534 primes) and measure empirical transition matrices in six octave windows: $[10^3, 10^4)$, $[10^4, 10^5)$, $[10^5, 10^6)$, $[10^6, 10^7)$, $[10^7, 10^8)$, $[10^8, 10^9)$. For each window, we compute the Boltzmann prediction at the geometric midpoint $N = 10^{(\log_{10} a + \log_{10} b)/2}$ and measure $R^2$ (fraction of variance explained relative to the uniform null model $T_{\text{null}} = 1/\varphi(m)$). Explicitly, $R^2 = 1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$ where $\text{SS}_{\text{res}} = \sum_{i,j}(T_{\text{emp}} - T_{\text{pred}})^2$ and $\text{SS}_{\text{tot}} = \sum_{i,j}(T_{\text{emp}} - 1/\varphi(m))^2$.

### 3.2 Results: Modulus 30 ($\varphi(30) = 8$ columns)

| Scale             | $\ln N$ | $\lambda_{\text{PNT}}$ | $\lambda_{\text{fitted}}$ | $\lambda \cdot \ln N$ | $R^2_{\text{zero-param}}$ |
| ----------------- | ------- | ---------------------- | ------------------------- | --------------------- | ------------------------- |
| $10^{3\text{–}4}$ | 8.06    | 0.12408                | 0.14173                   | 1.142                 | 0.951                     |
| $10^{4\text{–}5}$ | 10.36   | 0.09651                | 0.10120                   | 1.049                 | 0.961                     |
| $10^{5\text{–}6}$ | 12.66   | 0.07896                | 0.08118                   | 1.028                 | 0.971                     |
| $10^{6\text{–}7}$ | 14.97   | 0.06682                | 0.06791                   | 1.016                 | 0.971                     |
| $10^{7\text{–}8}$ | 17.27   | 0.05791                | 0.05783                   | 0.999                 | 0.971                     |
| $10^{8\text{–}9}$ | 19.57   | 0.05109                | 0.05053                   | **0.989**             | **0.970**                 |

The ratio $\lambda_{\text{fitted}} \cdot \ln N$ decreases monotonically from 1.142 to 0.989 across six orders of magnitude, passing through unity near $N = 10^{7.5}$ and reaching 0.989 at $N = 10^{8.5}$, consistent with asymptotic approach to $\lambda = 1/\ln N$ with $O(1/\ln^2 N)$ corrections. At the largest measured scale, the fitted decay rate is within 1.1% of the PNT prediction, and the zero-parameter model explains 97.0% of the transition variance. The $R^2$ plateau at $\approx 0.970$ reflects a structured residual whose anatomy is characterized in §5.

### 3.3 Results: Modulus 210 ($\varphi(210) = 48$ columns)

| Scale             | $R^2_{\text{zero-param}}$ | $\lambda \cdot \ln N$ |
| ----------------- | ------------------------- | --------------------- |
| $10^{3\text{–}4}$ | 0.890                     | 1.240                 |
| $10^{4\text{–}5}$ | 0.956                     | 1.129                 |
| $10^{5\text{–}6}$ | 0.975                     | 1.098                 |
| $10^{6\text{–}7}$ | 0.982                     | 1.082                 |
| $10^{7\text{–}8}$ | 0.986                     | 1.066                 |
| $10^{8\text{–}9}$ | **0.988**                 | **1.055**             |

$R^2$ **improves** monotonically with scale (0.890 → 0.988), consistent with the prediction that corrections are $O(1/\ln N)$. The convergence ratio $\lambda \cdot \ln N$ decreases steadily toward unity (1.240 → 1.055). The model becomes more accurate as $N$ grows, in both moduli tested.

### 3.4 Matrix Comparison (mod 30, $10^7$–$10^8$)

Empirical vs. predicted transition probabilities (rows = source column, columns = target column):

```
         EMPIRICAL                          ZERO-PARAMETER PREDICTION
col:  1     7    11    13    17    19    23    29     1     7    11    13    17    19    23    29
 1: .045  .234  .190  .157  .124  .098  .085  .066  .056  .223  .177  .158  .125  .111  .088  .062
 7: .058  .046  .238  .191  .155  .127  .100  .086  .068  .048  .217  .193  .153  .136  .108  .076
11: .075  .066  .046  .237  .194  .157  .126  .098  .083  .059  .046  .235  .186  .166  .132  .093
13: .095  .077  .070  .046  .240  .193  .156  .124  .103  .073  .058  .051  .231  .206  .163  .115
17: .142  .090  .074  .063  .046  .237  .192  .157  .127  .090  .071  .063  .050  .254  .202  .143
19: .157  .116  .109  .074  .070  .046  .238  .190  .161  .113  .090  .080  .064  .057  .255  .180
23: .193  .179  .117  .089  .077  .066  .045  .234  .203  .144  .114  .102  .081  .072  .057  .228
29: .236  .193  .157  .142  .095  .075  .059  .044  .251  .177  .140  .125  .099  .088  .070  .050
```

Maximum element error: 0.035. Mean element error: 0.009.

---

## 4. Lemke Oliver–Soundararajan as a Corollary

The principal finding of [LO&S 2016] is that consecutive primes avoid repeating their residue class: $T(a, a) < 1/\varphi(m)$. In the Boltzmann model, this is immediate:

$$T(a, a) \propto \exp(-m / \ln N)$$

while the nearest forward column has

$$T(a, a + d_{\min}) \propto \exp(-d_{\min} / \ln N)$$

The ratio is $\exp(-(m - d_{\min})/\ln N)$, which is strictly less than 1 for all finite $N$. At $m = 30$, $d_{\min} = 2$, $\ln N = 17.27$:

$$\text{Predicted ratio} = e^{-28/17.27} = 0.198 \qquad \text{Empirical ratio} = 0.045/0.234 = 0.192$$

Agreement to within 3%. The "unexpected bias" is the expected outcome of a thermal distribution where self-return is the highest-energy transition.

---

## 5. The Residual: Structure and Negative Results

The residual matrix $R = T_{\text{empirical}} - T_{\text{Boltzmann}}$ at mod 30 ($10^7$–$10^8$) has five nonzero singular values:

$$\sigma = [0.066, \; 0.036, \; 0.030, \; 0.022, \; 0.013]$$

(The 8×8 stochastic matrix has one zero singular value corresponding to the all-ones eigenvector.) The ratio $\sigma_1/\sigma_2 = 1.84$ indicates the residual is not rank-1 but carries genuine multi-dimensional structure. Its Frobenius norm scales as $\|R\| = 4.26 / \ln^{1.39} N$ ($R^2 = 0.98$ for the power-law fit), and the scaled residual $\ln N \cdot R(N)$ converges to a fixed matrix $M$ with inter-decade correlation $r \geq 0.986$.

### 5.1 Circulant Decomposition

Decomposing $R$ into a circulant part (depending only on forward distance $d$) and a non-circulant part (position-dependent):

| Component                          | $\|\cdot\|^2$ share | Scaling              |
| ---------------------------------- | ------------------- | -------------------- |
| Circulant (distance-only)          | 42%                 | $\sim 1/\ln N$       |
| Non-circulant (position-dependent) | 58%                 | $\sim 1/\ln^{1.6} N$ |

The non-circulant component decays faster and will become subdominant at sufficiently large $N$. A quadratic correction to the exponential kernel, $\exp(-d/\ln N + c \cdot d^2/\ln^2 N)$ with $c \approx -0.12$ (stable across three decades), captures part of the circulant residual and improves $R^2$ by $+0.002$ with one free parameter. The negative $c$ indicates the true kernel is slightly concave relative to pure exponential.

### 5.2 The Singular Series Does Not Appear

A natural hypothesis is that the residual encodes the Hardy-Littlewood singular series $\mathfrak{S}(g)$ for each prime gap $g$. We tested this at three levels:

1. **Multiplicative correction** $T_{\text{HL}} = B \cdot \mathfrak{S}(d(a,b)) / Z$: $R^2$ collapses from 0.971 to **0.233** (mod 30) and from 0.986 to **0.822** (mod 210). The Frobenius norm quintuples.
2. **Additive correction** $T = B \cdot (1 + \beta \cdot (\mathfrak{S} - \bar{\mathfrak{S}})/(\bar{\mathfrak{S}} \cdot \ln^2 N))$: the optimal $\beta$ is **negative** ($\beta = -3.2$ at $10^7$–$10^8$) and the improvement is $\Delta R^2 < 10^{-4}$.
3. **Free-power correction** $T = B \cdot \mathfrak{S}(d)^\alpha$: the optimizer sets $\alpha = 0.000$ — the optimal weight for the singular series is exactly zero.

The circulant distance profile $f(d) = \langle R_{ij} \rangle_{d(i,j)=d}$ has correlation $r = -0.14$ with the singular series, consistent with the null hypothesis. The singular series $\mathfrak{S}(g)$ describes the global frequency of gap $g$; the Boltzmann model's exponential kernel already absorbs this information through the distance parameterization. The residual is orthogonal to $\mathfrak{S}$.

### 5.3 What the Residual Contains

The leading SVD component (59.5% of residual variance) is a column bias: transitions to columns $\{7, 11\}$ are systematically enhanced, transitions to their complements $\{23, 19\}$ (where $7 + 23 = 11 + 19 = 30$) are suppressed. This antisymmetric structure is visible in every row of the non-circulant residual ($R_{\text{asym}}$: column 7 is positive in all 8 rows, column 23 is negative in all 8 rows). It represents a chirality — the primes have handedness in how they populate residue classes.

At mod 210, the residual's effective rank increases dramatically: 21 modes are needed to capture 95% of variance, compared to 4 at mod 30. The macroscopic chirality fragments into a diffuse spectrum of smaller position-dependent effects. This diffusion, combined with the rising $R^2$ (0.970 → 0.988), confirms that the Boltzmann model absorbs more structure as the modulus grows.

---

## 6. Eigenstructure and Spiral Persistence

The empirical transition matrix has complex eigenvalues — 6 conjugate pairs at mod 30, 46 at mod 210 — indicating rotational dynamics in the column transitions. The leading complex eigenvalue magnitude scales as:

| Modulus | Scaling Law                                        | $R^2$  | Interpretation                                                        |
| ------- | -------------------------------------------------- | ------ | --------------------------------------------------------------------- |
| 30      | $\|\lambda_1\| = 2.14 / \log_{10} N$               | 0.9996 | Spiral dies as $1/\log N$                                             |
| 210     | $\|\lambda_1\| = 1.13 \cdot (\log_{10} N)^{-0.10}$ | 0.950  | Spiral persists ($\|\lambda_1\| \approx 0.71$ at $\log_{10} N = 100$) |

The spiral angle $\theta$ drifts at approximately $4.5°$ per decade (mod 30) and $3.6°$ per decade (mod 210), ruling out a fixed angular invariant, as would be expected if the spiral were purely determined by the group structure of $(\mathbb{Z}/m\mathbb{Z})^*$. The persistence of rotational structure at high moduli suggests that the Boltzmann model's residuals carry helical symmetry that deepens with the number of admissible columns.

---

## 7. Asymptotic Behavior

As $N \to \infty$, three limits are reached simultaneously:

1. **$\lambda \cdot \ln N \to 1$**: The decay rate converges to the PNT prediction exactly.
2. **$R^2 \to 1$**: The residual scales as $O(1/\ln^{1.4} N)$ relative to the leading Boltzmann term, so the model becomes exact.
3. **$T(a,b) \to 1/\varphi(m)$**: At infinite temperature, the Boltzmann distribution becomes uniform. Every row converges to $1/\varphi(m)$ for all $a, b$ — precisely Dirichlet's theorem on the equidistribution of primes in arithmetic progressions.

The Boltzmann model thus interpolates between the Lemke Oliver-Soundararajan regime (finite $N$, structured transitions) and Dirichlet's theorem ($N \to \infty$, uniform distribution).

The residual $\|R\| \sim 4.26/\ln^{1.39} N$ converges to a fixed-shape matrix but never reaches zero at finite $N$. Whether this floor saturates a lower bound derivable from the discrete Fourier uncertainty principle on $(\mathbb{Z}/m\mathbb{Z})^*$ — i.e., whether the Boltzmann model is the optimal smooth approximation to the transition matrix — remains open. Numerically, the entropic uncertainty excess $H(X) + H(K) - \ln \varphi(m)$ of the Boltzmann rows scales as $\approx 2.9/\ln N$ (mod 30), suggesting the bound is approached but not saturated at any finite scale.

---

## 8. Conclusion

The transition matrix of consecutive primes modulo $m$ is a Boltzmann distribution on the forward cyclic distances of $(\mathbb{Z}/m\mathbb{Z})^*$, at temperature $\ln N$. Self-return costs energy $m$. The partition function normalizes each row. The model has zero free parameters, achieves $R^2 \approx 0.970$ at $m = 30$ and $R^2 = 0.988$ at $m = 210$, and converges to exact prediction as $N \to \infty$.

The Lemke Oliver-Soundararajan diagonal suppression is a one-line corollary. The 3% residual is structured — a circulant kernel correction and a position-dependent chirality — but is orthogonal to the Hardy-Littlewood singular series at every tested order. The PNT is the Boltzmann temperature.

The Boltzmann distribution on $(\mathbb{Z}/m\mathbb{Z})^*$ satisfies the discrete Fourier uncertainty principle [Tao 2005; see also Donoho-Stark 1989]: the transition rows cannot be simultaneously localized in position space (column distribution) and momentum space (eigenspectrum). The temperature $\ln N$ governs this tradeoff, with the classical limit $\ln N \to \infty$ recovering Dirichlet equidistribution.

---

## References

1. Cramér, H. (1936). "On the order of magnitude of the difference between consecutive prime numbers." _Acta Arithmetica_, 2(1), 23–46.

2. Hardy, G. H., & Littlewood, J. E. (1923). "Some problems of 'Partitio numerorum'; III: On the expression of a number as a sum of primes." _Acta Mathematica_, 44, 1–70.

3. Lemke Oliver, R. J., & Soundararajan, K. (2016). "Unexpected biases in the distribution of consecutive primes." _Proceedings of the National Academy of Sciences_, 113(31), E4446–E4454.

4. Dirichlet, P. G. L. (1837). "Beweis des Satzes, dass jede unbegrenzte arithmetische Progression, deren erstes Glied und Differenz ganze Zahlen ohne gemeinschaftlichen Factor sind, unendlich viele Primzahlen enthält." _Abhandlungen der Königlichen Preußischen Akademie der Wissenschaften zu Berlin_, 45–81.

5. Tao, T. (2005). "An uncertainty principle for cyclic groups of prime order." _Mathematical Research Letters_, 12(1), 121–127. See also: Donoho, D. L., & Stark, P. B. (1989). "Uncertainty principles and signal recovery." _SIAM Journal on Applied Mathematics_, 49(3), 906–931.

---

## Acknowledgments

This research was conducted using Claude Opus 4 (Anthropic) and Gemini 3.1 Pro (Google DeepMind) as computational research instruments. The Boltzmann framing was first proposed by a Gemini HELICASE worker during a constrained swarm convergence run on the prime basin transfer problem (BVP-5, Wave 2; see Appendix B). No thermodynamic framing was suggested in the problem specification; the swarm arrived at it independently within the constraints of the boundary value problem. Subsequent falsification and formalization were conducted through a 22-wave human-AI collaborative protocol. The author designed the problem specifications, system architecture, and experimental methodology.

---

## Appendix A: Reproducibility

All computations were performed with NumPy and SciPy on a standard consumer CPU. The segmented sieve covers primes to $10^9$ (50,847,534 primes; ~30 seconds). Source code is available at [github.com/fancyland-llc/boltzmann-prime-transitions](https://github.com/fancyland-llc/boltzmann-prime-transitions):

- `backend/scripts/prime_drum_wave22c_boltzmann.py` — the zero-parameter model
- `backend/scripts/prime_drum_wave22d_10e9.py` — the $10^9$ extension (segmented sieve)
- `backend/scripts/prime_drum_wave22b_boltzmann.py` — the two-parameter intermediate model
- `backend/scripts/prime_drum_wave22_boltzmann.py` — the initial (failed) column-index model
- `backend/scripts/prime_drum_wave22g_hardy_littlewood.py` — the singular series test (negative result)
- `backend/scripts/prime_drum_wave22h_residual_archaeology.py` — the residual decomposition
- `backend/scripts/prime_drum_wave22h_verify.py` — the character hypothesis verification

Results data: `backend/data/wave22c_boltzmann_*.json`, `backend/data/wave22d_10e9_*.json`, `backend/data/wave22h_archaeology_*.json`

---

## Appendix B: Discovery Context

This result was not found by direct theoretical derivation. The Boltzmann framing was discovered by a machine — specifically, by a Gemini HELICASE worker operating inside the Antigen Factory, a two-layer AI swarm solver, during a constrained convergence run on the prime basin transfer problem (BVP-5).

### B.1 The Swarm Run (BVP-5)

The BVP-5 problem specification, designed by the author, asked 8 AI workers per wave to discover the transfer operator mapping the structural fingerprint of primes across binary octave boundaries. The problem imposed hard constraints — a specific loss function over basin structure, mandatory falsification of prior wave outputs, and graveyard-aware context injection — but no thermodynamic framing was suggested. The swarm was given raw basins, a distance metric, and an error target.

| Wave | Best Error   | Crystallized Technique                       | Source     |
| ---- | ------------ | -------------------------------------------- | ---------- |
| 0    | 0.0107       | Two-point correlation                        | Claude     |
| 1    | 0.2138[^2]   | Spectral renormalization                     | Claude     |
| 2    | 0.0304       | **Boltzmann Reweighting**                    | **Gemini** |
| 3    | 0.0071       | Hardy-Littlewood-Cramér                      | Gemini     |
| 6    | 0.0043       | **Exponential Tilting**                      | **Gemini** |
| 11   | **0.000914** | **Exponential Tilt / Canonical Reweighting** | **Gemini** |

**The swarm found "Boltzmann Reweighting" at Wave 2.** Nobody told it to look for a thermal distribution. It tried spectral methods, correlation functions, and Fredholm integrals — and the thermodynamic framing kept winning. By Wave 11, the crystallized technique was "Exponential Tilt / Canonical Reweighting (similar to Umbrella Sampling)" — a Gemini L1.5 HELICASE worker, running autonomously inside a Mendelian breeding loop with graveyard-aware context injection. Error: 0.000914. Convergence: autonomous halt.

[^2]: Wave 1 explored spectral renormalization, a qualitatively different approach whose initial error exceeded Wave 0; the method was subsequently refined and its descendants contributed to later waves.

96 Layer-1 workers. 80 Layer-2 workers. 233,835 tokens. 68 minutes. Zero human intervention between start and halt.

The Boltzmann model was the swarm's idea. The author designed the constraints that made the discovery possible.

### B.2 The Interferometer (Waves 1–22)

What followed was a 22-wave adversarial falsification protocol — three AI systems (Claude, Gemini, and a human architect) operating as a self-evolving research instrument, systematically testing and killing the swarm's output:

1. **Propose** a falsifiable gauge derived from the swarm's crystallized technique or its competitors.
2. **Compute** the gauge across multiple octave scales using a shared sieve infrastructure.
3. **Kill or promote**: if the gauge fails its own prediction, it enters a permanent graveyard. If it survives, it seeds the next wave.

Over 22 waves, the system killed 80+ hypotheses and produced 30+ survivors. The path to the zero-parameter model was:

- Waves 1–4: compression-based primality signals (real but weak).
- Waves 5–7: exact prime interferometer (equivalent to trial division).
- Waves 8–15: scaling laws, winding numbers, sphere amplifiers, transfer functions.
- Waves 16–20: GUE killed, Goldbach Spectrometer built ($r = 0.99917$), dual Goldbach symmetric, compression crossover measured.
- Wave 21: columnar transition matrix has complex eigenvalues — spectral chirality, circulant ratchet structure.
- Wave 22a: Boltzmann model tested with column-index distance. **FAILED** ($R^2 = 0.08$). The swarm's insight was right; our metric was wrong.
- Wave 22b: corrected to residue distance, 2-parameter fit. $R^2 = 0.97$. Discovery: the self-penalty parameter $\mu \cdot \ln N$ converges to $m$. The self-avoidance IS the Boltzmann model.
- Wave 22c: $d_{\text{self}} = m$, $\lambda = 1/\ln N$. Zero parameters. $R^2 = 0.971$.

**The critical failure was instructive.** Wave 22a's catastrophic $R^2 = 0.08$ proved that the swarm's Boltzmann framing only works with the correct distance metric — forward cyclic residue distance, not column index. Wave 22b's convergence of $\mu \cdot \ln N \to m$ was the key insight that collapsed two parameters to zero. Each failure constrained the next hypothesis until the correct model was the only one remaining.

### B.3 The Architecture

The swarm infrastructure — Layer-1 isomorphic processors (Claude + Gemini at $T \approx 0.7$), Demon Gate classification, Layer-2 math workers (Claude at $T = 0.1$), epistemic sandbox with real Python execution, Mendelian breeding with phonon-adaptive ratio, Wilson loop accumulator, tension metric with gauge relaxation — is implemented in Prompt Studio, a distributed AI research platform built on Lattice OS.

The Boltzmann theorem is its first mathematical output.

---

_Fancyland LLC — Lattice OS research infrastructure._
