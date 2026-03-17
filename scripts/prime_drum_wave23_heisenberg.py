"""
WAVE 23 — THE HEISENBERG TEST
Does the Boltzmann prime transition matrix saturate the entropic uncertainty bound?

The entropic uncertainty principle on finite abelian groups (Maassen-Uffink / Hirschman-Beckner):
    H(X) + H(K) >= ln(phi(m))
where H(X) = Shannon entropy of the position distribution (transition row),
      H(K) = Shannon entropy of the momentum distribution (character transform of sqrt(row)),
      phi(m) = number of admissible columns.

This bound is satisfied by ANY probability distribution. The question is:
1. How CLOSE to the bound are the Boltzmann prime transitions? (saturation)
2. Is this closeness SPECIAL compared to generic distributions? (non-triviality)
3. Does the empirical data match the Boltzmann prediction? (verification)

CRITICAL DISTINCTION:
- "Primes satisfy the uncertainty principle" = tautology (everything does)
- "Primes nearly SATURATE the uncertainty bound" = potentially profound
"""

import numpy as np
import json
import os

# ============================================================
# 1. SETUP: MOD 30 PRIME PERIODIC TABLE
# ============================================================
m = 30
cols = [1, 7, 11, 13, 17, 19, 23, 29]
phi = len(cols)
PLANCK = np.log(phi)  # ln(8) ≈ 2.0794 — the absolute floor

print("=" * 72)
print("  WAVE 23: THE HEISENBERG TEST")
print("  Entropic Uncertainty of Prime Transitions mod 30")
print("=" * 72)
print(f"\n  Modulus m = {m}, φ(m) = {phi}")
print(f"  Heisenberg bound: ln({phi}) = {PLANCK:.6f} nats")

# ============================================================
# 2. CHARACTER TABLE OF (Z/30Z)* ≅ Z/2 × Z/4
# ============================================================
# Generators: 11 (order 2), 17 (order 4)
# Every element 11^a · 17^b mod 30, a∈{0,1}, b∈{0,1,2,3}
# Characters: χ_{j,k}(11^a · 17^b) = (-1)^{aj} · i^{bk}

col_to_ab = {}
for a in range(2):
    for b in range(4):
        val = pow(11, a, m) * pow(17, b, m) % m
        col_to_ab[val] = (a, b)

char_table = np.zeros((phi, phi), dtype=complex)
char_labels = []
idx = 0
for j in range(2):
    for k in range(4):
        for ci, c in enumerate(cols):
            a, b = col_to_ab[c]
            char_table[idx, ci] = ((-1) ** (a * j)) * (1j ** (b * k))
        char_labels.append(f"χ({j},{k})")
        idx += 1

# Verify orthogonality
ortho = char_table @ char_table.conj().T / phi
ortho_err = np.max(np.abs(ortho - np.eye(phi)))
print(f"\n  Character table orthogonality check: max off-diagonal = {ortho_err:.2e}")
assert ortho_err < 1e-10, "Character table is NOT orthogonal!"
print("  ✓ Character table verified.")

# ============================================================
# 3. CORE FUNCTIONS
# ============================================================

def forward_dist(a, b, m=30):
    d = (b - a) % m
    return m if d == 0 else d


def boltzmann_matrix(ln_N):
    """Zero-parameter Boltzmann transition matrix at temperature ln(N)."""
    T_mat = np.zeros((phi, phi))
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            T_mat[i, j] = np.exp(-forward_dist(a, b) / ln_N)
        T_mat[i] /= T_mat[i].sum()
    return T_mat


def entropic_uncertainty(prob_row, char_table, phi):
    """
    Given a probability distribution p on φ(m) elements:
    1. Compute position entropy H(X) = -Σ p·ln(p)
    2. Take Born amplitude ψ = √p
    3. Transform to momentum: ψ̂(χ) = Σ ψ(x)·χ̄(x) / √φ
    4. Compute momentum distribution P_K = |ψ̂|²
    5. Compute momentum entropy H(K) = -Σ P_K·ln(P_K)
    6. Return H(X), H(K), excess = H(X)+H(K)-ln(φ)
    """
    p = prob_row / prob_row.sum()
    
    # Position entropy
    hx = -np.sum(p * np.log(p + 1e-30))
    
    # Born amplitude
    psi = np.sqrt(p)
    
    # Character (Fourier) transform
    psi_k = char_table @ psi / np.sqrt(phi)
    p_k = np.abs(psi_k) ** 2
    
    # Momentum entropy
    hk = -np.sum(p_k * np.log(p_k + 1e-30))
    
    excess = hx + hk - np.log(phi)
    return hx, hk, excess


def row_averaged_uncertainty(T_mat, char_table, phi):
    """Return mean H(X), H(K), excess across all rows of a transition matrix."""
    hxs, hks, exs = [], [], []
    for i in range(phi):
        hx, hk, ex = entropic_uncertainty(T_mat[i], char_table, phi)
        hxs.append(hx)
        hks.append(hk)
        exs.append(ex)
    return np.mean(hxs), np.mean(hks), np.mean(exs)


# ============================================================
# 4. THEORETICAL BOLTZMANN TRAJECTORY
# ============================================================
print("\n" + "=" * 72)
print("  SECTION A: BOLTZMANN TRAJECTORY THROUGH PHASE SPACE")
print("=" * 72)

ln_Ns = np.logspace(np.log10(1.5), 2.5, 500)
theory_hx, theory_hk, theory_excess = [], [], []

for ln_N in ln_Ns:
    T_mat = boltzmann_matrix(ln_N)
    mhx, mhk, mex = row_averaged_uncertainty(T_mat, char_table, phi)
    theory_hx.append(mhx)
    theory_hk.append(mhk)
    theory_excess.append(mex)

theory_excess = np.array(theory_excess)
theory_hx = np.array(theory_hx)
theory_hk = np.array(theory_hk)

peak_idx = np.argmax(theory_excess)
print(f"\n  Peak excess over Heisenberg bound:")
print(f"    {theory_excess[peak_idx]:.6f} nats at ln(N) = {ln_Ns[peak_idx]:.2f}")
print(f"    = {theory_excess[peak_idx]/PLANCK*100:.3f}% of the bound")
print(f"\n  At the empirical scales:")

scales = [
    ("10^3–4", 8.06),
    ("10^4–5", 10.36),
    ("10^5–6", 12.66),
    ("10^6–7", 14.97),
    ("10^7–8", 17.27),
    ("10^8–9", 19.57),
]

print(f"\n  {'Scale':<12} {'ln(N)':>6} {'H(X)':>8} {'H(K)':>8} {'Sum':>8} {'Excess':>10} {'% bound':>8}")
print("  " + "-" * 62)
for name, ln_N in scales:
    T_mat = boltzmann_matrix(ln_N)
    mhx, mhk, mex = row_averaged_uncertainty(T_mat, char_table, phi)
    print(f"  {name:<12} {ln_N:>6.2f} {mhx:>8.4f} {mhk:>8.4f} {mhx+mhk:>8.4f} {mex:>10.6f} {mex/PLANCK*100:>7.3f}%")


# ============================================================
# 5. EMPIRICAL DATA (MOD 30, 10^7–10^8)
# ============================================================
print("\n" + "=" * 72)
print("  SECTION B: EMPIRICAL PRIME DATA vs BOLTZMANN PREDICTION")
print("=" * 72)

T_emp = np.array([
    [.045, .234, .190, .157, .124, .098, .085, .066],   # from col 1
    [.058, .046, .238, .191, .155, .127, .100, .086],   # from col 7
    [.075, .066, .046, .237, .194, .157, .126, .098],   # from col 11
    [.095, .077, .070, .046, .240, .193, .156, .124],   # from col 13
    [.142, .090, .074, .063, .046, .237, .192, .157],   # from col 17
    [.157, .116, .109, .074, .070, .046, .238, .190],   # from col 19
    [.193, .179, .117, .089, .077, .066, .045, .234],   # from col 23
    [.236, .193, .157, .142, .095, .075, .059, .044],   # from col 29
])
for i in range(phi):
    T_emp[i] /= T_emp[i].sum()

T_boltz = boltzmann_matrix(17.27)

print(f"\n  {'Row':>6} {'H(X)_emp':>10} {'H(K)_emp':>10} {'Exc_emp':>10} {'H(X)_bol':>10} {'H(K)_bol':>10} {'Exc_bol':>10}")
print("  " + "-" * 66)

emp_exs, boltz_exs = [], []
for i in range(phi):
    hx_e, hk_e, ex_e = entropic_uncertainty(T_emp[i], char_table, phi)
    hx_b, hk_b, ex_b = entropic_uncertainty(T_boltz[i], char_table, phi)
    emp_exs.append(ex_e)
    boltz_exs.append(ex_b)
    print(f"  col {cols[i]:>2}: {hx_e:>10.5f} {hk_e:>10.5f} {ex_e:>10.6f} {hx_b:>10.5f} {hk_b:>10.5f} {ex_b:>10.6f}")

emp_mean = np.mean(emp_exs)
boltz_mean = np.mean(boltz_exs)
print(f"\n  Mean empirical excess:  {emp_mean:.6f} nats ({emp_mean/PLANCK*100:.3f}% of bound)")
print(f"  Mean Boltzmann excess:  {boltz_mean:.6f} nats ({boltz_mean/PLANCK*100:.3f}% of bound)")
print(f"  Empirical − Boltzmann:  {emp_mean - boltz_mean:.6f} nats")


# ============================================================
# 6. NULL COMPARISON: RANDOM DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 72)
print("  SECTION C: NULL COMPARISON — IS BOLTZMANN SPECIAL?")
print("=" * 72)

np.random.seed(42)
n_trials = 50000

# Test 1: Uniform random distributions (Dirichlet α=1)
rand_excesses = []
for _ in range(n_trials):
    p = np.random.dirichlet(np.ones(phi))
    _, _, ex = entropic_uncertainty(p, char_table, phi)
    rand_excesses.append(ex)
rand_excesses = np.array(rand_excesses)

print(f"\n  Test 1: Uniform random distributions on {phi} elements (n={n_trials})")
print(f"    Mean excess:   {rand_excesses.mean():.6f} ({rand_excesses.mean()/PLANCK*100:.3f}% of bound)")
print(f"    Std:           {rand_excesses.std():.6f}")
print(f"    5th pctile:    {np.percentile(rand_excesses, 5):.6f}")
print(f"    95th pctile:   {np.percentile(rand_excesses, 95):.6f}")
print(f"    Min:           {rand_excesses.min():.6f}")
print(f"    Boltzmann:     {boltz_mean:.6f}")
print(f"    Boltzmann pctile: {np.mean(rand_excesses <= boltz_mean)*100:.1f}%")

# Test 2: Random distributions with MATCHED H(X)
# Among distributions as spread as the Boltzmann row, does Boltzmann minimize H(K)?
target_hx = np.mean([entropic_uncertainty(T_boltz[i], char_table, phi)[0] for i in range(phi)])
print(f"\n  Test 2: Random dists with matched position entropy H(X) ≈ {target_hx:.3f}")

matched_hks = []
matched_exs = []
attempts = 0
for _ in range(200000):
    p = np.random.dirichlet(np.ones(phi))
    hx_test = -np.sum(p * np.log(p + 1e-30))
    if abs(hx_test - target_hx) < 0.03:
        _, hk, ex = entropic_uncertainty(p, char_table, phi)
        matched_hks.append(hk)
        matched_exs.append(ex)
        attempts += 1

if attempts > 0:
    matched_hks = np.array(matched_hks)
    matched_exs = np.array(matched_exs)
    boltz_hk_mean = np.mean([entropic_uncertainty(T_boltz[i], char_table, phi)[1] for i in range(phi)])
    
    print(f"    Matched samples: {attempts}")
    print(f"    Mean H(K) [random]:    {matched_hks.mean():.5f}")
    print(f"    Mean H(K) [Boltzmann]: {boltz_hk_mean:.5f}")
    print(f"    Boltzmann H(K) pctile: {np.mean(matched_hks <= boltz_hk_mean)*100:.1f}%")
    print(f"    Mean excess [random]:    {matched_exs.mean():.6f}")
    print(f"    Mean excess [Boltzmann]: {boltz_mean:.6f}")
    print(f"    Boltzmann excess pctile: {np.mean(matched_exs <= boltz_mean)*100:.1f}%")

# Test 3: Random exponential (Boltzmann-shaped) distributions with random distances
print(f"\n  Test 3: Random Boltzmann-shaped distributions (exponential on shuffled distances)")
boltz_shaped_excesses = []
for _ in range(n_trials):
    # Random non-repeating distances from the same set as mod 30
    dists = np.random.choice(range(1, m + 1), phi, replace=False).astype(float)
    T = 17.27  # same temperature
    p = np.exp(-dists / T)
    p /= p.sum()
    _, _, ex = entropic_uncertainty(p, char_table, phi)
    boltz_shaped_excesses.append(ex)

boltz_shaped_excesses = np.array(boltz_shaped_excesses)
print(f"    Mean excess [random Boltzmann-shaped]: {boltz_shaped_excesses.mean():.6f} ({boltz_shaped_excesses.mean()/PLANCK*100:.3f}%)")
print(f"    Std:      {boltz_shaped_excesses.std():.6f}")
print(f"    Prime Boltzmann:  {boltz_mean:.6f}")
print(f"    Percentile: {np.mean(boltz_shaped_excesses <= boltz_mean)*100:.1f}%")


# ============================================================
# 7. SATURATION TEST: HOW TIGHT IS THE BOUND?
# ============================================================
print("\n" + "=" * 72)
print("  SECTION D: THE SATURATION QUESTION")
print("=" * 72)

# The bound is tight at T→0 (delta functions) and T→∞ (uniform)
# At intermediate T, there's a gap. How big is it?

print(f"\n  Theoretical maximum excess along Boltzmann trajectory:")
print(f"    Peak: {theory_excess[peak_idx]:.6f} nats at ln(N) = {ln_Ns[peak_idx]:.2f}")
print(f"    = {theory_excess[peak_idx]/PLANCK*100:.3f}% above the Heisenberg bound")

# Compare: for a GENERIC exponential distribution on [1,m] at T=17.27
# (i.e., not using the specific prime distances)
# What's the theoretical excess?

# The prime-specific distances from column 1
dists_from_1 = [forward_dist(cols[0], b) for b in cols]
print(f"\n  Forward distances from column 1: {dists_from_1}")
print(f"  Distance set: min={min(dists_from_1)}, max={max(dists_from_1)}, mean={np.mean(dists_from_1):.1f}")

# Compare all 8 source columns
print(f"\n  Per-row analysis at ln(N) = 17.27:")
for i in range(phi):
    dists = [forward_dist(cols[i], b) for b in cols]
    hx_e, hk_e, ex_e = entropic_uncertainty(T_emp[i], char_table, phi)
    hx_b, hk_b, ex_b = entropic_uncertainty(T_boltz[i], char_table, phi)
    print(f"    Col {cols[i]:>2}: dists={dists}  Boltz excess={ex_b:.6f}  Emp excess={ex_e:.6f}  Δ={ex_e-ex_b:+.6f}")


# ============================================================
# 8. THE DEFINITIVE TEST: EXCESS vs 1/ln(N) SCALING
# ============================================================
print("\n" + "=" * 72)
print("  SECTION E: SCALING LAW OF THE EXCESS")
print("=" * 72)

# If the Boltzmann model approaches a minimum uncertainty state as N→∞,
# the excess should scale as some power of 1/ln(N).
# Fit: excess = A / ln(N)^α

from scipy.optimize import curve_fit

# Use the theoretical trajectory at large scales
mask = ln_Ns > 5
ln_N_fit = ln_Ns[mask]
exc_fit = theory_excess[mask]

def power_model(x, A, alpha):
    return A / x ** alpha

try:
    popt, pcov = curve_fit(power_model, ln_N_fit, exc_fit, p0=[1.0, 2.0])
    A_fit, alpha_fit = popt
    exc_pred = power_model(ln_N_fit, *popt)
    ss_res = np.sum((exc_fit - exc_pred) ** 2)
    ss_tot = np.sum((exc_fit - np.mean(exc_fit)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"\n  Excess scaling law: excess ≈ {A_fit:.4f} / ln(N)^{alpha_fit:.3f}")
    print(f"  R² = {r2:.6f}")
    print(f"  Interpretation: The gap above the Heisenberg bound closes as 1/ln(N)^{alpha_fit:.2f}")
    
    # Predict when excess < 0.1% of bound
    for target_pct in [1.0, 0.1, 0.01]:
        target_exc = target_pct / 100 * PLANCK
        ln_N_needed = (A_fit / target_exc) ** (1 / alpha_fit)
        N_needed = np.exp(ln_N_needed)
        print(f"  Excess < {target_pct:.2f}% of bound at ln(N) > {ln_N_needed:.1f} (N > 10^{np.log10(N_needed):.1f})")
except Exception as e:
    print(f"  Curve fit failed: {e}")
    alpha_fit = None


# ============================================================
# 9. MOD 210 CHECK
# ============================================================
print("\n" + "=" * 72)
print("  SECTION F: MOD 210 VERIFICATION")
print("=" * 72)

# For mod 210, φ(210) = 48
m210 = 210
cols210 = [r for r in range(1, m210) if np.gcd(r, m210) == 1]
phi210 = len(cols210)
PLANCK210 = np.log(phi210)

print(f"\n  Modulus m = {m210}, φ(m) = {phi210}")
print(f"  Heisenberg bound: ln({phi210}) = {PLANCK210:.6f} nats")

# For mod 210, (Z/210Z)* ≅ Z/2 × Z/2 × Z/2 × Z/2 × Z/4 ≅ (Z/2)^3 × Z/4 ... wait
# Actually (Z/210Z)* ≅ (Z/2Z)* × (Z/3Z)* × (Z/5Z)* × (Z/7Z)* ≅ {1} × Z/2 × Z/4 × Z/6
# φ(210) = 1 × 2 × 4 × 6 = 48
# We can use the DFT directly instead of building the full character table
# The DFT on (Z/mZ)* via its characters is equivalent to the ordinary DFT
# when we properly enumerate the group

# For mod 210, use the general DFT approach
# Map each coprime residue to an index
col210_idx = {c: i for i, c in enumerate(cols210)}

# Build Boltzmann matrix for mod 210
def boltzmann_matrix_210(ln_N):
    T_mat = np.zeros((phi210, phi210))
    for i, a in enumerate(cols210):
        for j, b in enumerate(cols210):
            d = (b - a) % m210
            if d == 0:
                d = m210
            T_mat[i, j] = np.exp(-d / ln_N)
        T_mat[i] /= T_mat[i].sum()
    return T_mat

# For mod 210, build character table via CRT decomposition
# (Z/210Z)* ≅ (Z/2Z)* × (Z/3Z)* × (Z/5Z)* × (Z/7Z)*
# ≅ {1} × Z/2 × Z/4 × Z/6
# Actually: φ(2)=1, φ(3)=2, φ(5)=4, φ(7)=6
# Group structure: {1} × Z_2 × Z_4 × Z_6

# More precisely:
# (Z/3Z)* = <2> ≅ Z/2, generator 2 mod 3
# (Z/5Z)* = <2> ≅ Z/4, generator 2 mod 5 (2^1=2, 2^2=4, 2^3=3, 2^4=1)
# (Z/7Z)* = <3> ≅ Z/6, generator 3 mod 7 (3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1)

# For each coprime r mod 210, find (r mod 3, r mod 5, r mod 7)
# Then map to generator exponents

def discrete_log(val, gen, mod):
    """Find k such that gen^k ≡ val (mod mod)."""
    x = 1
    for k in range(mod):
        if x % mod == val % mod:
            return k
        x = (x * gen) % mod
    return -1

# Map each column to exponent coordinates
col210_to_exp = {}
for c in cols210:
    # r mod 3 in (Z/3Z)*: generator 2
    e3 = discrete_log(c % 3, 2, 3) if c % 3 != 0 else 0
    # r mod 5 in (Z/5Z)*: generator 2
    e5 = discrete_log(c % 5, 2, 5) if c % 5 != 0 else 0
    # r mod 7 in (Z/7Z)*: generator 3
    e7 = discrete_log(c % 7, 3, 7) if c % 7 != 0 else 0
    col210_to_exp[c] = (e3, e5, e7)

# Build character table: χ_{j3,j5,j7}(r) = ω_2^{e3·j3} · ω_4^{e5·j5} · ω_6^{e7·j7}
char_table_210 = np.zeros((phi210, phi210), dtype=complex)
chi_idx = 0
for j3 in range(2):
    for j5 in range(4):
        for j7 in range(6):
            for ci, c in enumerate(cols210):
                e3, e5, e7 = col210_to_exp[c]
                val = np.exp(2j * np.pi * e3 * j3 / 2) * \
                      np.exp(2j * np.pi * e5 * j5 / 4) * \
                      np.exp(2j * np.pi * e7 * j7 / 6)
                char_table_210[chi_idx, ci] = val
            chi_idx += 1

# Verify orthogonality
ortho210 = char_table_210 @ char_table_210.conj().T / phi210
ortho210_err = np.max(np.abs(ortho210 - np.eye(phi210)))
print(f"  Character table orthogonality check: max off-diagonal = {ortho210_err:.2e}")
if ortho210_err > 0.01:
    print("  ⚠ Character table may have issues. Using DFT matrix as fallback.")
    # Fallback: use the DFT on the group directly
    # This is an approximation — the DFT matrix on {0,...,47} instead of the true group
    char_table_210 = np.fft.fft(np.eye(phi210)) / np.sqrt(phi210) * np.sqrt(phi210)
    # Actually use a proper unitary DFT
    F = np.zeros((phi210, phi210), dtype=complex)
    for i in range(phi210):
        for j in range(phi210):
            F[i, j] = np.exp(-2j * np.pi * i * j / phi210)
    char_table_210 = F
    ortho210 = char_table_210 @ char_table_210.conj().T / phi210
    ortho210_err = np.max(np.abs(ortho210 - np.eye(phi210)))
    print(f"  DFT fallback orthogonality: max off-diagonal = {ortho210_err:.2e}")
else:
    print("  ✓ Character table verified for mod 210.")

def entropic_uncertainty_210(prob_row):
    p = prob_row / prob_row.sum()
    hx = -np.sum(p * np.log(p + 1e-30))
    psi = np.sqrt(p)
    psi_k = char_table_210 @ psi / np.sqrt(phi210)
    p_k = np.abs(psi_k) ** 2
    hk = -np.sum(p_k * np.log(p_k + 1e-30))
    return hx, hk, hx + hk - np.log(phi210)

# Test at a few scales
print(f"\n  {'Scale':<12} {'ln(N)':>6} {'Mean H(X)':>10} {'Mean H(K)':>10} {'Mean Excess':>12} {'% bound':>8}")
print("  " + "-" * 60)

for name, ln_N in [("10^5–6", 12.66), ("10^7–8", 17.27), ("10^8–9", 19.57)]:
    T_mat = boltzmann_matrix_210(ln_N)
    hxs, hks, exs = [], [], []
    for i in range(phi210):
        hx, hk, ex = entropic_uncertainty_210(T_mat[i])
        hxs.append(hx)
        hks.append(hk)
        exs.append(ex)
    mhx, mhk, mex = np.mean(hxs), np.mean(hks), np.mean(exs)
    print(f"  {name:<12} {ln_N:>6.2f} {mhx:>10.4f} {mhk:>10.4f} {mex:>12.6f} {mex/PLANCK210*100:>7.3f}%")


# ============================================================
# 10. FINAL VERDICT
# ============================================================
print("\n" + "=" * 72)
print("  FINAL VERDICT")
print("=" * 72)

print(f"\n  Heisenberg bound (mod 30):  ln(8) = {PLANCK:.6f} nats")
print(f"  Peak Boltzmann excess:      {theory_excess[peak_idx]:.6f} nats = {theory_excess[peak_idx]/PLANCK*100:.3f}% above bound")
print(f"  Empirical excess (10^7–8):  {emp_mean:.6f} nats = {emp_mean/PLANCK*100:.3f}% above bound")
print(f"  Random distribution excess: {rand_excesses.mean():.6f} nats = {rand_excesses.mean()/PLANCK*100:.3f}% above bound")

ratio = emp_mean / rand_excesses.mean()
print(f"\n  Boltzmann / Random excess ratio: {ratio:.4f}")

if theory_excess[peak_idx] / PLANCK < 0.03:
    print(f"\n  ★★★ RESULT: The Boltzmann prime transitions stay within 3% of the")
    print(f"       Heisenberg bound at ALL temperatures. These are NEAR-MINIMUM")
    print(f"       UNCERTAINTY STATES. The uncertainty principle is essentially TIGHT.")
elif theory_excess[peak_idx] / PLANCK < 0.10:
    print(f"\n  ★★ RESULT: The Boltzmann prime transitions stay within 10% of the")
    print(f"      Heisenberg bound. The uncertainty principle is APPROXIMATELY tight.")
    print(f"      The primes are significantly closer to the bound than random distributions.")
elif theory_excess[peak_idx] / PLANCK < 0.25:
    print(f"\n  ★ RESULT: The Boltzmann prime transitions stay within 25% of the")
    print(f"     Heisenberg bound. There is meaningful proximity but not saturation.")
else:
    print(f"\n  ○ RESULT: The Boltzmann prime transitions deviate significantly from the")
    print(f"    Heisenberg bound ({theory_excess[peak_idx]/PLANCK*100:.1f}%). The uncertainty principle")
    print(f"    is satisfied but not constraining at these scales.")

# Comparison verdict
pctile = np.mean(rand_excesses <= boltz_mean) * 100
if pctile < 5:
    print(f"\n  The Boltzmann excess is in the {pctile:.1f}th percentile of random distributions.")
    print(f"  → The prime structure produces ANOMALOUSLY LOW excess. This is non-trivial.")
elif pctile < 25:
    print(f"\n  The Boltzmann excess is in the {pctile:.1f}th percentile of random distributions.")
    print(f"  → The prime structure is moderately closer to the bound than random.")
else:
    print(f"\n  The Boltzmann excess is in the {pctile:.1f}th percentile of random distributions.")
    print(f"  → The prime structure is NOT anomalously close to the bound vs. random distributions.")

# Save comprehensive results
results = {
    "mod_30": {
        "planck_limit": float(PLANCK),
        "peak_excess": float(theory_excess[peak_idx]),
        "peak_excess_pct": float(theory_excess[peak_idx] / PLANCK * 100),
        "peak_ln_N": float(ln_Ns[peak_idx]),
        "empirical_mean_excess": float(emp_mean),
        "empirical_mean_excess_pct": float(emp_mean / PLANCK * 100),
        "boltzmann_mean_excess": float(boltz_mean),
        "boltzmann_mean_excess_pct": float(boltz_mean / PLANCK * 100),
        "random_mean_excess": float(rand_excesses.mean()),
        "random_mean_excess_pct": float(rand_excesses.mean() / PLANCK * 100),
        "random_std_excess": float(rand_excesses.std()),
        "boltzmann_percentile_vs_random": float(pctile),
        "scaling_law_alpha": float(alpha_fit) if alpha_fit else None,
    },
}

os.makedirs("backend/data", exist_ok=True)
with open("backend/data/wave23_heisenberg_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to backend/data/wave23_heisenberg_results.json")
print("\n" + "=" * 72)
