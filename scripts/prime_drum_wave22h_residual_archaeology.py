#!/usr/bin/env python3
"""
Wave 22h — Residual Archaeology

The Boltzmann model captures R²=0.971 (mod 30) with zero parameters.
Wave 22g killed the multiplicative HL hypothesis catastrophically (R² → 0.233).

This wave dissects the 3% residual to find what it actually IS.

EXPERIMENTS:
  A. Additive HL at O(1/ln^p N) — Claude reviewer's corrected formulation
  B. Scale evolution — is ||R|| ~ 1/lnN or 1/ln²N? Is the SHAPE stable?
  C. Circulant decomposition — distance-only vs position-dependent
  D. SVD anatomy — what the rank-3 structure physically represents
  E. λ refinement — is the residual just rate miscalibration?
  F. Eigenvalue structure — rotation and chirality
  G. Homeostasis test — Gemini's anti-correlation hypothesis
  H. Mod 210 — do the findings replicate?
"""

import numpy as np
from math import gcd, log
from scipy.optimize import minimize_scalar, minimize, curve_fit
import time
import json
import os

# ═══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════

def simple_sieve(limit):
    is_prime = [False, False] + [True] * (limit - 1)
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [p for p in range(2, limit + 1) if is_prime[p]]

def twin_prime_constant(max_prime=100000):
    primes = simple_sieve(max_prime)
    Pi2 = 1.0
    for p in primes:
        if p >= 3:
            Pi2 *= 1 - 1.0 / (p - 1)**2
    return Pi2

def singular_series(g, Pi2):
    if g <= 0 or g % 2 != 0:
        return 0
    S = 2 * Pi2
    n = g
    p = 3
    while p * p <= n:
        if n % p == 0:
            S *= (p - 1) / (p - 2)
            while n % p == 0:
                n //= p
        p += 2
    if n > 2:
        S *= (n - 1) / (n - 2)
    return S

def sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def admissible_columns(m):
    return sorted([r for r in range(1, m) if gcd(r, m) == 1])

def distance_matrix(cols, m):
    n = len(cols)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = m if i == j else (cols[j] - cols[i]) % m
    return D

def empirical_matrix(primes, cols, m):
    n_cols = len(cols)
    col_idx = {c: i for i, c in enumerate(cols)}
    working = [p for p in primes if p > m]
    T = np.zeros((n_cols, n_cols))
    for k in range(len(working) - 1):
        r_from = working[k] % m
        r_to = working[k + 1] % m
        if r_from in col_idx and r_to in col_idx:
            T[col_idx[r_from], col_idx[r_to]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return T / row_sums

def boltzmann_matrix(D, ln_N):
    B = np.exp(-D / ln_N)
    return B / B.sum(axis=1, keepdims=True)

def circulant_decompose(R, D, m):
    """Split R into circulant (distance-only) + non-circulant (position-dependent)."""
    n = R.shape[0]
    unique_d = sorted(set(int(D[i,j]) for i in range(n) for j in range(n)))
    d_avg = {}
    d_count = {}
    for d in unique_d:
        entries = [(i,j) for i in range(n) for j in range(n) if int(D[i,j]) == d]
        d_avg[d] = np.mean([R[i,j] for i,j in entries])
        d_count[d] = len(entries)
    R_circ = np.zeros_like(R)
    for i in range(n):
        for j in range(n):
            R_circ[i,j] = d_avg[int(D[i,j])]
    R_asym = R - R_circ
    return R_circ, R_asym, d_avg, d_count, unique_d


# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  WAVE 22h — RESIDUAL ARCHAEOLOGY")
    print("  What IS the 3% Boltzmann residual?")
    print("=" * 70)
    print()

    start = time.time()

    # Setup
    Pi2 = twin_prime_constant(100000)
    C2 = 2 * Pi2
    m = 30
    cols = admissible_columns(m)
    n_cols = len(cols)
    D = distance_matrix(cols, m)

    S_mat = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            S_mat[i, j] = singular_series(int(D[i, j]), Pi2)

    LIMIT = 10**8
    print(f"Sieving primes to {LIMIT:,}...")
    all_primes = sieve_primes(LIMIT)
    print(f"  Found {len(all_primes):,} primes")
    print()

    windows = [
        (10**4, 10**5, 4.5),
        (10**5, 10**6, 5.5),
        (10**6, 10**7, 6.5),
        (10**7, 10**8, 7.5),
    ]

    # Pre-compute all residuals
    residuals = {}
    empiricals = {}
    boltzmanns = {}

    for lo, hi, log_mid in windows:
        ln_N = log_mid * np.log(10)
        mask = (all_primes >= lo) & (all_primes < hi)
        primes_w = all_primes[mask].tolist()
        T_emp = empirical_matrix(primes_w, cols, m)
        B = boltzmann_matrix(D, ln_N)
        R = T_emp - B
        residuals[log_mid] = R
        empiricals[log_mid] = T_emp
        boltzmanns[log_mid] = B

    # ═══════════════════════════════════════════════════════
    # SECTION A: ADDITIVE HL CORRECTION
    # ═══════════════════════════════════════════════════════

    print("=" * 70)
    print("  A. ADDITIVE HL CORRECTION — T = B·(1 + β·(S−S̄)/(S̄·lnᵖN))")
    print("  Claude reviewer: right mechanism, wrong scale")
    print("=" * 70)
    print()

    S_row_mean = S_mat.mean(axis=1, keepdims=True)
    delta_S = (S_mat - S_row_mean) / S_row_mean

    for lo, hi, log_mid in windows:
        ln_N = log_mid * np.log(10)
        T_emp = empiricals[log_mid]
        B = boltzmanns[log_mid]
        R = residuals[log_mid]

        T_null = np.ones_like(T_emp) / n_cols
        ss_tot = np.sum((T_emp - T_null)**2)
        r2_B = 1 - np.sum(R**2) / ss_tot

        # β=1, p=2 (Claude's exact prediction)
        T_a1 = B * (1 + delta_S / ln_N**2)
        T_a1 = T_a1 / T_a1.sum(axis=1, keepdims=True)
        r2_a1 = 1 - np.sum((T_emp - T_a1)**2) / ss_tot

        # Free β, p=2
        def obj_b(beta, _ln=ln_N, _B=B, _dS=delta_S, _Te=T_emp):
            T_m = _B * (1 + beta * _dS / _ln**2)
            T_m = T_m / T_m.sum(axis=1, keepdims=True)
            return np.sum((_Te - T_m)**2)
        res = minimize_scalar(obj_b, bounds=(-100, 100), method='bounded')
        beta2 = res.x
        r2_b2 = 1 - res.fun / ss_tot

        # Free β AND free p — find the natural scaling
        def obj_bp(params, _ln=ln_N, _B=B, _dS=delta_S, _Te=T_emp):
            beta, power = params
            if power < 0:
                return 1e10
            T_m = _B * (1 + beta * _dS / max(_ln**power, 1e-10))
            T_m = np.maximum(T_m, 1e-15)
            T_m = T_m / T_m.sum(axis=1, keepdims=True)
            return np.sum((_Te - T_m)**2)
        res2 = minimize(obj_bp, [beta2, 2.0], method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-15})
        beta_f, power_f = res2.x
        r2_free = 1 - res2.fun / ss_tot

        lo_str = f"10^{log_mid-0.5:.0f}–10^{log_mid+0.5:.0f}"
        print(f"  {lo_str} (lnN={ln_N:.2f})")
        print(f"    R²(Boltzmann):            {r2_B:.6f}")
        print(f"    R²(+HL, β=1, p=2):       {r2_a1:.6f}  Δ={r2_a1-r2_B:+.7f}")
        print(f"    R²(+HL, β={beta2:+.2f}, p=2):  {r2_b2:.6f}  Δ={r2_b2-r2_B:+.7f}")
        print(f"    R²(+HL, β={beta_f:+.2f}, p={power_f:.2f}): {r2_free:.6f}  Δ={r2_free-r2_B:+.7f}")
        print()

    print("  VERDICT A: Does the additive HL correction help?")
    print("  (Examine the Δ values — positive = improvement)")
    print()

    # ═══════════════════════════════════════════════════════
    # SECTION B: SCALE EVOLUTION — O(1/lnN) or O(1/ln²N)?
    # ═══════════════════════════════════════════════════════

    print("=" * 70)
    print("  B. SCALE EVOLUTION — What order is the residual?")
    print("=" * 70)
    print()

    ln_vals = []
    norm_vals = []
    norm_circ_vals = []
    norm_asym_vals = []

    print(f"  {'Scale':>12}  {'lnN':>6}  {'||R||':>8}  {'lnN·||R||':>10}  {'ln²N·||R||':>11}")
    print(f"  {'─'*60}")

    for log_mid in [4.5, 5.5, 6.5, 7.5]:
        ln_N = log_mid * np.log(10)
        R = residuals[log_mid]
        R_c, R_a, _, _, _ = circulant_decompose(R, D, m)

        norm = np.linalg.norm(R, 'fro')
        ln_vals.append(ln_N)
        norm_vals.append(norm)
        norm_circ_vals.append(np.linalg.norm(R_c, 'fro'))
        norm_asym_vals.append(np.linalg.norm(R_a, 'fro'))

        lo_str = f"10^{log_mid-0.5:.0f}–10^{log_mid+0.5:.0f}"
        print(f"  {lo_str:>12}  {ln_N:>6.2f}  {norm:>8.6f}  {ln_N*norm:>10.6f}  {ln_N**2*norm:>11.6f}")

    # Fit power law: ||R|| = c / lnN^α
    def power_law(x, c, alpha):
        return c / x**alpha

    ln_arr = np.array(ln_vals)
    norm_arr = np.array(norm_vals)
    try:
        popt, _ = curve_fit(power_law, ln_arr, norm_arr, p0=[1.0, 1.0])
        c_fit, alpha_fit = popt
        pred = power_law(ln_arr, *popt)
        r2_pl = 1 - np.sum((norm_arr - pred)**2) / np.sum((norm_arr - norm_arr.mean())**2)
        print(f"\n  Power law fit: ||R|| = {c_fit:.4f} / lnN^{alpha_fit:.4f}  (R² = {r2_pl:.4f})")
    except Exception as e:
        print(f"\n  Power law fit failed: {e}")

    # Variation coefficient: how stable is lnN·||R|| vs ln²N·||R||?
    scaled_1 = ln_arr * norm_arr
    scaled_2 = ln_arr**2 * norm_arr
    cv_1 = np.std(scaled_1) / np.mean(scaled_1)
    cv_2 = np.std(scaled_2) / np.mean(scaled_2)
    print(f"\n  Stability test (coefficient of variation, lower = more constant):")
    print(f"    CV(lnN · ||R||)  = {cv_1:.4f}  (mean = {np.mean(scaled_1):.6f})")
    print(f"    CV(ln²N · ||R||) = {cv_2:.4f}  (mean = {np.mean(scaled_2):.6f})")
    print(f"    → The residual scales as 1/lnN^{alpha_fit:.2f}")
    print()

    # Component evolution
    print(f"  Component breakdown:")
    print(f"  {'Scale':>12}  {'||R_circ||':>10}  {'||R_asym||':>10}  {'circ%':>6}  {'asym%':>6}")
    for i, log_mid in enumerate([4.5, 5.5, 6.5, 7.5]):
        lo_str = f"10^{log_mid-0.5:.0f}–10^{log_mid+0.5:.0f}"
        total_sq = norm_vals[i]**2
        print(f"  {lo_str:>12}  {norm_circ_vals[i]:>10.6f}  {norm_asym_vals[i]:>10.6f}  "
              f"{100*norm_circ_vals[i]**2/total_sq:>5.1f}%  {100*norm_asym_vals[i]**2/total_sq:>5.1f}%")

    # Fit scaling for each component
    try:
        popt_c, _ = curve_fit(power_law, ln_arr, np.array(norm_circ_vals), p0=[1.0, 1.0])
        popt_a, _ = curve_fit(power_law, ln_arr, np.array(norm_asym_vals), p0=[1.0, 1.0])
        print(f"\n  ||R_circ|| ~ 1/lnN^{popt_c[1]:.3f}")
        print(f"  ||R_asym|| ~ 1/lnN^{popt_a[1]:.3f}")
    except:
        pass
    print()

    # ═══════════════════════════════════════════════════════
    # SECTION C: lnN · R(N) SHAPE STABILITY
    # ═══════════════════════════════════════════════════════

    print("=" * 70)
    print("  C. SHAPE STABILITY — Is lnN·R(N) ≈ constant matrix M?")
    print("  If yes, then R = M/lnN for a KNOWN fixed matrix M.")
    print("=" * 70)
    print()

    pairs = [(4.5, 7.5), (5.5, 7.5), (6.5, 7.5), (5.5, 6.5)]
    for (la, lb) in pairs:
        lna = la * np.log(10)
        lnb = lb * np.log(10)
        Ma = lna * residuals[la]
        Mb = lnb * residuals[lb]
        corr = np.corrcoef(Ma.flatten(), Mb.flatten())[0, 1]
        diff = np.linalg.norm(Ma - Mb, 'fro')
        avg = 0.5 * (np.linalg.norm(Ma, 'fro') + np.linalg.norm(Mb, 'fro'))
        sa = f"10^{la-0.5:.0f}–10^{la+0.5:.0f}"
        sb = f"10^{lb-0.5:.0f}–10^{lb+0.5:.0f}"
        print(f"  lnN·R at {sa} vs {sb}:  corr={corr:+.4f}  ||Δ||/||M̄||={diff/avg:.4f}")

    # Same with ln²N
    print()
    for (la, lb) in [(5.5, 7.5), (6.5, 7.5)]:
        lna = la * np.log(10)
        lnb = lb * np.log(10)
        Ma = lna**2 * residuals[la]
        Mb = lnb**2 * residuals[lb]
        corr = np.corrcoef(Ma.flatten(), Mb.flatten())[0, 1]
        diff = np.linalg.norm(Ma - Mb, 'fro')
        avg = 0.5 * (np.linalg.norm(Ma, 'fro') + np.linalg.norm(Mb, 'fro'))
        sa = f"10^{la-0.5:.0f}–10^{la+0.5:.0f}"
        sb = f"10^{lb-0.5:.0f}–10^{lb+0.5:.0f}"
        print(f"  ln²N·R at {sa} vs {sb}: corr={corr:+.4f}  ||Δ||/||M̄||={diff/avg:.4f}")

    # If the shape is stable, show the limiting matrix M = lnN · R at largest scale
    ln_N_main = 7.5 * np.log(10)
    M_limit = ln_N_main * residuals[7.5]
    print(f"\n  Limiting matrix M = lnN · R (at 10^7–10^8, lnN={ln_N_main:.2f}):")
    header = "         " + "".join(f"{c:>8}" for c in cols)
    print(f"  {header}")
    for i, ci in enumerate(cols):
        row_str = f"    {ci:>3} "
        for j in range(n_cols):
            row_str += f" {M_limit[i,j]:+7.4f}"
        print(row_str)
    print()

    # ═══════════════════════════════════════════════════════
    # SECTION D: CIRCULANT DECOMPOSITION
    # ═══════════════════════════════════════════════════════

    print("=" * 70)
    print("  D. CIRCULANT DECOMPOSITION (10^7–10^8)")
    print("  R = R_circ (depends only on distance) + R_asym (position-specific)")
    print("=" * 70)
    print()

    R_main = residuals[7.5]
    R_circ, R_asym, d_avg, d_count, unique_d = circulant_decompose(R_main, D, m)

    norm_R = np.linalg.norm(R_main, 'fro')
    norm_c = np.linalg.norm(R_circ, 'fro')
    norm_a = np.linalg.norm(R_asym, 'fro')

    print(f"  ||R||²       = {norm_R**2:.8f}  (100.0%)")
    print(f"  ||R_circ||²  = {norm_c**2:.8f}  ({100*norm_c**2/norm_R**2:.1f}%)")
    print(f"  ||R_asym||²  = {norm_a**2:.8f}  ({100*norm_a**2/norm_R**2:.1f}%)")
    print()

    # Distance profile: f(d) = circulant signal
    S_bar = S_mat.mean()
    print(f"  Distance profile (S̄ = {S_bar:.4f}):")
    print(f"    {'d':>3}  {'n':>3}  {'f(d)':>10}  {'S(d)':>8}  {'(S−S̄)/S̄':>10}  direction")
    off_d = [d for d in unique_d if d != m]
    f_vals = []
    S_delta_vals = []
    for d in unique_d:
        Sd = singular_series(d if d < m else 2, Pi2)  # d=30 → use S(2) as placeholder
        if d == m:
            Sd_str = "  (self)"
        else:
            Sd_str = f"  {Sd:>8.4f}"
        delta = (Sd - S_bar) / S_bar if d != m else float('nan')
        direction = ""
        if d != m:
            f_vals.append(d_avg[d])
            S_delta_vals.append(delta)
            if d_avg[d] > 0 and delta > 0:
                direction = "  S↑ R↑ (concordant)"
            elif d_avg[d] < 0 and delta < 0:
                direction = "  S↓ R↓ (concordant)"
            elif d_avg[d] > 0 and delta < 0:
                direction = "  S↓ R↑ (ANTI)"
            elif d_avg[d] < 0 and delta > 0:
                direction = "  S↑ R↓ (ANTI)"
        delta_str = f"  {delta:>+10.4f}" if d != m else "       n/a"
        print(f"    {d:>3}  {d_count[d]:>3}  {d_avg[d]:>+10.7f}{Sd_str}{delta_str}{direction}")

    f_arr = np.array(f_vals)
    S_d_arr = np.array(S_delta_vals)
    corr_fS = np.corrcoef(f_arr, S_d_arr)[0, 1] if len(f_arr) > 2 else float('nan')
    print(f"\n  Correlation f(d) vs (S(d)−S̄)/S̄: r = {corr_fS:.4f}")
    print(f"  → {'ANTI-correlated (homeostasis confirmed)' if corr_fS < -0.3 else 'CONCORDANT (homeostasis rejected)' if corr_fS > 0.3 else 'WEAK/AMBIGUOUS'}")
    print()

    # Non-circulant part
    print(f"  Non-circulant R_asym (position-dependent deviations):")
    header = "         " + "".join(f"{c:>8}" for c in cols)
    print(f"  {header}")
    for i, ci in enumerate(cols):
        row_str = f"    {ci:>3} "
        for j in range(n_cols):
            row_str += f" {R_asym[i,j]:+7.5f}"
        print(row_str)
    print()

    # ═══════════════════════════════════════════════════════
    # SECTION E: SVD ANATOMY
    # ═══════════════════════════════════════════════════════

    print("=" * 70)
    print("  E. SVD ANATOMY — What the rank-3 structure IS")
    print("=" * 70)
    print()

    U, S_svd, Vt = np.linalg.svd(R_main)
    cum_var = np.cumsum(S_svd**2) / np.sum(S_svd**2) * 100

    print(f"  Singular values: {', '.join(f'{s:.6f}' for s in S_svd)}")
    print(f"  Cumulative var:  {', '.join(f'{v:.1f}%' for v in cum_var)}")
    print()

    # Build feature vectors for the 8 residue classes
    col_arr = np.array(cols, dtype=float)
    features = {
        'position': col_arr - col_arr.mean(),
        'χ₃ (mod 3)': np.array([+1 if c%3==1 else -1 for c in cols], dtype=float),
        'χ₅ (mod 5)': np.array([+1 if (c%5) in {1,4} else -1 for c in cols], dtype=float),
        'χ₇ (mod 7)': np.array([+1 if (c%7) in {1,2,4} else -1 for c in cols], dtype=float),
        'sym (r vs 30-r)': np.array([-1,-1,-1,-1,+1,+1,+1,+1], dtype=float),
        'edge dist': np.array([min(c, m-c) for c in cols], dtype=float),
    }
    # Normalize features
    for k in features:
        f = features[k]
        features[k] = (f - f.mean()) / (f.std() + 1e-10)

    for k in range(min(4, len(S_svd))):
        pct = S_svd[k]**2 / np.sum(S_svd**2) * 100
        print(f"  Mode {k+1} (σ={S_svd[k]:.6f}, {pct:.1f}% of var):")
        print(f"    u{k+1} = [{', '.join(f'{u:+.4f}' for u in U[:, k])}]")
        print(f"    v{k+1} = [{', '.join(f'{v:+.4f}' for v in Vt[k, :])}]")
        print(f"    Residue class correlations with u{k+1}:")
        for fname, fvec in features.items():
            r = np.corrcoef(U[:, k], fvec)[0, 1]
            star = " ★" if abs(r) > 0.7 else ""
            print(f"      {fname:<18}  r = {r:+.4f}{star}")
        print()

    # Show rank-1 and rank-2 approximations
    R_rank1 = S_svd[0] * np.outer(U[:, 0], Vt[0, :])
    R_rank2 = R_rank1 + S_svd[1] * np.outer(U[:, 1], Vt[1, :])
    R_rank3 = R_rank2 + S_svd[2] * np.outer(U[:, 2], Vt[2, :])

    print(f"  Rank-1 approximation (captures {cum_var[0]:.1f}%):")
    header = "         " + "".join(f"{c:>8}" for c in cols)
    print(f"  {header}")
    for i, ci in enumerate(cols):
        row_str = f"    {ci:>3} "
        for j in range(n_cols):
            row_str += f" {R_rank1[i,j]:+7.5f}"
        print(row_str)
    print()

    # ═══════════════════════════════════════════════════════
    # SECTION F: λ REFINEMENT
    # ═══════════════════════════════════════════════════════

    print("=" * 70)
    print("  F. λ REFINEMENT — Is the residual just rate miscalibration?")
    print("=" * 70)
    print()

    for lo, hi, log_mid in windows:
        ln_N = log_mid * np.log(10)
        T_emp = empiricals[log_mid]
        T_null = np.ones_like(T_emp) / n_cols
        ss_tot = np.sum((T_emp - T_null)**2)

        # Standard Boltzmann: λ = 1/lnN
        B0 = boltzmann_matrix(D, ln_N)
        r2_0 = 1 - np.sum((T_emp - B0)**2) / ss_tot

        # Fit λ = f/lnN (1 parameter)
        def obj_f(f, _ln=ln_N, _D=D, _Te=T_emp):
            T_m = boltzmann_matrix(_D, _ln * f)
            return np.sum((_Te - T_m)**2)
        res = minimize_scalar(obj_f, bounds=(0.5, 2.0), method='bounded')
        f_opt = res.x
        r2_f = 1 - res.fun / ss_tot

        # Fit exp(-d/lnN + c·d²/ln²N) (1 parameter — quadratic distance correction)
        def obj_quad(c, _ln=ln_N, _D=D, _Te=T_emp):
            exponent = -_D / _ln + c * _D**2 / _ln**2
            T_m = np.exp(exponent)
            T_m = T_m / T_m.sum(axis=1, keepdims=True)
            return np.sum((_Te - T_m)**2)
        res_q = minimize_scalar(obj_quad, bounds=(-0.5, 0.5), method='bounded')
        c_quad = res_q.x
        r2_q = 1 - res_q.fun / ss_tot

        # Fit both f AND c (2 parameters)
        def obj_fc(params, _ln=ln_N, _D=D, _Te=T_emp):
            f, c = params
            if f <= 0:
                return 1e10
            exponent = -_D / (_ln * f) + c * _D**2 / _ln**2
            T_m = np.exp(exponent)
            T_m = T_m / T_m.sum(axis=1, keepdims=True)
            return np.sum((_Te - T_m)**2)
        res_fc = minimize(obj_fc, [f_opt, c_quad], method='Nelder-Mead',
                         options={'maxiter': 5000, 'fatol': 1e-15})
        f_fc, c_fc = res_fc.x
        r2_fc = 1 - res_fc.fun / ss_tot

        lo_str = f"10^{log_mid-0.5:.0f}–10^{log_mid+0.5:.0f}"
        print(f"  {lo_str} (lnN={ln_N:.2f}):")
        print(f"    R²(λ=1/lnN):              {r2_0:.6f}")
        print(f"    R²(λ={f_opt:.4f}/lnN):         {r2_f:.6f}  Δ={r2_f-r2_0:+.7f}  (1 param)")
        print(f"    R²(+d²/ln²N, c={c_quad:+.5f}): {r2_q:.6f}  Δ={r2_q-r2_0:+.7f}  (1 param)")
        print(f"    R²(f={f_fc:.4f}, c={c_fc:+.5f}):   {r2_fc:.6f}  Δ={r2_fc-r2_0:+.7f}  (2 params)")
        print()

    # ═══════════════════════════════════════════════════════
    # SECTION G: EIGENVALUE STRUCTURE
    # ═══════════════════════════════════════════════════════

    print("=" * 70)
    print("  G. EIGENVALUE STRUCTURE (10^7–10^8)")
    print("=" * 70)
    print()

    T_emp_main = empiricals[7.5]
    B_main = boltzmanns[7.5]

    eig_emp = np.linalg.eigvals(T_emp_main)
    eig_B = np.linalg.eigvals(B_main)
    eig_R = np.linalg.eigvals(R_main)

    def fmt_eig(e):
        if abs(e.imag) > 1e-8:
            return f"{e.real:+.6f} {e.imag:+.6f}i  |λ|={abs(e):.6f}  arg={np.degrees(np.angle(e)):+.1f}°"
        return f"{e.real:+.6f}                |λ|={abs(e):.6f}"

    print("  T_empirical eigenvalues (sorted by |λ|):")
    for e in sorted(eig_emp, key=lambda x: -abs(x)):
        print(f"    {fmt_eig(e)}")

    print("\n  B_Boltzmann eigenvalues:")
    for e in sorted(eig_B, key=lambda x: -abs(x)):
        print(f"    {fmt_eig(e)}")

    print("\n  R_residual eigenvalues:")
    for e in sorted(eig_R, key=lambda x: -abs(x)):
        print(f"    {fmt_eig(e)}")

    # Compare: eigenvalue-by-eigenvalue deviation
    eig_emp_s = sorted(eig_emp, key=lambda x: (-abs(x), np.angle(x)))
    eig_B_s = sorted(eig_B, key=lambda x: (-abs(x), np.angle(x)))
    print("\n  Eigenvalue deviation (empirical - Boltzmann):")
    for i, (ee, eb) in enumerate(zip(eig_emp_s, eig_B_s)):
        diff = ee - eb
        print(f"    λ{i+1}: emp={fmt_eig(ee)[:25]}  B={fmt_eig(eb)[:25]}  Δ={diff.real:+.6f}{diff.imag:+.6f}i")

    # ═══════════════════════════════════════════════════════
    # SECTION H: HOMEOSTASIS TEST (Gemini)
    # ═══════════════════════════════════════════════════════

    print()
    print("=" * 70)
    print("  H. HOMEOSTASIS — Does the mod ring suppress common gaps?")
    print("  Gemini: high S(g) → negative residual = structural self-defense")
    print("=" * 70)
    print()

    # Test across all scales
    for log_mid in [4.5, 5.5, 6.5, 7.5]:
        R_s = residuals[log_mid]
        _, _, d_avg_s, _, _ = circulant_decompose(R_s, D, m)
        f_s = np.array([d_avg_s[d] for d in off_d])
        S_s = np.array([singular_series(d, Pi2) for d in off_d])
        corr_s = np.corrcoef(f_s, S_s)[0, 1]
        lo_str = f"10^{log_mid-0.5:.0f}–10^{log_mid+0.5:.0f}"
        print(f"  {lo_str}: corr(f(d), S(d)) = {corr_s:+.4f}")

    # Entry-level test (off-diagonal)
    mask_off = ~np.eye(n_cols, dtype=bool)
    R_flat = R_main[mask_off]
    S_flat = S_mat[mask_off]
    corr_entry = np.corrcoef(R_flat, S_flat)[0, 1]
    print(f"\n  Entry-level (off-diag, 10^7–10^8):")
    print(f"    corr(R[i,j], S(d[i,j])) = {corr_entry:+.4f}")

    # The deep test: within each distance class, is there structure?
    print(f"\n  Within-distance deviations (non-circulant part):")
    R_asym_flat = R_asym[mask_off]
    # Group by distance
    for d in sorted(off_d)[:8]:  # Show first 8 distances
        mask_d = np.abs(D[mask_off] - d) < 0.5
        if mask_d.sum() > 0:
            vals = R_asym[mask_off][mask_d]
            print(f"    d={d:>2}: std(R_asym) = {np.std(vals):.6f}  "
                  f"range=[{np.min(vals):+.6f}, {np.max(vals):+.6f}]  n={mask_d.sum()}")

    # ═══════════════════════════════════════════════════════
    # SECTION I: MOD 210 — KEY TESTS
    # ═══════════════════════════════════════════════════════

    print()
    print("=" * 70)
    print("  I. MOD 210 — Replication at higher modulus")
    print("=" * 70)
    print()

    m2 = 210
    cols2 = admissible_columns(m2)
    n_cols2 = len(cols2)
    D2 = distance_matrix(cols2, m2)

    S_mat2 = np.zeros((n_cols2, n_cols2))
    for i in range(n_cols2):
        for j in range(n_cols2):
            S_mat2[i, j] = singular_series(int(D2[i, j]), Pi2)

    ln_N = 7.5 * np.log(10)
    mask = (all_primes >= 10**7) & (all_primes < 10**8)
    primes_w = all_primes[mask].tolist()
    T_emp2 = empirical_matrix(primes_w, cols2, m2)
    B2 = boltzmann_matrix(D2, ln_N)
    R2 = T_emp2 - B2

    T_null2 = np.ones_like(T_emp2) / n_cols2
    ss_tot2 = np.sum((T_emp2 - T_null2)**2)
    r2_B2 = 1 - np.sum(R2**2) / ss_tot2

    # Circulant decomposition
    R2_circ, R2_asym, d_avg2, d_count2, unique_d2 = circulant_decompose(R2, D2, m2)
    norm_R2 = np.linalg.norm(R2, 'fro')
    norm_R2c = np.linalg.norm(R2_circ, 'fro')
    norm_R2a = np.linalg.norm(R2_asym, 'fro')

    print(f"  R²(Boltzmann): {r2_B2:.6f}")
    print(f"  ||R||²       = {norm_R2**2:.8f}  (100.0%)")
    print(f"  ||R_circ||²  = {norm_R2c**2:.8f}  ({100*norm_R2c**2/norm_R2**2:.1f}%)")
    print(f"  ||R_asym||²  = {norm_R2a**2:.8f}  ({100*norm_R2a**2/norm_R2**2:.1f}%)")

    # Additive HL
    S_rm2 = S_mat2.mean(axis=1, keepdims=True)
    dS2 = (S_mat2 - S_rm2) / S_rm2
    def obj_b210(beta):
        T_m = B2 * (1 + beta * dS2 / ln_N**2)
        T_m = T_m / T_m.sum(axis=1, keepdims=True)
        return np.sum((T_emp2 - T_m)**2)
    res210 = minimize_scalar(obj_b210, bounds=(-100, 100), method='bounded')
    r2_210 = 1 - res210.fun / ss_tot2
    print(f"\n  Additive HL (β={res210.x:+.2f}/ln²N): R²={r2_210:.6f}  Δ={r2_210-r2_B2:+.7f}")

    # λ refinement
    def obj_f210(f):
        T_m = boltzmann_matrix(D2, ln_N * f)
        return np.sum((T_emp2 - T_m)**2)
    resf210 = minimize_scalar(obj_f210, bounds=(0.5, 2.0), method='bounded')
    r2_f210 = 1 - resf210.fun / ss_tot2
    print(f"  λ refinement (f={resf210.x:.4f}):     R²={r2_f210:.6f}  Δ={r2_f210-r2_B2:+.7f}")

    # Homeostasis
    off_d2 = [d for d in unique_d2 if d != m2]
    f_off2 = np.array([d_avg2[d] for d in off_d2])
    S_off2 = np.array([singular_series(d, Pi2) for d in off_d2])
    corr_h210 = np.corrcoef(f_off2, S_off2)[0, 1]
    print(f"\n  Homeostasis corr(f(d), S(d)): r = {corr_h210:+.4f}")

    # SVD
    U2, S2_svd, Vt2 = np.linalg.svd(R2)
    cum2 = np.cumsum(S2_svd**2) / np.sum(S2_svd**2) * 100
    print(f"\n  Residual SVD top 5: [{', '.join(f'{s:.6f}' for s in S2_svd[:5])}]")
    print(f"  Cumulative var:     [{', '.join(f'{v:.1f}%' for v in cum2[:5])}]")
    
    # Effective rank (how many modes to capture 90%)
    for threshold in [50, 75, 90, 95]:
        rank = np.searchsorted(cum2, threshold) + 1
        print(f"  Modes for {threshold}% variance: {rank}")

    # ═══════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    results = {
        "wave": "22h",
        "title": "Residual Archaeology",
        "timestamp": int(time.time() * 1000),
        "mod30": {
            "R2_boltzmann": float(1 - np.sum(residuals[7.5]**2) / np.sum((empiricals[7.5] - np.ones_like(empiricals[7.5])/n_cols)**2)),
            "norm_R": float(norm_R),
            "circulant_fraction": float(norm_c**2 / norm_R**2),
            "asymmetric_fraction": float(norm_a**2 / norm_R**2),
            "svd_top5": [float(s) for s in S_svd[:5]],
            "homeostasis_corr": float(corr_fS),
        },
        "mod210": {
            "R2_boltzmann": float(r2_B2),
            "norm_R": float(norm_R2),
            "circulant_fraction": float(norm_R2c**2 / norm_R2**2),
            "asymmetric_fraction": float(norm_R2a**2 / norm_R2**2),
            "homeostasis_corr": float(corr_h210),
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    ts = results["timestamp"]
    outpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"wave22h_archaeology_{ts}.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {outpath}")
