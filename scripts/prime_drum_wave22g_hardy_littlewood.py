#!/usr/bin/env python3
"""
Wave 22g — The Hardy-Littlewood Second-Order Term

Does the Boltzmann residual = the singular series correction?

T(a→b) ≈ [exp(-d(a,b)/ln(N)) / Z] · S(d(a,b)) / S̄

where S(g) is the Hardy-Littlewood singular series for gap g.

If R² jumps from 0.971 to 0.99+, the second-order term is confirmed.
"""

import numpy as np
from math import gcd, log
from scipy.optimize import minimize_scalar
import time


# ═══════════════════════════════════════════════════════
# HARDY-LITTLEWOOD SINGULAR SERIES
# ═══════════════════════════════════════════════════════

def simple_sieve(limit):
    is_prime = [False, False] + [True] * (limit - 1)
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [p for p in range(2, limit + 1) if is_prime[p]]


def twin_prime_constant(max_prime=10000):
    """
    C_2 = 2 · ∏_{p>2} p(p-2)/(p-1)²
    
    The product converges slowly. Using primes up to 10000 gives ~8 digits.
    Known value: C_2 ≈ 1.3203236...
    Wait — convention matters.
    
    The twin prime constant is usually:
      Π₂ = ∏_{p≥3} (1 - 1/(p-1)²) ≈ 0.6601618...
    
    And S(2) = 2·Π₂ ≈ 1.3203236...
    
    S(g) for general even g:
      S(g) = 2·Π₂ · ∏_{p|g, p≥3} (p-1)/(p-2)
    
    Let me compute Π₂ directly.
    """
    primes = simple_sieve(max_prime)
    Pi2 = 1.0
    for p in primes:
        if p >= 3:
            Pi2 *= 1 - 1.0 / (p - 1)**2
    return Pi2


def singular_series(g, Pi2):
    """
    S(g) = 2·Π₂ · ∏_{p|g, p odd} (p-1)/(p-2)
    
    For even g only. For odd g, S(g) = 0 (no odd-gap prime pairs except (2,p)).
    For g=0 this is undefined; we use g=m for self-transitions.
    """
    if g <= 0:
        return 0
    if g % 2 != 0:
        return 0  # No prime pairs with odd gap (except involving 2)
    
    S = 2 * Pi2
    
    # Factor g and apply corrections for odd prime factors
    n = g
    p = 3
    while p * p <= n:
        if n % p == 0:
            S *= (p - 1) / (p - 2)
            while n % p == 0:
                n //= p
        p += 2
    if n > 2:  # remaining odd prime factor
        S *= (n - 1) / (n - 2)
    
    return S


# ═══════════════════════════════════════════════════════
# PRIME SIEVE (for empirical matrix)
# ═══════════════════════════════════════════════════════

def sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


def admissible_columns(m):
    return sorted([r for r in range(1, m) if gcd(r, m) == 1])


def corrected_distance_matrix(cols, m):
    n = len(cols)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = m
            else:
                D[i, j] = (cols[j] - cols[i]) % m
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


if __name__ == "__main__":
    print("=" * 70)
    print("  WAVE 22g — THE HARDY-LITTLEWOOD SECOND-ORDER TERM")
    print("  Does the 3% residual = the singular series?")
    print("=" * 70)
    print()
    
    start = time.time()
    
    # ═══════════════════════════════════
    # Step 1: Compute the constants
    # ═══════════════════════════════════
    
    Pi2 = twin_prime_constant(100000)
    C2 = 2 * Pi2
    print(f"Twin prime constant Π₂ = {Pi2:.10f}")
    print(f"S(2) = 2·Π₂ = C₂ = {C2:.10f}")
    print(f"(Known value: Π₂ ≈ 0.6601618159, C₂ ≈ 1.3203236318)")
    print()
    
    # Show S(g) for all even gaps up to 30
    print("Singular series S(g) for even gaps:")
    print(f"  {'g':>3}  {'S(g)':>10}  {'S(g)/C₂':>10}  Odd prime factors of g")
    print(f"  {'-'*50}")
    for g in range(2, 32, 2):
        Sg = singular_series(g, Pi2)
        factors = []
        n = g
        p = 3
        while p * p <= n:
            if n % p == 0:
                factors.append(str(p))
                while n % p == 0:
                    n //= p
            p += 2
        if n > 2:
            factors.append(str(n))
        factor_str = "×".join(factors) if factors else "(none)"
        print(f"  {g:>3}  {Sg:>10.6f}  {Sg/C2:>10.4f}  {factor_str}")
    print()
    
    # ═══════════════════════════════════
    # Step 2: Build matrices at mod 30
    # ═══════════════════════════════════
    
    m = 30
    cols = admissible_columns(m)
    n_cols = len(cols)
    D = corrected_distance_matrix(cols, m)
    
    # S(g) matrix: S(d(a,b)) for each (a,b) pair
    S_mat = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            g = int(D[i, j])
            S_mat[i, j] = singular_series(g, Pi2)
    
    print("Distance matrix d(a,b) mod 30:")
    header = "      " + "".join(f"{c:>5}" for c in cols)
    print(header)
    for i, ci in enumerate(cols):
        row = f"  {ci:>3} " + "".join(f"{int(D[i,j]):>5}" for j in range(n_cols))
        print(row)
    print()
    
    print("Singular series matrix S(d(a,b)):")
    header = "      " + "".join(f"{c:>8}" for c in cols)
    print(header)
    for i, ci in enumerate(cols):
        row = f"  {ci:>3} " + "".join(f"{S_mat[i,j]:>8.4f}" for j in range(n_cols))
        print(row)
    print()
    
    # ═══════════════════════════════════
    # Step 3: Sieve and measure
    # ═══════════════════════════════════
    
    LIMIT = 10**8
    print(f"Sieving primes to {LIMIT:,}...")
    all_primes = sieve_primes(LIMIT)
    print(f"  Found {len(all_primes):,} primes")
    print()
    
    # Test at multiple scales
    windows = [
        (10**5, 10**6, 5.5),
        (10**6, 10**7, 6.5),
        (10**7, 10**8, 7.5),
    ]
    
    print("=" * 70)
    print("  RESULTS: Boltzmann vs Boltzmann + Hardy-Littlewood")
    print("=" * 70)
    print()
    
    for lo, hi, log_mid in windows:
        ln_N = log_mid * np.log(10)
        
        mask = (all_primes >= lo) & (all_primes < hi)
        primes_window = all_primes[mask].tolist()
        
        T_emp = empirical_matrix(primes_window, cols, m)
        
        # Model 1: Pure Boltzmann (zero parameters)
        B = np.exp(-D / ln_N)
        B = B / B.sum(axis=1, keepdims=True)
        
        # Model 2: Boltzmann × S(g) (zero parameters)
        # T_corrected[i,j] = B[i,j] * S(d(i,j)) / Z_i
        T_HL = B * S_mat
        T_HL = T_HL / T_HL.sum(axis=1, keepdims=True)
        
        # Model 3: Boltzmann × S(g)^α (one parameter — strength of HL correction)  
        def objective_alpha(alpha):
            T_model = B * (S_mat ** alpha)
            T_model = T_model / T_model.sum(axis=1, keepdims=True)
            return np.sum((T_emp - T_model)**2)
        
        result = minimize_scalar(objective_alpha, bounds=(0, 5), method='bounded')
        alpha_opt = result.x
        T_HL_fit = B * (S_mat ** alpha_opt)
        T_HL_fit = T_HL_fit / T_HL_fit.sum(axis=1, keepdims=True)
        
        # R² calculations
        T_null = np.ones_like(T_emp) / n_cols
        ss_tot = np.sum((T_emp - T_null)**2)
        
        ss_res_B = np.sum((T_emp - B)**2)
        ss_res_HL = np.sum((T_emp - T_HL)**2)
        ss_res_HL_fit = np.sum((T_emp - T_HL_fit)**2)
        
        r2_B = 1 - ss_res_B / ss_tot
        r2_HL = 1 - ss_res_HL / ss_tot
        r2_HL_fit = 1 - ss_res_HL_fit / ss_tot
        
        # Residual analysis
        resid_B = T_emp - B
        resid_HL = T_emp - T_HL
        
        # Correlation between predicted correction and actual residual
        correction = T_HL - B  # The HL correction term
        corr = np.corrcoef(resid_B.flatten(), correction.flatten())[0, 1]
        
        print(f"  Scale 10^{log_mid-0.5:.0f}–10^{log_mid+0.5:.0f} "
              f"(ln(N) = {ln_N:.2f}, {len(primes_window):,} primes)")
        print(f"  {'─'*60}")
        print(f"    R²(Boltzmann only):      {r2_B:.6f}  (0 params)")
        print(f"    R²(Boltzmann + HL):      {r2_HL:.6f}  (0 params)")
        print(f"    R²(Boltzmann + HL^α):    {r2_HL_fit:.6f}  (1 param, α={alpha_opt:.4f})")
        print(f"    Improvement (0-param):   {r2_HL - r2_B:+.6f}")
        print(f"    Improvement (1-param):   {r2_HL_fit - r2_B:+.6f}")
        print(f"    Correlation(residual, HL correction): {corr:.4f}")
        print(f"    Max |residual| Boltzmann: {np.max(np.abs(resid_B)):.6f}")
        print(f"    Max |residual| Boltz+HL:  {np.max(np.abs(resid_HL)):.6f}")
        print(f"    Mean |residual| Boltzmann: {np.mean(np.abs(resid_B)):.6f}")
        print(f"    Mean |residual| Boltz+HL:  {np.mean(np.abs(resid_HL)):.6f}")
        print()
        
        # At the largest scale, show the matrices
        if log_mid == 7.5:
            print("  MATRIX COMPARISON (mod 30, 10^7–10^8):")
            print()
            
            for name, T_show in [
                ("EMPIRICAL", T_emp),
                ("BOLTZMANN (0 params)", B),
                ("BOLTZMANN + HL (0 params)", T_HL),
                (f"BOLTZMANN + HL^{alpha_opt:.3f} (1 param)", T_HL_fit),
            ]:
                print(f"    {name}:")
                header = "         " + "".join(f"{c:>7}" for c in cols)
                print(f"    {header}")
                for i, ci in enumerate(cols):
                    row_str = f"      {ci:>3} "
                    for j in range(n_cols):
                        row_str += f" {T_show[i,j]:.4f}"
                    print(row_str)
                print()
            
            # Residual comparison
            print("    RESIDUAL (Empirical - Boltzmann):")
            header = "         " + "".join(f"{c:>7}" for c in cols)
            print(f"    {header}")
            for i, ci in enumerate(cols):
                row_str = f"      {ci:>3} "
                for j in range(n_cols):
                    row_str += f" {resid_B[i,j]:+.4f}"
                print(row_str)
            print()
            
            print("    RESIDUAL (Empirical - Boltzmann - HL):")
            header = "         " + "".join(f"{c:>7}" for c in cols)
            print(f"    {header}")
            for i, ci in enumerate(cols):
                row_str = f"      {ci:>3} "
                for j in range(n_cols):
                    row_str += f" {resid_HL[i,j]:+.4f}"
                print(row_str)
            print()
            
            # SVD of residuals
            U_B, S_B, Vt_B = np.linalg.svd(resid_B)
            U_HL, S_HL, Vt_HL = np.linalg.svd(resid_HL)
            print(f"    Residual SVD (Boltzmann):   σ = [{', '.join(f'{s:.4f}' for s in S_B[:5])}]")
            print(f"    Residual SVD (Boltz+HL):    σ = [{', '.join(f'{s:.4f}' for s in S_HL[:5])}]")
            print(f"    Frobenius norm reduction: {np.linalg.norm(resid_B):.6f} → {np.linalg.norm(resid_HL):.6f}")
            print()
    
    # ═══════════════════════════════════
    # Step 4: Now try mod 210
    # ═══════════════════════════════════
    
    print()
    print("=" * 70)
    print("  MOD 210 — Does the HL correction work at higher modulus?")
    print("=" * 70)
    print()
    
    m2 = 210
    cols2 = admissible_columns(m2)
    n_cols2 = len(cols2)
    D2 = corrected_distance_matrix(cols2, m2)
    
    S_mat2 = np.zeros((n_cols2, n_cols2))
    for i in range(n_cols2):
        for j in range(n_cols2):
            g = int(D2[i, j])
            S_mat2[i, j] = singular_series(g, Pi2)
    
    # Check: how many distinct S values at mod 210?
    unique_S = np.unique(np.round(S_mat2, 6))
    print(f"  Distinct S(g) values in 48×48 matrix: {len(unique_S)}")
    print(f"  Range: [{np.min(S_mat2):.4f}, {np.max(S_mat2):.4f}]")
    print()
    
    for lo, hi, log_mid in [(10**7, 10**8, 7.5)]:
        ln_N = log_mid * np.log(10)
        
        mask = (all_primes >= lo) & (all_primes < hi)
        primes_window = all_primes[mask].tolist()
        
        T_emp2 = empirical_matrix(primes_window, cols2, m2)
        
        B2 = np.exp(-D2 / ln_N)
        B2 = B2 / B2.sum(axis=1, keepdims=True)
        
        T_HL2 = B2 * S_mat2
        T_HL2 = T_HL2 / T_HL2.sum(axis=1, keepdims=True)
        
        # Fitted alpha
        def obj2(alpha):
            T_m = B2 * (S_mat2 ** alpha)
            T_m = T_m / T_m.sum(axis=1, keepdims=True)
            return np.sum((T_emp2 - T_m)**2)
        
        res2 = minimize_scalar(obj2, bounds=(0, 5), method='bounded')
        alpha2 = res2.x
        T_HL2_fit = B2 * (S_mat2 ** alpha2)
        T_HL2_fit = T_HL2_fit / T_HL2_fit.sum(axis=1, keepdims=True)
        
        T_null2 = np.ones_like(T_emp2) / n_cols2
        ss_tot2 = np.sum((T_emp2 - T_null2)**2)
        
        r2_B2 = 1 - np.sum((T_emp2 - B2)**2) / ss_tot2
        r2_HL2 = 1 - np.sum((T_emp2 - T_HL2)**2) / ss_tot2
        r2_HL2_fit = 1 - np.sum((T_emp2 - T_HL2_fit)**2) / ss_tot2
        
        resid_B2 = T_emp2 - B2
        resid_HL2 = T_emp2 - T_HL2
        corr2 = np.corrcoef(resid_B2.flatten(), (T_HL2 - B2).flatten())[0, 1]
        
        print(f"  Scale 10^7–10^8, mod 210 (48 columns, {len(primes_window):,} primes)")
        print(f"  {'─'*60}")
        print(f"    R²(Boltzmann only):      {r2_B2:.6f}  (0 params)")
        print(f"    R²(Boltzmann + HL):      {r2_HL2:.6f}  (0 params)")
        print(f"    R²(Boltzmann + HL^α):    {r2_HL2_fit:.6f}  (1 param, α={alpha2:.4f})")
        print(f"    Improvement (0-param):   {r2_HL2 - r2_B2:+.6f}")
        print(f"    Improvement (1-param):   {r2_HL2_fit - r2_B2:+.6f}")
        print(f"    Correlation(residual, HL correction): {corr2:.4f}")
        print(f"    Frobenius: {np.linalg.norm(resid_B2):.6f} → {np.linalg.norm(resid_HL2):.6f}")
        print()
    
    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s")
