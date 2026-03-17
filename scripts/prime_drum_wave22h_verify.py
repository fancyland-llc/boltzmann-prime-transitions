#!/usr/bin/env python3
"""
Wave 22h-verify — Independent Verification of the Chebyshev-Character Hypothesis

The prediction from wave 22h:
  At mod 30: χ₇ (Legendre symbol mod 7) correlates with SVD Mode 2 at r=0.84
  At mod 210: χ₇ should VANISH (p=7 absorbed into modulus)
  At mod 210: χ₁₁ should APPEAR as the new dominant external character

This script tests that prediction with:
  1. Full character construction for (Z/mZ)* 
  2. Proper Legendre symbols (handling p|m correctly)
  3. All SVD modes checked against all relevant characters
  4. Maximum projection across modes (not just top-5)
  5. Product characters (χ₃χ₅, χ₃χ₇, etc.)
  6. Comparison: mod 30 vs mod 210 side by side
  7. A third modulus (mod 6) as baseline
"""

import numpy as np
from math import gcd, log
from scipy.stats import pearsonr
import time


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


def legendre_symbol(a, p):
    """Compute (a/p) for odd prime p."""
    a = a % p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def build_characters(cols, m):
    """
    Build dictionary of character vectors for the admissible columns.
    
    For modulus m = ∏ pᵢ, the GROUP characters of (Z/mZ)* are products of
    characters from each (Z/pᵢZ)*. 
    
    But the CHEBYSHEV BIAS comes from ALL primes, including those NOT in m.
    For an external prime p (p∤m), its Legendre symbol χ_p applied to the 
    column values creates a vector that is NOT a group character of (Z/mZ)*,
    but may correlate with the SVD modes of the residual.
    
    We test both internal and external characters.
    """
    chars = {}
    
    # External Legendre symbols: primes NOT dividing m
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]:
        if m % p == 0:
            # p divides m — this is an INTERNAL prime
            # The Legendre symbol mod p IS a legitimate character of (Z/mZ)*
            # (it factors through the projection (Z/mZ)* → (Z/pZ)*)
            # Label it as internal
            vec = np.array([legendre_symbol(c, p) for c in cols], dtype=float)
            if np.any(vec == 0):
                # Some column is ≡ 0 mod p — shouldn't happen if p|m 
                # and cols are coprime to m, but check
                chars[f'χ_{p} (INTERNAL, p|m)'] = vec
            else:
                chars[f'χ_{p} (INTERNAL, p|m)'] = vec
        else:
            # p does NOT divide m — external character
            vec = np.array([legendre_symbol(c, p) for c in cols], dtype=float)
            # Note: no column is ≡ 0 mod p (since gcd(c,m)=1 and p∤m... 
            # well, c could still ≡ 0 mod p). Handle zeros.
            n_zero = np.sum(vec == 0)
            chars[f'χ_{p} (external)'] = vec
    
    # Some product characters for enrichment
    for p1, p2 in [(3,5), (3,7), (5,7), (3,11), (5,11), (7,11)]:
        v1 = np.array([legendre_symbol(c, p1) for c in cols], dtype=float)
        v2 = np.array([legendre_symbol(c, p2) for c in cols], dtype=float)
        prod = v1 * v2
        # Replace any zeros with nan and skip if too many
        if np.sum(prod == 0) < len(cols) // 2:
            key = f'χ_{p1}·χ_{p2}'
            if p1 * p2 <= m or m % p1 != 0 or m % p2 != 0:
                chars[key] = prod
    
    # Geometric features
    col_arr = np.array(cols, dtype=float)
    chars['position'] = col_arr - col_arr.mean()
    chars['edge_dist'] = np.array([min(c, m - c) for c in cols], dtype=float)
    chars['sym (r vs m-r)'] = np.array([-1 if c < m/2 else +1 for c in cols], dtype=float)
    
    return chars


def analyze_modulus(m, all_primes, lo, hi, log_mid):
    """Full SVD + character analysis for one modulus."""
    cols = admissible_columns(m)
    n_cols = len(cols)
    D = distance_matrix(cols, m)
    ln_N = log_mid * np.log(10)
    
    mask = (all_primes >= lo) & (all_primes < hi)
    primes_w = all_primes[mask].tolist()
    
    T_emp = empirical_matrix(primes_w, cols, m)
    B = boltzmann_matrix(D, ln_N)
    R = T_emp - B
    
    T_null = np.ones_like(T_emp) / n_cols
    ss_tot = np.sum((T_emp - T_null)**2)
    r2 = 1 - np.sum(R**2) / ss_tot
    
    U, S_svd, Vt = np.linalg.svd(R)
    cum_var = np.cumsum(S_svd**2) / np.sum(S_svd**2) * 100
    
    chars = build_characters(cols, m)
    
    return {
        'cols': cols, 'n_cols': n_cols, 'D': D,
        'T_emp': T_emp, 'B': B, 'R': R,
        'r2': r2, 'U': U, 'S_svd': S_svd, 'Vt': Vt,
        'cum_var': cum_var, 'chars': chars,
        'n_primes': len(primes_w),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  WAVE 22h-VERIFY — CHARACTER HYPOTHESIS VERIFICATION")
    print("  Does χ₇ vanish at mod 210? Does χ₁₁ take over?")
    print("=" * 70)
    print()
    
    start = time.time()
    
    LIMIT = 10**8
    print(f"Sieving primes to {LIMIT:,}...")
    all_primes = sieve_primes(LIMIT)
    print(f"  Found {len(all_primes):,} primes")
    print()
    
    # ═══════════════════════════════════════════════════════
    # ANALYZE THREE MODULI
    # ═══════════════════════════════════════════════════════
    
    moduli = [
        (6,   "2·3",     "Baseline"),
        (30,  "2·3·5",   "The original"),
        (210, "2·3·5·7", "χ₇ absorbed?"),
    ]
    
    results = {}
    for m, factoring, label in moduli:
        print(f"{'─'*70}")
        print(f"  MOD {m} = {factoring}  ({label})")
        print(f"  φ({m}) = {len(admissible_columns(m))}")
        print(f"{'─'*70}")
        print()
        
        res = analyze_modulus(m, all_primes, 10**7, 10**8, 7.5)
        results[m] = res
        
        print(f"  R²(Boltzmann) = {res['r2']:.6f}")
        print(f"  Primes used: {res['n_primes']:,}")
        print()
        
        # SVD summary
        n_show = min(8, len(res['S_svd']))
        print(f"  SVD (top {n_show} modes):")
        for k in range(n_show):
            pct = res['S_svd'][k]**2 / np.sum(res['S_svd']**2) * 100
            print(f"    Mode {k+1}: σ={res['S_svd'][k]:.6f}  ({pct:.1f}%  cum={res['cum_var'][k]:.1f}%)")
        print()
        
        # Character correlations — comprehensive scan
        print(f"  CHARACTER CORRELATIONS (max |r| across U and V vectors):")
        print(f"  {'Character':<25} {'Best |r|':>8} {'Mode':>5} {'Vector':>7} {'Notes':<20}")
        print(f"  {'─'*70}")
        
        char_results = []
        for cname, cvec in res['chars'].items():
            best_r = 0
            best_mode = -1
            best_side = ""
            
            # Remove zeros from correlation if any
            valid = cvec != 0
            if np.sum(valid) < 4:
                continue
            
            n_modes_check = min(len(res['S_svd']), 10)
            for k in range(n_modes_check):
                # Check U (left singular vectors = row structure)
                u_vec = res['U'][:, k]
                if np.sum(valid) == len(cvec):
                    r_u = abs(np.corrcoef(u_vec, cvec)[0, 1])
                else:
                    r_u = abs(np.corrcoef(u_vec[valid], cvec[valid])[0, 1])
                if r_u > best_r:
                    best_r = r_u
                    best_mode = k + 1
                    best_side = "U (row)"
                
                # Check V (right singular vectors = column structure)
                v_vec = res['Vt'][k, :]
                if np.sum(valid) == len(cvec):
                    r_v = abs(np.corrcoef(v_vec, cvec)[0, 1])
                else:
                    r_v = abs(np.corrcoef(v_vec[valid], cvec[valid])[0, 1])
                if r_v > best_r:
                    best_r = r_v
                    best_mode = k + 1
                    best_side = "V (col)"
            
            notes = ""
            if best_r > 0.7:
                notes = "★★★ STRONG"
            elif best_r > 0.4:
                notes = "★★ MODERATE"
            elif best_r > 0.25:
                notes = "★ WEAK"
            
            char_results.append((cname, best_r, best_mode, best_side, notes))
        
        char_results.sort(key=lambda x: -x[1])
        for cname, br, bm, bs, notes in char_results:
            print(f"  {cname:<25} {br:>8.4f} {bm:>5} {bs:>7} {notes:<20}")
        print()
    
    # ═══════════════════════════════════════════════════════
    # THE KEY COMPARISON: χ₇ at mod 30 vs mod 210
    # ═══════════════════════════════════════════════════════
    
    print()
    print("=" * 70)
    print("  THE CRITICAL TEST: Does χ₇ vanish when p=7 enters the modulus?")
    print("=" * 70)
    print()
    
    # At mod 30
    res30 = results[30]
    chi7_30 = np.array([legendre_symbol(c, 7) for c in res30['cols']], dtype=float)
    # Handle the zero at c=7: 7%7=0
    valid_30 = chi7_30 != 0
    
    best_30 = 0
    best_mode_30 = -1
    for k in range(min(8, len(res30['S_svd']))):
        for vec, label in [(res30['U'][:, k], 'U'), (res30['Vt'][k, :], 'V')]:
            if np.all(valid_30):
                r = abs(np.corrcoef(vec, chi7_30)[0, 1])
            else:
                r = abs(np.corrcoef(vec[valid_30], chi7_30[valid_30])[0, 1])
            if r > best_30:
                best_30 = r
                best_mode_30 = k + 1
    
    # At mod 210
    res210 = results[210]
    chi7_210 = np.array([legendre_symbol(c, 7) for c in res210['cols']], dtype=float)
    valid_210 = chi7_210 != 0
    
    best_210 = 0
    best_mode_210 = -1
    for k in range(min(20, len(res210['S_svd']))):
        for vec, label in [(res210['U'][:, k], 'U'), (res210['Vt'][k, :], 'V')]:
            if np.all(valid_210):
                r = abs(np.corrcoef(vec, chi7_210)[0, 1])
            else:
                r = abs(np.corrcoef(vec[valid_210], chi7_210[valid_210])[0, 1])
            if r > best_210:
                best_210 = r
                best_mode_210 = k + 1
    
    print(f"  χ₇ at mod 30:  max |r| = {best_30:.4f}  (Mode {best_mode_30})")
    print(f"  χ₇ at mod 210: max |r| = {best_210:.4f}  (Mode {best_mode_210})")
    print(f"  Ratio: {best_210/best_30:.4f}")
    print(f"  VERDICT: {'χ₇ ABSORBED (vanished from residual)' if best_210 < 0.35 else 'χ₇ PERSISTS (still in residual)'}")
    print()
    
    # Now check: what REPLACED it?
    print("  What character dominates the mod 210 residual?")
    print()
    
    # Check ALL external primes at mod 210
    external_primes_210 = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    print(f"  External prime characters at mod 210:")
    print(f"  {'Prime':>6}  {'max |r|':>8}  {'Mode':>5}")
    
    for p in external_primes_210:
        chi_p = np.array([legendre_symbol(c, p) for c in res210['cols']], dtype=float)
        valid_p = chi_p != 0
        if np.sum(valid_p) < 4:
            continue
        
        best_p = 0
        best_mode_p = -1
        for k in range(min(20, len(res210['S_svd']))):
            for vec in [res210['U'][:, k], res210['Vt'][k, :]]:
                if np.all(valid_p):
                    r = abs(np.corrcoef(vec, chi_p)[0, 1])
                else:
                    r = abs(np.corrcoef(vec[valid_p], chi_p[valid_p])[0, 1])
                if r > best_p:
                    best_p = r
                    best_mode_p = k + 1
        
        star = " ★" if best_p > 0.3 else ""
        print(f"  {p:>6}  {best_p:>8.4f}  {best_mode_p:>5}{star}")
    
    # ═══════════════════════════════════════════════════════
    # DIMENSIONALITY CORRECTION
    # ═══════════════════════════════════════════════════════
    
    print()
    print("=" * 70)
    print("  DIMENSIONALITY ANALYSIS")
    print("  Is the weak mod-210 signal just dilution from 8→48 columns?")
    print("=" * 70)
    print()
    
    # At mod 30, the character has 8 entries and projects onto 8 SVD modes.
    # At mod 210, the character has 48 entries and projects onto 48 SVD modes.
    # A random 48-vector's max correlation with any of 20 SVD modes has an 
    # expected value much higher than a random 8-vector with 8 modes.
    
    # Null distribution: random sign vectors
    np.random.seed(42)
    n_trials = 10000
    
    for m in [30, 210]:
        res = results[m]
        n = res['n_cols']
        n_modes = min(10, len(res['S_svd']))
        
        max_corrs = []
        for _ in range(n_trials):
            rand_vec = np.random.choice([-1, 1], size=n).astype(float)
            best = 0
            for k in range(n_modes):
                for vec in [res['U'][:, k], res['Vt'][k, :]]:
                    r = abs(np.corrcoef(vec, rand_vec)[0, 1])
                    if r > best:
                        best = r
                best_overall = max(best, best)
            max_corrs.append(best)
        
        max_corrs = np.array(max_corrs)
        p95 = np.percentile(max_corrs, 95)
        p99 = np.percentile(max_corrs, 99)
        mean_null = np.mean(max_corrs)
        
        print(f"  Mod {m} (n={n}, {n_modes} modes checked):")
        print(f"    Null distribution (random ±1 vectors):")
        print(f"      Mean max |r| = {mean_null:.4f}")
        print(f"      95th pctile  = {p95:.4f}")
        print(f"      99th pctile  = {p99:.4f}")
        
        # Now test actual characters against this null
        print(f"    Actual character max |r| values:")
        for cname, cvec in res['chars'].items():
            if 'external' not in cname and 'INTERNAL' not in cname:
                continue
            valid = cvec != 0
            if np.sum(valid) < 4:
                continue
            best = 0
            for k in range(n_modes):
                for vec in [res['U'][:, k], res['Vt'][k, :]]:
                    if np.all(valid):
                        r = abs(np.corrcoef(vec, cvec)[0, 1])
                    else:
                        r = abs(np.corrcoef(vec[valid], cvec[valid])[0, 1])
                    if r > best:
                        best = r
            sig = ""
            if best > p99:
                sig = " ★★ (p<0.01)"
            elif best > p95:
                sig = " ★ (p<0.05)"
            print(f"      {cname:<30}  |r|={best:.4f}{sig}")
        print()
    
    # ═══════════════════════════════════════════════════════
    # PROJECTION STRENGTH: total variance explained by character
    # ═══════════════════════════════════════════════════════
    
    print("=" * 70)
    print("  TOTAL CHARACTER PROJECTION (variance in R explained by χ)")
    print("  R_χ = sum over modes of (r_χ,mode)² · σ²_mode")
    print("=" * 70)
    print()
    
    for m in [30, 210]:
        res = results[m]
        n_modes = min(len(res['S_svd']), 20)
        total_var = np.sum(res['S_svd']**2)
        
        print(f"  Mod {m}:")
        
        proj_results = []
        for cname, cvec in res['chars'].items():
            if 'position' in cname or 'edge' in cname or 'sym' in cname:
                continue
            valid = cvec != 0
            if np.sum(valid) < 4:
                continue
            
            # Normalize the character vector
            cv_clean = cvec.copy()
            cv_clean[~valid] = 0
            cv_norm = cv_clean / (np.linalg.norm(cv_clean) + 1e-15)
            
            # Project R onto χ(a)·χ(b)ᵀ outer product
            # This measures how much of R is explained by the tensor χ⊗χ
            R_mat = res['R']
            projection = np.sum(R_mat * np.outer(cv_norm, cv_norm))
            
            # Also compute: how much variance is in the χ-direction
            # across all SVD modes
            var_explained = 0
            for k in range(n_modes):
                r_u = np.dot(res['U'][:, k], cv_norm)
                r_v = np.dot(res['Vt'][k, :], cv_norm)
                var_explained += (r_u * r_v * res['S_svd'][k])**2
            
            frac = var_explained / total_var * 100 if total_var > 0 else 0
            proj_results.append((cname, projection, var_explained, frac))
        
        proj_results.sort(key=lambda x: -x[3])
        print(f"    {'Character':<30} {'Projection':>10} {'Var expl':>10} {'% of ||R||²':>12}")
        for cname, proj, ve, frac in proj_results[:15]:
            print(f"    {cname:<30} {proj:>+10.6f} {ve:>10.8f} {frac:>11.4f}%")
        print()
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
