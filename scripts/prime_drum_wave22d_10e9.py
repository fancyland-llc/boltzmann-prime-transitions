#!/usr/bin/env python3
"""
Wave 22d — Extend Boltzmann verification to 10^9.
Adds the 10^8-10^9 window to the convergence table.
Uses segmented sieve to handle memory for 10^9.
"""

import numpy as np
from math import gcd, log, sqrt
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
from collections import defaultdict
import json
import time
import os

def segmented_sieve(limit):
    """Segmented sieve of Eratosthenes for large limits."""
    segment_size = 10**6
    
    # Small primes up to sqrt(limit)
    sqrt_limit = int(sqrt(limit)) + 1
    is_prime_small = np.ones(sqrt_limit + 1, dtype=bool)
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, int(sqrt(sqrt_limit)) + 1):
        if is_prime_small[i]:
            is_prime_small[i*i::i] = False
    small_primes = np.where(is_prime_small)[0]
    
    all_primes = list(small_primes)
    
    # Sieve segments
    for low in range(sqrt_limit + 1, limit + 1, segment_size):
        high = min(low + segment_size - 1, limit)
        seg = np.ones(high - low + 1, dtype=bool)
        
        for p in small_primes:
            if p < 2:
                continue
            start = ((low + p - 1) // p) * p
            if start < p * p:
                start = p * p
            if start > high:
                continue
            seg[start - low::p] = False
        
        primes_in_seg = np.where(seg)[0] + low
        all_primes.extend(primes_in_seg.tolist())
    
    return np.array(all_primes, dtype=np.int64)

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

def zero_param_boltzmann(ln_N, cols, m):
    D = corrected_distance_matrix(cols, m)
    T = np.exp(-D / ln_N)
    row_sums = T.sum(axis=1, keepdims=True)
    return T / row_sums

def fitted_boltzmann(lam, cols, m):
    D = corrected_distance_matrix(cols, m)
    T = np.exp(-lam * D)
    row_sums = T.sum(axis=1, keepdims=True)
    return T / row_sums

def measure_window(primes_in_range, cols, m, log10_mid):
    n_cols = len(cols)
    ln_N = log10_mid * np.log(10)
    
    working = [p for p in primes_in_range if p > m]
    
    T = np.zeros((n_cols, n_cols))
    col_idx = {c: i for i, c in enumerate(cols)}
    for k in range(len(working) - 1):
        r_from = working[k] % m
        r_to = working[k + 1] % m
        if r_from in col_idx and r_to in col_idx:
            T[col_idx[r_from], col_idx[r_to]] += 1
    
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_emp = T / row_sums
    
    # Zero-parameter model
    T_zero = zero_param_boltzmann(ln_N, cols, m)
    T_null = np.ones_like(T_emp) / n_cols
    ss_tot = np.sum((T_emp - T_null)**2)
    ss_res_zero = np.sum((T_emp - T_zero)**2)
    r2_zero = 1 - ss_res_zero / ss_tot
    resid_zero = T_emp - T_zero
    
    # One-parameter fit
    def objective(lam):
        T_model = fitted_boltzmann(lam, cols, m)
        return np.linalg.norm(T_emp - T_model, 'fro')
    
    result = minimize_scalar(objective, bounds=(0.001, 2.0), method='bounded')
    lam_opt = result.x
    T_fit = fitted_boltzmann(lam_opt, cols, m)
    ss_res_fit = np.sum((T_emp - T_fit)**2)
    r2_fit = 1 - ss_res_fit / ss_tot
    
    return {
        "n_primes": len(working),
        "log10_mid": log10_mid,
        "ln_N": ln_N,
        "lambda_pnt": 1.0 / ln_N,
        "r2_zero_param": float(r2_zero),
        "lambda_fitted": float(lam_opt),
        "r2_fitted": float(r2_fit),
        "lambda_ratio": float(lam_opt * ln_N),
        "max_err_zero": float(np.max(np.abs(resid_zero))),
        "mean_err_zero": float(np.mean(np.abs(resid_zero))),
        "T_empirical": T_emp.tolist(),
        "T_zero_param": T_zero.tolist(),
        "residual_zero": resid_zero.tolist(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  WAVE 22d — EXTENDING TO 10^9")
    print("  Adding the 10^8-10^9 window to the convergence table")
    print("=" * 70)
    print()
    
    start = time.time()
    
    print("Sieving primes to 10^9 (segmented sieve)...")
    all_primes = segmented_sieve(10**9)
    print(f"  Found {len(all_primes):,} primes in {time.time()-start:.1f}s")
    print()
    
    # All windows including the new 10^8-10^9
    windows = [
        (10**3, 10**4, 3.5),
        (10**4, 10**5, 4.5),
        (10**5, 10**6, 5.5),
        (10**6, 10**7, 6.5),
        (10**7, 10**8, 7.5),
        (10**8, 10**9, 8.5),  # NEW
    ]
    
    for m in [30, 210]:
        cols = admissible_columns(m)
        n_cols = len(cols)
        
        print(f"{'='*60}")
        print(f"  MODULUS {m} (phi={n_cols})")
        print(f"{'='*60}")
        print()
        
        print(f"  {'Scale':<14} {'ln(N)':>7} {'lam_PNT':>10} {'lam_fit':>10} "
              f"{'lam*lnN':>8} {'R2_zero':>10} {'R2_fit':>10}")
        print(f"  {'-'*72}")
        
        all_results = []
        for lo, hi, log_mid in windows:
            mask = (all_primes >= lo) & (all_primes < hi)
            primes_in_range = all_primes[mask].tolist()
            if len(primes_in_range) < 100:
                continue
            
            r = measure_window(primes_in_range, cols, m, log_mid)
            all_results.append(r)
            
            marker = " <<<" if log_mid == 8.5 else ""
            print(f"  10^{log_mid-0.5:.0f}-10^{log_mid+0.5:.0f}  "
                  f"{r['ln_N']:>7.2f} {r['lambda_pnt']:>10.5f} {r['lambda_fitted']:>10.5f} "
                  f"{r['lambda_ratio']:>8.4f} {r['r2_zero_param']:>10.6f} {r['r2_fitted']:>10.6f}{marker}")
        
        print()
        
        # Show the new window's matrix if mod 30
        if m == 30 and all_results:
            new = all_results[-1]  # 10^8-10^9
            print(f"  NEW WINDOW 10^8-10^9: {new['n_primes']:,} primes")
            print(f"  R²(zero-param) = {new['r2_zero_param']:.6f}")
            print(f"  λ·ln(N) = {new['lambda_ratio']:.4f}")
            print(f"  Max err = {new['max_err_zero']:.6f}")
            print(f"  Mean err = {new['mean_err_zero']:.6f}")
            print()
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Save results
    output = {
        "experiment": "Wave 22d — 10^9 extension",
        "timestamp": int(time.time() * 1000),
        "elapsed_seconds": elapsed,
        "max_prime": 10**9,
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                           f'wave22d_10e9_{int(time.time()*1000)}.json')
    out_path = os.path.normpath(out_path)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {out_path}")
