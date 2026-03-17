#!/usr/bin/env python3
"""
Wave 22f — The modulus scaling law.
R²(∞) vs φ(m): does the Boltzmann model become exact as m → ∞?

Tests m = 6, 30, 210, 2310 at the same scales.
m=2310 has φ(2310) = 480 columns.
"""

import numpy as np
from math import gcd, log
from scipy.optimize import minimize_scalar
import json
import time
import os


def sieve_primes(limit):
    if limit < 2:
        return []
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
    r2_zero = 1 - ss_res_zero / ss_tot if ss_tot > 0 else 0
    resid_zero = T_emp - T_zero
    
    # One-parameter fit
    def objective(lam):
        T_model = fitted_boltzmann(lam, cols, m)
        return np.linalg.norm(T_emp - T_model, 'fro')
    
    result = minimize_scalar(objective, bounds=(0.001, 2.0), method='bounded')
    lam_opt = result.x
    T_fit = fitted_boltzmann(lam_opt, cols, m)
    ss_res_fit = np.sum((T_emp - T_fit)**2)
    r2_fit = 1 - ss_res_fit / ss_tot if ss_tot > 0 else 0
    
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
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  WAVE 22f — THE MODULUS SCALING LAW")
    print("  R²(∞) vs φ(m): does the model become exact as m → ∞?")
    print("=" * 70)
    print()
    
    start = time.time()
    
    # Sieve to 10^8 (covers all windows we need for the scaling law)
    LIMIT = 10**8
    print(f"Sieving primes to {LIMIT:,}...")
    all_primes = sieve_primes(LIMIT)
    print(f"  Found {len(all_primes):,} primes in {time.time()-start:.1f}s")
    print()
    
    windows = [
        (10**3,  10**4,  3.5),
        (10**4,  10**5,  4.5),
        (10**5,  10**6,  5.5),
        (10**6,  10**7,  6.5),
        (10**7,  10**8,  7.5),
    ]
    
    moduli = [6, 30, 210, 2310]
    
    all_results = {}
    
    for m in moduli:
        cols = admissible_columns(m)
        n_cols = len(cols)
        phi_m = n_cols
        
        t0 = time.time()
        
        # Precompute distance matrix once
        D = corrected_distance_matrix(cols, m)
        
        print(f"{'='*70}")
        print(f"  m = {m}, φ(m) = {phi_m}, φ(m)/m = {phi_m/m:.4f}")
        print(f"  Distance matrix: {n_cols}×{n_cols} ({n_cols**2:,} entries)")
        print(f"{'='*70}")
        print()
        
        print(f"  {'Scale':<14} {'ln(N)':>7} {'lam_PNT':>10} {'lam_fit':>10} "
              f"{'lam*lnN':>8} {'R2_zero':>10} {'R2_fit':>10} {'n_primes':>10}")
        print(f"  {'-'*82}")
        
        results_for_m = []
        for lo, hi, log_mid in windows:
            mask = (all_primes >= lo) & (all_primes < hi)
            primes_in_range = all_primes[mask].tolist()
            if len(primes_in_range) < 100:
                print(f"  10^{log_mid-0.5:.0f}-10^{log_mid+0.5:.0f}  SKIPPED ({len(primes_in_range)} primes)")
                continue
            
            r = measure_window(primes_in_range, cols, m, log_mid)
            results_for_m.append(r)
            
            print(f"  10^{log_mid-0.5:.0f}-10^{log_mid+0.5:.0f}  "
                  f"{r['ln_N']:>7.2f} {r['lambda_pnt']:>10.5f} {r['lambda_fitted']:>10.5f} "
                  f"{r['lambda_ratio']:>8.4f} {r['r2_zero_param']:>10.6f} {r['r2_fitted']:>10.6f}"
                  f" {r['n_primes']:>10,}")
        
        elapsed_m = time.time() - t0
        print(f"\n  Time for m={m}: {elapsed_m:.1f}s")
        print()
        
        all_results[f"mod_{m}"] = {
            "modulus": m,
            "phi_m": phi_m,
            "phi_over_m": phi_m / m,
            "results": results_for_m,
        }
    
    # ═══════════════════════════════════════════
    # THE SCALING LAW TABLE
    # ═══════════════════════════════════════════
    
    print()
    print("=" * 70)
    print("  THE SCALING LAW: R² vs φ(m) at largest measured scale")
    print("=" * 70)
    print()
    
    print(f"  {'m':>6} {'φ(m)':>6} {'φ(m)/m':>8} {'R²(zero)':>10} {'R²(fit)':>10} "
          f"{'λ·ln(N)':>10} {'1-R²':>10}")
    print(f"  {'-'*64}")
    
    phi_vals = []
    r2_vals = []
    one_minus_r2 = []
    
    for m in moduli:
        key = f"mod_{m}"
        data = all_results[key]
        if data["results"]:
            best = data["results"][-1]  # largest scale
            r2 = best["r2_zero_param"]
            residual = 1 - r2
            phi = data["phi_m"]
            
            phi_vals.append(phi)
            r2_vals.append(r2)
            one_minus_r2.append(residual)
            
            print(f"  {m:>6} {phi:>6} {data['phi_over_m']:>8.4f} "
                  f"{r2:>10.6f} {best['r2_fitted']:>10.6f} "
                  f"{best['lambda_ratio']:>10.4f} {residual:>10.6f}")
    
    print()
    
    # Fit: 1 - R² = A / φ(m)^α ?
    if len(phi_vals) >= 3:
        log_phi = np.log(np.array(phi_vals, dtype=float))
        log_resid = np.log(np.array(one_minus_r2))
        
        coeffs = np.polyfit(log_phi, log_resid, 1)
        alpha = -coeffs[0]
        A = np.exp(coeffs[1])
        
        # Predicted values
        print(f"  POWER LAW FIT: 1 - R² = {A:.4f} / φ(m)^{alpha:.4f}")
        print()
        
        for phi, actual in zip(phi_vals, one_minus_r2):
            predicted = A / phi**alpha
            print(f"    φ={phi:>4}: actual 1-R² = {actual:.6f}, "
                  f"predicted = {predicted:.6f}, ratio = {actual/predicted:.3f}")
        
        print()
        
        # Extrapolation
        for m_test, phi_test in [(2310*13, 480*12), (30030, 5760)]:
            pred_resid = A / phi_test**alpha
            pred_r2 = 1 - pred_resid
            print(f"    Extrapolation: φ={phi_test:>5} → R² ≈ {pred_r2:.6f} "
                  f"(1-R² = {pred_resid:.6f})")
        
        # Does R² → 1 as φ → ∞?
        print()
        if alpha > 0:
            print(f"  ✓ α = {alpha:.4f} > 0 → R² → 1 as φ(m) → ∞")
            print(f"    The Boltzmann model is asymptotically exact in the large-modulus limit.")
        else:
            print(f"  ✗ α = {alpha:.4f} ≤ 0 → R² does NOT approach 1")
        
        # Also fit: 1 - R² vs ln(φ(m))
        coeffs_log = np.polyfit(log_phi, one_minus_r2, 1)
        print(f"\n  LINEAR IN ln(φ): 1 - R² = {coeffs_log[0]:.6f}·ln(φ) + {coeffs_log[1]:.6f}")
        r2_logfit = 1 - np.sum((one_minus_r2 - np.polyval(coeffs_log, log_phi))**2) / \
                        np.sum((one_minus_r2 - np.mean(one_minus_r2))**2)
        print(f"  R² of this fit: {r2_logfit:.6f}")
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Save
    output = {
        "experiment": "Wave 22f — Modulus Scaling Law",
        "timestamp": int(time.time() * 1000),
        "elapsed_seconds": elapsed,
        "max_prime": int(LIMIT),
        "results": all_results,
        "scaling_law": {
            "phi_values": phi_vals,
            "r2_values": r2_vals,
            "one_minus_r2": one_minus_r2,
        }
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                           f'wave22f_scaling_{int(time.time()*1000)}.json')
    out_path = os.path.normpath(out_path)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {out_path}")
