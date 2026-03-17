#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  WAVE 22c — THE ZERO-PARAMETER BOLTZMANN MODEL                        ║
║                                                                        ║
║  The insight: self-avoidance is NOT a separate parameter.             ║
║  Going from column a back to column a requires a gap ≥ m.            ║
║  The minimum gap for self-return IS m (the modulus).                  ║
║                                                                        ║
║  So: d(a,a) = m, not 0: the self-transition goes "around the cycle." ║
║                                                                        ║
║  This gives a ZERO FREE PARAMETER model:                              ║
║    T(a→b) = exp(-d(a,b) / ln(N)) / Z                                 ║
║  where                                                                 ║
║    d(a,b) = (b-a) mod m  if b ≠ a                                    ║
║    d(a,a) = m            (full cycle for self-return)                 ║
║    λ = 1/ln(N)           (from the PNT — no fitting)                 ║
║                                                                        ║
║  Physical derivation:                                                  ║
║    P(gap=g) ≈ exp(-g/ln(N))/ln(N)                                    ║
║    P(a→b) ∝ Σ_{k≥0} exp(-(d + km)/ln(N))   [b≠a, d>0]              ║
║    P(a→a) ∝ Σ_{k≥1} exp(-km/ln(N))         [must advance ≥m]        ║
║                                                                        ║
║  The self-avoidance ratio:                                            ║
║    P(a→a) / P(a→b_nearest) = exp(-(m-d_min)/ln(N))                   ║
║    At m=30, d_min=2, ln(N)=17.27: ratio = 0.198                      ║
║    Empirical ratio: 0.045/0.234 = 0.192  ← MATCH                     ║
║                                                                        ║
║  Author: Tony M. (Architect) + Claude Opus 4 (Correspondent)          ║
║  Date: March 17, 2026                                                  ║
║  EPOCH 004, Wave 22c                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from math import gcd, log, pi, atan2
from scipy.stats import pearsonr
from collections import defaultdict
import json
import time
import os

# ═══════════════════════════════════════════════════════
# SIEVE + HELPERS
# ═══════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════
# THE CORRECTED DISTANCE: d_self = m
# ═══════════════════════════════════════════════════════

def corrected_distance_matrix(cols, m):
    """
    Forward residue distance with the physical correction:
    d(a,b) = (b-a) mod m  for b ≠ a
    d(a,a) = m             (self-return requires full cycle)
    """
    n = len(cols)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = m  # Self-return distance = full modulus
            else:
                D[i, j] = (cols[j] - cols[i]) % m
    return D


# ═══════════════════════════════════════════════════════
# ZERO-PARAMETER MODEL: λ = 1/ln(N), d corrected
# ═══════════════════════════════════════════════════════

def zero_param_boltzmann(ln_N, cols, m):
    """
    Zero free parameters:
    T[i,j] = exp(-d(i,j) / ln(N)) / Z_i
    With d_self = m and λ = 1/ln(N).
    """
    D = corrected_distance_matrix(cols, m)
    T = np.exp(-D / ln_N)
    row_sums = T.sum(axis=1, keepdims=True)
    return T / row_sums


def fitted_boltzmann(lam, cols, m):
    """
    One-parameter model where only λ is fit (d corrected).
    """
    D = corrected_distance_matrix(cols, m)
    T = np.exp(-lam * D)
    row_sums = T.sum(axis=1, keepdims=True)
    return T / row_sums


# ═══════════════════════════════════════════════════════
# MEASURE AND COMPARE
# ═══════════════════════════════════════════════════════

def measure_and_fit(primes_in_range, cols, m, log10_mid):
    n_cols = len(cols)
    ln_N = log10_mid * np.log(10)
    
    working = [p for p in primes_in_range if p > m]
    
    # Build empirical transition matrix
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
    
    # ─── ZERO-PARAMETER MODEL ───
    T_zero = zero_param_boltzmann(ln_N, cols, m)
    
    T_null = np.ones_like(T_emp) / n_cols
    ss_tot = np.sum((T_emp - T_null)**2)
    
    ss_res_zero = np.sum((T_emp - T_zero)**2)
    r2_zero = 1 - ss_res_zero / ss_tot
    frob_zero = float(np.linalg.norm(T_emp - T_zero, 'fro'))
    resid_zero = T_emp - T_zero
    max_err_zero = float(np.max(np.abs(resid_zero)))
    mean_err_zero = float(np.mean(np.abs(resid_zero)))
    
    # ─── ONE-PARAMETER FIT (λ free, d corrected) ───
    from scipy.optimize import minimize_scalar
    
    def objective(lam):
        T_model = fitted_boltzmann(lam, cols, m)
        return np.linalg.norm(T_emp - T_model, 'fro')
    
    result = minimize_scalar(objective, bounds=(0.001, 2.0), method='bounded')
    lam_opt = result.x
    T_fit = fitted_boltzmann(lam_opt, cols, m)
    
    ss_res_fit = np.sum((T_emp - T_fit)**2)
    r2_fit = 1 - ss_res_fit / ss_tot
    resid_fit = T_emp - T_fit
    
    # ─── EIGENSTRUCTURE ───
    eigenvalues = np.linalg.eigvals(T_emp)
    eig_order = np.argsort(-np.abs(eigenvalues))
    eigenvalues_sorted = eigenvalues[eig_order]
    
    complex_eigs = [(e, abs(e), atan2(e.imag, e.real) * 180 / pi) 
                   for e in eigenvalues_sorted 
                   if abs(e.imag) > 1e-10]
    
    leading_mag = 0
    leading_angle = 0
    if complex_eigs:
        best = max(complex_eigs, key=lambda x: x[1])
        leading_mag = best[1]
        leading_angle = best[2]
    
    return {
        "n_primes": len(working),
        "log10_mid": log10_mid,
        "ln_N": ln_N,
        
        # Zero-parameter model
        "lambda_pnt": 1.0 / ln_N,
        "r2_zero_param": float(r2_zero),
        "frob_zero": float(frob_zero),
        "max_err_zero": max_err_zero,
        "mean_err_zero": mean_err_zero,
        
        # One-parameter fit
        "lambda_fitted": float(lam_opt),
        "r2_fitted": float(r2_fit),
        "lambda_ratio": float(lam_opt * ln_N),  # Should → 1 if λ = 1/ln(N)
        "frob_fitted": float(result.fun),
        "max_err_fitted": float(np.max(np.abs(resid_fit))),
        "mean_err_fitted": float(np.mean(np.abs(resid_fit))),
        
        # Eigenstructure
        "leading_mag": leading_mag,
        "leading_angle": leading_angle,
        
        # Matrices (for comparison)
        "T_empirical": T_emp.tolist(),
        "T_zero_param": T_zero.tolist(),
        "T_fitted": T_fit.tolist(),
        "residual_zero": resid_zero.tolist(),
    }


def run_experiment(max_prime=10**8):
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  WAVE 22c — THE ZERO-PARAMETER BOLTZMANN MODEL             ║")
    print("║                                                            ║")
    print("║  d(a,b) = (b-a) mod m   [forward distance]                ║")
    print("║  d(a,a) = m             [self = full cycle]                ║")
    print("║  λ = 1/ln(N)            [PNT — no fitting]                ║")
    print("║                                                            ║")
    print("║  ZERO free parameters.                                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    start_time = time.time()
    
    print(f"Sieving primes up to {max_prime:,}...")
    all_primes = sieve_primes(max_prime)
    print(f"  Found {len(all_primes):,} primes")
    print()
    
    octave_specs = [
        (10**3, 10**4, "10^3–10^4", 3.5),
        (10**4, 10**5, "10^4–10^5", 4.5),
        (10**5, 10**6, "10^5–10^6", 5.5),
        (10**6, 10**7, "10^6–10^7", 6.5),
        (10**7, 10**8, "10^7–10^8", 7.5),
    ]
    
    all_results = {}
    
    for m in [6, 30, 210]:
        cols = admissible_columns(m)
        n_cols = len(cols)
        
        D = corrected_distance_matrix(cols, m)
        
        print(f"{'='*70}")
        print(f"  MODULUS m = {m}  |  φ(m) = {n_cols} columns")
        print(f"  Columns: {cols[:16]}{'...' if n_cols > 16 else ''}")
        print(f"{'='*70}")
        
        if n_cols <= 8:
            print(f"\n  Corrected distance matrix (d_self = {m}):")
            header = "       " + "".join(f"{c:>5}" for c in cols)
            print(f"    {header}")
            for i, ci in enumerate(cols):
                row_str = f"    {ci:>3} "
                for j in range(n_cols):
                    row_str += f" {int(D[i,j]):>4}"
                print(row_str)
        print()
        
        results = []
        log_ns = []
        lambdas_pnt = []
        lambdas_fit = []
        ratios = []
        r2_zeros = []
        r2_fits = []
        angles = []
        mags = []
        
        for lo, hi, label, log_mid in octave_specs:
            mask = (all_primes >= lo) & (all_primes < hi)
            primes_in_range = all_primes[mask].tolist()
            
            if len(primes_in_range) < 100:
                continue
            
            r = measure_and_fit(primes_in_range, cols, m, log_mid)
            results.append(r)
            
            log_ns.append(log_mid)
            lambdas_pnt.append(r['lambda_pnt'])
            lambdas_fit.append(r['lambda_fitted'])
            ratios.append(r['lambda_ratio'])
            r2_zeros.append(r['r2_zero_param'])
            r2_fits.append(r['r2_fitted'])
            if r['leading_angle'] != 0:
                angles.append(r['leading_angle'])
            mags.append(r['leading_mag'])
            
            print(f"  {label} ({r['n_primes']:,} primes):")
            print(f"    ┌─ ZERO-PARAMETER MODEL (λ=1/ln(N), d_self=m):")
            print(f"    │  λ = 1/ln(N) = {r['lambda_pnt']:.6f}")
            print(f"    │  R² = {r['r2_zero_param']:.6f}")
            print(f"    │  Max error = {r['max_err_zero']:.6f}")
            print(f"    │  Mean error = {r['mean_err_zero']:.6f}")
            print(f"    │")
            print(f"    ├─ FITTED λ (with corrected d):")
            print(f"    │  λ = {r['lambda_fitted']:.6f}")
            print(f"    │  R² = {r['r2_fitted']:.6f}")
            print(f"    │  λ·ln(N) = {r['lambda_ratio']:.4f} (should → 1.0)")
            print(f"    │  Max error = {r['max_err_fitted']:.6f}")
            print(f"    │")
            print(f"    └─ EIGENSTRUCTURE:")
            if r['leading_mag'] > 0:
                print(f"       |λ₁| = {r['leading_mag']:.6f}, θ = {r['leading_angle']:.2f}°")
            else:
                print(f"       No complex eigenvalues")
            print()
        
        # ═══════════════════════════════════
        # SUMMARY TABLES
        # ═══════════════════════════════════
        
        print(f"  {'─'*60}")
        print(f"  SUMMARY: mod {m}")
        print(f"  {'─'*60}")
        print()
        
        print(f"    {'Scale':<12} {'λ_PNT':>10} {'λ_fit':>10} {'λ·ln(N)':>8} {'R²_0':>10} {'R²_fit':>10}")
        print(f"    {'─'*62}")
        for i, logn in enumerate(log_ns):
            print(f"    log₁₀N={logn:<4.1f} {lambdas_pnt[i]:>10.6f} {lambdas_fit[i]:>10.6f} "
                  f"{ratios[i]:>8.4f} {r2_zeros[i]:>10.6f} {r2_fits[i]:>10.6f}")
        
        print()
        print(f"    Mean λ·ln(N) = {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
        print(f"    Mean R²(zero-param) = {np.mean(r2_zeros):.6f}")
        print(f"    Mean R²(fitted) = {np.mean(r2_fits):.6f}")
        print()
        
        # Trend in λ·ln(N) — is it converging to 1?
        log_arr = np.array(log_ns)
        ratio_arr = np.array(ratios)
        
        if len(log_ns) >= 3:
            coeffs = np.polyfit(log_arr, ratio_arr, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            # Extrapolate to large N
            for target in [10, 20, 50, 100]:
                pred = slope * target + intercept
                print(f"    Extrapolation: log₁₀N={target:>4} → λ·ln(N) ≈ {pred:.4f}")
            print(f"    Trend: λ·ln(N) = {slope:.4f}·log₁₀N + {intercept:.4f}")
        print()
        
        # ═══════════════════════════════════
        # MATRIX COMPARISON AT LARGEST SCALE
        # ═══════════════════════════════════
        if results and n_cols <= 8:
            best = results[-1]
            T_emp = np.array(best['T_empirical'])
            T_zp = np.array(best['T_zero_param'])
            T_ft = np.array(best['T_fitted'])
            resid = np.array(best['residual_zero'])
            
            print(f"  {'─'*60}")
            print(f"  MATRIX COMPARISON (mod {m}, scale {octave_specs[-1][2]})")
            print(f"  {'─'*60}")
            print()
            
            for name, T_show, r2 in [
                ("EMPIRICAL", T_emp, 1.0),
                (f"ZERO-PARAM (λ=1/ln(N)={best['lambda_pnt']:.4f})", T_zp, best['r2_zero_param']),
                (f"FITTED (λ={best['lambda_fitted']:.4f})", T_ft, best['r2_fitted']),
            ]:
                print(f"    {name} (R² = {r2:.4f}):")
                header = "         " + "".join(f"{c:>7}" for c in cols)
                print(f"    {header}")
                for i, ci in enumerate(cols):
                    row_str = f"      {ci:>3} "
                    for j in range(n_cols):
                        row_str += f" {T_show[i, j]:5.3f} "
                    print(row_str)
                print()
            
            print(f"    RESIDUAL (Empirical - Zero-param):")
            header = "         " + "".join(f"{c:>7}" for c in cols)
            print(f"    {header}")
            for i, ci in enumerate(cols):
                row_str = f"      {ci:>3} "
                for j in range(n_cols):
                    d = resid[i, j]
                    marker = "*" if abs(d) > 0.005 else " "
                    row_str += f" {d:+5.3f}{marker}"
                print(row_str)
            print(f"    (* = |residual| > 0.005)")
            
            # SVD of residuals
            U, S, Vt = np.linalg.svd(resid)
            print(f"\n    Residual SVD: σ = [{', '.join(f'{s:.4f}' for s in S[:5])}]")
            
            # What fraction of variance is explained?
            T_null_local = np.ones_like(T_emp) / n_cols
            var_emp = np.sum((T_emp - T_null_local)**2)
            var_resid = np.sum(resid**2)
            print(f"    Variance explained: {(1-var_resid/var_emp)*100:.2f}%")
            print()
        
        # ═══════════════════════════════════
        # EIGENVALUE SCALING
        # ═══════════════════════════════════
        if len(mags) >= 3:
            mags_arr = np.array(mags)
            valid = mags_arr > 0
            if np.sum(valid) >= 3:
                log_mags = np.log(mags_arr[valid])
                log_logns = np.log(log_arr[valid])
                coeffs_m = np.polyfit(log_logns, log_mags, 1)
                beta = -coeffs_m[0]
                a_m = np.exp(coeffs_m[1])
                
                print(f"  Spiral: |λ₁| = {a_m:.4f} · log₁₀(N)^(-{beta:.4f})")
                
                pred_mags = a_m * log_arr[valid]**(-beta)
                ss_res = np.sum((mags_arr[valid] - pred_mags)**2)
                ss_tot = np.sum((mags_arr[valid] - np.mean(mags_arr[valid]))**2)
                r2_mag = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                print(f"  R² = {r2_mag:.6f}")
                
                for tgt in [10, 50, 100, 308]:
                    print(f"  |λ₁|(log₁₀N={tgt}) ≈ {a_m * tgt**(-beta):.4f}")
                print()
        
        # ═══════════════════════════════════
        # θ SCALING  
        # ═══════════════════════════════════
        if len(angles) >= 3:
            angles_arr = np.array(angles)
            angle_logns = log_arr[:len(angles_arr)]
            
            print(f"  Spiral angle: θ drift = {np.polyfit(angle_logns, angles_arr, 1)[0]:.4f}°/decade")
            
            # Does θ → 360/n_cols · k for integer k?
            expected_uniform = 360.0 / n_cols
            for logn, ang in zip(log_ns, angles):
                k = ang / expected_uniform
                print(f"    log₁₀N={logn:.1f}: θ = {ang:.2f}°, θ/(360/{n_cols}) = {k:.3f}")
            print()
        
        all_results[f"mod_{m}"] = {
            "modulus": m,
            "phi_m": n_cols,
            "columns": cols,
            "scales": results,
            "ratios_lambda_lnN": ratios,
            "r2_zero_params": r2_zeros,
            "r2_fitted": r2_fits,
        }
    
    elapsed = time.time() - start_time
    
    # ═══════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════
    print()
    print(f"{'='*70}")
    print(f"  GRAND SUMMARY — ZERO-PARAMETER BOLTZMANN MODEL")
    print(f"  T(a→b) = exp(-d(a,b) / ln(N)) / Z")
    print(f"  d_self = m, d_other = (b-a) mod m")
    print(f"  λ = 1/ln(N) from PNT. ZERO fitted parameters.")
    print(f"{'='*70}")
    print()
    
    for m_key, data in all_results.items():
        m = data['modulus']
        r2s = data['r2_zero_params']
        rats = data['ratios_lambda_lnN']
        
        mean_r2 = np.mean(r2s) if r2s else 0
        best_r2 = max(r2s) if r2s else 0
        mean_rat = np.mean(rats) if rats else 0
        
        verdict = (
            "████ EXCELLENT ████" if mean_r2 > 0.95 else
            "███ STRONG ███" if mean_r2 > 0.90 else
            "██ GOOD ██" if mean_r2 > 0.80 else
            "█ PARTIAL █" if mean_r2 > 0.50 else
            "░ WEAK ░"
        )
        
        print(f"  mod {m:>3}: R²(zero-param) = {mean_r2:.4f} (best: {best_r2:.4f})  "
              f"λ·ln(N) = {mean_rat:.4f}  [{verdict}]")
    
    print()
    print(f"Total time: {elapsed:.1f}s")
    
    # Save
    output = {
        "experiment": "Wave 22c — Zero-Parameter Boltzmann",
        "timestamp": int(time.time() * 1000),
        "elapsed_seconds": elapsed,
        "max_prime": int(max_prime),
        "results": all_results,
    }
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                           f'wave22c_boltzmann_{int(time.time()*1000)}.json')
    out_path = os.path.normpath(out_path)
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {out_path}")
    
    return output


if __name__ == "__main__":
    run_experiment(10**8)
