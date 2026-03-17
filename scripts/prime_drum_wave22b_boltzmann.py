#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  WAVE 22b — THE CORRECTED BOLTZMANN FIT                                ║
║                                                                        ║
║  Wave 22a failed because it used column INDEX distance (0,1,...,7).    ║
║  The columns are NOT equally spaced on the number line.                ║
║  Column gaps mod 30: {6, 4, 2, 4, 2, 4, 6, 2}.                       ║
║                                                                        ║
║  The correct model:                                                    ║
║    T(a→b) ∝ exp(-d_residue(a,b) / ln(N))                             ║
║  where d_residue = (b - a) mod 30 is the ACTUAL number-line distance. ║
║                                                                        ║
║  Physical derivation:                                                  ║
║    P(gap = g) ≈ exp(-g/ln(N)) / ln(N)     [Cramér-like]              ║
║    P(a→b) ∝ Σ_k exp(-(d + k·30)/ln(N))   [sum over wraps]           ║
║           = exp(-d/ln(N)) / (1 - exp(-30/ln(N)))                      ║
║                                                                        ║
║  This gives λ = 1/ln(N) EXACTLY, with the PNT as the driver.         ║
║                                                                        ║
║  Three predictions (corrected):                                        ║
║  1. T_ij = exp(-d_residue(i,j) / ln(N)) / Z   with R² >> 0.95       ║
║  2. θ(N) = constant or has interpretable drift                        ║
║  3. |λ₁(N)| scaling — spiral persists or dies                        ║
║                                                                        ║
║  Author: Tony M. (Architect) + Claude Opus 4 (Correspondent)          ║
║  Date: March 17, 2026                                                  ║
║  EPOCH 004, Wave 22b                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from math import gcd, log, pi, atan2, exp
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import pearsonr
from collections import defaultdict
import json
import time
import os

# ═══════════════════════════════════════════════════════
# SIEVE
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
# RESIDUE DISTANCE (the correct metric)
# ═══════════════════════════════════════════════════════

def residue_forward_distance(a, b, m):
    """
    Forward distance from residue a to residue b on the number line, mod m.
    This is (b - a) mod m — the actual gap needed.
    """
    return (b - a) % m


def build_distance_matrix(cols, m):
    """Build matrix of forward residue distances between all column pairs."""
    n = len(cols)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = residue_forward_distance(cols[i], cols[j], m)
    return D


# ═══════════════════════════════════════════════════════
# BOLTZMANN MODEL WITH RESIDUE DISTANCE
# ═══════════════════════════════════════════════════════

def boltzmann_matrix_residue(lam, cols, m):
    """
    Boltzmann transition matrix using actual residue distances:
    T[i,j] = exp(-λ * d_residue(cols[i], cols[j])) / Z_i
    """
    n = len(cols)
    D = build_distance_matrix(cols, m)
    T = np.exp(-lam * D)
    # Normalize each row
    row_sums = T.sum(axis=1, keepdims=True)
    return T / row_sums


def fit_lambda_residue(T_empirical, cols, m):
    """
    Fit λ by minimizing Frobenius norm between empirical and Boltzmann matrix.
    """
    def objective(lam):
        T_model = boltzmann_matrix_residue(lam, cols, m)
        return np.linalg.norm(T_empirical - T_model, 'fro')
    
    # Search over λ from 0.001 to 2.0
    result = minimize_scalar(objective, bounds=(0.001, 2.0), method='bounded')
    lam_opt = result.x
    min_error = result.fun
    
    T_model = boltzmann_matrix_residue(lam_opt, cols, m)
    
    # R² in Frobenius sense
    T_null = np.ones_like(T_empirical) / T_empirical.shape[0]
    ss_res = np.sum((T_empirical - T_model)**2)
    ss_tot = np.sum((T_empirical - T_null)**2)
    r_squared = 1 - ss_res / ss_tot
    
    # Element-wise statistics
    residual = T_empirical - T_model
    max_err = float(np.max(np.abs(residual)))
    mean_err = float(np.mean(np.abs(residual)))
    
    # Per-row λ fit (for consistency check)
    D = build_distance_matrix(cols, m)
    row_lambdas = []
    row_r2s = []
    for i in range(len(cols)):
        distances = D[i, :]
        log_probs = np.log(np.maximum(T_empirical[i, :], 1e-15))
        
        # Linear fit: log(T[i,j]) = -λ * d_ij + const
        valid = distances > 0  # exclude self (d=0 maps to d=30 for forward)
        # Actually for self, d = 0 mod 30 = 0 → means self = exp(0)/Z = 1/Z
        # But the self-avoidance breaks this. Let's fit all points.
        coeffs = np.polyfit(distances, log_probs, 1)
        row_lam = -coeffs[0]
        pred = coeffs[0] * distances + coeffs[1]
        ss_r = np.sum((log_probs - pred)**2)
        ss_t = np.sum((log_probs - np.mean(log_probs))**2)
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        
        row_lambdas.append(row_lam)
        row_r2s.append(r2)
    
    return {
        "lambda_opt": float(lam_opt),
        "frobenius_error": float(min_error),
        "r_squared": float(r_squared),
        "max_element_error": max_err,
        "mean_element_error": mean_err,
        "residual_matrix": residual.tolist(),
        "row_lambdas": row_lambdas,
        "row_r2s": row_r2s,
        "mean_row_lambda": float(np.mean(row_lambdas)),
        "std_row_lambda": float(np.std(row_lambdas)),
        "cv_row_lambda": float(np.std(row_lambdas) / abs(np.mean(row_lambdas))) if np.mean(row_lambdas) != 0 else float('inf'),
        "mean_row_r2": float(np.mean(row_r2s)),
        "T_model": T_model.tolist(),
    }


# ═══════════════════════════════════════════════════════
# ALSO FIT: TWO-PARAMETER MODEL WITH SELF-AVOIDANCE
# ═══════════════════════════════════════════════════════

def boltzmann_matrix_with_self_penalty(lam, mu, cols, m):
    """
    Extended model: T[i,j] = exp(-λ * d - μ * δ(i,j)) / Z
    where δ(i,j) = 1 if same column (d=0), 0 otherwise.
    This explicitly models self-avoidance as a separate parameter.
    """
    n = len(cols)
    D = build_distance_matrix(cols, m)
    
    unnorm = np.exp(-lam * D)
    # Apply self-penalty on diagonal
    for i in range(n):
        unnorm[i, i] *= np.exp(-mu)
    
    row_sums = unnorm.sum(axis=1, keepdims=True)
    return unnorm / row_sums


def fit_two_param(T_empirical, cols, m):
    """Fit λ and μ (self-avoidance penalty) simultaneously."""
    def objective(params):
        lam, mu = params
        if lam < 0.001 or mu < -5 or mu > 20:
            return 1e10
        T_model = boltzmann_matrix_with_self_penalty(lam, mu, cols, m)
        return np.linalg.norm(T_empirical - T_model, 'fro')
    
    result = minimize(objective, x0=[0.1, 1.0], method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    lam_opt, mu_opt = result.x
    T_model = boltzmann_matrix_with_self_penalty(lam_opt, mu_opt, cols, m)
    
    T_null = np.ones_like(T_empirical) / T_empirical.shape[0]
    ss_res = np.sum((T_empirical - T_model)**2)
    ss_tot = np.sum((T_empirical - T_null)**2)
    r_squared = 1 - ss_res / ss_tot
    
    residual = T_empirical - T_model
    
    return {
        "lambda_opt": float(lam_opt),
        "mu_opt": float(mu_opt),
        "frobenius_error": float(result.fun),
        "r_squared": float(r_squared),
        "max_element_error": float(np.max(np.abs(residual))),
        "mean_element_error": float(np.mean(np.abs(residual))),
        "residual_matrix": residual.tolist(),
        "T_model": T_model.tolist(),
    }


# ═══════════════════════════════════════════════════════
# PNT PREDICTION: λ_predicted = 1/ln(N)
# ═══════════════════════════════════════════════════════

def pnt_lambda(log10_N):
    """PNT prediction: λ = 1/ln(N) where ln(N) = log10(N) * ln(10)."""
    ln_N = log10_N * np.log(10)
    return 1.0 / ln_N


# ═══════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════

def run_experiment(max_prime=10**8):
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  WAVE 22b — THE CORRECTED BOLTZMANN FIT                    ║")
    print("║  Using RESIDUE DISTANCE, not column index distance.        ║")
    print("║  Prediction: λ = 1/ln(N) exactly (from the PNT).          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    start_time = time.time()
    
    print(f"Sieving primes up to {max_prime:,}...")
    t0 = time.time()
    all_primes = sieve_primes(max_prime)
    print(f"  Found {len(all_primes):,} primes in {time.time()-t0:.1f}s")
    print()
    
    moduli = [30, 210]
    
    octave_specs = [
        (10**3, 10**4, "10^3–10^4", 3.5),
        (10**4, 10**5, "10^4–10^5", 4.5),
        (10**5, 10**6, "10^5–10^6", 5.5),
        (10**6, 10**7, "10^6–10^7", 6.5),
        (10**7, 10**8, "10^7–10^8", 7.5),
    ]
    
    all_results = {}
    
    for m in moduli:
        cols = admissible_columns(m)
        n_cols = len(cols)
        
        # Show the distance matrix
        D = build_distance_matrix(cols, m)
        
        print(f"{'='*70}")
        print(f"  MODULUS m = {m}  |  φ(m) = {n_cols} columns")
        print(f"  Columns: {cols[:12]}{'...' if n_cols > 12 else ''}")
        print(f"{'='*70}")
        
        if n_cols <= 8:
            print(f"\n  Forward RESIDUE distance matrix:")
            header = "       " + "".join(f"{c:>5}" for c in cols)
            print(f"    {header}")
            for i, ci in enumerate(cols):
                row_str = f"    {ci:>3} "
                for j in range(n_cols):
                    row_str += f" {int(D[i,j]):>4}"
                print(row_str)
            print(f"\n  Compare to column INDEX distance: 0, 1, 2, ..., {n_cols-1}")
            print(f"  The residue distances are {list(map(int, D[0,:]))} — NOT uniform!")
        print()
        
        scale_results = []
        log_ns = []
        lambdas_1param = []
        lambdas_2param = []
        mus = []
        pnt_lambdas = []
        angles = []
        magnitudes = []
        
        for lo, hi, label, log_mid in octave_specs:
            mask = (all_primes >= lo) & (all_primes < hi)
            primes_in_range = all_primes[mask].tolist()
            
            if len(primes_in_range) < 100:
                continue
            
            working_primes = [p for p in primes_in_range if p > m]
            
            # Build empirical transition matrix
            T = np.zeros((n_cols, n_cols))
            col_idx = {c: i for i, c in enumerate(cols)}
            for k in range(len(working_primes) - 1):
                r_from = working_primes[k] % m
                r_to = working_primes[k + 1] % m
                if r_from in col_idx and r_to in col_idx:
                    T[col_idx[r_from], col_idx[r_to]] += 1
            
            row_sums = T.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            T_norm = T / row_sums
            
            # ─── FIT 1: One-parameter Boltzmann (residue distance) ───
            fit1 = fit_lambda_residue(T_norm, cols, m)
            
            # ─── FIT 2: Two-parameter (λ + self-avoidance μ) ───
            fit2 = fit_two_param(T_norm, cols, m)
            
            # ─── PNT prediction ───
            lam_pnt = pnt_lambda(log_mid)
            T_pnt = boltzmann_matrix_residue(lam_pnt, cols, m)
            T_null = np.ones_like(T_norm) / n_cols
            ss_res_pnt = np.sum((T_norm - T_pnt)**2)
            ss_tot_pnt = np.sum((T_norm - T_null)**2)
            r2_pnt = 1 - ss_res_pnt / ss_tot_pnt
            frob_pnt = float(np.linalg.norm(T_norm - T_pnt, 'fro'))
            max_err_pnt = float(np.max(np.abs(T_norm - T_pnt)))
            
            # ─── Eigenstructure ───
            eigenvalues = np.linalg.eigvals(T_norm)
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
            
            result = {
                "label": label,
                "n_primes": len(working_primes),
                "log10_N_mid": log_mid,
                "ln_N_mid": log_mid * np.log(10),
                
                # 1-param fit
                "lambda_1param": fit1['lambda_opt'],
                "r2_1param": fit1['r_squared'],
                "frob_1param": fit1['frobenius_error'],
                "max_err_1param": fit1['max_element_error'],
                "mean_err_1param": fit1['mean_element_error'],
                "row_lambdas": fit1['row_lambdas'],
                "row_r2s": fit1['row_r2s'],
                "cv_row_lambda": fit1['cv_row_lambda'],
                
                # 2-param fit
                "lambda_2param": fit2['lambda_opt'],
                "mu_2param": fit2['mu_opt'],
                "r2_2param": fit2['r_squared'],
                "frob_2param": fit2['frobenius_error'],
                "max_err_2param": fit2['max_element_error'],
                "mean_err_2param": fit2['mean_element_error'],
                
                # PNT prediction (zero free parameters)
                "lambda_pnt": float(lam_pnt),
                "r2_pnt": float(r2_pnt),
                "frob_pnt": frob_pnt,
                "max_err_pnt": max_err_pnt,
                
                # Eigenstructure
                "leading_complex_magnitude": leading_mag,
                "leading_complex_angle_deg": leading_angle,
                "n_complex": len(complex_eigs),
                
                # The transition matrices
                "T_empirical": T_norm.tolist(),
                "T_1param": fit1['T_model'],
                "T_2param": fit2['T_model'],
            }
            
            scale_results.append(result)
            log_ns.append(log_mid)
            lambdas_1param.append(fit1['lambda_opt'])
            lambdas_2param.append(fit2['lambda_opt'])
            mus.append(fit2['mu_opt'])
            pnt_lambdas.append(float(lam_pnt))
            if leading_angle != 0:
                angles.append(leading_angle)
            magnitudes.append(leading_mag)
            
            # ─── Print ───
            print(f"  {label} ({len(working_primes):,} primes):")
            print(f"    ┌─ 1-PARAM BOLTZMANN (residue distance):")
            print(f"    │  λ_fitted = {fit1['lambda_opt']:.6f}")
            print(f"    │  R² = {fit1['r_squared']:.6f}")
            print(f"    │  Max error = {fit1['max_element_error']:.6f}")
            print(f"    │  Mean error = {fit1['mean_element_error']:.6f}")
            print(f"    │  Row λ CV = {fit1['cv_row_lambda']:.4f}")
            print(f"    │")
            print(f"    ├─ 2-PARAM BOLTZMANN (+ self-avoidance μ):")
            print(f"    │  λ = {fit2['lambda_opt']:.6f}, μ = {fit2['mu_opt']:.4f}")
            print(f"    │  R² = {fit2['r_squared']:.6f}")
            print(f"    │  Max error = {fit2['max_element_error']:.6f}")
            print(f"    │  Mean error = {fit2['mean_element_error']:.6f}")
            print(f"    │")
            print(f"    ├─ PNT PREDICTION (ZERO free parameters):")
            print(f"    │  λ_PNT = 1/ln(N) = {lam_pnt:.6f}")
            print(f"    │  R² = {r2_pnt:.6f}")
            print(f"    │  Frobenius = {frob_pnt:.6f}")
            print(f"    │  λ_fitted/λ_PNT = {fit1['lambda_opt']/lam_pnt:.4f}")
            print(f"    │")
            print(f"    └─ EIGENSTRUCTURE:")
            if leading_mag > 0:
                print(f"       |λ₁| = {leading_mag:.6f}, θ = {leading_angle:.2f}°")
            else:
                print(f"       No complex eigenvalues")
            print()
        
        # ═══════════════════════════════════════════════════════
        # SCALING LAW ANALYSIS
        # ═══════════════════════════════════════════════════════
        
        log_ns_arr = np.array(log_ns)
        lambdas_arr = np.array(lambdas_1param)
        pnt_arr = np.array(pnt_lambdas)
        
        print(f"  {'='*60}")
        print(f"  PREDICTION 1: λ = 1/ln(N)  (mod {m})")
        print(f"  {'='*60}")
        print()
        
        # Compare fitted λ to PNT prediction
        print(f"    {'Scale':<12} {'λ_fitted':>10} {'λ_PNT':>10} {'Ratio':>8} {'Δλ':>10}")
        print(f"    {'─'*52}")
        ratios = []
        for i, (logn, lam_f, lam_p) in enumerate(zip(log_ns, lambdas_1param, pnt_lambdas)):
            ratio = lam_f / lam_p
            ratios.append(ratio)
            print(f"    log₁₀N={logn:<4.1f} {lam_f:>10.6f} {lam_p:>10.6f} {ratio:>8.4f} {lam_f-lam_p:>+10.6f}")
        
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        print()
        print(f"    Mean ratio λ_fitted/λ_PNT = {mean_ratio:.4f} ± {std_ratio:.4f}")
        print()
        
        # If ratio is constant, then λ = c/ln(N) with c ≈ mean_ratio
        # The pure PNT prediction is c = 1
        
        # Fit λ = c / ln(N)
        ln_N_mids = log_ns_arr * np.log(10)
        inv_lnN = 1.0 / ln_N_mids
        
        # OLS: λ = c * (1/lnN), forced through origin
        c_fit = float(np.sum(lambdas_arr * inv_lnN) / np.sum(inv_lnN**2))
        pred_c = c_fit * inv_lnN
        ss_res_c = np.sum((lambdas_arr - pred_c)**2)
        ss_tot = np.sum((lambdas_arr - np.mean(lambdas_arr))**2)
        r2_c = 1 - ss_res_c / ss_tot if ss_tot > 0 else 0
        
        print(f"    Best fit: λ = {c_fit:.4f} / ln(N)")
        print(f"    R² = {r2_c:.6f}")
        print(f"    PNT predicts c = 1.0, fitted c = {c_fit:.4f}")
        print()
        
        for logn, lam, pred in zip(log_ns, lambdas_1param, pred_c):
            print(f"      ln(N)={logn*np.log(10):.2f}: λ_obs={lam:.6f}, "
                  f"λ_pred={pred:.6f}, residual={lam-pred:+.6f}")
        print()
        
        # Also check: λ = c / ln(N) + d (two-param, with intercept)
        coeffs_2p = np.polyfit(inv_lnN, lambdas_arr, 1)
        c_2p = coeffs_2p[0]
        d_2p = coeffs_2p[1]
        pred_2p = c_2p * inv_lnN + d_2p
        ss_res_2p = np.sum((lambdas_arr - pred_2p)**2)
        r2_2p = 1 - ss_res_2p / ss_tot if ss_tot > 0 else 0
        
        print(f"    With intercept: λ = {c_2p:.4f} / ln(N) + {d_2p:.6f}")
        print(f"    R² = {r2_2p:.6f}")
        print(f"    Asymptotic λ(N→∞) = {d_2p:.6f}")
        print()
        
        # Verdict
        r2_pnt_pred = 1 - np.sum((lambdas_arr - pnt_arr)**2) / ss_tot if ss_tot > 0 else 0
        
        print(f"    ╔══════════════════════════════════════════════════════╗")
        print(f"    ║  PREDICTION 1 RESULTS (mod {m})                     ║")
        print(f"    ║  Zero-parameter PNT (λ=1/lnN):    R² = {r2_pnt_pred:>8.4f}   ║")
        print(f"    ║  One-parameter (λ=c/lnN):          R² = {r2_c:>8.4f}   ║")
        print(f"    ║  Two-parameter (λ=c/lnN+d):        R² = {r2_2p:>8.4f}   ║")
        print(f"    ║  Fitted c = {c_fit:.4f} (PNT predicts 1.0)           ║")
        
        if r2_c > 0.99:
            print(f"    ║  STATUS: ████ λ = c/ln(N) CONFIRMED ████          ║")
        elif r2_c > 0.95:
            print(f"    ║  STATUS: ███ STRONG ███                           ║")
        elif r2_c > 0.90:
            print(f"    ║  STATUS: ██ GOOD ██                               ║")
        else:
            print(f"    ║  STATUS: █ NEEDS MORE SCALES █                    ║")
        print(f"    ╚══════════════════════════════════════════════════════╝")
        print()
        
        # ═══════════════════════════════════════
        # PREDICTION 2: θ evolution
        # ═══════════════════════════════════════
        print(f"  {'='*60}")
        print(f"  PREDICTION 2: θ(N) spiral angle (mod {m})")
        print(f"  {'='*60}")
        print()
        
        if len(angles) >= 3:
            angles_arr = np.array(angles)
            theta_mean = np.mean(angles_arr)
            theta_std = np.std(angles_arr)
            theta_cv = theta_std / abs(theta_mean) if theta_mean != 0 else float('inf')
            
            angle_logns = log_ns_arr[:len(angles_arr)]
            coeffs_th = np.polyfit(angle_logns, angles_arr, 1)
            theta_slope = coeffs_th[0]
            
            for logn, ang in zip(log_ns, angles):
                cols_per = n_cols * ang / 360
                print(f"    log₁₀N={logn:.1f}: θ = {ang:7.2f}° ({cols_per:.2f} cols/step)")
            
            print()
            print(f"    Mean θ = {theta_mean:.2f}° ± {theta_std:.2f}°")
            print(f"    CV(θ) = {theta_cv:.4f}")
            print(f"    Drift: dθ/d(log₁₀N) = {theta_slope:.4f}°/decade")
            
            # Check if θ ∝ log(N) — would that make sense?
            r_theta_logn, p_theta = pearsonr(angle_logns, angles_arr) if len(angles_arr) >= 3 else (0, 1)
            print(f"    r(θ, log₁₀N) = {r_theta_logn:.4f}, p = {p_theta:.6f}")
            
            # If θ drifts linearly with logN, what's the slope?
            if abs(r_theta_logn) > 0.95:
                print(f"    θ ≈ {theta_slope:.2f} · log₁₀(N) + {coeffs_th[1]:.2f}")
                print(f"    The spiral is WINDING with scale — not constant.")
            
            print()
        
        # ═══════════════════════════════════════
        # PREDICTION 3: |λ₁| persistence
        # ═══════════════════════════════════════
        print(f"  {'='*60}")
        print(f"  PREDICTION 3: |λ₁(N)| — spiral persistence (mod {m})")
        print(f"  {'='*60}")
        print()
        
        if len(magnitudes) >= 3:
            mags_arr = np.array(magnitudes)
            
            for logn, mag in zip(log_ns, magnitudes):
                bar = '█' * int(mag * 40)
                print(f"    log₁₀N={logn:.1f}: |λ₁| = {mag:.6f}  {bar}")
            
            # Power law fit
            valid = mags_arr > 0
            log_mags = np.log(mags_arr[valid])
            log_logns = np.log(log_ns_arr[valid])
            
            coeffs_m = np.polyfit(log_logns, log_mags, 1)
            beta = -coeffs_m[0]
            a_mag = np.exp(coeffs_m[1])
            pred_mags = a_mag * log_ns_arr[valid]**(-beta)
            ss_res_m = np.sum((mags_arr[valid] - pred_mags)**2)
            ss_tot_m = np.sum((mags_arr[valid] - np.mean(mags_arr[valid]))**2)
            r2_mag = 1 - ss_res_m / ss_tot_m if ss_tot_m > 0 else 0
            
            print()
            print(f"    Fit: |λ₁| = {a_mag:.4f} · log₁₀(N)^(-{beta:.4f})")
            print(f"    R² = {r2_mag:.6f}")
            
            for target in [10, 20, 50, 100, 308]:  # 308 = log10(10^308) ≈ biggest float
                pred = a_mag * target**(-beta)
                print(f"    Extrapolation: log₁₀N={target:>4} → |λ₁| ≈ {pred:.4f}")
            print()
        
        # ═══════════════════════════════════════
        # MATRIX COMPARISON (mod 30 only)
        # ═══════════════════════════════════════
        if m == 30 and scale_results:
            best = scale_results[-1]
            T_emp = np.array(best['T_empirical'])
            
            # Get the three models
            T_1p = np.array(best['T_1param'])
            T_2p = np.array(best['T_2param'])
            T_pnt_model = boltzmann_matrix_residue(best['lambda_pnt'], cols, m)
            
            print(f"  {'='*60}")
            print(f"  MATRIX COMPARISON (mod 30, scale {best['label']})")
            print(f"  {'='*60}")
            print()
            
            for name, T_model, r2 in [
                ("EMPIRICAL", T_emp, 1.0),
                (f"1-PARAM (λ={best['lambda_1param']:.4f})", T_1p, best['r2_1param']),
                (f"2-PARAM (λ={best['lambda_2param']:.4f}, μ={best['mu_2param']:.4f})", T_2p, best['r2_2param']),
                (f"PNT (λ=1/ln(N)={best['lambda_pnt']:.4f})", T_pnt_model, best['r2_pnt']),
            ]:
                print(f"    {name} (R² = {r2:.4f}):")
                header = "         " + "".join(f"{c:>7}" for c in cols)
                print(f"    {header}")
                for i, ci in enumerate(cols):
                    row_str = f"      {ci:>3} "
                    for j in range(n_cols):
                        row_str += f" {T_model[i, j]:5.3f} "
                    print(row_str)
                print()
            
            # Residual for best model
            print(f"    RESIDUAL (Empirical - 2-param):")
            header = "         " + "".join(f"{c:>7}" for c in cols)
            print(f"    {header}")
            resid = T_emp - T_2p
            for i, ci in enumerate(cols):
                row_str = f"      {ci:>3} "
                for j in range(n_cols):
                    d = resid[i, j]
                    if abs(d) > 0.003:
                        row_str += f" {d:+5.3f}*"
                    else:
                        row_str += f" {d:+5.3f} "
                print(row_str)
            print(f"    (* = |residual| > 0.003)")
            
            # SVD of residual — what structure remains?
            U, S, Vt = np.linalg.svd(resid)
            print(f"\n    Residual singular values: {[f'{s:.4f}' for s in S[:5]]}")
            print(f"    Rank structure: σ₁/σ₂ = {S[0]/S[1]:.4f}" if S[1] > 0 else "")
            print()
        
        # ═══════════════════════════════════════
        # TWO-PARAM SCALING
        # ═══════════════════════════════════════
        print(f"  {'='*60}")
        print(f"  TWO-PARAMETER MODEL SCALING (mod {m})")
        print(f"  {'='*60}")
        print()
        
        mus_arr = np.array(mus)
        lam2_arr = np.array(lambdas_2param)
        
        print(f"    {'Scale':<12} {'λ':>10} {'μ':>10} {'R²':>10} {'exp(-μ)':>10}")
        print(f"    {'─'*54}")
        for logn, l2, mu, sr in zip(log_ns, lambdas_2param, mus, scale_results):
            print(f"    log₁₀N={logn:<4.1f} {l2:>10.6f} {mu:>10.4f} {sr['r2_2param']:>10.6f} {np.exp(-mu):>10.6f}")
        
        # Fit μ vs log(N)
        if len(mus) >= 3:
            coeffs_mu = np.polyfit(log_ns_arr, mus_arr, 1)
            mu_slope = coeffs_mu[0]
            print(f"\n    μ slope: dμ/d(log₁₀N) = {mu_slope:.4f}")
            print(f"    μ interpretation: self-avoidance penalty")
            print(f"    exp(-μ) = self-transition suppression factor")
        print()
        
        all_results[f"mod_{m}"] = {
            "modulus": m,
            "phi_m": n_cols,
            "columns": cols,
            "scales": scale_results,
            "log_ns": log_ns,
            "lambdas_1param": lambdas_1param,
            "lambdas_2param": lambdas_2param,
            "mus": mus,
            "pnt_lambdas": pnt_lambdas,
            "angles": angles,
            "magnitudes": magnitudes,
        }
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Save
    output = {
        "experiment": "Wave 22b — Corrected Boltzmann Fit",
        "timestamp": int(time.time() * 1000),
        "elapsed_seconds": elapsed,
        "max_prime": int(max_prime),
        "total_primes": int(len(all_primes)),
        "results": all_results,
    }
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                           f'wave22b_boltzmann_{int(time.time()*1000)}.json')
    out_path = os.path.normpath(out_path)
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Results saved to: {out_path}")
    
    return output


if __name__ == "__main__":
    results = run_experiment(max_prime=10**8)
