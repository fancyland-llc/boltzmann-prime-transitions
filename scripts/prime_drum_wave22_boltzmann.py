#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  WAVE 22 — THE BOLTZMANN FIT                                           ║
║                                                                        ║
║  Three falsifiable predictions from Wave 21's columnar interferometer:  ║
║                                                                        ║
║  1. The transition matrix is Boltzmann: T_ij = exp(-λ·d_ij) / Z        ║
║     where d_ij is forward cyclic distance.                             ║
║     Prediction: λ ∝ 1/log(N). Temperature = log(N).                   ║
║                                                                        ║
║  2. The argument θ of the leading complex eigenvalue (mod 210)         ║
║     is CONSTANT across scales — a geometric property of the            ║
║     admissible residue structure, not a scale artifact.                ║
║                                                                        ║
║  3. The magnitude |λ₁| of the leading complex eigenvalue              ║
║     either approaches 1 (spiral persists) or 0 (spiral dies).         ║
║                                                                        ║
║  Author: Tony M. (Architect) + Claude Opus 4 (Correspondent)          ║
║  Date: March 17, 2026                                                  ║
║  EPOCH 004, Wave 22                                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from math import gcd, log, pi, atan2
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from collections import defaultdict
import json
import time
import os

# ═══════════════════════════════════════════════════════
# SIEVE
# ═══════════════════════════════════════════════════════

def sieve_primes(limit):
    """Return list of primes up to limit using numpy sieve."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


def admissible_columns(m):
    """Get residue classes mod m that are coprime to m."""
    return sorted([r for r in range(1, m) if gcd(r, m) == 1])


# ═══════════════════════════════════════════════════════
# FORWARD CYCLIC DISTANCE
# ═══════════════════════════════════════════════════════

def forward_distance(i, j, n_cols):
    """Forward distance from column index i to column index j in cyclic group."""
    return (j - i) % n_cols


# ═══════════════════════════════════════════════════════
# BOLTZMANN MODEL
# ═══════════════════════════════════════════════════════

def boltzmann_row(lam, n_cols):
    """
    Generate a single row of the Boltzmann transition matrix.
    T[d] = exp(-λ * d) / Z  for d = 0, 1, ..., n_cols-1
    where d is forward cyclic distance.
    """
    distances = np.arange(n_cols)
    unnorm = np.exp(-lam * distances)
    Z = np.sum(unnorm)
    return unnorm / Z


def boltzmann_matrix(lam, n_cols):
    """Full Boltzmann transition matrix: T[i,j] = exp(-λ * d_fwd(i,j)) / Z."""
    T = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            d = forward_distance(i, j, n_cols)
            T[i, j] = d  # store distances first
    
    # Apply exponential
    T_exp = np.exp(-lam * T)
    # Normalize each row
    row_sums = T_exp.sum(axis=1, keepdims=True)
    return T_exp / row_sums


def fit_lambda_from_row(row, n_cols):
    """
    Fit λ from a single row of the transition matrix.
    The row should have T[d] = exp(-λ·d)/Z for d=0..n_cols-1.
    
    Take log: ln(T[d]) = -λ·d - ln(Z)
    This is linear regression: ln(T) vs d, slope = -λ.
    """
    distances = np.arange(n_cols)
    log_probs = np.log(np.maximum(row, 1e-15))  # avoid log(0)
    
    # Linear fit: log_prob = -λ * d + const
    coeffs = np.polyfit(distances, log_probs, 1)
    lam = -coeffs[0]  # slope is -λ
    intercept = coeffs[1]
    
    # Residuals
    predicted = coeffs[0] * distances + coeffs[1]
    residuals = log_probs - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    r_squared = 1 - np.sum(residuals**2) / np.sum((log_probs - np.mean(log_probs))**2)
    
    return lam, intercept, rmse, r_squared, residuals


def fit_lambda_matrix(T_norm, n_cols):
    """
    Fit λ from the full transition matrix by averaging over all rows.
    Each row should give the same λ if the matrix is truly circulant.
    """
    lambdas = []
    r_squareds = []
    all_residuals = []
    
    for i in range(n_cols):
        # Reorder row so that distances are 0, 1, 2, ..., n_cols-1
        reordered = np.zeros(n_cols)
        for j in range(n_cols):
            d = forward_distance(i, j, n_cols)
            reordered[d] = T_norm[i, j]
        
        lam, intercept, rmse, r_sq, resid = fit_lambda_from_row(reordered, n_cols)
        lambdas.append(lam)
        r_squareds.append(r_sq)
        all_residuals.append(resid)
    
    return {
        "lambdas": lambdas,
        "mean_lambda": float(np.mean(lambdas)),
        "std_lambda": float(np.std(lambdas)),
        "cv_lambda": float(np.std(lambdas) / np.mean(lambdas)) if np.mean(lambdas) != 0 else float('inf'),
        "mean_r_squared": float(np.mean(r_squareds)),
        "r_squareds": r_squareds,
        "residual_matrix": [r.tolist() for r in all_residuals],
    }


# ═══════════════════════════════════════════════════════
# MEASUREMENT ENGINE
# ═══════════════════════════════════════════════════════

def measure_at_scale(primes, m, label=""):
    """
    Given primes in a specific range, compute:
    1. Transition matrix
    2. λ (Boltzmann decay constant) 
    3. Eigenstructure (complex eigenvalues)
    4. Chirality decomposition
    """
    cols = admissible_columns(m)
    n_cols = len(cols)
    col_idx = {c: i for i, c in enumerate(cols)}
    
    working_primes = [p for p in primes if p > m]
    if len(working_primes) < 100:
        return None
    
    # Build transition matrix
    T = np.zeros((n_cols, n_cols))
    for i in range(len(working_primes) - 1):
        r_from = working_primes[i] % m
        r_to = working_primes[i + 1] % m
        if r_from in col_idx and r_to in col_idx:
            T[col_idx[r_from], col_idx[r_to]] += 1
    
    # Normalize
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_norm = T / row_sums
    
    # ─── FIT 1: Boltzmann λ ───
    boltzmann_fit = fit_lambda_matrix(T_norm, n_cols)
    
    # Compare empirical matrix to fitted Boltzmann matrix
    lam_fit = boltzmann_fit['mean_lambda']
    T_boltzmann = boltzmann_matrix(lam_fit, n_cols)
    frobenius_error = float(np.linalg.norm(T_norm - T_boltzmann, 'fro'))
    max_element_error = float(np.max(np.abs(T_norm - T_boltzmann)))
    mean_element_error = float(np.mean(np.abs(T_norm - T_boltzmann)))
    
    # ─── FIT 2: Eigenstructure ───
    eigenvalues = np.linalg.eigvals(T_norm)
    eig_order = np.argsort(-np.abs(eigenvalues))
    eigenvalues_sorted = eigenvalues[eig_order]
    
    # Find leading complex eigenvalue
    complex_eigs = [(e, abs(e), atan2(e.imag, e.real) * 180 / pi) 
                    for e in eigenvalues_sorted 
                    if abs(e.imag) > 1e-10]
    
    leading_complex = None
    leading_magnitude = 0
    leading_angle = 0
    if complex_eigs:
        # Take the one with largest magnitude
        best = max(complex_eigs, key=lambda x: x[1])
        leading_complex = best[0]
        leading_magnitude = best[1]
        leading_angle = best[2]
    
    # ─── FIT 3: Chirality decomposition ───
    T_null = np.ones((n_cols, n_cols)) / n_cols
    T_sym = (T_norm + T_norm.T) / 2
    T_antisym = (T_norm - T_norm.T) / 2
    
    sym_norm = float(np.linalg.norm(T_sym - T_null, 'fro'))
    antisym_norm = float(np.linalg.norm(T_antisym, 'fro'))
    chirality_ratio = antisym_norm / (sym_norm + antisym_norm) if (sym_norm + antisym_norm) > 0 else 0
    
    # Diagonal (self-avoidance)
    diag_vals = np.diag(T_norm)
    diag_null = 1.0 / n_cols
    mean_diag_bias = float(np.mean(diag_vals) - diag_null)
    
    # ─── Residual structure analysis ───
    # Are the Boltzmann residuals structureless?
    residual_matrix = T_norm - T_boltzmann
    
    # Check if residuals have systematic structure via SVD
    U, S, Vt = np.linalg.svd(residual_matrix)
    # If structureless, singular values should be small and flat
    top_singular = float(S[0]) if len(S) > 0 else 0
    second_singular = float(S[1]) if len(S) > 1 else 0
    
    # Residual Frobenius norm relative to T
    relative_error = frobenius_error / np.linalg.norm(T_norm, 'fro')
    
    return {
        "label": label,
        "modulus": m,
        "phi_m": n_cols,
        "n_primes": len(working_primes),
        "range": [int(working_primes[0]), int(working_primes[-1])],
        "transition_matrix": T_norm.tolist(),
        
        # Boltzmann fit
        "lambda_fit": lam_fit,
        "lambda_std": boltzmann_fit['std_lambda'],
        "lambda_cv": boltzmann_fit['cv_lambda'],
        "per_row_lambdas": boltzmann_fit['lambdas'],
        "boltzmann_r_squared": boltzmann_fit['mean_r_squared'],
        "per_row_r_squared": boltzmann_fit['r_squareds'],
        "frobenius_error": frobenius_error,
        "max_element_error": max_element_error,
        "mean_element_error": mean_element_error,
        "relative_error": float(relative_error),
        
        # Residual structure
        "residual_top_singular": top_singular,
        "residual_second_singular": second_singular,
        "residual_rank_ratio": second_singular / top_singular if top_singular > 0 else 0,
        
        # Eigenstructure
        "eigenvalues_real": [float(e.real) for e in eigenvalues_sorted[:6]],
        "eigenvalues_imag": [float(e.imag) for e in eigenvalues_sorted[:6]],
        "n_complex_eigenvalues": len(complex_eigs),
        "leading_complex_magnitude": leading_magnitude,
        "leading_complex_angle_deg": leading_angle,
        "leading_complex_real": float(leading_complex.real) if leading_complex is not None else None,
        "leading_complex_imag": float(leading_complex.imag) if leading_complex is not None else None,
        
        # Chirality
        "chirality_ratio": chirality_ratio,
        "sym_norm": sym_norm,
        "antisym_norm": antisym_norm,
        "mean_diag_bias": mean_diag_bias,
    }


# ═══════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════

def run_experiment(max_prime=10**8):
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  WAVE 22 — THE BOLTZMANN FIT                               ║")
    print("║  Three predictions. One experiment. New math or graveyard.  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    start_time = time.time()
    
    # Sieve
    print(f"Sieving primes up to {max_prime:,}...")
    t0 = time.time()
    all_primes = sieve_primes(max_prime)
    print(f"  Found {len(all_primes):,} primes in {time.time()-t0:.1f}s")
    print()
    
    moduli = [30, 210]
    
    # Build octave ranges — use finer granularity for better scaling law fit
    octave_specs = [
        (10**3, 10**4, "10^3–10^4", 3.5),      # log10 midpoint
        (10**4, 10**5, "10^4–10^5", 4.5),
        (10**5, 10**6, "10^5–10^6", 5.5),
        (10**6, 10**7, "10^6–10^7", 6.5),
        (10**7, 10**8, "10^7–10^8", 7.5),
    ]
    
    all_results = {}
    
    for m in moduli:
        cols = admissible_columns(m)
        n_cols = len(cols)
        
        print(f"{'='*70}")
        print(f"  MODULUS m = {m}  |  φ(m) = {n_cols} columns")
        print(f"{'='*70}")
        print()
        
        scale_results = []
        log_ns = []
        lambdas = []
        angles = []
        magnitudes = []
        chiralities = []
        
        for lo, hi, label, log_mid in octave_specs:
            mask = (all_primes >= lo) & (all_primes < hi)
            primes_in_range = all_primes[mask].tolist()
            
            if len(primes_in_range) < 100:
                continue
            
            result = measure_at_scale(primes_in_range, m, label)
            if result is None:
                continue
            
            scale_results.append(result)
            log_ns.append(log_mid)
            lambdas.append(result['lambda_fit'])
            if result['leading_complex_angle_deg'] != 0:
                angles.append(result['leading_complex_angle_deg'])
            magnitudes.append(result['leading_complex_magnitude'])
            chiralities.append(result['chirality_ratio'])
            
            # Print per-scale results
            print(f"  {label} ({result['n_primes']:,} primes):")
            print(f"    BOLTZMANN FIT:")
            print(f"      λ = {result['lambda_fit']:.6f} ± {result['lambda_std']:.6f} "
                  f"(CV = {result['lambda_cv']:.4f})")
            print(f"      R² = {result['boltzmann_r_squared']:.6f}")
            print(f"      Frobenius error = {result['frobenius_error']:.6f}")
            print(f"      Max element error = {result['max_element_error']:.6f}")
            print(f"      Mean element error = {result['mean_element_error']:.6f}")
            print(f"      Relative error = {result['relative_error']:.4f}")
            print(f"      Residual top SVs: σ₁={result['residual_top_singular']:.6f}, "
                  f"σ₂={result['residual_second_singular']:.6f}")
            
            if result['n_complex_eigenvalues'] > 0:
                print(f"    EIGENSTRUCTURE:")
                print(f"      Leading complex λ = {result['leading_complex_real']:.4f} "
                      f"± {abs(result['leading_complex_imag']):.4f}i")
                print(f"      |λ₁| = {result['leading_complex_magnitude']:.6f}")
                print(f"      θ = {result['leading_complex_angle_deg']:.2f}°")
                print(f"      Steps per full rotation: "
                      f"{360/abs(result['leading_complex_angle_deg']):.1f}" 
                      if result['leading_complex_angle_deg'] != 0 else "      N/A")
            
            print(f"    CHIRALITY: {result['chirality_ratio']:.6f}")
            print(f"    SELF-AVOIDANCE: {result['mean_diag_bias']:+.6f}")
            print()
        
        # ═══════════════════════════════════════════════════════
        # SCALING LAW FITS
        # ═══════════════════════════════════════════════════════
        print(f"  {'─'*60}")
        print(f"  SCALING LAWS (mod {m})")
        print(f"  {'─'*60}")
        print()
        
        log_ns_arr = np.array(log_ns)
        lambdas_arr = np.array(lambdas)
        
        # ─── PREDICTION 1: λ ∝ 1/log(N) ───
        # Fit λ = a / log10(N) + b
        # Also try λ = a / log(N) (natural log, no offset)
        print(f"  PREDICTION 1: λ(N) = a/log(N)")
        print()
        
        if len(log_ns) >= 3:
            # Fit 1a: λ = a / log10(N)  (one parameter, forced through origin)
            inv_logN = 1.0 / log_ns_arr
            
            # Simple: λ = a * (1/logN)
            a_simple = float(np.sum(lambdas_arr * inv_logN) / np.sum(inv_logN**2))
            pred_simple = a_simple * inv_logN
            resid_simple = lambdas_arr - pred_simple
            ss_res = np.sum(resid_simple**2)
            ss_tot = np.sum((lambdas_arr - np.mean(lambdas_arr))**2)
            r2_simple = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            print(f"    Fit A: λ = {a_simple:.4f} / log₁₀(N)")
            print(f"      R² = {r2_simple:.6f}")
            for logn, lam, pred in zip(log_ns, lambdas, pred_simple):
                print(f"      log₁₀N={logn:.1f}: λ_obs={lam:.6f}, λ_pred={pred:.6f}, "
                      f"residual={lam-pred:+.6f}")
            print()
            
            # Fit 1b: λ = a / log10(N) + b  (two parameters)
            coeffs_1b = np.polyfit(inv_logN, lambdas_arr, 1)
            a_2param = coeffs_1b[0]
            b_2param = coeffs_1b[1]
            pred_2param = a_2param * inv_logN + b_2param
            resid_2param = lambdas_arr - pred_2param
            ss_res_2 = np.sum(resid_2param**2)
            r2_2param = 1 - ss_res_2 / ss_tot if ss_tot > 0 else 0
            
            print(f"    Fit B: λ = {a_2param:.4f} / log₁₀(N) + {b_2param:.6f}")
            print(f"      R² = {r2_2param:.6f}")
            print(f"      Asymptotic λ(N→∞) = {b_2param:.6f}")
            for logn, lam, pred in zip(log_ns, lambdas, pred_2param):
                print(f"      log₁₀N={logn:.1f}: λ_obs={lam:.6f}, λ_pred={pred:.6f}, "
                      f"residual={lam-pred:+.6f}")
            print()
            
            # Fit 1c: λ = a * log10(N)^(-α)  (power law)
            log_log = np.log(log_ns_arr)
            log_lambda = np.log(lambdas_arr)
            coeffs_power = np.polyfit(log_log, log_lambda, 1)
            alpha = -coeffs_power[0]
            a_power = np.exp(coeffs_power[1])
            pred_power = a_power * log_ns_arr**(-alpha)
            resid_power = lambdas_arr - pred_power
            ss_res_p = np.sum(resid_power**2)
            r2_power = 1 - ss_res_p / ss_tot if ss_tot > 0 else 0
            
            print(f"    Fit C: λ = {a_power:.4f} * log₁₀(N)^(-{alpha:.4f})")
            print(f"      R² = {r2_power:.6f}")
            print(f"      Exponent α = {alpha:.4f} (1.0 = exact 1/logN)")
            for logn, lam, pred in zip(log_ns, lambdas, pred_power):
                print(f"      log₁₀N={logn:.1f}: λ_obs={lam:.6f}, λ_pred={pred:.6f}, "
                      f"residual={lam-pred:+.6f}")
            print()
            
            # Natural log version: λ = a / ln(N) 
            ln_N_mids = log_ns_arr * np.log(10)  # convert to natural log
            inv_lnN = 1.0 / ln_N_mids
            a_nat = float(np.sum(lambdas_arr * inv_lnN) / np.sum(inv_lnN**2))
            pred_nat = a_nat * inv_lnN
            resid_nat = lambdas_arr - pred_nat
            ss_res_nat = np.sum(resid_nat**2)
            r2_nat = 1 - ss_res_nat / ss_tot if ss_tot > 0 else 0
            
            print(f"    Fit D: λ = {a_nat:.4f} / ln(N)")
            print(f"      R² = {r2_nat:.6f}")
            print(f"      Temperature T = 1/λ = ln(N) / {a_nat:.4f}")
            print(f"      ──── THIS IS THE PNT CONNECTION ────")
            print(f"      Mean prime gap ≈ ln(N). Boltzmann T ∝ ln(N).")
            print(f"      The sieve temperature IS the mean prime gap.")
            for logn, lnN, lam, pred in zip(log_ns, ln_N_mids, lambdas, pred_nat):
                print(f"      ln(N)={lnN:.2f}: λ_obs={lam:.6f}, λ_pred={pred:.6f}, "
                      f"residual={lam-pred:+.6f}")
            print()
            
            # Verdict on Prediction 1
            best_r2 = max(r2_simple, r2_2param, r2_power, r2_nat)
            best_name = ['A: a/log₁₀N', 'B: a/log₁₀N+b', 'C: power law', 'D: a/ln(N)'][
                [r2_simple, r2_2param, r2_power, r2_nat].index(best_r2)]
            
            print(f"    ╔══════════════════════════════════════════════╗")
            print(f"    ║  PREDICTION 1 VERDICT                       ║")
            print(f"    ║  Best fit: {best_name:<35s}║")
            print(f"    ║  R² = {best_r2:.6f}                            ║")
            if best_r2 > 0.99:
                print(f"    ║  STATUS: ████ CONFIRMED ████                ║")
            elif best_r2 > 0.95:
                print(f"    ║  STATUS: ███ STRONG ███                     ║")
            elif best_r2 > 0.90:
                print(f"    ║  STATUS: ██ GOOD ██                         ║")
            else:
                print(f"    ║  STATUS: █ WEAK █                           ║")
            print(f"    ╚══════════════════════════════════════════════╝")
            print()
        
        # ─── PREDICTION 2: θ constant across scales ───
        print(f"  PREDICTION 2: θ(N) = constant (scale-invariant spiral)")
        print()
        
        if len(angles) >= 3:
            angles_arr = np.array(angles)
            theta_mean = float(np.mean(angles_arr))
            theta_std = float(np.std(angles_arr))
            theta_cv = theta_std / abs(theta_mean) if theta_mean != 0 else float('inf')
            
            # Fit: θ = a + b * log10(N)
            # We only have angles where complex eigs exist
            angle_logns = log_ns_arr[:len(angles_arr)]
            if len(angle_logns) == len(angles_arr):
                coeffs_theta = np.polyfit(angle_logns, angles_arr, 1)
                theta_slope = coeffs_theta[0]
                theta_intercept = coeffs_theta[1]
                pred_theta = theta_slope * angle_logns + theta_intercept
                resid_theta = angles_arr - pred_theta
                ss_res_theta = np.sum(resid_theta**2)
                ss_tot_theta = np.sum((angles_arr - np.mean(angles_arr))**2)
                r2_theta = 1 - ss_res_theta / ss_tot_theta if ss_tot_theta > 0 else 0
            else:
                theta_slope = 0
                r2_theta = 0
            
            print(f"    θ values across scales:")
            for i, (logn, ang) in enumerate(zip(log_ns, angles)):
                cols_per_step = n_cols * ang / 360
                print(f"      log₁₀N={logn:.1f}: θ = {ang:.2f}° "
                      f"({cols_per_step:.2f} columns/step)")
            
            print(f"    Mean θ = {theta_mean:.2f}° ± {theta_std:.2f}°")
            print(f"    CV(θ) = {theta_cv:.4f}")
            print(f"    Drift: dθ/d(logN) = {theta_slope:.4f}°/decade")
            
            print()
            print(f"    ╔══════════════════════════════════════════════╗")
            print(f"    ║  PREDICTION 2 VERDICT                       ║")
            if theta_cv < 0.02:
                print(f"    ║  CV = {theta_cv:.4f} < 0.02                      ║")
                print(f"    ║  STATUS: ████ CONSTANT ████                ║")
                print(f"    ║  θ IS scale-invariant                      ║")
            elif theta_cv < 0.05:
                print(f"    ║  CV = {theta_cv:.4f} < 0.05                      ║")
                print(f"    ║  STATUS: ███ APPROXIMATELY CONSTANT ███   ║")
            elif abs(theta_slope) > 1.0:
                print(f"    ║  Drift = {theta_slope:.2f}°/decade                  ║")
                print(f"    ║  STATUS: ██ DRIFTING ██                    ║")
                print(f"    ║  θ is NOT scale-invariant                  ║")
            else:
                print(f"    ║  CV = {theta_cv:.4f}, drift = {theta_slope:.2f}°/decade       ║")
                print(f"    ║  STATUS: █ MARGINALLY CONSTANT █           ║")
            print(f"    ╚══════════════════════════════════════════════╝")
            print()
        
        # ─── PREDICTION 3: |λ₁| → 1 or → 0? ───
        print(f"  PREDICTION 3: |λ₁(N)| scaling — spiral persists or dies?")
        print()
        
        if len(magnitudes) >= 3:
            magnitudes_arr = np.array(magnitudes)
            
            # Fit |λ₁| = a * log10(N)^(-β)
            log_mags = np.log(magnitudes_arr[magnitudes_arr > 0])
            log_logns = np.log(log_ns_arr[:len(log_mags)])
            
            if len(log_mags) >= 2:
                coeffs_mag = np.polyfit(log_logns, log_mags, 1)
                beta = -coeffs_mag[0]
                a_mag = np.exp(coeffs_mag[1])
                pred_mags = a_mag * log_ns_arr[:len(log_mags)]**(-beta)
                resid_mags = magnitudes_arr[:len(log_mags)] - pred_mags
                ss_res_m = np.sum(resid_mags**2)
                ss_tot_m = np.sum((magnitudes_arr[:len(log_mags)] - 
                                   np.mean(magnitudes_arr[:len(log_mags)]))**2)
                r2_mag = 1 - ss_res_m / ss_tot_m if ss_tot_m > 0 else 0
            else:
                beta = 0
                a_mag = 0
                r2_mag = 0
            
            print(f"    |λ₁| values across scales:")
            for logn, mag in zip(log_ns, magnitudes):
                bar = '█' * int(mag * 40)
                print(f"      log₁₀N={logn:.1f}: |λ₁| = {mag:.6f}  {bar}")
            
            print()
            print(f"    Power law fit: |λ₁| = {a_mag:.4f} * log₁₀(N)^(-{beta:.4f})")
            print(f"    R² = {r2_mag:.6f}")
            print(f"    Exponent β = {beta:.4f}")
            
            # Extrapolate
            for target_logN in [10, 15, 20, 50, 100]:
                pred_val = a_mag * target_logN**(-beta)
                print(f"    Extrapolation: log₁₀N={target_logN} → |λ₁| ≈ {pred_val:.4f}")
            
            print()
            print(f"    ╔══════════════════════════════════════════════╗")
            print(f"    ║  PREDICTION 3 VERDICT                       ║")
            if beta < 0.1:
                print(f"    ║  β = {beta:.4f} ≈ 0                            ║")
                print(f"    ║  STATUS: ████ SPIRAL PERSISTS ████         ║")
                print(f"    ║  |λ₁| → constant > 0 as N → ∞            ║")
            elif beta > 0.8:
                print(f"    ║  β = {beta:.4f}                                 ║")
                print(f"    ║  STATUS: ████ SPIRAL DIES ████             ║")
                print(f"    ║  |λ₁| → 0 as N → ∞                       ║")
            else:
                print(f"    ║  β = {beta:.4f}                                 ║")
                print(f"    ║  STATUS: ██ SLOW DECAY ██                  ║")
                print(f"    ║  Spiral weakens but persists to large N    ║")
            print(f"    ╚══════════════════════════════════════════════╝")
            print()
        
        # ─── BOLTZMANN MATRIX COMPARISON ───
        print(f"  {'─'*60}")
        print(f"  BOLTZMANN MODEL vs EMPIRICAL (largest scale)")
        print(f"  {'─'*60}")
        print()
        
        if scale_results:
            best = scale_results[-1]  # largest scale
            lam = best['lambda_fit']
            T_emp = np.array(best['transition_matrix'])
            T_bol = boltzmann_matrix(lam, n_cols)
            
            if n_cols <= 8:
                print(f"    Empirical transition matrix (mod {m}):")
                header = "       " + "".join(f"{c:>7}" for c in cols)
                print(f"    {header}")
                for i, ci in enumerate(cols):
                    row_str = f"    {ci:>3} "
                    for j in range(n_cols):
                        row_str += f" {T_emp[i, j]:5.3f} "
                    print(row_str)
                print()
                
                print(f"    Boltzmann model (λ = {lam:.6f}):")
                print(f"    {header}")
                for i, ci in enumerate(cols):
                    row_str = f"    {ci:>3} "
                    for j in range(n_cols):
                        row_str += f" {T_bol[i, j]:5.3f} "
                    print(row_str)
                print()
                
                print(f"    Residual (Empirical - Boltzmann):")
                print(f"    {header}")
                for i, ci in enumerate(cols):
                    row_str = f"    {ci:>3} "
                    for j in range(n_cols):
                        diff = T_emp[i, j] - T_bol[i, j]
                        if abs(diff) > 0.005:
                            row_str += f" {diff:+5.3f}*"
                        else:
                            row_str += f" {diff:+5.3f} "
                    print(row_str)
                print(f"    (* = residual > 0.005)")
                print()
                
                print(f"    Frobenius error: {best['frobenius_error']:.6f}")
                print(f"    Max element error: {best['max_element_error']:.6f}")
                print(f"    Mean element error: {best['mean_element_error']:.6f}")
                print(f"    Relative error: {best['relative_error']:.4f}")
                print()
        
        all_results[f"mod_{m}"] = {
            "modulus": m,
            "phi_m": n_cols,
            "columns": cols,
            "scales": scale_results,
            "log_ns": log_ns,
            "lambdas": lambdas,
            "angles": angles,
            "magnitudes": magnitudes,
            "chiralities": chiralities,
        }
    
    # ═══════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ═══════════════════════════════════════════════════════
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  GRAND SUMMARY                                             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    for m_key, m_data in all_results.items():
        m = m_data['modulus']
        print(f"  mod {m} (φ={m_data['phi_m']}):")
        
        if m_data['lambdas']:
            lams = m_data['lambdas']
            print(f"    λ range: {min(lams):.6f} – {max(lams):.6f}")
            print(f"    λ ratio (first/last): {lams[0]/lams[-1]:.4f}")
            print(f"    log₁₀N ratio (last/first): {m_data['log_ns'][-1]/m_data['log_ns'][0]:.4f}")
        
        if m_data['angles']:
            angs = m_data['angles']
            print(f"    θ range: {min(angs):.2f}° – {max(angs):.2f}°")
            print(f"    θ CV: {np.std(angs)/abs(np.mean(angs)):.4f}")
        
        if m_data['magnitudes']:
            mags = m_data['magnitudes']
            print(f"    |λ₁| range: {min(mags):.6f} – {max(mags):.6f}")
        
        print()
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")
    
    # ═══════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════
    
    output = {
        "experiment": "Wave 22 — The Boltzmann Fit",
        "timestamp": int(time.time() * 1000),
        "elapsed_seconds": elapsed,
        "max_prime": int(max_prime),
        "total_primes": int(len(all_primes)),
        "moduli_tested": moduli,
        "results": all_results,
    }
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                           f'wave22_boltzmann_{int(time.time()*1000)}.json')
    out_path = os.path.normpath(out_path)
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {out_path}")
    
    return output


if __name__ == "__main__":
    results = run_experiment(max_prime=10**8)
