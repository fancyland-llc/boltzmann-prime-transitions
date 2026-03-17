#!/usr/bin/env python3
"""
Wave 22e — Extend Boltzmann verification to 10^10.
Memory-efficient: streams primes through segmented sieve,
counts transitions on the fly without storing the full prime list.
"""

import numpy as np
from math import gcd, log, sqrt, isqrt
from scipy.optimize import minimize_scalar
import json
import time
import os


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


def streaming_transition_count(limit, windows, moduli):
    """
    Segmented sieve to `limit`. For each (modulus, window), count
    transitions between consecutive primes on the fly.
    Returns dict: (m, window_idx) -> (count_matrix, n_primes_in_window)
    
    Memory: O(sqrt(limit)) for small primes + O(segment_size) for sieve segment.
    Never stores the full prime list.
    """
    segment_size = 2 * 10**6  # 2M per segment for speed
    sqrt_limit = isqrt(limit) + 1
    
    # Small primes via simple sieve
    is_prime_small = np.ones(sqrt_limit + 1, dtype=bool)
    is_prime_small[0] = is_prime_small[1] = False
    for i in range(2, int(sqrt(sqrt_limit)) + 1):
        if is_prime_small[i]:
            is_prime_small[i*i::i] = False
    small_primes = np.where(is_prime_small)[0]
    
    print(f"  Small primes (up to {sqrt_limit:,}): {len(small_primes):,}")
    
    # Build col_idx maps
    col_idx_map = {}
    for m in moduli:
        cols = admissible_columns(m)
        col_idx_map[m] = {c: i for i, c in enumerate(cols)}
    
    # Initialize transition count matrices and state
    # State: last prime's residue index per modulus (to carry across segments)
    counts = {}
    n_primes = {}
    last_residue_idx = {}  # per (m, window_idx): last residue class index
    last_prime_window = {}  # per m: which window the last prime was in
    
    for m in moduli:
        n_cols = len(admissible_columns(m))
        for wi in range(len(windows)):
            counts[(m, wi)] = np.zeros((n_cols, n_cols), dtype=np.int64)
            n_primes[(m, wi)] = 0
    
    # For tracking the very last prime seen (globally, for transition across segments)
    for m in moduli:
        last_residue_idx[m] = None
        last_prime_window[m] = None
    
    # Process small primes first (they are primes too!)
    total_primes = 0
    for p in small_primes:
        total_primes += 1
        for m in moduli:
            r = int(p % m)
            cidx = col_idx_map[m]
            if r not in cidx:
                continue
            ri = cidx[r]
            
            # Which window?
            wi = None
            for idx, (lo, hi, _) in enumerate(windows):
                if lo <= p < hi:
                    wi = idx
                    break
            
            if wi is not None:
                n_primes[(m, wi)] += 1
                if last_residue_idx[m] is not None:
                    # Count transition from last prime to this one
                    # But only if they're in the same window OR we want cross-window
                    # For transition matrix: we count within-window consecutive primes
                    # Actually: we need CONSECUTIVE primes that are both in the window
                    if last_prime_window[m] == wi:
                        counts[(m, wi)][last_residue_idx[m], ri] += 1
                    elif last_prime_window[m] is not None and last_prime_window[m] == wi:
                        counts[(m, wi)][last_residue_idx[m], ri] += 1
            
            last_residue_idx[m] = ri
            last_prime_window[m] = wi if wi is not None else last_prime_window[m]
    
    # Segmented sieve
    n_segments = (limit - sqrt_limit) // segment_size + 1
    last_report = time.time()
    
    for seg_idx in range(n_segments):
        low = sqrt_limit + 1 + seg_idx * segment_size
        high = min(low + segment_size - 1, limit)
        if low > limit:
            break
        
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
        
        # Extract primes in this segment
        prime_offsets = np.where(seg)[0]
        primes_in_seg = prime_offsets.astype(np.int64) + low
        total_primes += len(primes_in_seg)
        
        # Process each prime
        for p in primes_in_seg:
            for m in moduli:
                r = int(p % m)
                cidx = col_idx_map[m]
                if r not in cidx:
                    continue
                ri = cidx[r]
                
                # Which window?
                wi = None
                for idx, (wlo, whi, _) in enumerate(windows):
                    if wlo <= p < whi:
                        wi = idx
                        break
                
                if wi is not None:
                    n_primes[(m, wi)] += 1
                    # Transition: if last prime was also in this window
                    if last_residue_idx[m] is not None and last_prime_window[m] == wi:
                        counts[(m, wi)][last_residue_idx[m], ri] += 1
                
                last_residue_idx[m] = ri
                if wi is not None:
                    last_prime_window[m] = wi
        
        # Progress report every 10 seconds
        now = time.time()
        if now - last_report > 10:
            pct = (high / limit) * 100
            print(f"    {pct:.1f}% (up to {high:,}, {total_primes:,} primes so far)")
            last_report = now
    
    print(f"  Total primes found: {total_primes:,}")
    return counts, n_primes


def analyze_window(counts_matrix, n_primes_count, cols, m, log10_mid):
    n_cols = len(cols)
    ln_N = log10_mid * np.log(10)
    
    T = counts_matrix.astype(np.float64)
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
        "n_primes": n_primes_count,
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
    print("  WAVE 22e — EXTENDING TO 10^10")
    print("  Streaming segmented sieve — constant memory")
    print("=" * 70)
    print()
    
    LIMIT = 10**10
    
    windows = [
        (10**3,  10**4,  3.5),
        (10**4,  10**5,  4.5),
        (10**5,  10**6,  5.5),
        (10**6,  10**7,  6.5),
        (10**7,  10**8,  7.5),
        (10**8,  10**9,  8.5),
        (10**9,  10**10, 9.5),  # NEW
    ]
    
    moduli = [30, 210]
    
    start = time.time()
    
    print("Streaming segmented sieve to 10^10...")
    counts, n_primes = streaming_transition_count(LIMIT, windows, moduli)
    sieve_time = time.time() - start
    print(f"  Sieve + counting: {sieve_time:.1f}s")
    print()
    
    all_results = {}
    
    for m in moduli:
        cols = admissible_columns(m)
        n_cols = len(cols)
        
        print(f"{'='*60}")
        print(f"  MODULUS {m} (phi={n_cols})")
        print(f"{'='*60}")
        print()
        
        print(f"  {'Scale':<14} {'ln(N)':>7} {'lam_PNT':>10} {'lam_fit':>10} "
              f"{'lam*lnN':>8} {'R2_zero':>10} {'R2_fit':>10} {'n_primes':>12}")
        print(f"  {'-'*84}")
        
        results_for_m = []
        for wi, (lo, hi, log_mid) in enumerate(windows):
            np_count = n_primes[(m, wi)]
            if np_count < 100:
                continue
            
            r = analyze_window(counts[(m, wi)], np_count, cols, m, log_mid)
            results_for_m.append(r)
            
            marker = " <<<" if log_mid == 9.5 else ""
            print(f"  10^{log_mid-0.5:.0f}-10^{log_mid+0.5:.0f}  "
                  f"{r['ln_N']:>7.2f} {r['lambda_pnt']:>10.5f} {r['lambda_fitted']:>10.5f} "
                  f"{r['lambda_ratio']:>8.4f} {r['r2_zero_param']:>10.6f} {r['r2_fitted']:>10.6f}"
                  f" {r['n_primes']:>12,}{marker}")
        
        print()
        
        if m == 30 and results_for_m:
            new = results_for_m[-1]
            print(f"  NEW WINDOW 10^9-10^10: {new['n_primes']:,} primes")
            print(f"  R²(zero-param) = {new['r2_zero_param']:.6f}")
            print(f"  λ·ln(N) = {new['lambda_ratio']:.4f}")
            print(f"  Max err = {new['max_err_zero']:.6f}")
            print(f"  Mean err = {new['mean_err_zero']:.6f}")
            print()
        
        all_results[f"mod_{m}"] = results_for_m
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Save results
    output = {
        "experiment": "Wave 22e — 10^10 extension (streaming)",
        "timestamp": int(time.time() * 1000),
        "elapsed_seconds": elapsed,
        "max_prime": int(LIMIT),
        "results": all_results,
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                           f'wave22e_10e10_{int(time.time()*1000)}.json')
    out_path = os.path.normpath(out_path)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {out_path}")
