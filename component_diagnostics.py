"""
component_diagnostics.py — Run AFTER score_all_candidates_like_ht()

PURPOSE:
  Answer the question: "Do all three components actually contribute
  to differentiating candidates, or is one dominant?"

  This script produces the tables and figures needed for the paper's
  methodology defense.

USAGE:
  python component_diagnostics.py

OUTPUT:
  1. Component statistics table (mean, std, range, coefficient of variation)
  2. Pairwise correlation matrix (are components redundant?)
  3. Weight sensitivity grid — how top-10 sites change across W1/W2/W3
  4. Component contribution plot — what % of score variance comes from each
  5. Rank stability analysis — how many sites remain in top-10 across weight configs
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from itertools import product

# ---- Import the model ----
from model_approach2 import (
    load_all,
    score_all_candidates_like_ht,
    norm_weight,
    MILE_M,
)


def run_diagnostics():
    print("=" * 70)
    print("COMPONENT DIAGNOSTICS")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1/6] Loading data...")
    state = load_all()

    # ---- Score with return_all to get component arrays ----
    print("\n[2/6] Scoring all candidates (W1=0.4, W2=0.3, W3=0.3)...")
    top_n, _, _, all_scored = score_all_candidates_like_ht(
        state,
        radius_miles=5.0,
        beta=2.0,
        K=3,
        W1=0.4, W2=0.3, W3=0.3,
        return_all=True,
    )

    # ---- Extract raw and normalised components ----
    P_raw = all_scored["potential_raw"].values
    A_raw = all_scored["access_score_dj"].values
    S_raw = all_scored["stores_per_10k"].values

    P_norm = all_scored["potential_norm"].values
    A_norm = all_scored["access_norm"].values
    S_norm = all_scored["s10k_norm"].values

    score = all_scored["pair_score"].values
    n = len(score)

    # ==============================================================
    # TABLE 1: Component Statistics
    # ==============================================================
    print("\n" + "=" * 70)
    print("TABLE 1: Component Statistics (Raw)")
    print("=" * 70)

    stats = {}
    for name, arr in [("P (Market Potential)", P_raw),
                       ("A (Accessibility)", A_raw),
                       ("S (Competition)", S_raw)]:
        valid = arr[np.isfinite(arr) & (arr > 0)]
        cv = np.std(valid) / np.mean(valid) if np.mean(valid) > 0 else 0
        stats[name] = {
            "mean": np.mean(valid),
            "std": np.std(valid),
            "min": np.min(valid),
            "max": np.max(valid),
            "CV": cv,
            "range": np.max(valid) - np.min(valid),
        }
        print(f"  {name}:")
        print(f"    mean={stats[name]['mean']:.4f}  std={stats[name]['std']:.4f}  "
              f"CV={cv:.3f}")
        print(f"    range=[{stats[name]['min']:.4f}, {stats[name]['max']:.4f}]  "
              f"spread={stats[name]['range']:.4f}")

    print("\n" + "=" * 70)
    print("TABLE 2: Component Statistics (After norm_weight to [0.01, 1.0])")
    print("=" * 70)

    for name, arr in [("P_norm", P_norm), ("A_norm", A_norm), ("S_norm", S_norm)]:
        valid = arr[np.isfinite(arr)]
        cv = np.std(valid) / np.mean(valid) if np.mean(valid) > 0 else 0
        print(f"  {name}: mean={np.mean(valid):.4f}  std={np.std(valid):.4f}  "
              f"CV={cv:.3f}  range=[{np.min(valid):.4f}, {np.max(valid):.4f}]")

    # ==============================================================
    # TABLE 2: Pairwise Correlation
    # ==============================================================
    print("\n" + "=" * 70)
    print("TABLE 3: Pairwise Spearman Rank Correlation (normalised components)")
    print("=" * 70)

    from scipy.stats import spearmanr

    components = {"P_norm": P_norm, "A_norm": A_norm, "S_norm": S_norm}
    names = list(components.keys())
    corr_matrix = np.zeros((3, 3))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            rho, _ = spearmanr(components[n1], components[n2])
            corr_matrix[i, j] = rho

    print(f"  {'':>8s}  {'P_norm':>8s}  {'A_norm':>8s}  {'S_norm':>8s}")
    for i, n1 in enumerate(names):
        row = "  " + f"{n1:>8s}"
        for j in range(3):
            row += f"  {corr_matrix[i,j]:>8.3f}"
        print(row)

    # KEY INSIGHT for paper:
    print("\n  INTERPRETATION:")
    pa_rho = corr_matrix[0, 1]
    ps_rho = corr_matrix[0, 2]
    as_rho = corr_matrix[1, 2]
    print(f"    P-A correlation: {pa_rho:.3f}", end="")
    if abs(pa_rho) > 0.7:
        print("  ← HIGH: these components are partially redundant")
    elif abs(pa_rho) < 0.3:
        print("  ← LOW: these components capture different information")
    else:
        print("  ← MODERATE")

    print(f"    P-S correlation: {ps_rho:.3f}", end="")
    if abs(ps_rho) > 0.7:
        print("  ← HIGH: demand-rich areas also competition-heavy")
    else:
        print("")

    print(f"    A-S correlation: {as_rho:.3f}")

    # ==============================================================
    # TABLE 3: Score Variance Decomposition
    # ==============================================================
    print("\n" + "=" * 70)
    print("TABLE 4: Score Variance Decomposition (W1=0.4, W2=0.3, W3=0.3)")
    print("=" * 70)

    W1, W2, W3 = 0.4, 0.3, 0.3
    contrib_P = W1 * P_norm
    contrib_A = W2 * A_norm
    contrib_S = W3 * S_norm  # Note: subtracted in score

    var_P = np.var(contrib_P)
    var_A = np.var(contrib_A)
    var_S = np.var(contrib_S)
    var_total = np.var(score)

    # Cross-terms exist, so contributions won't sum to 100% exactly
    print(f"  Var(W1*P_norm) = {var_P:.6f}  ({100*var_P/var_total:.1f}% of Var(Score))")
    print(f"  Var(W2*A_norm) = {var_A:.6f}  ({100*var_A/var_total:.1f}% of Var(Score))")
    print(f"  Var(W3*S_norm) = {var_S:.6f}  ({100*var_S/var_total:.1f}% of Var(Score))")
    print(f"  Var(Score)     = {var_total:.6f}")
    print(f"\n  NOTE: Cross-covariance terms account for remainder.")

    if var_A / var_total < 0.05:
        print(f"\n  ⚠ ACCESSIBILITY contributes <5% of score variance.")
        print(f"    This is because Charlotte's road network is uniform (A_raw std={np.std(A_raw):.4f}).")
        print(f"    After norm_weight(), the variance increases, but A still has low")
        print(f"    discrimination power in this city. In cities with more heterogeneous")
        print(f"    networks, A would contribute more.")

    # ==============================================================
    # TABLE 4: Weight Sensitivity Grid
    # ==============================================================
    print("\n" + "=" * 70)
    print("TABLE 5: Weight Sensitivity — Top-10 Site Stability")
    print("=" * 70)

    # Generate weight grid (W1, W2, W3 summing to 1)
    weight_configs = []
    for w1 in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for w2 in [0.0, 0.2, 0.4, 0.6]:
            w3 = round(1.0 - w1 - w2, 2)
            if 0 <= w3 <= 1.0:
                weight_configs.append((w1, w2, w3))

    # Collect top-10 site indices for each config
    top10_sets = {}
    baseline_config = (0.4, 0.3, 0.3)

    print(f"\n  Testing {len(weight_configs)} weight configurations...")
    print(f"  {'W1':>5s} {'W2':>5s} {'W3':>5s} | {'Top-10 overlap with baseline':>30s}")
    print(f"  {'-'*5} {'-'*5} {'-'*5} | {'-'*30}")

    # Baseline top-10
    baseline_score = baseline_config[0] * P_norm + baseline_config[1] * A_norm \
                   - baseline_config[2] * S_norm
    baseline_top10 = set(np.argsort(-baseline_score)[:10])

    for w1, w2, w3 in weight_configs:
        s = w1 * P_norm + w2 * A_norm - w3 * S_norm
        top10_idx = set(np.argsort(-s)[:10])
        top10_sets[(w1, w2, w3)] = top10_idx
        overlap = len(top10_idx & baseline_top10)
        marker = "  ← BASELINE" if (w1, w2, w3) == baseline_config else ""
        print(f"  {w1:5.2f} {w2:5.2f} {w3:5.2f} | {overlap:2d}/10 overlap{marker}")

    # Count how many sites appear in top-10 across ALL configs
    from collections import Counter
    all_top10 = Counter()
    for idx_set in top10_sets.values():
        all_top10.update(idx_set)

    n_configs = len(weight_configs)
    robust_sites = [(idx, count) for idx, count in all_top10.most_common()
                    if count >= 0.7 * n_configs]
    print(f"\n  Sites in top-10 across ≥70% of configs: {len(robust_sites)}")
    for idx, count in robust_sites[:5]:
        print(f"    Candidate #{idx}: appears in {count}/{n_configs} configs "
              f"({100*count/n_configs:.0f}%)")

    # ==============================================================
    # TABLE 5: What Each Component "Does" — Rank Change Analysis
    # ==============================================================
    print("\n" + "=" * 70)
    print("TABLE 6: Component Impact — Rank Changes When Removing Each")
    print("=" * 70)

    configs_ablation = {
        "Full (0.4, 0.3, 0.3)": (0.4, 0.3, 0.3),
        "No Competition (0.57, 0.43, 0.0)": (0.57, 0.43, 0.0),
        "No Accessibility (0.57, 0.0, 0.43)": (0.57, 0.0, 0.43),
        "No Demand (0.0, 0.57, 0.43)": (0.0, 0.57, 0.43),
        "Demand Only (1.0, 0.0, 0.0)": (1.0, 0.0, 0.0),
        "Access Only (0.0, 1.0, 0.0)": (0.0, 1.0, 0.0),
        "Competition Only (0.0, 0.0, 1.0)": (0.0, 0.0, 1.0),
    }

    full_score = 0.4 * P_norm + 0.3 * A_norm - 0.3 * S_norm
    full_ranking = np.argsort(-full_score)
    full_top10 = set(full_ranking[:10])

    print(f"\n  {'Configuration':<35s} | {'Top-10 Overlap':>14s} | {'Spearman rho':>12s}")
    print(f"  {'-'*35} | {'-'*14} | {'-'*12}")

    for label, (w1, w2, w3) in configs_ablation.items():
        s = w1 * P_norm + w2 * A_norm - w3 * S_norm
        ranking = np.argsort(-s)
        top10 = set(ranking[:10])
        overlap = len(top10 & full_top10)
        rho, _ = spearmanr(full_score, s)
        print(f"  {label:<35s} | {overlap:>6d}/10     | {rho:>12.4f}")

    # ==============================================================
    # RECOMMENDATIONS FOR PAPER
    # ==============================================================
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR YOUR PAPER")
    print("=" * 70)

    print("""
  Based on these diagnostics, here is what you can claim:

  1. COMPONENT INDEPENDENCE: If P-A, P-S, A-S correlations are all < 0.7,
     the three components capture genuinely different spatial information.
     This justifies having three separate components rather than just one.

  2. VARIANCE CONTRIBUTION: Report what % of score variance each
     component contributes. If A contributes < 5%, explain WHY
     (Charlotte network uniformity) and note this is city-specific.

  3. WEIGHT SENSITIVITY: If top-10 sites change when weights change,
     the framework IS responsive to all three components.
     If top-10 is stable regardless of weights, demand dominates and
     the other components add robustness but not discrimination.

  4. ABLATION: The rank correlation when removing each component tells
     you how much that component matters. If removing accessibility
     gives rho > 0.99 with the full model, it's not contributing.
     If removing demand gives rho < 0.5, demand is essential.

  FOR THE PAPER, include:
    - Table of component statistics (raw + normalised)
    - Correlation matrix
    - Variance decomposition
    - Weight sensitivity grid (or a subset)
    - Ablation rank correlations

  This gives you a rigorous defense of the three-component design.
""")

    # ---- Save results ----
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    diagnostics = {
        "n_candidates": int(n),
        "component_stats_raw": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in stats.items()
        },
        "correlation_matrix": {
            "labels": names,
            "values": corr_matrix.tolist(),
        },
        "variance_decomposition": {
            "W1": 0.4, "W2": 0.3, "W3": 0.3,
            "var_P_contrib": float(var_P),
            "var_A_contrib": float(var_A),
            "var_S_contrib": float(var_S),
            "var_total": float(var_total),
            "pct_P": float(100 * var_P / var_total),
            "pct_A": float(100 * var_A / var_total),
            "pct_S": float(100 * var_S / var_total),
        },
        "robust_sites_70pct": [
            {"candidate_idx": int(idx), "config_count": int(count),
             "pct": float(100 * count / n_configs)}
            for idx, count in robust_sites[:10]
        ],
    }

    out_path = os.path.join(results_dir, "component_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_diagnostics()
