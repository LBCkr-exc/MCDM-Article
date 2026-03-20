"""
monte_carlo_criteria_sensitivity.py
====================================
Monte Carlo sensitivity analysis on MCDM criteria values.

Answers reviewer comment Q8:
"The framework explores uncertainty in weights but assumes deterministic
criteria values. However, mechanical and environmental data inherently
contain variability. Monte Carlo propagation or interval MCDM approaches
would improve robustness credibility."

Method
------
Criteria values are perturbed by multiplicative Gaussian noise:
    x_perturbed = x_baseline * (1 + epsilon)
    epsilon ~ N(0, CV^2),  CV = 10%
    clipped to [0.5 * x, 2.0 * x] to prevent physically implausible values.

The full CRITIC-weighted TOPSIS pipeline is recomputed for each of
N = 5000 independent simulation runs.

Outputs
-------
- Per-material rank statistics (mean, std, min, max, stability %)
- Rank frequency matrix
- Global Spearman rank correlation vs baseline
- Critical boundary analysis (NFC vs conventional materials)
- CV sensitivity sweep (5%, 10%, 15%, 20%)

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.stats import spearmanr

# ── Data ──────────────────────────────────────────────────────────────────────

MATERIALS = [
    'Flax/Epoxy', 'Carbon/Epoxy', 'Glass/Epoxy',
    'Aluminium', 'Steel', 'Hemp/Epoxy', 'Jute/Epoxy'
]

CRITERIA = ['EF', 'Density', 'T.Mod', 'T.Str', 'Elong', 'F.Mod', 'F.Str', 'CTE', 'Cost']

# Raw decision matrix (7 materials × 9 criteria)
# Units: EF(mPt/kg), Density(g/cm³), T.Mod(GPa), T.Str(MPa),
#        Elong(%), F.Mod(GPa), F.Str(MPa), CTE(µm/m°C), Cost(€/kg)
RAW = np.array([
    [0.59,  1.35,  35,   610,  1.80,  39,   520, 2.0e-5, 3.8],   # Flax/Epoxy
    [4.98,  1.42, 120,  2135,  1.15, 134,  1815, 2.98e-6, 20.0],  # Carbon/Epoxy
    [0.69,  1.80,  40,  1135,  1.30,  43,   965, 3.5e-5,  4.0],   # Glass/Epoxy
    [1.58,  2.70,  70,   330,  3.40,  70,   325, 2.3e-5, 10.0],   # Aluminium
    [0.77,  7.90, 200,   750, 40.00, 200,   900, 1.7e-5,  2.5],   # Steel
    [0.55,  1.31,  28,   398,  2.00,  32,   340, 2.5e-5,  4.0],   # Hemp/Epoxy
    [0.57,  1.30,  12,   335,  2.50,  13,   285, 3.0e-5,  3.0],   # Jute/Epoxy
])

# True = benefit (higher is better), False = cost (lower is better)
IS_BENEFIT = [False, False, True, True, True, True, True, False, False]

# CRITIC weights (sum = 1)
W_CRITIC = np.array([0.1036, 0.1068, 0.0791, 0.0971, 0.1091,
                     0.0772, 0.0873, 0.2260, 0.1138])

# ── MCDM functions ────────────────────────────────────────────────────────────

def normalize_minmax(X, is_benefit):
    """Min-max normalisation. All columns → higher = better after inversion."""
    X = np.asarray(X, dtype=float)
    R = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        mn, mx = col.min(), col.max()
        if mx == mn:
            R[:, j] = 0.0
        else:
            scaled = (col - mn) / (mx - mn)
            R[:, j] = scaled if is_benefit[j] else 1.0 - scaled
    return R


def topsis(R, w):
    """TOPSIS closeness score (higher = better)."""
    V = R * w
    A_pos = V.max(axis=0)
    A_neg = V.min(axis=0)
    d_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    return d_neg / (d_pos + d_neg + 1e-12)


def ranks_descending(scores):
    """Rank from highest (1) to lowest score."""
    return np.argsort(np.argsort(-scores)) + 1


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def run_monte_carlo(raw, is_benefit, weights, cv=0.10, n_runs=5000, seed=42):
    """
    Perturb criteria values with multiplicative Gaussian noise and
    recompute TOPSIS rankings for each run.

    Parameters
    ----------
    raw       : (n_materials, n_criteria) baseline decision matrix
    is_benefit: list of bool, benefit/cost direction per criterion
    weights   : (n_criteria,) weight vector
    cv        : coefficient of variation for noise (default 0.10 = 10%)
    n_runs    : number of Monte Carlo simulations
    seed      : random seed for reproducibility

    Returns
    -------
    baseline_ranks : (n_materials,) integer ranks from baseline data
    all_ranks      : (n_runs, n_materials) integer ranks per simulation
    all_scores     : (n_runs, n_materials) TOPSIS scores per simulation
    """
    rng = np.random.default_rng(seed)

    # Baseline
    R0 = normalize_minmax(raw, is_benefit)
    base_scores = topsis(R0, weights)
    baseline_ranks = ranks_descending(base_scores)

    all_ranks  = np.zeros((n_runs, raw.shape[0]), dtype=int)
    all_scores = np.zeros((n_runs, raw.shape[0]))

    for i in range(n_runs):
        # Multiplicative perturbation: x' = x * (1 + epsilon)
        epsilon = rng.normal(0, cv, size=raw.shape)
        noise   = np.clip(1.0 + epsilon, 0.5, 2.0)
        X_pert  = raw * noise
        R_pert  = normalize_minmax(X_pert, is_benefit)
        sc      = topsis(R_pert, weights)
        all_ranks[i]  = ranks_descending(sc)
        all_scores[i] = sc

    return baseline_ranks, all_ranks, all_scores


# ── Results reporting ─────────────────────────────────────────────────────────

def report(materials, baseline_ranks, all_ranks, cv):
    n_runs = all_ranks.shape[0]
    print(f"\n{'='*65}")
    print(f"MONTE CARLO CRITERIA SENSITIVITY  (CV={cv*100:.0f}%, N={n_runs}, seed=42)")
    print(f"{'='*65}")

    # Global Spearman
    rhos = [spearmanr(baseline_ranks, all_ranks[i])[0] for i in range(n_runs)]
    rhos = np.array(rhos)
    print(f"\nGlobal Spearman ρ (baseline vs MC runs):")
    print(f"  Mean  = {rhos.mean():.4f}")
    print(f"  Std   = {rhos.std():.4f}")
    print(f"  Min   = {rhos.min():.4f}")
    print(f"  5th % = {np.percentile(rhos, 5):.4f}")
    print(f"  P(ρ > 0.9) = {(rhos > 0.9).mean()*100:.1f}%")
    print(f"  P(ρ > 0.8) = {(rhos > 0.8).mean()*100:.1f}%")

    # Per-material statistics
    print(f"\n{'Material':<22} {'Base':>5} {'Mean':>6} {'Std':>5} "
          f"{'Min':>4} {'Max':>4} {'Exact%':>7} {'±1%':>7}")
    print("-" * 65)
    for j, m in enumerate(materials):
        r = all_ranks[:, j]
        exact = (r == baseline_ranks[j]).mean() * 100
        pm1   = (np.abs(r - baseline_ranks[j]) <= 1).mean() * 100
        print(f"{m:<22} {baseline_ranks[j]:>5} {r.mean():>6.2f} {r.std():>5.2f} "
              f"{r.min():>4} {r.max():>4} {exact:>6.1f}% {pm1:>6.1f}%")

    # Rank frequency matrix
    n_mat = len(materials)
    print(f"\nRank frequency matrix (% of {n_runs} runs):")
    print(f"{'Material':<22}", end='')
    for r in range(1, n_mat + 1):
        print(f"  R{r}%", end='')
    print()
    for j, m in enumerate(materials):
        print(f"{m:<22}", end='')
        for r in range(1, n_mat + 1):
            pct = (all_ranks[:, j] == r).mean() * 100
            print(f" {pct:>5.1f}", end='')
        print()

    # Critical boundaries (NFC vs conventional)
    nfc_idx  = [0, 5, 6]   # Flax, Hemp, Jute
    conv_idx = [2, 3]       # Glass, Aluminium
    print("\nCritical boundary stability (NFC vs lower-tier conventional):")
    for ni in nfc_idx:
        for ci in conv_idx:
            p = (all_ranks[:, ni] < all_ranks[:, ci]).mean() * 100
            print(f"  {materials[ni]} outranks {materials[ci]}: {p:.1f}%")


def cv_sweep(raw, is_benefit, weights, baseline_ranks,
             cv_values=(0.05, 0.10, 0.15, 0.20), n_runs=3000):
    """Sweep over different CV levels to assess robustness of conclusions."""
    print(f"\nCV sensitivity sweep (N={n_runs} per level):")
    print(f"{'CV':>5}  {'mean ρ':>8}  {'5th %':>7}  {'P(ρ>0.9)':>10}")
    print("-" * 40)
    for cv in cv_values:
        rng = np.random.default_rng(42)
        rhos = []
        for _ in range(n_runs):
            noise  = np.clip(1.0 + rng.normal(0, cv, size=raw.shape), 0.3, 3.0)
            R_pert = normalize_minmax(raw * noise, is_benefit)
            r      = ranks_descending(topsis(R_pert, weights))
            rhos.append(spearmanr(baseline_ranks, r)[0])
        rhos = np.array(rhos)
        print(f"{cv*100:>4.0f}%  {rhos.mean():>8.4f}  "
              f"{np.percentile(rhos,5):>7.4f}  {(rhos>0.9).mean()*100:>9.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    baseline_ranks, all_ranks, all_scores = run_monte_carlo(
        RAW, IS_BENEFIT, W_CRITIC, cv=0.10, n_runs=5000, seed=42
    )
    report(MATERIALS, baseline_ranks, all_ranks, cv=0.10)
    cv_sweep(RAW, IS_BENEFIT, W_CRITIC, baseline_ranks)

    print("\nDone. Results fully reproducible with seed=42.")
