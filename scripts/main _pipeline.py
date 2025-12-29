
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

INPUT_FILE = os.path.join(DATA_DIR, "materials_datas.csv")
ranking_file = os.path.join(DATA_DIR, "MCDM_ranking.csv")


matplotlib.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_KNEE = "#D62728"
CMAP = "viridis"
CMAP_CORR = "coolwarm"
CMAP_HEAT = "cividis"



def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(OUTPUT_DIR)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)


def normalize_benefit_cost(X: np.ndarray, is_benefit: List[bool]) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    A, J = X.shape
    R = np.zeros_like(X)
    for j in range(J):
        col = X[:, j]
        mn, mx = col.min(), col.max()
        if mx == mn:
            R[:, j] = 0
        else:
            scaled = (col - mn) / (mx - mn)
            R[:, j] = scaled if is_benefit[j] else 1 - scaled
    return R


def objectives(R: np.ndarray, w: np.ndarray) -> Tuple[float, float, float]:
    scores = R @ w
    f1 = float(np.max(scores))
    f2 = float(np.min(scores))
    imbalance = float(np.sum((w - 1/len(w))**2))
    f3 = -imbalance
    return f1, f2, f3


# NSGA-II 

def dominates(p, q):
    return np.all(p >= q) and np.any(p > q)


def fast_non_dominated_sort(F):
    N = len(F)
    S = [[] for _ in range(N)]
    n = np.zeros(N, int)
    rank = np.zeros(N, int)
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts


def crowding_distance(front, F):
    if len(front) == 0:
        return np.zeros(0)
    m = F.shape[1]
    dist = np.zeros(len(front))

    for j in range(m):
        idx = np.argsort(F[front, j])
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        fmin, fmax = F[front, j].min(), F[front, j].max()
        if fmax == fmin:
            continue
        for k in range(1, len(front) - 1):
            dist[idx[k]] += (F[front[idx[k + 1]], j] - F[front[idx[k - 1]], j]) / (fmax - fmin)
    return dist


class NSGA2Weights:
    def __init__(self, R, pop=160, gen=200, seed=9):
        self.R = R
        self.A, self.J = R.shape
        self.pop_size = pop
        self.generations = gen
        self.rng = np.random.default_rng(seed)

    def _init_pop(self):
        return self.rng.normal(0, 1, size=(self.pop_size, self.J))

    def _evaluate(self, Z):
        W = np.apply_along_axis(softmax, 1, Z)
        F = np.array([objectives(self.R, w) for w in W])
        return W, F

    def run(self):
        Z = self._init_pop()
        W, F = self._evaluate(Z)

        for g in range(self.generations):
            fronts = fast_non_dominated_sort(F)
            rank = np.full(self.pop_size, np.inf)
            crowd = np.zeros(self.pop_size)

            for r, fr in enumerate(fronts):
                cd = crowding_distance(fr, F)
                for i_local, idx in enumerate(fr):
                    rank[idx] = r
                    crowd[idx] = cd[i_local]

            # Tournament selection
            pool = np.arange(self.pop_size)
            mating = [pool[self.rng.integers(0, len(pool))] for _ in range(self.pop_size)]

            # Variation
            Z_off = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = Z[mating[i]], Z[mating[i+1]]
                # Crossover
                c1 = p1.copy()
                c2 = p2.copy()
                # Mutation
                c1 += self.rng.normal(0, 0.1, size=self.J)
                c2 += self.rng.normal(0, 0.1, size=self.J)
                Z_off.append(c1)
                Z_off.append(c2)

            Z_comb = np.vstack([Z, Z_off])
            W_comb, F_comb = self._evaluate(Z_comb)

            fronts = fast_non_dominated_sort(F_comb)
            Z_next, W_next, F_next = [], [], []

            for fr in fronts:
                if len(Z_next) + len(fr) <= self.pop_size:
                    Z_next.extend(Z_comb[fr])
                    W_next.extend(W_comb[fr])
                    F_next.extend(F_comb[fr])
                else:
                    cd = crowding_distance(fr, F_comb)
                    order = np.argsort(-cd)
                    needed = self.pop_size - len(Z_next)
                    chosen = [fr[o] for o in order[:needed]]
                    Z_next.extend(Z_comb[chosen])
                    W_next.extend(W_comb[chosen])
                    F_next.extend(F_comb[chosen])
                    break

            Z = np.vstack(Z_next)
            W = np.vstack(W_next)
            F = np.vstack(F_next)

        # Final Pareto front
        fronts = fast_non_dominated_sort(F)
        idx = fronts[0]
        return W[idx], F[idx], W, F



# MCDM FUNCTIONS

def entropy_weights(R):
    A, J = R.shape
    P = R / (R.sum(axis=0) + 1e-12)
    P = np.clip(P, 1e-12, 1)
    E = -(1/np.log(A)) * np.sum(P * np.log(P), axis=0)
    d = 1 - E
    return d / d.sum()


def critic_weights(R):
    norm = R / (np.linalg.norm(R, axis=0) + 1e-12)
    sigma = norm.std(axis=0)
    corr = np.corrcoef(norm, rowvar=False)
    corr = np.nan_to_num(corr)
    info = sigma * np.sum(1 - np.abs(corr), axis=0)
    return info / info.sum()


def topsis_scores(R, w):
    norm = R / (np.linalg.norm(R, axis=0) + 1e-12)
    Vw = norm * w
    ideal = Vw.max(axis=0)
    anti = Vw.min(axis=0)
    Dp = np.linalg.norm(Vw - ideal, axis=1)
    Dm = np.linalg.norm(Vw - anti, axis=1)
    return Dm / (Dp + Dm + 1e-12)


def vikor_scores(R, w):
    fstar = R.max(axis=0)
    fmin = R.min(axis=0)
    D = (fstar - R) / (fstar - fmin + 1e-12)
    S = np.sum(w * D, axis=1)
    Rj = np.max(w * D, axis=1)
    Smin, Smax = S.min(), S.max()
    Rmin, Rmax = Rj.min(), Rj.max()
    Q = 0.5*(S - Smin)/(Smax - Smin + 1e-12) + 0.5*(Rj - Rmin)/(Rmax - Rmin + 1e-12)
    return Q


def aras_scores(R, w):
    x0 = R.max(axis=0)
    X = np.vstack([x0, R])
    Z = X / (X.sum(axis=0) + 1e-12)
    S = np.sum(Z * w, axis=1)
    return S[1:] / (S[0] + 1e-12)


def ranks_from_scores(scores, descending=True):
    order = np.argsort(-scores if descending else scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores)+1)
    return ranks



# VISUALISATIONS


def plot_rank_stability(rank_df, out):
    materials = rank_df.index
    methods = rank_df.columns

    plt.figure(figsize=(11, 6))

    for m in methods:
        y = rank_df[m].values.astype(float)  
        y += np.random.normal(0, 0.05, size=len(y))  
        plt.plot(materials, y, marker="o", label=m)

    plt.gca().invert_yaxis()
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Rank (1 = best)")
    plt.title("Rank Stability Across 9 Methods")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(os.path.join(out, "rank_stability.png"), dpi=600)
    plt.savefig(os.path.join(out, "rank_stability.pdf"))
    plt.close()


def plot_weight_distributions(w_ent, w_crit, labels, out):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    titles = [
        "Entropy–TOPSIS", "CRITIC–TOPSIS",
        "Entropy–VIKOR",  "CRITIC–VIKOR",
        "Entropy–ARAS",   "CRITIC–ARAS"
    ]
    Ws = [w_ent, w_crit, w_ent, w_crit, w_ent, w_crit]

    for ax, title, w in zip(axes, titles, Ws):
        ax.bar(labels, w, color="steelblue", edgecolor="black")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "weights_six_methods.png"), dpi=600)
    plt.savefig(os.path.join(out, "weights_six_methods.pdf"))
    plt.close()


def plot_projection_on_pareto(W_pareto, F_pareto, R, w_ent, w_crit, out):
    """
    Projection of all six MCDM methods onto the NSGA-II Pareto surface.
    Ensures:
        - six distinct markers,
        - proper legend placement,
        - same f1/f2 values as the 3D Pareto plot,
        - no legend overlap with colorbar.
    """

    # Pareto values 
    f1 = F_pareto[:, 0]
    f2 = F_pareto[:, 1]
    f3 = -F_pareto[:, 2]

   
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(f1, f2, c=f3, cmap="viridis", s=40)
    cbar = plt.colorbar(sc)
    cbar.set_label("Imbalance")

   
    method_weights = {
        "Entropy–TOPSIS": w_ent,
        "CRITIC–TOPSIS":  w_crit,
        "Entropy–VIKOR":  w_ent,
        "CRITIC–VIKOR":   w_crit,
        "Entropy–ARAS":   w_ent,
        "CRITIC–ARAS":    w_crit
    }

 
    markers = ["D", "s", "P", "X", "h", "o"]
    colors = ["red", "blue", "green", "black", "orange", "purple"]

    for (name, w), marker, col in zip(method_weights.items(), markers, colors):
        S = R @ w
        proj_f1 = S.max()
        proj_f2 = S.min()

        plt.scatter(
            proj_f1, proj_f2,
            s=180,
            marker=marker,
            color=col,
            edgecolor="black",
            linewidth=1.0,
            label=name
        )

        
        plt.text(proj_f1 + 0.002, proj_f2 + 0.002, name, fontsize=8)

    plt.xlabel("f1 = max S (Quality)")
    plt.ylabel("f2 = min S (Robustness)")
    plt.title("Projection of MCDM Weight Vectors onto the NSGA-II Pareto Surface")

    
    plt.legend(bbox_to_anchor=(1.30, 1.0), loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "mcdm_projection_pareto_FIXED.png"), dpi=600)
    plt.savefig(os.path.join(out, "mcdm_projection_pareto_FIXED.pdf"))
    plt.close()

def plot_pareto_3d(F_pareto, out):
    f1 = F_pareto[:, 0]
    f2 = F_pareto[:, 1]
    f3 = -F_pareto[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(f1, f2, f3, c=f3, cmap="viridis", s=40)

    ax.set_xlabel("f1 = max S (quality)")
    ax.set_ylabel("f2 = min S (robustness)")
    ax.set_zlabel("imbalance = -f3")

    cbar = plt.colorbar(sc)
    cbar.set_label("imbalance")

    plt.tight_layout()
    fig.savefig(os.path.join(out, "pareto_3D_fixed.png"), dpi=600)
    fig.savefig(os.path.join(out, "pareto_3D_fixed.pdf"))
    plt.close()


def plot_comparison_table(rank_df, out):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rank_df.values,
        rowLabels=rank_df.index,
        colLabels=rank_df.columns,
        loc="center",
        cellLoc="center"
    )
    tbl.scale(1, 1.4)
    plt.title("Side-by-Side Comparison: NSGA-II vs MCDM Rankings")
    plt.savefig(os.path.join(out, "comparison_table.png"), dpi=600)
    plt.savefig(os.path.join(out, "comparison_table.pdf"))
    plt.close()

def plot_pareto_3d(F_pareto, out):

    from mpl_toolkits.mplot3d import Axes3D  

    f1 = F_pareto[:, 0]          # max S (quality)
    f2 = F_pareto[:, 1]          # min S (robustness)
    f3 = -F_pareto[:, 2]         # imbalance >=0

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    p = ax.scatter(
        f1, f2, f3,
        c=f3, cmap="viridis",
        s=35, alpha=0.9
    )

    ax.set_xlabel("f1 = max S (quality)")
    ax.set_ylabel("f2 = min S (robustness)")
    ax.set_zlabel("imbalance = −f3 (lower = better)")
    ax.set_title("Pareto Front — 3D (colored by imbalance)")

    fig.colorbar(p, ax=ax, shrink=0.6, label="Imbalance")

    plt.tight_layout()
    plt.savefig(os.path.join(out, "pareto_3d.png"), dpi=600)
    plt.savefig(os.path.join(out, "pareto_3d.pdf"))
    plt.close()

def plot_pareto_3d_with_highlights(F_pareto, W_pareto, R, w_ent, w_crit, out):

    from mpl_toolkits.mplot3d import Axes3D

    # Extract objectives
    f1 = F_pareto[:, 0]
    f2 = F_pareto[:, 1]
    f3 = -F_pareto[:, 2]   # imbalance >= 0

    # Identify NSGA-II special solutions
    idx_knee     = np.argmax(f1 + f2)
    idx_balanced = np.argmin(f3)
    idx_robust   = np.argmax(f2)

    pts_nsga = {
        "NSGA-knee":     (f1[idx_knee], f2[idx_knee], f3[idx_knee]),
        "NSGA-balanced": (f1[idx_balanced], f2[idx_balanced], f3[idx_balanced]),
        "NSGA-robust":   (f1[idx_robust], f2[idx_robust], f3[idx_robust]),
    }

    
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Pareto cloud
    p = ax.scatter(f1, f2, f3, c=f3, cmap="viridis", s=35, alpha=0.9)

    cbar = fig.colorbar(p, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label("Imbalance = −f3 (lower = better)", rotation=270, labelpad=20)

    # NSGA-II highlighted solutions
    ax.scatter(*pts_nsga["NSGA-knee"],     s=150, c="red",   marker="^", edgecolor="black", label="NSGA–Knee")
    ax.scatter(*pts_nsga["NSGA-balanced"], s=150, c="yellow",  marker="o", edgecolor="black", label="NSGA–Balanced")
    ax.scatter(*pts_nsga["NSGA-robust"],   s=150, c="green", marker="s", edgecolor="black", label="NSGA–Robust")

    # GROUPED MCDM METHODS
    mcdm_groups = {
        "Entropy-based MCDM (TOPSIS, VIKOR, ARAS)": {
            "weights": w_ent,
            "marker": "D",
            "color": "#1f77b4"
        },
        "CRITIC-based MCDM (TOPSIS, VIKOR, ARAS)": {
            "weights": w_crit,
            "marker": "s",
            "color": "#ff7f0e"
        }
    }

    for i, (group_name, info) in enumerate(mcdm_groups.items()):
        w = info["weights"]
        S = R @ w
        f1p = float(S.max())
        f2p = float(S.min())
        f3p = float(np.sum((w - 1/len(w))**2))

        
        jitter = (i + 1) * 0.004

        ax.scatter(
            f1p + jitter,
            f2p + jitter,
            f3p + jitter,
            s=180,
            marker=info["marker"],
            color=info["color"],
            edgecolor="black",
            linewidth=1.1,
            label=group_name
        )

    
    ax.set_xlabel("f1 = max S (quality)")
    ax.set_ylabel("f2 = min S (robustness)")
    ax.set_title("Pareto Front — 3D with NSGA-II Highlights + MCDM Projections")

    
    ax.legend(
        bbox_to_anchor=(1.42, 0.92),   
        loc="upper left",
        frameon=True,
        title="Highlighted Solutions"
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out, "pareto_3d_highlighted.png"), dpi=600)
    plt.savefig(os.path.join(out, "pareto_3d_highlighted.pdf"))
    plt.close()

def plot_weight_heatmap_entropy_critic_nsga(w_ent, w_crit, F_pareto, W_pareto, criteria_labels, out_dir):

    # Extract NSGA-II representative solutions
    f1 = F_pareto[:, 0]
    f2 = F_pareto[:, 1]
    f3 = -F_pareto[:, 2]

    idx_knee     = np.argmax(f1 + f2)
    idx_balanced = np.argmin(f3)
    idx_robust   = np.argmax(f2)

    W_knee     = W_pareto[idx_knee]
    W_balanced = W_pareto[idx_balanced]
    W_robust   = W_pareto[idx_robust]

    # Construct heatmap matrix
    heatmap_matrix = np.vstack([
        w_ent,
        w_crit,
        W_knee,
        W_balanced,
        W_robust
    ])

    row_labels = [
        "Entropy",
        "CRITIC",
        "NSGA-Knee",
        "NSGA-Balanced",
        "NSGA-Robust"
    ]

    # Plot heatmap
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        cmap="viridis",
        xticklabels=criteria_labels,
        yticklabels=row_labels,
        fmt=".2f",
        cbar_kws={"label": "Weight value"}
    )

    plt.title("Comparison of Entropy, CRITIC, and NSGA-II Representative Weight Vectors")
    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(out_dir, "weights_heatmap_entropy_critic_nsga.png"), dpi=600)
    plt.savefig(os.path.join(out_dir, "weights_heatmap_entropy_critic_nsga.pdf"))
    plt.close()


# MAIN PIPELINE


if __name__ == "__main__":

    print("\n=== LOADING DATA ===")
    df = pd.read_csv(INPUT_FILE, sep=";", decimal=",")
    alt_names = df.iloc[:,0].tolist()
    X = df.iloc[:,1:].values

    is_benefit = [False, False, True, True, True, True, True, False, False]

    R = normalize_benefit_cost(X, is_benefit)

    print("Running NSGA-II…")
    nsga = NSGA2Weights(R)
    W_pareto, F_pareto, W_all, F_all = nsga.run()

    print("Computing MCDM…")
    w_ent = entropy_weights(R)
    w_crit = critic_weights(R)

    ranks = {
        "NSGA_knee": None,
        "NSGA_balanced": None,
        "NSGA_robust": None,
    }

    # NSGA reference rankings
    S_knee = R @ W_pareto[np.argmax(F_pareto[:,0])]
    S_bal  = R @ W_pareto[np.argmin(-F_pareto[:,2])]
    S_rob  = R @ W_pareto[np.argmax(F_pareto[:,1])]

    ranks["NSGA_knee"] = ranks_from_scores(S_knee)
    ranks["NSGA_balanced"] = ranks_from_scores(S_bal)
    ranks["NSGA_robust"] = ranks_from_scores(S_rob)

    
    print("Loading user MCDM ranking table...")

    ranking_file = r"C:\Users\lb298\Documents\NSGA II article\MCDM_ranking.csv"

    # DEBUG 1 — Check file exists
    print("Checking file exists:", os.path.exists(ranking_file))

    # DEBUG 2 — Attempt reading, print shape
    df_mcdm = pd.read_csv(ranking_file, sep=";", decimal=",", encoding="utf-8")
    print("Loaded MCDM CSV shape:", df_mcdm.shape)
    print("Columns:", df_mcdm.columns.tolist())

    # DEBUG 3 — Identify first column
    first_col = df_mcdm.columns[0]
    print("First column detected as:", first_col)

    df_mcdm = df_mcdm.set_index(first_col)
    print("Index set. New index example:", df_mcdm.index[:3].tolist())

    
    
    df_mcdm.index = (
        df_mcdm.index.str.strip()
                      .str.lower()
                      .str.replace(" ", "")
                      .str.replace("-", "")
                      .str.replace("_", "")
                      .str.replace("/", "")
    )
    print("Normalized index:", df_mcdm.index.tolist())

    alt_norm = [
        a.strip().lower()
         .replace(" ", "")
         .replace("-", "")
         .replace("_", "")
         .replace("/", "")
        for a in alt_names
    ]
    print("Normalized alt_names:", alt_norm)

    matches = [name in df_mcdm.index for name in alt_norm]
    print("Match mask:", matches)

    df_mcdm = df_mcdm.reindex(alt_norm)
    print("Reindexed MCDM shape:", df_mcdm.shape)
    print("Row sample after reindex:", df_mcdm.head())

    # Restore original names
    df_mcdm.index = alt_names
    print("Index restored to original materials:", df_mcdm.index.tolist())


    # Insert NSGA-II rankings
    
    print("Inserting NSGA-II reference rankings...")

    df_mcdm.insert(0, "NSGA_knee",     ranks_from_scores(S_knee))
    df_mcdm.insert(1, "NSGA_balanced", ranks_from_scores(S_bal))
    df_mcdm.insert(2, "NSGA_robust",   ranks_from_scores(S_rob))

    print("NSGA columns inserted successfully.")

    # Export final ranking table
    rank_df = df_mcdm.copy()
    rank_df.to_csv(os.path.join(OUTPUT_DIR, "final_rankings.csv"))

print("→ Starting plot_rank_stability...")
plot_rank_stability(rank_df, OUTPUT_DIR)
print("✓ rank_stability OK")

print("→ Starting plot_weight_distributions...")
plot_weight_distributions(
    w_ent, w_crit,
    ["EF", "Density", "Tensile Mod.", "Tensile Str.", "Elongation",
     "Flex Mod.", "Flex Str.", "CTE", "Cost"],
    OUTPUT_DIR
)
print("✓ weights_distributions OK")

print("→ Starting plot_projection_on_pareto...")
plot_projection_on_pareto(W_pareto, F_pareto, R, w_ent, w_crit, OUTPUT_DIR)
print("✓ projection_pareto OK")

print("→ Starting plot_comparison_table...")
plot_comparison_table(rank_df, OUTPUT_DIR)
print("✓ comparison_table OK")

print("\n=== ALL DONE ===")
print(f"All plots saved in: {OUTPUT_DIR}")
plot_pareto_3d(F_pareto, OUTPUT_DIR)
plot_pareto_3d_with_highlights(F_pareto, W_pareto, R, w_ent, w_crit, OUTPUT_DIR)
plot_weight_heatmap_entropy_critic_nsga(
    w_ent,
    w_crit,
    F_pareto,
    W_pareto,
    ["EF", "Density", "Tensile Mod.", "Tensile Str.",
     "Elongation", "Flex Mod.", "Flex Str.", "CTE", "Cost"],
    OUTPUT_DIR
)

