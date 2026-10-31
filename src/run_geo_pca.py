
from pathlib import Path
import json

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/geo/PDAML1_EOC2/normalized/"
    "PDAML1_EOC2_normalized.h5ad"
)

OUTPUT_DIR = ROOT / "data/processed/geo/PDAML1_EOC2/pca"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "PDAML1_EOC2_PCA.h5ad"

N_COMPONENTS = 50

sc.settings.verbosity = 2


# ============================================================
# 2. LOAD NORMALIZED DATASET
# ============================================================

print("\nLoading normalized dataset...")

adata = sc.read_h5ad(INPUT_FILE)

print(adata)
print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)


# ============================================================
# 3. SELECT HIGHLY VARIABLE GENES
# ============================================================

print("\nSelecting highly variable genes...")

if "highly_variable" not in adata.var.columns:
    raise ValueError("HVG annotations not found in dataset.")

hvg_mask = adata.var["highly_variable"].to_numpy()
hvg_count = int(hvg_mask.sum())

if hvg_count < 2:
    raise ValueError("Insufficient highly variable genes for PCA.")

adata_pca = adata[:, hvg_mask].copy()

print("Selected genes:", adata_pca.n_vars)


# ============================================================
# 4. SCALE GENE EXPRESSION
# ============================================================

print("\nScaling gene expression...")

sc.pp.scale(
    adata_pca,
    zero_center=True,
    max_value=10
)

print("Scaling completed.")


# ============================================================
# 5. PERFORM PCA
# ============================================================

print("\nRunning PCA...")

n_comps = min(
    N_COMPONENTS,
    adata_pca.n_obs - 1,
    adata_pca.n_vars - 1
)

sc.tl.pca(
    adata_pca,
    n_comps=n_comps,
    svd_solver="arpack",
    random_state=42
)

print("PCA completed.")
print("Components:", n_comps)


# ============================================================
# 6. EXTRACT EXPLAINED VARIANCE
# ============================================================

variance_ratio = adata_pca.uns["pca"]["variance_ratio"]

variance_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(variance_ratio))],
    "explained_variance_ratio": variance_ratio,
    "cumulative_variance": np.cumsum(variance_ratio)
})

variance_df.to_csv(
    OUTPUT_DIR / "pca_explained_variance.csv",
    index=False
)

print("\nExplained variance:")
print(variance_df.head(10).to_string(index=False))

print(
    "\nCumulative variance explained by first 10 PCs:",
    round(float(np.sum(variance_ratio[:10]) * 100), 2),
    "%"
)

print(
    "Cumulative variance explained by all PCs:",
    round(float(np.sum(variance_ratio) * 100), 2),
    "%"
)


# ============================================================
# 7. PCA SCREE PLOT
# ============================================================

print("\nGenerating PCA scree plot...")

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(variance_ratio) + 1),
    variance_ratio * 100,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.title("PCA Scree Plot")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "pca_scree_plot.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. CUMULATIVE VARIANCE PLOT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(variance_ratio) + 1),
    np.cumsum(variance_ratio) * 100,
    marker="o"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Cumulative PCA Explained Variance")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "pca_cumulative_variance.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. PCA SCATTER PLOT: PC1 VS PC2
# ============================================================

print("\nGenerating PCA scatter plot...")

pca_coordinates = adata_pca.obsm["X_pca"]

plt.figure(figsize=(8, 6))

plt.scatter(
    pca_coordinates[:, 0],
    pca_coordinates[:, 1],
    s=8,
    alpha=0.6
)

plt.xlabel(
    f"PC1 ({variance_ratio[0] * 100:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({variance_ratio[1] * 100:.2f}% variance)"
)

plt.title("GEO Dataset: PCA Projection")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "pca_pc1_pc2.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. SAVE PCA CELL COORDINATES
# ============================================================

print("\nSaving PCA coordinates...")

pca_columns = [
    f"PC{i+1}" for i in range(n_comps)
]

pca_df = pd.DataFrame(
    pca_coordinates,
    index=adata_pca.obs_names,
    columns=pca_columns
)

pca_df.index.name = "cell_barcode"

pca_df.to_csv(
    OUTPUT_DIR / "pca_cell_coordinates.csv"
)


# ============================================================
# 11. SAVE PCA LOADINGS
# ============================================================

loadings = adata_pca.varm["PCs"]

loadings_df = pd.DataFrame(
    loadings,
    index=adata_pca.var_names,
    columns=pca_columns
)

loadings_df.index.name = "gene"

loadings_df.to_csv(
    OUTPUT_DIR / "pca_gene_loadings.csv"
)


# ============================================================
# 12. SAVE PCA ANN DATA
# ============================================================

print("\nSaving PCA AnnData object...")

adata_pca.write_h5ad(OUTPUT_FILE)

print("Saved:", OUTPUT_FILE)


# ============================================================
# 13. SAVE SUMMARY
# ============================================================

summary = {
    "dataset": "PDAML1_EOC2",
    "input_file": str(INPUT_FILE),
    "output_file": str(OUTPUT_FILE),
    "cells": int(adata_pca.n_obs),
    "total_genes_in_input": int(adata.n_vars),
    "highly_variable_genes_used": int(adata_pca.n_vars),
    "pca_components": int(n_comps),
    "first_pc_variance_percent": float(variance_ratio[0] * 100),
    "second_pc_variance_percent": float(variance_ratio[1] * 100),
    "first_10_pcs_cumulative_variance_percent": float(
        np.sum(variance_ratio[:min(10, n_comps)]) * 100
    ),
    "all_components_cumulative_variance_percent": float(
        np.sum(variance_ratio) * 100
    ),
    "scaling": "zero-centered, clipped at 10",
    "pca_solver": "arpack"
}

with open(
    OUTPUT_DIR / "pca_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=4)


# ============================================================
# 14. FINAL REPORT
# ============================================================

print("\n========================================")
print("GEO PCA COMPLETED")
print("========================================")

print("Cells:", adata_pca.n_obs)
print("HVGs used:", adata_pca.n_vars)
print("PCA components:", n_comps)

print("\nOutput directory:", OUTPUT_DIR)

print("\nGenerated files:")
print("- PDAML1_EOC2_PCA.h5ad")
print("- pca_cell_coordinates.csv")
print("- pca_gene_loadings.csv")
print("- pca_explained_variance.csv")
print("- pca_summary.json")
print("- plots/pca_scree_plot.png")
print("- plots/pca_cumulative_variance.png")
print("- plots/pca_pc1_pc2.png")

print("\n========================================")
