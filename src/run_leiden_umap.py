
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
    ROOT / "data/processed/geo/PDAML1_EOC2/pca/"
    "PDAML1_EOC2_PCA.h5ad"
)

OUTPUT_DIR = (
    ROOT / "data/processed/geo/PDAML1_EOC2/leiden_umap"
)

PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

N_NEIGHBORS = 15
N_PCS = 30
RESOLUTIONS = [0.3, 0.5, 0.8]

sc.settings.verbosity = 2


# ============================================================
# 2. LOAD PCA DATASET
# ============================================================

print("\nLoading PCA dataset...")

adata = sc.read_h5ad(INPUT_FILE)

print(adata)

print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)


# ============================================================
# 3. VALIDATE PCA REPRESENTATION
# ============================================================

print("\nChecking PCA representation...")

if "X_pca" not in adata.obsm:
    raise ValueError("PCA coordinates not found in AnnData.")

available_pcs = adata.obsm["X_pca"].shape[1]

n_pcs = min(N_PCS, available_pcs)

print("Available PCs:", available_pcs)
print("Using PCs:", n_pcs)


# ============================================================
# 4. CONSTRUCT K-NEAREST-NEIGHBOR GRAPH
# ============================================================

print("\nConstructing KNN graph...")

sc.pp.neighbors(
    adata,
    n_neighbors=N_NEIGHBORS,
    n_pcs=n_pcs,
    use_rep="X_pca",
    random_state=42
)

print("KNN graph constructed.")


# ============================================================
# 5. GENERATE UMAP EMBEDDING
# ============================================================

print("\nGenerating UMAP...")

sc.tl.umap(
    adata,
    min_dist=0.3,
    spread=1.0,
    random_state=42
)

print("UMAP generated.")


# ============================================================
# 6. RUN LEIDEN AT MULTIPLE RESOLUTIONS
# ============================================================

print("\nRunning Leiden clustering...")

cluster_summary = []

for resolution in RESOLUTIONS:

    column_name = f"leiden_{str(resolution).replace('.', '_')}"

    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=column_name,
        random_state=42,
        flavor="igraph",
        n_iterations=2
    )

    cluster_counts = adata.obs[column_name].value_counts()

    number_of_clusters = len(cluster_counts)

    print(f"\nResolution: {resolution}")
    print("Number of clusters:", number_of_clusters)
    print(cluster_counts.sort_index())

    for cluster, count in cluster_counts.items():

        cluster_summary.append({
            "resolution": resolution,
            "cluster": cluster,
            "number_of_cells": int(count),
            "percentage": round(
                count / adata.n_obs * 100, 2
            )
        })


# ============================================================
# 7. SAVE CLUSTER SUMMARY
# ============================================================

summary_df = pd.DataFrame(cluster_summary)

summary_df.to_csv(
    OUTPUT_DIR / "cluster_summary.csv",
    index=False
)


# ============================================================
# 8. GENERATE UMAP PLOTS
# ============================================================

print("\nGenerating UMAP visualizations...")

for resolution in RESOLUTIONS:

    column_name = f"leiden_{str(resolution).replace('.', '_')}"

    sc.pl.umap(
        adata,
        color=column_name,
        title=f"Leiden Clustering (Resolution {resolution})",
        legend_loc="on data",
        frameon=False,
        show=False
    )

    plt.savefig(
        PLOT_DIR / f"umap_{column_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 9. GENERATE PCA AND UMAP COMPARISON
# ============================================================

sc.pl.umap(
    adata,
    color="sample_id",
    title="UMAP - Sample Distribution",
    frameon=False,
    show=False
)

plt.savefig(
    PLOT_DIR / "umap_sample_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. FIND CLUSTER MARKER GENES
# ============================================================

print("\nFinding cluster marker genes...")

# Use the highest Leiden resolution as the initial
# marker-discovery partition.
marker_resolution = RESOLUTIONS[-1]

marker_column = (
    f"leiden_{str(marker_resolution).replace('.', '_')}"
)

# Rank genes using the original log-normalized expression
# values, not the scaled PCA representation.
if "log1p" not in adata.uns:
    raise ValueError("Log-normalization metadata not found.")

sc.tl.rank_genes_groups(
    adata,
    groupby=marker_column,
    method="wilcoxon",
    use_raw=False,
    key_added="cluster_markers"
)

print("Cluster marker analysis completed.")


# ============================================================
# 11. EXPORT MARKER GENES
# ============================================================

marker_frames = []

for cluster in adata.obs[marker_column].cat.categories:

    marker_df = sc.get.rank_genes_groups_df(
        adata,
        group=cluster,
        key="cluster_markers"
    )

    marker_df["cluster"] = cluster

    marker_frames.append(marker_df)

all_markers = pd.concat(
    marker_frames,
    ignore_index=True
)

all_markers.to_csv(
    OUTPUT_DIR / "cluster_marker_genes.csv",
    index=False
)

# Export top 20 genes per cluster.
top_markers = (
    all_markers
    .sort_values(
        ["cluster", "pvals_adj", "scores"],
        ascending=[True, True, False]
    )
    .groupby("cluster")
    .head(20)
)

top_markers.to_csv(
    OUTPUT_DIR / "top_20_markers_per_cluster.csv",
    index=False
)


# ============================================================
# 12. GENERATE MARKER HEATMAP
# ============================================================

print("\nGenerating marker heatmap...")

sc.pl.rank_genes_groups_heatmap(
    adata,
    groupby=marker_column,
    key="cluster_markers",
    n_genes=5,
    show=False,
    swap_axes=True
)

plt.savefig(
    PLOT_DIR / "cluster_marker_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. SAVE CELL-LEVEL CLUSTER ASSIGNMENTS
# ============================================================

print("\nSaving cell cluster assignments...")

cell_results = adata.obs.copy()

# Include UMAP coordinates.
cell_results["UMAP1"] = adata.obsm["X_umap"][:, 0]
cell_results["UMAP2"] = adata.obsm["X_umap"][:, 1]

# Include PCA coordinates for reference.
for i in range(n_pcs):
    cell_results[f"PC{i+1}"] = adata.obsm["X_pca"][:, i]

cell_results.index.name = "cell_barcode"

cell_results.to_csv(
    OUTPUT_DIR / "cell_cluster_assignments.csv"
)


# ============================================================
# 14. SAVE COMPLETE ANN DATA
# ============================================================

print("\nSaving complete Leiden and UMAP dataset...")

OUTPUT_FILE = OUTPUT_DIR / "PDAML1_EOC2_Leiden_UMAP.h5ad"

adata.write_h5ad(OUTPUT_FILE)


# ============================================================
# 15. SAVE ANALYSIS SUMMARY
# ============================================================

summary = {
    "dataset": "PDAML1_EOC2",
    "input_file": str(INPUT_FILE),
    "output_file": str(OUTPUT_FILE),
    "number_of_cells": int(adata.n_obs),
    "number_of_genes": int(adata.n_vars),
    "available_pcs": int(available_pcs),
    "pcs_used": int(n_pcs),
    "knn_neighbors": N_NEIGHBORS,
    "umap_min_dist": 0.3,
    "leiden_resolutions": RESOLUTIONS,
    "marker_analysis_resolution": marker_resolution,
    "number_of_clusters": {
        column: int(adata.obs[column].nunique())
        for column in [
            f"leiden_{str(r).replace('.', '_')}"
            for r in RESOLUTIONS
        ]
    }
}

with open(
    OUTPUT_DIR / "analysis_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=4)


# ============================================================
# 16. FINAL REPORT
# ============================================================

print("\n==========================================")
print("LEIDEN CLUSTERING AND UMAP COMPLETED")
print("==========================================")

print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)
print("PCA components used:", n_pcs)

for resolution in RESOLUTIONS:

    column_name = f"leiden_{str(resolution).replace('.', '_')}"

    print(
        f"Resolution {resolution}: "
        f"{adata.obs[column_name].nunique()} clusters"
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("- PDAML1_EOC2_Leiden_UMAP.h5ad")
print("- cell_cluster_assignments.csv")
print("- cluster_summary.csv")
print("- cluster_marker_genes.csv")
print("- top_20_markers_per_cluster.csv")
print("- analysis_summary.json")
print("- plots/umap_leiden_*.png")
print("- plots/umap_sample_distribution.png")
print("- plots/cluster_marker_heatmap.png")

print("\n==========================================")
