
from pathlib import Path
import json
import re

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/geo/PDAML1_EOC2/"
    "PDAML1_EOC2_raw.h5ad"
)

OUTPUT_DIR = (
    ROOT / "data/processed/geo/PDAML1_EOC2/"
    "multimodal_analysis"
)

PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# RNA QC thresholds: provisional, to be reviewed.
MIN_GENES = 200
MAX_GENES = 6000
MIN_COUNTS = 500
MAX_COUNTS = 50000
MAX_MT_PERCENT = 20

# RNA processing.
TARGET_SUM = 10000
N_TOP_GENES = 2000
N_PCS = 50
N_NEIGHBORS = 15
N_PCS_NEIGHBORS = 30

# Leiden clustering.
RESOLUTIONS = [0.3, 0.5, 0.8]
PRIMARY_RESOLUTION = 0.5

sc.settings.verbosity = 2


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def to_dense(matrix):
    """Convert sparse or dense matrix to NumPy array."""
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def safe_name(value):
    """Create filesystem-safe names."""
    return str(value).replace(".", "_")


# ============================================================
# 3. LOAD ORIGINAL RAW DATA
# ============================================================

print("\n==============================================")
print("LOADING ORIGINAL GEO DATA")
print("==============================================")

adata = sc.read_h5ad(INPUT_FILE)

print(adata)
print("Total cells:", adata.n_obs)
print("Total features:", adata.n_vars)

if "feature_type" not in adata.var.columns:
    raise ValueError("feature_type metadata is missing.")

feature_types = adata.var["feature_type"].astype("string")

rna_mask = (
    feature_types == "Gene Expression"
).fillna(False).to_numpy()

protein_mask = (
    feature_types == "Antibody Capture"
).fillna(False).to_numpy()

if not rna_mask.any():
    raise ValueError("No Gene Expression features found.")

if not protein_mask.any():
    raise ValueError("No Antibody Capture features found.")

print("\nRNA features:", int(rna_mask.sum()))
print("Antibody features:", int(protein_mask.sum()))


# ============================================================
# 4. EXTRACT RNA COUNTS
# ============================================================

print("\n==============================================")
print("RNA-ONLY PROCESSING")
print("==============================================")

rna = adata[:, rna_mask].copy()

# Preserve original counts.
rna.layers["counts"] = rna.X.copy()

# Identify mitochondrial genes from gene symbols.
symbols = (
    rna.var["gene_symbol"]
    .astype("string")
    .fillna("")
    .str.upper()
)

rna.var["mt"] = symbols.str.startswith("MT-").to_numpy()

print("RNA cells:", rna.n_obs)
print("RNA genes:", rna.n_vars)
print("Mitochondrial features:", int(rna.var["mt"].sum()))


# ============================================================
# 5. RNA-SPECIFIC QUALITY CONTROL
# ============================================================

print("\nCalculating RNA QC metrics...")

sc.pp.calculate_qc_metrics(
    rna,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True
)

qc_mask = (
    (rna.obs["n_genes_by_counts"] >= MIN_GENES)
    & (rna.obs["n_genes_by_counts"] <= MAX_GENES)
    & (rna.obs["total_counts"] >= MIN_COUNTS)
    & (rna.obs["total_counts"] <= MAX_COUNTS)
    & (rna.obs["pct_counts_mt"] <= MAX_MT_PERCENT)
)

rna.obs["pass_rna_qc"] = qc_mask

qc_decisions = rna.obs[
    [
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
        "pass_rna_qc"
    ]
].copy()

qc_decisions.to_csv(
    OUTPUT_DIR / "rna_qc_decisions.csv"
)

print("\nRNA QC results:")
print("Original cells:", rna.n_obs)
print("Retained cells:", int(qc_mask.sum()))
print("Removed cells:", int((~qc_mask).sum()))
print(
    "Retention:",
    round(float(qc_mask.mean() * 100), 2),
    "%"
)

# Apply RNA-specific filtering.
rna = rna[qc_mask].copy()

if rna.n_obs < 3:
    raise ValueError("Too few cells remain after RNA QC.")

print("Filtered RNA dataset:", rna)


# ============================================================
# 6. RNA NORMALIZATION AND LOG TRANSFORMATION
# ============================================================

print("\nNormalizing RNA counts...")

sc.pp.normalize_total(
    rna,
    target_sum=TARGET_SUM
)

sc.pp.log1p(rna)

# Preserve normalized full-gene expression for later
# marker analysis.
rna.raw = rna.copy()

print("RNA normalization completed.")


# ============================================================
# 7. HIGHLY VARIABLE GENES
# ============================================================

print("\nIdentifying RNA highly variable genes...")

sc.pp.highly_variable_genes(
    rna,
    flavor="seurat",
    n_top_genes=N_TOP_GENES
)

hvg_count = int(rna.var["highly_variable"].sum())

rna.var[
    rna.var["highly_variable"]
].reset_index().to_csv(
    OUTPUT_DIR / "rna_highly_variable_genes.csv",
    index=False
)

print("HVGs identified:", hvg_count)

if hvg_count < 2:
    raise ValueError("Insufficient HVGs for PCA.")


# ============================================================
# 8. SCALE AND RUN RNA PCA
# ============================================================

print("\nRunning RNA PCA...")

# Use only HVGs for dimensionality reduction.
work = rna[:, rna.var["highly_variable"]].copy()

sc.pp.scale(
    work,
    zero_center=True,
    max_value=10
)

n_components = min(
    N_PCS,
    work.n_obs - 1,
    work.n_vars - 1
)

sc.tl.pca(
    work,
    n_comps=n_components,
    svd_solver="arpack",
    random_state=42
)

variance_ratio = work.uns["pca"]["variance_ratio"]

variance_df = pd.DataFrame({
    "PC": [
        f"PC{i+1}" for i in range(len(variance_ratio))
    ],
    "explained_variance_ratio": variance_ratio,
    "cumulative_variance": np.cumsum(variance_ratio)
})

variance_df.to_csv(
    OUTPUT_DIR / "rna_pca_explained_variance.csv",
    index=False
)

print("PCA completed.")
print(
    "Variance explained by first 10 PCs:",
    round(float(variance_ratio[:10].sum() * 100), 2),
    "%"
)


# ============================================================
# 9. PCA VISUALIZATION
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(variance_ratio) + 1),
    variance_ratio * 100,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.title("RNA-only PCA Scree Plot")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "rna_pca_scree.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. KNN GRAPH
# ============================================================

print("\nConstructing RNA KNN graph...")

pcs_for_neighbors = min(
    N_PCS_NEIGHBORS,
    n_components
)

sc.pp.neighbors(
    work,
    n_neighbors=N_NEIGHBORS,
    n_pcs=pcs_for_neighbors,
    use_rep="X_pca",
    random_state=42
)

print("KNN graph completed.")


# ============================================================
# 11. UMAP
# ============================================================

print("\nGenerating RNA UMAP...")

sc.tl.umap(
    work,
    min_dist=0.3,
    spread=1.0,
    random_state=42
)

print("UMAP completed.")


# ============================================================
# 12. LEIDEN CLUSTERING
# ============================================================

print("\nRunning RNA-only Leiden clustering...")

cluster_summary = []

for resolution in RESOLUTIONS:

    column = f"leiden_{safe_name(resolution)}"

    sc.tl.leiden(
        work,
        resolution=resolution,
        key_added=column,
        random_state=42,
        flavor="igraph",
        n_iterations=2
    )

    counts = work.obs[column].value_counts()

    print(f"\nResolution {resolution}")
    print("Clusters:", len(counts))
    print(counts.sort_index())

    for cluster, count in counts.items():

        cluster_summary.append({
            "resolution": resolution,
            "cluster": cluster,
            "number_of_cells": int(count),
            "percentage": round(
                count / work.n_obs * 100, 2
            )
        })

    sc.pl.umap(
        work,
        color=column,
        title=f"RNA Leiden Resolution {resolution}",
        legend_loc="on data",
        frameon=False,
        show=False
    )

    plt.savefig(
        PLOT_DIR / f"rna_umap_{column}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


pd.DataFrame(cluster_summary).to_csv(
    OUTPUT_DIR / "rna_cluster_summary.csv",
    index=False
)


# ============================================================
# 13. TRANSFER RNA EMBEDDINGS AND CLUSTERS
#     TO FULL RNA DATASET
# ============================================================

print("\nSaving RNA cluster assignments...")

for resolution in RESOLUTIONS:

    column = f"leiden_{safe_name(resolution)}"

    rna.obs[column] = (
        work.obs[column]
        .reindex(rna.obs_names)
        .astype("category")
    )

rna.obsm["X_pca"] = work.obsm["X_pca"].copy()
rna.obsm["X_umap"] = work.obsm["X_umap"].copy()

rna.uns["pca"] = work.uns["pca"].copy()
rna.uns["neighbors"] = work.uns["neighbors"].copy()
rna.uns["umap"] = work.uns["umap"].copy()

rna.obsp["connectivities"] = (
    work.obsp["connectivities"].copy()
)

rna.obsp["distances"] = (
    work.obsp["distances"].copy()
)

primary_column = (
    f"leiden_{safe_name(PRIMARY_RESOLUTION)}"
)


# ============================================================
# 14. RNA CLUSTER MARKER ANALYSIS
# ============================================================

print("\nFinding RNA cluster marker genes...")

sc.tl.rank_genes_groups(
    rna,
    groupby=primary_column,
    method="wilcoxon",
    use_raw=False,
    key_added="rna_cluster_markers"
)

marker_frames = []

for cluster in rna.obs[primary_column].cat.categories:

    marker_df = sc.get.rank_genes_groups_df(
        rna,
        group=cluster,
        key="rna_cluster_markers"
    )

    marker_df["cluster"] = cluster

    marker_frames.append(marker_df)

all_rna_markers = pd.concat(
    marker_frames,
    ignore_index=True
)

all_rna_markers.to_csv(
    OUTPUT_DIR / "rna_cluster_marker_genes.csv",
    index=False
)

top_rna_markers = (
    all_rna_markers
    .sort_values(
        ["cluster", "pvals_adj", "scores"],
        ascending=[True, True, False]
    )
    .groupby("cluster")
    .head(20)
)

top_rna_markers.to_csv(
    OUTPUT_DIR / "rna_top_20_markers_per_cluster.csv",
    index=False
)

sc.pl.rank_genes_groups_heatmap(
    rna,
    groupby=primary_column,
    key="rna_cluster_markers",
    n_genes=5,
    show_gene_labels=True,
    show=False
)

plt.savefig(
    PLOT_DIR / "rna_cluster_marker_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. SAVE RNA CELL ASSIGNMENTS AND DATASET
# ============================================================

rna_cells = rna.obs.copy()

rna_cells["UMAP1"] = rna.obsm["X_umap"][:, 0]
rna_cells["UMAP2"] = rna.obsm["X_umap"][:, 1]

for i in range(n_components):
    rna_cells[f"PC{i+1}"] = rna.obsm["X_pca"][:, i]

rna_cells.index.name = "cell_barcode"

rna_cells.to_csv(
    OUTPUT_DIR / "rna_cell_cluster_assignments.csv"
)

rna.write_h5ad(
    OUTPUT_DIR / "RNA_only_analysis.h5ad"
)

print("\nRNA-only analysis saved.")


# ============================================================
# 16. EXTRACT MATCHING ANTIBODY COUNTS
# ============================================================

print("\n==============================================")
print("ANTIBODY CAPTURE PROCESSING")
print("==============================================")

# Keep precisely the same cells that passed RNA QC.
protein = adata[
    rna.obs_names,
    protein_mask
].copy()

# Preserve original antibody counts.
protein.layers["counts"] = protein.X.copy()

protein.obs = rna.obs.copy()

print("Antibody cells:", protein.n_obs)
print("Antibody features:", protein.n_vars)


# ============================================================
# 17. ANTIBODY FEATURE METADATA
# ============================================================

protein.var.reset_index().to_csv(
    OUTPUT_DIR / "antibody_feature_metadata.csv",
    index=False
)

antibody_symbols = (
    protein.var["gene_symbol"]
    .astype("string")
    .fillna("")
    .str.strip()
)

print("\nAntibody features:")
print(antibody_symbols.tolist())


# ============================================================
# 18. ANTIBODY CLR NORMALIZATION
# ============================================================

print("\nApplying antibody CLR-style normalization...")

protein_counts = to_dense(
    protein.layers["counts"]
).astype(np.float32)

# Centered log-ratio transformation across antibody
# features within each cell.
log_protein = np.log1p(protein_counts)

protein_clr = (
    log_protein -
    log_protein.mean(axis=1, keepdims=True)
)

protein.X = protein_clr.astype(np.float32)

protein.uns["normalization"] = {
    "method": "CLR-style",
    "formula": "log1p(count) - per-cell mean(log1p(counts))",
    "axis": "antibody features within each cell"
}

protein.write_h5ad(
    OUTPUT_DIR / "Antibody_CLR_analysis.h5ad"
)

print("Antibody normalization completed.")


# ============================================================
# 19. EXPORT ANTIBODY NORMALIZED MATRIX
# ============================================================

protein_df = pd.DataFrame(
    protein.X,
    index=protein.obs_names,
    columns=protein.var_names
)

protein_df.index.name = "cell_barcode"

protein_df.to_csv(
    OUTPUT_DIR / "antibody_clr_expression.csv"
)


# ============================================================
# 20. COMPARE RNA AND ANTIBODY MARKERS
# ============================================================

print("\n==============================================")
print("RNA-PROTEIN MARKER COMPARISON")
print("==============================================")

# Map antibody features to corresponding RNA genes.
# Some antibody names represent a protein complex or
# a protein measured by multiple RNA genes.

protein_to_rna = {
    "CD8": ["CD8A", "CD8B"],
    "HLA-DRDPDQ": [
        "HLA-DRA", "HLA-DRB1",
        "HLA-DPA1", "HLA-DPB1"
    ]
}

# Create RNA gene-symbol lookup.
rna_symbols = (
    rna.var["gene_symbol"]
    .astype("string")
    .fillna("")
    .str.strip()
    .str.upper()
)

rna_symbol_indices = {}

for idx, symbol in enumerate(rna_symbols):

    if symbol:
        rna_symbol_indices.setdefault(
            symbol, []
        ).append(idx)


# Map antibody symbols to RNA symbols.
comparison_map = []

for protein_idx, protein_symbol in enumerate(antibody_symbols):

    name = str(protein_symbol).upper().strip()

    # Remove common TotalSeq suffixes.
    base_name = re.sub(
        r"[_ -]?TOTALSEQ.*$",
        "",
        name
    )

    target_genes = protein_to_rna.get(
        base_name,
        [base_name]
    )

    matching_indices = []

    for gene in target_genes:
        matching_indices.extend(
            rna_symbol_indices.get(gene, [])
        )

    comparison_map.append({
        "protein_index": protein_idx,
        "protein_feature": name,
        "base_marker": base_name,
        "matching_rna_genes": ";".join(target_genes),
        "rna_match_found": len(matching_indices) > 0,
        "rna_indices": matching_indices
    })


comparison_metadata = pd.DataFrame([
    {
        key: value
        for key, value in item.items()
        if key != "rna_indices"
    }
    for item in comparison_map
])

comparison_metadata.to_csv(
    OUTPUT_DIR / "rna_protein_feature_mapping.csv",
    index=False
)


# ============================================================
# 21. CLUSTER-LEVEL RNA AND PROTEIN EXPRESSION
# ============================================================

comparison_rows = []

rna_counts = to_dense(
    rna.layers["counts"]
)

rna_expression = to_dense(
    rna.X
)

protein_counts = to_dense(
    protein.layers["counts"]
)

protein_expression = to_dense(
    protein.X
)

cluster_labels = rna.obs[primary_column].astype(str).to_numpy()

for cluster in sorted(set(cluster_labels)):

    cell_mask = cluster_labels == cluster

    for item in comparison_map:

        p_idx = item["protein_index"]

        p_raw = protein_counts[cell_mask, p_idx]
        p_clr = protein_expression[cell_mask, p_idx]

        rna_indices = item["rna_indices"]

        if rna_indices:

            r_raw = rna_counts[
                cell_mask
            ][:, rna_indices].sum(axis=1)

            r_log = rna_expression[
                cell_mask
            ][:, rna_indices].mean(axis=1)

            rna_mean = float(np.mean(r_log))

            rna_pct = float(
                np.mean(np.asarray(r_raw).ravel() > 0) * 100
            )

        else:

            rna_mean = np.nan
            rna_pct = np.nan

        comparison_rows.append({

            "cluster": cluster,
            "number_of_cells": int(cell_mask.sum()),
            "protein_feature": item["protein_feature"],
            "matching_rna_genes": item["matching_rna_genes"],
            "rna_match_found": item["rna_match_found"],
            "rna_mean_log_expression": rna_mean,
            "rna_percent_expressing": rna_pct,
            "protein_mean_clr": float(np.mean(p_clr)),
            "protein_percent_detected": float(
                np.mean(p_raw > 0) * 100
            )

        })


comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    OUTPUT_DIR / "rna_protein_cluster_comparison.csv",
    index=False
)

print("RNA-protein comparison completed.")


# ============================================================
# 22. ANTIBODY CLUSTER HEATMAP
# ============================================================

print("\nGenerating antibody cluster heatmap...")

cluster_means = []

cluster_order = sorted(set(cluster_labels))

for cluster in cluster_order:

    cell_mask = cluster_labels == cluster

    cluster_means.append(
        protein_expression[cell_mask].mean(axis=0)
    )

cluster_means = np.asarray(cluster_means)

plt.figure(figsize=(14, 7))

plt.imshow(
    cluster_means,
    aspect="auto",
    interpolation="nearest"
)

plt.yticks(
    range(len(cluster_order)),
    cluster_order
)

plt.xticks(
    range(protein.n_vars),
    antibody_symbols.tolist(),
    rotation=90,
    fontsize=7
)

plt.xlabel("Antibody features")
plt.ylabel("RNA Leiden cluster")
plt.title("Antibody CLR Expression by RNA Cluster")

plt.colorbar(label="Mean CLR expression")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "antibody_cluster_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. SAVE FINAL ANALYSIS SUMMARY
# ============================================================

summary = {

    "dataset": "PDAML1_EOC2",

    "input_file": str(INPUT_FILE),

    "original_cells": int(adata.n_obs),

    "rna_features": int(rna_mask.sum()),

    "antibody_features": int(protein_mask.sum()),

    "rna_cells_after_qc": int(rna.n_obs),

    "rna_genes": int(rna.n_vars),

    "highly_variable_genes": int(hvg_count),

    "pca_components": int(n_components),

    "pca_neighbors_components": int(pcs_for_neighbors),

    "pca_first_10_variance_percent": float(
        variance_ratio[:10].sum() * 100
    ),

    "knn_neighbors": N_NEIGHBORS,

    "leiden_resolutions": RESOLUTIONS,

    "primary_resolution": PRIMARY_RESOLUTION,

    "primary_cluster_count": int(
        rna.obs[primary_column].nunique()
    ),

    "antibody_normalization": "CLR-style",

    "rna_protein_comparison": "cluster-level",

    "multimodal_integration_performed": False

}

with open(
    OUTPUT_DIR / "multimodal_analysis_summary.json",
    "w"
) as f:

    json.dump(summary, f, indent=4)


# ============================================================
# 24. FINAL REPORT
# ============================================================

print("\n==============================================")
print("COMPLETE GEO MULTIMODAL WORKFLOW FINISHED")
print("==============================================")

print("\nRNA RESULTS")
print("Cells:", rna.n_obs)
print("Genes:", rna.n_vars)
print("HVGs:", hvg_count)
print("PCA components:", n_components)

for resolution in RESOLUTIONS:

    column = f"leiden_{safe_name(resolution)}"

    print(
        f"Leiden {resolution}:",
        rna.obs[column].nunique(),
        "clusters"
    )

print("\nANTIBODY RESULTS")
print("Cells:", protein.n_obs)
print("Features:", protein.n_vars)

print("\nRNA-PROTEIN COMPARISON")
print("Comparisons:", len(comparison_df))
print(
    "RNA matches:",
    int(comparison_metadata["rna_match_found"].sum()),
    "of",
    len(comparison_metadata)
)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nIMPORTANT OUTPUT FILES:")
print("- RNA_only_analysis.h5ad")
print("- Antibody_CLR_analysis.h5ad")
print("- rna_cell_cluster_assignments.csv")
print("- rna_cluster_summary.csv")
print("- rna_cluster_marker_genes.csv")
print("- rna_top_20_markers_per_cluster.csv")
print("- rna_protein_cluster_comparison.csv")
print("- rna_protein_feature_mapping.csv")
print("- antibody_clr_expression.csv")
print("- antibody_feature_metadata.csv")
print("- multimodal_analysis_summary.json")

print("\nPLOTS:")
print("- plots/rna_pca_scree.png")
print("- plots/rna_umap_leiden_*.png")
print("- plots/rna_cluster_marker_heatmap.png")
print("- plots/antibody_cluster_heatmap.png")

print("\n==============================================")
