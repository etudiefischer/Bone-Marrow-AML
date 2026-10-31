
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data/processed/geo/PDAML1_EOC2/PDAML1_EOC2_raw.h5ad"
)

OUTPUT_DIR = (
    ROOT / "data/processed/geo/PDAML1_EOC2/qc"
)

PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Initial QC thresholds.
# These are configurable and must be reviewed against the
# distributions before being treated as final thresholds.

MIN_GENES = 200
MAX_GENES = 6000
MIN_COUNTS = 500
MAX_COUNTS = 50000
MAX_MT_PERCENT = 20

print("\n========== GEO QUALITY CONTROL ==========")


# ============================================================
# 2. LOAD ANNData
# ============================================================

print("\nLoading AnnData...")

adata = sc.read_h5ad(INPUT_FILE)

print(adata)

# Work on a copy so original extracted data stays unchanged
adata = adata.copy()


# ============================================================
# 3. IDENTIFY DUPLICATE GENE SYMBOLS
# ============================================================

print("\n========== GENE METADATA INSPECTION ==========")

gene_symbols = adata.var["gene_symbol"].astype(str)

duplicate_mask = gene_symbols.duplicated(keep=False)

duplicate_genes = (
    adata.var.loc[
        duplicate_mask,
        ["gene_id", "gene_symbol", "feature_type"]
    ]
    .sort_values("gene_symbol")
)

duplicate_genes.to_csv(
    OUTPUT_DIR / "duplicate_gene_symbols.csv"
)

print("Total genes:", adata.n_vars)
print("Duplicate gene-symbol entries:", len(duplicate_genes))
print(
    "Unique duplicated symbols:",
    duplicate_genes["gene_symbol"].nunique()
)


# ============================================================
# 4. IDENTIFY MITOCHONDRIAL GENES
# ============================================================

print("\n========== MITOCHONDRIAL GENE DETECTION ==========")

# Use gene symbols rather than unique AnnData variable names.
symbols = (
    adata.var["gene_symbol"]
    .astype(str)
    .str.upper()
)

# Human mitochondrial genes commonly begin with MT-
mt_mask = symbols.str.startswith("MT-")

adata.var["mt"] = mt_mask.values

print("Mitochondrial genes detected:", int(mt_mask.sum()))

if mt_mask.sum() > 0:

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    print(
        "Median mitochondrial percentage:",
        round(float(adata.obs["pct_counts_mt"].median()), 2)
    )

else:

    adata.obs["pct_counts_mt"] = 0.0

    print(
        "WARNING: No MT- genes detected. "
        "Verify gene naming and organism before interpreting "
        "mitochondrial QC."
    )


# ============================================================
# 5. BASIC QC STATISTICS
# ============================================================

print("\n========== QC STATISTICS ==========")

qc_columns = [
    "total_counts",
    "n_genes_by_counts",
    "pct_counts_mt"
]

qc_summary = adata.obs[qc_columns].describe().T

print(qc_summary)

qc_summary.to_csv(
    OUTPUT_DIR / "qc_statistics.csv"
)


# ============================================================
# 6. GENERATE QC DISTRIBUTION PLOTS
# ============================================================

print("\nGenerating QC plots...")

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# Plot 1: Number of genes per cell
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    adata.obs["n_genes_by_counts"],
    bins=50,
    kde=True
)

plt.axvline(
    MIN_GENES,
    linestyle="--",
    label=f"Minimum: {MIN_GENES}"
)

plt.axvline(
    MAX_GENES,
    linestyle="--",
    label=f"Maximum: {MAX_GENES}"
)

plt.title("Number of Detected Genes per Cell")
plt.xlabel("Number of genes")
plt.ylabel("Number of cells")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "genes_per_cell.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Plot 2: Total counts per cell
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    adata.obs["total_counts"],
    bins=50,
    kde=True
)

plt.axvline(
    MIN_COUNTS,
    linestyle="--",
    label=f"Minimum: {MIN_COUNTS}"
)

plt.axvline(
    MAX_COUNTS,
    linestyle="--",
    label=f"Maximum: {MAX_COUNTS}"
)

plt.title("Total Expression Counts per Cell")
plt.xlabel("Total counts")
plt.ylabel("Number of cells")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "counts_per_cell.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Plot 3: Mitochondrial percentage
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    adata.obs["pct_counts_mt"],
    bins=40,
    kde=True
)

plt.axvline(
    MAX_MT_PERCENT,
    linestyle="--",
    label=f"Maximum: {MAX_MT_PERCENT}%"
)

plt.title("Mitochondrial Gene Percentage")
plt.xlabel("Mitochondrial counts (%)")
plt.ylabel("Number of cells")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "mitochondrial_percentage.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Plot 4: QC violin plots
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, column, title in zip(
    axes,
    qc_columns,
    [
        "Total Counts",
        "Detected Genes",
        "Mitochondrial Percentage"
    ]
):

    sns.violinplot(
        y=adata.obs[column],
        ax=ax,
        inner="quartile"
    )

    ax.set_title(title)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "qc_violin_plots.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. APPLY CONFIGURABLE QC FILTERS
# ============================================================

print("\n========== APPLYING QC FILTERS ==========")

before_filtering = adata.n_obs

# Individual filtering flags
adata.obs["pass_min_genes"] = (
    adata.obs["n_genes_by_counts"] >= MIN_GENES
)

adata.obs["pass_max_genes"] = (
    adata.obs["n_genes_by_counts"] <= MAX_GENES
)

adata.obs["pass_min_counts"] = (
    adata.obs["total_counts"] >= MIN_COUNTS
)

adata.obs["pass_max_counts"] = (
    adata.obs["total_counts"] <= MAX_COUNTS
)

adata.obs["pass_mt_filter"] = (
    adata.obs["pct_counts_mt"] <= MAX_MT_PERCENT
)

adata.obs["pass_qc"] = (
    adata.obs["pass_min_genes"]
    & adata.obs["pass_max_genes"]
    & adata.obs["pass_min_counts"]
    & adata.obs["pass_max_counts"]
    & adata.obs["pass_mt_filter"]
)

# Save cell-level QC decisions
adata.obs.to_csv(
    OUTPUT_DIR / "cell_qc_decisions.csv"
)

# Filter only after recording all QC decisions
filtered = adata[
    adata.obs["pass_qc"].values
].copy()

after_filtering = filtered.n_obs

removed_cells = before_filtering - after_filtering

print("Cells before filtering:", before_filtering)
print("Cells retained:", after_filtering)
print("Cells removed:", removed_cells)

print(
    "Retention percentage:",
    round(after_filtering / before_filtering * 100, 2)
)


# ============================================================
# 8. SAVE FILTERED ANNData
# ============================================================

print("\nSaving filtered dataset...")

filtered.write_h5ad(
    OUTPUT_DIR / "PDAML1_EOC2_filtered.h5ad",
    compression="gzip"
)


# ============================================================
# 9. SAVE QC SUMMARY
# ============================================================

summary = {

    "sample_id": "PDAML1_EOC2",

    "cells_before_filtering": int(before_filtering),

    "cells_after_filtering": int(after_filtering),

    "cells_removed": int(removed_cells),

    "retention_percentage": round(
        after_filtering / before_filtering * 100,
        2
    ),

    "genes": int(adata.n_vars),

    "thresholds": {
        "min_genes": MIN_GENES,
        "max_genes": MAX_GENES,
        "min_counts": MIN_COUNTS,
        "max_counts": MAX_COUNTS,
        "max_mitochondrial_percentage": MAX_MT_PERCENT
    },

    "median_genes_per_cell": float(
        adata.obs["n_genes_by_counts"].median()
    ),

    "median_counts_per_cell": float(
        adata.obs["total_counts"].median()
    ),

    "median_mitochondrial_percentage": float(
        adata.obs["pct_counts_mt"].median()
    )

}

with open(
    OUTPUT_DIR / "qc_summary.json",
    "w"
) as file:

    json.dump(summary, file, indent=4)


# ============================================================
# 10. FINAL REPORT
# ============================================================

print("\n==============================================")
print("GEO QUALITY CONTROL COMPLETED")
print("==============================================")

print("Original cells:", before_filtering)
print("Filtered cells:", after_filtering)
print("Removed cells:", removed_cells)

print("\nGenerated outputs:")

print(OUTPUT_DIR / "PDAML1_EOC2_filtered.h5ad")
print(OUTPUT_DIR / "duplicate_gene_symbols.csv")
print(OUTPUT_DIR / "qc_statistics.csv")
print(OUTPUT_DIR / "cell_qc_decisions.csv")
print(OUTPUT_DIR / "qc_summary.json")

print("\nPlots saved in:", PLOT_DIR)

print("\n==============================================")
