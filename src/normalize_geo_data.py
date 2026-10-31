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
    ROOT / "data/processed/geo/PDAML1_EOC2/qc/"
    "PDAML1_EOC2_filtered.h5ad"
)

OUTPUT_DIR = ROOT / "data/processed/geo/PDAML1_EOC2/normalized"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "PDAML1_EOC2_normalized.h5ad"

TARGET_SUM = 10000
N_TOP_GENES = 2000

sc.settings.verbosity = 2


# ============================================================
# 2. LOAD FILTERED DATASET
# ============================================================

print("\nLoading filtered AnnData dataset...")

adata = sc.read_h5ad(INPUT_FILE)

print(adata)
print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)


# ============================================================
# 3. PRESERVE ORIGINAL RAW COUNTS
# ============================================================

print("\nPreserving original raw counts...")

adata.layers["counts"] = adata.X.copy()

print("Original counts stored in adata.layers['counts']")


# ============================================================
# 4. NORMALIZATION
# ============================================================

print("\nNormalizing counts to 10,000 per cell...")

sc.pp.normalize_total(
    adata,
    target_sum=TARGET_SUM
)

print("Normalization completed.")


# ============================================================
# 5. LOG TRANSFORMATION
# ============================================================

print("\nApplying log1p transformation...")

sc.pp.log1p(adata)

print("Log transformation completed.")


# ============================================================
# 6. HIGHLY VARIABLE GENE SELECTION
# ============================================================

print("\nIdentifying highly variable genes...")

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat",
    n_top_genes=N_TOP_GENES
)

hvg_count = int(adata.var["highly_variable"].sum())

print("Highly variable genes identified:", hvg_count)


# ============================================================
# 7. SAVE HIGHLY VARIABLE GENE METADATA
# ============================================================

hvg_df = adata.var[
    adata.var["highly_variable"]
].copy()

hvg_df.to_csv(
    OUTPUT_DIR / "highly_variable_genes.csv"
)

print("HVG metadata saved.")


# ============================================================
# 8. GENERATE HIGHLY VARIABLE GENE PLOT
# ============================================================

print("\nGenerating highly variable gene plot...")

sc.pl.highly_variable_genes(
    adata,
    show=False
)

plt.savefig(
    PLOT_DIR / "highly_variable_genes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. EXPRESSION DISTRIBUTION PLOT
# ============================================================

print("\nGenerating expression distribution plot...")

# Use a manageable sample of genes for visualization.
sample_genes = adata.var_names[:min(20, adata.n_vars)]

sc.pl.violin(
    adata,
    keys=list(sample_genes),
    jitter=0.2,
    show=False
)

plt.savefig(
    PLOT_DIR / "expression_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. SAVE NORMALIZED ANN DATA
# ============================================================

print("\nSaving normalized AnnData...")

adata.write_h5ad(OUTPUT_FILE)

print("Saved:", OUTPUT_FILE)


# ============================================================
# 11. GENERATE SUMMARY STATISTICS
# ============================================================

summary = {
    "dataset": "PDAML1_EOC2",
    "input_file": str(INPUT_FILE),
    "output_file": str(OUTPUT_FILE),
    "cells": int(adata.n_obs),
    "genes": int(adata.n_vars),
    "normalization_target_sum": TARGET_SUM,
    "log_transformation": "log1p",
    "hvg_method": "seurat",
    "requested_hvg_count": N_TOP_GENES,
    "identified_hvg_count": hvg_count,
    "raw_counts_preserved": True,
    "raw_counts_layer": "counts"
}

with open(
    OUTPUT_DIR / "normalization_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=4)


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("GEO NORMALIZATION COMPLETED")
print("========================================")

print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)
print("Highly variable genes:", hvg_count)

print("\nOutput files:")
print(OUTPUT_FILE)
print(OUTPUT_DIR / "highly_variable_genes.csv")
print(OUTPUT_DIR / "normalization_summary.json")
print(PLOT_DIR / "highly_variable_genes.png")
print(PLOT_DIR / "expression_distribution.png")

print("\n========================================")