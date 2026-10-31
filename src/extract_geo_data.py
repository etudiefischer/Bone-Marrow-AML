
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from scipy.io import mmread
from scipy.sparse import csr_matrix


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GEO_DIR = ROOT / "data/raw/geo/PDAML1_EOC2"

OUTPUT_DIR = ROOT / "data/processed/geo/PDAML1_EOC2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_FILE = GEO_DIR / "matrix.mtx.gz"
FEATURES_FILE = GEO_DIR / "features.tsv.gz"
BARCODES_FILE = GEO_DIR / "barcodes.tsv.gz"

for path in [MATRIX_FILE, FEATURES_FILE, BARCODES_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GEO file: {path}")


# ============================================================
# 2. LOAD GENE AND BARCODE METADATA
# ============================================================

print("\n========== LOADING GEO METADATA ==========")

features = pd.read_csv(
    FEATURES_FILE,
    sep="\t",
    header=None,
    dtype=str
)

barcodes = pd.read_csv(
    BARCODES_FILE,
    sep="\t",
    header=None,
    dtype=str
)

print("Features shape:", features.shape)
print("Barcodes shape:", barcodes.shape)

# Standard 10x-style features file:
# Column 0 = gene ID
# Column 1 = gene name
# Column 2 = feature type (if available)

gene_ids = features.iloc[:, 0].astype(str).values

if features.shape[1] >= 2:
    gene_names = features.iloc[:, 1].astype(str).values
else:
    gene_names = gene_ids.copy()

if features.shape[1] >= 3:
    feature_types = features.iloc[:, 2].astype(str).values
else:
    feature_types = np.array(
        ["Gene Expression"] * len(gene_ids)
    )

barcode_names = barcodes.iloc[:, 0].astype(str).values


# ============================================================
# 3. LOAD EXPRESSION MATRIX
# ============================================================

print("\n========== LOADING EXPRESSION MATRIX ==========")

with gzip.open(MATRIX_FILE, "rb") as file:
    matrix = mmread(file)

matrix = csr_matrix(matrix)

print("Original matrix shape:", matrix.shape)

# Matrix Market orientation can differ.
# Determine orientation using metadata dimensions.

if (
    matrix.shape[0] == len(gene_ids)
    and matrix.shape[1] == len(barcode_names)
):

    print("Matrix orientation: Genes x Cells")

    matrix = matrix.T.tocsr()

elif (
    matrix.shape[0] == len(barcode_names)
    and matrix.shape[1] == len(gene_ids)
):

    print("Matrix orientation: Cells x Genes")

    matrix = matrix.tocsr()

else:

    raise ValueError(
        "Matrix dimensions do not match feature/barcode files.\n"
        f"Matrix: {matrix.shape}\n"
        f"Genes: {len(gene_ids)}\n"
        f"Barcodes: {len(barcode_names)}"
    )

print("Final matrix shape:", matrix.shape)


# ============================================================
# 4. VALIDATE COUNTS
# ============================================================

print("\n========== VALIDATING COUNTS ==========")

if matrix.nnz == 0:
    raise ValueError("Expression matrix contains no nonzero values.")

if np.any(matrix.data < 0):
    raise ValueError("Negative expression counts detected.")

if not np.allclose(matrix.data, np.round(matrix.data)):
    raise ValueError(
        "Matrix contains non-integer values. "
        "Inspect the source before treating it as raw counts."
    )

print("Nonzero entries:", matrix.nnz)
print("Total expression counts:", int(matrix.sum()))


# ============================================================
# 5. CREATE ANNData OBJECT
# ============================================================

print("\n========== CREATING ANNData ==========")

obs = pd.DataFrame(
    {
        "sample_id": "PDAML1_EOC2",
        "geo_accession": "GSM8376306"
    },
    index=barcode_names
)

var = pd.DataFrame(
    {
        "gene_id": gene_ids,
        "gene_symbol": gene_names,
        "feature_type": feature_types
    },
    index=gene_names
)

adata = sc.AnnData(
    X=matrix,
    obs=obs,
    var=var
)

# Make duplicate gene names unique while preserving
# the original gene symbols in var["gene_symbol"].
adata.var_names_make_unique()

print(adata)


# ============================================================
# 6. CALCULATE BASIC QUALITY CONTROL METRICS
# ============================================================

print("\n========== CALCULATING QC METRICS ==========")

sc.pp.calculate_qc_metrics(
    adata,
    percent_top=None,
    log1p=False,
    inplace=True
)

qc_summary = pd.DataFrame(
    {
        "metric": [
            "number_of_cells",
            "number_of_genes",
            "total_counts",
            "nonzero_expression_entries",
            "median_counts_per_cell",
            "median_genes_per_cell",
            "mean_counts_per_cell",
            "mean_genes_per_cell"
        ],
        "value": [
            adata.n_obs,
            adata.n_vars,
            int(adata.X.sum()),
            int(adata.X.nnz),
            float(adata.obs["total_counts"].median()),
            float(adata.obs["n_genes_by_counts"].median()),
            float(adata.obs["total_counts"].mean()),
            float(adata.obs["n_genes_by_counts"].mean())
        ]
    }
)

print(qc_summary.to_string(index=False))


# ============================================================
# 7. SAVE EXTRACTED DATASET
# ============================================================

print("\n========== SAVING GEO DATA ==========")

# Preserve original, unnormalized counts.
adata.write_h5ad(
    OUTPUT_DIR / "PDAML1_EOC2_raw.h5ad",
    compression="gzip"
)

# Save separate metadata and QC tables.
adata.obs.to_csv(
    OUTPUT_DIR / "barcode_metadata.csv"
)

adata.var.to_csv(
    OUTPUT_DIR / "gene_metadata.csv"
)

adata.obs.to_csv(
    OUTPUT_DIR / "cell_qc_metrics.csv"
)

qc_summary.to_csv(
    OUTPUT_DIR / "qc_summary.csv",
    index=False
)

# Save a machine-readable dataset summary.
summary = {
    "geo_accession": "GSE271406",
    "sample_accession": "GSM8376306",
    "sample_id": "PDAML1_EOC2",
    "number_of_cells": int(adata.n_obs),
    "number_of_genes": int(adata.n_vars),
    "total_counts": int(adata.X.sum()),
    "nonzero_entries": int(adata.X.nnz),
    "matrix_format": "AnnData",
    "normalization_performed": False,
    "filtering_performed": False
}

with open(OUTPUT_DIR / "dataset_summary.json", "w") as file:
    json.dump(summary, file, indent=4)


# ============================================================
# 8. FINAL REPORT
# ============================================================

print("\n==============================================")
print("GEO DATA EXTRACTION COMPLETED")
print("==============================================")

print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)
print("Total counts:", int(adata.X.sum()))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("1. PDAML1_EOC2_raw.h5ad")
print("2. barcode_metadata.csv")
print("3. gene_metadata.csv")
print("4. cell_qc_metrics.csv")
print("5. qc_summary.csv")
print("6. dataset_summary.json")

print("\nOriginal expression counts preserved.")
print("No normalization or filtering performed.")

print("\n==============================================")
