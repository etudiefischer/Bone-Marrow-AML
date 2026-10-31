
from pathlib import Path

import scanpy as sc
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/geo/PDAML1_EOC2/normalized/"
    "PDAML1_EOC2_normalized.h5ad"
)

OUTPUT_DIR = (
    ROOT / "data/processed/geo/PDAML1_EOC2/marker_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading normalized GEO dataset...")

adata = sc.read_h5ad(INPUT_FILE)

print(adata)
print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)


# ============================================================
# 3. PREPARE GENE SYMBOLS
# ============================================================

print("\nPreparing gene symbols...")

# Convert categorical values to regular Python strings.
gene_symbols = (
    adata.var["gene_symbol"]
    .astype("string")
    .fillna("")
    .str.strip()
    .str.upper()
)

# Map each gene symbol to all matching feature indices.
symbol_to_indices = {}

for idx, symbol in enumerate(gene_symbols):
    if symbol and symbol != "NAN":
        symbol_to_indices.setdefault(symbol, []).append(idx)


# ============================================================
# 4. DEFINE BIOLOGICAL MARKER GROUPS
# ============================================================

marker_groups = {

    "AML_myeloid": [
        "CD34", "CD33", "MPO", "KIT", "FLT3",
        "ELANE", "PRTN3", "WT1", "RUNX1",
        "ANPEP", "CEBPA", "HOXA9", "MEIS1"
    ],

    "T_cells": [
        "CD3D", "CD3E", "CD2", "CD4",
        "CD8A", "CD8B", "IL7R", "TRAC"
    ],

    "B_cells": [
        "MS4A1", "CD79A", "CD79B",
        "CD19", "CD37", "CD74"
    ],

    "NK_cells": [
        "NKG7", "GNLY", "KLRD1",
        "PRF1", "GZMB", "GZMK"
    ],

    "Monocytes": [
        "CD14", "LYZ", "FCGR3A",
        "S100A8", "S100A9", "CTSS"
    ],

    "Macrophages": [
        "C1QA", "C1QB", "C1QC",
        "CD68", "CSF1R", "MRC1"
    ],

    "Neutrophils": [
        "FCGR3B", "CSF3R", "MPO",
        "S100A8", "S100A9", "CXCR2"
    ],

    "Dendritic_cells": [
        "FCER1A", "CD1C", "CLEC10A",
        "CLEC9A", "ITGAX"
    ]
}


# ============================================================
# 5. INSPECT MARKER EXPRESSION
# ============================================================

print("\n========== MARKER INSPECTION ==========")

results = []

for group, markers in marker_groups.items():

    print(f"\n{group}")
    print("-" * 60)

    for marker in markers:

        indices = symbol_to_indices.get(marker, [])

        present = len(indices) > 0

        if present:

            # Retrieve expression for all matching gene features.
            expression = adata.X[:, indices]

            # Convert sparse matrix to dense if necessary.
            if hasattr(expression, "toarray"):
                expression = expression.toarray()

            # Aggregate duplicate gene-symbol features.
            expression = np.asarray(expression).sum(axis=1).ravel()

            expressing_cells = int(np.count_nonzero(expression > 0))
            mean_expression = float(np.mean(expression))
            median_expression = float(np.median(expression))

            percentage_expressing = (
                expressing_cells / adata.n_obs
            ) * 100

        else:

            expressing_cells = 0
            mean_expression = 0.0
            median_expression = 0.0
            percentage_expressing = 0.0

        results.append({

            "cell_group": group,
            "gene": marker,
            "present_in_dataset": present,
            "number_of_features": len(indices),
            "expressing_cells": expressing_cells,
            "percentage_expressing_cells": round(
                percentage_expressing, 2
            ),
            "mean_expression": round(mean_expression, 4),
            "median_expression": round(median_expression, 4)

        })

        status = "FOUND" if present else "NOT FOUND"

        print(
            f"{marker:12} | {status:10} | "
            f"Expressing cells: {expressing_cells:5} | "
            f"{percentage_expressing:.2f}%"
        )


# ============================================================
# 6. SAVE INDIVIDUAL MARKER RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "marker_availability.csv",
    index=False
)

print("\nIndividual marker results saved.")


# ============================================================
# 7. GROUP-LEVEL SUMMARY
# ============================================================

summary = (
    results_df.groupby("cell_group")
    .agg(
        total_markers=("gene", "count"),
        available_markers=("present_in_dataset", "sum"),
        markers_with_expression=(
            "expressing_cells",
            lambda x: int((x > 0).sum())
        )
    )
    .reset_index()
)

summary["availability_percent"] = (
    summary["available_markers"] /
    summary["total_markers"] * 100
).round(2)

summary.to_csv(
    OUTPUT_DIR / "marker_group_summary.csv",
    index=False
)


# ============================================================
# 8. SAVE FOUND MARKERS ONLY
# ============================================================

found_markers = results_df[
    results_df["present_in_dataset"]
]

found_markers.to_csv(
    OUTPUT_DIR / "found_markers.csv",
    index=False
)


# ============================================================
# 9. PRINT FINAL REPORT
# ============================================================

print("\n==============================================")
print("GEO MARKER INSPECTION COMPLETED")
print("==============================================")

print("\nGroup-level summary:")

print(
    summary.to_string(index=False)
)

print("\nTotal markers checked:", len(results_df))

print(
    "Markers present:",
    int(results_df["present_in_dataset"].sum())
)

print(
    "Markers with expression:",
    int((results_df["expressing_cells"] > 0).sum())
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("- marker_availability.csv")
print("- marker_group_summary.csv")
print("- found_markers.csv")

print("\n==============================================")
