# src/identify_aml_candidates.py

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RNA_FILE = (
    ROOT / "data/processed/geo/PDAML1_EOC2/"
    "multimodal_analysis/RNA_only_analysis.h5ad"
)

PROTEIN_FILE = (
    ROOT / "data/processed/geo/PDAML1_EOC2/"
    "multimodal_analysis/Antibody_CLR_analysis.h5ad"
)

OUTPUT_DIR = (
    ROOT / "data/processed/geo/PDAML1_EOC2/"
    "aml_candidate_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Candidate ranking threshold: top 10% of the combined
# RNA-based blast-like score. This is exploratory, not diagnostic.
CANDIDATE_PERCENTILE = 90

# ============================================================
# 2. MARKER PANELS
# ============================================================

# RNA markers associated with AML, immature myeloid cells,
# hematopoietic progenitors and myeloid differentiation.
RNA_MARKERS = {
    "AML_associated": [
        "CD34", "CD33", "KIT", "FLT3", "MPO",
        "HOXA9", "MEIS1", "WT1", "RUNX1",
        "CEBPA", "ANPEP"
    ],

    "Immature_progenitor": [
        "CD34", "KIT", "PROM1", "GATA2",
        "HLF", "MEIS1", "HOXA9", "SOX4"
    ],

    "Myeloid_differentiation": [
        "CD33", "MPO", "ELANE", "PRTN3",
        "LYZ", "S100A8", "S100A9", "FCGR3A",
        "CTSS", "CSF1R", "CEBPA"
    ],

    "Monocyte_macrophage": [
        "CD14", "LYZ", "FCGR3A", "LST1",
        "MS4A7", "AIF1", "CSF1R"
    ],

    "T_cell_exclusion": [
        "CD3D", "CD3E", "CD3G", "TRAC"
    ],

    "B_cell_exclusion": [
        "MS4A1", "CD79A", "CD79B", "CD19"
    ],

    "Erythroid_exclusion": [
        "GYPA", "HBA1", "HBA2", "HBB",
        "ALAS2", "SLC4A1"
    ],

    "Proliferation": [
        "MKI67", "TOP2A", "TYMS",
        "STMN1", "PCLAF"
    ]
}

# Antibody panel. The script automatically uses only
# features actually present in your antibody dataset.
PROTEIN_MARKERS = {
    "AML_associated": [
        "CD34", "CD33", "CD117", "CD123",
        "CD13", "CD38", "CLEC12A"
    ],

    "Myeloid": [
        "CD14", "CD33", "CD13", "CLEC12A"
    ],

    "Progenitor": [
        "CD34", "CD117", "CD123"
    ],

    "HLA_DR": [
        "HLA-DRDPDQ"
    ]
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def dense_column(matrix, index):
    """Extract one matrix column without densifying the full matrix."""
    col = matrix[:, index]
    if sp.issparse(col):
        return col.toarray().ravel()
    return np.asarray(col).ravel()


def get_gene_symbols(adata):
    """Get gene symbols from var metadata or var_names."""

    fallback = pd.Series(
        adata.var_names.astype(str),
        index=adata.var.index
    )

    for column in ["gene_symbol", "gene_symbols", "symbol"]:
        if column in adata.var.columns:
            symbols = adata.var[column].astype("string")
            symbols = symbols.fillna(fallback)
            symbols = symbols.replace("", pd.NA).fillna(fallback)

            return symbols.astype(str).str.upper().tolist()

    return adata.var_names.astype(str).str.upper().tolist()


def make_feature_map(feature_names):
    """Map a feature name to its matrix column."""
    mapping = {}

    for i, name in enumerate(feature_names):
        name = str(name).upper()
        mapping.setdefault(name, []).append(i)

    return mapping


def find_rna_features(symbols, marker_list):
    """Find all matching columns for a list of RNA symbols."""
    indices = []

    for marker in marker_list:
        indices.extend(symbols.get(marker.upper(), []))

    return sorted(set(indices))


def mean_marker_expression(matrix, indices, n_cells):
    """Calculate per-cell average expression across selected markers."""
    if not indices:
        return np.zeros(n_cells)

    selected = matrix[:, indices]

    if sp.issparse(selected):
        return np.asarray(selected.mean(axis=1)).ravel()

    return np.asarray(selected).mean(axis=1).ravel()


def zscore(values):
    """Standardize a vector while handling constant values."""
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)

    if not np.isfinite(sd) or sd == 0:
        return np.zeros(len(values))

    return (values - np.nanmean(values)) / sd


def find_protein_feature(feature_map, marker):
    """
    Match antibody features such as CD8_TotalSeq or
    HLA-DRDPDQ_TotalSeq to the requested marker.
    """
    marker = marker.upper()

    aliases = {
        "CD117": ["CD117", "CD117_TOTALSEQ", "KIT"],
        "CD123": ["CD123", "CD123_TOTALSEQ"],
        "CD13": ["CD13", "CD13_TOTALSEQ"],
        "CD34": ["CD34", "CD34_TOTALSEQ"],
        "CD33": ["CD33", "CD33_TOTALSEQ"],
        "CD38": ["CD38", "CD38_TOTALSEQ"],
        "CLEC12A": ["CLEC12A", "CLEC12A_TOTALSEQ"],
        "CD14": ["CD14", "CD14_TOTALSEQ"],
        "HLA-DRDPDQ": [
            "HLA-DRDPDQ", "HLA-DRDPDQ_TOTALSEQ"
        ]
    }

    possible = aliases.get(marker, [marker])

    for candidate in possible:
        if candidate in feature_map:
            return feature_map[candidate]

    # Also support feature names with suffixes.
    for name, indices in feature_map.items():
        clean_name = name.replace("_TOTALSEQ", "")
        if clean_name == marker:
            return indices

    return []


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\nLoading RNA and antibody datasets...")

if not RNA_FILE.exists():
    raise FileNotFoundError(f"RNA file not found: {RNA_FILE}")

if not PROTEIN_FILE.exists():
    raise FileNotFoundError(f"Antibody file not found: {PROTEIN_FILE}")

rna = sc.read_h5ad(RNA_FILE)
protein = sc.read_h5ad(PROTEIN_FILE)

print("RNA dataset:", rna.shape)
print("Antibody dataset:", protein.shape)

# Check barcode alignment.
if not rna.obs_names.is_unique or not protein.obs_names.is_unique:
    raise ValueError("Duplicate cell barcodes found.")

common_cells = rna.obs_names.intersection(protein.obs_names)

if len(common_cells) == 0:
    raise ValueError(
        "RNA and antibody datasets have no matching cell barcodes."
    )

# Reorder both datasets to the same cells.
rna = rna[common_cells].copy()
protein = protein[common_cells].copy()

if not np.array_equal(rna.obs_names, protein.obs_names):
    raise ValueError("RNA and antibody barcode ordering does not match.")

print("Common aligned cells:", len(common_cells))


# ============================================================
# 5. IDENTIFY RNA FEATURES
# ============================================================

print("\nIdentifying RNA marker genes...")

gene_symbols_list = get_gene_symbols(rna)
gene_map = make_feature_map(gene_symbols_list)

# Use normalized/log-transformed RNA expression in X.
rna_matrix = rna.X
n_cells = rna.n_obs

rna_marker_values = pd.DataFrame(index=rna.obs_names)

for group, markers in RNA_MARKERS.items():
    indices = find_rna_features(gene_map, markers)

    if indices:
        rna_marker_values[group] = mean_marker_expression(
            rna_matrix, indices, n_cells
        )
    else:
        rna_marker_values[group] = np.zeros(n_cells)

    found = sorted({
        gene_symbols_list[i] for i in indices
    })

    print(f"{group}: {len(found)} markers found - {found}")


# Export individual marker expression.
individual_rna_markers = {}

for group, markers in RNA_MARKERS.items():
    for marker in markers:
        indices = gene_map.get(marker.upper(), [])

        if indices:
            individual_rna_markers[marker] = mean_marker_expression(
                rna_matrix, indices, n_cells
            )

rna_marker_df = pd.DataFrame(
    individual_rna_markers,
    index=rna.obs_names
)

rna_marker_df.to_csv(
    OUTPUT_DIR / "individual_rna_marker_expression.csv"
)


# ============================================================
# 6. IDENTIFY ANTIBODY FEATURES
# ============================================================

print("\nAnalyzing antibody features...")

protein_feature_names = [str(x).upper() for x in protein.var_names]
protein_map = make_feature_map(protein_feature_names)

protein_matrix = protein.X

protein_marker_values = pd.DataFrame(index=protein.obs_names)

for group, markers in PROTEIN_MARKERS.items():
    group_values = []

    for marker in markers:
        indices = find_protein_feature(protein_map, marker)

        if indices:
            group_values.extend(indices)

    group_values = sorted(set(group_values))

    if group_values:
        protein_marker_values[group] = mean_marker_expression(
            protein_matrix, group_values, n_cells
        )
    else:
        protein_marker_values[group] = np.zeros(n_cells)

    print(
        f"{group}: {len(group_values)} antibody features found"
    )

# Export individual protein CLR signals.
individual_protein_markers = {}

for marker in sorted({
    marker
    for markers in PROTEIN_MARKERS.values()
    for marker in markers
}):
    indices = find_protein_feature(protein_map, marker)

    if indices:
        individual_protein_markers[marker] = mean_marker_expression(
            protein_matrix, indices, n_cells
        )

protein_marker_df = pd.DataFrame(
    individual_protein_markers,
    index=protein.obs_names
)

protein_marker_df.to_csv(
    OUTPUT_DIR / "individual_protein_CLR_expression.csv"
)


# ============================================================
# 7. CREATE CELL-LEVEL RNA SCORES
# ============================================================

print("\nCalculating RNA-based candidate scores...")

# Separate biological dimensions.
rna_scores = pd.DataFrame(index=rna.obs_names)

for group in RNA_MARKERS:
    rna_scores[group] = zscore(rna_marker_values[group])

# Immature/progenitor and myeloid signals are considered together.
rna_scores["RNA_blast_like_score"] = (
    rna_scores["AML_associated"] * 0.40
    + rna_scores["Immature_progenitor"] * 0.35
    + rna_scores["Myeloid_differentiation"] * 0.25
)

# A separate myeloid score helps distinguish myeloid cells
# from lymphoid and erythroid populations.
rna_scores["RNA_myeloid_score"] = (
    rna_scores["Myeloid_differentiation"] * 0.65
    + rna_scores["Monocyte_macrophage"] * 0.35
)

rna_scores["RNA_lineage_exclusion_score"] = (
    rna_scores["T_cell_exclusion"]
    + rna_scores["B_cell_exclusion"]
    + rna_scores["Erythroid_exclusion"]
) / 3

# A candidate should have immature/progenitor and myeloid evidence,
# not simply high expression of one generic AML-associated marker.
rna_scores["RNA_candidate_score"] = (
    rna_scores["RNA_blast_like_score"] * 0.65
    + rna_scores["RNA_myeloid_score"] * 0.35
    - rna_scores["RNA_lineage_exclusion_score"] * 0.20
)


# ============================================================
# 8. CREATE CELL-LEVEL PROTEIN SCORES
# ============================================================

print("\nCalculating antibody-based candidate scores...")

protein_scores = pd.DataFrame(index=protein.obs_names)

for group in PROTEIN_MARKERS:
    protein_scores[group] = zscore(protein_marker_values[group])

protein_scores["Protein_blast_like_score"] = (
    protein_scores["AML_associated"] * 0.60
    + protein_scores["Progenitor"] * 0.30
    + protein_scores["Myeloid"] * 0.10
)


# ============================================================
# 9. COMBINE RNA AND PROTEIN EVIDENCE
# ============================================================

results = pd.concat(
    [
        rna_scores,
        protein_scores,
        rna_marker_values.add_prefix("RNA_mean_"),
        protein_marker_values.add_prefix("Protein_CLR_mean_"),
        rna_marker_df.add_prefix("RNA_gene_"),
        protein_marker_df.add_prefix("Protein_")
    ],
    axis=1
)

# Preserve original cell metadata.
metadata = rna.obs.copy()

results = results.join(metadata, how="left")

# Preserve existing RNA cluster assignments if available.
cluster_columns = [
    col for col in rna.obs.columns
    if "leiden" in col.lower()
]

if cluster_columns:
    print("Cluster columns found:", cluster_columns)

    # Prefer the resolution 0.5 column if it exists.
    preferred = [
        col for col in cluster_columns
        if "0.5" in col or "0_5" in col
    ]

    cluster_column = preferred[0] if preferred else cluster_columns[0]

    results["RNA_cluster"] = rna.obs[cluster_column].astype(str)

else:
    cluster_column = None
    results["RNA_cluster"] = "Not_available"


# Standardize the RNA and protein scores before combining.
# Protein CLR is treated as supporting evidence, not a diagnosis.
results["Combined_candidate_score"] = (
    zscore(results["RNA_candidate_score"]) * 0.75
    + zscore(results["Protein_blast_like_score"]) * 0.25
)

threshold = np.nanpercentile(
    results["Combined_candidate_score"],
    CANDIDATE_PERCENTILE
)

results["Candidate_rank"] = results[
    "Combined_candidate_score"
].rank(ascending=False, method="min")

results["Candidate_percentile"] = results[
    "Combined_candidate_score"
].rank(pct=True) * 100

# ============================================================
# 10. CONSERVATIVE CANDIDATE CLASSIFICATION
# ============================================================

# Count the number of detected RNA markers in important panels.
# Detection here means normalized RNA expression > 0.
# It is a screening indicator, not a clinical positivity threshold.

def count_detected_markers(marker_list):
    count = np.zeros(n_cells, dtype=int)

    for marker in marker_list:
        indices = gene_map.get(marker.upper(), [])

        if indices:
            expression = mean_marker_expression(
                rna_matrix, indices, n_cells
            )
            count += (expression > 0).astype(int)

    return count


results["AML_RNA_markers_detected"] = count_detected_markers(
    RNA_MARKERS["AML_associated"]
)

results["Progenitor_RNA_markers_detected"] = count_detected_markers(
    RNA_MARKERS["Immature_progenitor"]
)

results["Myeloid_RNA_markers_detected"] = count_detected_markers(
    RNA_MARKERS["Myeloid_differentiation"]
)

results["T_cell_markers_detected"] = count_detected_markers(
    RNA_MARKERS["T_cell_exclusion"]
)

results["B_cell_markers_detected"] = count_detected_markers(
    RNA_MARKERS["B_cell_exclusion"]
)

results["Erythroid_markers_detected"] = count_detected_markers(
    RNA_MARKERS["Erythroid_exclusion"]
)

# Rank-based candidate selection.
# Requiring multiple RNA markers avoids flagging cells on one gene alone.
candidate_mask = (
    (results["Candidate_percentile"] >= CANDIDATE_PERCENTILE)
    & (results["AML_RNA_markers_detected"] >= 2)
    & (results["Myeloid_RNA_markers_detected"] >= 2)
)

results["Candidate_status"] = "Not_high_priority"

results.loc[
    candidate_mask,
    "Candidate_status"
] = "High_priority_AML_candidate"

# Identify cells with an immature/myeloid RNA profile but weaker
# combined evidence. Keep these as a separate exploratory group.
suggestive_mask = (
    (results["RNA_candidate_score"] >= results["RNA_candidate_score"].quantile(0.75))
    & (results["AML_RNA_markers_detected"] >= 2)
    & (results["Myeloid_RNA_markers_detected"] >= 1)
    & (~candidate_mask)
)

results.loc[
    suggestive_mask,
    "Candidate_status"
] = "Suggestive_myeloid_immature"

# Flag cells with strong lymphoid/erythroid signals for review.
results["Alternative_lineage_signal"] = (
    (results["T_cell_markers_detected"] >= 2)
    | (results["B_cell_markers_detected"] >= 2)
    | (results["Erythroid_markers_detected"] >= 2)
)

# Do not automatically discard these cells. Some AML cells can
# express unexpected markers; this is a review flag only.
results.loc[
    candidate_mask & results["Alternative_lineage_signal"],
    "Candidate_status"
] = "High_priority_candidate_review_lineage_signal"


# ============================================================
# 11. SAVE CELL-LEVEL RESULTS
# ============================================================

results = results.sort_values(
    "Combined_candidate_score",
    ascending=False
)

results.to_csv(
    OUTPUT_DIR / "all_cells_aml_candidate_scores.csv"
)

high_priority = results[
    results["Candidate_status"].str.contains(
        "High_priority", na=False
    )
].copy()

suggestive = results[
    results["Candidate_status"] == "Suggestive_myeloid_immature"
].copy()

high_priority.to_csv(
    OUTPUT_DIR / "high_priority_aml_candidates.csv"
)

suggestive.to_csv(
    OUTPUT_DIR / "suggestive_immature_myeloid_cells.csv"
)


# ============================================================
# 12. CLUSTER-LEVEL SUMMARY
# ============================================================

cluster_summary = (
    results.groupby("RNA_cluster", observed=True)
    .agg(
        n_cells=("Combined_candidate_score", "size"),
        mean_candidate_score=("Combined_candidate_score", "mean"),
        median_candidate_score=("Combined_candidate_score", "median"),
        mean_RNA_candidate_score=("RNA_candidate_score", "mean"),
        mean_protein_score=("Protein_blast_like_score", "mean"),
        mean_AML_RNA_markers=("AML_RNA_markers_detected", "mean"),
        mean_progenitor_markers=("Progenitor_RNA_markers_detected", "mean"),
        mean_myeloid_markers=("Myeloid_RNA_markers_detected", "mean"),
        high_priority_candidates=(
            "Candidate_status",
            lambda x: x.str.contains("High_priority", na=False).sum()
        ),
        suggestive_cells=(
            "Candidate_status",
            lambda x: (x == "Suggestive_myeloid_immature").sum()
        )
    )
    .reset_index()
    .sort_values("mean_candidate_score", ascending=False)
)

cluster_summary.to_csv(
    OUTPUT_DIR / "aml_candidate_cluster_summary.csv",
    index=False
)


# ============================================================
# 13. MARKER CO-EXPRESSION SUMMARY
# ============================================================

coexpression_columns = [
    col for col in results.columns
    if col.startswith("RNA_gene_")
    or col.startswith("Protein_")
]

# Export the candidate cells with their individual marker values.
candidate_details = results[
    results["Candidate_status"].str.contains(
        "High_priority|Suggestive", na=False
    )
].copy()

candidate_details.to_csv(
    OUTPUT_DIR / "candidate_marker_coexpression.csv"
)


# ============================================================
# 14. VISUALIZATION: CANDIDATE SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    results["Combined_candidate_score"],
    bins=40,
    kde=True
)

plt.axvline(
    threshold,
    linestyle="--",
    label=f"Top {100 - CANDIDATE_PERCENTILE}% threshold"
)

plt.xlabel("Combined AML candidate score")
plt.ylabel("Number of cells")
plt.title("Distribution of AML Candidate Scores")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "candidate_score_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. VISUALIZATION: CLUSTER CANDIDATE COUNTS
# ============================================================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=cluster_summary,
    x="RNA_cluster",
    y="high_priority_candidates"
)

plt.xlabel("RNA Leiden cluster")
plt.ylabel("High-priority candidate cells")
plt.title("High-Priority AML Candidate Cells by RNA Cluster")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "candidate_cells_by_cluster.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. VISUALIZATION: RNA MARKER HEATMAP
# ============================================================

heatmap_markers = [
    "CD34", "CD33", "KIT", "FLT3", "MPO",
    "HOXA9", "MEIS1", "WT1", "RUNX1",
    "CEBPA", "ANPEP", "ELANE", "PRTN3"
]

available_heatmap_markers = [
    marker for marker in heatmap_markers
    if marker in rna_marker_df.columns
]

if available_heatmap_markers and len(high_priority) > 0:

    top_cells = high_priority.head(100)

    heatmap_data = rna_marker_df.loc[
        top_cells.index,
        available_heatmap_markers
    ]

    plt.figure(
        figsize=(
            max(10, len(available_heatmap_markers) * 0.8),
            max(6, len(heatmap_data) * 0.12)
        )
    )

    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        xticklabels=True,
        yticklabels=False
    )

    plt.title("RNA Marker Expression in Top AML Candidate Cells")
    plt.xlabel("RNA marker")
    plt.ylabel("Candidate cells")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top_candidate_rna_marker_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

summary = {
    "sample": "PDAML1_EOC2",
    "total_cells_analyzed": int(len(results)),
    "high_priority_candidate_cells": int(len(high_priority)),
    "suggestive_immature_myeloid_cells": int(len(suggestive)),
    "candidate_percentile_threshold": CANDIDATE_PERCENTILE,
    "combined_score_threshold": float(threshold),
    "rna_markers_available": sorted(
        set(gene_symbols_list).intersection(
            marker
            for group in RNA_MARKERS.values()
            for marker in group
        )
    ),
    "protein_features_available": [
        str(x) for x in protein.var_names
    ],
    "cluster_column_used": cluster_column,
    "cluster_summary": cluster_summary.to_dict(orient="records"),
    "interpretation": (
        "These are exploratory AML/blast-like candidate rankings. "
        "They are not confirmed malignant AML cells. "
        "Expression of AML-associated markers does not establish "
        "leukemic clonality. Clinical immunophenotyping and "
        "patient-specific genetic evidence are required for validation."
    )
}

with open(
    OUTPUT_DIR / "aml_candidate_analysis_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=4, default=str)


# ============================================================
# 18. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("AML CANDIDATE IDENTIFICATION COMPLETE")
print("=" * 65)

print(f"Total cells analyzed: {len(results)}")
print(f"High-priority candidates: {len(high_priority)}")
print(f"Suggestive immature/myeloid cells: {len(suggestive)}")

print("\nTop 15 candidate cells:")
display_columns = [
    "RNA_cluster",
    "Combined_candidate_score",
    "RNA_candidate_score",
    "Protein_blast_like_score",
    "AML_RNA_markers_detected",
    "Progenitor_RNA_markers_detected",
    "Myeloid_RNA_markers_detected",
    "Candidate_status"
]

print(
    results[display_columns].head(15).to_string()
)

print("\nCluster summary:")
print(cluster_summary.to_string(index=False))

print("\nOutput files saved to:")
print(OUTPUT_DIR)

print("\nIMPORTANT:")
print(
    "These results identify suggestive AML/blast-like candidates. "
    "They do NOT prove that the cells are malignant AML blasts."
)