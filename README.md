# Bone Marrow AML — AI-Driven Spatial Mapping

An AI-assisted computational pathology and bioinformatics project focused on nucleus segmentation, spatial morphology analysis and exploratory molecular characterization of Acute Myeloid Leukemia (AML) bone marrow.

## Project Overview

This project investigates computational approaches for analyzing AML bone marrow microscopy images and molecular data.

The workflow combines pretrained deep-learning segmentation models, spatial feature extraction and exploratory single-cell gene-expression analysis.

### Main Objectives

- Evaluate pretrained nucleus segmentation models (StarDist and Cellpose).
- Extract nucleus-level spatial coordinates and morphological features.
- Explore AML-related gene-expression profiles and cell populations.
- Investigate marker-associated fluorescence images.
- Establish a foundation for future spatial analysis of leukemia and immune-cell organization.

## Datasets

The project uses publicly available datasets associated with the primary research publication.

| Dataset | Description |
|---|---|
| Nucleus_train.zip | Nucleus microscopy images, ground-truth masks and auxiliary flow files. |
| Membrane_train.zip | Cell membrane images, ground-truth masks and flow files. |
| AML_images.zip | AML bone marrow microscopy images, including DAPI, B2M and CD34-labelled image sets. |
| GSE271406 | Single-cell gene-expression and antibody-capture data. |

### Dataset Downloads

**Zenodo — AML microscopy datasets**

https://zenodo.org/records/13334134

Contains:
- `Nucleus_train.zip`
- `Membrane_train.zip`
- `AML_images.zip`

**Gene Expression Omnibus — GSE271406**

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271406

Contains single-cell gene-expression data and antibody-capture features.

Download the datasets and place them in the following structure:

```text
data/
├── raw/
│   ├── cosmx/
│   │   ├── archives/
│   │   │   ├── Nucleus_train.zip
│   │   │   └── Membrane_train.zip
│   │   └── AML_images.zip
│   │
│   └── geo/
│       └── PDAML1_EOC2/
│           ├── barcodes.tsv.gz
│           ├── features.tsv.gz
│           └── matrix.mtx.gz
│
└── processed/
```

The large image archives can be read directly from ZIP files by the relevant inspection scripts.

## Research Paper

**Primary Research Reference**

Gui, G., Bingham, M. A., et al. (2025).

*Single-cell spatial transcriptomics reveals immunotherapy-driven bone marrow niche remodeling in AML.*

Science Advances, 11(28), eadw4871.

https://doi.org/10.1126/sciadv.adw4871

Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12239967/

Dataset: https://zenodo.org/records/13334134

Original research code: https://github.com/chenzhaolab2023/AML-Spatial

This publication provides the primary biological and methodological reference for the project, including CosMx spatial transcriptomics, image processing, cell segmentation, marker-based characterization and AML bone marrow niche analysis.

## Technology Stack

- **Python 3.11**
- PyTorch
- Cellpose
- StarDist
- Scanpy
- Squidpy
- AnnData
- NumPy and Pandas
- OpenCV
- Matplotlib
- SciPy
- scikit-learn

## Installation

Clone the repository:

```bash
git clone <https://github.com/etudiefischer/Bone-Marrow-AML>
cd Bone-Marrow-AML
```

Create a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The following are six important entry-point scripts.

**1. Inspect microscopy samples**

```bash
python src/inspect_samples.py
```

Inspects available microscopy images and dataset characteristics.

**2. Evaluate StarDist across all FOVs**

```bash
python src/evaluate_stardist_all.py
```

Runs pretrained StarDist segmentation evaluation across matched nucleus image-mask fields of view.

**3. Extract spatial coordinates**

```bash
python src/prepare_spatial_dataset.py
```

Extracts nucleus centroids and morphological features from ground-truth masks.

**4. Validate the spatial dataset**

```bash
python src/validate_spatial_dataset.py
```

Checks coordinate validity, missing values, duplicate cell identifiers and FOV consistency.

**5. Extract GEO expression data**

```bash
python src/extract_geo_data.py
```

Converts the downloaded GEO expression matrix and metadata into AnnData format.

**6. Run GEO multimodal analysis**

```bash
python src/rebuild_geo_multimodal.py
```

Performs RNA-only quality control, normalization, highly variable gene selection, PCA, UMAP, Leiden clustering and exploratory antibody-feature comparison.

Additional scripts for Cellpose benchmarking, AML candidate scoring, image inspection and visualization are available under `src/`.

## Current Project Results

| Analysis | Result |
|---|---:|
| Nucleus image-mask FOVs evaluated | 298 |
| Patients in nucleus dataset | 5 |
| Extracted nuclei | 224,225 |
| Full StarDist mean Dice score | 0.7186 |
| Full StarDist instance F1 | 0.6692 |
| RNA QC-passing cells | 2,815 |
| RNA Leiden clusters | 10 |
| AML marker images inspected | 142 |

The current milestone establishes a nucleus segmentation and spatial morphology pipeline alongside independent exploratory molecular characterization.

Direct cell-level integration and leukemia–immune spatial interaction analysis remain future work.

## Repository Structure

```text
Bone-Marrow-AML/
│
├── src/
│   ├── evaluation/
│   ├── segmentation/
│   ├── tests/
│   └── *.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
├── LICENSE
└── README.md
```

## License

This project is released under the MIT License. Refer to the `LICENSE` file for details.

Dataset usage remains subject to the original dataset's terms and attribution requirements.
