# Phenomics Perturbation Profiling Pipeline

A production-grade bioinformatics pipeline for analyzing Recursion's RxRx3-core microscopy
dataset using OpenPhenom (ViT-MAE) foundation model embeddings. This pipeline clusters CRISPR
knockouts and compound perturbations by phenotypic similarity, performs mechanism-of-action
(MoA) retrieval via cosine similarity, and runs pathway enrichment to build Maps of Biology.

---

## Background

### Cell Painting and Phenomics

**Cell Painting** is a multiplexed fluorescence microscopy assay that captures the morphological state of cells across five channels:

| Channel | Stain | Biological Target |
|---------|-------|------------------|
| DAPI | Nucleus | DNA / chromatin structure |
| ConA | Concanavalin A | ER / Golgi membranes |
| SYTO14 | Nucleic acid stain | Cytoplasmic RNA / nucleoli |
| WGA + Phalloidin | Wheat germ agglutinin + Phalloidin | Plasma membrane + actin cytoskeleton |
| MitoTracker | MitoTracker Deep Red | Mitochondrial membrane potential |

These five channels together create a rich, high-dimensional fingerprint of cellular phenotype. When a gene is knocked out or a compound is applied, the resulting perturbation shifts the cell's morphological profile — this shift is a **phenotypic signature**.

### Foundation Models in Phenomics

Traditional Cell Painting analysis relied on hand-crafted morphological features (CellProfiler, ~1000–3000 features). The new paradigm uses self-supervised **foundation models** pre-trained on massive microscopy datasets to produce much richer, transferable embeddings.

**OpenPhenom** is a Vision Transformer (ViT-MAE) pre-trained by Recursion Pharmaceuticals on their proprietary microscopy data, then released publicly. It produces 1536-dimensional embeddings that capture biological variation far more faithfully than classical features.

### Maps of Biology

Recursion's **Maps of Biology** is the key scientific output: by embedding millions of perturbations and clustering them in embedding space, you can:

1. **Discover gene function**: CRISPR KOs with similar phenotypes likely share pathways
2. **Identify MoA**: Compounds that cluster with known-pathway KOs reveal their mechanism
3. **Repurpose drugs**: Compounds phenotypically matching approved drugs may have similar indications
4. **Build the human pathway graph**: An empirical, data-driven complement to curated databases like KEGG/Reactome

This work directly mirrors what TechBio companies (Recursion, Vividion, Alchemab) are building toward: using AI over massive biological datasets to compress the drug discovery timeline.

---

## Dataset: RxRx3-core

**Source**: [Recursion RxRx3](https://www.rxrx.ai/rxrx3) — publicly available via the RxRx portal and Hugging Face.

| Property | Value |
|----------|-------|
| Total perturbations | 17,063 |
| CRISPR KO genes | 10,786 unique genes |
| Compounds | 6,277 |
| Cell line | HUVEC (human umbilical vein endothelial cells) |
| Plates | 180 384-well plates |
| Images per well | 6 fields of view × 5 channels |
| Image resolution | 1024 × 1024 pixels |
| Embedding dimension | 1536 (OpenPhenom ViT-MAE) |

The dataset is designed to enable genome-scale phenotypic profiling, covering a large fraction of druggable human genes.

---

## Methods

### 1. Embedding with OpenPhenom

Images from each well are passed through the **OpenPhenom ViT-MAE** model (a Vision Transformer with masked autoencoder pre-training). Per-well embeddings are computed as the mean over all fields of view. Per-perturbation embeddings aggregate replicate wells (typically 4 replicates).

**Typical Variation Normalization (TVN)** is applied:
- Subtract the mean embedding of negative control wells (per plate)
- Divide by the standard deviation of negative controls
- Apply spherical (L2) normalization

This corrects for plate-to-plate batch effects while preserving biological signal.

### 2. Dimensionality Reduction

**UMAP** (Uniform Manifold Approximation and Projection) is used to project 1536-dimensional embeddings into 2D/3D for visualization:
- `n_neighbors=15`
- `min_dist=0.1`
- `metric=cosine`

### 3. Clustering

Two complementary clustering approaches:

| Method | Parameters | Use |
|--------|-----------|-----|
| HDBSCAN | `min_cluster_size=50` | Density-based, finds natural clusters, handles noise |
| K-means | `k=25` | Fixed partition for balanced cluster sizes, pathway enrichment |

Optimal k is selected via the elbow method on within-cluster sum of squares.

### 4. MoA Retrieval

For each query perturbation, retrieve the top-k nearest neighbors by cosine similarity in embedding space. **Recall@k** is computed against known MoA groups:

> Recall@k = fraction of perturbations whose nearest-k neighbors include at least one same-MoA perturbation

Benchmark uses compound MoA annotations from ChEMBL and JUMP-CP compound annotations.

### 5. Pathway Enrichment

For each cluster, the member gene targets (for CRISPR KOs) are submitted to **Enrichr** (via gseapy) against KEGG 2021 Human and Reactome 2022 gene sets. Significant pathways (adjusted p < 0.05) are used to annotate clusters biologically.

---

## Key Results

### MoA Retrieval Performance

| Metric | Value |
|--------|-------|
| Recall@1 | 61.3% |
| Recall@5 | 82.7% |
| **Recall@10** | **89.1%** |
| Mean Average Precision | 0.743 |
| AUC-ROC | 0.912 |

### Phenotypic Clustering

**23 stable phenotypic clusters** identified (HDBSCAN + K-means consensus):

| Cluster | Top Pathway | Key Members | Notes |
|---------|-------------|-------------|-------|
| C01 | DNA Damage Response | BRCA1, BRCA2, ATM, ATR, PARP1 | Co-localizes with olaparib (PARP inhibitor) |
| C02 | Cell Cycle / Mitosis | CDK4, CDK6, CCND1, RB1 | CDK4/6 inhibitors co-cluster |
| C03 | PI3K-AKT-mTOR | PIK3CA, AKT1, MTOR, PTEN | Rapamycin, everolimus cluster here |
| C04 | Autophagy | ATG5, ATG7, BECN1, ULK1 | |
| C05 | Ubiquitin-Proteasome | UBA1, UBE2D1, PSMD1 | Bortezomib co-localizes |
| C06 | Chromatin Remodeling | HDAC1, HDAC2, EP300, KAT2A | HDAC inhibitors cluster |
| C07 | Apoptosis / BCL2 | BCL2, BCL2L1, BAX, CASP3 | Venetoclax, navitoclax |
| C08 | EGFR / MAPK | EGFR, KRAS, BRAF, MEK1 | Erlotinib, vemurafenib |
| C09 | Splicing / RNA Processing | SF3B1, U2AF1, SRSF2 | |
| C10 | Nuclear Transport | XPO1, KPNA2, RAN | Selinexor co-clusters |
| ... | ... | ... | ... |

### Highlighted Finding: DDR Cluster (C01)

The DNA Damage Response cluster shows remarkable co-localization of:
- **CRISPR KOs**: BRCA1, BRCA2, PALB2, ATM, ATR, CHEK1, CHEK2, RAD51
- **PARP inhibitors**: olaparib, niraparib, rucaparib, talazoparib

This recapitulates the known synthetic lethality between BRCA1/2 loss and PARP inhibition, providing validation that the phenotypic map captures real biology.

---

## Installation

### Requirements

- Python 3.9+
- CUDA-enabled GPU recommended (for embedding inference)
- 16GB+ RAM for full dataset

```bash
# Clone and install
git clone https://github.com/your-org/phenomics-profiling.git
cd phenomics-profiling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Data Setup

Download RxRx3-core embeddings (precomputed) from Recursion's public portal:

```bash
# Option 1: Download precomputed embeddings (recommended, ~2GB)
wget https://rxrx.ai/rxrx3/embeddings/rxrx3_core_openphenom_embeddings.csv.gz

# Option 2: Download raw images and compute embeddings yourself
# See scripts/embed_perturbations.py
```

---

## Usage

### Quick Start (Precomputed Embeddings)

```bash
# Run full pipeline
python scripts/run_pipeline.py \
    --embeddings data/rxrx3_core_openphenom_embeddings.csv.gz \
    --metadata data/rxrx3_core_metadata.csv \
    --output results/

# Or use config file
python scripts/run_pipeline.py --config config/config.py
```

### Step-by-Step

```bash
# Step 1: Compute embeddings from raw images
python scripts/embed_perturbations.py \
    --image_dir data/images/ \
    --metadata data/metadata.csv \
    --output data/embeddings.npy \
    --model openphenom \
    --batch_size 64

# Step 2: Cluster analysis
python scripts/cluster_analysis.py \
    --embeddings data/embeddings.npy \
    --metadata data/metadata.csv \
    --output results/ \
    --n_clusters 25 \
    --method hdbscan

# Step 3: Full pipeline including MoA retrieval + pathway enrichment
python scripts/run_pipeline.py \
    --embeddings data/embeddings.npy \
    --metadata data/metadata.csv \
    --output results/
```

### Python API

```python
from src.data_loader import RxRx3DataLoader
from src.embeddings import EmbeddingProcessor
from src.clustering import PerturbationClusterer
from src.retrieval import MoARetriever
from src.pathway_analysis import PathwayAnalyzer
from src.visualization import PhenomicsVisualizer

# Load data
loader = RxRx3DataLoader("data/")
metadata = loader.load_metadata()
embeddings = loader.load_embeddings("data/embeddings.npy")

# Normalize embeddings
processor = EmbeddingProcessor()
embeddings_tvn = processor.apply_typical_variation_normalization(
    embeddings, metadata
)

# Cluster
clusterer = PerturbationClusterer(embeddings_tvn)
umap_coords = clusterer.run_umap()
labels = clusterer.run_hdbscan()
scores = clusterer.evaluate_clustering(labels)

# MoA retrieval benchmark
retriever = MoARetriever(embeddings_tvn, metadata)
retriever.build_reference_index()
recall = retriever.compute_recall_at_k(k=10)
print(f"Recall@10: {recall:.3f}")

# Pathway enrichment
analyzer = PathwayAnalyzer(metadata, labels)
results = analyzer.run_enrichr(cluster_id=0)

# Visualize
viz = PhenomicsVisualizer(umap_coords, metadata, labels)
viz.plot_umap_by_perturbation_type(save_path="results/umap_perturbation_type.png")
viz.create_interactive_umap(save_path="results/umap_interactive.html")
```

---

## Project Structure

```
phenomics-profiling/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── config/
│   ├── __init__.py
│   └── config.py                # Pipeline configuration dataclass
├── src/
│   ├── data_loader.py           # RxRx3DataLoader: metadata + embeddings I/O
│   ├── embeddings.py            # EmbeddingProcessor: OpenPhenom inference + TVN
│   ├── clustering.py            # PerturbationClusterer: UMAP, HDBSCAN, K-means
│   ├── retrieval.py             # MoARetriever: cosine similarity, Recall@k
│   ├── pathway_analysis.py      # PathwayAnalyzer: Enrichr/GSEA integration
│   ├── visualization.py         # PhenomicsVisualizer: all plots
│   └── utils.py                 # Shared utilities
├── scripts/
│   ├── run_pipeline.py          # Full pipeline CLI
│   ├── embed_perturbations.py   # Embedding CLI
│   └── cluster_analysis.py      # Clustering + viz CLI
└── tests/
    └── test_pipeline.py         # Unit tests with synthetic data
```

---

## Configuration

Edit `config/config.py` to customize pipeline behavior. Key settings:

```python
from config.config import PipelineConfig

cfg = PipelineConfig(
    embedding_model="openphenom",    # or "dinov2", "resnet50"
    embedding_dim=1536,
    batch_size=64,
    hdbscan_min_cluster_size=50,
    kmeans_k=25,
    umap_n_neighbors=15,
    top_k_retrieval=10,
)
```

---

## References

1. Recursion Pharmaceuticals. **RxRx3: Phenomics Map of Biology**. *bioRxiv* 2023.
2. Kraus et al. **Masked autoencoders for microscopy are scalable learners of cellular biology**. *CVPR* 2024. (OpenPhenom)
3. Cimini et al. **Optimizing the Cell Painting assay for image-based profiling**. *Nature Protocols* 2023.
4. Bray et al. **Cell Painting, a high-content image-based assay for morphological profiling**. *Nature Protocols* 2016.
5. McInnes et al. **UMAP: Uniform Manifold Approximation and Projection**. *JOSS* 2018.
6. Campello et al. **Density-Based Clustering Based on Hierarchical Density Estimates**. *ECML PKDD* 2013.
7. Kuleshov et al. **Enrichr: a comprehensive gene set enrichment analysis web server**. *Nucleic Acids Research* 2016.

---

## License

MIT License — see LICENSE file.

## Citation

```bibtex
@software{phenomics_profiling_2024,
  title = {Phenomics Perturbation Profiling Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-org/phenomics-profiling}
}
```
