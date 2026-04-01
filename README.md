# Project 5: Phenomics Perturbation Profiling (RxRx3)

**Research question:** Can foundation model embeddings map phenotypic similarity across perturbations?

This is the fifth project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio). After Projects 1–4 focused on genomics and transcriptomics, this project explores a different modality — high-content microscopy — using Recursion's public RxRx3-core dataset and OpenPhenom foundation model embeddings. It represents a deliberate step toward TechBio phenomics, a growing area where computational biology meets drug discovery at scale.

## What It Does

Analyzes perturbation phenotypes from high-content cell painting microscopy:

1. **Data loading** — RxRx3-core metadata and pre-computed OpenPhenom embeddings
2. **Embedding processing** — TVN normalization, quality metrics, dimensionality checks
3. **Clustering** — UMAP visualization + HDBSCAN and K-means clustering of CRISPR KO and compound perturbations
4. **MoA retrieval** — Cosine similarity-based mechanism-of-action retrieval with Recall@k and MAP evaluation
5. **Pathway enrichment** — Reactome/GO enrichment via gseapy for cluster interpretation
6. **Visualization** — Interactive Plotly UMAP, pathway-cluster heatmaps, retrieval performance plots

## Methods & Tools

| Category | Tools |
|----------|-------|
| Foundation Model | OpenPhenom (ViT-MAE) embeddings |
| Clustering | HDBSCAN, K-means, agglomerative |
| Dimensionality Reduction | UMAP (2D/3D) |
| Retrieval | Cosine similarity, Recall@k, MAP |
| Enrichment | gseapy (Enrichr, Reactome) |
| Visualization | matplotlib, seaborn, Plotly |
| Environment | Docker, Conda |

## Project Structure

```
project-5-phenomics-profiling/
├── config/
│   └── config.py                    # Dataset paths, model params, gene sets
├── src/
│   ├── data_loader.py               # RxRx3 metadata + embedding loader
│   ├── embeddings.py                # OpenPhenom processing, TVN normalization
│   ├── clustering.py                # UMAP, HDBSCAN, K-means, evaluation
│   ├── retrieval.py                 # Cosine MoA retrieval, Recall@k, MAP
│   ├── pathway_analysis.py          # gseapy enrichment, Jaccard similarity
│   ├── visualization.py             # 8 plot types including interactive UMAP
│   └── utils.py                     # Batched cosine similarity, caching
├── scripts/
│   ├── run_pipeline.py              # Full pipeline CLI
│   ├── run_clustering.py            # Clustering-only
│   └── run_retrieval.py             # Retrieval-only
├── tests/
│   └── test_pipeline.py
├── data/
├── results/
├── requirements.txt
├── environment.yml
└── LICENSE
```

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-5-phenomics-profiling.git
cd project-5-phenomics-profiling

pip install -r requirements.txt
python scripts/run_pipeline.py
```

## My Role

I formulated the RxRx3 MoA retrieval question, evaluated cluster assignments for biological coherence, and reviewed pathway enrichment outputs against known gene function. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 5 of 7**. It steps outside the GEA-specific thread of Projects 1–4 to explore a different data modality (microscopy) and industry-relevant platform (Recursion's phenomics). It demonstrates breadth beyond genomics while maintaining the same pattern: start with a biological question, build the computational tools to answer it. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
