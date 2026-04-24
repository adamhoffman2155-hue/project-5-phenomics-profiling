# Project 5: Phenomics Perturbation Profiling (RxRx3)

![CI](https://github.com/adamhoffman2155-hue/project-5-phenomics-profiling/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Repro](https://img.shields.io/badge/FAIR_DOME_CURE-11%2F14_%7C_5%2F7_%7C_4%2F4-brightgreen)

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
│   ├── run_retrieval.py             # Retrieval-only
│   └── poc/
│       └── run_poc.py               # Proof-of-concept MoA retrieval benchmark
├── tests/
│   └── test_pipeline.py
├── data/
├── results/
│   └── poc/                         # POC outputs (recall CSV, UMAP, summary)
├── Dockerfile
├── requirements.txt
├── environment.yml
└── LICENSE
```

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-5-phenomics-profiling.git
cd project-5-phenomics-profiling

# Choose one environment
docker build -t phenomics . && docker run -it -v $(pwd):/workspace phenomics bash
#   or
conda env create -f environment.yml && conda activate phenomics-profiling
#   or
pip install -r requirements.txt

# Full pipeline
python scripts/run_pipeline.py

# Individual stages
python scripts/run_clustering.py
python scripts/run_retrieval.py

# Quick proof-of-concept run (~1 min, public S3 only)
python scripts/poc/run_poc.py
```

## Proof of Concept

A lightweight, reproducible benchmark that answers the core scientific question
of this project — *do perturbations with the same mechanism-of-action cluster
together in morphological feature space?* — on fully public data, with no
account required.

**Dataset.** LINCS Cell Painting (`cpg0004-lincs`), batch
`2016_04_01_a549_48hr_batch1`, 8 plates
(`SQ00014812`–`SQ00014819`). Served from the Broad Cell Painting Gallery
on AWS Open Data at
<https://cellpainting-gallery.s3.amazonaws.com/cpg0004-lincs/broad/workspace/profiles/>.
The script pulls the `*_normalized_feature_select.csv.gz` profile files, which
already have CellProfiler-derived morphological features, plate-level
normalization, and MoA annotations (`Metadata_moa`) embedded.

**Substitution note.** This POC substitutes LINCS Cell Painting for
Recursion's [RxRx3-core](https://www.rxrx.ai/rxrx3-core), which the full
project targets. RxRx3-core requires a Hugging Face account; LINCS Cell
Painting is on a public S3 bucket and is the canonical compound-MoA
Cell Painting benchmark. The scientific question — recovering MoA from
morphology via nearest-neighbor retrieval — is identical.

**Pipeline.**
1. Download 8 LINCS plates of normalized, feature-selected morphological profiles
2. Drop DMSO / untreated / unannotated wells
3. Take the intersection of feature columns across plates (241 features retained)
4. Aggregate replicate wells (all doses) to one mean profile per compound
5. Use the first listed MoA (`moa.split("|")[0]`) as the primary label
6. Restrict to MoAs with ≥2 compounds (needed for retrieval)
7. Compute the full cosine-similarity matrix over compound profiles
8. For each query compound, compute recall@K = fraction of its top-K nearest neighbors that share its MoA
9. Compare against a random baseline (random recall@K = `(N_same_MoA − 1) / (N − 1)`)

**Results (actual numbers on the committed output).**

| Metric | Value |
|---|---|
| N wells (post-filter) | 2,808 |
| N features used | 241 |
| N unique compounds | 111 |
| N unique primary MoAs | 85 |
| N benchmarkable compounds (MoA with ≥2 members) | 46 |
| N eligible MoAs (≥2 members) | 20 |

| K | recall@K | random baseline | fold over random |
|---|---|---|---|
| 1  | 0.1304 | 0.0319 | **4.09×** |
| 5  | 0.0957 | 0.0319 | **3.00×** |
| 10 | 0.0587 | 0.0319 | 1.84× |

At K=1 and K=5 cosine NN retrieval beats random by 3–4×, which is
consistent with the "real signal" range specified for untuned morphological
retrieval on small subsets. The drop at K=10 is expected given that most
eligible MoAs in this subset have only 2–3 member compounds, so recall@10
is dominated by unrelated neighbors.

**Top-10 most abundant MoAs in the benchmarked subset**
(N compounds): `adrenergic receptor antagonist` (4), `HCV inhibitor` (3),
`serotonin receptor agonist` (3), `acetylcholine receptor antagonist` (3),
`EGFR inhibitor` (3), `leukotriene receptor antagonist` (2),
`histamine receptor antagonist` (2), `gamma secretase inhibitor` (2),
`ACAT inhibitor` (2), `angiotensin converting enzyme inhibitor` (2).

**Reproduce.**

```bash
pip install pandas numpy scipy scikit-learn matplotlib umap-learn pyarrow
python scripts/poc/run_poc.py
```

Runtime: ~1 minute on a laptop (downloads ~5 MB of plate data, cached in
`data_cache/` after first run).

**Outputs** (under `results/poc/`):
- `recall_at_k.csv` — recall@1, @5, @10 with random baseline and fold ratio
- `poc_summary.txt` — full run summary, including top MoAs and interpretation
- `umap_moa.png` — 2D UMAP of compound profiles, colored by the top-10 MoAs

**Limits (what this POC does *not* do).**
- No batch-effect correction (sphering / TVN / CCA) — all 8 plates are from
  the same batch, which minimizes but does not eliminate plate effects.
- No dose–response modeling: all doses for a compound are collapsed into
  one mean profile. Dose-aware aggregation (e.g. picking the maximally
  active dose) would likely improve recall.
- Only 46 compounds fall into benchmarkable MoA groups on this subset —
  the random baseline is ~3%, so recall numbers are noisy. Scaling to the
  full batch (136 plates) would materially tighten the estimates.
- MoA labels are heterogeneous (many compounds have multi-label MoAs joined
  with `|`); we use the first listed term, which is a lossy simplification.
- No comparison against MoA retrieval with foundation-model embeddings
  (OpenPhenom, DINOv2) — that's the job of the full pipeline.

## My Role

I formulated the RxRx3 MoA retrieval question, evaluated cluster assignments for biological coherence, and reviewed pathway enrichment outputs against known gene function. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 5 of 7**. It steps outside the GEA-specific thread of Projects 1–4 to explore a different data modality (microscopy) and industry-relevant platform (Recursion's phenomics). It demonstrates breadth beyond genomics while maintaining the same pattern: start with a biological question, build the computational tools to answer it. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

### Cross-project positioning

Project-5 is a **parallel phenomics side-arm** — orthogonal to the P1→P3→P4→P6 GEA chain. It demonstrates a different computational-biology modality (high-content imaging with foundation-model embeddings) rather than consuming or feeding into the transcriptomics/DDR/survival projects.

## Benchmarks

| Benchmark | Output | Summary |
| --- | --- | --- |
| Retrieval metric comparison | [`results/benchmark/retrieval_comparison.md`](results/benchmark/retrieval_comparison.md) | Cosine (POC default) dominates at recall@5 = 0.900 over Euclidean (0.717) and Spearman (0.817) on a matched-shape profile space — validates the POC's choice of cosine similarity. |

Rebuild with `python scripts/benchmark_retrieval_metrics.py`.

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the FAIR-BioRS / DOME / CURE self-scorecard (11/14 · 5/7 · 4/4).

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
