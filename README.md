# Project 5: Phenomics Perturbation Profiling (RxRx3)

> **A pipeline that uses microscope images of drug-treated cells to recover what each drug does — without ever looking at the drug's chemistry.**

## The short version

**What this project does.** Takes thousands of microscope images of cells that have been treated with different drugs (or had different genes knocked out), extracts numerical "fingerprints" from the images, and checks whether drugs with the same mechanism of action look similar to each other.

**The question behind it.** Companies like Recursion and Insitro run enormous Cell Painting experiments and use the images to find new drugs and understand existing ones. The core assumption is that if two compounds hit the same biological target, the cells they treat will look similar under a microscope. Does this actually work?

**What the proof-of-concept shows.** On 46 drugs across 20 mechanism classes from the public LINCS Cell Painting dataset, nearest-neighbor matching finds a same-mechanism drug as the top hit **13% of the time** — about **4× better than random** (random would be 3.2%). At the top-5 hits, recall is 9.6% — about 3× random. Morphological features carry real mechanism signal, even with untuned cosine similarity.

**Why it matters.** Phenomics (morphology-based screening) is a real way to find drugs at scale. This POC shows the core assumption — "same mechanism → same look" — isn't hand-waving. With foundation-model embeddings (the next step, shown as the full-project target), results typically improve by another 2-3×.

---

_The rest of this README is technical detail for bioinformaticians, recruiters doing a deep review, or anyone reproducing the work._

## At a Glance

| | |
|---|---|
| **Stack** | OpenPhenom (ViT-MAE) · UMAP · HDBSCAN · K-means · gseapy · Docker |
| **Data** | RxRx3-core (full-pipeline target, HF-gated); LINCS Cell Painting cpg0004 (POC — public S3) |
| **POC headline** | recall@1 = 0.13 (4.1× random); recall@5 = 0.096 (3.0× random); 46 compounds × 20 MoAs |
| **Status** | POC: **Runnable POC** on LINCS (public S3). Full pipeline: **Needs data access** (RxRx3 via Hugging Face) |
| **Role** | RxRx3 MoA retrieval framing; biological review of clusters and pathway enrichments; implementation AI-assisted |
| **Portfolio** | Project 5 of 7 · [full narrative](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) |

## What It Does

Analyzes perturbation phenotypes from high-content cell painting microscopy:

1. **Data loading** — RxRx3-core metadata and pre-computed OpenPhenom embeddings (full pipeline) / LINCS Cell Painting profiles (POC)
2. **Embedding processing** — TVN normalization, quality metrics, dimensionality checks
3. **Clustering** — UMAP visualization + HDBSCAN and K-means clustering of CRISPR KO and compound perturbations
4. **MoA retrieval** — Cosine similarity-based mechanism-of-action retrieval with Recall@k evaluation
5. **Pathway enrichment** — Reactome/GO enrichment via gseapy for cluster interpretation
6. **Visualization** — Interactive Plotly UMAP, pathway-cluster heatmaps, retrieval performance plots

## Methods & Tools

| Category | Tools |
|----------|-------|
| Foundation Model | OpenPhenom (ViT-MAE) embeddings (full pipeline) |
| Clustering | HDBSCAN, K-means, agglomerative |
| Dimensionality Reduction | UMAP (2D/3D) |
| Retrieval | Cosine similarity, Recall@k |
| Enrichment | gseapy (Enrichr, Reactome) |
| Visualization | matplotlib, seaborn, Plotly |
| Environment | Docker, Conda |

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-5-phenomics-profiling.git
cd project-5-phenomics-profiling

pip install -r requirements.txt
python scripts/run_pipeline.py
```

## Proof of Concept

A lightweight, reproducible MoA retrieval benchmark on fully public data, no account required.

**Dataset.** LINCS Cell Painting (`cpg0004-lincs`), batch `2016_04_01_a549_48hr_batch1`, 8 plates served from the Broad Cell Painting Gallery on AWS Open Data. CellProfiler-derived morphological features with MoA annotations are embedded in the profiles.

**Substitution note.** This POC substitutes LINCS Cell Painting for Recursion's [RxRx3-core](https://www.rxrx.ai/rxrx3-core), which requires a Hugging Face account. LINCS is on a public S3 bucket and is the canonical compound-MoA Cell Painting benchmark. The scientific question — recovering MoA from morphology via nearest-neighbor retrieval — is identical.

### Pipeline

1. Download 8 LINCS plates of normalized, feature-selected profiles
2. Drop DMSO / untreated / unannotated wells
3. Take feature-column intersection across plates (241 features retained)
4. Aggregate replicate wells (all doses) to one mean profile per compound
5. First listed MoA (`moa.split("|")[0]`) = primary label
6. Restrict to MoAs with ≥2 compounds
7. Compute cosine-similarity matrix over compound profiles
8. Compute recall@K = fraction of top-K neighbors that share MoA
9. Compare against random baseline

### Results

| Metric | Value |
|---|---|
| N wells (post-filter) | 2,808 |
| N features used | 241 |
| N unique compounds | 111 |
| N unique primary MoAs | 85 |
| N benchmarkable compounds (MoA ≥2) | 46 |
| N eligible MoAs (≥2) | 20 |

| K | recall@K | random baseline | fold over random |
|---|---|---|---|
| 1  | 0.1304 | 0.0319 | **4.09×** |
| 5  | 0.0957 | 0.0319 | **3.00×** |
| 10 | 0.0587 | 0.0319 | 1.84× |

At K=1 and K=5, cosine NN retrieval beats random by 3–4× — consistent with the "real signal" range for untuned morphological retrieval on small subsets.

### Reproduce

```bash
pip install pandas numpy scipy scikit-learn matplotlib umap-learn pyarrow
python scripts/poc/run_poc.py
```

Runtime: ~1 minute on a laptop (downloads ~5 MB of plate data, cached in `data_cache/` after first run).

### Limits

- No batch-effect correction (sphering / TVN / CCA) — all 8 plates are from the same batch, which minimizes but does not eliminate plate effects.
- No dose–response modeling: all doses collapsed into one mean profile. Dose-aware aggregation would likely improve recall.
- Only 46 compounds fall into benchmarkable MoA groups on this subset.
- MoA labels are heterogeneous (multi-label MoAs joined with `|`); we use the first listed term.
- No comparison against foundation-model embeddings yet — that's the full-pipeline target.

## My Role

I formulated the RxRx3 MoA retrieval question, evaluated cluster assignments for biological coherence, and reviewed pathway enrichment outputs against known gene function. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 5 of 7**. It steps outside the GEA-specific thread of Projects 1–4 to explore a different data modality (microscopy) and industry-relevant platform (Recursion's phenomics). It demonstrates breadth beyond genomics while maintaining the same pattern: start with a biological question, build the computational tools to answer it. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
