"""
Phenomics Perturbation Profiling Pipeline — source package.

Submodules
----------
data_loader     : RxRx3DataLoader — metadata and embedding I/O
embeddings      : EmbeddingProcessor — OpenPhenom inference + TVN
clustering      : PerturbationClusterer — UMAP, HDBSCAN, K-means
retrieval       : MoARetriever — cosine similarity, Recall@k, MAP
pathway_analysis: PathwayAnalyzer — Enrichr / GSEA pathway enrichment
visualization   : PhenomicsVisualizer — all plots and figures
utils           : shared utility functions
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("phenomics-profiling")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = [
    "data_loader",
    "embeddings",
    "clustering",
    "retrieval",
    "pathway_analysis",
    "visualization",
    "utils",
]
