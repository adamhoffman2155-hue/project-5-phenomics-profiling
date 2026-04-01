"""
Pipeline configuration for the Phenomics Perturbation Profiling Pipeline.

Defines dataclasses for all configurable parameters: paths, model settings,
clustering, retrieval, UMAP, and pathway enrichment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class PathConfig:
    """File system paths used throughout the pipeline."""

    # Root data directory
    data_dir: str = "data"

    # Raw data
    raw_dir: str = "data/raw"
    image_dir: str = "data/raw/images"
    metadata_csv: str = "data/raw/metadata.csv"

    # Embeddings
    embeddings_dir: str = "data/embeddings"
    embeddings_file: str = "data/embeddings/openphenom_embeddings.npy"
    embedding_index_csv: str = "data/embeddings/embedding_index.csv"
    normalized_embeddings_file: str = "data/embeddings/normalized_embeddings.npy"

    # Results
    results_dir: str = "results"
    clustering_dir: str = "results/clustering"
    retrieval_dir: str = "results/retrieval"
    pathway_dir: str = "results/pathway_enrichment"
    figures_dir: str = "results/figures"

    def make_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        dirs = [
            self.data_dir,
            self.raw_dir,
            self.embeddings_dir,
            self.results_dir,
            self.clustering_dir,
            self.retrieval_dir,
            self.pathway_dir,
            self.figures_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingConfig:
    """Settings for OpenPhenom embedding extraction."""

    # Model
    model_name: str = "openphenom"
    # HuggingFace hub ID for the ViT-MAE model
    model_hub_id: str = "recursionpharma/OpenPhenom"
    # Local cache path for downloaded model weights
    model_cache_dir: str = "models/"
    # Embedding output dimensionality from ViT-L/16 CLS token
    embedding_dim: int = 1536

    # Inference
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"  # "cpu" for CPU-only inference

    # Image preprocessing
    image_size: int = 224  # Resize input to this for ViT
    # Per-channel mean and std for RxRx3 5-channel normalization
    # Approximated from the RxRx3 training distribution
    channel_means: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406, 0.485, 0.456]
    )
    channel_stds: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225, 0.229, 0.224]
    )

    # RxRx3 channel names in order
    channel_names: List[str] = field(
        default_factory=lambda: [
            "DAPI",           # Channel 1: Nucleus / DNA
            "ConA",           # Channel 2: ER / Golgi
            "SYTO14",         # Channel 3: Nucleoli / cytoplasmic RNA
            "WGA+Phalloidin", # Channel 4: Plasma membrane / actin
            "MitoTracker",    # Channel 5: Mitochondria
        ]
    )

    # Aggregation
    # How to aggregate multiple fields of view per well
    fov_aggregation: str = "mean"  # "mean" or "median"
    # How to aggregate multiple replicate wells per perturbation
    replicate_aggregation: str = "mean"

    # Normalization
    # Apply Typical Variation Normalization (TVN) after embedding
    apply_tvn: bool = True
    # Apply L2 (spherical) normalization
    apply_l2_norm: bool = True


@dataclass
class ClusteringConfig:
    """Settings for dimensionality reduction and clustering."""

    # UMAP
    umap_n_components: int = 2          # 2 for visualization, 3 for 3D
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    umap_random_state: int = 42
    umap_low_memory: bool = False        # Set True for >50k points

    # HDBSCAN (density-based, automatic cluster count)
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int = 10
    hdbscan_metric: str = "euclidean"    # Applied on UMAP 2D coords
    hdbscan_cluster_selection_method: str = "eom"  # "eom" or "leaf"

    # K-means
    kmeans_k: int = 25
    kmeans_max_iter: int = 500
    kmeans_n_init: int = 20
    kmeans_random_state: int = 42

    # Agglomerative
    agglomerative_n_clusters: int = 25
    agglomerative_linkage: str = "ward"

    # Elbow method for optimal k selection
    elbow_k_range_min: int = 5
    elbow_k_range_max: int = 50
    elbow_k_step: int = 5


@dataclass
class RetrievalConfig:
    """Settings for MoA retrieval benchmarking."""

    # Top-k nearest neighbors to retrieve
    top_k: int = 10
    # k values for Recall@k evaluation
    recall_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    # Similarity metric for retrieval index
    similarity_metric: str = "cosine"   # "cosine" or "euclidean"
    # Whether to use approximate nearest neighbors (faiss) for large datasets
    use_approximate_nn: bool = False
    # faiss index type if approximate
    faiss_index_type: str = "Flat"      # "Flat", "IVFFlat", "HNSW"

    # Benchmark ground truth
    # MoA group labels column in metadata
    moa_column: str = "moa"
    # Minimum group size to include in benchmark
    min_moa_group_size: int = 2
    # Whether to include cross-modal retrieval (compound ↔ CRISPR KO)
    cross_modal_retrieval: bool = True


@dataclass
class PathwayConfig:
    """Settings for pathway enrichment analysis."""

    # Gene sets to query via Enrichr
    gene_sets: List[str] = field(
        default_factory=lambda: [
            "KEGG_2021_Human",
            "Reactome_2022",
            "GO_Biological_Process_2023",
            "WikiPathway_2023_Human",
        ]
    )
    # Adjusted p-value cutoff for significance
    p_cutoff: float = 0.05
    # Minimum number of genes in a cluster to run enrichment
    min_genes_for_enrichment: int = 5
    # Top N pathways to display per cluster
    top_n_pathways: int = 10
    # Organism
    organism: str = "human"
    # GSEA preranked parameters (for full GSEA mode)
    gsea_min_size: int = 15
    gsea_max_size: int = 500
    gsea_permutation_num: int = 1000


@dataclass
class PipelineConfig:
    """
    Top-level configuration for the Phenomics Perturbation Profiling Pipeline.

    Aggregates all sub-configs and provides convenience properties.

    Example
    -------
    >>> cfg = PipelineConfig()
    >>> cfg.embedding.embedding_dim
    1536
    >>> cfg.clustering.hdbscan_min_cluster_size
    50
    """

    paths: PathConfig = field(default_factory=PathConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pathway: PathwayConfig = field(default_factory=PathwayConfig)

    # Global settings
    random_seed: int = 42
    n_jobs: int = -1            # -1 = use all CPUs
    verbose: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Experiment name for output file naming
    experiment_name: str = "rxrx3_openphenom"

    # Perturbation type labels used in metadata
    crispr_label: str = "CRISPR"
    compound_label: str = "compound"
    control_label: str = "negcon"       # Negative control label

    # Metadata column names (RxRx3 schema)
    perturbation_type_col: str = "perturbation_type"
    gene_col: str = "gene"
    compound_col: str = "compound"
    concentration_col: str = "concentration_um"
    cell_line_col: str = "cell_line"
    plate_col: str = "plate"
    well_col: str = "well"
    site_col: str = "site"
    replicate_col: str = "replicate"

    def __post_init__(self) -> None:
        """Validate config values after initialization."""
        assert self.embedding.embedding_dim > 0, "embedding_dim must be positive"
        assert self.clustering.kmeans_k > 1, "kmeans_k must be >= 2"
        assert self.retrieval.top_k > 0, "top_k must be positive"
        assert 0.0 < self.pathway.p_cutoff < 1.0, "p_cutoff must be in (0, 1)"

    @property
    def model_name(self) -> str:
        """Shortcut to embedding model name."""
        return self.embedding.model_name

    @property
    def embedding_dim(self) -> int:
        """Shortcut to embedding dimension."""
        return self.embedding.embedding_dim

    @property
    def batch_size(self) -> int:
        """Shortcut to inference batch size."""
        return self.embedding.batch_size

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary for logging/JSON export."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Reconstruct PipelineConfig from a plain dictionary."""
        return cls(
            paths=PathConfig(**d.get("paths", {})),
            embedding=EmbeddingConfig(**d.get("embedding", {})),
            clustering=ClusteringConfig(**d.get("clustering", {})),
            retrieval=RetrievalConfig(**d.get("retrieval", {})),
            pathway=PathwayConfig(**d.get("pathway", {})),
            **{k: v for k, v in d.items()
               if k not in {"paths", "embedding", "clustering", "retrieval", "pathway"}},
        )


# ---------------------------------------------------------------------------
# Default singleton instance
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = PipelineConfig()
