"""
Configuration for the Phenomics Profiling Pipeline.

Contains all hyperparameters, directory paths, and constants used
throughout the pipeline. Mirrors settings relevant to OpenPhenom ViT-MAE
embeddings and the RxRx3 high-content screening dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Embedding parameters (OpenPhenom ViT-MAE)
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 768

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
N_SYNTHETIC_PERTURBATIONS: int = 300
N_GENES_PER_PERTURBATION: int = 5
N_PLATES: int = 6
WELLS_PER_PLATE: int = 384

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
HDBSCAN_MIN_CLUSTER_SIZE: int = 15
KMEANS_K_RANGE: range = range(3, 12)

# ---------------------------------------------------------------------------
# UMAP / dimensionality reduction
# ---------------------------------------------------------------------------
UMAP_N_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
UMAP_N_COMPONENTS: int = 2
UMAP_METRIC: str = "cosine"

# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------
TOP_K_VALUES: list[int] = [1, 5, 10, 20]

# ---------------------------------------------------------------------------
# Perturbation types
# ---------------------------------------------------------------------------
PERTURBATION_TYPES: list[str] = ["crispr_ko", "compound"]

# ---------------------------------------------------------------------------
# Mechanism-of-Action (MoA) categories
# ---------------------------------------------------------------------------
MOA_CATEGORIES: list[str] = [
    "apoptosis_inducer",
    "cell_cycle_inhibitor",
    "dna_damage_response",
    "kinase_inhibitor",
    "epigenetic_modifier",
    "proteasome_inhibitor",
    "autophagy_modulator",
    "mtor_signaling",
    "mapk_pathway",
    "nf_kb_inhibitor",
]

# ---------------------------------------------------------------------------
# Pathway gene sets (mock, used when gseapy is not available)
# ---------------------------------------------------------------------------
PATHWAY_GENE_SETS = {
    "apoptosis": [
        "BAX",
        "BCL2",
        "CASP3",
        "CASP8",
        "CASP9",
        "CYCS",
        "APAF1",
        "BID",
        "BAK1",
        "XIAP",
        "BIRC5",
        "MCL1",
        "FADD",
        "FAS",
        "TNFRSF10A",
        "TNFRSF10B",
        "DIABLO",
        "PMAIP1",
        "BBC3",
        "BOK",
    ],
    "cell_cycle": [
        "CDK1",
        "CDK2",
        "CDK4",
        "CDK6",
        "CCND1",
        "CCNE1",
        "CCNA2",
        "CCNB1",
        "RB1",
        "TP53",
        "CDKN1A",
        "CDKN2A",
        "E2F1",
        "CDC25A",
        "PLK1",
        "AURKA",
        "AURKB",
        "BUB1",
        "MAD2L1",
        "TTK",
    ],
    "dna_repair": [
        "BRCA1",
        "BRCA2",
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "RAD51",
        "TP53BP1",
        "MDC1",
        "RNF8",
        "PARP1",
        "XRCC1",
        "MLH1",
        "MSH2",
        "MSH6",
        "ERCC1",
        "XPC",
        "XPA",
        "POLB",
        "LIG3",
    ],
    "immune_response": [
        "TNF",
        "IL6",
        "IL1B",
        "IFNG",
        "CXCL8",
        "CCL2",
        "NFKB1",
        "RELA",
        "JAK1",
        "JAK2",
        "STAT1",
        "STAT3",
        "TLR4",
        "MYD88",
        "IRAK4",
        "TRAF6",
        "IRF3",
        "CGAS",
        "STING1",
        "MAVS",
    ],
    "metabolism": [
        "HK2",
        "PKM",
        "LDHA",
        "PDK1",
        "CS",
        "IDH1",
        "IDH2",
        "OGDH",
        "SDHA",
        "SDHB",
        "FH",
        "MDH2",
        "ACLY",
        "FASN",
        "SCD",
        "CPT1A",
        "ACOX1",
        "GLS",
        "SLC1A5",
        "SLC7A11",
    ],
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR: str = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR: str = os.path.join(OUTPUT_DIR, "results")


# ---------------------------------------------------------------------------
# Convenience dataclass that bundles everything
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Immutable snapshot of all pipeline parameters.

    Attributes are available both as lowercase (Pythonic) names and
    uppercase aliases for backward-compatibility with older scripts.
    """

    embedding_dim: int = EMBEDDING_DIM
    n_synthetic_perturbations: int = N_SYNTHETIC_PERTURBATIONS
    n_genes_per_perturbation: int = N_GENES_PER_PERTURBATION
    n_plates: int = N_PLATES
    wells_per_plate: int = WELLS_PER_PLATE
    hdbscan_min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE
    kmeans_k_range: range = KMEANS_K_RANGE
    umap_n_neighbors: int = UMAP_N_NEIGHBORS
    umap_min_dist: float = UMAP_MIN_DIST
    umap_n_components: int = UMAP_N_COMPONENTS
    umap_metric: str = UMAP_METRIC
    top_k_values: list[int] = field(default_factory=lambda: list(TOP_K_VALUES))
    perturbation_types: list[str] = field(default_factory=lambda: list(PERTURBATION_TYPES))
    moa_categories: list[str] = field(default_factory=lambda: list(MOA_CATEGORIES))
    random_seed: int = RANDOM_SEED
    output_dir: str = OUTPUT_DIR
    data_dir: str = DATA_DIR
    figures_dir: str = FIGURES_DIR
    results_dir: str = RESULTS_DIR

    # Uppercase aliases -------------------------------------------------------
    @property
    def EMBEDDING_DIM(self) -> int:  # noqa: N802
        return self.embedding_dim

    @property
    def N_SYNTHETIC_PERTURBATIONS(self) -> int:  # noqa: N802
        return self.n_synthetic_perturbations

    @N_SYNTHETIC_PERTURBATIONS.setter
    def N_SYNTHETIC_PERTURBATIONS(self, value: int) -> None:  # noqa: N802
        self.n_synthetic_perturbations = value

    @property
    def N_GENES_PER_PERTURBATION(self) -> int:  # noqa: N802
        return self.n_genes_per_perturbation

    @property
    def HDBSCAN_MIN_CLUSTER_SIZE(self) -> int:  # noqa: N802
        return self.hdbscan_min_cluster_size

    @property
    def KMEANS_K_RANGE(self) -> range:  # noqa: N802
        return self.kmeans_k_range

    @property
    def UMAP_N_NEIGHBORS(self) -> int:  # noqa: N802
        return self.umap_n_neighbors

    @property
    def UMAP_MIN_DIST(self) -> float:  # noqa: N802
        return self.umap_min_dist

    @property
    def TOP_K_VALUES(self) -> list[int]:  # noqa: N802
        return self.top_k_values

    @property
    def PERTURBATION_TYPES(self) -> list[str]:  # noqa: N802
        return self.perturbation_types

    @property
    def MOA_CATEGORIES(self) -> list[str]:  # noqa: N802
        return self.moa_categories

    @property
    def RANDOM_SEED(self) -> int:  # noqa: N802
        return self.random_seed

    @RANDOM_SEED.setter
    def RANDOM_SEED(self, value: int) -> None:  # noqa: N802
        self.random_seed = value

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        for d in (self.output_dir, self.data_dir, self.figures_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)


def get_config() -> PipelineConfig:
    """Return a default pipeline configuration."""
    return PipelineConfig()
