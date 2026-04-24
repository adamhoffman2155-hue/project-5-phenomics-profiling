"""
Data loading and synthetic data generation for the Phenomics Profiling Pipeline.

Generates realistic synthetic morphological-profiling data that mimics
OpenPhenom ViT-MAE embeddings from the RxRx3 dataset. Each perturbation
gets a 768-dimensional embedding with subtle MoA-driven cluster structure.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger("phenomics.data_loader")


# ------------------------------------------------------------------
# Gene symbol pool
# ------------------------------------------------------------------
_GENE_POOL: list[str] = [
    "TP53",
    "BRCA1",
    "BRCA2",
    "EGFR",
    "KRAS",
    "BRAF",
    "PIK3CA",
    "PTEN",
    "MYC",
    "AKT1",
    "MTOR",
    "CDK4",
    "CDK6",
    "RB1",
    "ATM",
    "ATR",
    "CHEK1",
    "CHEK2",
    "MDM2",
    "CDKN2A",
    "BCL2",
    "BAX",
    "CASP3",
    "CASP8",
    "CASP9",
    "JAK2",
    "STAT3",
    "NFKB1",
    "TNF",
    "IL6",
    "HIF1A",
    "VEGFA",
    "ERBB2",
    "FGFR1",
    "MET",
    "ALK",
    "ROS1",
    "NTRK1",
    "MAP2K1",
    "MAPK1",
    "RAF1",
    "SRC",
    "ABL1",
    "FLT3",
    "KIT",
    "PDGFRA",
    "RET",
    "SMO",
    "NOTCH1",
    "WNT1",
    "CTNNB1",
    "APC",
    "SMAD4",
    "TGFBR1",
    "HDAC1",
    "HDAC2",
    "EZH2",
    "DNMT1",
    "BRD4",
    "CREBBP",
    "EP300",
    "KDM5A",
    "IDH1",
    "IDH2",
    "SDHB",
    "FH",
    "VHL",
    "NF1",
    "NF2",
    "TSC1",
    "TSC2",
    "STK11",
    "FBXW7",
    "ARID1A",
    "SMARCA4",
    "BAP1",
    "POLE",
    "MSH2",
    "MLH1",
    "MSH6",
    "PMS2",
    "RAD51",
    "PALB2",
    "FANCA",
    "FANCD2",
    "BLM",
    "WRN",
    "RECQL4",
    "XRCC1",
    "ERCC1",
    "XPA",
]


# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------


def generate_synthetic_data(
    config: object,
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Create a synthetic dataset that resembles RxRx3 morphological profiles.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.

    Returns
    -------
    metadata : pd.DataFrame
        Columns: perturbation_id, perturbation_type, gene_symbol,
        moa_label, plate_id, well_id
    embeddings : ndarray of shape (n_perturbations, embedding_dim)
        Embedding matrix with MoA-correlated cluster structure.
    """
    rng = np.random.default_rng(config.random_seed)
    n = config.n_synthetic_perturbations
    dim = config.embedding_dim
    n_moa = len(config.moa_categories)

    logger.info(
        "Generating synthetic data: %d perturbations, dim=%d, %d MoA classes",
        n,
        dim,
        n_moa,
    )

    # --- MoA centroids (unit-norm directions in embedding space) ----------
    raw_centroids = rng.standard_normal((n_moa, dim))
    centroid_norms = np.linalg.norm(raw_centroids, axis=1, keepdims=True)
    centroids = raw_centroids / centroid_norms * 3.0  # scale for separation

    # --- Assign each perturbation to an MoA and a type ------------------
    moa_indices = rng.integers(0, n_moa, size=n)
    moa_labels = [config.moa_categories[i] for i in moa_indices]
    perturbation_types = rng.choice(config.perturbation_types, size=n).tolist()

    # --- Build embeddings with cluster structure -------------------------
    noise = rng.standard_normal((n, dim)).astype(np.float64)
    embeddings = centroids[moa_indices] + noise * 0.8

    # Add a small perturbation-type shift so compound / crispr sub-clusters
    type_shift = rng.standard_normal((len(config.perturbation_types), dim)) * 0.3
    type_map = {t: i for i, t in enumerate(config.perturbation_types)}
    for idx in range(n):
        embeddings[idx] += type_shift[type_map[perturbation_types[idx]]]

    # --- Plate and well assignments --------------------------------------
    plate_ids = [f"plate_{i % config.n_plates + 1}" for i in range(n)]
    well_rows = [chr(ord("A") + (i % 16)) for i in range(n)]
    well_cols = [(i % 24) + 1 for i in range(n)]
    well_ids = [f"{r}{c:02d}" for r, c in zip(well_rows, well_cols, strict=False)]

    # --- Gene symbols (multiple per perturbation) ------------------------
    gene_symbols: list[str] = []
    for _i in range(n):
        genes = rng.choice(
            _GENE_POOL,
            size=min(config.n_genes_per_perturbation, len(_GENE_POOL)),
            replace=False,
        )
        gene_symbols.append(";".join(genes))

    # --- Perturbation identifiers ----------------------------------------
    perturbation_ids = [f"PERT-{i:04d}" for i in range(n)]

    metadata = pd.DataFrame(
        {
            "perturbation_id": perturbation_ids,
            "perturbation_type": perturbation_types,
            "gene_symbol": gene_symbols,
            "moa_label": moa_labels,
            "plate_id": plate_ids,
            "well_id": well_ids,
        }
    )

    logger.info("Synthetic data generated \u2014 metadata shape %s", metadata.shape)
    return metadata, embeddings.astype(np.float64)


# ------------------------------------------------------------------
# Placeholder loaders for real data
# ------------------------------------------------------------------


def load_rxrx3_metadata(path: str) -> pd.DataFrame:
    """Load RxRx3 experiment metadata from a CSV or Parquet file.

    Parameters
    ----------
    path : str
        Path to the metadata file.

    Returns
    -------
    pd.DataFrame
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_embeddings(path: str) -> NDArray[np.float64]:
    """Load precomputed embeddings from a .npy file.

    Parameters
    ----------
    path : str
        Path to the ``.npy`` file containing the embedding matrix.

    Returns
    -------
    ndarray of shape (n_samples, embedding_dim)
    """
    return np.load(path).astype(np.float64)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def validate_data(
    metadata: pd.DataFrame,
    embeddings: NDArray[np.float64],
) -> dict[str, bool]:
    """Run sanity checks on metadata and embeddings.

    Parameters
    ----------
    metadata : pd.DataFrame
    embeddings : ndarray

    Returns
    -------
    dict
        Keys are check names; values are ``True`` when the check passes.

    Raises
    ------
    ValueError
        If any critical check fails.
    """
    checks: dict[str, bool] = {}

    # Row count match
    checks["row_count_match"] = len(metadata) == embeddings.shape[0]
    if not checks["row_count_match"]:
        raise ValueError(
            f"Row count mismatch: metadata has {len(metadata)} rows, "
            f"embeddings have {embeddings.shape[0]} rows."
        )

    # Embedding dimensionality
    checks["embedding_is_2d"] = embeddings.ndim == 2
    if not checks["embedding_is_2d"]:
        raise ValueError(f"Embeddings must be 2-D, got ndim={embeddings.ndim}")

    # No NaN / Inf
    checks["no_nan_embeddings"] = not np.isnan(embeddings).any()
    checks["no_inf_embeddings"] = not np.isinf(embeddings).any()
    if not checks["no_nan_embeddings"]:
        raise ValueError("Embeddings contain NaN values.")
    if not checks["no_inf_embeddings"]:
        raise ValueError("Embeddings contain Inf values.")

    # Required columns
    required_cols = {"perturbation_id", "perturbation_type", "moa_label", "plate_id"}
    present_cols = set(metadata.columns)
    checks["required_columns_present"] = required_cols.issubset(present_cols)
    if not checks["required_columns_present"]:
        missing = required_cols - present_cols
        raise ValueError(f"Metadata is missing columns: {missing}")

    # No duplicate perturbation IDs
    checks["unique_perturbation_ids"] = metadata["perturbation_id"].is_unique

    logger.info("Data validation passed: %s", checks)
    return checks
