"""
Embedding processing and quality assessment for the Phenomics Profiling Pipeline.

Implements Typical Variation Normalization (TVN), embedding quality metrics,
PCA variance analysis, and batch-effect assessment.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger("phenomics.embeddings")


# ------------------------------------------------------------------
# Typical Variation Normalization (TVN)
# ------------------------------------------------------------------


def tvn_normalize(
    embeddings: NDArray[np.float64],
    plate_ids: list[str] | NDArray | Sequence[str],
) -> NDArray[np.float64]:
    """Apply Typical Variation Normalization per plate.

    TVN whitens each plate's embeddings to remove plate-specific
    technical variation while preserving biological signal. For each
    plate the procedure is:

    1. Center by subtracting the plate mean.
    2. Compute the plate covariance and its inverse square root.
    3. Project the centered embeddings through the whitening transform.

    After per-plate whitening the full matrix is re-centered to zero mean.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, dim)
        Raw embedding matrix.
    plate_ids : list of str
        Plate identifier for each sample (same length as *embeddings*).

    Returns
    -------
    ndarray of shape (n_samples, dim)
        TVN-normalised embeddings.
    """
    if len(plate_ids) != embeddings.shape[0]:
        raise ValueError(
            f"plate_ids length ({len(plate_ids)}) != embeddings rows ({embeddings.shape[0]})"
        )

    unique_plates = sorted(set(plate_ids))
    plate_arr = np.array(plate_ids)
    normalized = np.empty_like(embeddings)

    for plate in unique_plates:
        mask = plate_arr == plate
        plate_emb = embeddings[mask]

        if plate_emb.shape[0] < 2:
            # Cannot whiten a single sample; just center globally later
            normalized[mask] = plate_emb
            continue

        # Center
        plate_mean = plate_emb.mean(axis=0)
        centered = plate_emb - plate_mean

        # Covariance and regularised inverse square root
        cov = np.cov(centered, rowvar=False)
        # Regularise for numerical stability
        reg = 1e-6 * np.eye(cov.shape[0])
        cov += reg

        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Clamp small eigenvalues
        eigvals = np.maximum(eigvals, 1e-10)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        whitened = centered @ inv_sqrt
        normalized[mask] = whitened

        logger.debug("TVN applied to plate %s (%d samples)", plate, mask.sum())

    # Global re-centering
    normalized -= normalized.mean(axis=0)

    logger.info(
        "TVN normalisation complete for %d plates, %d samples",
        len(unique_plates),
        embeddings.shape[0],
    )
    return normalized


# ------------------------------------------------------------------
# Embedding quality metrics
# ------------------------------------------------------------------


def compute_embedding_quality(
    embeddings: NDArray[np.float64],
) -> dict[str, float]:
    """Compute summary quality metrics for an embedding matrix.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, dim)

    Returns
    -------
    dict with keys:
        - mean_norm: average L2 norm of embedding vectors
        - std_norm: standard deviation of L2 norms
        - total_variance: sum of per-feature variances
        - sparsity: fraction of near-zero entries (|x| < 0.01)
        - effective_dimensionality: participation ratio of PCA eigenvalues
    """
    norms = np.linalg.norm(embeddings, axis=1)
    variances = np.var(embeddings, axis=0)
    total_var = float(variances.sum())

    # Sparsity
    sparsity = float((np.abs(embeddings) < 0.01).mean())

    # Effective dimensionality via participation ratio
    cov = np.cov(embeddings, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    eigvals_norm = eigvals / (eigvals.sum() + 1e-12)
    participation_ratio = float(1.0 / (np.sum(eigvals_norm**2) + 1e-12))

    mean_var = float(variances.mean())

    quality = {
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "total_variance": total_var,
        "mean_variance": mean_var,
        "sparsity": sparsity,
        "effective_dimensionality": participation_ratio,
    }

    logger.info("Embedding quality: %s", quality)
    return quality


# ------------------------------------------------------------------
# PCA variance explained
# ------------------------------------------------------------------


def pca_variance_explained(
    embeddings: NDArray[np.float64],
    n_components: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute cumulative variance explained by the top PCA components.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, dim)
    n_components : int
        Number of components to report.

    Returns
    -------
    individual : ndarray of shape (n_components,)
        Fraction of variance explained by each component.
    cumulative : ndarray of shape (n_components,)
        Cumulative fraction of variance explained.
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings)

    individual = pca.explained_variance_ratio_
    cumulative = np.cumsum(individual)

    logger.info(
        "PCA: top-%d components explain %.1f%% variance",
        n_components,
        cumulative[-1] * 100,
    )
    return individual, cumulative


# ------------------------------------------------------------------
# Batch-effect assessment
# ------------------------------------------------------------------


def assess_batch_effects(
    embeddings: NDArray[np.float64],
    plate_ids: list[str] | NDArray | Sequence[str],
) -> dict[str, float]:
    """Quantify batch effects using plate identity as batch label.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, dim)
    plate_ids : list of str

    Returns
    -------
    dict with keys:
        - silhouette_by_plate: silhouette score treating plates as clusters
          (high = strong batch effect)
        - kruskal_wallis_p: p-value from Kruskal-Wallis test on PC1 by plate
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    unique_plates = list(set(plate_ids))

    if len(unique_plates) < 2:
        logger.warning("Only one plate found; cannot assess batch effects.")
        return {"silhouette_by_plate": 0.0, "kruskal_wallis_p": 1.0}

    plate_labels = np.array(plate_ids)

    # Use PCA to reduce before silhouette (faster, more stable)
    n_comp = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(embeddings)

    sil = silhouette_score(
        reduced,
        plate_labels,
        metric="euclidean",
        sample_size=min(5000, len(plate_labels)),
        random_state=42,
    )

    # Kruskal-Wallis on first PC
    pc1 = reduced[:, 0]
    groups = [pc1[plate_labels == p] for p in unique_plates]
    if all(len(g) >= 1 for g in groups):
        stat, pval = stats.kruskal(*groups)
    else:
        pval = 1.0

    results = {
        "silhouette_by_plate": float(sil),
        "kruskal_wallis_p": float(pval),
    }
    logger.info("Batch effect assessment: %s", results)
    return results
