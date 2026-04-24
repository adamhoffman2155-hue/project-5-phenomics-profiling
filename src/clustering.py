"""
Clustering and dimensionality reduction for the Phenomics Profiling Pipeline.

Provides UMAP (with t-SNE fallback), HDBSCAN, K-Means, the elbow method,
and standard clustering evaluation metrics.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger("phenomics.clustering")


# ------------------------------------------------------------------
# UMAP / t-SNE dimensionality reduction
# ------------------------------------------------------------------


def run_umap(
    embeddings: NDArray[np.float64],
    n_components: int = 2,
    config: object | None = None,
) -> NDArray[np.float64]:
    """Reduce embedding dimensionality with UMAP (or t-SNE fallback).

    If ``umap-learn`` is installed it will be used. Otherwise the
    function transparently falls back to scikit-learn's t-SNE so the
    rest of the pipeline still works.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, dim)
    n_components : int
        Target dimensionality (default 2).
    config : PipelineConfig, optional
        If provided, UMAP hyper-parameters are read from the config.

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        Low-dimensional coordinates.
    """
    n_neighbors = 15
    min_dist = 0.1
    metric = "cosine"
    seed = 42

    if config is not None:
        n_neighbors = getattr(config, "umap_n_neighbors", n_neighbors)
        min_dist = getattr(config, "umap_min_dist", min_dist)
        metric = getattr(config, "umap_metric", metric)
        seed = getattr(config, "random_seed", seed)

    try:
        import umap  # type: ignore[import-untyped]

        logger.info(
            "Running UMAP (n_neighbors=%d, min_dist=%.2f, metric=%s)",
            n_neighbors,
            min_dist,
            metric,
        )
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        coords = reducer.fit_transform(embeddings)
    except ImportError:
        logger.warning(
            "umap-learn not installed \u2014 falling back to t-SNE "
            "(pip install umap-learn for UMAP support)"
        )
        from sklearn.manifold import TSNE

        perplexity = min(30.0, embeddings.shape[0] - 1)
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            metric="cosine",
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(embeddings)

    logger.info("Dimensionality reduction complete \u2014 output shape %s", coords.shape)
    return coords.astype(np.float64)


# ------------------------------------------------------------------
# HDBSCAN clustering
# ------------------------------------------------------------------


def run_hdbscan(
    embeddings_2d: NDArray[np.float64],
    min_cluster_size: int = 15,
) -> NDArray[np.int64]:
    """Cluster low-dimensional embeddings with HDBSCAN.

    Falls back to DBSCAN from scikit-learn if ``hdbscan`` is not
    installed.

    Parameters
    ----------
    embeddings_2d : ndarray of shape (n_samples, n_features)
    min_cluster_size : int

    Returns
    -------
    ndarray of shape (n_samples,)
        Cluster labels (-1 for noise).
    """
    try:
        import hdbscan as hdb  # type: ignore[import-untyped]

        logger.info("Running HDBSCAN (min_cluster_size=%d)", min_cluster_size)
        clusterer = hdb.HDBSCAN(
            min_cluster_size=min_cluster_size,
            gen_min_span_tree=True,
        )
        labels = clusterer.fit_predict(embeddings_2d)
    except ImportError:
        logger.warning(
            "hdbscan not installed \u2014 falling back to sklearn DBSCAN "
            "(pip install hdbscan for HDBSCAN support)"
        )
        from sklearn.cluster import DBSCAN

        # Estimate eps from the data
        from sklearn.neighbors import NearestNeighbors

        k = min(min_cluster_size, embeddings_2d.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(embeddings_2d)
        distances, _ = nn.kneighbors(embeddings_2d)
        eps = float(np.median(distances[:, -1]) * 1.2)

        clusterer = DBSCAN(eps=eps, min_samples=max(3, min_cluster_size // 3))
        labels = clusterer.fit_predict(embeddings_2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("HDBSCAN/DBSCAN found %d clusters, %d noise points", n_clusters, n_noise)
    return np.asarray(labels, dtype=np.int64)


# ------------------------------------------------------------------
# K-Means clustering
# ------------------------------------------------------------------


def run_kmeans(
    embeddings: NDArray[np.float64],
    k: int,
    random_state: int = 42,
) -> NDArray[np.int64]:
    """Run K-Means clustering.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, dim)
    k : int
        Number of clusters.
    random_state : int

    Returns
    -------
    ndarray of shape (n_samples,)
        Cluster labels (0 .. k-1).
    """
    logger.info("Running K-Means with k=%d", k)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(embeddings)
    logger.info("K-Means inertia=%.2f", km.inertia_)
    return np.asarray(labels, dtype=np.int64)


# ------------------------------------------------------------------
# Elbow method
# ------------------------------------------------------------------


def elbow_method(
    embeddings: NDArray[np.float64],
    k_range: range | None = None,
    random_state: int = 42,
) -> dict[int, float]:
    """Compute K-Means inertia for a range of *k* values.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, dim)
    k_range : range, optional
        Defaults to ``range(2, 13)``.
    random_state : int

    Returns
    -------
    dict mapping k -> inertia
    """
    if k_range is None:
        k_range = range(2, 13)

    inertias: dict[int, float] = {}
    for k in k_range:
        if k > embeddings.shape[0]:
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        km.fit(embeddings)
        inertias[k] = float(km.inertia_)
        logger.debug("Elbow: k=%d  inertia=%.2f", k, km.inertia_)

    logger.info("Elbow method computed for k in %s", list(inertias.keys()))
    return inertias


# ------------------------------------------------------------------
# Clustering evaluation
# ------------------------------------------------------------------


def evaluate_clustering(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> dict[str, float]:
    """Evaluate clustering quality with standard metrics.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, dim)
    labels : ndarray of shape (n_samples,)
        Cluster assignments (noise label -1 is excluded from metrics).

    Returns
    -------
    dict with keys:
        - n_clusters: number of clusters (excluding noise)
        - n_noise: number of noise points
        - silhouette: silhouette coefficient
        - davies_bouldin: Davies-Bouldin index
        - calinski_harabasz: Calinski-Harabasz index
    """
    valid = labels >= 0
    n_clusters = len(set(labels[valid]))
    n_noise = int((~valid).sum())

    metrics: dict[str, float] = {
        "n_clusters": float(n_clusters),
        "n_noise": float(n_noise),
    }

    if n_clusters < 2:
        logger.warning("Fewer than 2 clusters (%d) \u2014 skipping evaluation metrics.", n_clusters)
        metrics.update(
            {"silhouette": 0.0, "davies_bouldin": float("inf"), "calinski_harabasz": 0.0}
        )
        return metrics

    emb_valid = embeddings[valid]
    lab_valid = labels[valid]

    metrics["silhouette"] = float(silhouette_score(emb_valid, lab_valid))
    metrics["davies_bouldin"] = float(davies_bouldin_score(emb_valid, lab_valid))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(emb_valid, lab_valid))

    logger.info("Clustering metrics: %s", metrics)
    return metrics
