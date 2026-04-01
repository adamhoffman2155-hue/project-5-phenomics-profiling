"""
PerturbationClusterer — dimensionality reduction and phenotypic clustering.

Implements:
- UMAP (2D / 3D) dimensionality reduction
- HDBSCAN density-based clustering
- K-means clustering with elbow-method k selection
- Agglomerative clustering
- Clustering evaluation (silhouette, Davies-Bouldin)
- Cluster labeling by pathway/gene enrichment
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class PerturbationClusterer:
    """
    Clusters phenomic perturbation embeddings using UMAP + HDBSCAN/K-means.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration. Uses sensible defaults if None.

    Attributes
    ----------
    umap_2d_ : np.ndarray, shape (N, 2)
        2D UMAP coordinates after run_umap().
    umap_3d_ : np.ndarray, shape (N, 3)
        3D UMAP coordinates after run_umap(n_components=3).
    hdbscan_labels_ : np.ndarray, shape (N,)
        HDBSCAN cluster labels (-1 = noise) after run_hdbscan().
    kmeans_labels_ : np.ndarray, shape (N,)
        K-means cluster labels after run_kmeans().

    Examples
    --------
    >>> clusterer = PerturbationClusterer()
    >>> umap_coords = clusterer.run_umap(embeddings)
    >>> labels = clusterer.run_hdbscan(umap_coords)
    >>> scores = clusterer.evaluate_clustering(umap_coords, labels)
    """

    def __init__(self, config=None):
        self.config = config

        # UMAP params
        cfg_c = config.clustering if config is not None else None
        self._umap_n_neighbors = cfg_c.umap_n_neighbors if cfg_c else 15
        self._umap_min_dist = cfg_c.umap_min_dist if cfg_c else 0.1
        self._umap_metric = cfg_c.umap_metric if cfg_c else "cosine"
        self._umap_random_state = cfg_c.umap_random_state if cfg_c else 42

        # HDBSCAN params
        self._hdbscan_min_cluster_size = cfg_c.hdbscan_min_cluster_size if cfg_c else 50
        self._hdbscan_min_samples = cfg_c.hdbscan_min_samples if cfg_c else 10
        self._hdbscan_metric = cfg_c.hdbscan_metric if cfg_c else "euclidean"
        self._hdbscan_method = (
            cfg_c.hdbscan_cluster_selection_method if cfg_c else "eom"
        )

        # K-means params
        self._kmeans_k = cfg_c.kmeans_k if cfg_c else 25
        self._kmeans_random_state = cfg_c.kmeans_random_state if cfg_c else 42
        self._kmeans_n_init = cfg_c.kmeans_n_init if cfg_c else 20

        # Agglomerative params
        self._agglom_n_clusters = cfg_c.agglomerative_n_clusters if cfg_c else 25
        self._agglom_linkage = cfg_c.agglomerative_linkage if cfg_c else "ward"

        # Elbow method range
        self._elbow_k_min = cfg_c.elbow_k_range_min if cfg_c else 5
        self._elbow_k_max = cfg_c.elbow_k_range_max if cfg_c else 50
        self._elbow_k_step = cfg_c.elbow_k_step if cfg_c else 5

        # Results storage
        self.umap_2d_: Optional[np.ndarray] = None
        self.umap_3d_: Optional[np.ndarray] = None
        self.hdbscan_labels_: Optional[np.ndarray] = None
        self.kmeans_labels_: Optional[np.ndarray] = None
        self.agglom_labels_: Optional[np.ndarray] = None
        self._wcss_by_k: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------

    def run_umap(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: Optional[int] = None,
        min_dist: Optional[float] = None,
        metric: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reduce high-dimensional embeddings to 2D or 3D using UMAP.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
            L2-normalized embedding matrix.
        n_components : int
            Output dimensionality (2 for visualization, 3 for 3D plots).
        n_neighbors, min_dist, metric, random_state
            UMAP hyperparameters. Defaults from config.

        Returns
        -------
        np.ndarray, shape (N, n_components)
            Low-dimensional UMAP coordinates.
        """
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP. Install with: pip install umap-learn"
            )

        n_neighbors = n_neighbors or self._umap_n_neighbors
        min_dist = min_dist if min_dist is not None else self._umap_min_dist
        metric = metric or self._umap_metric
        random_state = random_state if random_state is not None else self._umap_random_state

        # Ensure float32 and L2-normalized for cosine metric
        embs = embeddings.astype(np.float32)
        if metric == "cosine":
            embs = normalize(embs, norm="l2")

        logger.info(
            f"Running UMAP: n={len(embs)}, D={embs.shape[1]} → {n_components}D | "
            f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
        )

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            low_memory=len(embs) > 50_000,
            verbose=False,
        )
        coords = reducer.fit_transform(embs).astype(np.float32)

        if n_components == 2:
            self.umap_2d_ = coords
            logger.info(f"UMAP 2D complete. Shape: {coords.shape}")
        elif n_components == 3:
            self.umap_3d_ = coords
            logger.info(f"UMAP 3D complete. Shape: {coords.shape}")

        return coords

    # ------------------------------------------------------------------
    # Clustering methods
    # ------------------------------------------------------------------

    def run_hdbscan(
        self,
        coords: np.ndarray,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        metric: Optional[str] = None,
        cluster_selection_method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Cluster UMAP coordinates with HDBSCAN (density-based, variable cluster count).

        HDBSCAN is preferred for phenomics data because it:
        - Automatically determines the number of clusters
        - Handles clusters of varying density
        - Labels low-confidence points as noise (-1)

        Parameters
        ----------
        coords : np.ndarray, shape (N, 2) or (N, D)
            Typically UMAP 2D coordinates, but can be used on full embeddings.
        min_cluster_size : int, optional
        min_samples : int, optional
        metric : str, optional
        cluster_selection_method : str, optional

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Cluster labels. -1 = noise / unclustered.
        """
        try:
            import hdbscan as hdbscan_lib
        except ImportError:
            raise ImportError(
                "hdbscan is required. Install with: pip install hdbscan"
            )

        min_cluster_size = min_cluster_size or self._hdbscan_min_cluster_size
        min_samples = min_samples or self._hdbscan_min_samples
        metric = metric or self._hdbscan_metric
        cluster_selection_method = cluster_selection_method or self._hdbscan_method

        # Adaptive min_cluster_size based on dataset size
        if len(coords) < 200 and min_cluster_size > 10:
            min_cluster_size = max(5, len(coords) // 20)
            logger.info(
                f"Small dataset ({len(coords)} pts): "
                f"adjusting min_cluster_size to {min_cluster_size}"
            )

        logger.info(
            f"Running HDBSCAN: min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, metric={metric}"
        )

        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(coords)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info(
            f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points "
            f"({100 * n_noise / len(labels):.1f}%)"
        )

        self.hdbscan_labels_ = labels
        return labels

    def run_kmeans(
        self,
        embeddings: np.ndarray,
        k: Optional[int] = None,
        random_state: Optional[int] = None,
        n_init: Optional[int] = None,
    ) -> np.ndarray:
        """
        Cluster embeddings with K-means.

        Applied directly to the normalized high-dimensional embeddings
        (not UMAP coordinates) for best partition quality.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        k : int, optional
            Number of clusters. Uses config value if None.
        random_state, n_init : optional

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Cluster labels in [0, k).
        """
        k = k or self._kmeans_k
        random_state = random_state if random_state is not None else self._kmeans_random_state
        n_init = n_init or self._kmeans_n_init

        # Ensure L2-normalized for cosine geometry
        embs = normalize(embeddings.astype(np.float32), norm="l2")

        logger.info(f"Running K-means: k={k}, n_init={n_init}")

        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=500,
        )
        labels = km.fit_predict(embs)

        # Store WCSS for elbow method
        self._wcss_by_k[k] = float(km.inertia_)

        logger.info(
            f"K-means done: k={k}, inertia={km.inertia_:.2f}, "
            f"iterations={km.n_iter_}"
        )
        self.kmeans_labels_ = labels
        return labels

    def run_agglomerative(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None,
        linkage: Optional[str] = None,
    ) -> np.ndarray:
        """
        Cluster with agglomerative hierarchical clustering.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        n_clusters : int, optional
        linkage : str, optional
            "ward", "complete", "average", or "single".

        Returns
        -------
        np.ndarray, shape (N,), dtype int
        """
        n_clusters = n_clusters or self._agglom_n_clusters
        linkage = linkage or self._agglom_linkage

        embs = normalize(embeddings.astype(np.float32), norm="l2")
        logger.info(
            f"Running Agglomerative: n_clusters={n_clusters}, linkage={linkage}"
        )

        agglom = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
        labels = agglom.fit_predict(embs)
        self.agglom_labels_ = labels

        logger.info(f"Agglomerative clustering done: {n_clusters} clusters")
        return labels

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_clustering(
        self,
        embeddings_or_coords: np.ndarray,
        labels: np.ndarray,
        sample_size: int = 5000,
    ) -> Dict[str, float]:
        """
        Evaluate clustering quality with multiple metrics.

        Metrics
        -------
        - silhouette_score : Mean silhouette coefficient [-1, 1], higher is better
        - davies_bouldin_score : Lower is better (0 = perfect)
        - n_clusters : Number of identified clusters (excluding noise)
        - noise_fraction : Fraction of noise points (HDBSCAN)
        - calinski_harabasz_score : Higher is better (variance ratio)

        Parameters
        ----------
        embeddings_or_coords : np.ndarray, shape (N, D)
            Can be UMAP coords or full embeddings.
        labels : np.ndarray, shape (N,)
            Cluster labels. -1 labels (noise) are excluded from evaluation.
        sample_size : int
            Subsample size for silhouette (expensive for large N).

        Returns
        -------
        dict
        """
        from sklearn.metrics import calinski_harabasz_score

        # Exclude noise points
        valid_mask = labels >= 0
        if valid_mask.sum() < 10:
            logger.warning("Too few non-noise points for clustering evaluation.")
            return {"n_clusters": 0, "noise_fraction": float((~valid_mask).mean())}

        X = embeddings_or_coords[valid_mask].astype(np.float32)
        y = labels[valid_mask]

        unique_labels = np.unique(y)
        n_clusters = len(unique_labels)
        noise_fraction = float((~valid_mask).mean())

        scores: Dict[str, float] = {
            "n_clusters": n_clusters,
            "noise_fraction": noise_fraction,
            "total_points": len(labels),
            "clustered_points": int(valid_mask.sum()),
        }

        if n_clusters < 2:
            logger.warning("Need >= 2 clusters for quality metrics.")
            return scores

        # Subsample for silhouette (O(N^2) memory)
        rng = np.random.default_rng(42)
        if len(X) > sample_size:
            idx = rng.choice(len(X), size=sample_size, replace=False)
            X_s, y_s = X[idx], y[idx]
            # Ensure all clusters represented
            if len(np.unique(y_s)) < 2:
                X_s, y_s = X, y
        else:
            X_s, y_s = X, y

        try:
            scores["silhouette_score"] = float(
                silhouette_score(X_s, y_s, metric="euclidean", random_state=42)
            )
        except Exception as e:
            logger.warning(f"Silhouette score failed: {e}")

        try:
            scores["davies_bouldin_score"] = float(davies_bouldin_score(X, y))
        except Exception as e:
            logger.warning(f"Davies-Bouldin score failed: {e}")

        try:
            scores["calinski_harabasz_score"] = float(
                calinski_harabasz_score(X, y)
            )
        except Exception as e:
            logger.warning(f"Calinski-Harabasz score failed: {e}")

        logger.info(
            f"Clustering evaluation: n_clusters={n_clusters}, "
            f"noise={noise_fraction:.3f}, "
            f"silhouette={scores.get('silhouette_score', 'N/A'):.3f}, "
            f"DB={scores.get('davies_bouldin_score', 'N/A'):.3f}"
        )
        return scores

    def find_optimal_k(
        self,
        embeddings: np.ndarray,
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        k_step: Optional[int] = None,
        plot: bool = False,
        save_path: Optional[str] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal K for K-means via the elbow method.

        Fits K-means for a range of k values and identifies the elbow
        point in the within-cluster sum of squares (WCSS) curve using
        the maximum second-derivative (kneedle) heuristic.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        k_min, k_max, k_step : optional
            Range of k to evaluate.
        plot : bool
            Plot the elbow curve.
        save_path : str, optional
            Path to save the plot.

        Returns
        -------
        optimal_k : int
        wcss_by_k : dict mapping k → WCSS
        """
        k_min = k_min or self._elbow_k_min
        k_max = k_max or self._elbow_k_max
        k_step = k_step or self._elbow_k_step

        k_values = list(range(k_min, k_max + 1, k_step))
        wcss_values: Dict[int, float] = {}

        embs = normalize(embeddings.astype(np.float32), norm="l2")
        logger.info(f"Elbow method: evaluating k ∈ {k_values}")

        for k in k_values:
            km = KMeans(
                n_clusters=k,
                n_init=10,
                max_iter=300,
                random_state=self._kmeans_random_state,
            )
            km.fit(embs)
            wcss_values[k] = float(km.inertia_)
            logger.debug(f"  k={k}: WCSS={km.inertia_:.2f}")

        self._wcss_by_k.update(wcss_values)

        # Elbow detection: maximum second derivative
        k_arr = np.array(k_values)
        wcss_arr = np.array([wcss_values[k] for k in k_values])

        # Normalize to [0, 1]
        wcss_norm = (wcss_arr - wcss_arr.min()) / (wcss_arr.max() - wcss_arr.min() + 1e-9)

        if len(k_values) >= 3:
            second_deriv = np.diff(np.diff(wcss_norm))
            elbow_idx = int(np.argmax(second_deriv)) + 1  # +1 for double diff offset
            optimal_k = int(k_arr[elbow_idx])
        else:
            optimal_k = k_values[len(k_values) // 2]

        logger.info(f"Elbow method: optimal k = {optimal_k}")

        if plot or save_path:
            self._plot_elbow_curve(k_values, wcss_values, optimal_k, save_path)

        return optimal_k, wcss_values

    def _plot_elbow_curve(
        self,
        k_values: List[int],
        wcss_values: Dict[int, float],
        optimal_k: int,
        save_path: Optional[str],
    ) -> None:
        """Plot and optionally save the K-means elbow curve."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Skipping elbow plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        wcss_list = [wcss_values[k] for k in k_values]
        ax.plot(k_values, wcss_list, "bo-", linewidth=2, markersize=6)
        ax.axvline(optimal_k, color="red", linestyle="--", label=f"Elbow k={optimal_k}")
        ax.set_xlabel("Number of clusters (k)", fontsize=12)
        ax.set_ylabel("Within-cluster sum of squares (WCSS)", fontsize=12)
        ax.set_title("K-means Elbow Method — RxRx3-core Embeddings", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Elbow plot saved to {save_path}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Cluster labeling
    # ------------------------------------------------------------------

    def label_clusters_by_enrichment(
        self,
        labels: np.ndarray,
        metadata: pd.DataFrame,
        gene_col: str = "gene",
        pathway_col: Optional[str] = None,
        top_n: int = 3,
    ) -> Dict[int, List[str]]:
        """
        Label each cluster by the most frequent gene targets or pathway annotations.

        For CRISPR KO clusters, the label is derived from the most common
        gene symbols. For compound clusters, the label comes from MoA annotations.

        Parameters
        ----------
        labels : np.ndarray, shape (N,)
        metadata : pd.DataFrame
        gene_col : str
        pathway_col : str, optional
            Column containing pre-annotated pathway labels.
        top_n : int
            Number of top labels to return per cluster.

        Returns
        -------
        dict mapping cluster_id → list of top label strings
        """
        unique_labels = sorted(set(labels) - {-1})
        cluster_names: Dict[int, List[str]] = {}

        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_meta = metadata.iloc[mask] if len(metadata) == len(labels) else metadata

            top_labels = []

            # Try gene labels
            if gene_col in cluster_meta.columns:
                genes = cluster_meta[gene_col].dropna()
                if len(genes) > 0:
                    top_labels = genes.value_counts().head(top_n).index.tolist()

            # Fall back to pathway column
            if not top_labels and pathway_col and pathway_col in cluster_meta.columns:
                paths = cluster_meta[pathway_col].dropna()
                if len(paths) > 0:
                    top_labels = paths.value_counts().head(top_n).index.tolist()

            # Fall back to perturbation type
            if not top_labels:
                top_labels = [f"Cluster_{cluster_id}"]

            cluster_names[cluster_id] = top_labels

        logger.info(
            f"Labeled {len(cluster_names)} clusters from metadata."
        )
        return cluster_names

    def get_cluster_members(
        self,
        labels: np.ndarray,
        metadata: pd.DataFrame,
        cluster_id: int,
    ) -> pd.DataFrame:
        """
        Return metadata rows for all perturbations in a given cluster.

        Parameters
        ----------
        labels : np.ndarray, shape (N,)
        metadata : pd.DataFrame, shape (N,)
        cluster_id : int

        Returns
        -------
        pd.DataFrame
        """
        mask = labels == cluster_id
        members = metadata.iloc[mask].copy() if len(metadata) == len(labels) else pd.DataFrame()
        logger.debug(f"Cluster {cluster_id}: {mask.sum()} members")
        return members

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str) -> None:
        """Save all clustering results (labels, UMAP coords, metrics) to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.umap_2d_ is not None:
            np.save(str(out / "umap_2d_coords.npy"), self.umap_2d_)
        if self.umap_3d_ is not None:
            np.save(str(out / "umap_3d_coords.npy"), self.umap_3d_)
        if self.hdbscan_labels_ is not None:
            np.save(str(out / "hdbscan_labels.npy"), self.hdbscan_labels_)
        if self.kmeans_labels_ is not None:
            np.save(str(out / "kmeans_labels.npy"), self.kmeans_labels_)
        if self._wcss_by_k:
            with open(str(out / "wcss_by_k.json"), "w") as f:
                json.dump(self._wcss_by_k, f, indent=2)

        logger.info(f"Clustering results saved to {out}")
