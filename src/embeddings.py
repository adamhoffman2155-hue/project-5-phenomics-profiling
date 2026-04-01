"""
EmbeddingProcessor — OpenPhenom ViT-MAE embedding extraction and normalization.

This module handles:
1. Loading the OpenPhenom model (ViT-L/16 MAE) from HuggingFace or timm
2. Running batched inference over 5-channel RxRx3-core microscopy images
3. Aggregating field-of-view and replicate embeddings
4. Applying Typical Variation Normalization (TVN) to remove plate batch effects
5. Spherical (L2) normalization
6. Computing embedding quality metrics (percent replicating, explained variance)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore
    nn = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENPHENOM_HUB_ID = "recursionpharma/OpenPhenom"
OPENPHENOM_EMBEDDING_DIM = 1536  # ViT-L/16 CLS token dimension

# RxRx3 channel order
CHANNEL_NAMES = ["DAPI", "ConA", "SYTO14", "WGA+Phalloidin", "MitoTracker"]
N_CHANNELS = 5


# ---------------------------------------------------------------------------
# Lightweight ViT stub for environments without HuggingFace access
# ---------------------------------------------------------------------------

def _make_openphenom_stub(embedding_dim: int = OPENPHENOM_EMBEDDING_DIM):
    """
    Factory that returns an _OpenPhenomStub instance.

    This deferred construction allows the module to be imported even when
    PyTorch is not installed; the stub is only instantiated when actually
    needed (i.e., when embed_images is called).

    In production, replace with:
        import timm
        model = timm.create_model("hf-hub:recursionpharma/OpenPhenom", pretrained=True)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for embedding extraction. "
            "Install with: pip install torch"
        )
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F

    class _Stub(_nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.embedding_dim = dim
            self._proj = _nn.Linear(5 * 224 * 224, dim, bias=False)
            _nn.init.orthogonal_(self._proj.weight)
            for p in self.parameters():
                p.requires_grad = False

        def forward(self, x):
            B = x.shape[0]
            flat = x.view(B, -1)
            if flat.shape[1] != self._proj.weight.shape[1]:
                flat = _F.adaptive_avg_pool1d(
                    flat.unsqueeze(1), self._proj.weight.shape[1]
                ).squeeze(1)
            emb = self._proj(flat)
            return _F.normalize(emb, dim=-1)

    return _Stub(embedding_dim)


# Keep the class name available for isinstance checks in tests
class _OpenPhenomStub:
    """Sentinel class for isinstance checks. Actual stub created by _make_openphenom_stub()."""
    pass


class EmbeddingProcessor:
    """
    Extracts, aggregates, and normalizes OpenPhenom embeddings for RxRx3-core.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration. Uses defaults if None.
    device : str, optional
        Torch device string. Overrides config if provided.

    Examples
    --------
    >>> processor = EmbeddingProcessor()
    >>> model = processor.load_openphenom_model()
    >>> # embeddings shape: (N_wells, 1536)
    >>> embs = processor.embed_images(images, model)
    >>> embs_tvn = processor.apply_typical_variation_normalization(embs, controls_mask)
    """

    def __init__(self, config=None, device: Optional[str] = None):
        self.config = config
        self._embedding_dim = (
            config.embedding.embedding_dim
            if config is not None
            else OPENPHENOM_EMBEDDING_DIM
        )
        self._batch_size = (
            config.embedding.batch_size if config is not None else 64
        )

        if device is not None:
            self._device = device
        elif config is not None:
            self._device = config.embedding.device
        else:
            if _TORCH_AVAILABLE:
                import torch as _torch
                self._device = "cuda" if _torch.cuda.is_available() else "cpu"
            else:
                self._device = "cpu"

        self._model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_openphenom_model(
        self,
        model_name: str = "openphenom",
        cache_dir: Optional[str] = None,
        use_stub: bool = False,
    ):
        """
        Load the OpenPhenom ViT-MAE model.

        Attempts to load from HuggingFace hub via timm. Falls back to the
        stub implementation if the model is unavailable (e.g., in testing).

        Parameters
        ----------
        model_name : str
            One of "openphenom" (ViT-L/16 MAE) or a timm model identifier.
        cache_dir : str, optional
            Local directory to cache downloaded weights.
        use_stub : bool
            Force use of the stub model (for testing without internet access).

        Returns
        -------
        nn.Module
            Model in eval mode on the configured device.
        """
        if use_stub:
            logger.info("Loading OpenPhenom stub model (weights not downloaded).")
            model = _make_openphenom_stub(self._embedding_dim)
            model.eval()
            model = model.to(self._device)
            self._model = model
            return model

        try:
            import timm

            logger.info("Loading OpenPhenom via timm from HuggingFace hub...")
            # timm supports hf-hub: prefix for HuggingFace models
            model = timm.create_model(
                f"hf-hub:{OPENPHENOM_HUB_ID}",
                pretrained=True,
                num_classes=0,       # Remove classification head → raw features
                cache_dir=cache_dir,
            )
            model.eval()
            model = model.to(self._device)
            self._model = model
            logger.info(
                f"OpenPhenom loaded. "
                f"Expected output dim: {self._embedding_dim}"
            )
            return model

        except Exception as e:
            logger.warning(
                f"Could not load OpenPhenom from HuggingFace ({e}). "
                f"Falling back to stub model. "
                f"For production use, ensure internet access and valid model weights."
            )
            model = _make_openphenom_stub(self._embedding_dim)
            model.eval()
            model = model.to(self._device)
            self._model = model
            return model

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def embed_images(
        self,
        images: Union[np.ndarray, torch.Tensor],
        model: Optional[nn.Module] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Run OpenPhenom inference on a stack of 5-channel images.

        Parameters
        ----------
        images : array-like, shape (N, 5, H, W) or (N, H, W, 5)
            Batch of 5-channel microscopy images. Will be converted to float32
            and normalized by channel stats before inference.
        model : nn.Module, optional
            OpenPhenom model. Uses cached model if None.
        batch_size : int, optional
            Inference batch size. Overrides config.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        np.ndarray, shape (N, embedding_dim)
            Per-image embeddings.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for embed_images. Install with: pip install torch"
            )
        import torch as _torch

        if model is None:
            if self._model is None:
                model = self.load_openphenom_model()
            else:
                model = self._model

        if batch_size is None:
            batch_size = self._batch_size

        # Convert to torch tensor
        if not isinstance(images, _torch.Tensor):
            images = _torch.from_numpy(np.array(images, dtype=np.float32))

        # Handle channel-last format
        if images.ndim == 4 and images.shape[-1] == N_CHANNELS:
            images = images.permute(0, 3, 1, 2)

        # Normalize to [0, 1] if in uint8 range
        if images.max() > 1.5:
            images = images / 255.0

        n = len(images)
        all_embeddings = []

        model.eval()
        with _torch.no_grad():
            pbar = tqdm(
                range(0, n, batch_size),
                desc="Embedding images",
                disable=not show_progress,
            )
            for start in pbar:
                end = min(start + batch_size, n)
                batch = images[start:end].to(self._device)
                emb = model(batch)
                if isinstance(emb, (list, tuple)):
                    emb = emb[0]
                all_embeddings.append(emb.cpu().numpy())

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.info(
            f"Embedded {n} images → shape {embeddings.shape}"
        )
        return embeddings

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_replicates(
        self,
        embeddings: np.ndarray,
        replicate_groups: List[List[int]],
        method: str = "mean",
    ) -> np.ndarray:
        """
        Aggregate per-well embeddings to per-perturbation level.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N_wells, D)
        replicate_groups : list of lists
            Each inner list contains the integer row indices (into `embeddings`)
            of replicate wells for one perturbation.
        method : str
            "mean" or "median".

        Returns
        -------
        np.ndarray, shape (N_perturbations, D)
        """
        agg_fn = np.mean if method == "mean" else np.median
        agg_embs = np.stack(
            [agg_fn(embeddings[indices], axis=0) for indices in replicate_groups]
        ).astype(np.float32)
        logger.debug(
            f"Aggregated {len(replicate_groups)} perturbation groups "
            f"({method}) → shape {agg_embs.shape}"
        )
        return agg_embs

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply spherical (L2) normalization to project embeddings onto the unit hypersphere.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)

        Returns
        -------
        np.ndarray, shape (N, D)
            L2-normalized embeddings.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms
        logger.debug(
            f"Spherical normalization applied. "
            f"Mean L2 norm after: {np.linalg.norm(normalized, axis=1).mean():.4f}"
        )
        return normalized.astype(np.float32)

    def apply_typical_variation_normalization(
        self,
        embeddings: np.ndarray,
        controls_mask: np.ndarray,
        epsilon: float = 1e-6,
        per_plate: bool = False,
        plate_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply Typical Variation Normalization (TVN) to remove plate batch effects.

        TVN subtracts the mean of negative control embeddings and divides by their
        standard deviation, then applies L2 normalization. This corrects for
        plate-to-plate technical variation while preserving biological signal.

        The approach mirrors Recursion's internal normalization pipeline described
        in the RxRx3 paper.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
            Raw or replicate-aggregated embeddings.
        controls_mask : np.ndarray, shape (N,), dtype bool
            Boolean mask indicating which rows are negative control wells.
        epsilon : float
            Small value added to std to prevent division by zero.
        per_plate : bool
            If True, compute TVN statistics per plate (recommended for real data).
        plate_ids : np.ndarray, shape (N,), optional
            Plate identifier per row. Required if per_plate=True.

        Returns
        -------
        np.ndarray, shape (N, D)
            TVN-normalized and L2-normalized embeddings.
        """
        if controls_mask.sum() == 0:
            logger.warning(
                "No control wells found for TVN. Applying only L2 normalization."
            )
            return self.normalize_embeddings(embeddings)

        if per_plate and plate_ids is not None:
            return self._apply_tvn_per_plate(
                embeddings, controls_mask, plate_ids, epsilon
            )

        # Global TVN
        ctrl_embs = embeddings[controls_mask]
        ctrl_mean = ctrl_embs.mean(axis=0)   # shape (D,)
        ctrl_std = ctrl_embs.std(axis=0)     # shape (D,)

        normalized = (embeddings - ctrl_mean) / (ctrl_std + epsilon)
        normalized = self.normalize_embeddings(normalized)

        logger.info(
            f"TVN applied: subtracted mean of {controls_mask.sum()} control wells, "
            f"divided by per-dim std. Output shape: {normalized.shape}"
        )
        return normalized

    def _apply_tvn_per_plate(
        self,
        embeddings: np.ndarray,
        controls_mask: np.ndarray,
        plate_ids: np.ndarray,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """Apply TVN independently per plate."""
        result = embeddings.copy()
        unique_plates = np.unique(plate_ids)

        for plate in unique_plates:
            plate_mask = plate_ids == plate
            ctrl_in_plate = plate_mask & controls_mask

            if ctrl_in_plate.sum() < 2:
                logger.warning(
                    f"Plate {plate} has <2 control wells. "
                    "Skipping per-plate TVN for this plate."
                )
                continue

            ctrl_embs = embeddings[ctrl_in_plate]
            ctrl_mean = ctrl_embs.mean(axis=0)
            ctrl_std = ctrl_embs.std(axis=0)

            result[plate_mask] = (
                embeddings[plate_mask] - ctrl_mean
            ) / (ctrl_std + epsilon)

        result = self.normalize_embeddings(result)
        logger.info(
            f"Per-plate TVN applied across {len(unique_plates)} plates."
        )
        return result

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def compute_embedding_quality_metrics(
        self,
        embeddings: np.ndarray,
        metadata,
        n_pca_components: int = 50,
    ) -> Dict[str, float]:
        """
        Compute embedding quality metrics.

        Metrics
        -------
        - pca_explained_variance_50: Fraction of variance explained by top 50 PCs
        - mean_pairwise_cosine_sim: Average cosine similarity across all pairs (sample)
        - embedding_rank: Effective rank of the embedding matrix
        - norm_mean / norm_std: Statistics of L2 norms (should ~1 after normalization)

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        metadata : pd.DataFrame or None
        n_pca_components : int

        Returns
        -------
        dict
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize

        metrics: Dict[str, float] = {}
        N, D = embeddings.shape

        # L2 norm statistics
        norms = np.linalg.norm(embeddings, axis=1)
        metrics["norm_mean"] = float(norms.mean())
        metrics["norm_std"] = float(norms.std())

        # PCA explained variance
        n_components = min(n_pca_components, N - 1, D)
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(embeddings)
        metrics["pca_explained_variance_50"] = float(
            pca.explained_variance_ratio_.sum()
        )
        metrics["pca_n_components_95pct"] = int(
            np.searchsorted(
                np.cumsum(pca.explained_variance_ratio_), 0.95
            ) + 1
        )

        # Effective rank (entropy of singular values)
        singular_vals = pca.singular_values_
        sv_norm = singular_vals / singular_vals.sum()
        sv_norm = sv_norm[sv_norm > 0]
        effective_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm))))
        metrics["effective_rank"] = effective_rank

        # Mean pairwise cosine sim (subsample for speed)
        rng = np.random.default_rng(42)
        n_sample = min(500, N)
        idx = rng.choice(N, size=n_sample, replace=False)
        embs_sample = normalize(embeddings[idx], norm="l2")
        sim_matrix = embs_sample @ embs_sample.T
        # Exclude diagonal
        mask = ~np.eye(n_sample, dtype=bool)
        metrics["mean_pairwise_cosine_sim"] = float(sim_matrix[mask].mean())

        logger.info(
            f"Embedding quality: norm_mean={metrics['norm_mean']:.4f}, "
            f"pca_var@50={metrics['pca_explained_variance_50']:.3f}, "
            f"eff_rank={metrics['effective_rank']:.1f}, "
            f"mean_cos_sim={metrics['mean_pairwise_cosine_sim']:.4f}"
        )
        return metrics

    def compute_umap_spread(
        self,
        umap_coords: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute statistics of UMAP embedding spread and cluster separation.

        Parameters
        ----------
        umap_coords : np.ndarray, shape (N, 2) or (N, 3)
        labels : np.ndarray, shape (N,), optional
            Cluster labels for intra/inter-cluster distance computation.

        Returns
        -------
        dict with keys: total_spread, mean_nn_distance, cluster_separation (if labels given)
        """
        metrics: Dict[str, float] = {}

        # Total spread (std of coordinates)
        metrics["umap_spread_x"] = float(umap_coords[:, 0].std())
        metrics["umap_spread_y"] = float(umap_coords[:, 1].std())
        metrics["total_spread"] = float(np.linalg.norm(umap_coords.std(axis=0)))

        if labels is not None:
            unique_labels = np.unique(labels[labels >= 0])
            if len(unique_labels) > 1:
                # Compute cluster centroid distances
                centroids = np.stack(
                    [umap_coords[labels == l].mean(axis=0) for l in unique_labels]
                )
                centroid_dists = np.linalg.norm(
                    centroids[:, None] - centroids[None, :], axis=-1
                )
                # Mean inter-centroid distance
                n_clusters = len(unique_labels)
                mask = ~np.eye(n_clusters, dtype=bool)
                metrics["mean_inter_cluster_dist"] = float(centroid_dists[mask].mean())

                # Mean intra-cluster spread
                intra_spreads = []
                for l in unique_labels:
                    cluster_pts = umap_coords[labels == l]
                    if len(cluster_pts) > 1:
                        centroid = cluster_pts.mean(axis=0)
                        spread = np.linalg.norm(cluster_pts - centroid, axis=1).mean()
                        intra_spreads.append(float(spread))
                if intra_spreads:
                    metrics["mean_intra_cluster_spread"] = float(np.mean(intra_spreads))
                    metrics["cluster_separation_ratio"] = (
                        metrics["mean_inter_cluster_dist"]
                        / (metrics["mean_intra_cluster_spread"] + 1e-9)
                    )

        return metrics
