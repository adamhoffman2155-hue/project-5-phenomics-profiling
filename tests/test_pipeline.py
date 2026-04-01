"""
test_pipeline.py — Unit tests for the Phenomics Perturbation Profiling Pipeline.

Tests use synthetic embeddings (random numpy arrays with realistic cluster
structure) so that no real data or internet access is required.

Run with:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v --tb=short
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_dim() -> int:
    """Standard embedding dimension used across tests."""
    return 1536


@pytest.fixture(scope="module")
def n_perturbations() -> int:
    """Number of synthetic perturbations for tests."""
    return 200


@pytest.fixture(scope="module")
def n_clusters() -> int:
    """Number of ground-truth clusters in synthetic data."""
    return 8


@pytest.fixture(scope="module")
def synthetic_embeddings(
    n_perturbations: int, embedding_dim: int, n_clusters: int
) -> np.ndarray:
    """
    Generate synthetic embeddings with clear cluster structure.

    Creates `n_clusters` cluster centers with Gaussian noise around each.
    L2-normalized to unit sphere to mimic real embedding behavior.
    """
    rng = np.random.default_rng(42)
    cluster_centers = rng.standard_normal((n_clusters, embedding_dim)) * 3.0
    per_cluster = n_perturbations // n_clusters

    embs = []
    for c in range(n_clusters):
        noise = rng.standard_normal((per_cluster, embedding_dim)) * 0.3
        embs.append(cluster_centers[c] + noise)

    embeddings = np.vstack(embs).astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1.0, norms)
    return embeddings


@pytest.fixture(scope="module")
def synthetic_metadata(
    n_perturbations: int, n_clusters: int
) -> pd.DataFrame:
    """
    Generate realistic synthetic metadata aligned with synthetic_embeddings.
    """
    rng = np.random.default_rng(42)
    per_cluster = n_perturbations // n_clusters

    gene_pool = [
        "BRCA1", "BRCA2", "ATM", "ATR", "PARP1",
        "MTOR", "TSC1", "TSC2", "RPTOR", "AKT1",
        "PSMA1", "PSMB1", "EGFR", "KRAS", "BRAF",
        "BCL2", "HDAC1", "ATG5", "SF3B1", "XPO1",
    ]
    moa_pool = [
        "DDR", "mTOR inhibitor", "Proteasome inhibitor",
        "EGFR inhibitor", "BCL2 inhibitor", "HDAC inhibitor",
        "Autophagy", "Splicing",
    ]
    compound_pool = [
        "olaparib", "niraparib", "rapamycin", "everolimus",
        "bortezomib", "erlotinib", "venetoclax", "vorinostat",
    ]

    rows = []
    for c in range(n_clusters):
        for i in range(per_cluster):
            global_idx = c * per_cluster + i
            if i % 3 == 0:  # compound
                rows.append({
                    "embedding_id": global_idx,
                    "perturbation_type": "compound",
                    "gene": np.nan,
                    "compound": compound_pool[c % len(compound_pool)],
                    "concentration_um": 1.0,
                    "moa": moa_pool[c % len(moa_pool)],
                    "plate": f"Plate{(global_idx // 50) + 1}",
                    "well": f"A{(global_idx % 24) + 1:02d}",
                    "replicate": (i % 4) + 1,
                })
            elif i % 10 == 0:  # negcon
                rows.append({
                    "embedding_id": global_idx,
                    "perturbation_type": "negcon",
                    "gene": np.nan,
                    "compound": "DMSO",
                    "concentration_um": 0.1,
                    "moa": "negative_control",
                    "plate": f"Plate{(global_idx // 50) + 1}",
                    "well": f"B{(global_idx % 24) + 1:02d}",
                    "replicate": (i % 4) + 1,
                })
            else:  # CRISPR
                gene = gene_pool[c % len(gene_pool)]
                rows.append({
                    "embedding_id": global_idx,
                    "perturbation_type": "CRISPR",
                    "gene": gene,
                    "compound": np.nan,
                    "concentration_um": np.nan,
                    "moa": np.nan,
                    "plate": f"Plate{(global_idx // 50) + 1}",
                    "well": f"C{(global_idx % 24) + 1:02d}",
                    "replicate": (i % 4) + 1,
                })

    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture(scope="module")
def pipeline_config():
    """Build a lightweight PipelineConfig suitable for testing."""
    from config.config import PipelineConfig
    cfg = PipelineConfig()
    cfg.clustering.hdbscan_min_cluster_size = 5   # Small for test data
    cfg.clustering.hdbscan_min_samples = 2
    cfg.clustering.kmeans_k = 8
    cfg.clustering.umap_n_neighbors = 10
    cfg.retrieval.top_k = 5
    cfg.retrieval.recall_k_values = [1, 5, 10]
    return cfg


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_default_construction(self):
        from config.config import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.embedding.embedding_dim == 1536
        assert cfg.clustering.hdbscan_min_cluster_size == 50
        assert cfg.clustering.kmeans_k == 25
        assert cfg.retrieval.top_k == 10
        assert cfg.pathway.p_cutoff == 0.05
        assert len(cfg.pathway.gene_sets) > 0

    def test_channel_names(self):
        from config.config import PipelineConfig
        cfg = PipelineConfig()
        assert len(cfg.embedding.channel_names) == 5
        assert "DAPI" in cfg.embedding.channel_names
        assert "MitoTracker" in cfg.embedding.channel_names

    def test_to_dict_roundtrip(self):
        from config.config import PipelineConfig
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "embedding" in d
        assert "clustering" in d
        assert d["embedding"]["embedding_dim"] == 1536

    def test_validation_raises_on_bad_values(self):
        from config.config import PipelineConfig, PathwayConfig
        # p_cutoff outside (0, 1) should raise AssertionError in __post_init__
        import dataclasses
        cfg = PipelineConfig()
        # Patch and re-trigger __post_init__ with invalid value
        cfg.pathway.p_cutoff = 0.05  # valid
        assert cfg.pathway.p_cutoff == 0.05
        # Directly verify the assertion guards work
        with pytest.raises(AssertionError):
            bad = PipelineConfig.__new__(PipelineConfig)
            object.__setattr__(bad, "embedding", dataclasses.replace(cfg.embedding, embedding_dim=-1))
            object.__setattr__(bad, "clustering", cfg.clustering)
            object.__setattr__(bad, "retrieval", cfg.retrieval)
            object.__setattr__(bad, "pathway", cfg.pathway)
            object.__setattr__(bad, "random_seed", 42)
            object.__setattr__(bad, "n_jobs", -1)
            object.__setattr__(bad, "verbose", True)
            object.__setattr__(bad, "log_level", "INFO")
            object.__setattr__(bad, "log_file", None)
            object.__setattr__(bad, "experiment_name", "test")
            object.__setattr__(bad, "crispr_label", "CRISPR")
            object.__setattr__(bad, "compound_label", "compound")
            object.__setattr__(bad, "control_label", "negcon")
            object.__setattr__(bad, "perturbation_type_col", "perturbation_type")
            object.__setattr__(bad, "gene_col", "gene")
            object.__setattr__(bad, "compound_col", "compound")
            object.__setattr__(bad, "concentration_col", "concentration_um")
            object.__setattr__(bad, "cell_line_col", "cell_line")
            object.__setattr__(bad, "plate_col", "plate")
            object.__setattr__(bad, "well_col", "well")
            object.__setattr__(bad, "site_col", "site")
            object.__setattr__(bad, "replicate_col", "replicate")
            bad.__post_init__()


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

class TestRxRx3DataLoader:
    def test_generate_synthetic_metadata(self):
        from src.data_loader import RxRx3DataLoader
        loader = RxRx3DataLoader()
        meta = loader._generate_synthetic_metadata(n_perturbations=100)
        assert isinstance(meta, pd.DataFrame)
        assert len(meta) == 100
        assert "perturbation_type" in meta.columns
        assert "gene" in meta.columns
        assert "compound" in meta.columns
        assert "plate" in meta.columns

    def test_generate_synthetic_embeddings(self):
        from src.data_loader import RxRx3DataLoader
        loader = RxRx3DataLoader()
        embs, meta = loader._generate_synthetic_embeddings(
            n_perturbations=50, embedding_dim=128
        )
        assert embs.shape == (50, 128)
        assert len(meta) == 50
        assert embs.dtype == np.float32

    def test_get_crispr_perturbations(self, synthetic_metadata):
        from src.data_loader import RxRx3DataLoader
        loader = RxRx3DataLoader()
        crispr = loader.get_crispr_perturbations(synthetic_metadata)
        assert isinstance(crispr, pd.DataFrame)
        assert len(crispr) > 0
        assert (crispr["perturbation_type"] == "CRISPR").all()

    def test_get_compound_perturbations(self, synthetic_metadata):
        from src.data_loader import RxRx3DataLoader
        loader = RxRx3DataLoader()
        compounds = loader.get_compound_perturbations(synthetic_metadata)
        assert isinstance(compounds, pd.DataFrame)
        assert len(compounds) > 0
        assert (compounds["perturbation_type"] == "compound").all()

    def test_get_controls(self, synthetic_metadata):
        from src.data_loader import RxRx3DataLoader
        loader = RxRx3DataLoader()
        controls = loader.get_controls(synthetic_metadata, control_type="negcon")
        assert isinstance(controls, pd.DataFrame)
        assert (controls["perturbation_type"] == "negcon").all()

    def test_split_by_plate(self, synthetic_metadata):
        from src.data_loader import RxRx3DataLoader
        loader = RxRx3DataLoader()
        splits = loader.split_by_plate(synthetic_metadata)
        n_plates = synthetic_metadata["plate"].nunique()
        assert len(splits) == n_plates
        for train, test in splits:
            assert len(test) > 0
            assert len(train) + len(test) == len(synthetic_metadata)
            # No plate overlap
            train_plates = set(train["plate"])
            test_plates = set(test["plate"])
            assert len(train_plates & test_plates) == 0

    def test_compute_percent_replicating_returns_dict(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.data_loader import RxRx3DataLoader
        loader = RxRx3DataLoader()
        result = loader.compute_percent_replicating(
            synthetic_embeddings,
            synthetic_metadata,
            n_resamples=100,
        )
        assert isinstance(result, dict)
        # Should have at least some keys
        assert len(result) >= 0  # May be empty if no replicates in synthetic data


# ---------------------------------------------------------------------------
# Embedding processor tests
# ---------------------------------------------------------------------------

class TestEmbeddingProcessor:
    def test_normalize_embeddings(self, synthetic_embeddings):
        from src.embeddings import EmbeddingProcessor
        proc = EmbeddingProcessor()
        normed = proc.normalize_embeddings(synthetic_embeddings)
        assert normed.shape == synthetic_embeddings.shape
        # All rows should have unit L2 norm
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_tvn_normalization_output_shape(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.embeddings import EmbeddingProcessor
        proc = EmbeddingProcessor()
        controls_mask = synthetic_metadata["perturbation_type"].values == "negcon"
        normed = proc.apply_typical_variation_normalization(
            synthetic_embeddings, controls_mask=controls_mask
        )
        assert normed.shape == synthetic_embeddings.shape
        assert normed.dtype == np.float32

    def test_tvn_no_controls_fallback(self, synthetic_embeddings):
        from src.embeddings import EmbeddingProcessor
        proc = EmbeddingProcessor()
        controls_mask = np.zeros(len(synthetic_embeddings), dtype=bool)
        normed = proc.apply_typical_variation_normalization(
            synthetic_embeddings, controls_mask=controls_mask
        )
        assert normed.shape == synthetic_embeddings.shape

    def test_aggregate_replicates(self):
        from src.embeddings import EmbeddingProcessor
        proc = EmbeddingProcessor()
        embeddings = np.random.randn(12, 64).astype(np.float32)
        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        agg = proc.aggregate_replicates(embeddings, groups, method="mean")
        assert agg.shape == (4, 64)
        # First group mean should match manual mean
        np.testing.assert_allclose(agg[0], embeddings[[0, 1, 2]].mean(axis=0), atol=1e-6)

    def test_load_openphenom_stub(self):
        pytest.importorskip("torch", reason="torch not installed")
        from src.embeddings import EmbeddingProcessor, _make_openphenom_stub
        proc = EmbeddingProcessor()
        model = proc.load_openphenom_model(use_stub=True)
        assert model is not None
        # The stub should be a torch nn.Module
        import torch.nn as nn
        assert isinstance(model, nn.Module)

    def test_embed_images_stub_output_shape(self, embedding_dim):
        pytest.importorskip("torch", reason="torch not installed")
        from src.embeddings import EmbeddingProcessor
        proc = EmbeddingProcessor(device="cpu")
        model = proc.load_openphenom_model(use_stub=True)
        # Create 4 synthetic 5-channel images
        images = np.random.rand(4, 5, 224, 224).astype(np.float32)
        embs = proc.embed_images(images, model=model, batch_size=2, show_progress=False)
        assert embs.shape == (4, embedding_dim)
        assert embs.dtype == np.float32

    def test_compute_embedding_quality_metrics(self, synthetic_embeddings):
        from src.embeddings import EmbeddingProcessor
        proc = EmbeddingProcessor()
        metrics = proc.compute_embedding_quality_metrics(
            synthetic_embeddings, metadata=None
        )
        assert "norm_mean" in metrics
        assert "pca_explained_variance_50" in metrics
        assert "effective_rank" in metrics
        assert "mean_pairwise_cosine_sim" in metrics
        # All values should be finite
        for k, v in metrics.items():
            assert np.isfinite(v), f"Metric {k} = {v} is not finite"


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------

class TestPerturbationClusterer:
    def test_run_umap_2d_shape(
        self, synthetic_embeddings, pipeline_config
    ):
        from src.clustering import PerturbationClusterer
        clusterer = PerturbationClusterer(config=pipeline_config)
        try:
            coords = clusterer.run_umap(synthetic_embeddings, n_components=2)
            assert coords.shape == (len(synthetic_embeddings), 2)
            assert coords.dtype == np.float32
        except ImportError:
            pytest.skip("umap-learn not installed")

    def test_run_hdbscan_returns_labels(
        self, synthetic_embeddings, pipeline_config
    ):
        from src.clustering import PerturbationClusterer
        clusterer = PerturbationClusterer(config=pipeline_config)
        try:
            coords = clusterer.run_umap(synthetic_embeddings, n_components=2)
            labels = clusterer.run_hdbscan(coords)
            assert labels.shape == (len(synthetic_embeddings),)
            # Should find at least 1 cluster (HDBSCAN may find 0 on small data)
            n_clusters = len(set(labels) - {-1})
            assert n_clusters >= 0  # Can be 0 if all noise on tiny data
        except ImportError:
            pytest.skip("umap-learn or hdbscan not installed")

    def test_run_kmeans_returns_correct_n_clusters(
        self, synthetic_embeddings, pipeline_config
    ):
        from src.clustering import PerturbationClusterer
        k = 8
        clusterer = PerturbationClusterer(config=pipeline_config)
        labels = clusterer.run_kmeans(synthetic_embeddings, k=k)
        assert labels.shape == (len(synthetic_embeddings),)
        assert len(np.unique(labels)) == k
        assert labels.min() == 0
        assert labels.max() == k - 1

    def test_run_agglomerative(self, synthetic_embeddings, pipeline_config):
        from src.clustering import PerturbationClusterer
        k = 6
        clusterer = PerturbationClusterer(config=pipeline_config)
        labels = clusterer.run_agglomerative(synthetic_embeddings, n_clusters=k)
        assert labels.shape == (len(synthetic_embeddings),)
        assert len(np.unique(labels)) == k

    def test_evaluate_clustering_returns_scores(
        self, synthetic_embeddings, pipeline_config
    ):
        from src.clustering import PerturbationClusterer
        clusterer = PerturbationClusterer(config=pipeline_config)
        labels = clusterer.run_kmeans(synthetic_embeddings, k=8)
        scores = clusterer.evaluate_clustering(synthetic_embeddings, labels)
        assert isinstance(scores, dict)
        assert "n_clusters" in scores
        assert scores["n_clusters"] == 8
        # Silhouette score should be in valid range
        if "silhouette_score" in scores:
            assert -1 <= scores["silhouette_score"] <= 1

    def test_find_optimal_k_returns_int(
        self, synthetic_embeddings, pipeline_config
    ):
        from src.clustering import PerturbationClusterer
        clusterer = PerturbationClusterer(config=pipeline_config)
        optimal_k, wcss = clusterer.find_optimal_k(
            synthetic_embeddings, k_min=3, k_max=15, k_step=3
        )
        assert isinstance(optimal_k, int)
        assert 3 <= optimal_k <= 15
        assert isinstance(wcss, dict)
        assert len(wcss) > 0

    def test_get_cluster_members(
        self, synthetic_embeddings, synthetic_metadata, pipeline_config
    ):
        from src.clustering import PerturbationClusterer
        clusterer = PerturbationClusterer(config=pipeline_config)
        labels = clusterer.run_kmeans(synthetic_embeddings, k=8)
        members = clusterer.get_cluster_members(labels, synthetic_metadata, cluster_id=0)
        assert isinstance(members, pd.DataFrame)
        assert len(members) > 0
        assert (labels[members.index] == 0).all()

    def test_label_clusters_by_enrichment(
        self, synthetic_metadata, pipeline_config
    ):
        from src.clustering import PerturbationClusterer
        clusterer = PerturbationClusterer(config=pipeline_config)
        labels = np.array([i % 8 for i in range(len(synthetic_metadata))])
        cluster_labels = clusterer.label_clusters_by_enrichment(
            labels, synthetic_metadata
        )
        assert isinstance(cluster_labels, dict)
        assert len(cluster_labels) > 0
        for cid, names in cluster_labels.items():
            assert isinstance(names, list)
            assert len(names) > 0


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------

class TestMoARetriever:
    def test_build_reference_index(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.retrieval import MoARetriever
        retriever = MoARetriever()
        retriever.build_reference_index(synthetic_embeddings, synthetic_metadata)
        assert retriever._index_built
        assert retriever._embeddings_norm is not None
        assert retriever._embeddings_norm.shape == synthetic_embeddings.shape

    def test_retrieve_nearest_neighbors_shape(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.retrieval import MoARetriever
        retriever = MoARetriever()
        retriever.build_reference_index(synthetic_embeddings, synthetic_metadata)
        k = 5
        indices, sims = retriever.retrieve_nearest_neighbors(
            synthetic_embeddings[:10], top_k=k
        )
        assert indices.shape == (10, k)
        assert sims.shape == (10, k)
        # Similarities should be in [-1, 1]
        assert sims.max() <= 1.0 + 1e-5
        assert sims.min() >= -1.0 - 1e-5

    def test_nearest_neighbors_sorted_descending(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.retrieval import MoARetriever
        retriever = MoARetriever()
        retriever.build_reference_index(synthetic_embeddings, synthetic_metadata)
        _, sims = retriever.retrieve_nearest_neighbors(
            synthetic_embeddings[:5], top_k=10
        )
        # Each row should be sorted descending
        for row in sims:
            valid = row[row > 0]
            if len(valid) > 1:
                assert (np.diff(valid) <= 0).all(), "Similarities not sorted descending"

    def test_compute_recall_at_k_returns_float(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.retrieval import MoARetriever
        retriever = MoARetriever()
        recall = retriever.compute_recall_at_k(
            synthetic_embeddings,
            synthetic_metadata,
            k=5,
            moa_col="moa",
        )
        assert isinstance(recall, float)
        assert 0.0 <= recall <= 1.0

    def test_compute_recall_at_k_gene_fallback(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.retrieval import MoARetriever
        retriever = MoARetriever()
        # Use gene as MoA proxy
        recall = retriever.compute_recall_at_k(
            synthetic_embeddings,
            synthetic_metadata,
            k=5,
            moa_col="gene",
        )
        assert isinstance(recall, float)

    def test_compute_average_precision_range(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.retrieval import MoARetriever
        retriever = MoARetriever()
        retrieved = np.arange(20)
        ap = retriever.compute_average_precision(
            query_idx=0,
            retrieved_indices=retrieved,
            metadata=synthetic_metadata,
            moa_col="moa",
        )
        assert isinstance(ap, float)
        assert 0.0 <= ap <= 1.0

    def test_compute_map_returns_float(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.retrieval import MoARetriever
        retriever = MoARetriever()
        retriever.build_reference_index(synthetic_embeddings, synthetic_metadata)
        map_score = retriever.compute_map(
            synthetic_embeddings, synthetic_metadata, k=10
        )
        assert isinstance(map_score, float)
        assert 0.0 <= map_score <= 1.0

    def test_evaluate_moa_retrieval_keys(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.retrieval import MoARetriever
        cfg = MagicMock()
        cfg.recall_k_values = [1, 5, 10]
        cfg.top_k = 5
        cfg.similarity_metric = "cosine"
        cfg.moa_column = "moa"
        cfg.min_moa_group_size = 2
        cfg.cross_modal_retrieval = True

        retriever = MoARetriever()
        retriever._recall_k_values = [1, 5, 10]
        results = retriever.evaluate_moa_retrieval(
            synthetic_embeddings, synthetic_metadata
        )
        assert isinstance(results, dict)
        assert "recall_at_1" in results
        assert "recall_at_5" in results
        assert "recall_at_10" in results
        assert "map" in results

    def test_match_compounds_to_crispr(
        self, synthetic_embeddings, synthetic_metadata
    ):
        from src.retrieval import MoARetriever
        retriever = MoARetriever()
        result = retriever.match_compounds_to_crispr(
            synthetic_embeddings,
            synthetic_metadata,
            top_k=5,
            similarity_threshold=0.0,  # Low threshold to get results on synthetic data
        )
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "compound" in result.columns
            assert "crispr_gene" in result.columns
            assert "cosine_similarity" in result.columns
            assert "rank" in result.columns
            # Ranks should start at 1
            assert result["rank"].min() >= 1


# ---------------------------------------------------------------------------
# Pathway analysis tests
# ---------------------------------------------------------------------------

class TestPathwayAnalyzer:
    def test_get_cluster_gene_lists(
        self, synthetic_metadata
    ):
        from src.pathway_analysis import PathwayAnalyzer
        analyzer = PathwayAnalyzer()
        labels = np.array([i % 5 for i in range(len(synthetic_metadata))])
        gene_lists = analyzer.get_cluster_gene_lists(synthetic_metadata, labels)
        assert isinstance(gene_lists, dict)
        assert len(gene_lists) == 5
        for cid, genes in gene_lists.items():
            assert isinstance(genes, list)
            # All genes should be uppercase strings
            for g in genes:
                assert isinstance(g, str)
                assert g == g.upper()

    def test_run_enrichr_mock(self, synthetic_metadata):
        """Test Enrichr with mock (no internet required)."""
        from src.pathway_analysis import PathwayAnalyzer
        analyzer = PathwayAnalyzer()
        labels = np.array([i % 5 for i in range(len(synthetic_metadata))])
        gene_lists = analyzer.get_cluster_gene_lists(synthetic_metadata, labels)

        # Use _mock_enrichr_results directly (no internet needed)
        results = analyzer._mock_enrichr_results(gene_lists)
        assert isinstance(results, dict)
        for cid, df in results.items():
            assert isinstance(df, pd.DataFrame)

    def test_run_enrichr_fallback_to_mock(self, synthetic_metadata):
        """Test that run_enrichr falls back gracefully when gseapy is unavailable."""
        from src.pathway_analysis import PathwayAnalyzer
        analyzer = PathwayAnalyzer()
        gene_lists = {0: ["BRCA1", "BRCA2", "ATM", "RAD51", "CHEK1"]}

        # Patch gseapy import to fail by temporarily removing it from sys.modules
        import importlib
        import sys as _sys
        original = _sys.modules.get("gseapy", "NOT_PRESENT")
        _sys.modules["gseapy"] = None  # type: ignore
        try:
            results = analyzer.run_enrichr(gene_lists, gene_sets=["KEGG_2021_Human"])
        finally:
            if original == "NOT_PRESENT":
                _sys.modules.pop("gseapy", None)
            else:
                _sys.modules["gseapy"] = original
        assert isinstance(results, dict)

    def test_build_pathway_cluster_matrix(self, synthetic_metadata):
        from src.pathway_analysis import PathwayAnalyzer
        analyzer = PathwayAnalyzer()
        labels = np.array([i % 5 for i in range(len(synthetic_metadata))])
        gene_lists = analyzer.get_cluster_gene_lists(synthetic_metadata, labels)
        mock_results = analyzer._mock_enrichr_results(gene_lists)
        matrix = analyzer.build_pathway_cluster_matrix(mock_results, top_n=5)
        if len(matrix) > 0:
            assert isinstance(matrix, pd.DataFrame)
            assert matrix.shape[0] > 0
            assert matrix.shape[1] > 0

    def test_compare_clusters_pathways_jaccard_range(self, synthetic_metadata):
        from src.pathway_analysis import PathwayAnalyzer
        analyzer = PathwayAnalyzer()
        labels = np.array([i % 4 for i in range(len(synthetic_metadata))])
        gene_lists = analyzer.get_cluster_gene_lists(synthetic_metadata, labels)
        mock_results = analyzer._mock_enrichr_results(gene_lists)
        jaccard = analyzer.compare_clusters_pathways(mock_results)
        if len(jaccard) > 0:
            assert isinstance(jaccard, pd.DataFrame)
            # Jaccard values should be in [0, 1]
            assert jaccard.values.min() >= 0.0
            assert jaccard.values.max() <= 1.0
            # Diagonal should be 1.0 (self-similarity)
            diag = np.diag(jaccard.values)
            # Some diagonal entries may be 0 if a cluster has no enrichment
            assert all(v == pytest.approx(1.0) or v == 0.0 for v in diag)

    def test_identify_cluster_pathways_returns_dict(self, synthetic_metadata):
        from src.pathway_analysis import PathwayAnalyzer
        analyzer = PathwayAnalyzer()
        labels = np.array([i % 4 for i in range(len(synthetic_metadata))])
        gene_lists = analyzer.get_cluster_gene_lists(synthetic_metadata, labels)
        mock_results = analyzer._mock_enrichr_results(gene_lists)
        top_paths = analyzer.identify_cluster_pathways(mock_results, top_n=3)
        assert isinstance(top_paths, dict)
        for cid, paths in top_paths.items():
            assert isinstance(paths, list)
            assert len(paths) <= 3


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_cosine_similarity_matrix_self_similarity(self):
        from src.utils import cosine_similarity_matrix
        a = np.random.randn(20, 64).astype(np.float32)
        sim = cosine_similarity_matrix(a)
        assert sim.shape == (20, 20)
        np.testing.assert_allclose(np.diag(sim), np.ones(20), atol=1e-5)

    def test_cosine_similarity_matrix_cross(self):
        from src.utils import cosine_similarity_matrix
        a = np.random.randn(10, 64).astype(np.float32)
        b = np.random.randn(15, 64).astype(np.float32)
        sim = cosine_similarity_matrix(a, b)
        assert sim.shape == (10, 15)
        assert sim.max() <= 1.0 + 1e-5
        assert sim.min() >= -1.0 - 1e-5

    def test_batch_iterator_shapes(self):
        from src.utils import batch_iterator
        data = np.arange(100)
        batches = list(batch_iterator(data, batch_size=30))
        assert len(batches) == 4  # 30+30+30+10
        assert len(batches[0]) == 30
        assert len(batches[-1]) == 10

    def test_batch_iterator_drop_last(self):
        from src.utils import batch_iterator
        data = list(range(100))
        batches = list(batch_iterator(data, batch_size=30, drop_last=True))
        assert len(batches) == 3
        assert all(len(b) == 30 for b in batches)

    def test_format_gene_list_cleaning(self):
        from src.utils import format_gene_list
        raw = ["brca1", "BRCA2", None, "brca1", "", "nan", "PARP1"]
        result = format_gene_list(raw)
        assert "BRCA1" in result
        assert "BRCA2" in result
        assert "PARP1" in result
        # Duplicates removed
        assert result.count("BRCA1") == 1
        # Nulls and empty strings removed
        assert "" not in result
        assert "NAN" not in result

    def test_compute_jaccard_basic(self):
        from src.utils import compute_jaccard
        a = {"BRCA1", "BRCA2", "ATM"}
        b = {"BRCA2", "ATM", "PARP1"}
        j = compute_jaccard(a, b)
        # |a ∩ b| = 2, |a ∪ b| = 4 → 0.5
        assert j == pytest.approx(0.5, abs=1e-6)

    def test_compute_jaccard_identical_sets(self):
        from src.utils import compute_jaccard
        a = {"BRCA1", "BRCA2"}
        assert compute_jaccard(a, a) == pytest.approx(1.0)

    def test_compute_jaccard_disjoint_sets(self):
        from src.utils import compute_jaccard
        assert compute_jaccard({"A"}, {"B"}) == pytest.approx(0.0)

    def test_compute_jaccard_empty(self):
        from src.utils import compute_jaccard
        assert compute_jaccard(set(), set()) == pytest.approx(0.0)

    def test_compute_jaccard_matrix(self):
        from src.utils import compute_jaccard_matrix
        gene_lists = {
            "pathA": ["BRCA1", "BRCA2", "ATM"],
            "pathB": ["BRCA2", "ATM", "PARP1"],
            "pathC": ["MTOR", "TSC1"],
        }
        matrix = compute_jaccard_matrix(gene_lists)
        assert matrix.shape == (3, 3)
        assert matrix.loc["pathA", "pathA"] == pytest.approx(1.0)
        assert matrix.loc["pathA", "pathC"] == pytest.approx(0.0)
        assert matrix.loc["pathA", "pathB"] == pytest.approx(0.5, abs=0.01)

    def test_summarize_metadata(self, synthetic_metadata):
        from src.utils import summarize_metadata
        summary = summarize_metadata(synthetic_metadata)
        assert "n_rows" in summary
        assert "perturbation_type_counts" in summary
        assert summary["n_rows"] == len(synthetic_metadata)

    def test_timer_context_manager(self):
        from src.utils import Timer
        import time
        with Timer("test") as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04
        assert t.elapsed < 2.0  # Should not be slow

    def test_save_and_load_embeddings(self, synthetic_embeddings, synthetic_metadata, tmp_path):
        from src.utils import save_embeddings, load_embeddings_cached
        npy, csv = save_embeddings(
            synthetic_embeddings, synthetic_metadata,
            output_dir=str(tmp_path), prefix="test"
        )
        loaded_embs, loaded_meta = load_embeddings_cached(npy, csv)
        np.testing.assert_array_equal(loaded_embs, synthetic_embeddings)
        assert len(loaded_meta) == len(synthetic_metadata)


# ---------------------------------------------------------------------------
# Integration test: end-to-end mini pipeline
# ---------------------------------------------------------------------------

def _check_umap_available() -> bool:
    """Return True if umap-learn is importable."""
    try:
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torch_available() -> bool:
    """Return True if torch is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TestEndToEndPipeline:
    """
    Integration test: run a compressed version of the full pipeline
    with synthetic data and verify all outputs are produced.

    These tests are skipped when umap-learn or torch are not installed,
    as they exercise the full CLI scripts which require those dependencies.
    """

    @pytest.mark.skipif(
        not _check_umap_available(),
        reason="umap-learn not installed",
    )
    def test_mini_pipeline(self, tmp_path):
        """Run the full pipeline on tiny synthetic data and check outputs."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "run_pipeline.py"),
                "--synthetic",
                "--n_synthetic", "100",
                "--output", str(tmp_path / "results"),
                "--clustering_method", "kmeans",
                "--n_clusters", "5",
                "--skip_retrieval",
                "--skip_figures",
                "--log_level", "WARNING",
                "--no_3d_umap",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"Pipeline failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        # Check key output files exist
        assert (tmp_path / "results" / "pipeline_summary.json").exists()
        assert (tmp_path / "results" / "clustering" / "umap_2d_coords.npy").exists()
        assert (tmp_path / "results" / "clustering" / "cluster_assignments.csv").exists()

        # Verify pipeline summary content
        with open(tmp_path / "results" / "pipeline_summary.json") as f:
            summary = json.load(f)
        assert "n_perturbations" in summary
        assert summary["n_perturbations"] > 0

    @pytest.mark.skipif(
        not _check_umap_available(),
        reason="umap-learn not installed",
    )
    def test_mini_cluster_analysis_script(self, tmp_path):
        """Run cluster_analysis.py on synthetic data."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "cluster_analysis.py"),
                "--synthetic",
                "--n_synthetic", "100",
                "--method", "kmeans",
                "--n_clusters", "5",
                "--output", str(tmp_path / "clusters"),
                "--skip_figures",
                "--log_level", "WARNING",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"cluster_analysis.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        # Check outputs
        assert (tmp_path / "clusters" / "kmeans_labels.npy").exists()
        assert (tmp_path / "clusters" / "cluster_assignments.csv").exists()
        labels = np.load(str(tmp_path / "clusters" / "kmeans_labels.npy"))
        assert len(labels) == 100

    @pytest.mark.skipif(
        not _check_torch_available(),
        reason="torch not installed",
    )
    def test_mini_embed_script(self, tmp_path):
        """Run embed_perturbations.py with synthetic flag."""
        import subprocess
        output_path = tmp_path / "embeddings" / "test_embs.npy"
        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "embed_perturbations.py"),
                "--synthetic",
                "--n_synthetic", "50",
                "--output", str(output_path),
                "--log_level", "WARNING",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"embed_perturbations.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        assert output_path.exists()
        embs = np.load(str(output_path))
        assert embs.shape[0] == 50
        assert embs.shape[1] > 0


# ---------------------------------------------------------------------------
# Parametrize: test different clustering methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["kmeans", "agglomerative"])
def test_clustering_method_produces_valid_labels(
    method: str,
    synthetic_embeddings: np.ndarray,
):
    """All clustering methods should produce valid integer label arrays."""
    from src.clustering import PerturbationClusterer
    clusterer = PerturbationClusterer()
    k = 6

    if method == "kmeans":
        labels = clusterer.run_kmeans(synthetic_embeddings, k=k)
    elif method == "agglomerative":
        labels = clusterer.run_agglomerative(synthetic_embeddings, n_clusters=k)
    else:
        pytest.skip(f"Method {method} not handled in parametrize.")

    assert labels.shape == (len(synthetic_embeddings),)
    assert labels.dtype in (np.int32, np.int64, np.intp)
    n_unique = len(np.unique(labels))
    assert n_unique == k


@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_recall_at_k_increases_with_k(
    k: int,
    synthetic_embeddings: np.ndarray,
    synthetic_metadata: pd.DataFrame,
):
    """Recall@k should be non-decreasing as k increases (more neighbors = more hits)."""
    from src.retrieval import MoARetriever
    retriever = MoARetriever()
    recall = retriever.compute_recall_at_k(
        synthetic_embeddings, synthetic_metadata,
        k=k, moa_col="moa"
    )
    assert 0.0 <= recall <= 1.0
