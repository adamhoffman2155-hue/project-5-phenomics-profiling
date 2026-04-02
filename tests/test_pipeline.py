"""Tests for phenomics profiling pipeline components."""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import PipelineConfig


@pytest.fixture
def config():
    cfg = PipelineConfig()
    cfg.N_SYNTHETIC_PERTURBATIONS = 50
    return cfg


@pytest.fixture
def synthetic_data(config):
    from src.data_loader import generate_synthetic_data
    return generate_synthetic_data(config)


def test_synthetic_data_shape(synthetic_data, config):
    metadata, embeddings = synthetic_data
    assert embeddings.shape[0] == config.N_SYNTHETIC_PERTURBATIONS
    assert embeddings.shape[1] == config.EMBEDDING_DIM
    assert len(metadata) == config.N_SYNTHETIC_PERTURBATIONS


def test_synthetic_metadata_columns(synthetic_data):
    metadata, _ = synthetic_data
    required = ["perturbation_id", "perturbation_type", "gene_symbol", "moa_label", "plate_id"]
    for col in required:
        assert col in metadata.columns, f"Missing column: {col}"


def test_tvn_normalization(synthetic_data):
    from src.embeddings import tvn_normalize
    metadata, embeddings = synthetic_data
    normalized = tvn_normalize(embeddings, metadata["plate_id"].values)
    assert normalized.shape == embeddings.shape
    assert not np.isnan(normalized).any()
    # Should be roughly zero-centered after TVN
    assert abs(normalized.mean()) < 0.1


def test_embedding_quality(synthetic_data):
    from src.embeddings import compute_embedding_quality
    _, embeddings = synthetic_data
    quality = compute_embedding_quality(embeddings)
    assert "mean_variance" in quality
    assert "sparsity" in quality
    assert quality["mean_variance"] > 0


def test_clustering_labels(synthetic_data):
    from src.clustering import run_kmeans
    _, embeddings = synthetic_data
    labels = run_kmeans(embeddings, k=5)
    assert len(labels) == embeddings.shape[0]
    assert len(set(labels)) <= 5


def test_clustering_evaluation(synthetic_data):
    from src.clustering import run_kmeans, evaluate_clustering
    _, embeddings = synthetic_data
    labels = run_kmeans(embeddings, k=5)
    metrics = evaluate_clustering(embeddings, labels)
    assert "silhouette" in metrics
    assert "davies_bouldin" in metrics
    assert "calinski_harabasz" in metrics
    assert -1 <= metrics["silhouette"] <= 1


def test_cosine_retrieval_shape(synthetic_data):
    from src.retrieval import compute_cosine_similarity
    _, embeddings = synthetic_data
    query = embeddings[0:1]
    sims = compute_cosine_similarity(query, embeddings)
    assert sims.shape == (1, embeddings.shape[0])
    assert np.allclose(sims.max(), 1.0, atol=0.01)


def test_cosine_single_vector(synthetic_data):
    from src.retrieval import compute_cosine_similarity
    _, embeddings = synthetic_data
    sims = compute_cosine_similarity(embeddings[0], embeddings)
    assert sims.shape == (1, embeddings.shape[0])


def test_recall_at_k_range(synthetic_data, config):
    from src.retrieval import compute_recall_at_k
    metadata, embeddings = synthetic_data
    recall = compute_recall_at_k(embeddings, metadata, [1, 5])
    for k, v in recall.items():
        assert 0.0 <= v <= 1.0, f"Recall@{k} = {v} out of range"
    # Recall should be non-decreasing
    assert recall[1] <= recall[5] + 1e-9


def test_pathway_enrichment():
    from src.pathway_analysis import enrich_cluster_genes
    genes = ["BRCA1", "TP53", "ATM", "PTEN", "RB1"]
    result = enrich_cluster_genes(genes)
    assert isinstance(result, pd.DataFrame)
    assert "pathway" in result.columns
    assert "p_value" in result.columns
    assert len(result) > 0


def test_pathway_enrichment_semicolon():
    from src.pathway_analysis import enrich_cluster_genes
    genes = ["BAX;BCL2;CASP3", "TP53;BRCA1"]
    result = enrich_cluster_genes(genes)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_jaccard_similarity():
    from src.pathway_analysis import compute_jaccard_similarity
    assert compute_jaccard_similarity(["A", "B", "C"], ["B", "C", "D"]) == pytest.approx(0.5)
    assert compute_jaccard_similarity([], []) == 0.0
    assert compute_jaccard_similarity(["A"], ["A"]) == 1.0
