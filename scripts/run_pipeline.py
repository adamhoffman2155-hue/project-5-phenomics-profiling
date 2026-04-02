#!/usr/bin/env python3
"""Full phenomics profiling pipeline: data → embeddings → clustering → retrieval → enrichment → plots."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import PipelineConfig
from src.data_loader import generate_synthetic_data, validate_data
from src.embeddings import tvn_normalize, compute_embedding_quality
from src.clustering import run_umap, run_hdbscan, run_kmeans, evaluate_clustering
from src.retrieval import compute_recall_at_k, compute_map
from src.pathway_analysis import enrich_cluster_genes, build_pathway_cluster_matrix
from src.visualization import (
    plot_umap_clusters,
    plot_retrieval_performance,
    plot_cluster_metrics,
    plot_pathway_heatmap,
)
from src.utils import setup_logging, set_seed, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Phenomics Perturbation Profiling Pipeline")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-perturbations", type=int, default=300, help="Number of synthetic perturbations")
    args = parser.parse_args()

    logger = setup_logging("phenomics_pipeline")
    config = PipelineConfig()
    config.RANDOM_SEED = args.seed
    config.N_SYNTHETIC_PERTURBATIONS = args.n_perturbations
    set_seed(config.RANDOM_SEED)
    ensure_dir(args.output)
    ensure_dir(os.path.join(args.output, "figures"))

    # Step 1: Generate or load data
    logger.info("Step 1: Generating synthetic data...")
    metadata, embeddings = generate_synthetic_data(config)
    validate_data(metadata, embeddings)
    logger.info(f"  Data: {embeddings.shape[0]} perturbations, {embeddings.shape[1]}-dim embeddings")

    # Step 2: Normalize embeddings
    logger.info("Step 2: TVN normalization...")
    embeddings_norm = tvn_normalize(embeddings, metadata["plate_id"].values)
    quality = compute_embedding_quality(embeddings_norm)
    logger.info(f"  Embedding quality: variance={quality['mean_variance']:.4f}, sparsity={quality['sparsity']:.4f}")

    # Step 3: Clustering
    logger.info("Step 3: UMAP + clustering...")
    umap_coords = run_umap(embeddings_norm, n_components=2)
    hdbscan_labels = run_hdbscan(umap_coords, min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE)
    n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    logger.info(f"  HDBSCAN found {n_clusters} clusters ({(hdbscan_labels == -1).sum()} noise points)")

    cluster_metrics = evaluate_clustering(embeddings_norm, hdbscan_labels)
    logger.info(f"  Silhouette: {cluster_metrics.get('silhouette', 'N/A'):.3f}")

    # Step 4: MoA retrieval
    logger.info("Step 4: MoA retrieval evaluation...")
    recall_dict = compute_recall_at_k(embeddings_norm, metadata, config.TOP_K_VALUES)
    map_score = compute_map(embeddings_norm, metadata)
    logger.info(f"  MAP: {map_score:.3f}")
    for k, v in recall_dict.items():
        logger.info(f"  Recall@{k}: {v:.3f}")

    # Step 5: Pathway enrichment
    logger.info("Step 5: Pathway enrichment...")
    cluster_genes = {}
    for label in sorted(set(hdbscan_labels)):
        if label == -1:
            continue
        mask = hdbscan_labels == label
        genes = metadata.loc[mask, "gene_symbol"].dropna().unique().tolist()
        if genes:
            cluster_genes[f"cluster_{label}"] = genes
    
    pathway_matrix = None
    if cluster_genes:
        enrichment = {k: enrich_cluster_genes(v) for k, v in cluster_genes.items()}
        pathway_matrix = build_pathway_cluster_matrix(enrichment)

    # Step 6: Visualization
    logger.info("Step 6: Generating plots...")
    fig_dir = os.path.join(args.output, "figures")
    plot_umap_clusters(umap_coords, hdbscan_labels, os.path.join(fig_dir, "umap_clusters.png"))
    plot_retrieval_performance(recall_dict, os.path.join(fig_dir, "retrieval_recall.png"))
    plot_cluster_metrics(cluster_metrics, os.path.join(fig_dir, "cluster_metrics.png"))
    if pathway_matrix is not None:
        plot_pathway_heatmap(pathway_matrix, os.path.join(fig_dir, "pathway_heatmap.png"))

    logger.info(f"Pipeline complete. Results saved to {args.output}/")


if __name__ == "__main__":
    main()
