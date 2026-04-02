#!/usr/bin/env python3
"""Clustering-only pipeline: load embeddings, normalize, cluster, evaluate."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import PipelineConfig
from src.data_loader import generate_synthetic_data
from src.embeddings import tvn_normalize
from src.clustering import run_umap, run_hdbscan, run_kmeans, elbow_method, evaluate_clustering
from src.visualization import plot_umap_clusters, plot_cluster_metrics
from src.utils import setup_logging, set_seed, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Clustering analysis")
    parser.add_argument("--output", default="results/clustering", help="Output directory")
    parser.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan")
    parser.add_argument("--k", type=int, default=8, help="K for K-means")
    args = parser.parse_args()

    logger = setup_logging("clustering")
    config = PipelineConfig()
    set_seed(config.RANDOM_SEED)
    ensure_dir(args.output)

    metadata, embeddings = generate_synthetic_data(config)
    embeddings_norm = tvn_normalize(embeddings, metadata["plate_id"].values)
    umap_coords = run_umap(embeddings_norm)

    if args.method == "hdbscan":
        labels = run_hdbscan(umap_coords, min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE)
    else:
        labels = run_kmeans(embeddings_norm, k=args.k)

    metrics = evaluate_clustering(embeddings_norm, labels)
    logger.info(f"Clustering metrics: {metrics}")

    plot_umap_clusters(umap_coords, labels, os.path.join(args.output, "umap_clusters.png"))
    plot_cluster_metrics(metrics, os.path.join(args.output, "cluster_metrics.png"))
    logger.info("Done.")


if __name__ == "__main__":
    main()
