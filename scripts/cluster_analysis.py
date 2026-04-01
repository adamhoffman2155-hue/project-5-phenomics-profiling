#!/usr/bin/env python3
"""
cluster_analysis.py — Cluster phenomic embeddings and generate all visualizations.

Takes a precomputed embeddings .npy file, runs UMAP dimensionality reduction,
clusters with HDBSCAN and/or K-means, evaluates cluster quality, and generates
a full suite of publication-ready figures.

Usage
-----
    python scripts/cluster_analysis.py --help

    # HDBSCAN clustering with all figures
    python scripts/cluster_analysis.py \\
        --embeddings data/embeddings/openphenom_embeddings.npy \\
        --embedding_index data/embeddings/embedding_index.csv \\
        --output results/clustering/ \\
        --method hdbscan

    # K-means with elbow method k selection
    python scripts/cluster_analysis.py \\
        --embeddings data/embeddings/openphenom_embeddings.npy \\
        --embedding_index data/embeddings/embedding_index.csv \\
        --output results/clustering/ \\
        --method kmeans \\
        --find_optimal_k \\
        --k_min 5 --k_max 40

    # Quick test with synthetic data
    python scripts/cluster_analysis.py \\
        --synthetic \\
        --n_synthetic 300 \\
        --output results/test_clustering/ \\
        --method both

    # Load existing UMAP coords (skip recomputing)
    python scripts/cluster_analysis.py \\
        --embeddings data/embeddings/openphenom_embeddings.npy \\
        --umap_coords results/clustering/umap_2d_coords.npy \\
        --embedding_index data/embeddings/embedding_index.csv \\
        --output results/clustering/ \\
        --skip_umap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.config import PipelineConfig
from src.clustering import PerturbationClusterer
from src.visualization import PhenomicsVisualizer
from src.utils import setup_logging, load_embeddings_cached, Timer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster phenomic embeddings and generate visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    in_group = parser.add_argument_group("Input")
    in_group.add_argument(
        "--embeddings", type=str, default=None,
        help="Path to .npy embeddings file.",
    )
    in_group.add_argument(
        "--embedding_index", type=str, default=None,
        help="Path to index CSV aligned to embeddings.",
    )
    in_group.add_argument(
        "--umap_coords", type=str, default=None,
        help="Path to precomputed UMAP coords .npy (skip UMAP if provided).",
    )
    in_group.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic embeddings (for testing).",
    )
    in_group.add_argument(
        "--n_synthetic", type=int, default=300,
        help="Number of synthetic embeddings.",
    )

    # UMAP
    umap_group = parser.add_argument_group("UMAP")
    umap_group.add_argument("--skip_umap", action="store_true")
    umap_group.add_argument("--umap_n_components", type=int, default=2)
    umap_group.add_argument("--umap_n_neighbors", type=int, default=15)
    umap_group.add_argument("--umap_min_dist", type=float, default=0.1)
    umap_group.add_argument("--umap_metric", type=str, default="cosine")
    umap_group.add_argument("--compute_3d", action="store_true",
                            help="Also compute 3D UMAP.")

    # Clustering
    cl_group = parser.add_argument_group("Clustering")
    cl_group.add_argument(
        "--method", type=str, default="hdbscan",
        choices=["hdbscan", "kmeans", "agglomerative", "both"],
    )
    cl_group.add_argument("--n_clusters", type=int, default=25,
                          help="K for K-means / agglomerative.")
    cl_group.add_argument("--hdbscan_min_cluster_size", type=int, default=50)
    cl_group.add_argument("--hdbscan_min_samples", type=int, default=10)
    cl_group.add_argument("--find_optimal_k", action="store_true",
                          help="Run elbow method to find optimal K.")
    cl_group.add_argument("--k_min", type=int, default=5)
    cl_group.add_argument("--k_max", type=int, default=50)
    cl_group.add_argument("--k_step", type=int, default=5)

    # Figures
    fig_group = parser.add_argument_group("Figures")
    fig_group.add_argument("--skip_figures", action="store_true")
    fig_group.add_argument("--no_interactive", action="store_true")
    fig_group.add_argument("--dpi", type=int, default=150)
    fig_group.add_argument(
        "--color_by", type=str, default="perturbation_type",
        help="Metadata column to color UMAP by.",
    )

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output", type=str, default="results/clustering/",
                           help="Output directory.")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_or_generate_data(
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata, or generate synthetic data."""
    if args.synthetic:
        from src.data_loader import RxRx3DataLoader
        loader = RxRx3DataLoader()
        logger.info(f"Generating {args.n_synthetic} synthetic embeddings...")
        embeddings, metadata = loader._generate_synthetic_embeddings(
            n_perturbations=args.n_synthetic
        )
        return embeddings, metadata

    if args.embeddings is None:
        logger.error("--embeddings is required (or use --synthetic).")
        sys.exit(1)

    embeddings, metadata = load_embeddings_cached(
        embeddings_path=args.embeddings,
        index_path=args.embedding_index,
    )
    if metadata is None:
        metadata = pd.DataFrame({"embedding_id": np.arange(len(embeddings))})

    return embeddings, metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    logger.info("Cluster Analysis Script")
    logger.info(f"Args: {vars(args)}")

    np.random.seed(args.seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- load
    with Timer("Data loading"):
        embeddings, metadata = load_or_generate_data(args)
    logger.info(
        f"Loaded: embeddings={embeddings.shape}, metadata={len(metadata)} rows"
    )

    # ---------------------------------------------------------------- config
    cfg = PipelineConfig()
    cfg.clustering.umap_n_neighbors = args.umap_n_neighbors
    cfg.clustering.umap_min_dist = args.umap_min_dist
    cfg.clustering.umap_metric = args.umap_metric
    cfg.clustering.hdbscan_min_cluster_size = args.hdbscan_min_cluster_size
    cfg.clustering.hdbscan_min_samples = args.hdbscan_min_samples
    cfg.clustering.kmeans_k = args.n_clusters
    cfg.clustering.elbow_k_range_min = args.k_min
    cfg.clustering.elbow_k_range_max = args.k_max
    cfg.clustering.elbow_k_step = args.k_step
    cfg.random_seed = args.seed

    clusterer = PerturbationClusterer(config=cfg)

    # ---------------------------------------------------------------- umap
    if args.skip_umap and args.umap_coords:
        logger.info(f"Loading precomputed UMAP coords from {args.umap_coords}")
        umap_2d = np.load(args.umap_coords)
    else:
        with Timer("UMAP"):
            umap_2d = clusterer.run_umap(
                embeddings,
                n_components=args.umap_n_components,
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
            )
        np.save(str(out / "umap_2d_coords.npy"), umap_2d)
        logger.info(f"UMAP 2D saved: {out / 'umap_2d_coords.npy'}")

    if args.compute_3d:
        with Timer("UMAP 3D"):
            umap_3d = clusterer.run_umap(embeddings, n_components=3)
            np.save(str(out / "umap_3d_coords.npy"), umap_3d)

    # ---------------------------------------------------------------- cluster
    labels_hdbscan = np.full(len(embeddings), -1, dtype=int)
    labels_kmeans = np.full(len(embeddings), -1, dtype=int)
    all_quality_scores = {}

    if args.method in ("hdbscan", "both"):
        with Timer("HDBSCAN"):
            labels_hdbscan = clusterer.run_hdbscan(
                umap_2d,
                min_cluster_size=args.hdbscan_min_cluster_size,
                min_samples=args.hdbscan_min_samples,
            )
        np.save(str(out / "hdbscan_labels.npy"), labels_hdbscan)

        hdbscan_scores = clusterer.evaluate_clustering(umap_2d, labels_hdbscan)
        all_quality_scores["hdbscan"] = hdbscan_scores
        logger.info(
            f"HDBSCAN: n_clusters={hdbscan_scores.get('n_clusters', '?')}, "
            f"silhouette={hdbscan_scores.get('silhouette_score', '?'):.3f}"
        )

    if args.method in ("kmeans", "both"):
        if args.find_optimal_k:
            with Timer("Elbow method"):
                optimal_k, wcss = clusterer.find_optimal_k(
                    embeddings,
                    k_min=args.k_min,
                    k_max=args.k_max,
                    k_step=args.k_step,
                    plot=True,
                    save_path=str(out / "elbow_curve.png"),
                )
            logger.info(f"Optimal k (elbow): {optimal_k}")
            k_to_use = optimal_k
        else:
            k_to_use = args.n_clusters
            wcss = {}

        with Timer("K-means"):
            labels_kmeans = clusterer.run_kmeans(embeddings, k=k_to_use)
        np.save(str(out / "kmeans_labels.npy"), labels_kmeans)

        kmeans_scores = clusterer.evaluate_clustering(embeddings, labels_kmeans)
        all_quality_scores["kmeans"] = kmeans_scores
        logger.info(
            f"K-means (k={k_to_use}): "
            f"silhouette={kmeans_scores.get('silhouette_score', '?'):.3f}"
        )

        if wcss:
            with open(str(out / "wcss_by_k.json"), "w") as f:
                json.dump(wcss, f, indent=2)

    if args.method in ("agglomerative",):
        with Timer("Agglomerative"):
            labels_agglom = clusterer.run_agglomerative(embeddings, n_clusters=args.n_clusters)
        np.save(str(out / "agglomerative_labels.npy"), labels_agglom)
        agglom_scores = clusterer.evaluate_clustering(embeddings, labels_agglom)
        all_quality_scores["agglomerative"] = agglom_scores

    # Save quality scores
    with open(str(out / "clustering_quality.json"), "w") as f:
        json.dump(all_quality_scores, f, indent=2)

    # ---------------------------------------------------------------- cluster assignments CSV
    primary_labels = (
        labels_hdbscan
        if args.method in ("hdbscan", "both") else labels_kmeans
    )

    assignments = metadata.copy()
    assignments["umap_1"] = umap_2d[:, 0]
    assignments["umap_2"] = umap_2d[:, 1]
    if args.method in ("hdbscan", "both"):
        assignments["hdbscan_cluster"] = labels_hdbscan
    if args.method in ("kmeans", "both"):
        assignments["kmeans_cluster"] = labels_kmeans
    assignments["primary_cluster"] = primary_labels

    # Cluster labeling
    cluster_names = clusterer.label_clusters_by_enrichment(
        primary_labels, metadata
    )
    cluster_name_map = {
        cid: names[0] if names else f"C{cid:02d}"
        for cid, names in cluster_names.items()
    }
    assignments["cluster_label"] = [
        cluster_name_map.get(l, "noise") for l in primary_labels
    ]

    assignments_path = str(out / "cluster_assignments.csv")
    assignments.to_csv(assignments_path, index=False)
    logger.info(f"Cluster assignments saved: {assignments_path}")

    # Cluster member summary
    unique_clusters = sorted(set(primary_labels) - {-1})
    cluster_summary = []
    for cid in unique_clusters:
        mask = primary_labels == cid
        cluster_summary.append({
            "cluster_id": cid,
            "n_members": int(mask.sum()),
            "label": cluster_name_map.get(cid, f"C{cid:02d}"),
        })

    summary_df = pd.DataFrame(cluster_summary)
    summary_df.to_csv(str(out / "cluster_summary.csv"), index=False)
    logger.info(f"Cluster summary:\n{summary_df.to_string(index=False)}")

    # ---------------------------------------------------------------- figures
    if not args.skip_figures:
        logger.info("\n--- Generating Figures ---")
        with Timer("Figures"):
            viz = PhenomicsVisualizer(config=cfg, dpi=args.dpi)

            # UMAP by perturbation type
            fig = viz.plot_umap_by_perturbation_type(
                umap_2d, metadata,
                save_path=str(out / "umap_perturbation_type.png"),
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

            # UMAP by cluster
            fig = viz.plot_umap_by_cluster(
                umap_2d, primary_labels,
                cluster_names=cluster_name_map,
                save_path=str(out / "umap_clusters.png"),
            )
            plt.close(fig)

            # UMAP by MoA (if available)
            if "moa" in metadata.columns:
                fig = viz.plot_umap_by_moa(
                    umap_2d, metadata,
                    save_path=str(out / "umap_moa.png"),
                )
                plt.close(fig)

            # Cluster heatmap
            fig = viz.plot_cluster_heatmap(
                embeddings, primary_labels, metadata,
                save_path=str(out / "cluster_heatmap.png"),
            )
            if fig:
                plt.close(fig)

            # Interactive UMAP
            if not args.no_interactive:
                viz.create_interactive_umap(
                    umap_2d, metadata, primary_labels,
                    color_by=args.color_by,
                    save_path=str(out / "interactive_umap.html"),
                )

        logger.info(f"Figures saved to {out}")
    else:
        logger.info("Skipping figures (--skip_figures).")

    # ---------------------------------------------------------------- done
    logger.info(f"\nCluster analysis complete. Output: {out.resolve()}")
    logger.info(f"Primary clusters: {len(unique_clusters)}")
    for row in cluster_summary:
        logger.info(
            f"  Cluster {row['cluster_id']:3d}: "
            f"{row['n_members']:4d} members | {row['label']}"
        )


if __name__ == "__main__":
    main()
