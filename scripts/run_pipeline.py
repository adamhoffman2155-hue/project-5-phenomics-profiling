#!/usr/bin/env python3
"""
run_pipeline.py — Full Phenomics Perturbation Profiling Pipeline CLI.

Orchestrates the complete pipeline:
  1. Load RxRx3-core metadata and precomputed embeddings
  2. Apply TVN normalization
  3. Run UMAP (2D + 3D)
  4. Cluster with HDBSCAN and K-means
  5. Evaluate clustering quality
  6. Build MoA retrieval index and benchmark Recall@k / MAP
  7. Run Enrichr pathway enrichment per cluster
  8. Generate and save all figures
  9. Export results (cluster assignments, retrieval scores, pathway enrichment)

Usage
-----
    python scripts/run_pipeline.py --help

    # Quick run with synthetic data (no real data needed)
    python scripts/run_pipeline.py --synthetic --output results/

    # Full run with precomputed embeddings
    python scripts/run_pipeline.py \\
        --embeddings data/embeddings/openphenom_embeddings.npy \\
        --metadata data/raw/metadata.csv \\
        --output results/ \\
        --n_clusters 25 \\
        --clustering_method hdbscan \\
        --run_pathway_enrichment

    # Skip embedding re-generation (use cached normalized embeddings)
    python scripts/run_pipeline.py \\
        --embeddings data/embeddings/openphenom_embeddings.npy \\
        --metadata data/raw/metadata.csv \\
        --output results/ \\
        --skip_normalization \\
        --skip_figures
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.config import PipelineConfig
from src.data_loader import RxRx3DataLoader
from src.embeddings import EmbeddingProcessor
from src.clustering import PerturbationClusterer
from src.retrieval import MoARetriever
from src.pathway_analysis import PathwayAnalyzer
from src.visualization import PhenomicsVisualizer
from src.utils import setup_logging, save_embeddings, Timer, summarize_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phenomics Perturbation Profiling Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data inputs
    io_group = parser.add_argument_group("Data I/O")
    io_group.add_argument(
        "--embeddings", type=str, default=None,
        help="Path to precomputed embeddings .npy file.",
    )
    io_group.add_argument(
        "--embedding_index", type=str, default=None,
        help="Path to embedding index CSV (metadata aligned to embeddings).",
    )
    io_group.add_argument(
        "--metadata", type=str, default=None,
        help="Path to RxRx3-core metadata.csv.",
    )
    io_group.add_argument(
        "--output", type=str, default="results/",
        help="Root output directory for all results.",
    )
    io_group.add_argument(
        "--synthetic", action="store_true",
        help="Generate and use synthetic data (for testing/demo).",
    )

    # Embedding
    emb_group = parser.add_argument_group("Embeddings")
    emb_group.add_argument(
        "--skip_normalization", action="store_true",
        help="Skip TVN normalization (use embeddings as-is).",
    )
    emb_group.add_argument(
        "--embedding_dim", type=int, default=1536,
        help="Embedding dimensionality.",
    )

    # Clustering
    cl_group = parser.add_argument_group("Clustering")
    cl_group.add_argument(
        "--clustering_method", type=str, default="hdbscan",
        choices=["hdbscan", "kmeans", "both"],
        help="Clustering method to use.",
    )
    cl_group.add_argument(
        "--n_clusters", type=int, default=25,
        help="Number of clusters for K-means.",
    )
    cl_group.add_argument(
        "--hdbscan_min_cluster_size", type=int, default=50,
        help="HDBSCAN min_cluster_size.",
    )
    cl_group.add_argument(
        "--umap_n_neighbors", type=int, default=15,
        help="UMAP n_neighbors.",
    )
    cl_group.add_argument(
        "--no_3d_umap", action="store_true",
        help="Skip 3D UMAP computation.",
    )

    # Retrieval
    ret_group = parser.add_argument_group("MoA Retrieval")
    ret_group.add_argument(
        "--top_k", type=int, default=10,
        help="Top-k for MoA retrieval evaluation.",
    )
    ret_group.add_argument(
        "--skip_retrieval", action="store_true",
        help="Skip MoA retrieval benchmark.",
    )

    # Pathway enrichment
    path_group = parser.add_argument_group("Pathway Enrichment")
    path_group.add_argument(
        "--run_pathway_enrichment", action="store_true",
        help="Run Enrichr pathway enrichment per cluster (requires internet).",
    )
    path_group.add_argument(
        "--gene_sets", type=str, nargs="+",
        default=["KEGG_2021_Human", "Reactome_2022"],
        help="Enrichr gene set libraries to query.",
    )

    # Figures
    fig_group = parser.add_argument_group("Figures")
    fig_group.add_argument(
        "--skip_figures", action="store_true",
        help="Skip figure generation.",
    )
    fig_group.add_argument(
        "--no_interactive", action="store_true",
        help="Skip interactive Plotly UMAP.",
    )

    # Misc
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    misc_group.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    misc_group.add_argument(
        "--log_file", type=str, default=None,
        help="Path to log file.",
    )
    misc_group.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Number of parallel jobs (-1 = all CPUs).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full phenomics profiling pipeline."""
    start_time = time.time()

    # ------------------------------------------------------------------ setup
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.info("=" * 70)
    logger.info("Phenomics Perturbation Profiling Pipeline")
    logger.info("=" * 70)
    logger.info(f"Arguments: {vars(args)}")

    # Set random seeds
    np.random.seed(args.seed)

    # Build config
    cfg = PipelineConfig()
    cfg.clustering.kmeans_k = args.n_clusters
    cfg.clustering.hdbscan_min_cluster_size = args.hdbscan_min_cluster_size
    cfg.clustering.umap_n_neighbors = args.umap_n_neighbors
    cfg.retrieval.top_k = args.top_k
    cfg.pathway.gene_sets = args.gene_sets
    cfg.random_seed = args.seed

    # Create output directories
    out = Path(args.output)
    dirs = {
        "root": out,
        "clustering": out / "clustering",
        "retrieval": out / "retrieval",
        "pathway": out / "pathway_enrichment",
        "figures": out / "figures",
        "embeddings": out / "embeddings",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    logger.info("\n--- Step 1: Loading Data ---")
    with Timer("Data loading"):
        loader = RxRx3DataLoader(config=cfg)

        if args.synthetic or (args.embeddings is None and args.metadata is None):
            logger.info("Using synthetic data (no embeddings path provided).")
            embeddings, metadata = loader._generate_synthetic_embeddings(
                n_perturbations=500, embedding_dim=args.embedding_dim
            )
        else:
            metadata = loader.load_metadata(args.metadata)
            embeddings, emb_index = loader.load_embeddings(
                embeddings_file=args.embeddings,
                index_file=args.embedding_index,
            )
            # Use embedding index as metadata if richer
            if emb_index is not None and len(emb_index) == len(embeddings):
                metadata = emb_index

    logger.info(f"Data loaded: {embeddings.shape} embeddings, {len(metadata)} rows")
    summary = summarize_metadata(metadata)
    logger.info(f"Metadata summary: {json.dumps(summary, indent=2)}")

    # Save metadata summary
    with open(str(out / "metadata_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------ norm
    if not args.skip_normalization:
        logger.info("\n--- Step 2: TVN Normalization ---")
        with Timer("TVN normalization"):
            processor = EmbeddingProcessor(config=cfg)

            # Identify control wells
            controls_mask = np.zeros(len(metadata), dtype=bool)
            if "perturbation_type" in metadata.columns:
                controls_mask = (
                    metadata["perturbation_type"].values == "negcon"
                )

            if controls_mask.sum() > 0:
                logger.info(f"Found {controls_mask.sum()} negative control wells for TVN.")
            else:
                logger.warning("No negative controls found; skipping TVN, applying L2 norm only.")

            plate_ids = (
                metadata["plate"].values
                if "plate" in metadata.columns else None
            )

            embeddings = processor.apply_typical_variation_normalization(
                embeddings,
                controls_mask=controls_mask,
                per_plate=(plate_ids is not None),
                plate_ids=plate_ids,
            )

            # Save normalized embeddings
            save_embeddings(
                embeddings, metadata,
                output_dir=str(dirs["embeddings"]),
                prefix="normalized_embeddings",
            )

            # Compute and log quality metrics
            qual_metrics = processor.compute_embedding_quality_metrics(embeddings, metadata)
            with open(str(dirs["embeddings"] / "embedding_quality.json"), "w") as f:
                json.dump(qual_metrics, f, indent=2)
    else:
        logger.info("Skipping TVN normalization (--skip_normalization).")

    # ------------------------------------------------------------------ umap
    logger.info("\n--- Step 3: UMAP Dimensionality Reduction ---")
    with Timer("UMAP 2D"):
        clusterer = PerturbationClusterer(config=cfg)
        umap_2d = clusterer.run_umap(embeddings, n_components=2)
        np.save(str(dirs["clustering"] / "umap_2d_coords.npy"), umap_2d)

    if not args.no_3d_umap:
        with Timer("UMAP 3D"):
            umap_3d = clusterer.run_umap(embeddings, n_components=3)
            np.save(str(dirs["clustering"] / "umap_3d_coords.npy"), umap_3d)

    # ------------------------------------------------------------------ cluster
    logger.info("\n--- Step 4: Clustering ---")
    cluster_assignments = metadata.copy()
    labels_hdbscan = np.full(len(embeddings), -1, dtype=int)
    labels_kmeans = np.full(len(embeddings), -1, dtype=int)

    if args.clustering_method in ("hdbscan", "both"):
        with Timer("HDBSCAN"):
            labels_hdbscan = clusterer.run_hdbscan(umap_2d)
            np.save(str(dirs["clustering"] / "hdbscan_labels.npy"), labels_hdbscan)
            cluster_assignments["hdbscan_cluster"] = labels_hdbscan

        hdbscan_scores = clusterer.evaluate_clustering(umap_2d, labels_hdbscan)
        logger.info(f"HDBSCAN scores: {hdbscan_scores}")
        with open(str(dirs["clustering"] / "hdbscan_quality.json"), "w") as f:
            json.dump(hdbscan_scores, f, indent=2)

    if args.clustering_method in ("kmeans", "both"):
        with Timer("K-means"):
            labels_kmeans = clusterer.run_kmeans(embeddings, k=args.n_clusters)
            np.save(str(dirs["clustering"] / "kmeans_labels.npy"), labels_kmeans)
            cluster_assignments["kmeans_cluster"] = labels_kmeans

        kmeans_scores = clusterer.evaluate_clustering(embeddings, labels_kmeans)
        logger.info(f"K-means scores: {kmeans_scores}")
        with open(str(dirs["clustering"] / "kmeans_quality.json"), "w") as f:
            json.dump(kmeans_scores, f, indent=2)

        # Elbow method
        with Timer("Elbow method"):
            optimal_k, wcss = clusterer.find_optimal_k(
                embeddings,
                plot=True,
                save_path=str(dirs["clustering"] / "elbow_curve.png"),
            )
        logger.info(f"Optimal k (elbow): {optimal_k}")
        with open(str(dirs["clustering"] / "wcss_by_k.json"), "w") as f:
            json.dump(wcss, f, indent=2)

    # Use HDBSCAN labels by default, fall back to K-means
    primary_labels = (
        labels_hdbscan
        if args.clustering_method in ("hdbscan", "both")
        else labels_kmeans
    )

    # Add UMAP coords to cluster assignments
    cluster_assignments["umap_1"] = umap_2d[:, 0]
    cluster_assignments["umap_2"] = umap_2d[:, 1]

    cluster_assignments_path = str(dirs["clustering"] / "cluster_assignments.csv")
    cluster_assignments.to_csv(cluster_assignments_path, index=False)
    logger.info(f"Cluster assignments saved: {cluster_assignments_path}")

    # ------------------------------------------------------------------ retrieval
    if not args.skip_retrieval:
        logger.info("\n--- Step 5: MoA Retrieval Benchmark ---")
        with Timer("MoA retrieval"):
            retriever = MoARetriever(config=cfg)
            retrieval_results = retriever.evaluate_moa_retrieval(
                embeddings, metadata,
                save_path=str(dirs["retrieval"] / "recall_at_k.json"),
            )
            logger.info(f"Retrieval results: {retrieval_results}")

            # Nearest-neighbor table
            nn_table = retriever.get_nearest_neighbor_table(
                embeddings, metadata, top_k=args.top_k
            )
            nn_table.to_csv(
                str(dirs["retrieval"] / "nearest_neighbors.csv"), index=False
            )

            # Cross-modal matching
            if "perturbation_type" in metadata.columns:
                cross_modal = retriever.match_compounds_to_crispr(
                    embeddings, metadata,
                    top_k=10, similarity_threshold=0.5,
                )
                if len(cross_modal) > 0:
                    cross_modal.to_csv(
                        str(dirs["retrieval"] / "compound_crispr_matches.csv"),
                        index=False,
                    )
                    logger.info(
                        f"Cross-modal matches saved: {len(cross_modal)} pairs"
                    )
    else:
        logger.info("Skipping retrieval benchmark (--skip_retrieval).")
        retrieval_results = {}

    # ------------------------------------------------------------------ pathway
    enrichment_results = {}
    if args.run_pathway_enrichment:
        logger.info("\n--- Step 6: Pathway Enrichment ---")
        with Timer("Pathway enrichment"):
            analyzer = PathwayAnalyzer(config=cfg)
            cluster_genes = analyzer.get_cluster_gene_lists(
                metadata, primary_labels
            )
            logger.info(
                f"Cluster gene lists: "
                f"{sum(len(v) for v in cluster_genes.values())} total genes "
                f"across {len(cluster_genes)} clusters"
            )

            enrichment_results = analyzer.run_enrichr(
                cluster_genes,
                gene_sets=args.gene_sets,
                outdir=str(dirs["pathway"] / "enrichr_cache"),
            )

            # Save enrichment results
            analyzer.save_enrichment_results(
                enrichment_results,
                output_dir=str(dirs["pathway"]),
            )

            # Build cluster-pathway matrix
            if enrichment_results:
                matrix = analyzer.build_pathway_cluster_matrix(enrichment_results)
                if len(matrix) > 0:
                    matrix.to_csv(
                        str(dirs["pathway"] / "cluster_pathway_matrix.csv")
                    )

                jaccard = analyzer.compare_clusters_pathways(enrichment_results)
                if len(jaccard) > 0:
                    jaccard.to_csv(
                        str(dirs["pathway"] / "pathway_jaccard.csv")
                    )

                top_paths = analyzer.identify_cluster_pathways(enrichment_results)
                with open(str(dirs["pathway"] / "top_pathways_per_cluster.json"), "w") as f:
                    json.dump(
                        {str(k): v for k, v in top_paths.items()}, f, indent=2
                    )
    else:
        logger.info("Pathway enrichment skipped (use --run_pathway_enrichment to enable).")

    # ------------------------------------------------------------------ percent replicating
    logger.info("\n--- Step 7: Percent Replicating ---")
    try:
        pr_metrics = loader.compute_percent_replicating(embeddings, metadata)
        logger.info(f"Percent replicating: {pr_metrics}")
        with open(str(out / "percent_replicating.json"), "w") as f:
            json.dump(pr_metrics, f, indent=2)
    except Exception as e:
        logger.warning(f"Percent replicating computation failed: {e}")

    # ------------------------------------------------------------------ figures
    if not args.skip_figures:
        logger.info("\n--- Step 8: Generating Figures ---")
        with Timer("Figure generation"):
            viz = PhenomicsVisualizer(config=cfg)

            recall_at_k_dict = {
                int(k.split("_")[-1]): v
                for k, v in retrieval_results.items()
                if k.startswith("recall_at_")
                and k.split("_")[-1].isdigit()
            }

            saved_figures = viz.save_all_figures(
                umap_coords=umap_2d,
                metadata=metadata,
                labels=primary_labels,
                embeddings=embeddings,
                enrichment_results=enrichment_results if enrichment_results else None,
                recall_at_k=recall_at_k_dict if recall_at_k_dict else None,
                output_dir=str(dirs["figures"]),
                map_score=retrieval_results.get("map"),
            )
            logger.info(f"Figures saved: {saved_figures}")
    else:
        logger.info("Skipping figure generation (--skip_figures).")

    # ------------------------------------------------------------------ summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete!")
    logger.info(f"Total elapsed time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Output directory: {out.resolve()}")

    # Write pipeline summary JSON
    pipeline_summary = {
        "elapsed_seconds": elapsed,
        "n_perturbations": len(metadata),
        "embedding_shape": list(embeddings.shape),
        "n_hdbscan_clusters": int(
            len(set(primary_labels)) - (1 if -1 in primary_labels else 0)
        ),
        "retrieval": retrieval_results,
        "output_dir": str(out.resolve()),
        "args": vars(args),
    }
    with open(str(out / "pipeline_summary.json"), "w") as f:
        json.dump(pipeline_summary, f, indent=2)

    logger.info(f"Pipeline summary: {out / 'pipeline_summary.json'}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
