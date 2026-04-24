#!/usr/bin/env python3
"""MoA retrieval evaluation pipeline."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import PipelineConfig
from src.data_loader import generate_synthetic_data
from src.embeddings import tvn_normalize
from src.retrieval import compute_map, compute_recall_at_k
from src.utils import ensure_dir, set_seed, setup_logging
from src.visualization import plot_retrieval_performance


def main():
    parser = argparse.ArgumentParser(description="MoA retrieval evaluation")
    parser.add_argument("--output", default="results/retrieval", help="Output directory")
    args = parser.parse_args()

    logger = setup_logging("retrieval")
    config = PipelineConfig()
    set_seed(config.RANDOM_SEED)
    ensure_dir(args.output)

    metadata, embeddings = generate_synthetic_data(config)
    embeddings_norm = tvn_normalize(embeddings, metadata["plate_id"].values)

    recall_dict = compute_recall_at_k(embeddings_norm, metadata, config.TOP_K_VALUES)
    map_score = compute_map(embeddings_norm, metadata)

    logger.info(f"MAP: {map_score:.3f}")
    for k, v in recall_dict.items():
        logger.info(f"Recall@{k}: {v:.3f}")

    plot_retrieval_performance(recall_dict, os.path.join(args.output, "recall_at_k.png"))
    logger.info("Done.")


if __name__ == "__main__":
    main()
