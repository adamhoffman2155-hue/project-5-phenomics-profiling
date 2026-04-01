#!/usr/bin/env python3
"""
embed_perturbations.py — Generate OpenPhenom embeddings from raw microscopy images.

Takes a directory of 5-channel TIFF images (or a metadata CSV pointing to images)
and runs the OpenPhenom ViT-MAE model to produce per-well embeddings, which are
then aggregated to per-perturbation level and saved as a .npy file.

Usage
-----
    python scripts/embed_perturbations.py --help

    # Embed from image directory
    python scripts/embed_perturbations.py \\
        --image_dir data/raw/images/ \\
        --metadata data/raw/metadata.csv \\
        --output data/embeddings/my_embeddings.npy \\
        --batch_size 64 \\
        --device cuda

    # Embed specific plates only
    python scripts/embed_perturbations.py \\
        --image_dir data/raw/images/ \\
        --metadata data/raw/metadata.csv \\
        --output data/embeddings/plate1_embeddings.npy \\
        --plates Plate1 Plate2 \\
        --batch_size 32

    # Test with synthetic images (no real data needed)
    python scripts/embed_perturbations.py \\
        --synthetic \\
        --n_synthetic 200 \\
        --output data/embeddings/test_embeddings.npy

Image format expected
---------------------
RxRx3-core images are 1080×1080 16-bit TIFFs with 5 channels stored as separate files:
    {experiment}/{plate}/{well}/w1.tif  (DAPI)
    {experiment}/{plate}/{well}/w2.tif  (ConA)
    {experiment}/{plate}/{well}/w3.tif  (SYTO14)
    {experiment}/{plate}/{well}/w4.tif  (WGA+Phalloidin)
    {experiment}/{plate}/{well}/w5.tif  (MitoTracker)

Or as multi-channel stacks:
    {experiment}/{plate}/{well}.tif  (shape: 5×H×W or H×W×5)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import PipelineConfig
from src.embeddings import EmbeddingProcessor, CHANNEL_NAMES, N_CHANNELS
from src.utils import setup_logging, save_embeddings, Timer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_rxrx3_image(
    image_path: str,
    target_size: int = 224,
    n_channels: int = N_CHANNELS,
) -> Optional[np.ndarray]:
    """
    Load a single RxRx3 image from disk.

    Supports:
    - Multi-channel TIFF stacks (shape H×W×5 or 5×H×W)
    - Directory of per-channel TIFFs (w1.tif, w2.tif, ..., w5.tif)

    Parameters
    ----------
    image_path : str
        Path to TIFF file or directory containing per-channel files.
    target_size : int
        Output spatial size (images resized to target_size × target_size).
    n_channels : int
        Expected number of channels.

    Returns
    -------
    np.ndarray, shape (n_channels, target_size, target_size), dtype float32
        Normalized to [0, 1].
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("pillow is required for image loading.")

    path = Path(image_path)

    # Case 1: directory of per-channel files
    if path.is_dir():
        channel_files = sorted(path.glob("w*.tif")) + sorted(path.glob("w*.tiff"))
        if len(channel_files) == 0:
            channel_files = sorted(path.glob("channel_*.tif"))

        if len(channel_files) < n_channels:
            logger.warning(
                f"Expected {n_channels} channel files in {path}, "
                f"found {len(channel_files)}. Padding with zeros."
            )

        channels = []
        for i in range(n_channels):
            if i < len(channel_files):
                ch_img = np.array(Image.open(str(channel_files[i])), dtype=np.float32)
                if ch_img.ndim == 3:
                    ch_img = ch_img.mean(axis=-1)  # collapse if RGB
            else:
                ch_img = np.zeros((target_size, target_size), dtype=np.float32)

            # Percentile normalization (2nd–98th percentile)
            p2 = np.percentile(ch_img, 2)
            p98 = np.percentile(ch_img, 98)
            if p98 > p2:
                ch_img = (ch_img - p2) / (p98 - p2)
            ch_img = np.clip(ch_img, 0, 1)

            # Resize to target_size
            if ch_img.shape != (target_size, target_size):
                pil_img = Image.fromarray((ch_img * 255).astype(np.uint8))
                pil_img = pil_img.resize(
                    (target_size, target_size), Image.Resampling.LANCZOS
                )
                ch_img = np.array(pil_img, dtype=np.float32) / 255.0

            channels.append(ch_img)

        return np.stack(channels, axis=0)

    # Case 2: single multi-channel TIFF
    elif path.suffix.lower() in (".tif", ".tiff"):
        try:
            img = np.array(Image.open(str(path)), dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

        # Handle various shapes
        if img.ndim == 2:
            # Grayscale — duplicate to n_channels
            img = np.stack([img] * n_channels, axis=0)
        elif img.ndim == 3:
            if img.shape[0] == n_channels:
                pass  # (C, H, W) — correct
            elif img.shape[-1] == n_channels:
                img = img.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
            elif img.shape[0] < n_channels:
                # Pad channels
                pad = np.zeros(
                    (n_channels - img.shape[0], img.shape[1], img.shape[2]),
                    dtype=np.float32,
                )
                img = np.concatenate([img, pad], axis=0)
            else:
                img = img[:n_channels]
        elif img.ndim == 4:
            img = img[0]  # Take first frame/time point

        # Normalize each channel independently
        for c in range(n_channels):
            ch = img[c]
            p2 = np.percentile(ch, 2)
            p98 = np.percentile(ch, 98)
            if p98 > p2:
                img[c] = np.clip((ch - p2) / (p98 - p2), 0, 1)
            else:
                img[c] = np.zeros_like(ch)

        # Resize
        if img.shape[1:] != (target_size, target_size):
            channels = []
            for c in range(n_channels):
                pil_img = Image.fromarray((img[c] * 255).astype(np.uint8))
                pil_img = pil_img.resize(
                    (target_size, target_size), Image.Resampling.LANCZOS
                )
                channels.append(np.array(pil_img, dtype=np.float32) / 255.0)
            img = np.stack(channels, axis=0)

        return img.astype(np.float32)

    else:
        logger.warning(f"Unsupported file format: {path.suffix}")
        return None


def discover_images(
    image_dir: str,
    metadata: pd.DataFrame,
    plates: Optional[List[str]] = None,
) -> List[Tuple[int, str]]:
    """
    Discover image files that correspond to metadata rows.

    Walks the image directory and matches files to metadata using
    plate/well identifiers. Returns a list of (metadata_row_idx, image_path).

    Parameters
    ----------
    image_dir : str
    metadata : pd.DataFrame
    plates : list of str, optional
        If provided, only include these plates.

    Returns
    -------
    list of (row_idx, image_path) tuples
    """
    img_dir = Path(image_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    plate_col = "plate" if "plate" in metadata.columns else None
    well_col = "well" if "well" in metadata.columns else None

    found = []
    for row_idx, row in metadata.iterrows():
        plate = str(row[plate_col]) if plate_col else "Plate1"
        well = str(row[well_col]) if well_col else f"row_{row_idx}"

        if plates and plate not in plates:
            continue

        # Try common RxRx3 path patterns
        candidate_paths = [
            img_dir / plate / well,                    # directory of per-channel TIFFs
            img_dir / plate / f"{well}.tif",           # single stack
            img_dir / f"{plate}_{well}.tif",            # flat structure
        ]

        image_path = None
        for cp in candidate_paths:
            if cp.exists():
                image_path = str(cp)
                break

        if image_path is None:
            logger.debug(f"No image found for plate={plate}, well={well}")
            continue

        found.append((row_idx, image_path))

    logger.info(f"Discovered {len(found)}/{len(metadata)} image–metadata pairs")
    return found


def generate_synthetic_images(
    n: int,
    n_channels: int = N_CHANNELS,
    image_size: int = 224,
    n_clusters: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate synthetic 5-channel microscopy images with cluster structure.

    Creates images with realistic cell-painting-like intensity patterns:
    cluster-specific "expression" patterns per channel, with Gaussian noise.

    Parameters
    ----------
    n : int
        Number of images.
    n_channels : int
    image_size : int
    n_clusters : int
    seed : int

    Returns
    -------
    images : np.ndarray, shape (n, n_channels, image_size, image_size)
    metadata : pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    # Each cluster has a distinct intensity profile across channels
    cluster_profiles = rng.uniform(0.1, 0.9, size=(n_clusters, n_channels))

    images = np.zeros((n, n_channels, image_size, image_size), dtype=np.float32)
    meta_rows = []

    for i in range(n):
        cluster_id = i % n_clusters
        profile = cluster_profiles[cluster_id]

        for c in range(n_channels):
            # Base intensity from cluster profile + noise
            base = profile[c]
            noise = rng.normal(0, 0.05, size=(image_size, image_size))
            # Add simulated "cells" as bright spots
            n_cells = rng.integers(5, 25)
            cell_img = np.full((image_size, image_size), base * 0.3, dtype=np.float32)
            for _ in range(n_cells):
                cy, cx = rng.integers(10, image_size - 10, size=2)
                radius = rng.integers(3, 12)
                y, x = np.ogrid[:image_size, :image_size]
                dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                cell_img += base * np.exp(-dist ** 2 / (2 * radius ** 2))

            images[i, c] = np.clip(cell_img + noise.astype(np.float32), 0, 1)

        # Metadata row
        plate_num = i // 96 + 1
        row_letter = chr(65 + (i % 16))
        col_num = (i // 16) % 24 + 1
        meta_rows.append({
            "experiment": "HUVEC-SYN-1",
            "plate": f"Plate{plate_num}",
            "well": f"{row_letter}{col_num:02d}",
            "site": 1,
            "cell_line": "HUVEC",
            "perturbation_type": "CRISPR" if i % 3 != 0 else "compound",
            "gene": f"GENE_{cluster_id:03d}" if i % 3 != 0 else None,
            "compound": f"COMPOUND_{cluster_id:02d}" if i % 3 == 0 else None,
            "concentration_um": 1.0 if i % 3 == 0 else None,
            "replicate": (i % 4) + 1,
            "embedding_id": i,
        })

    metadata = pd.DataFrame(meta_rows)
    logger.info(
        f"Generated {n} synthetic images: "
        f"shape={images.shape}, {n_clusters} cluster patterns"
    )
    return images, metadata


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate OpenPhenom embeddings from RxRx3 microscopy images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--image_dir", type=str, default=None,
        help="Directory containing RxRx3 microscopy images.",
    )
    input_group.add_argument(
        "--metadata", type=str, default=None,
        help="Path to metadata CSV with plate/well information.",
    )
    input_group.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic images (no real data needed, for testing).",
    )
    input_group.add_argument(
        "--n_synthetic", type=int, default=200,
        help="Number of synthetic images to generate.",
    )
    input_group.add_argument(
        "--plates", type=str, nargs="+", default=None,
        help="Only process these plates (by plate name).",
    )

    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model", type=str, default="openphenom",
        choices=["openphenom", "stub"],
        help="Embedding model to use.",
    )
    model_group.add_argument(
        "--model_cache_dir", type=str, default="models/",
        help="Directory to cache model weights.",
    )
    model_group.add_argument(
        "--device", type=str, default="cuda",
        help="PyTorch device: 'cuda' or 'cpu'.",
    )
    model_group.add_argument(
        "--batch_size", type=int, default=64,
        help="Inference batch size.",
    )
    model_group.add_argument(
        "--image_size", type=int, default=224,
        help="Target image size for model input (resized from raw).",
    )

    # Aggregation
    agg_group = parser.add_argument_group("Aggregation")
    agg_group.add_argument(
        "--aggregate_replicates", action="store_true", default=True,
        help="Aggregate replicate wells to perturbation-level embeddings.",
    )
    agg_group.add_argument(
        "--agg_method", type=str, default="mean",
        choices=["mean", "median"],
    )
    agg_group.add_argument(
        "--no_tvn", action="store_true",
        help="Skip TVN normalization.",
    )

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output", type=str, default="data/embeddings/embeddings.npy",
        help="Output path for embeddings .npy file.",
    )
    out_group.add_argument(
        "--output_index", type=str, default=None,
        help="Output path for index CSV. Defaults to same dir as --output.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    logger.info("OpenPhenom Embedding Script")
    logger.info(f"Args: {vars(args)}")

    np.random.seed(args.seed)
    t0 = time.time()

    # ---------------------------------------------------------------- data
    if args.synthetic:
        logger.info(f"Generating {args.n_synthetic} synthetic images...")
        images, metadata = generate_synthetic_images(
            n=args.n_synthetic,
            image_size=args.image_size,
            seed=args.seed,
        )
        image_meta_pairs = [(i, None) for i in range(len(images))]
        preloaded_images = images

    else:
        if not args.image_dir:
            logger.error("--image_dir is required when not using --synthetic")
            sys.exit(1)

        if args.metadata:
            metadata = pd.read_csv(args.metadata)
        else:
            logger.warning("No metadata CSV provided. Creating minimal metadata.")
            metadata = pd.DataFrame({"embedding_id": [], "plate": [], "well": []})

        image_meta_pairs = discover_images(
            args.image_dir, metadata, plates=args.plates
        )
        if len(image_meta_pairs) == 0:
            logger.error("No image–metadata pairs found. Check --image_dir and --metadata.")
            sys.exit(1)

        preloaded_images = None

    # ---------------------------------------------------------------- model
    cfg = PipelineConfig()
    cfg.embedding.batch_size = args.batch_size
    cfg.embedding.image_size = args.image_size
    cfg.embedding.device = args.device

    processor = EmbeddingProcessor(config=cfg, device=args.device)
    use_stub = (args.model == "stub") or args.synthetic
    model = processor.load_openphenom_model(
        model_name=args.model,
        cache_dir=args.model_cache_dir,
        use_stub=use_stub,
    )

    # ---------------------------------------------------------------- embed
    logger.info("\n--- Embedding ---")
    with Timer("Embedding"):
        if preloaded_images is not None:
            # Synthetic images already in memory
            embeddings = processor.embed_images(
                preloaded_images, model=model, batch_size=args.batch_size
            )
        else:
            # Stream images from disk in batches
            all_embs = []
            valid_indices = []

            for batch_start in range(0, len(image_meta_pairs), args.batch_size):
                batch_pairs = image_meta_pairs[batch_start:batch_start + args.batch_size]
                batch_imgs = []
                batch_valid_idx = []

                for row_idx, img_path in batch_pairs:
                    img = load_rxrx3_image(img_path, target_size=args.image_size)
                    if img is not None:
                        batch_imgs.append(img)
                        batch_valid_idx.append(row_idx)

                if not batch_imgs:
                    continue

                batch_array = np.stack(batch_imgs)
                batch_embs = processor.embed_images(
                    batch_array, model=model, batch_size=args.batch_size,
                    show_progress=False,
                )
                all_embs.append(batch_embs)
                valid_indices.extend(batch_valid_idx)

                logger.info(
                    f"Processed {len(valid_indices)}/{len(image_meta_pairs)} images"
                )

            if not all_embs:
                logger.error("No embeddings generated.")
                sys.exit(1)

            embeddings = np.vstack(all_embs)
            metadata = metadata.iloc[valid_indices].reset_index(drop=True)
            metadata["embedding_id"] = np.arange(len(metadata))

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # ---------------------------------------------------------------- normalize
    if not args.no_tvn:
        logger.info("Applying TVN normalization...")
        controls_mask = np.zeros(len(metadata), dtype=bool)
        if "perturbation_type" in metadata.columns:
            controls_mask = metadata["perturbation_type"].values == "negcon"

        with Timer("TVN"):
            embeddings = processor.apply_typical_variation_normalization(
                embeddings, controls_mask=controls_mask
            )

    # ---------------------------------------------------------------- save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    index_path = args.output_index or str(
        output_path.parent / (output_path.stem + "_index.csv")
    )

    npy_path, csv_path = save_embeddings(
        embeddings, metadata,
        output_dir=str(output_path.parent),
        prefix=output_path.stem,
    )

    elapsed = time.time() - t0
    logger.info(f"\nDone! Elapsed: {elapsed:.1f}s")
    logger.info(f"Embeddings saved: {npy_path}")
    logger.info(f"Index saved: {csv_path}")
    logger.info(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
