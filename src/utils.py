"""
Shared utility functions for the Phenomics Perturbation Profiling Pipeline.

Provides:
- setup_logging(): configure structured logging
- cosine_similarity_matrix(): pairwise cosine similarity
- batch_iterator(): memory-efficient batching
- load_gene_sets(): load Enrichr/KEGG gene set dictionaries
- save_embeddings() / load_embeddings_cached(): fast .npy embedding I/O
- format_gene_list(): clean gene symbol lists
- compute_jaccard(): Jaccard similarity between sets
- Timer: context manager for execution timing
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Configure structured logging for the pipeline.

    Sets up a root logger with a StreamHandler (always) and optionally a
    FileHandler. Uses a clean format with timestamps.

    Parameters
    ----------
    level : str
        Log level: "DEBUG", "INFO", "WARNING", "ERROR".
    log_file : str, optional
        Path to write log file. If None, logs go to stderr only.
    format_str : str, optional
        Custom log format string.

    Returns
    -------
    logging.Logger
        Configured root logger.

    Examples
    --------
    >>> logger = setup_logging(level="INFO", log_file="pipeline.log")
    >>> logger.info("Pipeline started.")
    """
    if format_str is None:
        format_str = (
            "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s"
        )

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: List[logging.Handler] = [
        logging.StreamHandler()
    ]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Quiet noisy third-party loggers
    for noisy in ("matplotlib", "PIL", "urllib3", "requests", "timm", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger("phenomics")
    logger.setLevel(numeric_level)
    return logger


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(
    a: np.ndarray,
    b: Optional[np.ndarray] = None,
    normalize: bool = True,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Compute the pairwise cosine similarity matrix between rows of `a` and `b`.

    If `b` is None, computes the self-similarity matrix of `a`.
    Uses batched matrix multiplication to avoid memory OOM for large arrays.

    Parameters
    ----------
    a : np.ndarray, shape (N, D)
    b : np.ndarray, shape (M, D), optional
        If None, b = a (self-similarity).
    normalize : bool
        If True, L2-normalize rows before computing dot products.
    batch_size : int
        Row batch size for `a` to avoid memory issues.

    Returns
    -------
    np.ndarray, shape (N, M)
        Cosine similarity matrix in [-1, 1].

    Examples
    --------
    >>> a = np.random.randn(100, 1536).astype(np.float32)
    >>> sim = cosine_similarity_matrix(a)
    >>> assert sim.shape == (100, 100)
    >>> assert np.allclose(np.diag(sim), 1.0, atol=1e-5)
    """
    if b is None:
        b = a

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    if normalize:
        a_norms = np.linalg.norm(a, axis=1, keepdims=True)
        a_norms = np.where(a_norms == 0, 1.0, a_norms)
        a = a / a_norms

        b_norms = np.linalg.norm(b, axis=1, keepdims=True)
        b_norms = np.where(b_norms == 0, 1.0, b_norms)
        b = b / b_norms

    N = len(a)
    M = len(b)
    result = np.empty((N, M), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        result[start:end] = a[start:end] @ b.T

    return result


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def batch_iterator(
    data: Union[np.ndarray, list, pd.DataFrame],
    batch_size: int,
    drop_last: bool = False,
) -> Iterator[Union[np.ndarray, list, pd.DataFrame]]:
    """
    Yield successive batches from `data`.

    Parameters
    ----------
    data : array-like
        Any indexable sequence (numpy array, list, DataFrame).
    batch_size : int
        Number of elements per batch.
    drop_last : bool
        If True, drop the last incomplete batch.

    Yields
    ------
    Batches of the same type as `data`.

    Examples
    --------
    >>> for batch in batch_iterator(np.arange(100), batch_size=32):
    ...     print(batch.shape)
    (32,)
    (32,)
    (32,)
    (4,)
    """
    n = len(data)
    for start in range(0, n, batch_size):
        end = start + batch_size
        if drop_last and end > n:
            break
        yield data[start:end]


# ---------------------------------------------------------------------------
# Gene sets
# ---------------------------------------------------------------------------

def load_gene_sets(
    gene_set_name: str = "KEGG_2021_Human",
    cache_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Load a gene set library as a dictionary.

    Attempts to load from a local JSON cache first, then fetches from
    the Enrichr API.

    Parameters
    ----------
    gene_set_name : str
        Enrichr library name, e.g. "KEGG_2021_Human", "Reactome_2022".
    cache_dir : str, optional
        Directory to cache downloaded gene sets.

    Returns
    -------
    dict mapping pathway_name → list of gene symbols

    Examples
    --------
    >>> gene_sets = load_gene_sets("KEGG_2021_Human")
    >>> print(gene_sets["DNA Repair"])
    ['BRCA1', 'BRCA2', 'ATM', ...]
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "phenomics_profiling" / "gene_sets"
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{gene_set_name}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            gene_sets = json.load(f)
        logging.getLogger(__name__).info(
            f"Loaded {len(gene_sets)} gene sets from cache: {cache_file}"
        )
        return gene_sets

    # Try fetching from Enrichr API
    try:
        import requests
        url = (
            f"https://maayanlab.cloud/Enrichr/geneSetLibrary"
            f"?mode=text&libraryName={gene_set_name}"
        )
        logging.getLogger(__name__).info(f"Fetching gene sets from Enrichr: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        gene_sets: Dict[str, List[str]] = {}
        for line in response.text.strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pathway = parts[0]
            # Enrichr format: pathway\t\tgene1\tgene2\t...
            genes = [g.strip() for g in parts[2:] if g.strip()]
            gene_sets[pathway] = genes

        # Cache to disk
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(gene_sets, f)
        logging.getLogger(__name__).info(
            f"Cached {len(gene_sets)} gene sets to {cache_file}"
        )
        return gene_sets

    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Could not fetch {gene_set_name} from Enrichr ({e}). "
            "Returning empty gene set dict."
        )
        return {}


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------

def save_embeddings(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str,
    prefix: str = "embeddings",
) -> Tuple[str, str]:
    """
    Save embeddings array and aligned metadata index to disk.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
    metadata : pd.DataFrame, shape (N,)
    output_dir : str
    prefix : str
        Filename prefix.

    Returns
    -------
    tuple of (npy_path, csv_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    npy_path = str(out / f"{prefix}.npy")
    csv_path = str(out / f"{prefix}_index.csv")

    np.save(npy_path, embeddings.astype(np.float32))
    metadata.to_csv(csv_path, index=False)

    logging.getLogger(__name__).info(
        f"Saved embeddings: {npy_path} ({embeddings.shape}), "
        f"index: {csv_path} ({len(metadata)} rows)"
    )
    return npy_path, csv_path


def load_embeddings_cached(
    embeddings_path: str,
    index_path: Optional[str] = None,
    mmap_mode: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """
    Load embeddings from a .npy file with optional memory mapping.

    Memory mapping (mmap_mode="r") allows loading large embedding files
    without reading them fully into RAM — useful for >1M perturbation datasets.

    Parameters
    ----------
    embeddings_path : str
        Path to .npy embeddings file.
    index_path : str, optional
        Path to companion index CSV. If None, returns None for metadata.
    mmap_mode : str, optional
        NumPy memory-map mode: None (load fully), "r" (read-only mmap),
        "r+" (read-write mmap).

    Returns
    -------
    embeddings : np.ndarray
    metadata : pd.DataFrame or None
    """
    emb_path = Path(embeddings_path)
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    embeddings = np.load(str(emb_path), mmap_mode=mmap_mode)
    logging.getLogger(__name__).info(
        f"Loaded embeddings: {embeddings.shape} from {emb_path}"
    )

    metadata = None
    if index_path and Path(index_path).exists():
        metadata = pd.read_csv(index_path)
        logging.getLogger(__name__).info(
            f"Loaded index: {len(metadata)} rows from {index_path}"
        )

    return embeddings, metadata


# ---------------------------------------------------------------------------
# Gene list utilities
# ---------------------------------------------------------------------------

def format_gene_list(
    genes: Union[List[str], pd.Series, np.ndarray],
    deduplicate: bool = True,
    uppercase: bool = True,
    remove_empty: bool = True,
    max_genes: Optional[int] = None,
) -> List[str]:
    """
    Clean and format a gene symbol list.

    Parameters
    ----------
    genes : list-like
        Raw gene symbols (may contain NaN, duplicates, mixed case).
    deduplicate : bool
    uppercase : bool
    remove_empty : bool
    max_genes : int, optional
        Truncate to this many genes (for Enrichr API limits).

    Returns
    -------
    list of str

    Examples
    --------
    >>> format_gene_list(["brca1", "BRCA2", None, "brca1", ""])
    ['BRCA1', 'BRCA2']
    """
    if isinstance(genes, (pd.Series, np.ndarray)):
        genes = genes.tolist()

    result = []
    for g in genes:
        if g is None or (isinstance(g, float) and np.isnan(g)):
            continue
        g_str = str(g).strip()
        if remove_empty and not g_str:
            continue
        if g_str.lower() in ("nan", "none", "null", "na", "n/a"):
            continue
        if uppercase:
            g_str = g_str.upper()
        result.append(g_str)

    if deduplicate:
        seen: Set[str] = set()
        unique = []
        for g in result:
            if g not in seen:
                seen.add(g)
                unique.append(g)
        result = unique

    if max_genes is not None:
        result = result[:max_genes]

    return result


# ---------------------------------------------------------------------------
# Set similarity
# ---------------------------------------------------------------------------

def compute_jaccard(
    set_a: Union[set, List],
    set_b: Union[set, List],
) -> float:
    """
    Compute Jaccard similarity coefficient between two sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    set_a, set_b : set or list-like

    Returns
    -------
    float in [0, 1]. Returns 0.0 if both sets are empty.

    Examples
    --------
    >>> compute_jaccard({"BRCA1", "BRCA2"}, {"BRCA2", "ATM"})
    0.333...
    """
    a = set(set_a)
    b = set(set_b)
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def compute_jaccard_matrix(
    gene_lists: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Compute pairwise Jaccard similarity matrix for a collection of gene lists.

    Parameters
    ----------
    gene_lists : dict
        Mapping of name → list of gene symbols.

    Returns
    -------
    pd.DataFrame, shape (N, N)
        Symmetric Jaccard similarity matrix.
    """
    names = sorted(gene_lists.keys())
    n = len(names)
    matrix = np.zeros((n, n), dtype=np.float32)

    sets = {name: set(genes) for name, genes in gene_lists.items()}

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            matrix[i, j] = compute_jaccard(sets[ni], sets[nj])

    return pd.DataFrame(matrix, index=names, columns=names)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """
    Context manager for timing code blocks.

    Examples
    --------
    >>> with Timer("UMAP fitting") as t:
    ...     umap_coords = reducer.fit_transform(embeddings)
    >>> print(f"Elapsed: {t.elapsed:.2f}s")
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start
        label = f"[{self.name}] " if self.name else ""
        logging.getLogger(__name__).info(
            f"{label}Elapsed: {self.elapsed:.2f}s"
        )


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for NumPy and Python's random module for reproducibility.

    Note: Does NOT set torch seeds (call torch.manual_seed separately).

    Parameters
    ----------
    seed : int
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_memory_usage_mb() -> float:
    """Return current process memory usage in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        return -1.0


def summarize_metadata(metadata: pd.DataFrame) -> Dict[str, object]:
    """
    Generate a concise summary dictionary of a metadata DataFrame.

    Parameters
    ----------
    metadata : pd.DataFrame

    Returns
    -------
    dict
    """
    summary: Dict[str, object] = {
        "n_rows": len(metadata),
        "n_columns": len(metadata.columns),
        "columns": list(metadata.columns),
    }

    if "perturbation_type" in metadata.columns:
        summary["perturbation_type_counts"] = (
            metadata["perturbation_type"].value_counts().to_dict()
        )

    if "gene" in metadata.columns:
        summary["n_unique_genes"] = int(metadata["gene"].dropna().nunique())

    if "compound" in metadata.columns:
        summary["n_unique_compounds"] = int(metadata["compound"].dropna().nunique())

    if "plate" in metadata.columns:
        summary["n_plates"] = int(metadata["plate"].nunique())

    if "moa" in metadata.columns:
        summary["n_unique_moa"] = int(metadata["moa"].dropna().nunique())

    return summary
