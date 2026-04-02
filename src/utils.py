"""
Utility helpers for the Phenomics Profiling Pipeline.

Provides logging setup, random-seed management, directory creation,
and a batched cosine-similarity implementation.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def setup_logging(
    level_or_name: int | str = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return the pipeline root logger.

    Parameters
    ----------
    level_or_name : int or str
        Logging level (e.g. ``logging.INFO``) **or** a logger name string.
        When a string is passed the level defaults to ``INFO`` and the
        string is used as the logger name suffix.
    log_file : str, optional
        If provided, also write logs to this file.

    Returns
    -------
    logging.Logger
    """
    if isinstance(level_or_name, str):
        logger_name = f"phenomics.{level_or_name}"
        level = logging.INFO
    else:
        logger_name = "phenomics"
        level = level_or_name

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s \u2014 %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def set_seed(seed: int) -> None:
    """Set the global random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Integer seed value.
    """
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not already exist.

    Returns
    -------
    str
        The same *path* for convenience.
    """
    os.makedirs(path, exist_ok=True)
    return path


def batched_cosine_similarity(
    queries: NDArray[np.float64],
    database: NDArray[np.float64],
    batch_size: int = 128,
) -> NDArray[np.float64]:
    """Compute cosine similarity between *queries* and *database* in batches.

    Parameters
    ----------
    queries : ndarray of shape (n_queries, dim)
    database : ndarray of shape (n_database, dim)
    batch_size : int
        Number of query rows processed at once.

    Returns
    -------
    ndarray of shape (n_queries, n_database)
        Cosine similarity matrix.
    """
    queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    database_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-12)

    n = queries_norm.shape[0]
    sim = np.empty((n, database_norm.shape[0]), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim[start:end] = queries_norm[start:end] @ database_norm.T

    return sim
