"""
Retrieval evaluation for the Phenomics Profiling Pipeline.

Implements cosine-similarity-based retrieval, Recall@k, Mean Average
Precision (mAP), and cross-modal (compound-to-CRISPR) matching.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger("phenomics.retrieval")


# ------------------------------------------------------------------
# Cosine similarity
# ------------------------------------------------------------------

def compute_cosine_similarity(
    query: NDArray[np.float64],
    database: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute cosine similarity between *query* and *database* vectors.

    Parameters
    ----------
    query : ndarray of shape (n_queries, dim)  or  (dim,)
    database : ndarray of shape (n_database, dim)

    Returns
    -------
    ndarray of shape (n_queries, n_database)
        Cosine similarity scores.
    """
    if query.ndim == 1:
        query = query[np.newaxis, :]

    q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    d_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-12)
    return (q_norm @ d_norm.T).astype(np.float64)


# ------------------------------------------------------------------
# Single-query MoA retrieval
# ------------------------------------------------------------------

def moa_retrieval(
    embeddings: NDArray[np.float64],
    metadata: pd.DataFrame,
    query_idx: int,
    top_k: int = 10,
) -> pd.DataFrame:
    """Retrieve the top-k most similar perturbations and check MoA match.

    Parameters
    ----------
    embeddings : ndarray of shape (n, dim)
    metadata : pd.DataFrame
        Must contain ``moa_label`` and ``perturbation_id`` columns.
    query_idx : int
        Row index of the query perturbation.
    top_k : int
        Number of neighbours to retrieve (excluding the query itself).

    Returns
    -------
    pd.DataFrame with columns:
        rank, perturbation_id, moa_label, similarity, moa_match
    """
    query_vec = embeddings[query_idx]
    sims = compute_cosine_similarity(query_vec, embeddings).flatten()

    # Exclude the query itself
    sims[query_idx] = -np.inf
    top_indices = np.argsort(sims)[::-1][:top_k]

    query_moa = metadata.iloc[query_idx]["moa_label"]
    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        rows.append(
            {
                "rank": rank,
                "perturbation_id": metadata.iloc[idx]["perturbation_id"],
                "moa_label": metadata.iloc[idx]["moa_label"],
                "similarity": float(sims[idx]),
                "moa_match": metadata.iloc[idx]["moa_label"] == query_moa,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Recall@k
# ------------------------------------------------------------------

def compute_recall_at_k(
    embeddings: NDArray[np.float64],
    metadata: pd.DataFrame,
    k_values: Optional[List[int]] = None,
) -> Dict[int, float]:
    """Compute Recall@k for MoA-based retrieval across all queries.

    For each perturbation, the k nearest neighbours (by cosine similarity)
    are retrieved and Recall@k is defined as the fraction of queries
    for which at least one of the top-k neighbours shares the same MoA.

    Parameters
    ----------
    embeddings : ndarray of shape (n, dim)
    metadata : pd.DataFrame
        Must contain ``moa_label``.
    k_values : list of int, optional
        Defaults to ``[1, 5, 10, 20]``.

    Returns
    -------
    dict mapping k -> recall (0..1)
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    n = embeddings.shape[0]
    sim_matrix = compute_cosine_similarity(embeddings, embeddings)
    # Zero out self-similarity
    np.fill_diagonal(sim_matrix, -np.inf)

    moa_labels = metadata["moa_label"].values
    sorted_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]

    recall: Dict[int, float] = {}
    for k in k_values:
        k_actual = min(k, n - 1)
        hits = 0
        for i in range(n):
            top_k_idx = sorted_indices[i, :k_actual]
            if any(moa_labels[j] == moa_labels[i] for j in top_k_idx):
                hits += 1
        recall[k] = hits / n
        logger.info("Recall@%d = %.4f", k, recall[k])

    return recall


# ------------------------------------------------------------------
# Mean Average Precision (mAP)
# ------------------------------------------------------------------

def compute_map(
    embeddings: NDArray[np.float64],
    metadata: pd.DataFrame,
) -> float:
    """Compute Mean Average Precision for MoA retrieval.

    For each query, the Average Precision (AP) is computed over the full
    ranking of database items. mAP is the mean of all per-query APs.

    Parameters
    ----------
    embeddings : ndarray of shape (n, dim)
    metadata : pd.DataFrame

    Returns
    -------
    float
        Mean Average Precision.
    """
    n = embeddings.shape[0]
    sim_matrix = compute_cosine_similarity(embeddings, embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)

    moa_labels = metadata["moa_label"].values
    sorted_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]

    aps: List[float] = []
    for i in range(n):
        query_moa = moa_labels[i]
        relevant = 0
        precision_sum = 0.0
        for rank, j in enumerate(sorted_indices[i], start=1):
            if moa_labels[j] == query_moa:
                relevant += 1
                precision_sum += relevant / rank
        # Total relevant items (excluding query itself)
        n_relevant = int((moa_labels == query_moa).sum()) - 1
        if n_relevant > 0:
            aps.append(precision_sum / n_relevant)
        else:
            aps.append(0.0)

    mean_ap = float(np.mean(aps))
    logger.info("Mean Average Precision = %.4f", mean_ap)
    return mean_ap


# ------------------------------------------------------------------
# Cross-modal retrieval
# ------------------------------------------------------------------

def cross_modal_retrieval(
    compound_embeddings: NDArray[np.float64],
    crispr_embeddings: NDArray[np.float64],
    compound_metadata: pd.DataFrame,
    crispr_metadata: pd.DataFrame,
    k_values: Optional[List[int]] = None,
) -> Dict[int, float]:
    """Evaluate compound-to-CRISPR cross-modal retrieval.

    For each compound perturbation, retrieve the most similar CRISPR
    perturbations and check whether they share the same MoA.

    Parameters
    ----------
    compound_embeddings : ndarray of shape (n_compounds, dim)
    crispr_embeddings : ndarray of shape (n_crispr, dim)
    compound_metadata : pd.DataFrame
    crispr_metadata : pd.DataFrame
    k_values : list of int, optional

    Returns
    -------
    dict mapping k -> recall
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    n_compounds = compound_embeddings.shape[0]
    n_crispr = crispr_embeddings.shape[0]

    sim = compute_cosine_similarity(compound_embeddings, crispr_embeddings)
    sorted_indices = np.argsort(sim, axis=1)[:, ::-1]

    compound_moas = compound_metadata["moa_label"].values
    crispr_moas = crispr_metadata["moa_label"].values

    recall: Dict[int, float] = {}
    for k in k_values:
        k_actual = min(k, n_crispr)
        hits = 0
        for i in range(n_compounds):
            top_k_idx = sorted_indices[i, :k_actual]
            if any(crispr_moas[j] == compound_moas[i] for j in top_k_idx):
                hits += 1
        recall[k] = hits / max(n_compounds, 1)
        logger.info("Cross-modal Recall@%d = %.4f", k, recall[k])

    return recall
