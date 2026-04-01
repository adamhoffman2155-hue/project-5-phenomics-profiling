"""
MoARetriever — mechanism-of-action (MoA) retrieval via cosine similarity.

Implements:
- Building a cosine similarity reference index from perturbation embeddings
- Top-k nearest-neighbor retrieval
- Recall@k computation for known MoA groups
- Average Precision and Mean Average Precision (MAP)
- Full MoA retrieval benchmark on compound and CRISPR KO pairs
- Cross-modal retrieval: matching compound phenotypes to CRISPR KO phenotypes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class MoARetriever:
    """
    Performs mechanism-of-action retrieval from phenomic embeddings.

    Builds a cosine similarity index and benchmarks retrieval against
    known MoA group annotations (from ChEMBL or JUMP-CP metadata).

    Parameters
    ----------
    config : PipelineConfig, optional

    Examples
    --------
    >>> retriever = MoARetriever()
    >>> retriever.build_reference_index(embeddings, metadata)
    >>> neighbors = retriever.retrieve_nearest_neighbors(query_idx=0, top_k=10)
    >>> recall = retriever.compute_recall_at_k(embeddings, metadata, k=10)
    """

    def __init__(self, config=None):
        self.config = config
        cfg_r = config.retrieval if config is not None else None
        self._top_k = cfg_r.top_k if cfg_r else 10
        self._recall_k_values = cfg_r.recall_k_values if cfg_r else [1, 5, 10, 20, 50]
        self._metric = cfg_r.similarity_metric if cfg_r else "cosine"
        self._moa_col = cfg_r.moa_column if cfg_r else "moa"
        self._min_group_size = cfg_r.min_moa_group_size if cfg_r else 2

        # Index state
        self._embeddings_norm: Optional[np.ndarray] = None
        self._metadata: Optional[pd.DataFrame] = None
        self._sim_matrix: Optional[np.ndarray] = None
        self._index_built: bool = False

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_reference_index(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        precompute_full_matrix: bool = False,
    ) -> None:
        """
        Build the reference index for nearest-neighbor retrieval.

        Normalizes embeddings to unit L2 norm so that dot product equals
        cosine similarity. Optionally precomputes the full N×N similarity
        matrix (memory-intensive for large N, but fast for repeated queries).

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        metadata : pd.DataFrame, shape (N,)
        precompute_full_matrix : bool
            If True, precomputes full N×N cosine similarity matrix.
            Avoid for N > 10,000 (memory: N² × 4 bytes).
        """
        self._embeddings_norm = normalize(
            embeddings.astype(np.float32), norm="l2"
        )
        self._metadata = metadata.reset_index(drop=True).copy()

        if precompute_full_matrix:
            logger.info(
                f"Precomputing full {len(embeddings)}×{len(embeddings)} "
                f"cosine similarity matrix..."
            )
            self._sim_matrix = (
                self._embeddings_norm @ self._embeddings_norm.T
            ).astype(np.float32)
            logger.info("Full similarity matrix ready.")
        else:
            self._sim_matrix = None

        self._index_built = True
        logger.info(
            f"Reference index built: {len(embeddings)} perturbations, "
            f"dim={embeddings.shape[1]}"
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_nearest_neighbors(
        self,
        query_embeddings: np.ndarray,
        top_k: Optional[int] = None,
        exclude_self: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k most similar perturbations for each query.

        Parameters
        ----------
        query_embeddings : np.ndarray, shape (Q, D) or (D,)
            Query embedding(s). Will be L2-normalized.
        top_k : int, optional
        exclude_self : bool
            If True, excludes each query from its own neighbor list
            (assumes queries are subset of the reference index).

        Returns
        -------
        indices : np.ndarray, shape (Q, top_k)
            Row indices into the reference index.
        similarities : np.ndarray, shape (Q, top_k)
            Cosine similarity values in descending order.
        """
        if not self._index_built:
            raise RuntimeError("Call build_reference_index() first.")

        top_k = top_k or self._top_k

        # Handle single query
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[np.newaxis, :]

        query_norm = normalize(query_embeddings.astype(np.float32), norm="l2")

        # Compute query–reference similarities
        if self._sim_matrix is not None:
            # Use precomputed matrix (query must be subset of reference)
            # For this case we expect query indices to be passed
            sim = query_norm @ self._embeddings_norm.T
        else:
            sim = query_norm @ self._embeddings_norm.T  # (Q, N)

        Q, N = sim.shape
        fetch_k = min(top_k + (1 if exclude_self else 0), N)

        all_indices = []
        all_sims = []

        for q in range(Q):
            row = sim[q]
            # Partial argsort: top fetch_k in descending order
            top_idx = np.argpartition(row, -fetch_k)[-fetch_k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            top_sim = row[top_idx]

            if exclude_self:
                # Remove exact matches (self-similarity = 1.0)
                self_mask = top_sim < 0.9999
                top_idx = top_idx[self_mask][:top_k]
                top_sim = top_sim[self_mask][:top_k]
            else:
                top_idx = top_idx[:top_k]
                top_sim = top_sim[:top_k]

            # Pad if fewer than top_k returned
            if len(top_idx) < top_k:
                pad = top_k - len(top_idx)
                top_idx = np.concatenate([top_idx, np.full(pad, -1, dtype=int)])
                top_sim = np.concatenate([top_sim, np.zeros(pad)])

            all_indices.append(top_idx)
            all_sims.append(top_sim)

        return np.stack(all_indices), np.stack(all_sims)

    # ------------------------------------------------------------------
    # Recall@k
    # ------------------------------------------------------------------

    def compute_recall_at_k(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        k: int = 10,
        moa_col: Optional[str] = None,
        perturbation_type: Optional[str] = None,
    ) -> float:
        """
        Compute Recall@k for MoA retrieval.

        Recall@k is the fraction of query perturbations for which at least
        one of the top-k retrieved perturbations shares the same MoA label.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        metadata : pd.DataFrame, shape (N,)
        k : int
        moa_col : str, optional
            Column in metadata containing MoA/target labels.
        perturbation_type : str, optional
            If set, only evaluate queries of this perturbation type.

        Returns
        -------
        float
            Recall@k in [0, 1].
        """
        moa_col = moa_col or self._moa_col

        if moa_col not in metadata.columns:
            # Fall back to gene-based grouping for CRISPR
            if "gene" in metadata.columns:
                moa_col = "gene"
            elif "compound" in metadata.columns:
                moa_col = "compound"
            else:
                raise ValueError(
                    f"MoA column '{moa_col}' not found in metadata."
                )

        meta = metadata.reset_index(drop=True).copy()

        # Filter by perturbation type
        if perturbation_type and "perturbation_type" in meta.columns:
            query_mask = meta["perturbation_type"] == perturbation_type
        else:
            query_mask = np.ones(len(meta), dtype=bool)

        # Require non-null MoA labels
        valid_moa = meta[moa_col].notna()
        # Exclude controls
        if "perturbation_type" in meta.columns:
            non_control = ~meta["perturbation_type"].isin(["negcon", "poscon"])
            query_mask = query_mask & valid_moa & non_control
        else:
            query_mask = query_mask & valid_moa

        # Only evaluate MoA groups with >= min_group_size members
        moa_counts = meta.loc[query_mask, moa_col].value_counts()
        valid_moas = set(moa_counts[moa_counts >= self._min_group_size].index)
        query_mask = query_mask & meta[moa_col].isin(valid_moas)

        query_indices = np.where(query_mask)[0]

        if len(query_indices) == 0:
            logger.warning("No valid query perturbations for Recall@k evaluation.")
            return 0.0

        # Build index if needed
        if not self._index_built:
            self.build_reference_index(embeddings, meta)

        query_embs = self._embeddings_norm[query_indices]
        nn_indices, nn_sims = self.retrieve_nearest_neighbors(
            query_embs, top_k=k, exclude_self=True
        )

        # Check if any retrieved neighbor shares MoA with query
        hits = 0
        for i, q_idx in enumerate(query_indices):
            q_moa = meta.loc[q_idx, moa_col]
            retrieved_moas = set(
                meta.loc[nn_indices[i][nn_indices[i] >= 0], moa_col].tolist()
            )
            if q_moa in retrieved_moas:
                hits += 1

        recall = hits / len(query_indices)
        logger.info(
            f"Recall@{k} = {recall:.4f} "
            f"({hits}/{len(query_indices)} hits, {len(valid_moas)} MoA groups)"
        )
        return recall

    def compute_average_precision(
        self,
        query_idx: int,
        retrieved_indices: np.ndarray,
        metadata: pd.DataFrame,
        moa_col: str = "moa",
    ) -> float:
        """
        Compute Average Precision for a single query.

        AP = (1 / R) * Σ_{k=1}^{K} P(k) * rel(k)
        where R is total number of relevant items, P(k) is precision at rank k,
        and rel(k) = 1 if rank-k item is relevant.

        Parameters
        ----------
        query_idx : int
            Row index of the query in metadata.
        retrieved_indices : np.ndarray, shape (K,)
            Ranked list of retrieved indices.
        metadata : pd.DataFrame
        moa_col : str

        Returns
        -------
        float : AP in [0, 1]
        """
        if moa_col not in metadata.columns:
            return 0.0

        q_moa = metadata.iloc[query_idx][moa_col]
        if pd.isna(q_moa):
            return 0.0

        # All relevant items (same MoA, excluding query itself)
        all_relevant = set(
            metadata[
                (metadata[moa_col] == q_moa)
                & (metadata.index != query_idx)
            ].index.tolist()
        )
        R = len(all_relevant)
        if R == 0:
            return 0.0

        # Compute AP
        n_relevant_so_far = 0
        ap = 0.0
        for rank, idx in enumerate(retrieved_indices, start=1):
            if idx < 0:
                continue
            if idx in all_relevant:
                n_relevant_so_far += 1
                ap += n_relevant_so_far / rank

        ap /= R
        return float(ap)

    def compute_map(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        k: int = 50,
        moa_col: Optional[str] = None,
    ) -> float:
        """
        Compute Mean Average Precision (MAP) across all query perturbations.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        metadata : pd.DataFrame, shape (N,)
        k : int
            Number of neighbors to retrieve per query.
        moa_col : str, optional

        Returns
        -------
        float : MAP in [0, 1]
        """
        moa_col = moa_col or self._moa_col
        meta = metadata.reset_index(drop=True).copy()

        if moa_col not in meta.columns:
            if "gene" in meta.columns:
                moa_col = "gene"
            else:
                logger.warning(f"MoA column '{moa_col}' not found. MAP=0.")
                return 0.0

        # Filter to valid queries
        valid_mask = (
            meta[moa_col].notna()
            & ~meta.get("perturbation_type", pd.Series("unknown", index=meta.index))
            .isin(["negcon", "poscon"])
        )
        query_indices = np.where(valid_mask)[0]

        if len(query_indices) == 0:
            return 0.0

        if not self._index_built:
            self.build_reference_index(embeddings, meta)

        query_embs = self._embeddings_norm[query_indices]
        nn_indices, _ = self.retrieve_nearest_neighbors(
            query_embs, top_k=k, exclude_self=True
        )

        ap_scores = []
        for i, q_idx in enumerate(query_indices):
            ap = self.compute_average_precision(
                q_idx, nn_indices[i], meta, moa_col=moa_col
            )
            ap_scores.append(ap)

        map_score = float(np.mean(ap_scores))
        logger.info(
            f"MAP@{k} = {map_score:.4f} "
            f"(over {len(query_indices)} queries)"
        )
        return map_score

    # ------------------------------------------------------------------
    # Full benchmark
    # ------------------------------------------------------------------

    def evaluate_moa_retrieval(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        moa_col: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run the full MoA retrieval benchmark.

        Computes Recall@k for k ∈ {1, 5, 10, 20, 50} and MAP.
        Reports results by perturbation type and MoA category.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        metadata : pd.DataFrame
        moa_col : str, optional
        save_path : str, optional
            Path to save JSON results.

        Returns
        -------
        dict of metric → float
        """
        if not self._index_built:
            self.build_reference_index(embeddings, metadata)

        results: Dict[str, float] = {}

        # Overall Recall@k
        for k in self._recall_k_values:
            recall = self.compute_recall_at_k(embeddings, metadata, k=k, moa_col=moa_col)
            results[f"recall_at_{k}"] = recall

        # MAP
        results["map"] = self.compute_map(embeddings, metadata, k=50, moa_col=moa_col)

        # Per-type breakdown
        if "perturbation_type" in metadata.columns:
            for ptype in metadata["perturbation_type"].unique():
                if ptype in ("negcon", "poscon"):
                    continue
                r10 = self.compute_recall_at_k(
                    embeddings, metadata, k=10,
                    moa_col=moa_col, perturbation_type=ptype
                )
                results[f"recall_at_10_{ptype}"] = r10

        # Log summary
        logger.info("=== MoA Retrieval Benchmark ===")
        for key, val in results.items():
            logger.info(f"  {key}: {val:.4f}")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Retrieval results saved to {save_path}")

        return results

    # ------------------------------------------------------------------
    # Cross-modal matching
    # ------------------------------------------------------------------

    def match_compounds_to_crispr(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> pd.DataFrame:
        """
        For each compound perturbation, find CRISPR KO perturbations whose
        phenotype is most similar (compound phenocopies gene KO).

        This is the basis for target identification: if a compound has a similar
        phenotype to a gene KO, the compound may act on that gene or pathway.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        metadata : pd.DataFrame, shape (N,)
        top_k : int
            Number of top CRISPR matches per compound.
        similarity_threshold : float
            Minimum cosine similarity to report.

        Returns
        -------
        pd.DataFrame
            Columns: compound, crispr_gene, cosine_similarity, rank
        """
        meta = metadata.reset_index(drop=True).copy()

        if "perturbation_type" not in meta.columns:
            logger.warning("No perturbation_type column. Returning empty DataFrame.")
            return pd.DataFrame()

        compound_mask = meta["perturbation_type"] == "compound"
        crispr_mask = meta["perturbation_type"] == "CRISPR"

        compound_indices = np.where(compound_mask)[0]
        crispr_indices = np.where(crispr_mask)[0]

        if len(compound_indices) == 0 or len(crispr_indices) == 0:
            logger.warning("No compound or CRISPR perturbations found.")
            return pd.DataFrame()

        embs_norm = normalize(embeddings.astype(np.float32), norm="l2")
        compound_embs = embs_norm[compound_indices]
        crispr_embs = embs_norm[crispr_indices]

        # Compute compound → CRISPR similarities
        sim = compound_embs @ crispr_embs.T  # (N_compounds, N_crispr)

        rows = []
        compound_col = "compound" if "compound" in meta.columns else "perturbation_type"
        gene_col = "gene" if "gene" in meta.columns else "perturbation_type"

        for c_local_idx, c_global_idx in enumerate(compound_indices):
            compound_name = meta.iloc[c_global_idx].get(compound_col, f"compound_{c_local_idx}")
            sim_row = sim[c_local_idx]

            top_local = np.argsort(sim_row)[::-1][:top_k]
            for rank, kr_local_idx in enumerate(top_local, start=1):
                kr_global_idx = crispr_indices[kr_local_idx]
                cos_sim = float(sim_row[kr_local_idx])
                if cos_sim < similarity_threshold:
                    break
                gene = meta.iloc[kr_global_idx].get(gene_col, f"gene_{kr_local_idx}")
                rows.append({
                    "compound": compound_name,
                    "crispr_gene": gene,
                    "cosine_similarity": cos_sim,
                    "rank": rank,
                })

        result = pd.DataFrame(rows)
        logger.info(
            f"Cross-modal matching: {len(compound_indices)} compounds → "
            f"{len(crispr_indices)} CRISPR KOs. Found {len(result)} matches "
            f"above threshold {similarity_threshold}."
        )
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_nearest_neighbor_table(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build a full nearest-neighbor table for all perturbations.

        Returns a long-format DataFrame with one row per (query, neighbor) pair,
        including metadata for both the query and neighbor.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        metadata : pd.DataFrame, shape (N,)
        top_k : int, optional

        Returns
        -------
        pd.DataFrame
        """
        top_k = top_k or self._top_k
        meta = metadata.reset_index(drop=True).copy()

        if not self._index_built:
            self.build_reference_index(embeddings, meta)

        nn_indices, nn_sims = self.retrieve_nearest_neighbors(
            self._embeddings_norm, top_k=top_k, exclude_self=True
        )

        rows = []
        for q_idx in range(len(meta)):
            for rank, (nn_idx, nn_sim) in enumerate(
                zip(nn_indices[q_idx], nn_sims[q_idx]), start=1
            ):
                if nn_idx < 0:
                    continue
                rows.append({
                    "query_idx": q_idx,
                    "neighbor_idx": int(nn_idx),
                    "rank": rank,
                    "cosine_similarity": float(nn_sim),
                })

        nn_df = pd.DataFrame(rows)
        logger.info(
            f"Nearest-neighbor table: {len(nn_df)} rows "
            f"({len(meta)} queries × {top_k} neighbors)"
        )
        return nn_df
