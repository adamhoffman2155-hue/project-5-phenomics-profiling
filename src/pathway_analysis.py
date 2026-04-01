"""
PathwayAnalyzer — pathway enrichment analysis for phenotypic clusters.

Uses gseapy to query Enrichr (or run GSEA preranked) for each cluster's
gene list against KEGG, Reactome, and GO gene sets.

Implements:
- run_gsea(): GSEA preranked analysis on gene ranking vectors
- run_enrichr(): Enrichr API query for cluster gene lists
- compute_pathway_scores(): Aggregate per-gene pathway membership scores
- build_pathway_cluster_matrix(): Cluster × Pathway enrichment heatmap data
- identify_cluster_pathways(): Top pathways per cluster
- compare_clusters_pathways(): Pairwise Jaccard similarity of pathway sets
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PathwayAnalyzer:
    """
    Pathway enrichment analysis for phenomics cluster gene sets.

    Parameters
    ----------
    config : PipelineConfig, optional

    Examples
    --------
    >>> analyzer = PathwayAnalyzer()
    >>> cluster_genes = analyzer.get_cluster_gene_lists(metadata, labels)
    >>> results = analyzer.run_enrichr(cluster_genes, gene_sets=["KEGG_2021_Human"])
    >>> matrix = analyzer.build_pathway_cluster_matrix(results)
    """

    def __init__(self, config=None):
        self.config = config
        cfg_p = config.pathway if config is not None else None

        self._gene_sets = (
            cfg_p.gene_sets if cfg_p
            else ["KEGG_2021_Human", "Reactome_2022"]
        )
        self._p_cutoff = cfg_p.p_cutoff if cfg_p else 0.05
        self._min_genes = cfg_p.min_genes_for_enrichment if cfg_p else 5
        self._top_n = cfg_p.top_n_pathways if cfg_p else 10
        self._organism = cfg_p.organism if cfg_p else "human"

        # GSEA params
        self._gsea_min_size = cfg_p.gsea_min_size if cfg_p else 15
        self._gsea_max_size = cfg_p.gsea_max_size if cfg_p else 500
        self._gsea_perms = cfg_p.gsea_permutation_num if cfg_p else 1000

    # ------------------------------------------------------------------
    # Gene list preparation
    # ------------------------------------------------------------------

    def get_cluster_gene_lists(
        self,
        metadata: pd.DataFrame,
        labels: np.ndarray,
        gene_col: str = "gene",
        perturbation_type_col: str = "perturbation_type",
        crispr_label: str = "CRISPR",
    ) -> Dict[int, List[str]]:
        """
        Extract the set of gene targets for each cluster from CRISPR KO members.

        Only CRISPR knockout perturbations contribute gene symbols. Compound
        perturbations do not contribute directly (their target genes are unknown
        without annotation). Noise points (label = -1) are excluded.

        Parameters
        ----------
        metadata : pd.DataFrame, shape (N,)
        labels : np.ndarray, shape (N,)
        gene_col : str
        perturbation_type_col : str
        crispr_label : str

        Returns
        -------
        dict mapping cluster_id (int) → sorted list of unique gene symbols
        """
        meta = metadata.reset_index(drop=True).copy()
        unique_labels = sorted(set(labels) - {-1})

        cluster_genes: Dict[int, List[str]] = {}
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            if mask.sum() == 0:
                continue

            cluster_meta = meta.iloc[mask]

            # Filter to CRISPR KO perturbations
            if perturbation_type_col in cluster_meta.columns:
                crispr_mask = cluster_meta[perturbation_type_col] == crispr_label
                crispr_rows = cluster_meta[crispr_mask]
            else:
                crispr_rows = cluster_meta

            if gene_col not in crispr_rows.columns:
                cluster_genes[cluster_id] = []
                continue

            genes = (
                crispr_rows[gene_col]
                .dropna()
                .str.strip()
                .str.upper()
                .unique()
                .tolist()
            )
            genes = sorted([g for g in genes if g and g != "NAN"])
            cluster_genes[cluster_id] = genes

            logger.debug(
                f"Cluster {cluster_id}: {len(genes)} unique CRISPR genes"
            )

        logger.info(
            f"Extracted gene lists for {len(cluster_genes)} clusters. "
            f"Total genes: {sum(len(v) for v in cluster_genes.values())}"
        )
        return cluster_genes

    # ------------------------------------------------------------------
    # Enrichr
    # ------------------------------------------------------------------

    def run_enrichr(
        self,
        cluster_gene_lists: Dict[int, List[str]],
        gene_sets: Optional[List[str]] = None,
        p_cutoff: Optional[float] = None,
        min_genes: Optional[int] = None,
        outdir: Optional[str] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Run Enrichr for each cluster's gene list.

        Submits gene lists to the Enrichr API via gseapy and returns
        significant pathways per cluster.

        Parameters
        ----------
        cluster_gene_lists : dict
            Mapping of cluster_id → list of gene symbols.
        gene_sets : list of str, optional
            Enrichr gene set libraries to query.
        p_cutoff : float, optional
            Adjusted p-value cutoff.
        min_genes : int, optional
            Minimum gene list size to run enrichment.
        outdir : str, optional
            Directory to save Enrichr result CSVs.

        Returns
        -------
        dict mapping cluster_id → pd.DataFrame with enrichment results.
        """
        try:
            import gseapy as gp
        except ImportError:
            logger.warning(
                "gseapy not available. Returning mock enrichment results."
            )
            return self._mock_enrichr_results(cluster_gene_lists)

        gene_sets = gene_sets or self._gene_sets
        p_cutoff = p_cutoff if p_cutoff is not None else self._p_cutoff
        min_genes = min_genes or self._min_genes

        results: Dict[int, pd.DataFrame] = {}

        for cluster_id, genes in cluster_gene_lists.items():
            if len(genes) < min_genes:
                logger.debug(
                    f"Cluster {cluster_id}: only {len(genes)} genes, "
                    f"skipping (min={min_genes})"
                )
                results[cluster_id] = pd.DataFrame()
                continue

            logger.info(
                f"Running Enrichr for Cluster {cluster_id} "
                f"({len(genes)} genes, {gene_sets})"
            )

            try:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=gene_sets,
                    organism=self._organism,
                    outdir=outdir or "tmp_enrichr",
                    no_plot=True,
                    verbose=False,
                )
                df = enr.results.copy()

                # Standardize column names
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]

                # Filter by adjusted p-value
                padj_col = next(
                    (c for c in df.columns if "adjust" in c or "padj" in c or "fdr" in c),
                    None
                )
                if padj_col and padj_col in df.columns:
                    df = df[df[padj_col] <= p_cutoff].copy()

                # Sort by p-value
                pval_col = next(
                    (c for c in df.columns if c in ("p_value", "pval", "p-value")),
                    None
                )
                if pval_col:
                    df = df.sort_values(pval_col)

                results[cluster_id] = df
                logger.info(
                    f"  Cluster {cluster_id}: {len(df)} significant pathways"
                )

            except Exception as e:
                logger.warning(
                    f"Enrichr failed for Cluster {cluster_id}: {e}"
                )
                results[cluster_id] = pd.DataFrame()

        return results

    def _mock_enrichr_results(
        self, cluster_gene_lists: Dict[int, List[str]]
    ) -> Dict[int, pd.DataFrame]:
        """
        Generate mock Enrichr results for testing (no internet access required).
        """
        pathway_catalog = [
            ("KEGG_2021_Human", "DNA Repair", 1.2e-5),
            ("KEGG_2021_Human", "Cell Cycle", 3.4e-4),
            ("KEGG_2021_Human", "mTOR signaling pathway", 2.1e-4),
            ("KEGG_2021_Human", "Proteasome", 5.6e-6),
            ("KEGG_2021_Human", "Spliceosome", 7.8e-4),
            ("Reactome_2022", "Homologous Recombination Repair", 1.5e-5),
            ("Reactome_2022", "Signaling by mTOR", 4.2e-4),
            ("Reactome_2022", "Ubiquitin mediated proteolysis", 8.9e-5),
            ("Reactome_2022", "Regulation of TP53 Expression and Degradation", 3.3e-3),
            ("Reactome_2022", "Mitotic Prometaphase", 6.7e-4),
        ]

        results: Dict[int, pd.DataFrame] = {}
        rng = np.random.default_rng(42)

        for cluster_id, genes in cluster_gene_lists.items():
            if len(genes) == 0:
                results[cluster_id] = pd.DataFrame()
                continue

            n_pathways = rng.integers(2, min(len(pathway_catalog), 6))
            chosen = rng.choice(len(pathway_catalog), size=n_pathways, replace=False)
            rows = []
            for idx in chosen:
                gs, term, base_p = pathway_catalog[idx]
                p = base_p * rng.uniform(0.5, 2.0)
                rows.append({
                    "gene_set_library": gs,
                    "term": term,
                    "overlap": f"{rng.integers(3, min(len(genes), 15))}/{rng.integers(20, 200)}",
                    "p_value": p,
                    "adjusted_p_value": min(p * len(pathway_catalog), 1.0),
                    "old_p_value": p,
                    "old_adjusted_p_value": min(p * len(pathway_catalog), 1.0),
                    "odds_ratio": float(rng.uniform(1.5, 8.0)),
                    "combined_score": float(rng.uniform(5.0, 80.0)),
                    "genes": ";".join(rng.choice(genes, size=min(5, len(genes)), replace=False).tolist()),
                })
            df = pd.DataFrame(rows).sort_values("p_value")
            results[cluster_id] = df

        return results

    # ------------------------------------------------------------------
    # GSEA
    # ------------------------------------------------------------------

    def run_gsea(
        self,
        gene_ranking: pd.Series,
        gene_sets: Optional[List[str]] = None,
        outdir: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        permutation_num: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Run GSEA preranked analysis on a gene ranking vector.

        The gene ranking is typically computed as:
            -log10(p_value) × sign(fold_change)
        or as the mean cosine similarity of a gene's perturbation profile
        to the centroid of a pathway of interest.

        Parameters
        ----------
        gene_ranking : pd.Series
            Index = gene symbols, values = ranking metric (higher = more relevant).
        gene_sets : list of str, optional
        outdir : str, optional
        min_size, max_size : int, optional
        permutation_num : int, optional

        Returns
        -------
        pd.DataFrame or None
            GSEA results with NES, p-value, FDR per pathway.
        """
        try:
            import gseapy as gp
        except ImportError:
            logger.warning("gseapy not installed. Cannot run GSEA.")
            return None

        gene_sets = gene_sets or self._gene_sets
        min_size = min_size or self._gsea_min_size
        max_size = max_size or self._gsea_max_size
        permutation_num = permutation_num or self._gsea_perms

        # gseapy expects a DataFrame with gene names as index, one data column
        rnk = gene_ranking.sort_values(ascending=False).to_frame(name="score")
        rnk.index.name = "gene"

        logger.info(
            f"Running GSEA preranked: {len(rnk)} genes, "
            f"gene_sets={gene_sets}, permutations={permutation_num}"
        )

        try:
            gsea = gp.prerank(
                rnk=rnk,
                gene_sets=gene_sets,
                outdir=outdir or "tmp_gsea",
                min_size=min_size,
                max_size=max_size,
                permutation_num=permutation_num,
                no_plot=True,
                verbose=False,
                seed=42,
            )
            df = gsea.res2d.copy()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            logger.info(f"GSEA complete: {len(df)} pathways tested")
            return df
        except Exception as e:
            logger.warning(f"GSEA failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Pathway score aggregation
    # ------------------------------------------------------------------

    def compute_pathway_scores(
        self,
        gene_sets_dict: Dict[str, List[str]],
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        gene_col: str = "gene",
    ) -> pd.DataFrame:
        """
        Compute a per-perturbation score for each pathway based on
        whether its gene target is a member of the pathway.

        Creates a binary (or soft) perturbation × pathway membership matrix
        that can be used for enrichment scoring without running Enrichr.

        Parameters
        ----------
        gene_sets_dict : dict
            Mapping of pathway_name → list of gene members.
        embeddings : np.ndarray, shape (N, D)
        metadata : pd.DataFrame, shape (N,)
        gene_col : str

        Returns
        -------
        pd.DataFrame, shape (N, n_pathways)
            Binary membership matrix.
        """
        if gene_col not in metadata.columns:
            logger.warning(f"Gene column '{gene_col}' not in metadata.")
            return pd.DataFrame()

        genes = metadata[gene_col].fillna("").str.upper().tolist()
        pathway_names = sorted(gene_sets_dict.keys())

        # Build binary matrix
        matrix = np.zeros((len(genes), len(pathway_names)), dtype=np.float32)
        for j, pname in enumerate(pathway_names):
            pathway_genes = set(g.upper() for g in gene_sets_dict[pname])
            for i, gene in enumerate(genes):
                if gene and gene in pathway_genes:
                    matrix[i, j] = 1.0

        df = pd.DataFrame(matrix, columns=pathway_names)
        logger.info(
            f"Pathway score matrix: {df.shape[0]} perturbations × "
            f"{df.shape[1]} pathways"
        )
        return df

    # ------------------------------------------------------------------
    # Cluster-pathway matrix
    # ------------------------------------------------------------------

    def build_pathway_cluster_matrix(
        self,
        enrichment_results: Dict[int, pd.DataFrame],
        top_n: Optional[int] = None,
        score_col: str = "combined_score",
    ) -> pd.DataFrame:
        """
        Build a cluster × pathway enrichment score matrix for heatmap visualization.

        Parameters
        ----------
        enrichment_results : dict
            Output of run_enrichr(): cluster_id → enrichment DataFrame.
        top_n : int, optional
            Use only the top N most enriched pathways (across all clusters).
        score_col : str
            Column to use as enrichment score. Common choices:
            "combined_score", "odds_ratio", "-log10(p_value)".

        Returns
        -------
        pd.DataFrame, shape (n_clusters, n_pathways)
            Cluster × pathway enrichment score matrix.
        """
        top_n = top_n or self._top_n

        # Collect all (cluster, pathway, score) tuples
        rows = []
        for cluster_id, df in enrichment_results.items():
            if df is None or len(df) == 0:
                continue
            sc_col = score_col if score_col in df.columns else (
                df.columns[df.columns.str.contains("score", case=False)][0]
                if any(df.columns.str.contains("score", case=False)) else df.columns[-1]
            )
            term_col = "term" if "term" in df.columns else df.columns[1]
            for _, row in df.iterrows():
                rows.append({
                    "cluster": cluster_id,
                    "pathway": row[term_col],
                    "score": float(row[sc_col]) if pd.notna(row[sc_col]) else 0.0,
                })

        if not rows:
            logger.warning("No enrichment results to build matrix from.")
            return pd.DataFrame()

        long_df = pd.DataFrame(rows)

        # Select top N pathways by max score across clusters
        top_pathways = (
            long_df.groupby("pathway")["score"]
            .max()
            .nlargest(top_n)
            .index.tolist()
        )
        long_df = long_df[long_df["pathway"].isin(top_pathways)]

        # Pivot to cluster × pathway
        matrix = long_df.pivot_table(
            index="cluster", columns="pathway", values="score", fill_value=0.0
        )
        matrix.index.name = "cluster"
        matrix.columns.name = "pathway"

        logger.info(
            f"Pathway-cluster matrix: {matrix.shape[0]} clusters × "
            f"{matrix.shape[1]} pathways"
        )
        return matrix

    # ------------------------------------------------------------------
    # Cluster labeling
    # ------------------------------------------------------------------

    def identify_cluster_pathways(
        self,
        enrichment_results: Dict[int, pd.DataFrame],
        top_n: Optional[int] = None,
        p_cutoff: Optional[float] = None,
    ) -> Dict[int, List[str]]:
        """
        Return the top N significant pathways for each cluster.

        Parameters
        ----------
        enrichment_results : dict
        top_n : int, optional
        p_cutoff : float, optional

        Returns
        -------
        dict mapping cluster_id → list of top pathway names
        """
        top_n = top_n or self._top_n
        p_cutoff = p_cutoff if p_cutoff is not None else self._p_cutoff

        cluster_pathways: Dict[int, List[str]] = {}
        for cluster_id, df in enrichment_results.items():
            if df is None or len(df) == 0:
                cluster_pathways[cluster_id] = []
                continue

            # Filter by p-value
            padj_col = next(
                (c for c in df.columns if "adjust" in c or "fdr" in c), None
            )
            if padj_col:
                df = df[df[padj_col] <= p_cutoff]

            term_col = "term" if "term" in df.columns else df.columns[1]
            top_terms = df[term_col].dropna().head(top_n).tolist()
            cluster_pathways[cluster_id] = top_terms

        return cluster_pathways

    # ------------------------------------------------------------------
    # Pairwise cluster comparison
    # ------------------------------------------------------------------

    def compare_clusters_pathways(
        self,
        enrichment_results: Dict[int, pd.DataFrame],
        p_cutoff: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute pairwise Jaccard similarity of significant pathway sets
        between clusters.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        A high Jaccard similarity indicates two clusters share similar
        enriched pathways, suggesting biological relatedness.

        Parameters
        ----------
        enrichment_results : dict
        p_cutoff : float, optional

        Returns
        -------
        pd.DataFrame, shape (n_clusters, n_clusters)
            Pairwise Jaccard similarity matrix.
        """
        p_cutoff = p_cutoff if p_cutoff is not None else self._p_cutoff
        cluster_pathway_sets: Dict[int, set] = {}

        for cluster_id, df in enrichment_results.items():
            if df is None or len(df) == 0:
                cluster_pathway_sets[cluster_id] = set()
                continue

            padj_col = next(
                (c for c in df.columns if "adjust" in c or "fdr" in c), None
            )
            if padj_col:
                sig_df = df[df[padj_col] <= p_cutoff]
            else:
                sig_df = df

            term_col = "term" if "term" in df.columns else df.columns[1]
            cluster_pathway_sets[cluster_id] = set(
                sig_df[term_col].dropna().str.strip().tolist()
            )

        cluster_ids = sorted(cluster_pathway_sets.keys())
        n = len(cluster_ids)
        jaccard_matrix = np.zeros((n, n), dtype=np.float32)

        for i, ci in enumerate(cluster_ids):
            for j, cj in enumerate(cluster_ids):
                set_i = cluster_pathway_sets[ci]
                set_j = cluster_pathway_sets[cj]
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0

        result = pd.DataFrame(
            jaccard_matrix,
            index=cluster_ids,
            columns=cluster_ids,
        )
        result.index.name = "cluster"
        result.columns.name = "cluster"

        logger.info(
            f"Pairwise pathway Jaccard matrix: {n}×{n} clusters"
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_enrichment_results(
        self,
        enrichment_results: Dict[int, pd.DataFrame],
        output_dir: str,
    ) -> None:
        """Save per-cluster enrichment results to CSV files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_results = []
        for cluster_id, df in enrichment_results.items():
            if df is None or len(df) == 0:
                continue
            df_copy = df.copy()
            df_copy.insert(0, "cluster_id", cluster_id)
            all_results.append(df_copy)

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(str(out / "enrichr_results.csv"), index=False)
            logger.info(
                f"Enrichment results saved: {len(combined)} rows → "
                f"{out / 'enrichr_results.csv'}"
            )
