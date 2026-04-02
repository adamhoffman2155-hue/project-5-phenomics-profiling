"""
Pathway enrichment analysis for the Phenomics Profiling Pipeline.

Provides gene-set enrichment (via gseapy or a mock implementation),
Jaccard similarity, and pathway-cluster enrichment matrices.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger("phenomics.pathway_analysis")


# ------------------------------------------------------------------
# Default mock gene sets (used when gseapy is unavailable)
# ------------------------------------------------------------------

_DEFAULT_GENE_SETS: Dict[str, List[str]] = {
    "apoptosis": [
        "BAX", "BCL2", "CASP3", "CASP8", "CASP9", "CYCS", "APAF1",
        "BID", "BAK1", "XIAP", "BIRC5", "MCL1", "FADD", "FAS",
        "TNFRSF10A", "TNFRSF10B", "DIABLO", "PMAIP1", "BBC3", "BOK",
    ],
    "cell_cycle": [
        "CDK1", "CDK2", "CDK4", "CDK6", "CCND1", "CCNE1", "CCNA2",
        "CCNB1", "RB1", "TP53", "CDKN1A", "CDKN2A", "E2F1", "CDC25A",
        "PLK1", "AURKA", "AURKB", "BUB1", "MAD2L1", "TTK",
    ],
    "dna_repair": [
        "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51",
        "TP53BP1", "MDC1", "RNF8", "PARP1", "XRCC1", "MLH1", "MSH2",
        "MSH6", "ERCC1", "XPC", "XPA", "POLB", "LIG3",
    ],
    "immune_response": [
        "TNF", "IL6", "IL1B", "IFNG", "CXCL8", "CCL2", "NFKB1",
        "RELA", "JAK1", "JAK2", "STAT1", "STAT3", "TLR4", "MYD88",
        "IRAK4", "TRAF6", "IRF3", "CGAS", "STING1", "MAVS",
    ],
    "metabolism": [
        "HK2", "PKM", "LDHA", "PDK1", "CS", "IDH1", "IDH2",
        "OGDH", "SDHA", "SDHB", "FH", "MDH2", "ACLY", "FASN",
        "SCD", "CPT1A", "ACOX1", "GLS", "SLC1A5", "SLC7A11",
    ],
}


# ------------------------------------------------------------------
# Enrichment
# ------------------------------------------------------------------

def enrich_cluster_genes(
    cluster_genes: List[str],
    gene_sets: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Run gene-set enrichment for a list of cluster genes.

    If ``gseapy`` is available, the function attempts to use the Enrichr
    API. Otherwise it falls back to a simple Fisher-exact-style mock
    enrichment against the built-in pathway gene sets.

    Parameters
    ----------
    cluster_genes : list of str
        Gene symbols found in the cluster.
    gene_sets : dict, optional
        Mapping of pathway name to gene list. Defaults to built-in sets.

    Returns
    -------
    pd.DataFrame with columns:
        pathway, overlap_count, overlap_genes, jaccard, p_value
    """
    if gene_sets is None:
        gene_sets = _DEFAULT_GENE_SETS

    # Expand any semicolon-separated gene symbols
    expanded: List[str] = []
    for g in cluster_genes:
        if ";" in g:
            expanded.extend(g.split(";"))
        else:
            expanded.append(g)
    cluster_genes = list(set(expanded))

    # Try gseapy first
    try:
        import gseapy as gp  # type: ignore[import-untyped]

        logger.info("Using gseapy Enrichr for enrichment analysis")
        enr = gp.enrichr(
            gene_list=cluster_genes,
            gene_sets="KEGG_2021_Human",
            organism="Human",
            outdir=None,
            no_plot=True,
        )
        result_df = enr.results[["Term", "Overlap", "Adjusted P-value"]].rename(
            columns={
                "Term": "pathway",
                "Overlap": "overlap_count",
                "Adjusted P-value": "p_value",
            }
        )
        result_df["overlap_genes"] = ""
        result_df["jaccard"] = 0.0
        return result_df.head(20)

    except (ImportError, Exception) as exc:
        logger.info(
            "gseapy not available or Enrichr failed (%s) \u2014 using mock enrichment",
            type(exc).__name__,
        )

    # Mock enrichment: Jaccard + hypergeometric p-value approximation
    from scipy.stats import hypergeom

    query_set: Set[str] = set(cluster_genes)
    # Universe of all genes across all pathways + query
    universe: Set[str] = set(cluster_genes)
    for genes in gene_sets.values():
        universe.update(genes)
    N = len(universe)

    rows = []
    for pathway, pathway_genes in gene_sets.items():
        pathway_set = set(pathway_genes)
        overlap = query_set & pathway_set
        overlap_count = len(overlap)
        jaccard = compute_jaccard_similarity(query_set, pathway_set)

        # Hypergeometric p-value
        # P(X >= overlap_count) where X ~ Hypergeom(N, K, n)
        K = len(pathway_set)
        n = len(query_set)
        if overlap_count > 0:
            pval = float(hypergeom.sf(overlap_count - 1, N, K, n))
        else:
            pval = 1.0

        rows.append(
            {
                "pathway": pathway,
                "overlap_count": overlap_count,
                "overlap_genes": ";".join(sorted(overlap)),
                "jaccard": jaccard,
                "p_value": pval,
            }
        )

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    logger.info("Mock enrichment complete \u2014 %d pathways tested", len(result))
    return result


# ------------------------------------------------------------------
# Jaccard similarity
# ------------------------------------------------------------------

def compute_jaccard_similarity(
    set1: Set[str] | List[str],
    set2: Set[str] | List[str],
) -> float:
    """Compute Jaccard similarity between two gene sets.

    Parameters
    ----------
    set1, set2 : set or list of str

    Returns
    -------
    float in [0, 1]
    """
    s1 = set(set1)
    s2 = set(set2)
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


# ------------------------------------------------------------------
# Pathway-cluster enrichment matrix
# ------------------------------------------------------------------

def build_pathway_cluster_matrix(
    cluster_data: Dict,
    pathways: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Build a clusters x pathways enrichment-score matrix.

    Accepts either:
    - A dict of cluster_label -> list of gene symbols (Jaccard is computed), or
    - A dict of cluster_label -> pd.DataFrame from ``enrich_cluster_genes``
      (the ``jaccard`` or ``p_value`` column is pivoted into the matrix).

    Parameters
    ----------
    cluster_data : dict
        Mapping of cluster label -> gene list **or** enrichment DataFrame.
    pathways : dict, optional
        Mapping of pathway name -> gene list. Defaults to built-in sets.

    Returns
    -------
    pd.DataFrame
        Rows = cluster labels, columns = pathway names.
    """
    if pathways is None:
        pathways = _DEFAULT_GENE_SETS

    cluster_labels = sorted(cluster_data.keys(), key=str)

    # Detect whether values are DataFrames (enrichment results) or gene lists
    sample_val = next(iter(cluster_data.values()))
    if isinstance(sample_val, pd.DataFrame):
        # Pivot enrichment DataFrames into a matrix
        all_pathways: Set[str] = set()
        for df in cluster_data.values():
            all_pathways.update(df["pathway"].tolist())
        sorted_pw = sorted(all_pathways)

        data: Dict[str, List[float]] = {pw: [] for pw in sorted_pw}
        for cl in cluster_labels:
            df = cluster_data[cl]
            pw_map = dict(zip(df["pathway"], df["jaccard"]))
            for pw in sorted_pw:
                data[pw].append(pw_map.get(pw, 0.0))

        matrix = pd.DataFrame(data, index=[str(c) for c in cluster_labels])
    else:
        # Gene-list mode: compute Jaccard against pathway gene sets
        data = {pw: [] for pw in pathways}
        for cl in cluster_labels:
            cl_genes = set(cluster_data[cl])
            for pw_name, pw_genes in pathways.items():
                data[pw_name].append(compute_jaccard_similarity(cl_genes, pw_genes))

        matrix = pd.DataFrame(data, index=[f"cluster_{c}" for c in cluster_labels])

    logger.info("Pathway-cluster matrix shape: %s", matrix.shape)
    return matrix
