"""
Visualization module for the Phenomics Profiling Pipeline.

Produces publication-quality figures for UMAP embeddings, retrieval
performance, clustering metrics, and pathway enrichment heatmaps.
All functions accept a *save_path* argument; when provided, the figure
is saved and the matplotlib figure is closed to free memory.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server / CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

logger = logging.getLogger("phenomics.visualization")

# Shared style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
_DPI = 150


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to *save_path* (creating parent dirs) or show it."""
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
        plt.close(fig)
    else:
        plt.close(fig)


# ------------------------------------------------------------------
# UMAP scatter plots
# ------------------------------------------------------------------

def plot_umap_clusters(
    umap_coords: NDArray[np.float64],
    labels: NDArray[np.int64],
    save_path: Optional[str] = None,
    title: str = "UMAP \u2014 Cluster assignments",
) -> None:
    """Scatter plot of UMAP coordinates coloured by cluster label.

    Parameters
    ----------
    umap_coords : ndarray of shape (n, 2)
    labels : ndarray of shape (n,)
    save_path : str, optional
    title : str
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", n_colors=max(len(unique_labels), 1))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}

    for lab in unique_labels:
        mask = labels == lab
        label_str = "Noise" if lab == -1 else f"Cluster {lab}"
        color = "lightgrey" if lab == -1 else color_map[lab]
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=[color],
            label=label_str,
            s=20,
            alpha=0.7,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_umap_perturbation_type(
    umap_coords: NDArray[np.float64],
    perturbation_types: List[str],
    save_path: Optional[str] = None,
    title: str = "UMAP \u2014 Perturbation type",
) -> None:
    """Scatter plot of UMAP coordinates coloured by perturbation type.

    Parameters
    ----------
    umap_coords : ndarray of shape (n, 2)
    perturbation_types : list of str
    save_path : str, optional
    title : str
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    types_arr = np.array(perturbation_types)
    unique_types = sorted(set(perturbation_types))
    palette = sns.color_palette("Set2", n_colors=len(unique_types))

    for i, ptype in enumerate(unique_types):
        mask = types_arr == ptype
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=[palette[i]],
            label=ptype,
            s=20,
            alpha=0.7,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=10)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# Retrieval performance
# ------------------------------------------------------------------

def plot_retrieval_performance(
    recall_dict: Dict[int, float],
    save_path: Optional[str] = None,
    title: str = "Retrieval \u2014 Recall@k",
) -> None:
    """Bar chart of Recall@k values.

    Parameters
    ----------
    recall_dict : dict mapping k -> recall
    save_path : str, optional
    title : str
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ks = sorted(recall_dict.keys())
    values = [recall_dict[k] for k in ks]
    bars = ax.bar([str(k) for k in ks], values, color=sns.color_palette("Blues_d", len(ks)))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title(title)
    ax.set_ylim(0, min(1.05, max(values) + 0.1))
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# Clustering metrics
# ------------------------------------------------------------------

def plot_cluster_metrics(
    metrics_dict: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Clustering evaluation metrics",
) -> None:
    """Horizontal bar chart of clustering quality metrics.

    Parameters
    ----------
    metrics_dict : dict mapping metric name -> value
    save_path : str, optional
    title : str
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    colors = sns.color_palette("viridis", len(names))
    bars = ax.barh(names, values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Value")
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ------------------------------------------------------------------
# Pathway enrichment heatmap
# ------------------------------------------------------------------

def plot_pathway_heatmap(
    pathway_matrix: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Pathway enrichment (Jaccard similarity)",
) -> None:
    """Heatmap of pathway enrichment scores per cluster.

    Parameters
    ----------
    pathway_matrix : pd.DataFrame
        Rows = clusters, columns = pathways.
    save_path : str, optional
    title : str
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(pathway_matrix) * 0.6)))

    sns.heatmap(
        pathway_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=max(0.3, pathway_matrix.values.max()),
    )

    ax.set_title(title)
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Pathway")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    _save_or_show(fig, save_path)
