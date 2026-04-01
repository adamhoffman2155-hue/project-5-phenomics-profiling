"""
PhenomicsVisualizer — visualization suite for phenomics perturbation profiling.

Generates publication-quality figures for:
- UMAP embeddings colored by perturbation type, cluster, MoA
- Cluster heatmaps (seaborn clustermap)
- Retrieval precision-recall curves
- Pathway enrichment barplots
- Interactive plotly UMAP with hover labels

All figures follow a consistent color palette and style inspired by
Recursion's published Maps of Biology figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

PERTURBATION_PALETTE = {
    "CRISPR": "#2196F3",        # Blue
    "compound": "#FF5722",      # Deep orange
    "negcon": "#9E9E9E",        # Grey
    "poscon": "#4CAF50",        # Green
    "other": "#AB47BC",         # Purple
}

CLUSTER_COLORMAP = "tab20"

MoA_PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#A8DADC", "#1D3557", "#F1FAEE", "#E76F51",
    "#8338EC", "#3A86FF", "#06D6A0", "#FFB703", "#FB8500",
    "#023047", "#219EBC", "#8ECAE6", "#126782", "#6A0572",
]


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------

class PhenomicsVisualizer:
    """
    Visualization suite for phenomics perturbation profiling results.

    Parameters
    ----------
    config : PipelineConfig, optional
    figsize : tuple
        Default figure size (width, height) in inches.
    dpi : int
        Default dots per inch for saved figures.

    Examples
    --------
    >>> viz = PhenomicsVisualizer()
    >>> viz.plot_umap_by_perturbation_type(umap_coords, metadata, save_path="umap.png")
    >>> viz.create_interactive_umap(umap_coords, metadata, labels, save_path="umap.html")
    """

    def __init__(self, config=None, figsize: Tuple[int, int] = (10, 8), dpi: int = 150):
        self.config = config
        self.figsize = figsize
        self.dpi = dpi
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure matplotlib style for all plots."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            plt.style.use("seaborn-v0_8-whitegrid")
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
                "figure.dpi": self.dpi,
            })
        except Exception:
            pass  # Style setup is non-critical

    # ------------------------------------------------------------------
    # UMAP scatter plots
    # ------------------------------------------------------------------

    def plot_umap_by_perturbation_type(
        self,
        umap_coords: np.ndarray,
        metadata: pd.DataFrame,
        perturbation_type_col: str = "perturbation_type",
        save_path: Optional[str] = None,
        alpha: float = 0.6,
        point_size: float = 8.0,
        title: str = "UMAP — Perturbation Type",
    ) -> "plt.Figure":
        """
        UMAP scatter plot colored by perturbation type (CRISPR / compound / control).

        Parameters
        ----------
        umap_coords : np.ndarray, shape (N, 2)
        metadata : pd.DataFrame, shape (N,)
        perturbation_type_col : str
        save_path : str, optional
        alpha : float
        point_size : float
        title : str

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)
        meta = metadata.reset_index(drop=True)

        if perturbation_type_col in meta.columns:
            types = meta[perturbation_type_col].fillna("other").values
            unique_types = sorted(set(types))
        else:
            types = np.array(["unknown"] * len(umap_coords))
            unique_types = ["unknown"]

        # Plot each type separately for legend control
        for ptype in unique_types:
            mask = types == ptype
            color = PERTURBATION_PALETTE.get(ptype, "#888888")
            # Draw controls under perturbations (lower zorder)
            zorder = 1 if ptype in ("negcon", "poscon") else 2
            _alpha = 0.2 if ptype in ("negcon",) else alpha
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=color,
                s=point_size * (0.5 if ptype == "negcon" else 1.0),
                alpha=_alpha,
                label=f"{ptype} (n={mask.sum():,})",
                zorder=zorder,
                linewidths=0,
                rasterized=True,
            )

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(title, fontsize=14, pad=15)
        ax.legend(
            loc="upper right",
            framealpha=0.9,
            markerscale=2.0,
            title="Perturbation type",
        )
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()

        if save_path:
            _save_figure(fig, save_path, self.dpi)

        return fig

    def plot_umap_by_cluster(
        self,
        umap_coords: np.ndarray,
        labels: np.ndarray,
        cluster_names: Optional[Dict[int, str]] = None,
        save_path: Optional[str] = None,
        alpha: float = 0.65,
        point_size: float = 8.0,
        title: str = "UMAP — Phenotypic Clusters (HDBSCAN)",
        show_noise: bool = True,
    ) -> "plt.Figure":
        """
        UMAP scatter plot colored by cluster assignment.

        Parameters
        ----------
        umap_coords : np.ndarray, shape (N, 2)
        labels : np.ndarray, shape (N,)
        cluster_names : dict, optional
            Mapping of cluster_id → display name.
        save_path : str, optional
        alpha : float
        point_size : float
        title : str
        show_noise : bool
            Whether to plot noise points (label=-1).

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig, ax = plt.subplots(figsize=self.figsize)

        unique_labels = sorted(set(labels))
        n_clusters = sum(1 for l in unique_labels if l >= 0)
        cmap = cm.get_cmap(CLUSTER_COLORMAP, max(n_clusters, 1))

        # Plot noise first (background)
        if show_noise and -1 in unique_labels:
            noise_mask = labels == -1
            ax.scatter(
                umap_coords[noise_mask, 0],
                umap_coords[noise_mask, 1],
                c="#CCCCCC",
                s=point_size * 0.4,
                alpha=0.3,
                label=f"Noise (n={noise_mask.sum():,})",
                zorder=1,
                linewidths=0,
                rasterized=True,
            )

        # Plot clusters
        cluster_idx = 0
        for cluster_id in unique_labels:
            if cluster_id < 0:
                continue
            mask = labels == cluster_id
            color = cmap(cluster_idx % n_clusters)
            name = (
                cluster_names.get(cluster_id, f"C{cluster_id:02d}")
                if cluster_names else f"C{cluster_id:02d}"
            )
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[color],
                s=point_size,
                alpha=alpha,
                label=f"{name} (n={mask.sum():,})",
                zorder=2,
                linewidths=0,
                rasterized=True,
            )
            # Label centroid
            centroid = umap_coords[mask].mean(axis=0)
            ax.text(
                centroid[0], centroid[1],
                str(cluster_id),
                fontsize=7, ha="center", va="center",
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.1", fc=color, alpha=0.7, lw=0),
            )
            cluster_idx += 1

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(f"{title} ({n_clusters} clusters)", fontsize=14, pad=15)

        # Place legend outside plot if many clusters
        if n_clusters <= 15:
            ax.legend(
                loc="upper right", framealpha=0.9,
                markerscale=2.0, ncol=1, fontsize=9
            )
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()

        if save_path:
            _save_figure(fig, save_path, self.dpi)

        return fig

    def plot_umap_by_moa(
        self,
        umap_coords: np.ndarray,
        metadata: pd.DataFrame,
        moa_col: str = "moa",
        top_n_moa: int = 12,
        save_path: Optional[str] = None,
        alpha: float = 0.7,
        point_size: float = 10.0,
        title: str = "UMAP — Mechanism of Action",
    ) -> "plt.Figure":
        """
        UMAP scatter plot colored by compound MoA annotation.

        Only annotated compound perturbations are colored; unannotated and
        CRISPR perturbations are shown in grey background.

        Parameters
        ----------
        umap_coords : np.ndarray, shape (N, 2)
        metadata : pd.DataFrame, shape (N,)
        moa_col : str
        top_n_moa : int
            Number of most common MoAs to color distinctly.
        save_path : str, optional

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)
        meta = metadata.reset_index(drop=True)

        # Plot background (unlabeled) in grey
        if moa_col in meta.columns:
            unlabeled_mask = meta[moa_col].isna() | (meta[moa_col] == "")
        else:
            unlabeled_mask = np.ones(len(umap_coords), dtype=bool)

        ax.scatter(
            umap_coords[unlabeled_mask, 0],
            umap_coords[unlabeled_mask, 1],
            c="#DDDDDD", s=point_size * 0.3, alpha=0.2, zorder=1,
            linewidths=0, rasterized=True,
        )

        if moa_col in meta.columns:
            # Top N MoAs by frequency
            moa_counts = meta[moa_col].dropna().value_counts()
            top_moas = moa_counts.head(top_n_moa).index.tolist()

            for moa_idx, moa in enumerate(top_moas):
                mask = meta[moa_col] == moa
                color = MoA_PALETTE[moa_idx % len(MoA_PALETTE)]
                ax.scatter(
                    umap_coords[mask, 0],
                    umap_coords[mask, 1],
                    c=color, s=point_size, alpha=alpha,
                    label=f"{moa} (n={mask.sum()})",
                    zorder=3, linewidths=0.3, edgecolors="white",
                    rasterized=True,
                )

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(title, fontsize=14, pad=15)
        ax.legend(
            loc="upper right", framealpha=0.9,
            markerscale=1.5, fontsize=8,
            title=f"Top {top_n_moa} MoAs",
        )
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()

        if save_path:
            _save_figure(fig, save_path, self.dpi)

        return fig

    # ------------------------------------------------------------------
    # Heatmaps
    # ------------------------------------------------------------------

    def plot_cluster_heatmap(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metadata: pd.DataFrame,
        gene_col: str = "gene",
        top_n_genes: int = 50,
        save_path: Optional[str] = None,
        title: str = "Phenotypic Cluster Heatmap — Top Genes",
    ) -> Optional["plt.Figure"]:
        """
        Seaborn clustermap of mean embeddings (PCA-projected) per cluster.

        Shows the top genes per cluster in a hierarchically clustered heatmap,
        revealing pathway relationships between clusters.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
        labels : np.ndarray, shape (N,)
        metadata : pd.DataFrame, shape (N,)
        gene_col : str
        top_n_genes : int
        save_path : str, optional

        Returns
        -------
        matplotlib Figure or None
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("seaborn not available. Skipping cluster heatmap.")
            return None

        meta = metadata.reset_index(drop=True)
        valid_mask = labels >= 0

        if valid_mask.sum() == 0:
            logger.warning("No valid cluster assignments for heatmap.")
            return None

        unique_labels = sorted(set(labels[valid_mask]))

        # Compute per-cluster mean embeddings
        cluster_embs = []
        cluster_ids = []
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            if mask.sum() == 0:
                continue
            cluster_embs.append(embeddings[mask].mean(axis=0))
            cluster_ids.append(cluster_id)

        if len(cluster_embs) < 2:
            logger.warning("Need >= 2 clusters for heatmap.")
            return None

        cluster_matrix = np.stack(cluster_embs)  # (n_clusters, D)

        # Reduce to top PCA components for visualization
        n_components = min(top_n_genes, cluster_matrix.shape[0] - 1, cluster_matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        cluster_pca = pca.fit_transform(
            StandardScaler().fit_transform(cluster_matrix)
        )

        heatmap_df = pd.DataFrame(
            cluster_pca,
            index=[f"C{cid:02d}" for cid in cluster_ids],
            columns=[f"PC{i+1}" for i in range(n_components)],
        )

        # Clustermap
        g = sns.clustermap(
            heatmap_df,
            cmap="RdBu_r",
            center=0,
            figsize=(max(12, n_components // 3), max(8, len(cluster_ids) // 2)),
            dendrogram_ratio=(0.15, 0.15),
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            xticklabels=True,
            yticklabels=True,
        )
        g.fig.suptitle(title, fontsize=13, y=1.01)
        g.ax_heatmap.set_xlabel("Principal Components (proxy for morphological features)")
        g.ax_heatmap.set_ylabel("Phenotypic Clusters")

        if save_path:
            _save_figure(g.fig, save_path, self.dpi)

        return g.fig

    # ------------------------------------------------------------------
    # Retrieval plots
    # ------------------------------------------------------------------

    def plot_retrieval_precision_recall(
        self,
        recall_at_k: Dict[int, float],
        precision_at_k: Optional[Dict[int, float]] = None,
        map_score: Optional[float] = None,
        save_path: Optional[str] = None,
        title: str = "MoA Retrieval — Recall@k",
        per_moa_results: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> "plt.Figure":
        """
        Plot Recall@k and optionally Precision@k curves.

        Parameters
        ----------
        recall_at_k : dict
            Mapping of k → Recall@k value (overall).
        precision_at_k : dict, optional
        map_score : float, optional
        save_path : str, optional
        title : str
        per_moa_results : dict, optional
            Mapping of MoA_name → {k → recall} for per-MoA curves.

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))

        k_vals = sorted(recall_at_k.keys())
        r_vals = [recall_at_k[k] for k in k_vals]

        ax.plot(k_vals, r_vals, "bo-", linewidth=2.5, markersize=8,
                label=f"Overall Recall@k (MAP={map_score:.3f})" if map_score else "Overall Recall@k",
                zorder=3)
        ax.fill_between(k_vals, r_vals, alpha=0.12, color="blue")

        if precision_at_k:
            p_vals = [precision_at_k.get(k, 0) for k in k_vals]
            ax.plot(k_vals, p_vals, "rs--", linewidth=2, markersize=7,
                    label="Precision@k", zorder=3)

        if per_moa_results:
            for moa_name, moa_recall in per_moa_results.items():
                mr_vals = [moa_recall.get(k, 0) for k in k_vals]
                ax.plot(k_vals, mr_vals, linewidth=1.2, alpha=0.6,
                        label=moa_name, zorder=2)

        # Annotate peak Recall@k
        if r_vals:
            max_k = k_vals[int(np.argmax(r_vals))]
            max_r = max(r_vals)
            ax.annotate(
                f"{max_r:.1%}",
                xy=(max_k, max_r),
                xytext=(max_k + 1, max_r - 0.05),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="grey"),
            )

        ax.set_xlabel("k (number of neighbors retrieved)", fontsize=12)
        ax.set_ylabel("Recall@k", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(
            __import__("matplotlib.ticker", fromlist=["PercentFormatter"]).PercentFormatter(xmax=1)
        )
        ax.legend(loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            _save_figure(fig, save_path, self.dpi)

        return fig

    # ------------------------------------------------------------------
    # Pathway plots
    # ------------------------------------------------------------------

    def plot_pathway_enrichment_barplot(
        self,
        enrichment_results: Dict[int, "pd.DataFrame"],
        cluster_ids: Optional[List[int]] = None,
        top_n: int = 10,
        score_col: str = "combined_score",
        term_col: str = "term",
        save_path: Optional[str] = None,
        n_cols: int = 3,
    ) -> "plt.Figure":
        """
        Faceted barplot of top enriched pathways per cluster.

        Parameters
        ----------
        enrichment_results : dict
            cluster_id → enrichment DataFrame from run_enrichr().
        cluster_ids : list of int, optional
            Which clusters to plot. Defaults to all with results.
        top_n : int
            Number of pathways to show per cluster.
        score_col : str
        term_col : str
        save_path : str, optional
        n_cols : int
            Number of subplot columns.

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if cluster_ids is None:
            cluster_ids = [
                cid for cid, df in enrichment_results.items()
                if df is not None and len(df) > 0
            ]

        if not cluster_ids:
            logger.warning("No enrichment results to plot.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No enrichment results", ha="center", va="center")
            return fig

        n_plots = len(cluster_ids)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            squeeze=False,
        )

        for plot_idx, cluster_id in enumerate(cluster_ids):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row][col]

            df = enrichment_results.get(cluster_id)
            if df is None or len(df) == 0:
                ax.text(0.5, 0.5, f"Cluster {cluster_id}\nNo results", ha="center", va="center")
                ax.axis("off")
                continue

            # Select columns
            _term_col = term_col if term_col in df.columns else df.columns[1]
            _score_col = score_col if score_col in df.columns else (
                df.select_dtypes(include=[np.number]).columns[0]
                if len(df.select_dtypes(include=[np.number]).columns) > 0
                else df.columns[-1]
            )

            plot_df = df.head(top_n)[[_term_col, _score_col]].copy()
            plot_df = plot_df.dropna()
            plot_df[_score_col] = pd.to_numeric(plot_df[_score_col], errors="coerce")
            plot_df = plot_df.dropna()
            plot_df = plot_df.sort_values(_score_col, ascending=True)

            # Truncate long pathway names
            plot_df[_term_col] = plot_df[_term_col].str[:45]

            colors = plt.cm.RdYlBu_r(
                np.linspace(0.2, 0.8, len(plot_df))
            )
            bars = ax.barh(
                plot_df[_term_col],
                plot_df[_score_col],
                color=colors,
            )
            ax.set_xlabel(score_col.replace("_", " ").title(), fontsize=9)
            ax.set_title(f"Cluster {cluster_id}", fontsize=11, fontweight="bold")
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="x", alpha=0.3)

        # Hide empty subplots
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].set_visible(False)

        fig.suptitle(
            "Pathway Enrichment per Cluster (Enrichr)",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()

        if save_path:
            _save_figure(fig, save_path, self.dpi)

        return fig

    # ------------------------------------------------------------------
    # Interactive UMAP (Plotly)
    # ------------------------------------------------------------------

    def create_interactive_umap(
        self,
        umap_coords: np.ndarray,
        metadata: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        color_by: str = "perturbation_type",
        hover_cols: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: str = "RxRx3-core — OpenPhenom UMAP (Interactive)",
        max_points: int = 20_000,
    ) -> Optional[object]:
        """
        Create an interactive Plotly scatter plot of UMAP embeddings.

        Includes hover labels with perturbation metadata (gene, compound, MoA).
        Can be saved as a self-contained HTML file.

        Parameters
        ----------
        umap_coords : np.ndarray, shape (N, 2)
        metadata : pd.DataFrame, shape (N,)
        labels : np.ndarray, shape (N,), optional
            Cluster labels to add as color dimension.
        color_by : str
            Column in metadata to color by. Also accepts "cluster".
        hover_cols : list of str, optional
            Additional metadata columns to show on hover.
        save_path : str, optional
            Path for HTML output (e.g., "results/umap.html").
        title : str
        max_points : int
            Subsample for performance if N > max_points.

        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not available. Skipping interactive UMAP.")
            return None

        meta = metadata.reset_index(drop=True).copy()

        # Add UMAP coordinates
        meta["UMAP_1"] = umap_coords[:, 0]
        meta["UMAP_2"] = umap_coords[:, 1]

        # Add cluster labels
        if labels is not None:
            meta["cluster"] = labels.astype(str)
            meta["cluster"] = meta["cluster"].replace("-1", "noise")

        # Subsample if too large
        if len(meta) > max_points:
            logger.info(
                f"Subsampling {len(meta)} → {max_points} points for interactive plot."
            )
            rng = np.random.default_rng(42)
            idx = rng.choice(len(meta), size=max_points, replace=False)
            meta = meta.iloc[idx].copy()

        # Determine color column
        if color_by == "cluster" and "cluster" in meta.columns:
            color_col = "cluster"
        elif color_by in meta.columns:
            color_col = color_by
        else:
            color_col = None

        # Default hover columns
        if hover_cols is None:
            hover_cols = [
                c for c in ["gene", "compound", "moa", "perturbation_type",
                             "plate", "concentration_um", "cluster"]
                if c in meta.columns
            ]

        # Build plot
        fig = px.scatter(
            meta,
            x="UMAP_1",
            y="UMAP_2",
            color=color_col,
            hover_data=hover_cols,
            title=title,
            opacity=0.65,
            size_max=6,
            template="plotly_white",
        )

        fig.update_traces(
            marker=dict(size=4, line=dict(width=0)),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            font=dict(family="Arial", size=12),
            legend=dict(
                title=color_col or "",
                itemsizing="constant",
            ),
            width=900,
            height=700,
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path, include_plotlyjs="cdn")
            logger.info(f"Interactive UMAP saved to {save_path}")

        return fig

    # ------------------------------------------------------------------
    # Batch save
    # ------------------------------------------------------------------

    def save_all_figures(
        self,
        umap_coords: np.ndarray,
        metadata: pd.DataFrame,
        labels: np.ndarray,
        embeddings: np.ndarray,
        enrichment_results: Optional[Dict[int, "pd.DataFrame"]],
        recall_at_k: Optional[Dict[int, float]],
        output_dir: str = "results/figures/",
        map_score: Optional[float] = None,
    ) -> List[str]:
        """
        Generate and save all standard pipeline figures.

        Parameters
        ----------
        umap_coords : np.ndarray, shape (N, 2)
        metadata : pd.DataFrame
        labels : np.ndarray, shape (N,)
        embeddings : np.ndarray, shape (N, D)
        enrichment_results : dict, optional
        recall_at_k : dict, optional
        output_dir : str
        map_score : float, optional

        Returns
        -------
        list of str
            Paths of all saved figure files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []

        # 1. UMAP by perturbation type
        fig = self.plot_umap_by_perturbation_type(
            umap_coords, metadata,
            save_path=str(out / "umap_by_perturbation_type.png"),
        )
        saved.append(str(out / "umap_by_perturbation_type.png"))
        import matplotlib.pyplot as plt
        plt.close(fig)

        # 2. UMAP by cluster
        fig = self.plot_umap_by_cluster(
            umap_coords, labels,
            save_path=str(out / "umap_by_cluster.png"),
        )
        saved.append(str(out / "umap_by_cluster.png"))
        plt.close(fig)

        # 3. UMAP by MoA
        if "moa" in metadata.columns:
            fig = self.plot_umap_by_moa(
                umap_coords, metadata,
                save_path=str(out / "umap_by_moa.png"),
            )
            saved.append(str(out / "umap_by_moa.png"))
            plt.close(fig)

        # 4. Cluster heatmap
        fig = self.plot_cluster_heatmap(
            embeddings, labels, metadata,
            save_path=str(out / "cluster_heatmap.png"),
        )
        if fig:
            saved.append(str(out / "cluster_heatmap.png"))
            plt.close(fig)

        # 5. Recall@k
        if recall_at_k:
            fig = self.plot_retrieval_precision_recall(
                recall_at_k,
                map_score=map_score,
                save_path=str(out / "retrieval_recall_at_k.png"),
            )
            saved.append(str(out / "retrieval_recall_at_k.png"))
            plt.close(fig)

        # 6. Pathway enrichment
        if enrichment_results:
            fig = self.plot_pathway_enrichment_barplot(
                enrichment_results,
                save_path=str(out / "pathway_enrichment_barplot.png"),
            )
            saved.append(str(out / "pathway_enrichment_barplot.png"))
            plt.close(fig)

        # 7. Interactive UMAP (HTML)
        self.create_interactive_umap(
            umap_coords, metadata, labels,
            save_path=str(out / "interactive_umap.html"),
        )
        saved.append(str(out / "interactive_umap.html"))

        logger.info(f"Saved {len(saved)} figures to {out}")
        return saved


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_figure(fig: "plt.Figure", save_path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk, creating parent directories as needed."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(save_path).suffix.lower()
    kwargs = {"dpi": dpi, "bbox_inches": "tight"}
    if ext == ".pdf":
        kwargs["format"] = "pdf"
    elif ext == ".svg":
        kwargs["format"] = "svg"
    fig.savefig(save_path, **kwargs)
    logger.info(f"Figure saved: {save_path}")
