"""
RxRx3DataLoader — loads metadata and precomputed embeddings for the
Recursion RxRx3-core phenomics dataset.

RxRx3-core schema
-----------------
The metadata CSV shipped with RxRx3-core has the following columns:

    experiment          str     e.g. "HUVEC-1"
    plate               str     e.g. "Plate1"
    well                str     e.g. "A01"
    site                int     field-of-view index (1–6)
    cell_line           str     "HUVEC"
    perturbation_type   str     "CRISPR" | "compound" | "negcon" | "poscon"
    gene                str     target gene symbol for CRISPR KOs, NaN otherwise
    guide_sequence      str     sgRNA sequence for CRISPR
    compound            str     compound name / InChIKey for compound wells
    concentration_um    float   compound concentration in µM
    moa                 str     mechanism of action annotation (if available)
    smiles              str     SMILES string for compounds
    replicate           int     biological replicate index (1–6)
    embedding_id        int     row index into the embeddings array
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name constants matching the RxRx3-core schema
# ---------------------------------------------------------------------------

EXPERIMENT_COL = "experiment"
PLATE_COL = "plate"
WELL_COL = "well"
SITE_COL = "site"
CELL_LINE_COL = "cell_line"
PERTURBATION_TYPE_COL = "perturbation_type"
GENE_COL = "gene"
GUIDE_COL = "guide_sequence"
COMPOUND_COL = "compound"
CONCENTRATION_COL = "concentration_um"
MOA_COL = "moa"
SMILES_COL = "smiles"
REPLICATE_COL = "replicate"
EMBEDDING_ID_COL = "embedding_id"

CRISPR_LABEL = "CRISPR"
COMPOUND_LABEL = "compound"
NEGCON_LABEL = "negcon"
POSCON_LABEL = "poscon"


class RxRx3DataLoader:
    """
    Loader for the RxRx3-core phenomics dataset.

    Handles metadata loading, embedding I/O, per-perturbation aggregation,
    leave-one-plate-out splitting, and percent-replicating computation.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing `raw/` and `embeddings/` sub-directories.
    config : PipelineConfig, optional
        Pipeline configuration object. If None, uses defaults.

    Examples
    --------
    >>> loader = RxRx3DataLoader("data/")
    >>> metadata = loader.load_metadata("data/raw/metadata.csv")
    >>> embeddings, index = loader.load_embeddings("data/embeddings/")
    >>> crispr = loader.get_crispr_perturbations(metadata)
    """

    def __init__(self, data_dir: str = "data/", config=None):
        self.data_dir = Path(data_dir)
        self.config = config
        self._metadata: Optional[pd.DataFrame] = None
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_index: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and validate the RxRx3-core metadata CSV.

        Parameters
        ----------
        metadata_path : str, optional
            Path to `metadata.csv`. Defaults to `{data_dir}/raw/metadata.csv`.

        Returns
        -------
        pd.DataFrame
            Cleaned metadata with standardized column names and dtypes.
        """
        if metadata_path is None:
            metadata_path = self.data_dir / "raw" / "metadata.csv"

        path = Path(metadata_path)
        if not path.exists():
            logger.warning(
                f"Metadata file not found at {path}. "
                "Generating synthetic metadata for demonstration."
            )
            return self._generate_synthetic_metadata()

        logger.info(f"Loading metadata from {path}")
        meta = pd.read_csv(path, low_memory=False)
        meta = self._validate_and_clean_metadata(meta)
        self._metadata = meta
        logger.info(
            f"Loaded {len(meta)} well records | "
            f"{meta[PERTURBATION_TYPE_COL].value_counts().to_dict()}"
        )
        return meta

    def _validate_and_clean_metadata(self, meta: pd.DataFrame) -> pd.DataFrame:
        """Validate schema and coerce dtypes."""
        required_cols = [
            PLATE_COL, WELL_COL, PERTURBATION_TYPE_COL,
        ]
        missing = [c for c in required_cols if c not in meta.columns]
        if missing:
            raise ValueError(f"Metadata is missing required columns: {missing}")

        # Coerce types
        if CONCENTRATION_COL in meta.columns:
            meta[CONCENTRATION_COL] = pd.to_numeric(
                meta[CONCENTRATION_COL], errors="coerce"
            )
        if REPLICATE_COL in meta.columns:
            meta[REPLICATE_COL] = pd.to_numeric(
                meta[REPLICATE_COL], errors="coerce"
            ).astype("Int64")

        # Standardize perturbation type labels
        pt = meta[PERTURBATION_TYPE_COL].str.strip().str.lower()
        label_map = {
            "crispr": CRISPR_LABEL,
            "crispr_ko": CRISPR_LABEL,
            "knockout": CRISPR_LABEL,
            "compound": COMPOUND_LABEL,
            "small_molecule": COMPOUND_LABEL,
            "negcon": NEGCON_LABEL,
            "negative_control": NEGCON_LABEL,
            "dmso": NEGCON_LABEL,
            "poscon": POSCON_LABEL,
            "positive_control": POSCON_LABEL,
        }
        meta[PERTURBATION_TYPE_COL] = pt.map(label_map).fillna(
            meta[PERTURBATION_TYPE_COL]
        )

        # Assign embedding_id if not present
        if EMBEDDING_ID_COL not in meta.columns:
            meta = meta.reset_index(drop=True)
            meta[EMBEDDING_ID_COL] = meta.index

        return meta.reset_index(drop=True)

    def _generate_synthetic_metadata(self, n_perturbations: int = 500) -> pd.DataFrame:
        """
        Generate a small synthetic metadata table for testing.

        Mimics the RxRx3-core schema with realistic column names and value
        distributions, but uses random data.
        """
        rng = np.random.default_rng(42)

        n_crispr = int(n_perturbations * 0.55)
        n_compound = int(n_perturbations * 0.35)
        n_negcon = n_perturbations - n_crispr - n_compound

        gene_pool = [
            "BRCA1", "BRCA2", "ATM", "ATR", "PARP1", "RAD51", "CHEK1", "CHEK2",
            "TP53", "MDM2", "CDK4", "CDK6", "CCND1", "RB1", "E2F1",
            "MTOR", "TSC1", "TSC2", "RPTOR", "AKT1", "PIK3CA", "PTEN",
            "EGFR", "KRAS", "BRAF", "MAP2K1", "MAPK1",
            "BCL2", "BCL2L1", "BAX", "CASP3", "CASP9",
            "HDAC1", "HDAC2", "EP300", "KAT2A",
            "PSMA1", "PSMB1", "PSMC1", "PSMD1", "UBA1",
            "ATG5", "ATG7", "BECN1", "ULK1", "SQSTM1",
            "SF3B1", "U2AF1", "SRSF1", "SRSF2",
            "XPO1", "KPNA2", "RAN",
        ]

        compound_pool = [
            ("olaparib", "PARP inhibitor", 1.0),
            ("niraparib", "PARP inhibitor", 1.0),
            ("rucaparib", "PARP inhibitor", 1.0),
            ("talazoparib", "PARP inhibitor", 0.1),
            ("rapamycin", "mTOR inhibitor", 0.1),
            ("everolimus", "mTOR inhibitor", 1.0),
            ("bortezomib", "Proteasome inhibitor", 0.01),
            ("carfilzomib", "Proteasome inhibitor", 0.1),
            ("erlotinib", "EGFR inhibitor", 1.0),
            ("gefitinib", "EGFR inhibitor", 1.0),
            ("vemurafenib", "BRAF inhibitor", 1.0),
            ("trametinib", "MEK inhibitor", 0.1),
            ("venetoclax", "BCL2 inhibitor", 1.0),
            ("navitoclax", "BCL2 inhibitor", 1.0),
            ("alisertib", "Aurora A inhibitor", 0.1),
            ("barasertib", "Aurora B inhibitor", 0.1),
            ("vorinostat", "HDAC inhibitor", 1.0),
            ("entinostat", "HDAC inhibitor", 1.0),
            ("selinexor", "XPO1 inhibitor", 0.1),
            ("staurosporine", "Kinase inhibitor", 0.001),
        ]

        rows = []
        plates = [f"Plate{i}" for i in range(1, 7)]

        # CRISPR KO wells
        for i in range(n_crispr):
            gene = rng.choice(gene_pool)
            plate = rng.choice(plates)
            rows.append({
                EXPERIMENT_COL: "HUVEC-1",
                PLATE_COL: plate,
                WELL_COL: f"{chr(65 + rng.integers(0, 16))}{rng.integers(1, 25):02d}",
                SITE_COL: int(rng.integers(1, 7)),
                CELL_LINE_COL: "HUVEC",
                PERTURBATION_TYPE_COL: CRISPR_LABEL,
                GENE_COL: gene,
                GUIDE_COL: f"GUIDE_{gene}_{i % 3 + 1}",
                COMPOUND_COL: np.nan,
                CONCENTRATION_COL: np.nan,
                MOA_COL: np.nan,
                SMILES_COL: np.nan,
                REPLICATE_COL: int(i % 4 + 1),
                EMBEDDING_ID_COL: i,
            })

        # Compound wells
        for i in range(n_compound):
            cname, moa, conc = compound_pool[i % len(compound_pool)]
            plate = rng.choice(plates)
            rows.append({
                EXPERIMENT_COL: "HUVEC-1",
                PLATE_COL: plate,
                WELL_COL: f"{chr(65 + rng.integers(0, 16))}{rng.integers(1, 25):02d}",
                SITE_COL: int(rng.integers(1, 7)),
                CELL_LINE_COL: "HUVEC",
                PERTURBATION_TYPE_COL: COMPOUND_LABEL,
                GENE_COL: np.nan,
                GUIDE_COL: np.nan,
                COMPOUND_COL: cname,
                CONCENTRATION_COL: conc,
                MOA_COL: moa,
                SMILES_COL: f"SMILES_{cname}",
                REPLICATE_COL: int(i % 4 + 1),
                EMBEDDING_ID_COL: n_crispr + i,
            })

        # Negative controls
        for i in range(n_negcon):
            plate = rng.choice(plates)
            rows.append({
                EXPERIMENT_COL: "HUVEC-1",
                PLATE_COL: plate,
                WELL_COL: f"{chr(65 + rng.integers(0, 16))}{rng.integers(1, 25):02d}",
                SITE_COL: int(rng.integers(1, 7)),
                CELL_LINE_COL: "HUVEC",
                PERTURBATION_TYPE_COL: NEGCON_LABEL,
                GENE_COL: np.nan,
                GUIDE_COL: np.nan,
                COMPOUND_COL: "DMSO",
                CONCENTRATION_COL: 0.1,
                MOA_COL: "negative_control",
                SMILES_COL: np.nan,
                REPLICATE_COL: int(i % 4 + 1),
                EMBEDDING_ID_COL: n_crispr + n_compound + i,
            })

        df = pd.DataFrame(rows).reset_index(drop=True)
        logger.info(f"Generated synthetic metadata: {len(df)} wells")
        return df

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def load_embeddings(
        self,
        embeddings_dir: Optional[str] = None,
        embeddings_file: Optional[str] = None,
        index_file: Optional[str] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load precomputed embeddings and their metadata index.

        Supports both `.npy` arrays (with a companion index CSV) and standalone
        embedding CSVs where rows are perturbations and columns are embedding dims.

        Parameters
        ----------
        embeddings_dir : str, optional
            Directory to search for embeddings. Defaults to `{data_dir}/embeddings/`.
        embeddings_file : str, optional
            Path to `.npy` embeddings array.
        index_file : str, optional
            Path to index CSV aligned to the embeddings array rows.

        Returns
        -------
        embeddings : np.ndarray, shape (N, D)
            Embedding matrix where N = number of perturbations, D = embedding dim.
        index : pd.DataFrame
            Metadata index aligned row-for-row with `embeddings`.
        """
        if embeddings_dir is None:
            embeddings_dir = self.data_dir / "embeddings"

        emb_dir = Path(embeddings_dir)

        # Resolve file paths
        if embeddings_file is None:
            candidates = list(emb_dir.glob("*.npy"))
            if candidates:
                embeddings_file = str(candidates[0])
            else:
                logger.warning("No .npy embeddings found. Generating synthetic embeddings.")
                return self._generate_synthetic_embeddings()

        emb_path = Path(embeddings_file)
        if not emb_path.exists():
            logger.warning(f"Embeddings file not found: {emb_path}. Generating synthetic.")
            return self._generate_synthetic_embeddings()

        logger.info(f"Loading embeddings from {emb_path}")
        embeddings = np.load(str(emb_path))

        # Load index
        if index_file is None:
            idx_candidates = list(emb_dir.glob("*index*.csv")) + list(emb_dir.glob("*meta*.csv"))
            if idx_candidates:
                index_file = str(idx_candidates[0])

        if index_file and Path(index_file).exists():
            logger.info(f"Loading embedding index from {index_file}")
            index = pd.read_csv(index_file)
        else:
            logger.warning("No index file found. Creating minimal index.")
            index = pd.DataFrame(
                {EMBEDDING_ID_COL: np.arange(len(embeddings))}
            )

        assert len(embeddings) == len(index), (
            f"Embeddings ({len(embeddings)}) and index ({len(index)}) length mismatch"
        )

        self._embeddings = embeddings
        self._embedding_index = index
        logger.info(
            f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}"
        )
        return embeddings, index

    def _generate_synthetic_embeddings(
        self, n_perturbations: int = 500, embedding_dim: int = 1536
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate synthetic embeddings with biologically plausible cluster structure.

        Creates 10 distinct cluster centers with Gaussian noise to simulate
        phenotypically similar groups (e.g., DDR genes, mTOR pathway, etc.).
        """
        rng = np.random.default_rng(42)
        n_clusters = 10
        cluster_size = n_perturbations // n_clusters

        cluster_centers = rng.standard_normal((n_clusters, embedding_dim)) * 2.0
        embeddings_list = []
        labels = []

        for c in range(n_clusters):
            noise = rng.standard_normal((cluster_size, embedding_dim)) * 0.3
            cluster_embs = cluster_centers[c] + noise
            embeddings_list.append(cluster_embs)
            labels.extend([c] * cluster_size)

        # Remainder
        remainder = n_perturbations - n_clusters * cluster_size
        if remainder > 0:
            noise = rng.standard_normal((remainder, embedding_dim)) * 1.0
            embeddings_list.append(noise)
            labels.extend([-1] * remainder)

        embeddings = np.vstack(embeddings_list).astype(np.float32)

        # L2-normalize (unit sphere)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        meta = self._generate_synthetic_metadata(n_perturbations=n_perturbations)
        # Trim or pad to match
        meta = meta.iloc[: len(embeddings)].copy()
        meta[EMBEDDING_ID_COL] = np.arange(len(embeddings))

        logger.info(
            f"Generated synthetic embeddings: shape={embeddings.shape}"
        )
        return embeddings, meta

    # ------------------------------------------------------------------
    # Perturbation subset getters
    # ------------------------------------------------------------------

    def get_crispr_perturbations(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Return rows corresponding to CRISPR knockout perturbations."""
        return metadata[
            metadata[PERTURBATION_TYPE_COL] == CRISPR_LABEL
        ].copy()

    def get_compound_perturbations(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Return rows corresponding to small-molecule compound perturbations."""
        return metadata[
            metadata[PERTURBATION_TYPE_COL] == COMPOUND_LABEL
        ].copy()

    def get_controls(
        self, metadata: pd.DataFrame, control_type: str = "negcon"
    ) -> pd.DataFrame:
        """
        Return control wells.

        Parameters
        ----------
        control_type : str
            "negcon" for negative controls, "poscon" for positive controls.
        """
        label = NEGCON_LABEL if control_type == "negcon" else POSCON_LABEL
        return metadata[metadata[PERTURBATION_TYPE_COL] == label].copy()

    # ------------------------------------------------------------------
    # Plate-level operations
    # ------------------------------------------------------------------

    def split_by_plate(
        self, metadata: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate leave-one-plate-out cross-validation splits.

        For each plate P, returns (train_metadata, test_metadata) where
        test_metadata is restricted to plate P and train_metadata contains
        all other plates.

        Parameters
        ----------
        metadata : pd.DataFrame
            Full metadata table.

        Returns
        -------
        List of (train_df, test_df) tuples — one per plate.
        """
        if PLATE_COL not in metadata.columns:
            raise ValueError(f"Metadata must contain '{PLATE_COL}' column.")

        plates = sorted(metadata[PLATE_COL].unique())
        splits = []
        for plate in plates:
            test_mask = metadata[PLATE_COL] == plate
            train_df = metadata[~test_mask].copy()
            test_df = metadata[test_mask].copy()
            splits.append((train_df, test_df))

        logger.info(
            f"Created {len(splits)} leave-one-plate-out splits "
            f"from {len(plates)} plates"
        )
        return splits

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def compute_percent_replicating(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        n_resamples: int = 1000,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Compute Percent Replicating (PR) for CRISPR and compound perturbations.

        Percent Replicating is the fraction of perturbations whose replicate
        similarity (within-perturbation cosine similarity) exceeds the 95th
        percentile of a null distribution constructed from cross-perturbation
        well pairs of the same type.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, D)
            Embedding matrix aligned to `metadata`.
        metadata : pd.DataFrame
            Metadata table with EMBEDDING_ID_COL and perturbation type.
        n_resamples : int
            Number of null distribution samples.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Keys: "CRISPR_percent_replicating", "compound_percent_replicating",
            "CRISPR_mean_replicate_sim", "compound_mean_replicate_sim".
        """
        rng = np.random.default_rng(random_state)
        results: Dict[str, float] = {}

        for pert_type, label in [("CRISPR", CRISPR_LABEL), ("compound", COMPOUND_LABEL)]:
            subset = metadata[metadata[PERTURBATION_TYPE_COL] == label].copy()
            if len(subset) < 4:
                results[f"{pert_type}_percent_replicating"] = float("nan")
                results[f"{pert_type}_mean_replicate_sim"] = float("nan")
                continue

            # Identify the grouping key (gene or compound)
            group_col = GENE_COL if pert_type == "CRISPR" else COMPOUND_COL
            if group_col not in subset.columns:
                continue

            subset = subset.dropna(subset=[group_col])
            emb_ids = subset[EMBEDDING_ID_COL].values.astype(int)
            valid_mask = emb_ids < len(embeddings)
            subset = subset[valid_mask]
            emb_ids = emb_ids[valid_mask]

            if len(subset) == 0:
                continue

            embs = embeddings[emb_ids]

            # Normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embs_norm = embs / norms

            # Build replicate groups
            groups = subset.groupby(group_col)[EMBEDDING_ID_COL].apply(list)

            # Within-group similarities
            within_sims = []
            for gname, ids in groups.items():
                ids_valid = [i for i in ids if i < len(embeddings)]
                if len(ids_valid) < 2:
                    continue
                local_idx = [
                    int(subset[subset[EMBEDDING_ID_COL] == i].index[0]) for i in ids_valid
                    if len(subset[subset[EMBEDDING_ID_COL] == i]) > 0
                ]
                if len(local_idx) < 2:
                    continue
                local_embs = embs_norm[
                    [subset.index.get_loc(i) for i in subset.index if
                     subset.loc[i, EMBEDDING_ID_COL] in ids_valid][:len(ids_valid)]
                ]
                if len(local_embs) < 2:
                    continue
                sim_matrix = local_embs @ local_embs.T
                n = len(local_embs)
                # Upper triangle (excluding diagonal)
                for r in range(n):
                    for c in range(r + 1, n):
                        within_sims.append(float(sim_matrix[r, c]))

            if not within_sims:
                results[f"{pert_type}_percent_replicating"] = float("nan")
                results[f"{pert_type}_mean_replicate_sim"] = float("nan")
                continue

            # Null distribution: random cross-perturbation pairs
            n_embs = len(embs_norm)
            null_sims = []
            for _ in range(n_resamples):
                i, j = rng.integers(0, n_embs, size=2)
                while i == j:
                    j = rng.integers(0, n_embs)
                null_sims.append(float(embs_norm[i] @ embs_norm[j]))

            null_95th = np.percentile(null_sims, 95)
            pct_rep = float(
                np.mean(np.array(within_sims) > null_95th) * 100
            )
            mean_rep_sim = float(np.mean(within_sims))

            results[f"{pert_type}_percent_replicating"] = pct_rep
            results[f"{pert_type}_mean_replicate_sim"] = mean_rep_sim

            logger.info(
                f"{pert_type}: Percent Replicating = {pct_rep:.1f}%, "
                f"Mean replicate cosine sim = {mean_rep_sim:.4f}"
            )

        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_to_perturbation_level(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        group_by: Optional[List[str]] = None,
        agg_method: str = "mean",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Aggregate per-well embeddings to per-perturbation level.

        Groups wells by perturbation identity (gene or compound + concentration)
        and aggregates embeddings by mean (or median).

        Parameters
        ----------
        embeddings : np.ndarray, shape (N_wells, D)
        metadata : pd.DataFrame, shape (N_wells,)
        group_by : list of str, optional
            Columns to group by. Defaults to [perturbation_type, gene/compound].
        agg_method : str
            "mean" or "median".

        Returns
        -------
        agg_embeddings : np.ndarray, shape (N_perturbations, D)
        agg_metadata : pd.DataFrame, shape (N_perturbations,)
        """
        if group_by is None:
            group_by = [PERTURBATION_TYPE_COL, GENE_COL, COMPOUND_COL,
                        CONCENTRATION_COL]
            group_by = [c for c in group_by if c in metadata.columns]

        agg_fn = np.mean if agg_method == "mean" else np.median

        agg_embs = []
        agg_rows = []

        groups = metadata.groupby(group_by, dropna=False)
        for key, grp in tqdm(groups, desc="Aggregating replicates", disable=not logger.isEnabledFor(logging.DEBUG)):
            ids = grp[EMBEDDING_ID_COL].values.astype(int)
            valid = ids[ids < len(embeddings)]
            if len(valid) == 0:
                continue
            agg_embs.append(agg_fn(embeddings[valid], axis=0))
            agg_rows.append(grp.iloc[0].to_dict())

        if not agg_embs:
            raise ValueError("No valid embeddings found after aggregation.")

        agg_embeddings = np.stack(agg_embs).astype(np.float32)
        agg_metadata = pd.DataFrame(agg_rows).reset_index(drop=True)
        agg_metadata[EMBEDDING_ID_COL] = np.arange(len(agg_metadata))

        logger.info(
            f"Aggregated {len(metadata)} wells → "
            f"{len(agg_metadata)} perturbations"
        )
        return agg_embeddings, agg_metadata

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """Cached metadata from last load_metadata() call."""
        return self._metadata

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Cached embeddings from last load_embeddings() call."""
        return self._embeddings
