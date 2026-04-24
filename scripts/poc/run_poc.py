#!/usr/bin/env python3
"""
POC: MoA retrieval benchmark on LINCS Cell Painting morphological profiles.

Scientific question: Can morphological profile similarity recover
mechanism-of-action (MoA) groupings for compounds?

Dataset: LINCS Cell Painting (cpg0004-lincs), batch
  2016_04_01_a549_48hr_batch1 from the Broad Cell Painting Gallery
  (AWS Open Data, public HTTPS, no account needed).

Substitution note: This replaces Recursion RxRx3-core (which requires a
Hugging Face account). The scientific question is identical.

Pipeline:
  1. Download multiple LINCS plates of normalized, feature-selected profiles
  2. Use embedded MoA annotations (Metadata_moa)
  3. Aggregate replicate wells (doses) to one profile per compound (mean)
  4. Cosine-similarity NN retrieval
  5. Recall@K (K=1,5,10) vs. random baseline
  6. UMAP colored by top-10 most abundant MoAs
"""

from __future__ import annotations

import io
import os
import sys
import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # repo root
RESULTS = REPO / "results" / "poc"
CACHE = REPO / "data_cache"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

BUCKET = "https://cellpainting-gallery.s3.amazonaws.com"
BATCH = "2016_04_01_a549_48hr_batch1"
# Pick 8 plates from the batch to get ~100+ unique compounds.
PLATES = [
    "SQ00014812", "SQ00014813", "SQ00014814", "SQ00014815",
    "SQ00014816", "SQ00014817", "SQ00014818", "SQ00014819",
]


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())
    return dest


def load_plate(plate_id: str) -> pd.DataFrame:
    url = (
        f"{BUCKET}/cpg0004-lincs/broad/workspace/profiles/{BATCH}/"
        f"{plate_id}/{plate_id}_normalized_feature_select.csv.gz"
    )
    dest = CACHE / f"{plate_id}_normalized_feature_select.csv.gz"
    download(url, dest)
    return pd.read_csv(dest)


def main() -> int:
    print("=" * 70)
    print("POC: MoA retrieval benchmark (LINCS Cell Painting)")
    print("=" * 70)

    print(f"\n[1/7] Loading {len(PLATES)} plates from batch {BATCH}")
    frames = []
    for p in PLATES:
        try:
            df = load_plate(p)
            frames.append(df)
            print(f"  {p}: {df.shape}")
        except Exception as e:
            print(f"  {p}: FAILED ({e})")
    if not frames:
        print("ERROR: no plates loaded", file=sys.stderr)
        return 1

    # Align columns across plates (intersection of feature columns)
    common_cols = set(frames[0].columns)
    for df in frames[1:]:
        common_cols &= set(df.columns)
    common_cols = [c for c in frames[0].columns if c in common_cols]
    data = pd.concat([df[common_cols] for df in frames], ignore_index=True)
    print(f"\nCombined shape: {data.shape}")

    print("\n[2/7] Filtering wells: drop DMSO / untreated / missing MoA")
    data = data[data["Metadata_broad_sample"].astype(str) != "DMSO"].copy()
    data = data[data["Metadata_broad_sample"].notna()]
    data = data[data["Metadata_moa"].notna()]
    data = data[data["Metadata_moa"].astype(str).str.strip() != ""]
    print(f"  wells remaining: {len(data)}")
    print(f"  unique compounds: {data['Metadata_broad_sample'].nunique()}")

    # Feature columns = non-metadata numeric columns
    feat_cols = [c for c in data.columns if not c.startswith("Metadata_")]
    feat_cols = [c for c in feat_cols if np.issubdtype(data[c].dtype, np.number)]
    # Drop features with any NaN across the full table
    feat_df = data[feat_cols]
    good_feats = feat_df.columns[feat_df.notna().all()].tolist()
    print(f"  feature columns (after NaN drop): {len(good_feats)}")

    print("\n[3/7] Aggregating replicates (all doses) per compound: mean profile")
    moa_per_cpd = (
        data.groupby("Metadata_broad_sample")["Metadata_moa"]
        .agg(lambda s: s.value_counts().index[0])
    )
    agg = data.groupby("Metadata_broad_sample")[good_feats].mean()
    agg["moa"] = moa_per_cpd
    agg["moa_primary"] = agg["moa"].astype(str).str.split("|").str[0].str.strip()
    agg = agg[agg["moa_primary"].str.len() > 0].copy()
    print(f"  compounds aggregated: {len(agg)}")
    print(f"  unique primary MoAs: {agg['moa_primary'].nunique()}")

    # Restrict to MoAs with at least 2 compounds (retrieval requires a target)
    moa_counts = agg["moa_primary"].value_counts()
    eligible_moas = moa_counts[moa_counts >= 2].index
    bench = agg[agg["moa_primary"].isin(eligible_moas)].copy()
    print(f"  benchmarkable compounds (MoA with >=2 members): {len(bench)}")
    print(f"  eligible MoAs: {len(eligible_moas)}")

    if len(bench) < 10:
        print("ERROR: too few benchmarkable compounds", file=sys.stderr)
        return 2

    X = bench[good_feats].to_numpy(dtype=np.float32)
    labels = bench["moa_primary"].to_numpy()

    print("\n[4/7] Computing cosine similarity matrix")
    S = cosine_similarity(X)
    np.fill_diagonal(S, -np.inf)

    print("\n[5/7] MoA retrieval: recall@K")
    Ks = [1, 5, 10]
    n = len(bench)
    order = np.argsort(-S, axis=1)

    hits_at = {k: [] for k in Ks}
    for i in range(n):
        true_moa = labels[i]
        same_moa_mask = (labels == true_moa)
        same_moa_mask[i] = False
        n_same = int(same_moa_mask.sum())
        if n_same == 0:
            continue
        top_order = order[i]
        for k in Ks:
            top_k = top_order[:k]
            hits = int(same_moa_mask[top_k].sum())
            hits_at[k].append(hits / k)

    mean_recall = {k: float(np.mean(hits_at[k])) for k in Ks}

    rand_per_query = []
    for i in range(n):
        true_moa = labels[i]
        n_same = int((labels == true_moa).sum()) - 1
        if n_same < 0:
            continue
        rand_per_query.append(n_same / (n - 1))
    random_baseline = float(np.mean(rand_per_query))

    print(f"  N benchmarkable compounds: {n}")
    for k in Ks:
        ratio = mean_recall[k] / random_baseline if random_baseline > 0 else float("nan")
        print(
            f"  recall@{k:<2d} = {mean_recall[k]:.4f}   "
            f"(random = {random_baseline:.4f}, {ratio:.2f}x)"
        )

    recall_df = pd.DataFrame(
        {
            "k": Ks,
            "recall_at_k": [mean_recall[k] for k in Ks],
            "random_baseline": [random_baseline] * len(Ks),
            "fold_over_random": [
                mean_recall[k] / random_baseline if random_baseline > 0 else np.nan
                for k in Ks
            ],
        }
    )
    recall_df.to_csv(RESULTS / "recall_at_k.csv", index=False)
    print(f"  wrote {RESULTS / 'recall_at_k.csv'}")

    print("\n[6/7] UMAP of compound profiles colored by top-10 MoAs")
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=15, min_dist=0.2, metric="cosine", random_state=0
        )
        emb = reducer.fit_transform(X)
    except Exception as e:
        print(f"  UMAP failed ({e}); falling back to PCA(2)")
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=0).fit_transform(X)

    top10_moas = bench["moa_primary"].value_counts().head(10).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 7))
    not_top = ~np.isin(labels, top10_moas)
    ax.scatter(
        emb[not_top, 0], emb[not_top, 1],
        s=12, c="lightgrey", alpha=0.5, label="other",
    )
    cmap = plt.get_cmap("tab10")
    for idx, moa in enumerate(top10_moas):
        mask = labels == moa
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            s=35, color=cmap(idx),
            label=f"{moa} (n={int(mask.sum())})",
            edgecolors="black", linewidths=0.3,
        )
    ax.set_title(
        f"LINCS Cell Painting: UMAP of {n} compound profiles\n"
        f"(colored by top-10 MoAs)"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, frameon=False,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "umap_moa.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {RESULTS / 'umap_moa.png'}")

    print("\n[7/7] Writing summary")
    lines = []
    lines.append("POC: MoA retrieval benchmark on LINCS Cell Painting")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Dataset: LINCS Cell Painting (cpg0004-lincs)")
    lines.append(f"  Batch: {BATCH}")
    lines.append(f"  Plates used: {', '.join(PLATES)}")
    lines.append(
        f"  Source: {BUCKET}/cpg0004-lincs/broad/workspace/profiles/"
    )
    lines.append(
        "  (Substitution for Recursion RxRx3-core, which requires an HF account.)"
    )
    lines.append("")
    lines.append(
        "Pipeline: per-compound mean profile over all replicate wells/doses;"
    )
    lines.append(
        "  feature-selected normalized profiles; cosine similarity NN retrieval."
    )
    lines.append("")
    lines.append(f"N wells (post-filter): {len(data)}")
    lines.append(f"N features used: {len(good_feats)}")
    lines.append(f"N unique compounds: {len(agg)}")
    lines.append(f"N unique primary MoAs: {agg['moa_primary'].nunique()}")
    lines.append(
        f"N benchmarkable compounds (MoA with >=2 members): {n}"
    )
    lines.append(f"N eligible MoAs (>=2 members): {len(eligible_moas)}")
    lines.append("")
    lines.append("Recall@K (cosine NN over morphological profile space):")
    for k in Ks:
        ratio = mean_recall[k] / random_baseline if random_baseline > 0 else float("nan")
        lines.append(
            f"  recall@{k:<2d} = {mean_recall[k]:.4f}   "
            f"random = {random_baseline:.4f}   ({ratio:.2f}x random)"
        )
    lines.append("")
    lines.append("Top-10 most abundant MoA classes (N compounds in benchmark):")
    for moa, cnt in bench["moa_primary"].value_counts().head(10).items():
        lines.append(f"  {cnt:4d}  {moa}")
    lines.append("")
    lines.append("Interpretation:")
    fold5 = mean_recall[5] / random_baseline if random_baseline > 0 else 0.0
    if fold5 >= 2.0:
        lines.append(
            f"  recall@5 is {fold5:.2f}x the random baseline -- "
            "real MoA signal is recoverable from morphology."
        )
    else:
        lines.append(
            f"  recall@5 is only {fold5:.2f}x random -- "
            "signal is weak on this subset (untuned cosine similarity; "
            "no batch correction or class-sphere CCA)."
        )
    lines.append("")
    lines.append("Reproduce:")
    lines.append("  python scripts/poc/run_poc.py")
    lines.append("")
    summary = "\n".join(lines)
    (RESULTS / "poc_summary.txt").write_text(summary)
    print(summary)

    # Regenerate results/poc/manifest.json so the portfolio site's
    # headline numbers stay in sync with the freshly-written CSVs.
    print("[8/7] Rebuilding manifest.json")
    import subprocess
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("build_manifest.py"))],
        check=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
