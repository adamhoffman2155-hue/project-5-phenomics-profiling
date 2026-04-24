#!/usr/bin/env python3
"""
Regenerate ``results/poc/manifest.json`` from the artefacts that
``run_poc.py`` writes:

  - ``recall_at_k.csv`` : MoA retrieval recall@K with random baseline
  - ``poc_summary.txt`` : cohort counts (wells, compounds, MoAs)

The portfolio site at ``bioinformatics-portfolio/shared/poc/project-5.json``
is a snapshot of this manifest; re-copy after running this script so
the portfolio's headline numbers stay in sync with the POC results.

Usage
-----
    python scripts/poc/build_manifest.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "results" / "poc"
OUT = POC / "manifest.json"

SUMMARY_PATTERNS = {
    "n_wells_post_filter": re.compile(r"N wells \(post-filter\):\s*(\d+)"),
    "n_features_used":   re.compile(r"N features used:\s*(\d+)"),
    "n_unique_compounds": re.compile(r"N unique compounds:\s*(\d+)"),
    "n_unique_moas":      re.compile(r"N unique primary MoAs:\s*(\d+)"),
    "n_benchmarkable_compounds": re.compile(
        r"N benchmarkable compounds \(MoA with >=2 members\):\s*(\d+)"
    ),
    "n_eligible_moas": re.compile(
        r"N eligible MoAs \(>=2 members\):\s*(\d+)"
    ),
}


def parse_summary_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, pat in SUMMARY_PATTERNS.items():
        m = pat.search(text)
        if m:
            counts[key] = int(m.group(1))
    return counts


def main() -> int:
    recall_csv = POC / "recall_at_k.csv"
    summary_txt = POC / "poc_summary.txt"
    for f in (recall_csv, summary_txt):
        if not f.is_file():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 1

    recall = pd.read_csv(recall_csv).set_index("k")
    counts = parse_summary_counts(summary_txt.read_text())

    def row(k: int) -> dict:
        r = recall.loc[k]
        return {
            "name": f"Recall@{k}",
            "value": round(float(r["recall_at_k"]), 4),
            "random_baseline": round(float(r["random_baseline"]), 4),
            "ratio_vs_random": round(float(r["fold_over_random"]), 2),
        }

    headline = row(5)
    headline["name"] = (
        "Recall@5 (cosine NN over morphological profiles)"
    )

    manifest = {
        "$schema": (
            "https://github.com/adamhoffman2155-hue/bioinformatics-portfolio/"
            "blob/main/shared/poc-manifest.schema.json"
        ),
        "project": "project-5-phenomics-profiling",
        "poc_title": "MoA retrieval benchmark on LINCS Cell Painting",
        "poc_version": "v1",
        "dataset": {
            "name": "LINCS Cell Painting (cpg0004-lincs)",
            "batch": "2016_04_01_a549_48hr_batch1",
            "plates": [
                "SQ00014812", "SQ00014813", "SQ00014814", "SQ00014815",
                "SQ00014816", "SQ00014817", "SQ00014818", "SQ00014819",
            ],
            "source": (
                "https://cellpainting-gallery.s3.amazonaws.com/"
                "cpg0004-lincs/broad/workspace/profiles/"
            ),
            "substitute_for": "Recursion RxRx3-core (requires HF account)",
            **counts,
        },
        "script": "scripts/poc/run_poc.py",
        "generated_at": date.today().isoformat(),
        "headline_metric": headline,
        "secondary_metrics": [row(1), row(10)],
        "headline_text": (
            f"Recall@5 = {float(recall.loc[5, 'recall_at_k']):.4f} "
            f"({float(recall.loc[5, 'fold_over_random']):.2f}× random "
            "baseline) — real MoA signal is recoverable from morphology."
        ),
        "artifacts": [
            "results/poc/poc_summary.txt",
            "results/poc/recall_at_k.csv",
        ],
    }

    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT}")
    print(f"  headline: {manifest['headline_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
