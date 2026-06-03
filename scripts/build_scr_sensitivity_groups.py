#!/usr/bin/env python3
"""Build SCR responder/learner flags for MVPA sensitivity analyses.

This script converts the outputs from analysis_scr.ipynb or
identify_fear_learning_subjects_scr.ipynb into one subject-level CSV. The MVPA
models can then filter rows without rerunning expensive decoding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mvpa_l2_common import normalize_subject_id, write_csv


FLAG_FILES = {
    "SCR_Physiological_Responder": "physiological_responder_subjects.txt",
    "SCR_Simple_Acquisition_Differential_Learner": "simple_CSplus_learner_subjects.txt",
    "SCR_Habituation_Adjusted_Learner": "habituation_adjusted_CSplus_learner_subjects.txt",
    "SCR_Late_Phase_Sensitivity_Learner": "late_phase_sensitivity_CSplus_learner_subjects.txt",
}


def read_subject_txt(path: Path) -> set[str]:
    if not path.exists():
        return set()
    subjects = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        subjects.add(normalize_subject_id(line))
    return subjects


def build_flags(scr_dir: Path) -> pd.DataFrame:
    subject_sets = {flag: read_subject_txt(scr_dir / filename) for flag, filename in FLAG_FILES.items()}
    all_subjects = sorted(set().union(*subject_sets.values()))

    learner_csv = scr_dir / "scr_acquisition_learner_subjects.csv"
    learner_df = None
    if learner_csv.exists():
        learner_df = pd.read_csv(learner_csv)
        id_col = "sub_ID" if "sub_ID" in learner_df.columns else learner_df.columns[0]
        learner_df["sub_ID"] = learner_df[id_col].map(normalize_subject_id)
        all_subjects = sorted(set(all_subjects).union(set(learner_df["sub_ID"].dropna())))

    out = pd.DataFrame({"sub_ID": all_subjects})
    for flag, subjects in subject_sets.items():
        out[flag] = out["sub_ID"].isin(subjects)

    if learner_df is not None:
        keep_cols = [
            c
            for c in learner_df.columns
            if c == "sub_ID" or c.startswith("physiological_") or "learner" in c or "diff" in c or "beta" in c
        ]
        extra = learner_df[keep_cols].drop_duplicates("sub_ID")
        out = out.merge(extra, on="sub_ID", how="left", suffixes=("", "_raw"))

    return out.sort_values("sub_ID").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scr-dir", type=Path, default=Path("scr_analysis_outputs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv"),
    )
    args = parser.parse_args()

    flags = build_flags(args.scr_dir)
    write_csv(flags, args.out)
    print(f"Wrote {len(flags)} subject SCR sensitivity rows -> {args.out}")
    for col in FLAG_FILES:
        print(f"  {col}: {int(flags[col].sum())}")


if __name__ == "__main__":
    main()

