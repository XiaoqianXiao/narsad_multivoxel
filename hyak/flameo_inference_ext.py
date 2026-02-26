#!/usr/bin/env python3
"""
FLAMEO mixed-effects group inference for extinction searchlight maps.
Mirrors cluster_inference_ext.py contrasts.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img


CS_LABELS = ["CS-", "CSS", "CSR"]


@dataclass
class SubjectInfo:
    group: str
    drug: str


def find_reference_lss(project_root: str, task: str) -> str:
    patterns = [
        os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/subjects",
            f"sub-*_task-{task}_contrast1.nii*",
        ),
        os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/wholeBrain_S4",
            f"sub-*_task-{task}_contrast1.nii*",
        ),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No LSS files found with patterns: {patterns}")


def build_master_mask_from_reference(
    glasser_path: str,
    tian_path: str,
    reference_img_path: str,
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    ref_img = nib.load(reference_img_path)
    img_g = nib.load(glasser_path)
    img_t = nib.load(tian_path)
    glasser_res = resample_to_img(img_g, ref_img, interpolation="nearest")
    tian_res = resample_to_img(img_t, ref_img, interpolation="nearest")
    data_g = glasser_res.get_fdata()
    data_t = tian_res.get_fdata()
    mask = (data_g > 0) | (data_t > 0)
    return mask, nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine)


def load_subject_meta(merged_dir: str, project_root: str, subj_ids: List[str]) -> pd.DataFrame:
    meta_path = os.path.join(merged_dir, "subj_meta.csv")
    if not os.path.exists(meta_path):
        parent_meta = os.path.join(os.path.dirname(merged_dir), "subj_meta.csv")
        if os.path.exists(parent_meta):
            meta_path = parent_meta
        else:
            drug_csv = os.path.join(project_root, "MRI/source_data/behav/drug_order.csv")
            if not os.path.exists(drug_csv):
                raise FileNotFoundError(f"Missing subj_meta.csv and {drug_csv}")
            meta_df = pd.read_csv(drug_csv)
            meta_df["subject_id"] = meta_df["subject_id"].astype(str)
            keep = meta_df["subject_id"].isin(subj_ids) | meta_df["subject_id"].isin([f"sub-{s}" for s in subj_ids])
            meta_df = meta_df.loc[keep, ["subject_id", "Group", "Drug"]].copy()
            meta_df = meta_df.rename(columns={"subject_id": "Subject"})
            return meta_df
    return pd.read_csv(meta_path)


def load_subject_maps(
    merged_dir: str,
    cond_list: List[str],
    mask: np.ndarray,
    project_root: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, SubjectInfo]]:
    subj_files = sorted(glob.glob(os.path.join(merged_dir, f"subjmap_{cond_list[0]}_*.nii.gz")))
    subj_ids = [os.path.basename(p).split("_")[-1].replace(".nii.gz", "") for p in subj_files]
    meta_df = load_subject_meta(merged_dir, project_root, subj_ids)
    subj_scores: Dict[str, Dict[str, np.ndarray]] = {}
    subj_info: Dict[str, SubjectInfo] = {}
    for row in meta_df.itertuples(index=False):
        s_id = str(row.Subject).replace("sub-", "").strip()
        subj_scores[s_id] = {}
        for cond in cond_list:
            map_path = os.path.join(merged_dir, f"subjmap_{cond}_{s_id}.nii.gz")
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"Missing subject map: {map_path}")
            data = nib.load(map_path).get_fdata()
            subj_scores[s_id][cond] = data[mask]
        subj_info[s_id] = SubjectInfo(group=row.Group, drug=row.Drug)
    return subj_scores, subj_info


def write_fsl_mat(path: str, design: np.ndarray) -> None:
    npts, nwaves = design.shape
    with open(path, "w") as f:
        f.write("/NumWaves\t%d\n" % nwaves)
        f.write("/NumPoints\t%d\n" % npts)
        f.write("/PPheights\t" + "\t".join(["1"] * nwaves) + "\n")
        f.write("/Matrix\n")
        for row in design:
            f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")


def write_fsl_con(path: str, cons: np.ndarray, names: List[str]) -> None:
    nwaves = cons.shape[1]
    ncon = cons.shape[0]
    with open(path, "w") as f:
        f.write("/NumWaves\t%d\n" % nwaves)
        f.write("/NumContrasts\t%d\n" % ncon)
        for i, name in enumerate(names, 1):
            f.write(f"/ContrastName{i}\t{name}\n")
        f.write("/Matrix\n")
        for row in cons:
            f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")


def write_fsl_grp(path: str, npts: int) -> None:
    with open(path, "w") as f:
        f.write("/NumWaves\t1\n")
        f.write("/NumPoints\t%d\n" % npts)
        f.write("/Matrix\n")
        for _ in range(npts):
            f.write("1\n")


def build_valid_mask(values: np.ndarray, mask: np.ndarray, ref_img: nib.Nifti1Image) -> str:
    finite_mask = np.all(np.isfinite(values), axis=0)
    var_mask = np.nanvar(values, axis=0) > 0
    valid = finite_mask & var_mask
    vol = np.zeros(mask.shape, dtype=np.uint8)
    vol[mask] = valid.astype(np.uint8)
    out_path = os.path.join(os.getcwd(), "analysis_valid_mask.nii.gz")
    nib.save(nib.Nifti1Image(vol, ref_img.affine), out_path)
    return out_path


def save_4d(path: str, data: np.ndarray, mask: np.ndarray, ref_img: nib.Nifti1Image) -> None:
    vol = np.zeros(mask.shape + (data.shape[0],), dtype=np.float32)
    for i in range(data.shape[0]):
        vol[..., i][mask] = np.nan_to_num(data[i], nan=0.0)
    nib.save(nib.Nifti1Image(vol, ref_img.affine), path)


def run_flameo(out_dir: str, cope_4d: str, var_4d: str, mask: str, mat: str, con: str, grp: str) -> None:
    cmd = [
        "flameo",
        "--cope", cope_4d,
        "--varcope", var_4d,
        "--mask", mask,
        "--dm", mat,
        "--tc", con,
        "--cs", grp,
        "--runmode=flame1",
        "--ld", out_dir,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="FLAMEO group inference for extinction searchlight")
    parser.add_argument("--project_root", default="/gscratch/fang/NARSAD")
    parser.add_argument("--merged_dir", default="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight/ext/merged")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--task", default="phase2")
    parser.add_argument("--glasser_atlas", default="/gscratch/fang/NARSAD/ROI/Glasser/HCP-MMP1_2mm.nii")
    parser.add_argument("--tian_atlas", default="/gscratch/fang/NARSAD/ROI/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz")
    parser.add_argument("--reference_lss", default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.merged_dir
    if args.reference_lss is None:
        args.reference_lss = find_reference_lss(args.project_root, args.task)

    mask, mask_img = build_master_mask_from_reference(
        args.glasser_atlas, args.tian_atlas, args.reference_lss
    )
    subj_scores, subj_info = load_subject_maps(args.merged_dir, CS_LABELS, mask, args.project_root)
    all_subs = sorted(subj_scores.keys())

    def select_subjects(group: str | None = None, drug: str | None = None) -> List[str]:
        selected = []
        for s_id in all_subs:
            info = subj_info[s_id]
            if group is not None and info.group != group:
                continue
            if drug is not None and info.drug != drug:
                continue
            selected.append(s_id)
        return selected

    work_base = os.path.join(args.out_dir, "flameo")
    os.makedirs(work_base, exist_ok=True)

    for cond in CS_LABELS:
        # 1) Placebo within-group (SAD/HC)
        for group in ["SAD", "HC"]:
            subs = select_subjects(group=group, drug="Placebo")
            if len(subs) < 3:
                continue
            values = np.stack([subj_scores[s][cond] for s in subs], axis=0)
            work = os.path.join(work_base, f"within_{cond}_{group}_PLC")
            os.makedirs(work, exist_ok=True)
            mask_path = build_valid_mask(values, mask, mask_img)
            cope_4d = os.path.join(work, "cope.nii.gz")
            var_4d = os.path.join(work, "varcope.nii.gz")
            save_4d(cope_4d, values, mask, mask_img)
            save_4d(var_4d, np.ones_like(values), mask, mask_img)
            mat = os.path.join(work, "design.mat")
            con = os.path.join(work, "design.con")
            grp = os.path.join(work, "design.grp")
            write_fsl_mat(mat, np.ones((len(subs), 1)))
            write_fsl_con(con, np.array([[1.0]]), ["mean"])
            write_fsl_grp(grp, len(subs))
            run_flameo(os.path.join(work, "flameo_out"), cope_4d, var_4d, mask_path, mat, con, grp)

        # 2) Placebo group differences (SAD vs HC)
        sad = select_subjects(group="SAD", drug="Placebo")
        hc = select_subjects(group="HC", drug="Placebo")
        if len(sad) >= 3 and len(hc) >= 3:
            subs = sad + hc
            values = np.stack([subj_scores[s][cond] for s in subs], axis=0)
            work = os.path.join(work_base, f"diff_{cond}_SAD-HC_PLC")
            os.makedirs(work, exist_ok=True)
            mask_path = build_valid_mask(values, mask, mask_img)
            cope_4d = os.path.join(work, "cope.nii.gz")
            var_4d = os.path.join(work, "varcope.nii.gz")
            save_4d(cope_4d, values, mask, mask_img)
            save_4d(var_4d, np.ones_like(values), mask, mask_img)
            mat = os.path.join(work, "design.mat")
            con = os.path.join(work, "design.con")
            grp = os.path.join(work, "design.grp")
            design = np.zeros((len(subs), 2))
            design[: len(sad), 0] = 1.0
            design[len(sad) :, 1] = 1.0
            write_fsl_mat(mat, design)
            write_fsl_con(con, np.array([[1.0, -1.0]]), ["SAD>HC"])
            write_fsl_grp(grp, len(subs))
            run_flameo(os.path.join(work, "flameo_out"), cope_4d, var_4d, mask_path, mat, con, grp)

        # 3) OXT-PLC modulation within group
        for group in ["SAD", "HC"]:
            oxt = select_subjects(group=group, drug="Oxytocin")
            plc = select_subjects(group=group, drug="Placebo")
            if len(oxt) < 3 or len(plc) < 3:
                continue
            subs = oxt + plc
            values = np.stack([subj_scores[s][cond] for s in subs], axis=0)
            work = os.path.join(work_base, f"mod_{cond}_{group}_OXT-PLC")
            os.makedirs(work, exist_ok=True)
            mask_path = build_valid_mask(values, mask, mask_img)
            cope_4d = os.path.join(work, "cope.nii.gz")
            var_4d = os.path.join(work, "varcope.nii.gz")
            save_4d(cope_4d, values, mask, mask_img)
            save_4d(var_4d, np.ones_like(values), mask, mask_img)
            mat = os.path.join(work, "design.mat")
            con = os.path.join(work, "design.con")
            grp = os.path.join(work, "design.grp")
            design = np.zeros((len(subs), 2))
            design[: len(oxt), 0] = 1.0
            design[len(oxt) :, 1] = 1.0
            write_fsl_mat(mat, design)
            write_fsl_con(con, np.array([[1.0, -1.0]]), ["OXT>PLC"])
            write_fsl_grp(grp, len(subs))
            run_flameo(os.path.join(work, "flameo_out"), cope_4d, var_4d, mask_path, mat, con, grp)


if __name__ == "__main__":
    main()
