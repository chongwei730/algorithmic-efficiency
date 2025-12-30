#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd


BASE_DIR = "/scratch.global/chen8596/mlcommons_experiments"

WORKLOADS = [
    "cifar",
    # "imagenet_resnet",
    # "imagenet_vit",
    # "librispeech_conformer",
]


VALID_THRESHOLD = {
    "cifar": 0.85,
    # "imagenet_resnet": 0.759,
    # "imagenet_vit": 0.78,
    # "librispeech_conformer": 0.085,
}



def trial_time_to_success(measurements_path: str, workload: str):
    df = pd.read_csv(measurements_path)


    is_eval = False
    if "validation/accuracy" in df.columns:
        is_eval = df["validation/accuracy"].notna()
    if "test/accuracy" in df.columns:
        is_eval = is_eval | df["test/accuracy"].notna()

    eval_df = df[is_eval]
    if len(eval_df) == 0:
        return False, None


    row = eval_df.iloc[-1]


    time_used = row.get("accumulated_submission_time", None)
    if time_used is None or pd.isna(time_used):

        time_used = row.get("total_duration", None)

    if time_used is None or pd.isna(time_used):
        return False, None


    target = VALID_THRESHOLD.get(workload, None)
    if target is None:
        return False, float(time_used)

    val_acc = row.get("validation/accuracy", None)
    if val_acc is None or pd.isna(val_acc):
        return False, float(time_used)

    success = float(val_acc) >= float(target)
    return success, float(time_used)


def score_by_sequential_trials(trial_measurement_paths, workload: str):

    total_time = 0.0

    for m_path in trial_measurement_paths:
        success, t = trial_time_to_success(m_path, workload)
        if t is None:
            continue

        total_time += t
        if success:
            return total_time

    return None




all_results = []

for workload in WORKLOADS:
    workload_dir = os.path.join(BASE_DIR, workload)
    if not os.path.isdir(workload_dir):
        print(f"[WARN] workload dir not found: {workload_dir}")
        continue

    rows = []

    for method in sorted(os.listdir(workload_dir)):
        method_dir = os.path.join(workload_dir, method)
        if not os.path.isdir(method_dir):
            continue

        trial_root = os.path.join(method_dir, f"{workload}_pytorch")
        if not os.path.isdir(trial_root):
            continue

     
        trial_paths = []
        for trial in sorted(os.listdir(trial_root)):
            trial_dir = os.path.join(trial_root, trial)
            if not os.path.isdir(trial_dir):
                continue

            m_path = os.path.join(trial_dir, "measurements.csv")
            if os.path.isfile(m_path):
                trial_paths.append(m_path)

        if len(trial_paths) == 0:
            continue

        score = score_by_sequential_trials(trial_paths, workload)
        if score is None:
            continue 

        rows.append(
            {
                "workload": workload,
                "method": method,
                "time_cost": score,
                "num_trials_used": len(trial_paths),
            }
        )

    if len(rows) == 0:
        print(f"[WARN] no successful submissions for workload={workload}")
        continue

    df = pd.DataFrame(rows).sort_values("time_cost", ascending=True)

    out_path = f"{workload}_time_cost.csv"
    df.to_csv(out_path, index=False)

    print(f"\n===== {workload} (OFFICIAL SCORE) =====\n")
    print(df.to_string(index=False))

    all_results.append(df)


if len(all_results) > 0:
    summary_df = pd.concat(all_results, ignore_index=True)
    summary_df = summary_df.sort_values("time_cost", ascending=True)
    summary_df.to_csv("summary_time_cost.csv", index=False)

    print("\n===== SUMMARY (ALL WORKLOADS) =====\n")
    print(summary_df.to_string(index=False))
