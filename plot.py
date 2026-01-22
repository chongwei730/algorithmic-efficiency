#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

# BASE_DIR = "/scratch.global/chen8596/mlcommons_experiments_test/"
BASE_DIR = "./test"
TRIAL_GLOB = "trial_*"

# Fallback columns
TRAIN_LOSS_COL = "loss"
LR_COL = "lr"

from scoring import scoring_utils

# Helper to pick validation metric for a workload (e.g. validation/accuracy or validation/mean_average_precision)
def get_validation_metric_for_workload(workload_dir_name):
    # scoring_utils expects a workload identifier with framework suffix (e.g., cifar_pytorch)
    try:
        metric, target = scoring_utils.get_workload_metrics_and_targets(workload_dir_name, split='validation')
        return metric
    except Exception:
        # Fallback to validation/accuracy if unknown workload
        return 'validation/accuracy'

# ============================================================
# UTILS
# ============================================================

def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_hparams(trial_dir):
    hpath = os.path.join(trial_dir, "hparams.json")
    if not os.path.exists(hpath):
        return {}
    with open(hpath, "r") as f:
        return json.load(f)


def make_label(trial_dir):
    """
    trial name + c1 + accum_steps + interval
    """
    hparams = load_hparams(trial_dir)
    c1 = hparams.get("c1", "NA")
    accum = hparams.get("accum_steps", "NA")
    interval = hparams.get("interval", "NA")
    return f"{os.path.basename(trial_dir)} | c1={c1} | accum={accum} | int={interval}"

# ============================================================
# MAIN
# ============================================================

algorithms = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
)

print("[INFO] algorithms:", algorithms)

for algo in algorithms:
    algo_root = os.path.join(BASE_DIR, algo)
    # iterate workloads under this algorithm
    workloads = sorted(
        d for d in os.listdir(algo_root)
        if os.path.isdir(os.path.join(algo_root, d))
    )

    if not workloads:
        continue

    for workload in workloads:
        workload_root = os.path.join(algo_root, workload)
        trials = sorted(glob.glob(os.path.join(workload_root, TRIAL_GLOB)))
        if len(trials) == 0:
            continue

        print(f"[INFO] {algo}/{workload}: {len(trials)} trials")

        # determine validation metric for this workload
        val_metric = get_validation_metric_for_workload(workload)

        # ------------------------
        # TRAIN LOSS
        # ------------------------
        plt.figure()
        curves = 0

        for t in trials:
            # prefer measurements.csv for training logs
            df = load_csv(os.path.join(t, "measurements.csv"))
            if df is None:
                df = load_csv(os.path.join(t, "eval_measurements.csv"))
            if df is None or TRAIN_LOSS_COL not in df.columns:
                continue

            mask = df[TRAIN_LOSS_COL].notna()
            if mask.sum() == 0:
                continue

            plt.plot(
                df.loc[mask, "global_step"],
                df.loc[mask, TRAIN_LOSS_COL],
                alpha=0.7,
                label=make_label(t),
            )
            curves += 1

        if curves > 0:
            plt.xlabel("global_step")
            plt.ylabel("train loss")
            plt.title(f"{algo}/{workload} – Training Loss")
            plt.grid(True)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(algo_root, f"{workload}_train_loss.png"), dpi=200)
        plt.close()

        # ------------------------
        # VALIDATION METRIC (varies by workload)
        # ------------------------
        plt.figure()
        curves = 0

        for t in trials:
            df = load_csv(os.path.join(t, "eval_measurements.csv"))
            # fallback to measurements.csv
            if df is None:
                df = load_csv(os.path.join(t, "measurements.csv"))
            if df is None or val_metric not in df.columns:
                continue

            mask = df[val_metric].notna()
            if mask.sum() == 0:
                continue

            plt.plot(
                df.loc[mask, "global_step"],
                df.loc[mask, val_metric],
                alpha=0.7,
                label=make_label(t),
            )
            curves += 1

        if curves > 0:
            plt.xlabel("global_step")
            plt.ylabel(val_metric)
            plt.title(f"{algo}/{workload} – {val_metric}")
            plt.grid(True)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(algo_root, f"{workload}_val_metric.png"), dpi=200)
        plt.close()

        # ------------------------
        # LEARNING RATE
        # ------------------------
        plt.figure()
        curves = 0

        for t in trials:
            df = load_csv(os.path.join(t, "measurements.csv"))
            if df is None or LR_COL not in df.columns:
                continue

            mask = df[LR_COL].notna()
            if mask.sum() == 0:
                continue

            plt.plot(
                df.loc[mask, "global_step"],
                df.loc[mask, LR_COL],
                alpha=0.7,
                label=make_label(t),
            )
            curves += 1

        if curves > 0:
            plt.xlabel("global_step")
            plt.ylabel("learning rate")
            plt.title(f"{algo}/{workload} – Learning Rate")
            plt.grid(True)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(algo_root, f"{workload}_learning_rate.png"), dpi=200)
        plt.close()

        print(f"[OK] saved plots for {algo}/{workload}")

print("[DONE] all algorithms processed")
