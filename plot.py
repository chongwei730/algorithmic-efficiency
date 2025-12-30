#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/scratch.global/chen8596/mlcommons_experiments/cifar"
WORKLOAD_DIR = "cifar_pytorch"
TRIAL_GLOB = "trial_*"

TRAIN_LOSS_COL = "train/loss"
VAL_ACC_COL = "validation/accuracy"
LR_COL = "lr"

# ============================================================
def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)



algorithms = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
)

print("[INFO] algorithms:", algorithms)


for algo in algorithms:
    algo_root = os.path.join(BASE_DIR, algo)
    workload_root = os.path.join(algo_root, WORKLOAD_DIR)

    if not os.path.isdir(workload_root):
        continue

    trials = sorted(glob.glob(os.path.join(workload_root, TRIAL_GLOB)))
    if len(trials) == 0:
        continue

    print(f"[INFO] {algo}: {len(trials)} trials")

    plt.figure()
    curves = 0

    for t in trials:
        csv_path = os.path.join(t, "measurements.csv")
        df = load_csv(csv_path)
        if df is None or TRAIN_LOSS_COL not in df.columns:
            continue

        mask = df[TRAIN_LOSS_COL].notna()
        if mask.sum() == 0:
            continue

        plt.plot(
            df.loc[mask, "global_step"],
            df.loc[mask, TRAIN_LOSS_COL],
            alpha=0.7,
            label=os.path.basename(t),
        )
        curves += 1

    if curves > 0:
        plt.xlabel("global_step")
        plt.ylabel("train loss")
        plt.title(f"{algo} Training Loss")
        plt.grid(True)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(algo_root, "train_loss.png"), dpi=200)
    plt.close()

    plt.figure()
    curves = 0

    for t in trials:
        csv_path = os.path.join(t, "measurements.csv")
        df = load_csv(csv_path)
        if df is None or VAL_ACC_COL not in df.columns:
            continue

        mask = df[VAL_ACC_COL].notna()
        if mask.sum() == 0:
            continue

        plt.plot(
            df.loc[mask, "global_step"],
            df.loc[mask, VAL_ACC_COL],
            alpha=0.7,
            label=os.path.basename(t),
        )
        curves += 1

    if curves > 0:
        plt.xlabel("global_step")
        plt.ylabel("validation accuracy")
        plt.title(f"{algo} – Validation Accuracy")
        plt.grid(True)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(algo_root, "val_accuracy.png"), dpi=200)
    plt.close()

    plt.figure()
    curves = 0

    for t in trials:
        csv_path = os.path.join(t, "measurements.csv")
        df = load_csv(csv_path)
        if df is None or LR_COL not in df.columns:
            continue

        mask = df[LR_COL].notna()
        if mask.sum() == 0:
            continue

        plt.plot(
            df.loc[mask, "global_step"],
            df.loc[mask, LR_COL],
            alpha=0.7,
            label=os.path.basename(t),
        )
        curves += 1

    if curves > 0:
        plt.xlabel("global_step")
        plt.ylabel("learning rate")
        plt.title(f"{algo} Learning Rate")
        plt.grid(True)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(algo_root, "learning_rate.png"), dpi=200)
    plt.close()

    print(f"[OK] saved plots for {algo}")

print("[DONE] all algorithms processed")
