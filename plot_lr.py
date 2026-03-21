#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
import copy
import algoperf.workloads.workloads as workloads_registry

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/scratch.global/chen8596/mlcommons_experiments/"
TRIAL_GLOB = "trial_*"

TRAIN_LOSS_COL = "train/loss"
LR_COL = "lr"

from scoring import scoring_utils

# ============================================================
# METRIC HELPERS
# ============================================================

def get_validation_metric_for_workload(workload_dir_name):
    try:
        metric, _ = scoring_utils.get_workload_metrics_and_targets(
            workload_dir_name, split='validation'
        )
        return metric
    except Exception:
        return 'validation/accuracy'


def get_step_hint_for_workload(workload_dir_name):
    """Return the workload `step_hint` if available, else None."""
    try:
        m = re.match(scoring_utils.WORKLOAD_NAME_PATTERN, workload_dir_name)
        if not m:
            return None
        workload_name = m.group(1)
        framework = m.group(2)
        workload_metadata = copy.copy(scoring_utils.WORKLOADS[workload_name])
        workload_metadata['workload_path'] = os.path.join(
            scoring_utils.BASE_WORKLOADS_DIR,
            workload_metadata['workload_path'] + f'{framework}',
            'workload.py',
        )
        workload_obj = workloads_registry.import_workload(
            workload_path=workload_metadata['workload_path'],
            workload_class_name=workload_metadata['workload_class_name'],
        )
        return int(workload_obj.step_hint)
    except Exception:
        return None


def get_training_metric_for_workload(workload_dir_name):
    try:
        metric, _ = scoring_utils.get_workload_metrics_and_targets(
            workload_dir_name, split='train'
        )
        return metric
    except Exception:
        return 'train/accuracy'

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


def compute_cyclic_lr(lr_max, cycles, warmup_factor, max_step, lr_min=1e-10):
    if max_step <= 0:
        return [], []

    T = max_step
    warmup_steps = int(warmup_factor * T)

    def warmup(t, phi, alpha):
        return lr_min + (lr_max - lr_min) / max(1, warmup_steps) * (
            t - phi * alpha * T
        )

    def cosine_decay(t, phi, alpha):
        denom = alpha * T - warmup_steps
        if denom <= 0:
            return lr_max
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos((t - warmup_steps - phi * alpha * T) * math.pi / denom)
        )

    steps = list(range(max_step + 1))
    lrs = []
    alpha = 1.0 / max(1, cycles)

    for t in steps:
        lr_t = lr_min
        for phi in range(cycles):
            start_warm = phi * alpha * T + warmup_steps
            end_cycle = (phi + 1) * alpha * T

            if t <= start_warm:
                lr_t = warmup(t, phi, alpha)
                break
            elif t <= end_cycle:
                lr_t = cosine_decay(t, phi, alpha)
                break

        lrs.append(lr_t)

    return steps, lrs


def make_label(trial_dir):
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
    if "cyclic" not in algo: 
        continue
    algo_root = os.path.join(BASE_DIR, algo)

    workloads = sorted(
        d for d in os.listdir(algo_root)
        if os.path.isdir(os.path.join(algo_root, d))
    )

    if not workloads:
        continue

    for workload in workloads:
        workload_root = os.path.join(algo_root, workload)
        trials = sorted(glob.glob(os.path.join(workload_root, TRIAL_GLOB)))
        print(workload)
        # if "cyclic" not in workload:
        #     continue



        # val_metric = get_validation_metric_for_workload(workload)
        # train_metric = get_training_metric_for_workload(workload)


        # ===================== LEARNING RATE =====================
        plt.figure()
        curves = 0

        for t in trials:
            df = load_csv(os.path.join(t, "measurements.csv"))
            if df is None:
                continue

            trial_name = os.path.basename(t).lower()
            hparams = load_hparams(t)
            print(trial_name)

            try:
                if (
                   all(k in hparams for k in ("learning_rate", "cycles", "warmup_factor"))
                ):
                    # prefer workload step_hint when available
                    step_hint = get_step_hint_for_workload(workload)
                    if step_hint is not None and step_hint > 0:
                        max_step = int(step_hint)
                    else:
                        max_step = int(df["global_step"].max())
                    steps, lrs = compute_cyclic_lr(
                        lr_max=float(hparams["learning_rate"]),
                        cycles=int(hparams["cycles"]),
                        warmup_factor=float(hparams["warmup_factor"]),
                        max_step=max_step,
                    )

                    if steps:
                        plt.plot(
                            steps,
                            lrs,
                            alpha=0.9,
                            label=make_label(t) + " (cyclic)",
                        )
                        curves += 1
                        continue
            except Exception:
                pass
            # Only plot cyclic (cosine) LR curves; skip raw LR traces.
            # If the trial wasn't cyclic, skip plotting here.
            # if "cyclic" not in trial_name:
            #     # optional: print debug info for non-cyclic trials
            #     print(f"[INFO] skipping non-cyclic LR plot for trial {t}")
            #     continue
        save_path = os.path.join(algo_root, f"{workload}_learning_rate.png")
        if curves > 0:
            plt.xlabel("global_step")
            plt.ylabel("learning rate")
            plt.ylim(1e-8, 1e-2)
            plt.title(f"{algo}/{workload} – Learning Rate")
            plt.grid(True)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close()

            print(f"[OK] saved plot: {save_path}")

print("[DONE]")