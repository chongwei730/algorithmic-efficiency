#!/usr/bin/env python3
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _find_best_train_col(df):
    cols = list(df.columns)
    cols_low = [c.lower() for c in cols]
    for i, c in enumerate(cols_low):
        if 'best' in c and 'train' in c and 'loss' in c:
            return cols[i]
    # fallback: try combinations
    for i, c in enumerate(cols_low):
        if 'train' in c and 'loss' in c:
            return cols[i]
    raise KeyError('Could not find best train loss column in dataframe. Columns: ' + ','.join(cols))


def _ensure_workload_trial_cols(df):
    cols = list(df.columns)
    # workload
    if 'workload' not in cols:
        for c in cols:
            if 'workload' in c.lower():
                df = df.rename(columns={c: 'workload'})
                break
    # trial
    if 'trial' not in df.columns:
        for c in cols:
            if 'trial' in c.lower():
                df = df.rename(columns={c: 'trial'})
                break
    return df


def read_summary(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = _ensure_workload_trial_cols(df)
    best_col = _find_best_train_col(df)
    return df, best_col


def make_plots(line_csv, cyclic_csv, out_dir='scoring_results/plots'):
    os.makedirs(out_dir, exist_ok=True)

    line_df, line_best_col = read_summary(line_csv)
    cyclic_df, cyclic_best_col = read_summary(cyclic_csv)

    # normalize trial strings
    line_df['trial'] = line_df['trial'].astype(str)
    cyclic_df['trial'] = cyclic_df['trial'].astype(str)

    workloads = sorted(set(line_df['workload'].unique()).union(set(cyclic_df['workload'].unique())))

    summary_rows = []

    for wl in workloads:
        ldf = line_df[line_df['workload'] == wl]
        cdf = cyclic_df[cyclic_df['workload'] == wl]
        trials = sorted(set(ldf['trial'].unique()).union(set(cdf['trial'].unique())))
        if not trials:
            continue

        trials_sorted = sorted(trials, key=lambda s: int(''.join([c for c in s if c.isdigit()]) or -1))

        line_vals = []
        cyclic_vals = []
        labels = []
        for t in trials_sorted:
            labels.append(t)
            lv = ldf.loc[ldf['trial'] == t, line_best_col]
            cv = cdf.loc[cdf['trial'] == t, cyclic_best_col]
            lvv = float(lv.values[0]) if not lv.empty else float('nan')
            cvv = float(cv.values[0]) if not cv.empty else float('nan')
            line_vals.append(lvv)
            cyclic_vals.append(cvv)

            summary_rows.append({
                'workload': wl,
                'trial': t,
                'line_best_train_loss': lvv,
                'cyclic_best_train_loss': cvv,
            })

        # plot side-by-side bars
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
        ax.bar(x - width/2, line_vals, width, label='line_search')
        ax.bar(x + width/2, cyclic_vals, width, label='cyclic')
        ax.set_ylabel('best train loss')
        ax.set_xlabel('trial')
        ax.set_title(f'{wl} — Best Train Loss per Trial (line vs cyclic)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y')
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{wl}_best_train_loss_comparison.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    # write combined csv
    summary_df = pd.DataFrame(summary_rows)
    summary_out = os.path.join(out_dir, 'line_vs_cyclic_best_train_loss_summary.csv')
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved summary CSV: {summary_out}")


if __name__ == '__main__':
    # defaults (can be overridden via CLI)
    line_csv = sys.argv[1] if len(sys.argv) > 1 else './scoring_results/Line_Search_AdamW_summary.csv'
    cyclic_csv = sys.argv[2] if len(sys.argv) > 2 else './scoring_results/external_cyclic_lr_summary.csv'
    make_plots(line_csv, cyclic_csv)
