#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from analyze_results import align_and_stack, load_npz_by_run_envs, plot_mean_std, summarize_single_run


DEFAULT_METRICS = [
    "reward",
    "viol_rate_roll",
    "viol_rate_cum",
    "resources",
    "allocated_prbs_sum",
]

METRIC_TITLES = {
    "reward": "Reward",
    "violation": "Violations (per-step count)",
    "viol_rate_roll": "Violation Rate (rolling)",
    "viol_rate_cum": "Violation Rate (cumulative)",
    "resources": "Resources",
    "allocated_prbs_sum": "Allocated PRBs (sum)",
    "prio_weighted": "Priority-weighted Violations",
}


# =========================
# Editable config
# =========================
ROOT_DIR = "./results/ablations"
SCENARIO_ID = 1
VARIANTS = ["full", "no_pred", "no_cross", "raw_reward"]
METRICS = DEFAULT_METRICS
SMOOTH_WINDOW = 2000
SMOOTH_RESOURCES = True
FIGSIZE = (12.0, 14.0)
DPI = 150
OUTPUT_PATH = None


def _discover_runs(root: str, scenario: int, variant: str):
    base = os.path.join(root, f"scenario_{scenario}", variant)
    runs = set()

    for path in glob.glob(os.path.join(base, "config_run*.json")):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("config_run"):
            try:
                runs.add(int(stem.replace("config_run", "")))
            except ValueError:
                pass

    for path in glob.glob(os.path.join(base, "history_run*_env*.npz")):
        stem = os.path.splitext(os.path.basename(path))[0]
        match = re.match(r"history_run(\d+)_env\d+$", stem)
        if match:
            runs.add(int(match.group(1)))

    return sorted(runs)


def collect_variant_series(root: str, scenario: int, variant: str):
    run_series = []
    loaded_runs = []

    for run in _discover_runs(root, scenario, variant):
        data_list, _ = load_npz_by_run_envs(root, scenario, variant, run)
        if not data_list:
            continue

        series_envs = []
        for data in data_list:
            _, ts = summarize_single_run(data, aux_csv=None)
            if ts:
                series_envs.append(ts)

        if not series_envs:
            continue

        aligned_envs = align_and_stack(series_envs)
        if not aligned_envs:
            continue

        run_series.append({k: np.mean(v, axis=1) for k, v in aligned_envs.items()})
        loaded_runs.append(run)

    return run_series, loaded_runs


def _metric_smooth(metric: str, smooth_window: int, smooth_resources: bool):
    if metric in {"viol_rate_roll", "viol_rate_cum"}:
        return None
    if metric in {"resources", "allocated_prbs_sum"} and not smooth_resources:
        return None
    return smooth_window if smooth_window and smooth_window > 1 else None


def _variant_label(variant: str, runs):
    if not runs:
        return f"{variant} (0 runs)"
    return f"{variant} ({len(runs)} runs: {','.join(str(r) for r in runs)})"


def main():
    metrics = [m for m in METRICS if m]
    if not metrics:
        raise SystemExit("No metrics requested.")

    variant_payloads = []
    for variant in VARIANTS:
        series_runs, loaded_runs = collect_variant_series(ROOT_DIR, SCENARIO_ID, variant)
        if not series_runs:
            print(f"[skip] variant={variant}: no completed run history found")
            continue
        aligned_runs = align_and_stack(series_runs)
        if not aligned_runs:
            print(f"[skip] variant={variant}: no alignable series found")
            continue
        variant_payloads.append((variant, loaded_runs, aligned_runs))
        print(f"[load] variant={variant}: runs={loaded_runs}")

    if not variant_payloads:
        raise SystemExit("No variants with usable history files were found.")

    fig, axs = plt.subplots(len(metrics), 1, figsize=FIGSIZE, constrained_layout=True)
    if len(metrics) == 1:
        axs = [axs]

    for ax, metric in zip(axs, metrics):
        plotted = False
        for variant, loaded_runs, aligned_runs in variant_payloads:
            if metric not in aligned_runs:
                continue
            smooth_k = _metric_smooth(metric, SMOOTH_WINDOW, SMOOTH_RESOURCES)
            plot_mean_std(ax, aligned_runs[metric], label=_variant_label(variant, loaded_runs), smooth_k=smooth_k)
            plotted = True

        title = METRIC_TITLES.get(metric, metric)
        ax.set_title(f"Scenario {SCENARIO_ID} - {title}")
        ax.set_xlabel("steps")
        ax.grid(True)
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"No data for metric: {metric}", ha="center", va="center", transform=ax.transAxes)

    out_dir = os.path.join(ROOT_DIR, f"scenario_{SCENARIO_ID}")
    os.makedirs(out_dir, exist_ok=True)

    if OUTPUT_PATH:
        out_path = OUTPUT_PATH
    else:
        suffix = "_".join(VARIANTS)
        out_path = os.path.join(out_dir, f"ablation_curves_{suffix}.png")

    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
