#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import glob
import json
import os

import numpy as np

from analyze_results import load_npz_by_run_envs, summarize_single_run, merge_metrics


MAIN_COLS = [
    "reward_eval_mean",
    "viol_rate_eval",
    "sla_rate_eval",
    "res_eval_mean",
]


# =========================
# Editable config
# =========================
ROOT_DIR = "./results/ablations"
SCENARIO_ID = 1
VARIANTS = ["full", "no_pred", "no_cross", "raw_reward"]


def _safe_mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def collect_variant_metrics(root: str, scenario: int, variant: str):
    base = os.path.join(root, f"scenario_{scenario}", variant)
    cfg_paths = sorted(glob.glob(os.path.join(base, "config_run*.json")))
    runs = []
    for p in cfg_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        run = int(stem.replace("config_run", ""))
        runs.append(run)
    runs = sorted(set(runs))

    run_metrics = []
    for run in runs:
        data_list, _ = load_npz_by_run_envs(root, scenario, variant, run)
        if not data_list:
            continue
        metrics_envs = []
        for data in data_list:
            m, _ = summarize_single_run(data, aux_csv=None)
            metrics_envs.append(m)
        run_metrics.append(merge_metrics(metrics_envs))
    return run_metrics


def main():
    out_rows = []
    json_out = {}
    for variant in VARIANTS:
        run_metrics = collect_variant_metrics(ROOT_DIR, SCENARIO_ID, variant)
        row = {"variant": variant, "runs": len(run_metrics)}
        json_out[variant] = {"runs": len(run_metrics), "per_run": run_metrics}
        for col in MAIN_COLS:
            mean, std = _safe_mean_std([m.get(col, np.nan) for m in run_metrics])
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            json_out[variant][col] = {"mean": mean, "std": std}
        out_rows.append(row)

    out_dir = os.path.join(ROOT_DIR, f"scenario_{SCENARIO_ID}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    json_path = os.path.join(out_dir, "ablation_summary.json")
    md_path = os.path.join(out_dir, "ablation_summary.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["variant"])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| variant | runs | reward_eval_mean | viol_rate_eval | sla_rate_eval | res_eval_mean |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in out_rows:
            f.write(
                f"| {row['variant']} | {row['runs']} | "
                f"{row['reward_eval_mean_mean']:.4f} +/- {row['reward_eval_mean_std']:.4f} | "
                f"{row['viol_rate_eval_mean']:.4f} +/- {row['viol_rate_eval_std']:.4f} | "
                f"{row['sla_rate_eval_mean']:.4f} +/- {row['sla_rate_eval_std']:.4f} | "
                f"{row['res_eval_mean_mean']:.4f} +/- {row['res_eval_mean_std']:.4f} |\n"
            )

    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
