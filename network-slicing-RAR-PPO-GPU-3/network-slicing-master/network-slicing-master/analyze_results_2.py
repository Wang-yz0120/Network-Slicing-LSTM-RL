#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from datetime import datetime

# ========================= CONFIG =========================
ROOT_RESULTS_DIR = "./results"     # 结果根目录
OUT_DIR          = "./results/figs"  # 图保存目录（仅在 SAVE_FIG=True 时用）

SCENARIOS = [1]                 # 例如 [0,1,2]；None 表示自动发现
ALGOS     = ["PPO", "RaRPPO"]   # 例如 ["PPO", "RaRPPO"]；None 自动发现
RUNS      = [0]                 # 例如 [0,1,2,3]；None 自动发现

SMOOTH_WINDOW = 50              # reward/resources/violation/prio 曲线平滑窗口
ROLL_RATE_WIN = 200             # 滚动违规率窗口；None/0 关闭
SAVE_FIG      = True            # 只想看表格就设 False
SHOW_BANDS    = True            # 多 run 画均值±std 阴影带；单 run 画单条
REPRESENTATIVE_RUN_FOR_CURVE = None  # SHOW_BANDS=False 时选哪条 run
FIGSIZE_BASE = (11, 21)         # 不拆分优先级时的图尺寸
# —— 新增：优先级子图显示方式 —— #
PRIO_PANEL_MODE = "split"       # "split"=拆成4张；"single"=一张合并
PRIO_SMOOTH_WINDOW = 100        # 优先级子图可单独平滑
PRIO_YLIM_PERCENTILE = (1, 99)  # 用分位数裁剪 y 轴以弱化极端点
# =========================================================

def ensure_dir(path): os.makedirs(path, exist_ok=True)

def rolling_mean(x, k):
    if not k or k <= 1: return x
    k = int(min(k, len(x)))
    if k <= 1: return x
    w = np.ones(k, dtype=float) / k
    return np.convolve(x, w, mode='valid')

def find_scenarios(root):
    out = []
    for p in glob.glob(os.path.join(root, "scenario_*")):
        try: out.append(int(os.path.basename(p).split("_")[1]))
        except: pass
    return sorted(out)

def find_algos(root, scenario):
    base = os.path.join(root, f"scenario_{scenario}")
    if not os.path.isdir(base): return []
    return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

def find_runs(root, scenario, algo):
    base = os.path.join(root, f"scenario_{scenario}", algo)
    runs = set()
    for pat in ["results_*.npz", "history_*.npz", "evaluation_*.npz"]:
        for f in glob.glob(os.path.join(base, pat)):
            stem = os.path.splitext(os.path.basename(f))[0]
            try: runs.add(int(stem.split("_")[1]))
            except: pass
    return sorted(runs)

def load_npz_by_run(root, scenario, algo, run):
    base = os.path.join(root, f"scenario_{scenario}", algo)
    for fname in [f"results_{run}.npz", f"history_{run}.npz", f"evaluation_{run}.npz"]:
        path = os.path.join(base, fname)
        if os.path.isfile(path): return np.load(path), path
    for pat in [f"results_{run}*.npz", f"history_{run}*.npz", f"evaluation_{run}*.npz"]:
        files = sorted(glob.glob(os.path.join(base, pat)))
        if files: return np.load(files[0]), files[0]
    return None, None

def _get_first_key_case_insensitive(data, candidates):
    keys = list(data.files)
    lower_map = {k.lower(): k for k in keys}
    for c in candidates:
        k = lower_map.get(c.lower())
        if k is not None:
            return k
    return None

def _to_violation_indicator(ts):
    # 每步 0/1 违规指示器
    if 'violation' in ts:
        v = ts['violation'].astype(np.float64)
        return (v > 0).astype(np.float64)
    if 'SLA' in ts:
        s = ts['SLA'].astype(np.float64)
        return (s <= 0).astype(np.float64)
    return None

def summarize_single_run(data):
    ts = {}

    # 常规序列
    for name, aliases in {
        'reward':    ['reward'],
        'resources': ['resources'],
        'violation': ['violation','violations','total_violations'],
    }.items():
        k = _get_first_key_case_insensitive(data, aliases)
        if k is not None: ts[name] = data[k].astype(np.float64)

    # SLA（可选）
    sla_key = _get_first_key_case_insensitive(data, ['SLA','sla','SLA_history','sla_history'])
    if sla_key is not None:
        ts['SLA'] = data[sla_key].astype(np.float64)

    # —— 优先级相关（可选）——
    for name, aliases in {
        'prio_weighted': ['prio_weighted','priority_weighted'],
        'prio_mmtc':     ['prio_mmtc'],
        'prio_embb_cbr': ['prio_embb_cbr','prio_cbr'],
        'prio_embb_vbr': ['prio_embb_vbr','prio_vbr'],
        'cnt_mmtc':      ['cnt_mmtc'],
        'cnt_embb_cbr':  ['cnt_embb_cbr'],
        'cnt_embb_vbr':  ['cnt_embb_vbr'],
    }.items():
        k = _get_first_key_case_insensitive(data, aliases)
        if k is not None:
            ts[name] = data[k].astype(np.float64)

    # 违规率（rolling & cumulative）
    z = _to_violation_indicator(ts)
    viol_rate_final = np.nan
    if z is not None and len(z) > 0:
        steps = np.arange(1, len(z) + 1, dtype=np.float64)
        ts['viol_rate_cum'] = np.cumsum(z) / steps
        viol_rate_final = float(ts['viol_rate_cum'][-1])
        ts['viol_rate_roll'] = (rolling_mean(z, ROLL_RATE_WIN)
                                if ROLL_RATE_WIN and len(z) >= ROLL_RATE_WIN else z)

    # 统一 SLA 为 0/1
    metrics = {}
    if 'SLA' in ts:
        sla_binary = (ts['SLA'] > 0).astype(np.float64)
        ts['SLA'] = sla_binary
    elif 'violation' in ts:
        sla_binary = (ts['violation'] <= 0).astype(np.float64)
        ts['SLA'] = sla_binary
    else:
        sla_binary = None

    if sla_binary is not None and len(sla_binary) > 0:
        metrics['sla_mean'] = float(np.mean(sla_binary))
        metrics['sla_rate'] = float(np.sum(sla_binary) / len(sla_binary))
    else:
        metrics['sla_mean'] = np.nan
        metrics['sla_rate'] = np.nan

    # 其他指标
    metrics['avg_reward'] = float(np.mean(ts['reward']))    if 'reward'    in ts else np.nan
    metrics['res_mean']   = float(np.mean(ts['resources'])) if 'resources' in ts else np.nan
    metrics['viol_total'] = float(np.sum(ts['violation']))  if 'violation' in ts else np.nan
    metrics['viol_mean']  = float(np.mean(ts['violation'])) if 'violation' in ts else np.nan
    metrics['viol_rate']  = viol_rate_final

    # —— 优先级汇总（若存在）——
    if 'prio_weighted' in ts:
        metrics['prio_weight_sum']  = float(np.sum(ts['prio_weighted']))
        metrics['prio_weight_mean'] = float(np.mean(ts['prio_weighted']))
    for k in ('prio_mmtc','prio_embb_cbr','prio_embb_vbr','cnt_mmtc','cnt_embb_cbr','cnt_embb_vbr'):
        if k in ts:
            metrics[k + '_sum'] = float(np.sum(ts[k]))

    return metrics, ts

def merge_metrics(metrics_list):
    agg = defaultdict(list)
    for m in metrics_list:
        for k, v in m.items(): agg[k].append(v)
    return {k: float(np.nanmean(v)) for k, v in agg.items()}

def align_and_stack(series_list):
    if not series_list: return {}
    keys = sorted(set().union(*[s.keys() for s in series_list]))
    out = {}
    for k in keys:
        seqs = [s[k] for s in series_list if (k in s and isinstance(s[k], np.ndarray))]
        if not seqs: continue
        T = min(len(s) for s in seqs)
        out[k] = np.stack([s[:T] for s in seqs], axis=1)
    return out

def plot_mean_std(ax, arr, label, smooth_k=None, ls='-'):
    mean, std = np.mean(arr, axis=1), np.std(arr, axis=1)
    if smooth_k and len(mean) >= smooth_k:
        mean, std = rolling_mean(mean, smooth_k), rolling_mean(std, smooth_k)
    x = np.arange(len(mean))
    ax.plot(x, mean, label=label, linestyle=ls, linewidth=1.6)
    ax.fill_between(x, mean-std, mean+std, alpha=0.15)

def print_table(title, rows, cols):
    print(f"\n{title}")
    line = "-" * (14 + len(cols)*15)
    print(line)
    header = "Algorithm".ljust(12) + " | " + " | ".join([c.ljust(12) for c in cols])
    print(header); print(line)
    for algo, m in rows:
        cells = []
        for c in cols:
            v = m.get(c, np.nan)
            if np.isnan(v): cells.append("nan".ljust(12))
            elif c == "viol_total": cells.append(f"{v:.0f}".ljust(12))
            elif c in ("viol_rate","sla_rate","sla_mean"): cells.append(f"{v:.4f}".ljust(12))
            else: cells.append(f"{v:.4f}".ljust(12))
        print(algo.ljust(12) + " | " + " | ".join(cells))
    print(line)

def _apply_percentile_ylim(ax, ydata, p_low=1, p_high=99):
    if ydata is None or len(ydata) == 0: return
    lo = np.nanpercentile(ydata, p_low)
    hi = np.nanpercentile(ydata, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi): return
    if hi <= lo: return
    pad = 0.05 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

def _plot_prio_panels(fig, grid_axes, algo, ts, label_prefix):
    """将优先级拆成 4 张小图：Total/mMTC/eMBB-CBR/eMBB-VBR"""
    series = {
        "Total":    ts.get('prio_weighted'),
        "mMTC":     ts.get('prio_mmtc'),
        "eMBB-CBR": ts.get('prio_embb_cbr'),
        "eMBB-VBR": ts.get('prio_embb_vbr'),
    }
    for ax, (name, arr) in zip(grid_axes, series.items()):
        if arr is None:     # 缺哪个就跳过
            ax.set_visible(False)
            continue
        y = arr
        k = PRIO_SMOOTH_WINDOW or SMOOTH_WINDOW
        if k and len(y) >= k: y = rolling_mean(y, k)
        ax.plot(y, label=f"{label_prefix} {algo}", linewidth=1.6)
        ax.set_title(name)
        ax.grid(True)
        _apply_percentile_ylim(ax, y, *PRIO_YLIM_PERCENTILE)

def main():
    if SAVE_FIG: ensure_dir(OUT_DIR)
    scenarios = SCENARIOS if SCENARIOS else find_scenarios(ROOT_RESULTS_DIR)
    if not scenarios:
        print("No scenarios found."); return

    overall_agg = defaultdict(list)

    # 表格列（含优先级）
    cols = [
        "avg_reward", "viol_rate", "sla_rate", "sla_mean",
        "viol_total", "viol_mean", "res_mean",
        "prio_weight_sum", "prio_mmtc_sum", "prio_embb_cbr_sum", "prio_embb_vbr_sum",
        "cnt_mmtc_sum", "cnt_embb_cbr_sum", "cnt_embb_vbr_sum"
    ]

    for scenario in scenarios:
        algos = ALGOS if ALGOS else find_algos(ROOT_RESULTS_DIR, scenario)
        if not algos:
            print(f"[scenario {scenario}] No algos.")
            continue

        per_algo_metrics = OrderedDict()
        per_algo_series_runs = OrderedDict()

        for algo in algos:
            runs = RUNS if RUNS else find_runs(ROOT_RESULTS_DIR, scenario, algo)
            if not runs:
                continue

            metrics_runs, series_runs = [], []
            for run in runs:
                data, _ = load_npz_by_run(ROOT_RESULTS_DIR, scenario, algo, run)
                if data is None: continue
                m, ts = summarize_single_run(data)
                metrics_runs.append(m); series_runs.append(ts)

            if not metrics_runs:
                continue

            merged = merge_metrics(metrics_runs)
            per_algo_metrics[algo] = merged
            per_algo_series_runs[algo] = series_runs
            overall_agg[algo].append(merged)

        # —— 场景统计表 —— 
        if per_algo_metrics:
            rows = list(per_algo_metrics.items())
            print_table(f"=== Scenario {scenario} ===", rows, cols)

        # —— 绘图 —— #
        if per_algo_series_runs and SAVE_FIG:
            # 计算需要的行数
            split_prio = (PRIO_PANEL_MODE.lower() == "split")
            if split_prio:
                n_rows = 5 + 4   # 5 张基础图 + 4 张优先级子图
                figsize = (FIGSIZE_BASE[0], FIGSIZE_BASE[1] + 6)
            else:
                n_rows = 6
                figsize = FIGSIZE_BASE

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig, axs = plt.subplots(n_rows, 1, figsize=figsize, constrained_layout=True)

            # 基础 5 图
            base_axes = axs[:5]
            ax_map = {
                'reward':         base_axes[0],
                'violation':      base_axes[1],
                'resources':      base_axes[2],
                'viol_rate_roll': base_axes[3],
                'viol_rate_cum':  base_axes[4],
            }

            # 绘制基础图
            for algo, series_runs in per_algo_series_runs.items():
                if SHOW_BANDS and len(series_runs) > 1:
                    aligned = align_and_stack(series_runs)
                    for k, ax in ax_map.items():
                        if k in aligned:
                            plot_mean_std(ax, aligned[k], label=algo,
                                          smooth_k=(0 if k.startswith("viol_rate") else SMOOTH_WINDOW))
                else:
                    idx = 0 if REPRESENTATIVE_RUN_FOR_CURVE is None else \
                          max(0, min(REPRESENTATIVE_RUN_FOR_CURVE, len(series_runs)-1))
                    ts = series_runs[idx]
                    for k, ax in ax_map.items():
                        if k in ts and isinstance(ts[k], np.ndarray):
                            y = ts[k]
                            if (not k.startswith("viol_rate")
                                and SMOOTH_WINDOW and len(y) >= SMOOTH_WINDOW):
                                y = rolling_mean(y, SMOOTH_WINDOW)
                            ax.plot(y, label=algo, linewidth=1.6)

            # 标题/网格/图例
            base_axes[0].set_title(f"Scenario {scenario} — Reward")
            base_axes[1].set_title(f"Scenario {scenario} — Violations (per-step count)")
            base_axes[2].set_title(f"Scenario {scenario} — Resources (sum PRBs)")
            base_axes[3].set_title(f"Scenario {scenario} — SLA Violation Rate (rolling={ROLL_RATE_WIN})")
            base_axes[4].set_title(f"Scenario {scenario} — SLA Violation Rate (cumulative)")
            for ax in base_axes: ax.grid(True); ax.legend(); ax.set_xlabel("steps")

            # 优先级：拆分 or 合并
            if split_prio:
                prio_axes = axs[5:]  # 4 张
                # 先统一清空子标题，绘制时再设
                for ax in prio_axes: ax.set_title("")
                # 按算法分别绘
                for algo, series_runs in per_algo_series_runs.items():
                    idx = 0 if REPRESENTATIVE_RUN_FOR_CURVE is None else \
                          max(0, min(REPRESENTATIVE_RUN_FOR_CURVE, len(series_runs)-1))
                    ts = series_runs[idx]
                    _plot_prio_panels(fig, prio_axes, algo, ts, label_prefix="")
                # 布局/标签
                prio_axes[0].set_ylabel("priority-weighted\nviolations")
                for ax in prio_axes:
                    ax.set_xlabel("steps")
                    ax.legend(loc="upper right")
            else:
                ax = axs[5]
                ax.set_title(f"Scenario {scenario} — Priority-weighted Violations (per-step)")
                for algo, series_runs in per_algo_series_runs.items():
                    idx = 0 if REPRESENTATIVE_RUN_FOR_CURVE is None else \
                          max(0, min(REPRESENTATIVE_RUN_FOR_CURVE, len(series_runs)-1))
                    ts = series_runs[idx]
                    if 'prio_weighted' in ts:
                        y = ts['prio_weighted']
                        k = PRIO_SMOOTH_WINDOW or SMOOTH_WINDOW
                        if k and len(y) >= k: y = rolling_mean(y, k)
                        ax.plot(y, label=f"{algo} total", linewidth=1.6)
                        for kname, lab, ls in [('prio_mmtc','mMTC','--'),
                                               ('prio_embb_cbr','eMBB-CBR',':'),
                                               ('prio_embb_vbr','eMBB-VBR','-.')]:
                            if kname in ts:
                                yk = ts[kname]
                                if PRIO_SMOOTH_WINDOW and len(yk) >= PRIO_SMOOTH_WINDOW:
                                    yk = rolling_mean(yk, PRIO_SMOOTH_WINDOW)
                                ax.plot(yk, label=f"{algo} {lab}", linestyle=ls, linewidth=1.6)
                # 自动 y 轴范围
                all_y = []
                for algo, series_runs in per_algo_series_runs.items():
                    ts = series_runs[0]
                    for k in ('prio_weighted','prio_mmtc','prio_embb_cbr','prio_embb_vbr'):
                        if k in ts: all_y.append(ts[k])
                if all_y:
                    all_y = np.concatenate(all_y)
                    _apply_percentile_ylim(ax, all_y, *PRIO_YLIM_PERCENTILE)
                ax.grid(True); ax.legend(); ax.set_xlabel("steps")

            out_path = os.path.join(OUT_DIR, f"scenario_{scenario}_comparison_{timestamp}.png")
            plt.savefig(out_path, dpi=150); plt.close(fig)
            print(f"Saved figure: {out_path}")

    # —— 所有场景综合表 —— 
    if overall_agg:
        overall = OrderedDict((algo, merge_metrics(lst)) for algo, lst in overall_agg.items())
        rows = list(overall.items())
        print_table("=== All Scenarios — Aggregated Metrics (mean over scenarios) ===", rows, [
            "avg_reward", "viol_rate", "sla_rate", "sla_mean",
            "viol_total", "viol_mean", "res_mean",
            "prio_weight_sum", "prio_mmtc_sum", "prio_embb_cbr_sum", "prio_embb_vbr_sum",
            "cnt_mmtc_sum", "cnt_embb_cbr_sum", "cnt_embb_vbr_sum"
        ])

if __name__ == "__main__":
    main()