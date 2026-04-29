# eval_lstm_on_npz_3N.py
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque

# =========================
# 你只改这里
# =========================
NPZ_GLOB = "./results/scenario_1/RaRPPO/lstm_predictions_run0.npz"   # 评估用轨迹（建议用“没参与预训练”的 run 做验证集）
CKPT_PATH = "./traffic_lstm_pretrained_3N.pth"          # 你的 3N 离线预训练权重
HISTORY_LEN = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 可选：只评估某些维度
REPORT_TOPK_WORST = 10
# =========================
# ====== 绘图开关 ======
PLOT = True
PLOT_DIR = "./lstm_eval_plots"
WINDOW = 20  # 时间滑窗长度（用于画 windowed Corr/R2）
# ======================

class TrafficLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _load_ckpt(path: str, feat_dim_expected: int | None = None):
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
        feat_dim = int(payload.get("feat_dim", payload.get("input_dim", 0)))
        hidden_size = int(payload.get("hidden_size", 128))
        num_layers = int(payload.get("num_layers", 1))
        scale = payload.get("scale", None)  # 你的 wrapper 走 scale 归一化就用这个
        mean = payload.get("mean", None)    # 如果你离线脚本走 z-score，就会有 mean/std
        std  = payload.get("std", None)
    else:
        raise RuntimeError("Unsupported checkpoint format")

    if feat_dim_expected is not None and feat_dim != feat_dim_expected:
        raise RuntimeError(f"ckpt feat_dim={feat_dim} != expected={feat_dim_expected}")

    model = TrafficLSTM(feat_dim, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()

    return model, feat_dim, scale, mean, std


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    # y_*: [M, D]
    eps = 1e-8
    err = y_pred - y_true
    mse = np.mean(err**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err), axis=0)

    var = np.var(y_true, axis=0)
    r2 = 1.0 - mse / (var + eps)

    # 相关系数（逐维）
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    corr = np.sum(yt * yp, axis=0) / (np.sqrt(np.sum(yt**2, axis=0) * np.sum(yp**2, axis=0)) + eps)

    # 归一化 RMSE：除以真实值的 std（更稳定）
    nrmse = rmse / (np.std(y_true, axis=0) + eps)

    return {"mae": mae, "rmse": rmse, "nrmse": nrmse, "r2": r2, "corr": corr}


def main():
    files = sorted(glob.glob(NPZ_GLOB))
    if not files:
        raise FileNotFoundError(f"no npz matched: {NPZ_GLOB}")

    # 先读一个 npz 拿 feat_dim
    tmp = np.load(files[0], allow_pickle=True)
    if "exog_vec" not in tmp:
        raise KeyError("npz lacks exog_vec (shape [T,3N]). 请确认你的 ReportWrapper 已保存 exog_vec。")
    feat_dim_expected = int(tmp["exog_vec"].shape[1])

    model, feat_dim, scale, mean, std = _load_ckpt(CKPT_PATH, feat_dim_expected=feat_dim_expected)

    all_true = []
    all_pred = []

    for f in files:
        data = np.load(f, allow_pickle=True)
        x = data["exog_vec"].astype(np.float32)   # [T,D]
        T, D = x.shape
        if T <= HISTORY_LEN + 1:
            continue

        # 归一化口径：优先用 ckpt 里的 mean/std（z-score），否则用 scale（除法），否则不归一化
        if mean is not None and std is not None:
            mean_ = np.asarray(mean, dtype=np.float32).reshape(1, -1)
            std_  = np.asarray(std,  dtype=np.float32).reshape(1, -1)
            x_norm = (x - mean_) / (std_ + 1e-6)
            inv = lambda y: y * (std_ + 1e-6) + mean_
        elif scale is not None:
            scale_ = np.asarray(scale, dtype=np.float32).reshape(1, -1)
            x_norm = x / (scale_ + 1e-8)
            inv = lambda y: y * (scale_ + 1e-8)
        else:
            x_norm = x
            inv = lambda y: y

        # teacher forcing one-step
        M = T - HISTORY_LEN
        y_true = x_norm[HISTORY_LEN: ]                 # [M, D]
        y_pred = np.zeros((M, D), dtype=np.float32)

        with torch.no_grad():
            for i in range(M):
                seq = x_norm[i:i+HISTORY_LEN]          # [H, D]
                inp = torch.from_numpy(seq[None, ...]).to(DEVICE)
                out = model(inp).cpu().numpy()[0]
                y_pred[i] = out

        # 反归一化后再计算误差（你也可以同时算 norm-space 的误差）
        y_true_raw = inv(y_true)
        y_pred_raw = inv(y_pred)

        all_true.append(y_true_raw)
        all_pred.append(y_pred_raw)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    ms = _metrics(y_true, y_pred)

    def summarize(name, arr):
        print(f"{name}: mean={arr.mean():.6f}  median={np.median(arr):.6f}  p90={np.percentile(arr,90):.6f}")

    print("== Overall ==")
    summarize("MAE", ms["mae"])
    summarize("RMSE", ms["rmse"])
    summarize("NRMSE", ms["nrmse"])
    summarize("R2", ms["r2"])
    summarize("Corr", ms["corr"])

    # 最差维度（便于你定位是哪类外生量难预测）
    worst = np.argsort(-ms["nrmse"])[:REPORT_TOPK_WORST]
    print("\n== Worst dims by NRMSE ==")
    for k, d in enumerate(worst):
        print(f"[{k}] dim={d:3d}  NRMSE={ms['nrmse'][d]:.4f}  RMSE={ms['rmse'][d]:.4f}  Corr={ms['corr'][d]:.4f}  R2={ms['r2'][d]:.4f}")

    # 分组统计：3N 中每切片三元组 (cbr, vbr, new_devices)
    # cbr: idx%3==0, vbr: idx%3==1, new_dev: idx%3==2
    idx_cbr = np.arange(feat_dim)[(np.arange(feat_dim) % 3) == 0]
    idx_vbr = np.arange(feat_dim)[(np.arange(feat_dim) % 3) == 1]
    idx_ndv = np.arange(feat_dim)[(np.arange(feat_dim) % 3) == 2]

    def group(name, idx):
        print(f"\n== Group {name} ==")
        summarize("MAE", ms["mae"][idx])
        summarize("RMSE", ms["rmse"][idx])
        summarize("NRMSE", ms["nrmse"][idx])
        summarize("Corr", ms["corr"][idx])

    group("CBR (dims %3==0)", idx_cbr)
    group("VBR (dims %3==1)", idx_vbr)
    group("NewDevices (dims %3==2)", idx_ndv)

        # =========================
    # 绘图：维度曲线 + 时间曲线
    # =========================
    if PLOT:
        os.makedirs(PLOT_DIR, exist_ok=True)

        D = feat_dim
        dims = np.arange(D)

        # ---------- 1) 按维度的指标曲线（每个 dim 一个点） ----------
        def _plot_by_dim(arr, title, ylabel, fname):
            plt.figure(figsize=(12, 5))
            plt.plot(dims, arr, linewidth=1.0)
            plt.xlabel("dimension (0..3N-1)")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
            plt.close()

        _plot_by_dim(ms["nrmse"], "NRMSE per-dimension", "NRMSE", "by_dim_nrmse.png")
        _plot_by_dim(ms["r2"],    "R2 per-dimension",    "R2",    "by_dim_r2.png")
        _plot_by_dim(ms["corr"],  "Corr per-dimension",  "Corr",  "by_dim_corr.png")

        # 也可以加一个直方图，看分布
        def _hist(arr, title, fname):
            plt.figure(figsize=(8, 5))
            plt.hist(arr[np.isfinite(arr)], bins=60)
            plt.xlabel(title)
            plt.ylabel("count")
            plt.title(f"Histogram: {title}")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
            plt.close()

        _hist(ms["nrmse"], "NRMSE", "hist_nrmse.png")
        _hist(ms["r2"], "R2", "hist_r2.png")
        _hist(ms["corr"], "Corr", "hist_corr.png")

        # ---------- 2) 按时间的误差曲线 ----------
        # y_true/y_pred: [M_total, D]
        Err = y_pred - y_true              # [M, D]
        AE = np.abs(Err)
        SE = Err * Err

        # per-step: 对维度聚合 -> 得到随时间变化的标量曲线
        # 注意：这里的 NRMSE(t) 用“每维 std”归一化后再聚合
        STD = np.std(y_true, axis=0, keepdims=True) + 1e-6  # [1,D]
        step_nrmse_mean = np.mean(np.sqrt(SE) / STD, axis=1)     # [M]
        step_nrmse_med  = np.median(np.sqrt(SE) / STD, axis=1)   # [M]
        step_mae_mean   = np.mean(AE, axis=1)                    # [M]
        step_mae_med    = np.median(AE, axis=1)                  # [M]

        def _plot_time(arr_list, labels, title, ylabel, fname, ylim=None):
            plt.figure(figsize=(12, 5))
            for a, lab in zip(arr_list, labels):
                plt.plot(a, label=lab, linewidth=1.0)
            plt.xlabel("t (one-step prediction index)")
            plt.ylabel(ylabel)
            plt.title(title)
            if ylim is not None:
                plt.ylim(ylim)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, fname), dpi=150)
            plt.close()

        _plot_time([step_nrmse_mean, step_nrmse_med],
                   ["step NRMSE mean", "step NRMSE median"],
                   "Per-step NRMSE over time", "NRMSE",
                   "time_step_nrmse.png")

        _plot_time([step_mae_mean, step_mae_med],
                   ["step MAE mean", "step MAE median"],
                   "Per-step MAE over time", "MAE (raw units)",
                   "time_step_mae.png")

        # ---------- 3) 按时间的 windowed Corr / R2（更接近“趋势对齐”） ----------
        W = int(WINDOW)
        corr_curve = np.full((y_true.shape[0],), np.nan, dtype=np.float32)
        r2_curve   = np.full((y_true.shape[0],), np.nan, dtype=np.float32)

        def _corr_vec(a, b):
            # a,b: [W,D]
            a0 = a - a.mean(axis=0, keepdims=True)
            b0 = b - b.mean(axis=0, keepdims=True)
            num = np.sum(a0 * b0, axis=0)
            den = np.sqrt(np.sum(a0*a0, axis=0) * np.sum(b0*b0, axis=0)) + 1e-8
            return num / den  # [D]

        def _r2_vec(a, b):
            mse = np.mean((b - a) ** 2, axis=0)
            var = np.var(a, axis=0) + 1e-6
            return 1.0 - mse / var

        for i in range(y_true.shape[0]):
            if i + 1 < W:
                continue
            a = y_true[i+1-W:i+1]
            b = y_pred[i+1-W:i+1]
            c = _corr_vec(a, b)
            r = _r2_vec(a, b)

            # 常数维度会导致 corr nan/0，用 nanmean 更稳
            corr_curve[i] = float(np.nanmean(c))
            r2_curve[i]   = float(np.nanmean(r))

        _plot_time([corr_curve],
                   [f"window Corr mean (W={W})"],
                   "Windowed Corr over time", "Corr",
                   "time_window_corr.png", ylim=(-1.0, 1.0))

        _plot_time([r2_curve],
                   [f"window R2 mean (W={W})"],
                   "Windowed R2 over time", "R2",
                   "time_window_r2.png")

        print("[PLOT] saved to:", os.path.abspath(PLOT_DIR))


if __name__ == "__main__":
    main()
