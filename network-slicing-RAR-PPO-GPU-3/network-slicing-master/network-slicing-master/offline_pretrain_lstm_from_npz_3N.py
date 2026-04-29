import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===========================
# 你只需要改这里
# ===========================
NPZ_PATH = "./lstm_datasets/history_0.npz"
OUT_PATH = "./traffic_lstm_pretrained_3N.pth"

# [新增] loss 曲线输出路径
LOSS_PNG_PATH = "./traffic_lstm_loss_curve.png"
SHOW_PLOT = False  # True 则训练结束后弹窗显示；服务器/无GUI环境建议 False

HISTORY_LEN = 10
HIDDEN_SIZE = 128
NUM_LAYERS = 1

BATCH_SIZE = 256
EPOCHS = 300
LR = 1e-3

USE_STD_SCALE = True     # True: 仅用 std 做 scale（与线上 x/scale 一致）
CLIP_GRAD_NORM = 5.0
SEED = 0
# ===========================

class SeqDataset(Dataset):
    def __init__(self, series: np.ndarray, history_len: int):
        assert series.ndim == 2
        self.series = series.astype(np.float32)
        self.H = int(history_len)

    def __len__(self):
        return max(0, self.series.shape[0] - self.H)

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.H]   # [H, D]
        y = self.series[idx + self.H]        # [D]
        return torch.from_numpy(x), torch.from_numpy(y)

class TrafficLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        y = self.fc(out[:, -1, :])
        return y

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"找不到 npz：{NPZ_PATH}")

    data = np.load(NPZ_PATH, allow_pickle=True)
    if "exog_vec" not in data:
        raise KeyError("npz 中没有 exog_vec。请先按我上面修改 ReportWrapper，重新跑一遍生成 npz。")

    series = data["exog_vec"].astype(np.float32)
    if series.ndim != 2:
        raise RuntimeError(f"exog_vec 维度不对：{series.shape}")

    T, D = series.shape
    if T <= HISTORY_LEN + 1:
        raise RuntimeError(f"样本不足：T={T}, HISTORY_LEN={HISTORY_LEN}")

    # 用 std 做 scale，匹配线上 wrapper 的 x/scale（不做减均值）
    scale = np.ones((D,), dtype=np.float32)
    series_norm = series.copy()
    if USE_STD_SCALE:
        std = series.std(axis=0).astype(np.float32) + 1e-6
        scale = std
        series_norm = series / scale

    ds = SeqDataset(series_norm, HISTORY_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrafficLSTM(input_dim=D, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # [新增] 记录每个 epoch 的平均 loss
    epoch_losses = []

    model.train()
    for ep in range(EPOCHS):
        total = 0.0
        n = 0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            opt.step()

            total += float(loss.item())
            n += 1

        avg_loss = total / max(1, n)
        epoch_losses.append(avg_loss)
        print(f"[epoch {ep+1:03d}/{EPOCHS}] loss={avg_loss:.6f}")

    # [新增] 训练完成后绘制并保存 loss 曲线（PNG）
    try:
        import matplotlib
        if not SHOW_PLOT:
            matplotlib.use("Agg")  # 无GUI环境也可保存
        import matplotlib.pyplot as plt

        xs = np.arange(1, EPOCHS + 1)
        plt.figure()
        plt.plot(xs, np.array(epoch_losses, dtype=np.float32))
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss Curve")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.savefig(LOSS_PNG_PATH, dpi=200)
        if SHOW_PLOT:
            plt.show()
        plt.close()
        print(f"[OK] saved loss curve -> {LOSS_PNG_PATH}")
    except Exception as e:
        print(f"[WARN] 画 loss 曲线失败：{repr(e)}（不影响模型保存）")

    ckpt = {
        "state_dict": model.state_dict(),
        "feat_dim": int(D),
        "history_len": int(HISTORY_LEN),
        "hidden_size": int(HIDDEN_SIZE),
        "num_layers": int(NUM_LAYERS),
        "scale": scale.astype(np.float32),
    }
    torch.save(ckpt, OUT_PATH)
    print(f"[OK] saved pretrained lstm -> {OUT_PATH}")

if __name__ == "__main__":
    main()

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# # ===========================
# # 你只需要改这里
# # ===========================
# NPZ_PATH = "./lstm_datasets/history_0.npz"
# OUT_PATH = "./traffic_lstm_pretrained_3N.pth"

# HISTORY_LEN = 10
# HIDDEN_SIZE = 128
# NUM_LAYERS = 1

# BATCH_SIZE = 256
# EPOCHS = 300
# LR = 1e-3

# USE_STD_SCALE = True     # True: 仅用 std 做 scale（与线上 x/scale 一致）
# CLIP_GRAD_NORM = 5.0
# SEED = 0
# # ===========================

# class SeqDataset(Dataset):
#     def __init__(self, series: np.ndarray, history_len: int):
#         assert series.ndim == 2
#         self.series = series.astype(np.float32)
#         self.H = int(history_len)

#     def __len__(self):
#         return max(0, self.series.shape[0] - self.H)

#     def __getitem__(self, idx):
#         x = self.series[idx: idx + self.H]   # [H, D]
#         y = self.series[idx + self.H]        # [D]
#         return torch.from_numpy(x), torch.from_numpy(y)

# class TrafficLSTM(nn.Module):
#     def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, input_dim)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         y = self.fc(out[:, -1, :])
#         return y

# def main():
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)

#     if not os.path.exists(NPZ_PATH):
#         raise FileNotFoundError(f"找不到 npz：{NPZ_PATH}")

#     data = np.load(NPZ_PATH, allow_pickle=True)
#     if "exog_vec" not in data:
#         raise KeyError("npz 中没有 exog_vec。请先按我上面修改 ReportWrapper，重新跑一遍生成 npz。")

#     series = data["exog_vec"].astype(np.float32)
#     if series.ndim != 2:
#         raise RuntimeError(f"exog_vec 维度不对：{series.shape}")

#     T, D = series.shape
#     if T <= HISTORY_LEN + 1:
#         raise RuntimeError(f"样本不足：T={T}, HISTORY_LEN={HISTORY_LEN}")

#     # 用 std 做 scale，匹配线上 wrapper 的 x/scale（不做减均值）
#     scale = np.ones((D,), dtype=np.float32)
#     series_norm = series.copy()
#     if USE_STD_SCALE:
#         std = series.std(axis=0).astype(np.float32) + 1e-6
#         scale = std
#         series_norm = series / scale

#     ds = SeqDataset(series_norm, HISTORY_LEN)
#     dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = TrafficLSTM(input_dim=D, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)

#     opt = torch.optim.Adam(model.parameters(), lr=LR)
#     loss_fn = nn.MSELoss()

#     model.train()
#     for ep in range(EPOCHS):
#         total = 0.0
#         n = 0
#         for x, y in dl:
#             x = x.to(device)
#             y = y.to(device)
#             pred = model(x)
#             loss = loss_fn(pred, y)

#             opt.zero_grad()
#             loss.backward()
#             if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
#             opt.step()

#             total += float(loss.item())
#             n += 1
#         print(f"[epoch {ep+1:03d}/{EPOCHS}] loss={total/max(1,n):.6f}")

#     ckpt = {
#         "state_dict": model.state_dict(),
#         "feat_dim": int(D),
#         "history_len": int(HISTORY_LEN),
#         "hidden_size": int(HIDDEN_SIZE),
#         "num_layers": int(NUM_LAYERS),
#         "scale": scale.astype(np.float32),
#     }
#     torch.save(ckpt, OUT_PATH)
#     print(f"[OK] saved pretrained lstm -> {OUT_PATH}")

# if __name__ == "__main__":
#     main()
