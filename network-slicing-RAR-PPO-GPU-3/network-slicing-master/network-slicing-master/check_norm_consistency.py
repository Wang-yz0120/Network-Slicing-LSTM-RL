import numpy as np
import torch

NPZ_PATH = "./lstm_datasets/history_0.npz"
CKPT_PATH = "./traffic_lstm_pretrained_3N.pth"

d = np.load(NPZ_PATH, allow_pickle=True)
x = d["exog_vec"].astype(np.float32)   # [T, 3N]
print("exog_vec shape:", x.shape)
print("raw stats: mean/std/min/max =", x.mean(), x.std(), x.min(), x.max())

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print("ckpt keys:", ckpt.keys())

# 你离线脚本可能保存 mean/std
mean = ckpt.get("mean", None)
std  = ckpt.get("std", None)
scale = ckpt.get("scale", None)

if mean is not None and std is not None:
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std  = np.asarray(std,  dtype=np.float32).reshape(1, -1)
    x_z = (x - mean) / (std + 1e-6)
    print("zscore stats: mean/std/min/max =", x_z.mean(), x_z.std(), x_z.min(), x_z.max())
else:
    print("ckpt has no mean/std, skip z-score check")

if scale is not None:
    scale = np.asarray(scale, dtype=np.float32).reshape(1, -1)
    x_sc = x / (scale + 1e-8)
    print("scale stats: mean/std/min/max =", x_sc.mean(), x_sc.std(), x_sc.min(), x_sc.max())
else:
    print("ckpt has no scale")
