#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =========================
# Editable config
# 直接改这里，然后运行:
# python offline_pretrain_lstm_multi_seed_3N.py
# =========================
INPUT_GLOBS = ["./results/ablations/scenario_1/full/history_run*_env*.npz"]
OUTPUT_PATH = "./traffic_lstm_pretrained_3N_multiseed_all.pth"
HISTORY_LEN = 10
HIDDEN_SIZE = 128
NUM_LAYERS = 1
BATCH_SIZE = 512
EPOCHS = 40
LR = 5e-4
STRIDE = 4
VAL_RATIO = 0.0
SEED = 0
CLIP_GRAD_NORM = 5.0
USE_STD_SCALE = True
PATIENCE = 0


@dataclass
class TrainConfig:
    input_globs: list[str]
    output_path: str
    history_len: int = 10
    hidden_size: int = 128
    num_layers: int = 1
    batch_size: int = 512
    epochs: int = 80
    lr: float = 1e-3
    stride: int = 4
    val_ratio: float = 0.2
    seed: int = 0
    clip_grad_norm: float = 5.0
    use_std_scale: bool = True
    patience: int = 12


class SeqDataset(Dataset):
    def __init__(self, series_list: list[np.ndarray], history_len: int, stride: int = 1):
        self.series_list = [np.asarray(s, dtype=np.float32) for s in series_list if len(s) > history_len]
        self.history_len = int(history_len)
        self.stride = max(1, int(stride))
        self.index: list[tuple[int, int]] = []
        for sid, series in enumerate(self.series_list):
            limit = series.shape[0] - self.history_len
            for start in range(0, limit, self.stride):
                self.index.append((sid, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, start = self.index[idx]
        series = self.series_list[sid]
        x = series[start : start + self.history_len]
        y = series[start + self.history_len]
        return torch.from_numpy(x), torch.from_numpy(y)


class TrafficLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _collect_files(input_globs: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in input_globs:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No npz files matched: {input_globs}")
    return files


def _load_series(files: list[str]) -> list[np.ndarray]:
    series_list = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        if "exog_vec" not in data:
            raise KeyError(f"{path} has no exog_vec")
        x = np.asarray(data["exog_vec"], dtype=np.float32)
        if x.ndim != 2:
            raise RuntimeError(f"{path}: exog_vec shape invalid: {x.shape}")
        series_list.append(x)
    return series_list


def _split_files(files: list[str], val_ratio: float, seed: int):
    if len(files) <= 1 or val_ratio <= 0:
        return files, []
    rng = np.random.default_rng(seed)
    order = list(files)
    rng.shuffle(order)
    n_val = int(round(len(order) * float(val_ratio)))
    n_val = min(max(n_val, 1), len(order) - 1)
    val_files = order[:n_val]
    train_files = order[n_val:]
    return train_files, val_files


def _eval_loss(model, loader, device):
    if loader is None:
        return float("nan")
    model.eval()
    total = 0.0
    count = 0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total += float(loss.item())
            count += 1
    return total / max(1, count)


def main():
    cfg = TrainConfig(
        input_globs=list(INPUT_GLOBS),
        output_path=OUTPUT_PATH,
        history_len=HISTORY_LEN,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        stride=STRIDE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        clip_grad_norm=CLIP_GRAD_NORM,
        use_std_scale=USE_STD_SCALE,
        patience=PATIENCE,
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    files = _collect_files(cfg.input_globs)
    train_files, val_files = _split_files(files, cfg.val_ratio, cfg.seed)
    train_series = _load_series(train_files)
    val_series = _load_series(val_files) if val_files else []
    if not train_series:
        raise RuntimeError("No training series loaded.")

    raw_train = np.concatenate(train_series, axis=0)
    if cfg.use_std_scale:
        scale = raw_train.std(axis=0).astype(np.float32) + 1e-6
    else:
        scale = np.ones((raw_train.shape[1],), dtype=np.float32)

    train_series_norm = [s / scale for s in train_series]
    val_series_norm = [s / scale for s in val_series]

    train_ds = SeqDataset(train_series_norm, cfg.history_len, stride=cfg.stride)
    val_ds = SeqDataset(val_series_norm, cfg.history_len, stride=cfg.stride) if val_series_norm else None
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrafficLSTM(
        input_dim=int(raw_train.shape[1]),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    print(f"[INFO] files={len(files)} train={len(train_files)} val={len(val_files)}")
    print(f"[INFO] train samples={len(train_ds)} val samples={(len(val_ds) if val_ds is not None else 0)} stride={cfg.stride}")
    print(f"[INFO] device={device} output={cfg.output_path}")

    for ep in range(cfg.epochs):
        model.train()
        total = 0.0
        count = 0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            opt.step()
            total += float(loss.item())
            count += 1

        train_loss = total / max(1, count)
        val_loss = _eval_loss(model, val_dl, device) if val_dl is not None else float("nan")
        print(f"[epoch {ep+1:03d}/{cfg.epochs}] train={train_loss:.6f} val={val_loss:.6f}")

        current = val_loss if np.isfinite(val_loss) else train_loss
        if current < best_val:
            best_val = current
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if cfg.patience > 0 and bad_epochs >= cfg.patience:
                print(f"[INFO] early stopping at epoch {ep+1}, best_epoch={best_epoch}, best_loss={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    payload = {
        "state_dict": model.state_dict(),
        "feat_dim": int(raw_train.shape[1]),
        "history_len": int(cfg.history_len),
        "hidden_size": int(cfg.hidden_size),
        "num_layers": int(cfg.num_layers),
        "scale": scale.astype(np.float32),
        "train_files": train_files,
        "val_files": val_files,
        "stride": int(cfg.stride),
        "config": asdict(cfg),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_val),
    }
    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    torch.save(payload, cfg.output_path)
    print(f"[OK] saved -> {cfg.output_path}")


if __name__ == "__main__":
    main()
