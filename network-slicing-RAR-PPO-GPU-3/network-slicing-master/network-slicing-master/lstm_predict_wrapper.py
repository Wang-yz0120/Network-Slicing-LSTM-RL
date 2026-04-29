# lstm_predict_wrapper.py
# -*- coding: utf-8 -*-
import os
from typing import Optional, Any, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


def _normalize_reset(ret):
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0], (ret[1] if isinstance(ret[1], dict) else {})
    return ret, {}


def _normalize_step(ret):
    if isinstance(ret, tuple) and len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        if not isinstance(info, dict):
            info = {}
        return obs, float(reward), bool(terminated), bool(truncated), info
    if isinstance(ret, tuple) and len(ret) == 4:
        obs, reward, done, info = ret
        if not isinstance(info, dict):
            info = {}
        return obs, float(reward), bool(done), False, info
    raise RuntimeError("Unsupported env.step return format")


class TrafficLSTM(nn.Module):
    # 输入/输出都是 3*N 的向量，按时间序列 [B,T,3N] -> 预测下一步 [B,3N]
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_dim)

    def forward(self, x, h=None):
        out, h = self.lstm(x, h)
        y = self.fc(out[:, -1, :])
        return y, h


class LSTMPredictWrapper(gym.Wrapper):
    """
    目标（外生负载预测）：
      - eMBB: 预测 [cbr_traffic, vbr_traffic]
      - mMTC: 预测 [new_devices]（本 step 新到达设备数）

    特征向量（每切片 3 维，拼成 3N）：
      - eMBB 切片： [cbr_traffic, vbr_traffic, 0]
      - mMTC 切片： [0, 0, new_devices]
    """
    def __init__(
        self,
        env: gym.Env,
        history_len: int = 10,
        hidden_size: int = 128,
        lr: float = 1e-3,
        device: Optional[str] = None,

        # 预训练权重
        pretrained_path: Optional[str] = None,

        # 在线微调（replay）
        online_finetune: bool = True,
        finetune_lr: Optional[float] = None,
        train_every: int = 8,
        warmup_steps: int = 500,
        replay_size: int = 20000,
        min_replay: int = 512,
        batch_size: int = 64,
        grad_clip: float = 1.0,
        weight_decay: float = 0.0,

        # 自动保存微调后的权重（可选）
        autosave_path: Optional[str] = None,
        autosave_every_updates: int = 5000,
    ):
        super().__init__(env)
        self.history_len = int(history_len)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 找到底层 node_b
        base = env
        for _ in range(10):
            if hasattr(base, "node_b"):
                break
            base = getattr(base, "env", None)
            if base is None:
                raise RuntimeError("LSTMPredictWrapper: cannot find env.node_b")
        self.node_b = base.node_b

        # 切片类型列表（按 gid 顺序）
        self.slice_types = []
        for l1 in self.node_b.slices_l1:
            for s in l1.slices_ran:
                self.slice_types.append(str(getattr(s, "type", "")).lower())

        self.n_slices = len(self.slice_types)
        if self.n_slices == 0:
            raise RuntimeError("LSTMPredictWrapper: no slices found")

        self.embb_indices = [i for i, t in enumerate(self.slice_types) if t == "embb"]
        self.mmtc_indices = [i for i, t in enumerate(self.slice_types) if t == "mmtc"]
        self.n_embb = len(self.embb_indices)
        self.n_mmtc = len(self.mmtc_indices)

        # 每切片 3 维：cbr_traffic, vbr_traffic, new_devices
        self.feat_dim = 3 * self.n_slices

        # 归一化尺度（先保持 1；如需可按场景调）
        self.scale = np.ones(self.feat_dim, dtype=np.float32)
        self._scale_loaded = False

        # 模型 + 优化器
        self.model = TrafficLSTM(input_dim=self.feat_dim, hidden_size=hidden_size).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 拼接到 observation 的预测维度：eMBB 2个 / mMTC 1个
        self.pred_dim = 2 * self.n_embb + 1 * self.n_mmtc

        # prediction scale warmup (median-based)
        self.pred_scale_warmup = 0
        self.pred_scale = 1
        self._scale_obs_vals = []
        self._scale_pred_vals = []

        # 扩展 observation_space
        orig = env.observation_space
        low = np.concatenate([orig.low, -np.ones(self.pred_dim) * np.inf]).astype(np.float32)
        high = np.concatenate([orig.high, np.ones(self.pred_dim) * np.inf]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 在线缓存
        self.history = deque(maxlen=self.history_len + 1)  # 归一化后的 x_t
        self.replay = deque(maxlen=int(replay_size))       # (x_in[T,D], x_out[D])

        self.online_finetune = bool(online_finetune)
        self.train_every = int(train_every)
        self.warmup_steps = int(warmup_steps)
        self.min_replay = int(min_replay)
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)

        self.autosave_path = autosave_path
        self.autosave_every_updates = int(autosave_every_updates)

        self._env_step = 0
        self._update_step = 0
        self.loss_history = []

        ########################
        self.pred_full_raw_hist = []  # 存储预测值（原始尺度）
        self.real_full_raw_hist = []  # 存储真实值（原始尺度）
        self._pending_pred_full_raw = None  # 用于对齐预测t-1与真实t
        ########################

        # 加载预训练
        if pretrained_path is not None and str(pretrained_path).strip():
            self.load(pretrained_path)
            if finetune_lr is None:
                finetune_lr = lr * 0.1
            self._set_lr(float(finetune_lr))

    def _set_lr(self, lr: float):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def save(self, path: str):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "feat_dim": self.feat_dim,
            "history_len": self.history_len,
            "scale": self.scale,
        }, path)

    # def load(self, path: str, strict: bool = True):
    #     # payload = torch.load(path, map_location=self.device)
    #     payload = torch.load(path, map_location=self.device, weights_only=False)
    #     sd = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    #     self.model.load_state_dict(sd, strict=strict)
    #     if isinstance(payload, dict) and "scale" in payload:
    #         sc = np.asarray(payload["scale"], dtype=np.float32).ravel()
    #         if sc.shape[0] == self.feat_dim:
    #             self.scale = sc

    def load(self, path: str, strict: bool = True):
        payload = torch.load(path, map_location=self.device, weights_only=False)

        # 1) state_dict
        sd = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        self.model.load_state_dict(sd, strict=strict)

        # 2) 强校验维度 + 读取 scale
        if not isinstance(payload, dict):
            raise RuntimeError("LSTMPredictWrapper.load: checkpoint payload is not a dict, cannot verify feat_dim/scale.")

        # 优先使用 feat_dim 校验（若存在）
        if "feat_dim" in payload:
            ckpt_dim = int(payload["feat_dim"])
            if ckpt_dim != int(self.feat_dim):
                raise RuntimeError(
                    f"LSTMPredictWrapper.load: feat_dim mismatch: ckpt={ckpt_dim} vs env={self.feat_dim}. "
                    "你可能加载错了 3 维(global3) 的 ckpt 或者场景 N 变化导致维度不同。"
                )

        if "scale" not in payload:
            raise RuntimeError("LSTMPredictWrapper.load: checkpoint has no 'scale'. 请确认你加载的是 3N 版本离线预训练 ckpt。")

        sc = np.asarray(payload["scale"], dtype=np.float32).ravel()
        if sc.shape[0] != int(self.feat_dim):
            raise RuntimeError(
                f"LSTMPredictWrapper.load: scale dim mismatch: ckpt_scale={sc.shape[0]} vs env_feat_dim={self.feat_dim}."
            )

        self.scale = sc
        self._scale_loaded = True

        # 可选：只打印一次，避免刷屏
        print(f"[LSTMPredictWrapper] loaded pretrained: feat_dim={self.feat_dim}, scale_min={self.scale.min():.6g}, scale_max={self.scale.max():.6g}")


    def _norm(self, x: np.ndarray) -> np.ndarray:
        return x / (self.scale + 1e-8)

    def _extract_feat_3N(self, info: Dict[str, Any]) -> np.ndarray:
        slices_dict = info.get("slices", {}) if isinstance(info, dict) else {}
        feat = np.zeros(self.feat_dim, dtype=np.float32)

        for gid in range(self.n_slices):
            si = slices_dict.get(gid, {}) if isinstance(slices_dict, dict) else {}
            t = str(si.get("type", "")).lower()
            base = 3 * gid

            if t == "embb":
                feat[base + 0] = float(si.get("cbr_traffic", 0.0))
                feat[base + 1] = float(si.get("vbr_traffic", 0.0))
                feat[base + 2] = 0.0
            elif t == "mmtc":
                feat[base + 0] = 0.0
                feat[base + 1] = 0.0
                feat[base + 2] = float(si.get("new_devices", 0.0))  # <-- 外生到达
            else:
                # 如果未来扩展了其它 type，尽量兜底
                feat[base + 0] = float(si.get("cbr_traffic", 0.0))
                feat[base + 1] = float(si.get("vbr_traffic", 0.0))
                feat[base + 2] = float(si.get("new_devices", 0.0))
        return feat

    def _short_pred_from_full(self, pred_full_norm: np.ndarray) -> np.ndarray:
        out = []
        for idx in self.embb_indices:
            base = 3 * idx
            out.append(pred_full_norm[base + 0])  # cbr_traffic
            out.append(pred_full_norm[base + 1])  # vbr_traffic
        for idx in self.mmtc_indices:
            base = 3 * idx
            out.append(pred_full_norm[base + 2])  # new_devices
        return np.array(out, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = _normalize_reset(self.env.reset(**kwargs))
        self.history.clear()
        self._env_step = 0
        pred_short = np.zeros(self.pred_dim, dtype=np.float32)
        obs_aug = np.concatenate([obs, pred_short], axis=-1).astype(np.float32)

        #########################################################################
        self.pred_full_raw_hist.clear()
        self.real_full_raw_hist.clear()
        self._pending_pred_full_raw = None

        #########################################################################
        return obs_aug, info

    def _maybe_train(self):
        if not self.online_finetune:
            return
        if self._env_step < self.warmup_steps:
            return
        if len(self.replay) < self.min_replay:
            return
        if self.train_every <= 0 or (self._env_step % self.train_every) != 0:
            return

        B = min(self.batch_size, len(self.replay))
        idx = np.random.choice(len(self.replay), size=B, replace=False)
        x_in = np.stack([self.replay[i][0] for i in idx], axis=0)   # [B,T,D]
        x_out = np.stack([self.replay[i][1] for i in idx], axis=0)  # [B,D]

        x_in_t = torch.from_numpy(x_in).float().to(self.device)
        x_out_t = torch.from_numpy(x_out).float().to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        y_hat, _ = self.model(x_in_t)
        loss = self.loss_fn(y_hat, x_out_t)
        loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        self.loss_history.append(float(loss.detach().cpu().item()))
        self._update_step += 1

        if self.autosave_path and (self._update_step % self.autosave_every_updates == 0):
            self.save(self.autosave_path)

    def step(self, action):
        obs, reward, terminated, truncated, info = _normalize_step(self.env.step(action))

        x_raw = self._extract_feat_3N(info)
        x_t = self._norm(x_raw)
        self.history.append(x_t)
        self._env_step += 1

        # 生成监督样本：(history_len -> next) 放入 replay
        if len(self.history) >= self.history_len + 1:
            seq = np.stack(list(self.history)[-(self.history_len + 1):], axis=0)  # [T+1,D]
            x_in = seq[:-1].astype(np.float32)   # [T,D]
            x_out = seq[-1].astype(np.float32)   # [D]
            self.replay.append((x_in, x_out))

        # 在线微调
        self._maybe_train()

        # 预测
        if len(self.history) >= self.history_len:
            seq = np.stack(list(self.history)[-self.history_len:], axis=0)  # [T,D]
            x_in_t = torch.from_numpy(seq[None, ...]).float().to(self.device)
            self.model.eval()
            with torch.no_grad():
                y_pred, _ = self.model(x_in_t)
            pred_full_norm = y_pred.cpu().numpy()[0].astype(np.float32)
        else:
            pred_full_norm = np.zeros(self.feat_dim, dtype=np.float32)

        # pred_short = self._short_pred_from_full(pred_full_norm)
        # obs_aug = np.concatenate([obs, pred_short], axis=-1).astype(np.float32)

        # return obs_aug, reward, terminated, truncated, info
        # —— 关键：把预测从归一化空间反变换回原量纲，再拼给 RL —— #
        # 如果 scale 没加载成功，这里直接报错，避免 silent wrong behavior
        if not getattr(self, "_scale_loaded", False):
            raise RuntimeError("LSTMPredictWrapper: scale not loaded. 请确认 pretrained_path 正确且 ckpt 含 scale。")

        pred_full_raw = pred_full_norm * (self.scale.astype(np.float32) + 1e-8)

        ############################################################################
        # ??(t-1) ?? ??(t)???
        if self._pending_pred_full_raw is not None:
            self.pred_full_raw_hist.append(self._pending_pred_full_raw.copy())
            self.real_full_raw_hist.append(x_raw.astype(np.float32).copy())
        self._pending_pred_full_raw = pred_full_raw.astype(np.float32).copy()

        ############################################################################
        ############################################################################
        # use raw prediction with fixed scale for PPO input
        #pred_short_raw = self._short_pred_from_full(pred_full_norm)
        pred_short_raw = self._short_pred_from_full(pred_full_raw)
        scale = float(self.pred_scale) if self.pred_scale is not None else 1.0
        pred_short = (pred_short_raw * scale).astype(np.float32)
        obs_aug = np.concatenate([obs, pred_short], axis=-1).astype(np.float32)

        print(f"[LSTM] Step: {self._env_step},x_raw: {x_raw}")
        print(f"[LSTM] Step: {self._env_step},befro LSTM obs: {obs}")
        print(f"[LSTM] Step: {self._env_step},after LSTM obs: {obs_aug}")
        print(f"[LSTM] Step: {self._env_step},pred_full_norm: {pred_full_norm}")
        print(f"[LSTM] Step: {self._env_step},pred_scale: {self.pred_scale}")
        print(f"[LSTM] Step: {self._env_step},pred_full_raw: {pred_full_raw}")
        ############################################################################
        return obs_aug, reward, terminated, truncated, info

    def save_predictions(self, path: str = "lstm_predictions.npz"):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        pred = np.asarray(self.pred_full_raw_hist, dtype=np.float32)
        real = np.asarray(self.real_full_raw_hist, dtype=np.float32)
        np.savez(
            path,
            pred_full=pred,
            real_full=real,
            slice_types=np.asarray(self.slice_types),
            feat_dim=np.asarray(self.feat_dim, dtype=np.int32),
            alignment=np.asarray("pred_t-1_vs_real_t"),
        )
        print(f"[LSTMPredictWrapper] saved predictions: {path} (pairs={pred.shape[0]})")

