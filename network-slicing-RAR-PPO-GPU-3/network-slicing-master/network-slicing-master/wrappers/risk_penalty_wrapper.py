# wrappers/risk_penalty_wrapper.py
# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym


# -------------------- helpers: unify Gym/Gymnasium -------------------- #
def _normalize_reset(ret):
    """
    Gymnasium: (obs, info)
    Old Gym:   obs
    """
    if isinstance(ret, tuple) and len(ret) == 2:
        obs, info = ret
        return obs, (info if isinstance(info, dict) else {})
    return ret, {}


def _normalize_step(ret):
    """
    Return (obs, reward, terminated, truncated, info) in all cases.
    """
    if isinstance(ret, tuple):
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return obs, float(reward), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
        if len(ret) == 4:
            obs, reward, done, info = ret
            return obs, float(reward), bool(done), False, (info if isinstance(info, dict) else {})
    # fallback (should not happen)
    obs, reward = ret[0], ret[1]
    return obs, float(reward), False, False, {}


def _space_dim(space):
    """
    Try to infer a flat dimension for a Box/Discrete action/observation space.
    """
    if hasattr(space, "shape") and space.shape is not None and len(space.shape) > 0:
        return int(np.prod(space.shape))
    if hasattr(space, "n") and space.n is not None:
        return int(space.n)
    # unknown: treat as 1
    return 1


# ====================== RiskPenaltyWrapper ====================== #
class RiskPenaltyWrapper(gym.Wrapper):
    """
    对奖励加入风险惩罚项（线性打分超阈值时惩罚）。
    - 兼容 Gym/Gymnasium：reset 返回 (obs, info)，step 返回 5 元组。
    - 关键修复：prev_obs 始终保存为 ndarray，而不是 (obs, info)。

    Args:
        lam (float): 风险权重 λ
        tau (float): 风险阈值 τ
        lr  (float): 线性模型学习率
        l2  (float): L2 衰减系数
    """
    def __init__(self, env, lam=0.2, tau=0.05, lr=5e-4, l2=1e-6):
        super().__init__(env)

        self.lam = float(lam)
        self.tau = float(tau)
        self.lr = float(lr)
        self.l2 = float(l2)

        # 推断特征维度：obs_dim + act_dim
        obs_dim = _space_dim(env.observation_space)
        act_dim = _space_dim(env.action_space)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # 线性模型参数
        self.w = np.zeros(self.obs_dim + self.act_dim, dtype=np.float32)

        # 历史观测（ndarray）
        self.prev_obs = None

        # 透传 space
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    # ---------- features ---------- #
    def _features(self, obs_prev, action):
        """
        将前一时刻 obs 与当前 action 展平后拼接成一维特征。
        obs_prev: ndarray-like
        action:   标量或向量
        """
        o = np.asarray(obs_prev, dtype=np.float32).ravel()
        a = np.asarray(action, dtype=np.float32).ravel()
        # 如离散动作，长度可能为 1，不做 one-hot（原实现即拼接）
        return np.concatenate([o, a], axis=0)

    # ---------- API ---------- #
    def reset(self, *, seed=None, options=None):
        obs, info = _normalize_reset(self.env.reset(seed=seed, options=options))
        # 只保存 obs（ndarray），避免把 (obs, info) 元组塞进来
        self.prev_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        # 用 prev_obs + 当前 action 计算风险特征
        if self.prev_obs is None:
            # 在极少数外层先 step 的情况下，先强制 reset 一下
            obs0, _ = _normalize_reset(self.env.reset())
            self.prev_obs = np.asarray(obs0, dtype=np.float32)

        x = self._features(self.prev_obs, action)  # 这里 prev_obs 必为 ndarray

        # 先与底层交互
        obs, reward, terminated, truncated, info = _normalize_step(self.env.step(action))

        # 线性打分 + hinge 风险惩罚（示例：max(0, y_hat - tau)）
        y_hat = float(np.dot(self.w, x))
        penalty = max(0.0, y_hat - self.tau)
        reward = float(reward) - self.lam * penalty

        # 线性模型一次 SGD（可按你原公式替换）
        grad = (1.0 if y_hat > self.tau else 0.0)  # d max(0, y_hat - tau)/dy_hat
        self.w = (1.0 - self.l2) * self.w - self.lr * grad * x

        # 更新 prev_obs（务必是数组）
        self.prev_obs = np.asarray(obs, dtype=np.float32)

        return obs, reward, terminated, truncated, info

