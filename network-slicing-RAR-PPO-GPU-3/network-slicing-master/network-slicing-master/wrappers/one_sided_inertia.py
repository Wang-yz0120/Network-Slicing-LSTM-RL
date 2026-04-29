# wrappers/one_sided_inertia.py
# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym


# -------------------- helpers -------------------- #
def _normalize_reset(ret):
    if isinstance(ret, tuple) and len(ret) == 2:
        obs, info = ret
        return obs, (info if isinstance(info, dict) else {})
    return ret, {}


def _normalize_step(ret):
    if isinstance(ret, tuple):
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return obs, float(reward), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
        if len(ret) == 4:
            obs, reward, done, info = ret
            return obs, float(reward), bool(done), False, (info if isinstance(info, dict) else {})
    obs, reward = ret[0], ret[1]
    return obs, float(reward), False, False, {}


# ====================== OneSidedInertia ====================== #
class OneSidedInertia(gym.Wrapper):
    """
    单边惯性惩罚：惩罚动作比上一步“向下跳”的幅度（例如减少资源）。
    兼容 Gym/Gymnasium（step 返回 5 元组）。
    """
    def __init__(self, env, mu=0.05):
        super().__init__(env)
        self.mu = float(mu)
        self.prev_action = None

        # 透传 space
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        self.prev_action = None
        obs, info = _normalize_reset(self.env.reset(seed=seed, options=options))
        return obs, info

    def step(self, action):
        # 先与底层交互
        obs, reward, terminated, truncated, info = _normalize_step(self.env.step(action))

        # 单边惯性：只惩罚下降（pa - a > 0）
        try:
            a = np.asarray(action, dtype=np.float32).ravel()
            pa = np.asarray(self.prev_action if self.prev_action is not None else a, dtype=np.float32).ravel()
            inertia_penalty = np.maximum(0.0, pa - a).sum()
            reward = float(reward) - self.mu * float(inertia_penalty)
        except Exception:
            # 若动作不是数值型或形状异常，则忽略惩罚
            pass

        self.prev_action = np.asarray(action, dtype=np.float32)

        return obs, reward, terminated, truncated, info

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import numpy as np
# import gymnasium as gym

# class OneSidedInertia(gym.Wrapper):
#     """
#     只惩罚“骤降”的动作，鼓励“提前多给、慢慢降”：
#       r'' = r' - μ * ||max(0, a_{t-1} - a_t)||_1
#     """
#     def __init__(self, env, mu=0.05):
#         super().__init__(env)
#         self.mu = float(mu)
#         self.prev_action = None

#     def reset(self, **kwargs):
#         self.prev_action = None
#         return self.env.reset(**kwargs)

#     def step(self, action):
#         obs, r, done, info = self.env.step(action)
#         if self.prev_action is not None:
#             pa = np.asarray(self.prev_action, dtype=np.float32).ravel()
#             a  = np.asarray(action, dtype=np.float32).ravel()
#             down = np.maximum(0.0, pa - a)
#             r = float(r) - self.mu * float(np.sum(down))
#         self.prev_action = action
#         return obs, r, done, info
