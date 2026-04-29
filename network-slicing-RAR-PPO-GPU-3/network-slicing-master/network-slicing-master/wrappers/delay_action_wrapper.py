# wrappers/delay_action_wrapper.py
# -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
from typing import Any, Dict, Tuple
from collections import deque


def _get_base_env(env, max_depth: int = 20):
    cur = env
    for _ in range(max_depth):
        if isinstance(cur, gym.Wrapper):
            cur = cur.env
        else:
            break
    return getattr(cur, "unwrapped", cur)


def _get_node_b_from_env(env):
    base = _get_base_env(env)
    return getattr(base, "node_b", None)


def _normalize_step(ret) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    # Normalize env.step(...) to (obs, reward, terminated, truncated, info).
    if isinstance(ret, tuple):
        if len(ret) == 5:
            obs, r, terminated, truncated, info = ret
            return obs, float(r), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
        if len(ret) == 4:
            obs, r, done, info = ret
            return obs, float(r), bool(done), False, (info if isinstance(info, dict) else {})
    obs, r = ret[0], ret[1]
    return obs, float(r), False, False, {}


class DelayActionWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        delay_steps: int = 1,
        default_action: np.ndarray | None = None,
        debug: bool = True,
    ):
        super().__init__(env)
        self.delay_steps = max(1, int(delay_steps))
        self._pending_actions = deque()
        self._default_action = default_action
        self.debug = bool(debug)

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        if self._default_action is None:
            self._default_action = self._build_default_action()
        self._default_action = np.asarray(self._default_action).copy()

    def _build_default_action(self) -> np.ndarray:
        # Assumes action shape = (2*n_slices + 1,)
        # Use 0 for cross-step, equal split for current-step, 0 for reserve.
        nb = _get_node_b_from_env(self.env)
        n_slices = None
        if hasattr(self.env, "n_slices"):
            n_slices = int(getattr(self.env, "n_slices"))
        elif nb is not None and hasattr(nb, "n_slices_l1"):
            n_slices = int(nb.n_slices_l1)
        else:
            # Fallback: infer from action_space
            n_slices = max(1, (self.action_space.shape[0] - 1) // 2)

        n_prbs = None
        if hasattr(self.env, "n_prbs"):
            n_prbs = int(getattr(self.env, "n_prbs"))
        elif nb is not None and hasattr(nb, "n_prbs"):
            n_prbs = int(nb.n_prbs)
        else:
            n_prbs = 0

        dtype = np.float32
        if hasattr(self.action_space, "dtype") and self.action_space.dtype is not None:
            dtype = self.action_space.dtype
        a = np.zeros((2 * n_slices + 1,), dtype=dtype)
        if n_slices > 0 and n_prbs > 0:
            per = float(n_prbs) / float(n_slices)
            a[n_slices:2 * n_slices] = per
        return a

    def reset(self, **kwargs):
        self._pending_actions.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        # Apply delayed action and buffer current raw action
        raw_action = np.asarray(action).copy()
        if len(self._pending_actions) < self.delay_steps:
            applied_action = self._default_action.copy()
        else:
            applied_action = self._pending_actions.popleft()
        self._pending_actions.append(raw_action)

        ret = self.env.step(applied_action)
        obs, reward, terminated, truncated, info = _normalize_step(ret)

        if not isinstance(info, dict):
            info = {}
        info["delayed_raw_action"] = raw_action.copy()
        info["delayed_applied_action"] = np.asarray(applied_action).copy()

        if self.debug:
            print(
                f"[DelayActionWrapper] raw_action={raw_action}, "
                f"applied_action={applied_action}, delay_steps={self.delay_steps}"
            )

        return obs, reward, terminated, truncated, info
