# wrappers/priority_reward_wrapper.py
# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym

def _normalize_step(ret):
    if isinstance(ret, tuple):
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return obs, float(reward), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
        if len(ret) == 4:
            obs, reward, done, info = ret
            return obs, float(reward), bool(done), False, (info if isinstance(info, dict) else {})
    # 兜底：旧式返回
    obs, reward = ret[0], ret[1]
    return obs, float(reward), False, False, {}

class PriorityRewardWrapper(gym.Wrapper):
    """
    优先级惩罚（与 SlaDense 的“eMBB 子SLA=CBR/VBR 分别评估，口径为 OR”一致）：
      - eMBB：weighted += w_cbr * viol_cbr + w_vbr * viol_vbr
               （若两者都违规，则两次计罚）
      - mMTC：weighted += w_mmtc * viol
    最终： reward -= lam * min(weighted, cap)

    weights 示例：
      {"mmtc": 3.0, "embb_cbr": 2.0, "embb_vbr": 1.0, "embb": 1.5}
    也支持对单个切片 idx 额外乘子：若 weights 里有整数键 idx，则作为乘子参与计算。
    """
    def __init__(self, env, weights=None, lam: float = 1.0, cap: float = 10.0):
        super().__init__(env)
        self.weights = weights or {"mmtc": 3.0, "embb_cbr": 2.0, "embb_vbr": 1.0, "embb": 1.5}
        self.lam = float(lam)
        self.cap = float(cap)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    # 在线调参
    def set_lambda(self, lam: float): self.lam = float(lam)
    def set_cap(self, cap: float): self.cap = float(cap)

    def _iter_slices(self, info):
        # NodeB.get_info() 中：info["slices"] 为 {global_idx: slice_info, ...}
        if isinstance(info, dict) and isinstance(info.get("slices"), dict):
            for gid in sorted(info["slices"].keys()):
                d = info["slices"][gid] if isinstance(info["slices"][gid], dict) else {}
                yield gid, d

    def step(self, action):
        obs, reward, terminated, truncated, info = _normalize_step(self.env.step(action))

        weighted = 0.0
        prio_mmtc = 0.0
        prio_embb_cbr = 0.0
        prio_embb_vbr = 0.0

        for idx, si in self._iter_slices(info):
            t = str(si.get("type", "")).lower()
            idx_factor = float(self.weights.get(idx, 1.0))  # 可选：对指定 idx 额外乘权

            if t == "embb":
                # 与 SlaDense 的分项口径一致：分别看两条子SLA
                vc = int(si.get("viol_cbr", 0))
                vv = int(si.get("viol_vbr", 0))
                w_cbr = float(self.weights.get("embb_cbr", self.weights.get("embb", 1.0))) * idx_factor
                w_vbr = float(self.weights.get("embb_vbr", self.weights.get("embb", 1.0))) * idx_factor

                if vc > 0:
                    weighted += w_cbr * vc
                    prio_embb_cbr += w_cbr * vc
                if vv > 0:
                    weighted += w_vbr * vv
                    prio_embb_vbr += w_vbr * vv

            elif t == "mmtc":
                v = int(si.get("viol", 0))
                w_m = float(self.weights.get("mmtc", 1.0)) * idx_factor
                if v > 0:
                    weighted += w_m * v
                    prio_mmtc += w_m * v

            else:
                # 未知类型：如果希望仍然惩罚，可用通用 embb 权重或 idx 权重
                v = int(si.get("viol", 0))
                if v > 0:
                    weighted += idx_factor * v

        weighted_capped = min(float(weighted), self.cap)
        reward = float(reward) - self.lam * weighted_capped

        # 写回分解项（便于 ReportWrapper 保存与可视化）
        if isinstance(info, dict):
            pt = info.setdefault("priority_terms", {})
            pt.update({
                "weighted": float(weighted),
                "weighted_capped": float(weighted_capped),
                "mmtc": float(prio_mmtc),
                "embb_cbr": float(prio_embb_cbr),
                "embb_vbr": float(prio_embb_vbr),
                "lam": float(self.lam),
                "cap": float(self.cap),
            })

        return obs, reward, terminated, truncated, info


