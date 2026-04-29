# wrappers/sla_dense_reward_wrapper.py
# -*- coding: utf-8 -*-
"""
SlaDenseRewardWrapper

基于 slice_ran.py 中的 SLA 判定逻辑，给每个 step 重新计算
“正/负裕度”并组合成稠密奖励。主要特性：

- 直接读取 node_b.step(...) 返回的 info["slices"]，确保与底座
  NodeB / SliceRAN 的违约判定保持一致；
- 对 eMBB 采用“流内 OR、切片内 AND”的口径：
    * CBR: (吞吐 OR PRB OR 队列) 满足即可；
    * VBR: 同上；
    * 切片满足 = CBR 满足 AND VBR 满足；
- 对 mMTC：delay / avg_rep 两个子指标分别计算裕度；
- pos_sum / neg_sum = 所有子 margin 绝对值的求和；
- reward 结构保持为：
    if any_violate:
        r = -barrier - gamma * neg_sum
    else:
        r = 1 + kappa * pos_sum - alpha * prb_cost
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Any, Dict, Tuple

# 调试开关（需要对拍时改成 True）
DEBUG = True


def _normalize_step(ret) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    """
    把 env.step(...) 的返回统一成
        (obs, reward, terminated, truncated, info)
    兼容老 Gym / Gymnasium / (obs, info) 这种形式。
    """
    if isinstance(ret, tuple):
        if len(ret) == 5:
            obs, r, terminated, truncated, info = ret
            return obs, float(r), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
        if len(ret) == 4:
            # 老 Gym: (obs, r, done, info)
            obs, r, done, info = ret
            return obs, float(r), bool(done), False, (info if isinstance(info, dict) else {})
        if len(ret) == 2:
            # (obs, info)，reward 从 info["reward"] 回退
            obs, info = ret
            base_r = 0.0
            if isinstance(info, dict):
                base_r = float(info.get("reward", 0.0))
            return obs, base_r, False, False, (info if isinstance(info, dict) else {})
    # 兜底（不太会进来）
    obs, r = ret[0], ret[1]
    return obs, float(r), False, False, {}


def _gte_margin(val: float, lower: float, eps: float = 1e-6) -> float:
    """
    约束 val >= lower 的裕度：
        m = val/lower - 1
    m >= 0 表示满足，m < 0 表示不满足，|m| 代表相对裕度。
    """
    return (float(val) / (float(lower) + eps)) - 1.0


def _lte_margin(val: float, upper: float, eps: float = 1e-6) -> float:
    """
    约束 val <= upper 的裕度：
        m = (upper - val)/upper
    m >= 0 表示满足，m < 0 表示不满足。
    """
    return (float(upper) - float(val)) / (float(upper) + eps)


class SlaDenseRewardWrapper(gym.Wrapper):
    """
    稠密奖励 wrapper（按 slice_ran 的 SLA 判定重写）：

    - 先执行 env.step(action)，利用底座仿真更新 NodeB / SliceRAN；
    - 然后读取 info["slices"][gid] 中的 per-slice 指标，对照
      每个 slice 的 SLA / 观测窗口重新计算每个子约束的 margin；
    - 所有子 margin 绝对值求和得到：
         pos_sum = sum(|m| for m>=0)
         neg_sum = sum(|m| for m<0)
    - 对违反 SLA 的 step： r = -barrier - gamma * neg_sum
      否则：             r =  1 + kappa * pos_sum - alpha * prb_cost
    """

    def __init__(
        self,
        env,
        barrier: float = 2.0,
        gamma: float = 1.0,
        kappa: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.1,
        clip_abs: float = 1.0,
    ):
        super().__init__(env)

        self.barrier = float(barrier)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.clip_abs = float(clip_abs)

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.slots_length = 1e-3
        # ---- 建立 gid -> slice meta 的映射，用于还原 margin 计算 ----
        # meta 中记录：
        #   type           : "embb" / "mmtc" / ...
        #   SLA            : 阈值字典（直接来自每个 SliceRAN 的 self.SLA）
        #   obsT           : observation_time
        #   slots_per_step : slots_per_step
        self.slice_meta: Dict[int, Dict[str, Any]] = {}
        self.n_prbs_total = None

        nb = getattr(env, "node_b", None)
        if nb is not None:
            self.n_prbs_total = getattr(nb, "n_prbs", None)
            gid = 0
            # NodeB._flatten_slices_info 中的 gid 增长顺序是：
            #   for l1 in slices_l1:
            #       for local_idx in l1.slices_ran:
            #           slices[gid] = ...
            #           gid += 1
            # 这里按同样顺序构造 meta，确保 gid 对齐。
            for l1 in getattr(nb, "slices_l1", []):
                slots_per_step = getattr(l1, "slots_per_step", None)
                for s in getattr(l1, "slices_ran", []):
                    meta = {
                        "type": getattr(s, "type", "unknown"),
                        "SLA": getattr(s, "SLA", {}),
                        "obsT": getattr(s, "observation_time", 1.0),
                        "slots_per_step": getattr(
                            s,
                            "slots_per_step",
                            slots_per_step if slots_per_step is not None else 1.0,
                        ),
                    }
                    self.slice_meta[gid] = meta
                    gid += 1

        # 这里的 meta/m 取的是最后一个 slice 的值，仅用于 debug 打印
        meta["obsT"] = meta["slots_per_step"] * self.slots_length
        if DEBUG:
            print("[SlaDense] init slice_meta:")
            for gid, m in self.slice_meta.items():
                print(f"  gid={gid}: type={m['type']}, SLA_keys={list(m['SLA'].keys())}")
            print(f"SLA DENSE slots_per_step:{m['slots_per_step']},obT:{m['obsT']}")

    # 供外部在线调参
    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def set_clip(self, clip_abs: float):
        self.clip_abs = float(clip_abs)

    def step(self, action):
        # 从底层 env 里拿 NodeB 的 remaining_prb
        nb = getattr(self.env, "node_b", None)
        rem = getattr(nb, "remaining_prb", None) if nb is not None else None
        raw = self.env.step(action)
        obs, base_r, terminated, truncated, info = _normalize_step(raw)

        slices_info = info.get("slices", None)
        if not isinstance(slices_info, dict) or not slices_info:
            # 没有 slice 详细信息就直接透传底座 reward
            return obs, base_r, terminated, truncated, info

        pos_margins = []
        neg_margins = []
        per_slice_flags = []  # 每个切片是否违约（有任一子 margin<0）

        # ---- 逐个 slice 计算 margin ----
        for gid, s_info in slices_info.items():
            if not isinstance(s_info, dict):
                # 防御：如果值不是 dict，跳过
                continue

            meta = self.slice_meta.get(gid, None)
            if meta is None:
                if DEBUG:
                    print(f"[SlaDense] unknown slice gid={gid}, keys={list(s_info.keys())}")
                continue

            s_type = str(meta.get("type", "unknown")).lower()
            SLA = meta.get("SLA", {}) or {}
            # obsT = float(meta.get("obsT", 1.0))
            obsT = meta["slots_per_step"] * self.slots_length
            slots_per_step = float(meta.get("slots_per_step", 1.0))

            local_has_neg = False

            # ---------- eMBB ---------- #
            if s_type == "embb":
                # CBR 三个子约束：吞吐 / PRB / 队列
                cbr_margins = []

                if "cbr_th" in s_info and "cbr_th" in SLA:
                    val = float(s_info["cbr_th"]) / max(obsT, 1e-6)
                    m = _gte_margin(val, SLA["cbr_th"])
                    cbr_margins.append(m)

                if "cbr_prb" in s_info and "cbr_prb" in SLA:
                    val = float(s_info["cbr_prb"]) / max(slots_per_step, 1e-6)
                    m = _gte_margin(val, SLA["cbr_prb"])
                    cbr_margins.append(m)

                if "cbr_queue" in s_info and "cbr_queue" in SLA:
                    val = float(s_info["cbr_queue"]) / max(slots_per_step, 1e-6)
                    m = _lte_margin(val, SLA["cbr_queue"])
                    cbr_margins.append(m)

                # VBR 三个子约束
                vbr_margins = []

                if "vbr_th" in s_info and "vbr_th" in SLA:
                    val = float(s_info["vbr_th"]) / max(obsT, 1e-6)
                    m = _gte_margin(val, SLA["vbr_th"])
                    vbr_margins.append(m)

                if "vbr_prb" in s_info and "vbr_prb" in SLA:
                    val = float(s_info["vbr_prb"]) / max(slots_per_step, 1e-6)
                    m = _gte_margin(val, SLA["vbr_prb"])
                    vbr_margins.append(m)

                if "vbr_queue" in s_info and "vbr_queue" in SLA:
                    val = float(s_info["vbr_queue"]) / max(slots_per_step, 1e-6)
                    m = _lte_margin(val, SLA["vbr_queue"])
                    vbr_margins.append(m)

                # 流内 OR：取各自 best margin
                flow_margins = []
                if cbr_margins:
                    flow_margins.append(max(cbr_margins))
                if vbr_margins:
                    flow_margins.append(max(vbr_margins))

                # 切片内 AND：两条流都要非负；有负则视为违约
                for m in flow_margins:
                    if m >= 0:
                        pos_margins.append(min(abs(m), self.clip_abs))
                    else:
                        neg_margins.append(min(abs(m), self.clip_abs))
                        local_has_neg = True

            # ---------- mMTC ---------- #
            elif s_type == "mmtc":
                # delay 约束：delay <= SLA_delay
                if "delay" in s_info and "delay" in SLA:
                    val = float(s_info["delay"]) / max(slots_per_step, 1e-6)
                    m = _lte_margin(val, SLA["delay"])
                    if m >= 0:
                        pos_margins.append(min(abs(m), self.clip_abs))
                    else:
                        neg_margins.append(min(abs(m), self.clip_abs))
                        local_has_neg = True

                # avg_rep 约束：avg_rep <= SLA_avg_rep（如果有的话）
                if "avg_rep" in s_info and "avg_rep" in SLA:
                    val = float(s_info["avg_rep"]) / max(slots_per_step, 1e-6)
                    m = _lte_margin(val, SLA["avg_rep"])
                    if m >= 0:
                        pos_margins.append(min(abs(m), self.clip_abs))
                    else:
                        neg_margins.append(min(abs(m), self.clip_abs))
                        local_has_neg = True

            else:
                # 其它类型暂时不做 margin，遵从底座 reward
                if DEBUG:
                    print(f"[SlaDense] skip slice gid={gid}, type={s_type}, keys={list(s_info.keys())}")

            per_slice_flags.append(1 if local_has_neg else 0)

        neg_sum = float(sum(neg_margins))
        pos_sum = float(sum(pos_margins))
        any_violate = any(per_slice_flags)

        # ---- PRB 使用率：当前动作总量 / 当前可用 PRB(remaining_prb) ----
        prb_cost = 0.0
        try:
            a = np.asarray(action, dtype=np.float32).ravel()
            tot = float(a.sum())  # 当前 step cross+now 的 PRB 总量（整数动作向量求和）

            # # 从底层 env 里拿 NodeB 的 remaining_prb
            # nb = getattr(self.env, "node_b", None)
            # rem = getattr(nb, "remaining_prb", None) if nb is not None else None

            if rem is not None and rem > 0:
                prb_cost = tot / float(rem)
            else:
                # 拿不到 remaining_prb 或为 0 时，退化为使用 tot，避免除零
                prb_cost = tot
        except Exception:
            prb_cost = 0.0

        # ---- 组合奖励：保持原有结构 ----
        if any_violate:
            r = -self.barrier * sum(per_slice_flags) - self.gamma * neg_sum + self.alpha * prb_cost
        else:
            r = self.kappa * pos_sum - self.beta * prb_cost

        dense = info.setdefault("dense_terms", {})
        dense.update(
            dict(
                any_violate=bool(any_violate),
                per_slice=per_slice_flags,
                neg_sum=neg_sum,
                pos_sum=pos_sum,
                prb_cost=float(prb_cost),
                r_dense=float(r),
            )
        )

        if DEBUG:
            nb_viol = info.get("violations", None)
            if isinstance(nb_viol, (list, tuple, np.ndarray)):
                nb_arr = np.array(nb_viol).astype(int).tolist()
            else:
                nb_arr = None
            print(
                "[SlaDense] per_slice(dense)={} node_b={} any_violate={} "
                "pos_sum={:.3f} neg_sum={:.3f} prb_cost={:.3f} r={:.3f}".format(
                    per_slice_flags,
                    nb_arr,
                    any_violate,
                    pos_sum,
                    neg_sum,
                    prb_cost,
                    r,
                )
            )

        return obs, float(r), terminated, truncated, info

# # wrappers/sla_dense_reward_wrapper.py
# # -*- coding: utf-8 -*-
# """
# SlaDenseRewardWrapper

# 基于 slice_ran.py 中的 SLA 判定逻辑，给每个 step 重新计算
# “正/负裕度”并组合成稠密奖励。主要特性：

# - 直接读取 node_b.step(...) 返回的 info["slices"]，确保与底座
#   NodeB / SliceRAN 的违约判定保持一致；
# - 对 eMBB 采用“流内 OR、切片内 AND”的口径：
#     * CBR: (吞吐 OR PRB OR 队列) 满足即可；
#     * VBR: 同上；
#     * 切片满足 = CBR 满足 AND VBR 满足；
# - 对 mMTC：delay / avg_rep 两个子指标分别计算裕度；
# - pos_sum / neg_sum = 所有子 margin 绝对值的求和；
# - reward 结构保持为：
#     if any_violate:
#         r = -barrier - gamma * neg_sum
#     else:
#         r = 1 + kappa * pos_sum - alpha * prb_cost
# """

# from __future__ import annotations

# import numpy as np
# import gymnasium as gym
# from typing import Any, Dict, Tuple

# # 调试开关（需要对拍时改成 True）
# DEBUG = True


# def _normalize_step(ret) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
#     """
#     把 env.step(...) 的返回统一成
#         (obs, reward, terminated, truncated, info)
#     兼容老 Gym / Gymnasium / (obs, info) 这种形式。
#     """
#     if isinstance(ret, tuple):
#         if len(ret) == 5:
#             obs, r, terminated, truncated, info = ret
#             return obs, float(r), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
#         if len(ret) == 4:
#             # 老 Gym: (obs, r, done, info)
#             obs, r, done, info = ret
#             return obs, float(r), bool(done), False, (info if isinstance(info, dict) else {})
#         if len(ret) == 2:
#             # (obs, info)，reward 从 info["reward"] 回退
#             obs, info = ret
#             base_r = 0.0
#             if isinstance(info, dict):
#                 base_r = float(info.get("reward", 0.0))
#             return obs, base_r, False, False, (info if isinstance(info, dict) else {})
#     # 兜底（不太会进来）
#     obs, r = ret[0], ret[1]
#     return obs, float(r), False, False, {}


# def _gte_margin(val: float, lower: float, eps: float = 1e-6) -> float:
#     """
#     约束 val >= lower 的裕度：
#         m = val/lower - 1
#     m >= 0 表示满足，m < 0 表示不满足，|m| 代表相对裕度。
#     """
#     return (float(val) / (float(lower) + eps)) - 1.0


# def _lte_margin(val: float, upper: float, eps: float = 1e-6) -> float:
#     """
#     约束 val <= upper 的裕度：
#         m = (upper - val)/upper
#     m >= 0 表示满足，m < 0 表示不满足。
#     """
#     return (float(upper) - float(val)) / (float(upper) + eps)


# class SlaDenseRewardWrapper(gym.Wrapper):
#     """
#     稠密奖励 wrapper（按 slice_ran 的 SLA 判定重写）：

#     - 先执行 env.step(action)，利用底座仿真更新 NodeB / SliceRAN；
#     - 然后读取 info["slices"][gid] 中的 per-slice 指标，对照
#       每个 slice 的 SLA / 观测窗口重新计算每个子约束的 margin；
#     - 所有子 margin 绝对值求和得到：
#          pos_sum = sum(|m| for m>=0)
#          neg_sum = sum(|m| for m<0)
#     - 对违反 SLA 的 step： r = -barrier - gamma * neg_sum
#       否则：             r =  1 + kappa * pos_sum - alpha * prb_cost
#     """

#     def __init__(
#         self,
#         env,
#         barrier: float = 2.0,
#         gamma: float = 1.0,
#         kappa: float = 0.5,
#         alpha: float = 0.1,
#         beta: float = 0.1,
#         clip_abs: float = 1.0,
#     ):
#         super().__init__(env)

#         self.barrier = float(barrier)
#         self.gamma = float(gamma)
#         self.kappa = float(kappa)
#         self.alpha = float(alpha)
#         self.beta = float(beta)
#         self.clip_abs = float(clip_abs)

#         self.observation_space = env.observation_space
#         self.action_space = env.action_space
#         self.slots_length = 1e-3
#         # ---- 建立 gid -> slice meta 的映射，用于还原 margin 计算 ----
#         # meta 中记录：
#         #   type           : "embb" / "mmtc" / ...
#         #   SLA            : 阈值字典（直接来自每个 SliceRAN 的 self.SLA）
#         #   obsT           : observation_time
#         #   slots_per_step : slots_per_step
#         self.slice_meta: Dict[int, Dict[str, Any]] = {}
#         self.n_prbs_total = None

#         nb = getattr(env, "node_b", None)
#         if nb is not None:
#             self.n_prbs_total = getattr(nb, "n_prbs", None)
#             gid = 0
#             # NodeB._flatten_slices_info 中的 gid 增长顺序是：
#             #   for l1 in slices_l1:
#             #       for local_idx in l1.slices_ran:
#             #           slices[gid] = ...
#             #           gid += 1
#             # 这里按同样顺序构造 meta，确保 gid 对齐。
#             for l1 in getattr(nb, "slices_l1", []):
#                 slots_per_step = getattr(l1, "slots_per_step", None)
#                 for s in getattr(l1, "slices_ran", []):
#                     meta = {
#                         "type": getattr(s, "type", "unknown"),
#                         "SLA": getattr(s, "SLA", {}),
#                         "obsT": getattr(s, "observation_time", 1.0),
#                         "slots_per_step": getattr(
#                             s,
#                             "slots_per_step",
#                             slots_per_step if slots_per_step is not None else 1.0,
#                         ),
#                     }
#                     self.slice_meta[gid] = meta
#                     gid += 1

#         meta["obsT"] = meta["slots_per_step"]*self.slots_length
#         if DEBUG:
#             print("[SlaDense] init slice_meta:")
#             for gid, m in self.slice_meta.items():
#                 print(f"  gid={gid}: type={m['type']}, SLA_keys={list(m['SLA'].keys())}")
#             print(f"slots_per_step:{m['slots_per_step']},obT:{m['obsT']}")

#     # 供外部在线调参
#     def set_alpha(self, alpha: float):
#         self.alpha = float(alpha)

#     def set_clip(self, clip_abs: float):
#         self.clip_abs = float(clip_abs)

#     def step(self, action):
#         raw = self.env.step(action)
#         obs, base_r, terminated, truncated, info = _normalize_step(raw)

#         slices_info = info.get("slices", None)
#         if not isinstance(slices_info, dict) or not slices_info:
#             # 没有 slice 详细信息就直接透传底座 reward
#             return obs, base_r, terminated, truncated, info

#         pos_margins = []
#         neg_margins = []
#         per_slice_flags = []  # 每个切片是否违约（有任一子 margin<0）

#         # ---- 逐个 slice 计算 margin ----
#         for gid, s_info in slices_info.items():
#             if not isinstance(s_info, dict):
#                 # 防御：如果值不是 dict，跳过
#                 continue

#             meta = self.slice_meta.get(gid, None)
#             if meta is None:
#                 if DEBUG:
#                     print(f"[SlaDense] unknown slice gid={gid}, keys={list(s_info.keys())}")
#                 continue

#             s_type = str(meta.get("type", "unknown")).lower()
#             SLA = meta.get("SLA", {}) or {}
#             # obsT = float(meta.get("obsT", 1.0))
#             obsT = meta["slots_per_step"]*self.slots_length
#             slots_per_step = float(meta.get("slots_per_step", 1.0))

#             local_has_neg = False

#             # ---------- eMBB ---------- #
#             if s_type == "embb":
#                 # CBR 三个子约束：吞吐 / PRB / 队列
#                 cbr_margins = []

#                 if "cbr_th" in s_info and "cbr_th" in SLA:
#                     val = float(s_info["cbr_th"]) / max(obsT, 1e-6)
#                     m = _gte_margin(val, SLA["cbr_th"])
#                     cbr_margins.append(m)

#                 if "cbr_prb" in s_info and "cbr_prb" in SLA:
#                     val = float(s_info["cbr_prb"]) / max(slots_per_step, 1e-6)
#                     m = _gte_margin(val, SLA["cbr_prb"])
#                     cbr_margins.append(m)

#                 if "cbr_queue" in s_info and "cbr_queue" in SLA:
#                     val = float(s_info["cbr_queue"]) / max(slots_per_step, 1e-6)
#                     m = _lte_margin(val, SLA["cbr_queue"])
#                     cbr_margins.append(m)

#                 # VBR 三个子约束
#                 vbr_margins = []

#                 if "vbr_th" in s_info and "vbr_th" in SLA:
#                     val = float(s_info["vbr_th"]) / max(obsT, 1e-6)
#                     m = _gte_margin(val, SLA["vbr_th"])
#                     vbr_margins.append(m)

#                 if "vbr_prb" in s_info and "vbr_prb" in SLA:
#                     val = float(s_info["vbr_prb"]) / max(slots_per_step, 1e-6)
#                     m = _gte_margin(val, SLA["vbr_prb"])
#                     vbr_margins.append(m)

#                 if "vbr_queue" in s_info and "vbr_queue" in SLA:
#                     val = float(s_info["vbr_queue"]) / max(slots_per_step, 1e-6)
#                     m = _lte_margin(val, SLA["vbr_queue"])
#                     vbr_margins.append(m)

#                 # 流内 OR：取各自 best margin
#                 flow_margins = []
#                 if cbr_margins:
#                     flow_margins.append(max(cbr_margins))
#                 if vbr_margins:
#                     flow_margins.append(max(vbr_margins))

#                 # 切片内 AND：两条流都要非负；有负则视为违约
#                 for m in flow_margins:
#                     if m >= 0:
#                         pos_margins.append(min(abs(m), self.clip_abs))
#                     else:
#                         neg_margins.append(min(abs(m), self.clip_abs))
#                         local_has_neg = True

#             # ---------- mMTC ---------- #
#             elif s_type == "mmtc":
#                 # delay 约束：delay <= SLA_delay
#                 if "delay" in s_info and "delay" in SLA:
#                     val = float(s_info["delay"]) / max(slots_per_step, 1e-6)
#                     m = _lte_margin(val, SLA["delay"])
#                     if m >= 0:
#                         pos_margins.append(min(abs(m), self.clip_abs))
#                     else:
#                         neg_margins.append(min(abs(m), self.clip_abs))
#                         local_has_neg = True

#                 # avg_rep 约束：avg_rep <= SLA_avg_rep（如果有的话）
#                 if "avg_rep" in s_info and "avg_rep" in SLA:
#                     val = float(s_info["avg_rep"]) / max(slots_per_step, 1e-6)
#                     m = _lte_margin(val, SLA["avg_rep"])
#                     if m >= 0:
#                         pos_margins.append(min(abs(m), self.clip_abs))
#                     else:
#                         neg_margins.append(min(abs(m), self.clip_abs))
#                         local_has_neg = True

#             else:
#                 # 其它类型暂时不做 margin，遵从底座 reward
#                 if DEBUG:
#                     print(f"[SlaDense] skip slice gid={gid}, type={s_type}, keys={list(s_info.keys())}")

#             per_slice_flags.append(1 if local_has_neg else 0)

#         neg_sum = float(sum(neg_margins))
#         pos_sum = float(sum(pos_margins))
#         # any_violate = (neg_sum > 0.0) or any(per_slice_flags)
#         any_violate = any(per_slice_flags)

#         # ---- PRB 使用率（或动作和） ----
#         prb_cost = 0.0
#         try:
#             a = np.asarray(action, dtype=np.float32).ravel()
#             tot = float(a.sum())
#             if self.n_prbs_total and self.n_prbs_total > 0:
#                 prb_cost = tot / float(self.n_prbs_total)
#             else:
#                 prb_cost = tot
#         except Exception:
#             prb_cost = 0.0

#         # ---- 组合奖励：保持原有结构 ----
#         # if any_violate:
#         #     r = -self.barrier - self.gamma * neg_sum + self.alpha * prb_cost
#         # else:
#         #     r = 3.0 + self.kappa * pos_sum - self.alpha * prb_cost
#         if any_violate:
#             r =  -self.barrier * sum(per_slice_flags) - self.gamma * neg_sum + self.alpha * prb_cost
#         else:
#             r =  self.kappa * pos_sum - self.beta * prb_cost
#         dense = info.setdefault("dense_terms", {})
#         dense.update(
#             dict(
#                 any_violate=bool(any_violate),
#                 per_slice=per_slice_flags,
#                 neg_sum=neg_sum,
#                 pos_sum=pos_sum,
#                 prb_cost=float(prb_cost),
#                 r_dense=float(r),
#             )
#         )

#         if DEBUG:
#             nb_viol = info.get("violations", None)
#             if isinstance(nb_viol, (list, tuple, np.ndarray)):
#                 nb_arr = np.array(nb_viol).astype(int).tolist()
#             else:
#                 nb_arr = None
#             print(
#                 "[SlaDense] per_slice(dense)={} node_b={} any_violate={} "
#                 "pos_sum={:.3f} neg_sum={:.3f} prb_cost={:.3f} r={:.3f}".format(
#                     per_slice_flags,
#                     nb_arr,
#                     any_violate,
#                     pos_sum,
#                     neg_sum,
#                     prb_cost,
#                     r,
#                 )
#             )

#         return obs, float(r), terminated, truncated, info



