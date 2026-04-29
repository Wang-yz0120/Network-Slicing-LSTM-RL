#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrappers for the slice environment (Gymnasium compatible)

ReportWrapper
DQNWrapper
TimerWrapper
"""
import os
import time
from itertools import product
import numpy as np
import gymnasium as gym
from gymnasium import spaces

PENALTY = 1000
SLICES = 5

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


def _dig_attr(obj, candidates):
    for dotted in candidates:
        cur = obj
        ok = True
        for part in dotted.split("."):
            try:
                cur = object.__getattribute__(cur, part)
            except AttributeError:
                ok = False
                break
        if ok:
            return cur
    return None

def _normalize_reset(ret):
    if isinstance(ret, tuple) and len(ret) == 2:
        obs, info = ret
        return obs, (info if isinstance(info, dict) else {})
    return ret, {}

def _normalize_step(ret):
    #print("ret type/len:", type(ret), (len(ret) if isinstance(ret, tuple) else None))
    if isinstance(ret, tuple):
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return obs, float(reward), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})
        if len(ret) == 4:
            obs, reward, done, info = ret
            return obs, float(reward), bool(done), False, (info if isinstance(info, dict) else {})
    obs, reward = ret[0], ret[1]
    return obs, float(reward), False, False, {}

# ============================== ReportWrapper ============================== #

class ReportWrapper(gym.Wrapper):
    """
    记录历史 + 兼容 gym/gymnasium 接口。
    这版“列表记账”，不会因为步数上限或阶段切换丢数据；
    资源记录为“当前 step 系统实际占用的 PRB 数”，
    = 本 step now 部分分配的 PRB + 所有 lease(跨 step 占用) 中 remain>0 的 PRB。
    缺键统一写 NaN，避免被误绘成 0。
    """
    def __init__(
        self,
        env,
        steps: int = 2000,            # 仅用于进度控制/预估；不会截断
        control_steps: int = 500,
        env_id: int = 1,
        extra_samples: int = 10,
        path: str = "./logs/",
        verbose: bool = False,
        n_prbs: int | None = None,
        use_cross_step: bool = True,
        debug: bool = False,
    ):
        super().__init__(env)
        self.use_cross_step = bool(use_cross_step)
        self.debug = bool(debug)

        # Normalize observation to match training scale
        n_slices = _dig_attr(self.env, ["n_slices", "env.n_slices"])
        if n_slices is None and hasattr(self.env, "action_space"):
            a = self.env.action_space
            if hasattr(a, "n") and a.n is not None:
                n_slices = int(a.n)
            elif hasattr(a, "shape") and a.shape:
                n_slices = int(a.shape[0])
        if n_slices is None:
            raise AttributeError("ReportWrapper: cannot infer n_slices; expose env.n_slices or action_space.")
        self.n_slices = int(n_slices)

        ####################################################################################################
        # Normalize observation to match training scale
        n_vars = None
        if hasattr(self.env, "observation_space") and getattr(self.env.observation_space, "shape", None):
            n_vars = int(self.env.observation_space.shape[0])

        # Normalize observation to match training scale
        if n_vars is None:
            n_vars = _dig_attr(self.env, ["n_variables", "env.n_variables"])

        if n_vars is None:
            raise AttributeError("ReportWrapper: cannot infer n_variables.")

        self.n_variables = int(n_vars)
        ####################################################################################################
        if n_prbs is not None:
            self.n_prbs = int(n_prbs)
        else:
            guess = _dig_attr(
                self.env,
                ["n_prbs","n_PRBs","n_prb","N_PRB","node_b.n_prbs","env.n_prbs","env.node_b.n_prbs"],
            )
            self.n_prbs = int(guess) if guess is not None else None

        # Normalize observation to match training scale
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2*self.n_slices + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_variables + 1,), dtype=np.float32)

        self.steps = int(steps)
        self.control_steps = int(control_steps)
        self.env_id = env_id
        self.verbose = verbose

        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.file_path = os.path.join(self.path, f"history_{env_id}.npz")
        self.extra_samples = int(extra_samples)

        self.t = 0
        self.obs = None

        # Normalize observation to match training scale
        self.hist = {
            "violation":      [],
            "reward":         [],
            "resources":      [],
            "prio_weighted":  [],
            "prio_mmtc":      [],
            "prio_embb_cbr":  [],
            "prio_embb_vbr":  [],
        }

        # =======================
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # =======================
        self.hist["exog_vec"] = []

        ##############################################################################################
        if self.debug:
            print(f"[DEBUG] ReportWrapper init: n_prbs={self.n_prbs}, n_slices={self.n_slices}, n_variables={self.n_variables}")

    def reset(self, *, seed=None, options=None):
        obs, info = _normalize_reset(self.env.reset(seed=seed, options=options))
        obs = np.clip(obs, -0.5, 1.5) - 0.5
        self.obs = obs
        return obs, info

    def _finalize_action(self, action):
        """
        连续/比例动作 -> 整数 PRB 向量（长度 = 2*n_slices，先 cross 后 now）
        口径：动作维度 = 2*n_slices + 1，最后一维为“未使用份额(null-bin)”
              可用总量 N_avail 优先取 env.node_b.remaining_prb，拿不到则用 n_prbs
              实际使用总量 N_use = floor( N_avail * sum(front 2S) / (sum(front 2S) + null_bin) )
              先把 N_use 按 cross/now 的总权重比例切为 N_cross/N_now，
              再各自用最大剩余法在每个组内做整数分配，保证两组分别保和。
        """
        # Normalize observation to match training scale
        nS = int(self.n_slices)
        # Finalize action before stepping env
        N_avail = getattr(self, "_peek_remaining_prb", None)
        if N_avail is None:
            N_avail = _dig_attr(self.env, [
                "node_b.remaining_prb", "env.node_b.remaining_prb",
                "remaining_prb", "env.remaining_prb"
            ])
        if N_avail is None:
            N_avail = _dig_attr(self.env, [
                "n_prbs","n_PRBs","n_prb","N_PRB","env.n_prbs","env.node_b.n_prbs"
            ])
        N_avail = int(N_avail) if N_avail is not None else int(self.n_prbs)
        if self.debug:
            print(f"[WRAPPER]:N_avail: {N_avail}")
        eps = 1e-8

        # Normalize observation to match training scale
        try:
            _ = len(action)
        except Exception:
            return int(action)

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        # Normalize observation to match training scale
        if a.size < 2*nS + 1:
        # Normalize observation to match training scale
            pad = 2*nS + 1 - a.size
            a = np.concatenate([a, np.zeros((pad,), dtype=np.float64)], axis=0)
        elif a.size > 2*nS + 1:
            a = a[:2*nS + 1]

        # Normalize observation to match training scale
        w_cross = np.maximum(0.0, a[:nS])
        w_now   = np.maximum(0.0, a[nS:2*nS])
        w_null  = float(max(0.0, a[2*nS]))

        sum_cross = float(w_cross.sum())
        sum_now   = float(w_now.sum())
        sum_w     = sum_cross + sum_now

        if (sum_w + w_null) <= eps or N_avail <= 0:
            return np.zeros((2*nS,), dtype=np.int64)

        N_use = int(np.floor(N_avail * (sum_w / (sum_w + w_null + eps))))
        if N_use <= 0:
            return np.zeros((2*nS,), dtype=np.int64)

        # Normalize observation to match training scale
        N_cross = int(np.floor(N_use * (sum_cross / (sum_w + eps))))
        N_now   = int(N_use - N_cross)

        # Normalize observation to match training scale
        def alloc_by_weights(w, total):
            if total <= 0 or w.sum() <= eps:
                return np.zeros_like(w, dtype=np.int64)
            raw  = (w / (float(w.sum()) + eps)) * float(total)
            base = np.floor(raw).astype(np.int64)
            rem  = int(total - int(base.sum()))
            if rem > 0:
                frac  = raw - base
                order = np.argsort(-frac)  # 小数部分大的先补
                for i in range(rem):
                    base[order[i % len(base)]] += 1
            return base

        if self.use_cross_step:
            a_cross = alloc_by_weights(w_cross, N_cross)
            a_now   = alloc_by_weights(w_now,   N_now)
        else:
            # Fair no-cross-step ablation:
            # merge cross-step intent into current-step allocation before rounding,
            # so total used PRBs remain unchanged while future reservation is disabled.
            merged_now = w_now + w_cross
            a_cross = np.zeros((nS,), dtype=np.int64)
            a_now   = alloc_by_weights(merged_now, N_use)

        # Build final integer action (cross + now)
        final_action = np.concatenate([a_cross, a_now], axis=0).astype(np.int64)
        return final_action

    def step(self, action):
        # Use a peeked remaining_prb for current-step action sizing (no state change)
        nb = _get_node_b_from_env(self.env)
        if nb is not None and hasattr(nb, "peek_remaining_prb_next_step"):
            self._peek_remaining_prb = nb.peek_remaining_prb_next_step()
            if self.debug:
                print(f"step:{self.t},[WRAPPER]: peek_remaining_prb:{self._peek_remaining_prb}")
        else:
            self._peek_remaining_prb = None
        # Finalize action before stepping env
        if self.debug:
            print(f"step:{self.t},[WRAPPER]: original action:{action}")
        final_action = self._finalize_action(action)
        obs, reward, terminated, truncated, info = _normalize_step(self.env.step(final_action))
        if self.debug:
            print(f"step:{self.t},[WRAPPER]: obs:{obs}")
            print(f"step: {self.t},[WRAPPER]: allocated prbs:{final_action}, reward:{reward}, violations: {info['violations']}")
        # Normalize observation to match training scale
        obs = np.clip(obs, -0.5, 1.5) - 0.5
        self.obs = obs

        # Resource usage: current-step PRB + active cross-step leases
        # (leases counted when remain>0)
        try:
            a_arr_int = np.asarray(final_action, dtype=np.int64).ravel()
        except Exception:
            a_arr_int = np.asarray([int(final_action)], dtype=np.int64)

        nS = self.n_slices

        # Normalize observation to match training scale
        if a_arr_int.size >= 2 * nS:
            now_use = int(a_arr_int[nS:2 * nS].sum())
        else:
        # Normalize observation to match training scale
            now_use = int(a_arr_int.sum())

        # Finalize action before stepping env
        cross_use = 0
        nb = _get_node_b_from_env(self.env)
        if nb is not None:
            try:
                leases = getattr(nb, "slice_leases", [])
                cross_use = sum(
                    int(lease.get("prb", 0))
                    for lease in leases
                    if lease.get("remain", 0) > 0
                )
            except Exception:
                cross_use = 0

        sum_prbs = now_use + cross_use

        # Normalize observation to match training scale
        viol_value = np.nan
        if isinstance(info, dict):
            if "total_violations" in info:
                try:
                    viol_value = int(info["total_violations"])
                except Exception:
                    pass
            elif "violations" in info:
                try:
                    v = info["violations"]
                    viol_value = int(np.asarray(v).sum())
                except Exception:
                    try:
                        viol_value = int(info["violations"])
                    except Exception:
                        pass

        # Normalize observation to match training scale
        pt = info.get("priority_terms", {}) if isinstance(info, dict) else {}
        prio_weighted = float(pt.get("prio_weighted", np.nan))
        prio_mmtc     = float(pt.get("prio_mmtc", np.nan))
        prio_cbr      = float(pt.get("prio_embb_cbr", np.nan))
        prio_vbr      = float(pt.get("prio_embb_vbr", np.nan))

        # Normalize observation to match training scale
        self.hist["violation"].append(viol_value)
        self.hist["reward"].append(float(reward))
        self.hist["resources"].append(sum_prbs)
        self.hist["prio_weighted"].append(prio_weighted)
        self.hist["prio_mmtc"].append(prio_mmtc)
        self.hist["prio_embb_cbr"].append(prio_cbr)
        self.hist["prio_embb_vbr"].append(prio_vbr)

        # Normalize observation to match training scale
        a_arr = a_arr_int.astype(np.float32)
        self.hist.setdefault("allocated_prbs", []).append(a_arr)

        # ==========================
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # ==========================
        exog = np.zeros((3 * self.n_slices,), dtype=np.float32)
        slices = info.get("slices", None) if isinstance(info, dict) else None
        if isinstance(slices, dict):
        # Normalize observation to match training scale
            for gid in range(self.n_slices):
                si = slices.get(gid, None)
                if not isinstance(si, dict):
                    continue
                s_type = str(si.get("type", "")).lower()
                base = 3 * gid

                if "embb" in s_type:
                    exog[base + 0] = float(si.get("cbr_traffic", 0.0))
                    exog[base + 1] = float(si.get("vbr_traffic", 0.0))
                    exog[base + 2] = 0.0
                elif "mmtc" in s_type:
                    exog[base + 0] = 0.0
                    exog[base + 1] = 0.0
                    exog[base + 2] = float(si.get("new_devices", 0.0))
                else:
        # Normalize observation to match training scale
                    exog[base + 0] = float(si.get("cbr_traffic", 0.0))
                    exog[base + 1] = float(si.get("vbr_traffic", 0.0))
                    exog[base + 2] = float(si.get("new_devices", 0.0))

        self.hist["exog_vec"].append(exog)
        # ==========================
        if self.debug:
            print(f"steps:{self.t},[WRAPPER]: all resources:{sum_prbs}")
        self.t += 1
        # Normalize observation to match training scale
        if (self.t % self.control_steps) == 0:
            self.save_results()
        # print(f"steps:{self.t},resources:{sum_prbs}")
        return obs, reward, terminated, truncated, info

    def save_results(self):
        """
        保存评测结果到 .npz：
          - allocated_prbs: [T, S] 的动作矩阵（若存在）
          - resources     : 每步系统实际占用的 PRB 总量（来自 hist['resources']）
          - 其他序列：reward / violation / prio_*
          - 新增：exog_vec : [T, 3N]（LSTM 离线训练用）
        """
        def _to_arr_1d(key, dtype=np.float32):
            if key not in self.hist:
                return None
            try:
                return np.asarray(self.hist[key], dtype=dtype)
            except Exception:
                return None

        def _stack_2d_from_list(key, dtype=np.float32):
            """
            把 self.hist[key]（元素为 1D 向量/列表）堆成 [T, D]。
            若不存在或为空返回 None。
            """
            if key not in self.hist or len(self.hist[key]) == 0:
                return None
            try:
                arr = np.stack([np.asarray(x, dtype=dtype) for x in self.hist[key]], axis=0)
                return arr
            except Exception:
                return None

        alloc = _stack_2d_from_list("allocated_prbs", dtype=np.float32)
        resources = _to_arr_1d("resources", dtype=np.float32)
        resources_source = "hist.resources"

        payload = dict(
            violation=_to_arr_1d("violation", np.float32),
            reward=_to_arr_1d("reward", np.float32),
            resources=(resources if resources is not None else np.zeros(0, dtype=np.float32)),
            prio_weighted=_to_arr_1d("prio_weighted", np.float32),
            prio_mmtc=_to_arr_1d("prio_mmtc", np.float32),
            prio_embb_cbr=_to_arr_1d("prio_embb_cbr", np.float32),
            prio_embb_vbr=_to_arr_1d("prio_embb_vbr", np.float32),
        )
        if alloc is not None:
            payload["allocated_prbs"] = alloc

        # ==========================
        # Normalize observation to match training scale
        # ==========================
        exog_mat = _stack_2d_from_list("exog_vec", dtype=np.float32)
        if exog_mat is not None:
            payload["exog_vec"] = exog_mat
        # ==========================

        np.savez(self.file_path, **payload)
        if self.debug:
            print(f"results saved by wrapper  (resources from: {resources_source})")

        # Normalize observation to match training scale
    def set_evaluation(self, eval_steps: int, new_path: str = None, change_name: bool = False):
        if new_path:
            self.path = new_path
            os.makedirs(self.path, exist_ok=True)
        if change_name:
            self.file_path = os.path.join(self.path, f"evaluation_{self.env_id}.npz")

# class ReportWrapper(gym.Wrapper):
#     """
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Resource usage: current-step PRB + active cross-step leases
        # (leases counted when remain>0)
        # Normalize observation to match training scale
#     """
#     def __init__(
#         self,
#         env,
        # Normalize observation to match training scale
#         control_steps: int = 500,
#         env_id: int = 1,
#         extra_samples: int = 10,
#         path: str = "./logs/",
#         verbose: bool = False,
#         n_prbs: int | None = None,
#     ):
#         super().__init__(env)

#         # -----------------------------
        # Normalize observation to match training scale
#         # -----------------------------
#         n_slices = _dig_attr(self.env, ["n_slices", "env.n_slices"])
#         if n_slices is None and hasattr(self.env, "action_space"):
#             a = self.env.action_space
#             if hasattr(a, "n") and a.n is not None:
#                 n_slices = int(a.n)
#             elif hasattr(a, "shape") and a.shape:
#                 n_slices = int(a.shape[0])
#         if n_slices is None:
#             raise AttributeError("ReportWrapper: cannot infer n_slices; expose env.n_slices or action_space.")
#         self.n_slices = int(n_slices)

#         ####################################################################################################
        # Normalize observation to match training scale
#         n_vars = None
#         if hasattr(self.env, "observation_space") and getattr(self.env.observation_space, "shape", None):
#             n_vars = int(self.env.observation_space.shape[0])

        # Normalize observation to match training scale
#         if n_vars is None:
#             n_vars = _dig_attr(self.env, ["n_variables", "env.n_variables"])

#         if n_vars is None:
#             raise AttributeError("ReportWrapper: cannot infer n_variables.")
#         self.n_variables = int(n_vars)

#         # -----------------------------
        # Normalize observation to match training scale
#         # -----------------------------
#         if n_prbs is not None:
#             self.n_prbs = int(n_prbs)
#         else:
#             guess = _dig_attr(
#                 self.env,
#                 ["n_prbs","n_PRBs","n_prb","N_PRB","node_b.n_prbs","env.n_prbs","env.node_b.n_prbs"],
#             )
#             self.n_prbs = int(guess) if guess is not None else None

#         # -----------------------------
        # Normalize observation to match training scale
#         # -----------------------------
#         self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2*self.n_slices + 1,), dtype=np.float32)
#         self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_variables + 1,), dtype=np.float32)

#         self.steps = int(steps)
#         self.control_steps = int(control_steps)
#         self.env_id = env_id
#         self.verbose = verbose

#         self.path = path
#         self.file_path = f"{path}history_{env_id}.npz"
#         self.extra_samples = int(extra_samples)

#         self.t = 0
#         self.obs = None

#         self.hist = {
#             "violation":      [],
#             "reward":         [],
#             "resources":      [],
#             "prio_weighted":  [],
#             "prio_mmtc":      [],
#             "prio_embb_cbr":  [],
#             "prio_embb_vbr":  [],
#         }

        # Normalize observation to match training scale
#         self.hist["cbr_traffic"] = []
#         self.hist["vbr_traffic"] = []
#         self.hist["new_devices"] = []

#         print(f"[DEBUG] ReportWrapper init: n_prbs={self.n_prbs}, n_slices={self.n_slices}, n_variables={self.n_variables}")

#     def reset(self, *, seed=None, options=None):
        # Normalize observation to match training scale
#         obs = np.clip(obs, -0.5, 1.5) - 0.5
#         self.obs = obs
#         return obs, info

#     def _finalize_action(self, action):
#         """
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
#         """
        # Normalize observation to match training scale
#         nS = int(self.n_slices)
        # Finalize action before stepping env
#         N_avail = _dig_attr(self.env, [
#             "node_b.remaining_prb", "env.node_b.remaining_prb",
#             "remaining_prb", "env.remaining_prb"
#         ])
#         if N_avail is None:
#             N_avail = _dig_attr(self.env, [
#                 "n_prbs","n_PRBs","n_prb","N_PRB","env.n_prbs","env.node_b.n_prbs"
#             ])
#         N_avail = int(N_avail) if N_avail is not None else int(self.n_prbs)
#         eps = 1e-8

        # Normalize observation to match training scale
#         try:
#             _ = len(action)
#         except Exception:
#             return int(action)

#         a = np.asarray(action, dtype=np.float64).reshape(-1)
        # Normalize observation to match training scale
#         if a.size < 2*nS + 1:
        # Normalize observation to match training scale
#             pad = 2*nS + 1 - a.size
#             a = np.concatenate([a, np.zeros((pad,), dtype=np.float64)], axis=0)
#         elif a.size > 2*nS + 1:
#             a = a[:2*nS + 1]

        # Normalize observation to match training scale
#         w_cross = np.maximum(0.0, a[:nS])
#         w_now   = np.maximum(0.0, a[nS:2*nS])
#         w_null  = float(max(0.0, a[2*nS]))

#         sum_cross = float(w_cross.sum())
#         sum_now   = float(w_now.sum())
#         sum_w     = sum_cross + sum_now

        # Normalize observation to match training scale
#         if (sum_w + w_null) <= eps or N_avail <= 0:
#             return np.zeros((2*nS,), dtype=np.int64)

        # Normalize observation to match training scale
#         N_use = int(np.floor(N_avail * (sum_w / (sum_w + w_null + eps))))
#         if N_use <= 0:
#             return np.zeros((2*nS,), dtype=np.int64)

        # Normalize observation to match training scale
#         N_cross = int(np.floor(N_use * (sum_cross / (sum_w + eps))))
#         N_now   = int(N_use - N_cross)

        # Normalize observation to match training scale
#         def alloc_by_weights(w, total):
#             if total <= 0 or w.sum() <= eps:
#                 return np.zeros_like(w, dtype=np.int64)
#             raw  = (w / (float(w.sum()) + eps)) * float(total)
#             base = np.floor(raw).astype(np.int64)
#             rem  = int(total - int(base.sum()))
#             if rem > 0:
#                 frac  = raw - base
        # Normalize observation to match training scale
#                 for i in range(rem):
#                     base[order[i % len(base)]] += 1
#             return base

#         a_cross = alloc_by_weights(w_cross, N_cross)
#         a_now   = alloc_by_weights(w_now,   N_now)

        # Build final integer action (cross + now)
#         final_action = np.concatenate([a_cross, a_now], axis=0).astype(np.int64)
#         return final_action

#     def step(self, action):
#         print(f"w original action:{action}")
#         final_action = self._finalize_action(action)
#         obs, reward, terminated, truncated, info = _normalize_step(self.env.step(final_action))
#         print(f"step:{self.t}, obs:{obs}")
#         print(f"step: {self.t}, allocated prbs:{final_action}, reward:{reward}, violations: {info['violations']}")

#         obs = np.clip(obs, -0.5, 1.5) - 0.5
#         self.obs = obs

        # Normalize observation to match training scale
#         try:
#             a_arr_int = np.asarray(final_action, dtype=np.int64).ravel()
#         except Exception:
#             a_arr_int = np.asarray([int(final_action)], dtype=np.int64)

#         nS = self.n_slices
#         if a_arr_int.size >= 2 * nS:
#             now_use = int(a_arr_int[nS:2 * nS].sum())
#         else:
#             now_use = int(a_arr_int.sum())

#         cross_use = 0
#         nb = _dig_attr(self.env, ["node_b", "env.node_b"])
#         if nb is not None:
#             try:
#                 leases = getattr(nb, "slice_leases", [])
#                 cross_use = sum(
#                     int(lease.get("prb", 0))
#                     for lease in leases
#                     if lease.get("remain", 0) > 0
#                 )
#             except Exception:
#                 cross_use = 0

#         sum_prbs = now_use + cross_use

#         viol_value = np.nan
#         if isinstance(info, dict):
#             if "total_violations" in info:
#                 try:
#                     viol_value = int(info["total_violations"])
#                 except Exception:
#                     pass
#             elif "violations" in info:
#                 try:
#                     v = info["violations"]
#                     viol_value = int(np.asarray(v).sum())
#                 except Exception:
#                     try:
#                         viol_value = int(info["violations"])
#                     except Exception:
#                         pass

#         pt = info.get("priority_terms", {}) if isinstance(info, dict) else {}
#         prio_weighted = float(pt.get("prio_weighted", np.nan))
#         prio_mmtc     = float(pt.get("prio_mmtc", np.nan))
#         prio_cbr      = float(pt.get("prio_embb_cbr", np.nan))
#         prio_vbr      = float(pt.get("prio_embb_vbr", np.nan))

        # Normalize observation to match training scale
#         self.hist["violation"].append(viol_value)
#         self.hist["reward"].append(float(reward))
#         self.hist["resources"].append(sum_prbs)
#         self.hist["prio_weighted"].append(prio_weighted)
#         self.hist["prio_mmtc"].append(prio_mmtc)
#         self.hist["prio_embb_cbr"].append(prio_cbr)
#         self.hist["prio_embb_vbr"].append(prio_vbr)

        # Normalize observation to match training scale
#         a_arr = a_arr_int.astype(np.float32)
#         self.hist.setdefault("allocated_prbs", []).append(a_arr)

        # Normalize observation to match training scale
#         cbr_sum = 0.0
#         vbr_sum = 0.0
#         new_dev_sum = 0.0
#         slices = info.get("slices", None) if isinstance(info, dict) else None
#         if isinstance(slices, dict):
#             for s in slices.values():
#                 if not isinstance(s, dict):
#                     continue
#                 s_type = str(s.get("type", "")).lower()
#                 if "embb" in s_type:
#                     cbr_sum += float(s.get("cbr_traffic", 0.0))
#                     vbr_sum += float(s.get("vbr_traffic", 0.0))
#                 elif "mmtc" in s_type:
#                     new_dev_sum += float(s.get("new_devices", 0.0))

#         self.hist.setdefault("cbr_traffic", []).append(cbr_sum)
#         self.hist.setdefault("vbr_traffic", []).append(vbr_sum)
#         self.hist.setdefault("new_devices", []).append(new_dev_sum)

#         self.t += 1
        # Normalize observation to match training scale
#         if (self.t % self.control_steps) == 0:
#             self.save_results()
#         print(f"steps:{self.t},resources:{sum_prbs}")
#         return obs, reward, terminated, truncated, info

#     def save_results(self):
#         """
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
#         """
#         def _to_arr_1d(key, dtype=np.float32):
#             if key not in self.hist:
#                 return None
#             try:
#                 return np.asarray(self.hist[key], dtype=dtype)
#             except Exception:
#                 return None

#         def _stack_2d_from_list(key, dtype=np.float32):
#             """
        # Normalize observation to match training scale
        # Normalize observation to match training scale
#             """
#             if key not in self.hist or len(self.hist[key]) == 0:
#                 return None
#             try:
#                 arr = np.stack([np.asarray(x, dtype=dtype) for x in self.hist[key]], axis=0)
#                 return arr
#             except Exception:
#                 return None

#         alloc = _stack_2d_from_list("allocated_prbs", dtype=np.float32)
#         resources = _to_arr_1d("resources", dtype=np.float32)
#         resources_source = "hist.resources"

#         payload = dict(
#             violation=_to_arr_1d("violation", np.float32),
#             reward=_to_arr_1d("reward", np.float32),
#             resources=(resources if resources is not None else np.zeros(0, dtype=np.float32)),
#             prio_weighted=_to_arr_1d("prio_weighted", np.float32),
#             prio_mmtc=_to_arr_1d("prio_mmtc", np.float32),
#             prio_embb_cbr=_to_arr_1d("prio_embb_cbr", np.float32),
#             prio_embb_vbr=_to_arr_1d("prio_embb_vbr", np.float32),

        # Normalize observation to match training scale
#             cbr_traffic=_to_arr_1d("cbr_traffic", np.float32),
#             vbr_traffic=_to_arr_1d("vbr_traffic", np.float32),
#             new_devices=_to_arr_1d("new_devices", np.float32),
#         )
#         if alloc is not None:
#             payload["allocated_prbs"] = alloc

#         np.savez(self.file_path, **payload)
#         print(f"results saved by wrapper  (resources from: {resources_source})")

        # Normalize observation to match training scale
#     def set_evaluation(self, eval_steps: int, new_path: str = None, change_name: bool = False):
#         if new_path: self.path = new_path
#         if change_name: self.file_path = f"{self.path}evaluation_{self.env_id}.npz"

# class ReportWrapper(gym.Wrapper):
#     """
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Resource usage: current-step PRB + active cross-step leases
        # (leases counted when remain>0)
        # Normalize observation to match training scale
#     """
#     def __init__(
#         self,
#         env,
        # Normalize observation to match training scale
#         control_steps: int = 500,
#         env_id: int = 1,
#         extra_samples: int = 10,
#         path: str = "./logs/",
#         verbose: bool = False,
#         n_prbs: int | None = None,
#     ):
#         super().__init__(env)

        # Normalize observation to match training scale
#         n_slices = _dig_attr(self.env, ["n_slices", "env.n_slices"])
#         if n_slices is None and hasattr(self.env, "action_space"):
#             a = self.env.action_space
#             if hasattr(a, "n") and a.n is not None:
#                 n_slices = int(a.n)
#             elif hasattr(a, "shape") and a.shape:
#                 n_slices = int(a.shape[0])
#         if n_slices is None:
#             raise AttributeError("ReportWrapper: cannot infer n_slices; expose env.n_slices or action_space.")
#         self.n_slices = int(n_slices)

#         ####################################################################################################
        # Normalize observation to match training scale
#         n_vars = None
#         if hasattr(self.env, "observation_space") and getattr(self.env.observation_space, "shape", None):
#             n_vars = int(self.env.observation_space.shape[0])

        # Normalize observation to match training scale
#         if n_vars is None:
#             n_vars = _dig_attr(self.env, ["n_variables", "env.n_variables"])

#         if n_vars is None:
#             raise AttributeError("ReportWrapper: cannot infer n_variables.")

#         self.n_variables = int(n_vars)
#         ####################################################################################################
#         if n_prbs is not None:
#             self.n_prbs = int(n_prbs)
#         else:
#             guess = _dig_attr(
#                 self.env,
#                 ["n_prbs","n_PRBs","n_prb","N_PRB","node_b.n_prbs","env.n_prbs","env.node_b.n_prbs"],
#             )
#             self.n_prbs = int(guess) if guess is not None else None

        # Normalize observation to match training scale
#         self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2*self.n_slices + 1,), dtype=np.float32)
#         self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_variables + 1,), dtype=np.float32)

#         self.steps = int(steps)
#         self.control_steps = int(control_steps)
#         self.env_id = env_id
#         self.verbose = verbose

#         self.path = path
#         self.file_path = f"{path}history_{env_id}.npz"
#         self.extra_samples = int(extra_samples)

#         self.t = 0
#         self.obs = None

        # Normalize observation to match training scale
#         self.hist = {
#             "violation":      [],
#             "reward":         [],
#             "resources":      [],
#             "prio_weighted":  [],
#             "prio_mmtc":      [],
#             "prio_embb_cbr":  [],
#             "prio_embb_vbr":  [],
#         }
#         ##############################################################################################
#         print(f"[DEBUG] ReportWrapper init: n_prbs={self.n_prbs}, n_slices={self.n_slices}, n_variables={self.n_variables}")

#     def reset(self, *, seed=None, options=None):
        # Normalize observation to match training scale
#         obs = np.clip(obs, -0.5, 1.5) - 0.5
#         self.obs = obs
#         return obs, info

#     def _finalize_action(self, action):
#         """
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
#         """
        # Normalize observation to match training scale
#         nS = int(self.n_slices)
        # Finalize action before stepping env
#         N_avail = _dig_attr(self.env, [
#             "node_b.remaining_prb", "env.node_b.remaining_prb",
#             "remaining_prb", "env.remaining_prb"
#         ])
#         if N_avail is None:
#             N_avail = _dig_attr(self.env, [
#                 "n_prbs","n_PRBs","n_prb","N_PRB","env.n_prbs","env.node_b.n_prbs"
#             ])
#         N_avail = int(N_avail) if N_avail is not None else int(self.n_prbs)
#         eps = 1e-8

        # Normalize observation to match training scale
#         try:
#             _ = len(action)
#         except Exception:
#             return int(action)

#         a = np.asarray(action, dtype=np.float64).reshape(-1)
        # Normalize observation to match training scale
#         if a.size < 2*nS + 1:
        # Normalize observation to match training scale
#             pad = 2*nS + 1 - a.size
#             a = np.concatenate([a, np.zeros((pad,), dtype=np.float64)], axis=0)
#         elif a.size > 2*nS + 1:
#             a = a[:2*nS + 1]

        # Normalize observation to match training scale
#         w_cross = np.maximum(0.0, a[:nS])
#         w_now   = np.maximum(0.0, a[nS:2*nS])
#         w_null  = float(max(0.0, a[2*nS]))

#         sum_cross = float(w_cross.sum())
#         sum_now   = float(w_now.sum())
#         sum_w     = sum_cross + sum_now

        # Normalize observation to match training scale
#         if (sum_w + w_null) <= eps or N_avail <= 0:
#             return np.zeros((2*nS,), dtype=np.int64)

        # Normalize observation to match training scale
#         N_use = int(np.floor(N_avail * (sum_w / (sum_w + w_null + eps))))
#         if N_use <= 0:
#             return np.zeros((2*nS,), dtype=np.int64)

        # Normalize observation to match training scale
#         N_cross = int(np.floor(N_use * (sum_cross / (sum_w + eps))))
#         N_now   = int(N_use - N_cross)

        # Normalize observation to match training scale
#         def alloc_by_weights(w, total):
#             if total <= 0 or w.sum() <= eps:
#                 return np.zeros_like(w, dtype=np.int64)
#             raw  = (w / (float(w.sum()) + eps)) * float(total)
#             base = np.floor(raw).astype(np.int64)
#             rem  = int(total - int(base.sum()))
#             if rem > 0:
#                 frac  = raw - base
        # Normalize observation to match training scale
#                 for i in range(rem):
#                     base[order[i % len(base)]] += 1
#             return base

#         a_cross = alloc_by_weights(w_cross, N_cross)
#         a_now   = alloc_by_weights(w_now,   N_now)

        # Build final integer action (cross + now)
#         final_action = np.concatenate([a_cross, a_now], axis=0).astype(np.int64)
#         return final_action

#     def step(self, action):
        # Finalize action before stepping env
#         print(f"w original action:{action}")
#         final_action = self._finalize_action(action)
#         obs, reward, terminated, truncated, info = _normalize_step(self.env.step(final_action))
#         print(f"step:{self.t}, obs:{obs}")
#         print(f"step: {self.t}, allocated prbs:{final_action}, reward:{reward}, violations: {info['violations']}")
        # Normalize observation to match training scale
#         obs = np.clip(obs, -0.5, 1.5) - 0.5
#         self.obs = obs

        # Resource usage: current-step PRB + active cross-step leases
        # (leases counted when remain>0)
#         try:
#             a_arr_int = np.asarray(final_action, dtype=np.int64).ravel()
#         except Exception:
#             a_arr_int = np.asarray([int(final_action)], dtype=np.int64)

#         nS = self.n_slices

        # Normalize observation to match training scale
#         if a_arr_int.size >= 2 * nS:
#             now_use = int(a_arr_int[nS:2 * nS].sum())
#         else:
        # Normalize observation to match training scale
#             now_use = int(a_arr_int.sum())

        # Finalize action before stepping env
#         cross_use = 0
#         nb = _dig_attr(self.env, ["node_b", "env.node_b"])
#         if nb is not None:
#             try:
#                 leases = getattr(nb, "slice_leases", [])
#                 cross_use = sum(
#                     int(lease.get("prb", 0))
#                     for lease in leases
#                     if lease.get("remain", 0) > 0
#                 )
#             except Exception:
#                 cross_use = 0

#         sum_prbs = now_use + cross_use

        # Normalize observation to match training scale
#         viol_value = np.nan
#         if isinstance(info, dict):
#             if "total_violations" in info:
#                 try:
#                     viol_value = int(info["total_violations"])
#                 except Exception:
#                     pass
#             elif "violations" in info:
#                 try:
#                     v = info["violations"]
#                     viol_value = int(np.asarray(v).sum())
#                 except Exception:
#                     try:
#                         viol_value = int(info["violations"])
#                     except Exception:
#                         pass

        # Normalize observation to match training scale
#         pt = info.get("priority_terms", {}) if isinstance(info, dict) else {}
#         prio_weighted = float(pt.get("prio_weighted", np.nan))
#         prio_mmtc     = float(pt.get("prio_mmtc", np.nan))
#         prio_cbr      = float(pt.get("prio_embb_cbr", np.nan))
#         prio_vbr      = float(pt.get("prio_embb_vbr", np.nan))

        # Normalize observation to match training scale
#         self.hist["violation"].append(viol_value)
#         self.hist["reward"].append(float(reward))
#         self.hist["resources"].append(sum_prbs)
#         self.hist["prio_weighted"].append(prio_weighted)
#         self.hist["prio_mmtc"].append(prio_mmtc)
#         self.hist["prio_embb_cbr"].append(prio_cbr)
#         self.hist["prio_embb_vbr"].append(prio_vbr)

        # Normalize observation to match training scale
#         a_arr = a_arr_int.astype(np.float32)
#         self.hist.setdefault("allocated_prbs", []).append(a_arr)

#         #########################################################
        # Normalize observation to match training scale
#         cbr_sum = 0.0
#         vbr_sum = 0.0
#         new_dev_sum = 0.0
#         slices = info.get("slices", None) if isinstance(info, dict) else None
#         if isinstance(slices, dict):
#             for s in slices.values():
#                 if not isinstance(s, dict):
#                     continue
#                 s_type = str(s.get("type", "")).lower()
#                 if "embb" in s_type:
#                     cbr_sum += float(s.get("cbr_traffic", 0.0))
#                     vbr_sum += float(s.get("vbr_traffic", 0.0))
#                 elif "mmtc" in s_type:
#                     new_dev_sum += float(s.get("new_devices", 0.0))

#         self.hist["cbr_traffic"].append(cbr_sum)
#         self.hist["vbr_traffic"].append(vbr_sum)
#         self.hist["new_devices"].append(new_dev_sum)
#         #########################################################
#         self.t += 1
        # Normalize observation to match training scale
#         if (self.t % self.control_steps) == 0:
#             self.save_results()
#         print(f"steps:{self.t},resources:{sum_prbs}")
#         return obs, reward, terminated, truncated, info

#     def save_results(self):
#         """
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
        # Normalize observation to match training scale
#         """
#         def _to_arr_1d(key, dtype=np.float32):
#             if key not in self.hist:
#                 return None
#             try:
#                 return np.asarray(self.hist[key], dtype=dtype)
#             except Exception:
#                 return None

#         def _stack_2d_from_list(key, dtype=np.float32):
#             """
        # Normalize observation to match training scale
        # Normalize observation to match training scale
#             """
#             if key not in self.hist or len(self.hist[key]) == 0:
#                 return None
#             try:
#                 rows = [np.asarray(x, dtype=dtype).ravel() for x in self.hist[key]]
#                 max_len = max(len(r) for r in rows)
        # Normalize observation to match training scale
#                 if all(len(r) == max_len for r in rows):
#                     return np.vstack(rows)
#                 else:
#                     out = np.zeros((len(rows), max_len), dtype=dtype)
#                     for i, r in enumerate(rows):
#                         out[i, :len(r)] = r
#                     return out
#             except Exception:
#                 return None

        # Normalize observation to match training scale
        # Normalize observation to match training scale

        # Normalize observation to match training scale
#         resources = _to_arr_1d("resources", dtype=np.float32)
#         resources_source = "hist.resources"

        # Normalize observation to match training scale
#         payload = dict(
#             violation=_to_arr_1d("violation", np.float32),
#             reward=_to_arr_1d("reward", np.float32),
#             resources=(resources if resources is not None else np.zeros(0, dtype=np.float32)),
#             prio_weighted=_to_arr_1d("prio_weighted", np.float32),
#             prio_mmtc=_to_arr_1d("prio_mmtc", np.float32),
#             prio_embb_cbr=_to_arr_1d("prio_embb_cbr", np.float32),
#             prio_embb_vbr=_to_arr_1d("prio_embb_vbr", np.float32),
#         )
#         if alloc is not None:
#             payload["allocated_prbs"] = alloc

#         np.savez(self.file_path, **payload)
#         print(f"results saved by wrapper  (resources from: {resources_source})")

        # Normalize observation to match training scale
#     def set_evaluation(self, eval_steps: int, new_path: str = None, change_name: bool = False):
#         if new_path: self.path = new_path
#         if change_name: self.file_path = f"{self.path}evaluation_{self.env_id}.npz"


        # Normalize observation to match training scale
class DQNWrapper(ReportWrapper):
    def __init__(self, env, steps=2000, control_steps=500, env_id=1, extra_samples=10, path='./logs/', verbose=False, n_prbs: int | None = None):
        super().__init__(env, steps=steps, control_steps=control_steps, env_id=env_id, extra_samples=extra_samples, path=path, verbose=verbose, n_prbs=n_prbs)
        g_eMBB = 2; max_eMBB = 51
        self.actions = []
        a = list(range(0, max_eMBB, g_eMBB))
        for (a1, a2) in product(a, a):
            if (self.n_prbs is None) or (a1 + a2 <= self.n_prbs):
                self.actions.append(np.array([a1, a2], dtype=np.int16))
        self.action_space = spaces.Discrete(len(self.actions))
    def step(self, action):
        # Use a peeked remaining_prb for current-step action sizing (no state change)
        nb = _get_node_b_from_env(self.env)
        if nb is not None and hasattr(nb, "peek_remaining_prb_next_step"):
            self._peek_remaining_prb = nb.peek_remaining_prb_next_step()
        else:
            self._peek_remaining_prb = None
        a = self.actions[int(action)]
        return super(DQNWrapper, self).step(a)

class TimerWrapper(gym.Wrapper):
    def __init__(self, env, steps=2000, n_prbs: int | None = None):
        super().__init__(env)
        n_slices = _dig_attr(self.env, ["n_slices", "env.n_slices"])
        if n_slices is None and hasattr(self.env, "action_space"):
            a = self.env.action_space
            if hasattr(a, "n") and a.n is not None: n_slices = int(a.n)
            elif hasattr(a, "shape") and a.shape:  n_slices = int(a.shape[0])
        if n_slices is None: raise AttributeError("TimerWrapper: cannot infer n_slices.")
        self.n_slices = int(n_slices)

        n_vars = _dig_attr(self.env, ["n_variables", "env.n_variables"])
        if n_vars is None and hasattr(self.env, "observation_space") and getattr(self.env.observation_space, "shape", None):
            n_vars = int(self.env.observation_space.shape[0])
        if n_vars is None: raise AttributeError("TimerWrapper: cannot infer n_variables.")
        self.n_variables = int(n_vars)

        if n_prbs is not None:
            self.n_prbs = int(n_prbs)
        else:
            guess = _dig_attr(self.env, ["n_prbs","n_PRBs","n_prb","N_PRB","node_b.n_prbs","env.n_prbs","env.node_b.n_prbs"])
            self.n_prbs = int(guess) if guess is not None else None

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_slices + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_variables,), dtype=np.float32)

        self.steps = int(steps)
        self.step_counter = 0
        self.simtime = 0.0
        self.time_samples = np.zeros((self.steps,), dtype=np.float32)
        self.obs = None
        print(f"n_prbs = {self.n_prbs if self.n_prbs is not None else 'None'}")
        print(f"n_slices = {self.n_slices}")

    def reset(self, *, seed=None, options=None):
        self.step_counter = 0
        self.simtime = 0.0
        obs, info = _normalize_reset(self.env.reset(seed=seed, options=options))
        obs = np.clip(obs, -0.5, 1.5) - 0.5
        self.obs = obs
        return obs, info

    def get_simtime(self): return self.simtime

    def step(self, action):
        # Use a peeked remaining_prb for current-step action sizing (no state change)
        nb = _get_node_b_from_env(self.env)
        if nb is not None and hasattr(nb, "peek_remaining_prb_next_step"):
            self._peek_remaining_prb = nb.peek_remaining_prb_next_step()
        else:
            self._peek_remaining_prb = None
        try: length = len(action)
        except Exception: length = 0
        if length and length > self.n_slices and (self.n_prbs is not None):
            action = np.asarray(action, dtype=np.float64)
            action = np.abs(action)
            t_action = float(action.sum()) or 1.0
            action = np.array([np.floor(self.n_prbs * action[i] / t_action) for i in range(self.n_slices)], dtype=np.int64)

        t1 = time.time()
        obs, reward, terminated, truncated, info = _normalize_step(self.env.step(action))
        self.simtime += time.time() - t1

        obs = np.clip(obs, -0.5, 1.5) - 0.5
        self.obs = obs
        self.step_counter += 1
        return obs, reward, terminated, truncated, info

