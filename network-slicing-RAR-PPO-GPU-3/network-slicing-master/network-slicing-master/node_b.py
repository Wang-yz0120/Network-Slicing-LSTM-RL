#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math

class NodeB():
    def __init__(self, slices_l1, slots_per_step, n_prbs, slot_length = 1e-3):
        self.slices_l1 = slices_l1
        self.n_slices_l1 = len(self.slices_l1)
        self.slots_per_step = slots_per_step
        self.n_prbs = n_prbs
        self.slot_length = slot_length
        ##########################################################################
        self.slice_leases = []  # 保存所有切片的租约 [{'slice_idx': int, 'prb': int, 'remain': int}]
        self.remaining_prb = self.n_prbs  # 剩余的可用PRB
        ##########################################################################
        self.reset()

    def reset(self):
        self.steps = 0
        for slice_l1 in self.slices_l1:
            slice_l1.reset()
        state = self.get_state()
        #####################################
        self.slice_leases.clear()  # 清空租约
        self.remaining_prb = int(self.n_prbs)  # 重置可用池为总PRB
        #####################################
        return state

    def get_n_variables(self):
        n_variables = 0
        for slice_l1 in self.slices_l1:
            # n_variables += slice_l1.get_n_variables()
            n_variables += slice_l1.get_n_variables() + 1
        return n_variables

    def reset_info(self):
        """Reset the info of the L1 slices for SLA assessment"""
        for l1 in self.slices_l1:
            l1.reset_info()

    def slot(self):
        """runs the system just for one time-slot"""
        for slice_l1 in self.slices_l1:
            slice_l1.slot()

    def get_state(self):
        state = np.array([], dtype=np.float32)
        for l1 in self.slices_l1:
            state = np.concatenate((state, l1.get_state()), axis=None)
        ##############################################################
        #新增各切片跨step prb量
        slices_occ = np.array(
        [
            self.get_remaining_prb_for_slice(i) / max(1, self.n_prbs)
            for i in range(self.n_slices_l1)
        ],
        dtype=np.float32,
        )
        state = np.concatenate((state, slices_occ), axis=None)
        # slices_occ = []
        # for i in range(self.n_slices_l1):
        #     occ_prbs = self.get_remaining_prb_for_slice(i)
        #     slices_occ.append(occ_prbs/max(1,self.n_prbs))
        # state = np.concatenate((state,slices_occ), axis = None)
        ##############################################################
        extra = np.array([self.remaining_prb / max(1, self.n_prbs)], dtype=np.float32)
        state = np.concatenate((state, extra), axis=None)# 新增状态用于显示剩余可用prb
        ##############################################################
        return state

    def _flatten_slices_info(self):
        """
        把所有 L1 的 {i: slice_ran.info} 扁平化成一个 dict：
        {'slices': {0: {...}, 1:{...}, ...}, 'total_violations': int}
        """
        flat = {}
        gid = 0
        for l1 in self.slices_l1:
            per_l1 = l1.get_info()  # 形如 {i: slice_ran.info}
            if isinstance(per_l1, dict):
                for _, s_info in per_l1.items():
                    # 确保必要键存在（兼容防御）
                    s_info.setdefault("type", "unknown")
                    s_info.setdefault("viol", 0)
                    # eMBB 可细分 CBR/VBR 的违约
                    s_info.setdefault("viol_cbr", 0)
                    s_info.setdefault("viol_vbr", 0)
                    flat[gid] = s_info
                    gid += 1
        total_viol = int(sum(int(si.get("viol", 0)) for si in flat.values()))
        return flat, total_viol

    def get_info(self, violations=0, SLA_labels=0):
        # 兼容保留：原有字段
        info = {
            'l1_info': [l1.get_info() for l1 in self.slices_l1],
            'SLA_labels': SLA_labels,
            'violations': violations,
            'n_prbs': [l1.n_prbs for l1 in self.slices_l1],
        }
        # 新增：顶层扁平化后的 per-slice 信息（优先级加权需要）
        slices_dict, total_viol = self._flatten_slices_info()
        info['slices'] = slices_dict
        info['total_violations'] = total_viol
        return info

    def compute_reward(self):
        """checks if the SLA is fulfilled for each slice"""
        SLA_labels = np.zeros(self.n_slices_l1, dtype=np.int64)
        violations = np.zeros(self.n_slices_l1, dtype=np.int64)
        for i, l1 in enumerate(self.slices_l1):
            SLA_labels[i], violations[i] = l1.compute_reward()
        return SLA_labels, violations

    def step(self, action):
        self.reset_info()

        # 释放到期租约
        self.release_expired_prb()
        print(f'step: {self.steps},[NODEB]:current lease: {self.slice_leases},remaining_prb: {self.remaining_prb}')
        # 解析动作（第三段是全局预留标量）
        a_cross, a_now = self.parse_action(action)


        # 为每个切片创建新的跨step租约（估时要用到当步PRB作为一次性清账）
        for slice_idx in range(self.n_slices_l1):
            p_cross = int(a_cross[slice_idx])
            p_curr  = int(a_now[slice_idx])
            if p_cross > 0:
                cross_t = self.calculate_cross_step_prb_usage(slice_idx, p_cross, p_curr)
                if cross_t > 0:
                    self.record_cross_step_usage(slice_idx, p_cross, cross_t)

        print(f'step: {self.steps},[NODEB]:current lease: {self.slice_leases},remaining_prb: {self.remaining_prb}')
        # 分配当步 PRB：当步 + 租约剩余（全局预留已扣，不会被用到）
        i_prb = 0
        for i, l1 in enumerate(self.slices_l1):
            total_prb = int(a_now[i]) + int(self.get_remaining_prb_for_slice(i))
            total_prb = max(0, min(total_prb, int(self.n_prbs)))  # 保险裁剪
            l1.set_prbs(i_prb, total_prb)
            i_prb += total_prb   # 修：原代码用了未定义变量 prbs【:contentReference[oaicite:3]{index=3}】

        # 跑 slots
        for _ in range(self.slots_per_step):
            self.slot()

        state = self.get_state()
        SLA_labels, violations = self.compute_reward()
        info = self.get_info(SLA_labels=SLA_labels, violations=violations)
        ######################################################################
        slices_occ = []
        for i in range(self.n_slices_l1):
            occ_prbs = self.get_remaining_prb_for_slice(i)
            slices_occ.append(occ_prbs)
        print(f"step: {self.steps},[NODEB]:slice_occ:{slices_occ}")
        ######################################################################
        self.steps += 1
        return state, info


    def release_expired_prb(self):
        """
        释放已过期的跨step PRB资源。
        """
        # 遍历租约，释放过期的PRB
        for lease in self.slice_leases[:]:
            lease['remain'] -= 1  # 减少剩余时间
            if lease['remain'] <= 0:
                self.remaining_prb += lease['prb']  # 释放PRB资源
                self.slice_leases.remove(lease)  # 移除过期的租约

    def parse_action(self, action):
        """
        返回：
        - cross_step_allocations: 长度 = n_slices 的向量，跨step占用的PRB
        - current_step_allocations: 长度 = n_slices 的向量，当step占用的PRB
        - reserve_global: 标量，表示本step全局预留、谁都不能用的PRB
        """
        n = int(self.n_slices_l1)  # 你文件里已有 n_slices_l1
        # 仍按“每切片两段”解析
        cross_step_allocations   = np.asarray(action[:n], dtype=np.int64)      # 跨step
        current_step_allocations = np.asarray(action[n:2*n], dtype=np.int64)   # 当step

        # 基本裁剪：不超过总PRB、不为负
        cross_step_allocations   = np.clip(cross_step_allocations,   0, self.n_prbs)
        current_step_allocations = np.clip(current_step_allocations, 0, self.n_prbs)
        return cross_step_allocations, current_step_allocations

    def calculate_cross_step_prb_usage(self, slice_idx: int, p_cross: int, p_curr: int) -> int:
        """
        返回值单位：step（按“步”计时）。推荐与 release_expired_prb() 的“每 step 衰减 1 次”搭配。
        """
        import math
        # ---- 保护与边界 ----
        p_cross = max(int(p_cross), 0)
        p_curr  = max(int(p_curr),  0)
        if p_cross == 0 and p_curr == 0:
            return 0

        T_MIN_STEPS = 1
        T_MAX_STEPS = 3          # <== 关键：最多只允许续 3 个 step

        # 取类型
        l1 = self.slices_l1[slice_idx]
        slices_dict, _ = self._flatten_slices_info()
        s_info = slices_dict.get(slice_idx, {})
        typ  = str(s_info.get("type", "")).lower()
        viol = bool(s_info.get("viol", 0))

        if typ == "embb" or getattr(l1, "type", "").lower() == "embb":
            # backlog/速率/到达（内部已做EWMA平滑更稳）
            B, r_bar, lam = self._embb_slice_stats(l1, use_ewma=True)  # 见下文小改
            S0      = p_curr  * r_bar     # 本步一次性清账
            C_cross = p_cross * r_bar     # 之后每步持续能力

            B_target = 0.0
            alpha    = 1.2                # 保守一点，防止租期过长
            numer = max(0.0, B - B_target - S0)
            denom = max(1e-6, C_cross - alpha * lam)

            T = 1 if numer <= 0 else int(math.ceil(numer / denom))
            if viol:
                T = int(math.ceil(T * 1.5))  # 违约时更保守
            return max(T_MIN_STEPS, min(T, T_MAX_STEPS))

        elif typ == "mmtc" or getattr(l1, "type", "").lower() == "mmtc":
            R_now = self._mmtc_slice_work(l1)
            if R_now <= 0:
                return T_MIN_STEPS
            S0   = min(p_curr, R_now)
            Rrem = max(0, R_now - S0)
            if viol:
                Rrem = int(math.ceil(Rrem * 1.3))  # 也适度放大，但别过头
            denom = max(1, p_cross)
            T = 1 if Rrem == 0 else int(math.ceil(Rrem / denom))
            return max(T_MIN_STEPS, min(T, T_MAX_STEPS))

        else:
            return T_MIN_STEPS

    def record_cross_step_usage(self, slice_idx, cross_step_allocation, cross_step_time):
        """
        记录每个切片的跨step PRB占用情况，将租约记录到字典中,并更新可用的prb总数。
        """
        lease = {'slice_idx': slice_idx, 'prb': cross_step_allocation, 'remain': cross_step_time}
        self.slice_leases.append(lease)
        self.remaining_prb -= cross_step_allocation

    def get_remaining_prb_for_slice(self, slice_idx):
        """
        获取该切片跨step占用PRB的剩余量。
        """
        remaining_prb = 0
        for lease in self.slice_leases:
            if lease['slice_idx'] == slice_idx and lease['remain'] > 0:
                remaining_prb += lease['prb']
        return remaining_prb

    def peek_remaining_prb_next_step(self):
        """
        预估“当前 step 释放过期 lease 后”的剩余 PRB 数量（不修改状态）。
        """
        released = 0
        for lease in self.slice_leases:
            if lease.get('remain', 0) <= 1:
                released += int(lease.get('prb', 0))
        print(f"step: {self.steps},[NODEB]:pre_released:{released}")
        return int(self.remaining_prb + released)

    # def estimate_transmission_rate(self, ue):
    #     """
    #     估算传输速率 C_ue（比特/step），根据信道质量（SNR）和MCS计算。
        
    #     :param ue: 用户设备（UE）
    #     :return: 估算的传输速率（比特/step）
    #     """
    #     # 假设通过信噪比（SNR）与MCS获得每符号的比特数（这里可以使用MCS代码集来计算）
    #     bits_per_symbol = self.mcs_codeset.mcs_rate_vs_error(ue.e_snr, error_bound=0.1)[1]
    #     # 每PRB的符号数
    #     sym_per_prb = self.sym_per_prb
    #     # 计算传输速率
    #     C_ue = bits_per_symbol * sym_per_prb
    #     return C_ue
    # node_b.py
    # def _embb_slice_stats(self, l1):
    #     """
    #     从 SliceL1eMBB 抽取三个指标：
    #     B: backlog bits（所有 UE 的 queue 之和）
    #     r_bar: 平均每PRB可传 bits（保守估计）
    #     lam: 近一步的到达量（bit/step）的保守估计
    #     """
    #     # 1) backlog 与到达：直接遍历 UE（这些字段来源于 slice_ran.UE）【调度路径见 scheduler.allocate / ue.traffic_step】
    #     B = 0.0
    #     lam = 0.0
    #     sum_bits = 0.0
    #     sum_prbs = 0.0

    #     ues = getattr(l1, "ues", []) or []
    #     for ue in ues:
    #         # 队列与到达
    #         B   += float(getattr(ue, "queue", 0.0))
    #         lam += float(getattr(ue, "new_bits", 0.0))
    #         # 最近一次调度统计（若刚经历过一个 slot，会有 bits/prbs）
    #         sum_bits += float(getattr(ue, "bits", 0.0))
    #         sum_prbs += float(getattr(ue, "prbs", 0.0))

    #     # 2) r_bar：优先用“最近一次 bits/prbs”的安全均值；没有时给兜底
    #     if sum_prbs > 0:
    #         r_bar = max(1.0, sum_bits / sum_prbs)
    #     else:
    #         # 兜底：从调度器拿到 sym_per_prb，并假设一个保守 bits_per_sym
    #         r_bar = 100.0
    #         try:
    #             sched = getattr(l1, "scheduler", None)
    #             sym_per_prb = getattr(sched, "sym_per_prb", None)
    #             if sym_per_prb:
    #                 # 假设一个保守的 bits_per_sym=1
    #                 r_bar = max(1.0, 1.0 * float(sym_per_prb))
    #         except Exception:
    #             pass

    #     return B, r_bar, lam
    def _embb_slice_stats(self, l1, use_ewma=False):
        B = 0.0; lam = 0.0; sum_bits = 0.0; sum_prbs = 0.0
        ues = getattr(l1, "ues", []) or []
        for ue in ues:
            B   += float(getattr(ue, "queue", 0.0))
            lam += float(getattr(ue, "new_bits", 0.0))
            sum_bits += float(getattr(ue, "bits", 0.0))
            sum_prbs += float(getattr(ue, "prbs", 0.0))
        r_bar = (sum_bits / sum_prbs) if sum_prbs > 0 else 100.0
        if use_ewma:
            k = 0.3  # 平滑系数，可调
            if not hasattr(self, "_rbar_ema"):
                self._rbar_ema = {}
            old = self._rbar_ema.get(id(l1), r_bar)
            r_bar = k * r_bar + (1 - k) * old
            self._rbar_ema[id(l1)] = r_bar
        return B, max(1.0, r_bar), lam

    def _mmtc_slice_work(self, l1):
        """
        从 SliceL1mMTC 抽取“总重复次数”作为工作量。
        """
        reps = getattr(l1, "repetitions", None)
        if reps is None:
            try:
                # mMTC 的 repetitions 存在于 SliceL1mMTC 对象中
                reps = l1.repetitions
            except Exception:
                reps = []
        try:
            import numpy as _np
            R_now = int(_np.sum(reps))
        except Exception:
            R_now = int(sum(int(x) for x in (reps or [])))
        return R_now
