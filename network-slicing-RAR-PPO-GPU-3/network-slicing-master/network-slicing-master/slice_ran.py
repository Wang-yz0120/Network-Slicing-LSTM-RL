#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Classes:

UE
SliceRANmMTC
SliceRANeMBB

"""

DEBUG = True
CBR = 0
VBR = 1

import numpy as np
from traffic_generators import VbrSource, CbrSource


# ===== MOD START =====
# [MOD] Two-state Markov-Modulated Poisson Process (MMPP) helper.
# - Supports TON-style burstiness: arrival rate switches between normal/burst states.
# - If you don't pass lambda_n/lambda_b in descriptions, this won't be used.
class TwoStateMMPP:
    """
    Two-state MMPP for UE arrivals.

    State 0 (normal): lambda_n (1/s)
    State 1 (burst) : lambda_b (1/s)

    mean_normal_sec: expected duration of normal state (seconds)
    mean_burst_sec : expected duration of burst state (seconds)

    We update state every slot (1ms by default). Transition probabilities per slot:
      p01 = slot_length / mean_normal_sec
      p10 = slot_length / mean_burst_sec
    """
    def __init__(self, rng, lambda_n, lambda_b, mean_normal_sec, mean_burst_sec, slot_length):
        self.rng = rng
        self.lambda_n = float(lambda_n)
        self.lambda_b = float(lambda_b)
        self.slot_length = float(slot_length)

        # avoid division by zero
        mean_normal_sec = max(float(mean_normal_sec), 1e-9)
        mean_burst_sec = max(float(mean_burst_sec), 1e-9)

        # per-slot transition probabilities
        self.p01 = min(self.slot_length / mean_normal_sec, 1.0)  # normal -> burst
        self.p10 = min(self.slot_length / mean_burst_sec, 1.0)   # burst  -> normal

        self.state = 0  # start in normal by default

    def step(self):
        u = self.rng.random()
        if self.state == 0:
            if u < self.p01:
                self.state = 1
        else:
            if u < self.p10:
                self.state = 0

    def current_lambda(self):
        return self.lambda_b if self.state == 1 else self.lambda_n
# ===== MOD END =====


class UE:
    '''
    eMBB UE contains a traffic source that can be CRB (GBR) or VBR (non-GBR)
    '''
    def __init__(self, id, slice_ran_id, traffic_source, type, global_prbs, snr_generator, window = 50, slot_length = 1e-3):
        self.id = id
        self.slice_ran_id = slice_ran_id
        self.traffic_source = traffic_source
        self.type = type
        self.th = 0
        self.b = 1/window
        self.a = 1 - self.b
        self.queue = 0
        self.slot_length = slot_length
        self.global_prbs = global_prbs
        self.snr_generator = snr_generator
        # per subframe variables
        self.snr = 0 # real error values per prb
        self.e_snr = 0 # estimated error
        self.new_bits = 0 # incoming bits
        self.bits = 0 # assigned bits
        self.prbs = 0 # assigned prbs
        self.p = 0 # reception probability
    
    def estimate_snr(self, snr):
        self.snr = snr
        self.e_snr = round(np.mean(snr))

    def traffic_step(self):
        self.new_bits = self.traffic_source.step()
        self.queue += self.new_bits
    
    def transmission_step(self, received):
        if not received:
            self.bits = 0
        self.queue = max(self.queue - self.bits, 0)
        self.th = self.a * self.th + self.b * self.bits / self.slot_length

    def __repr__(self):
        return 'UE {}'.format(self.id)

class MTCdevice:
    def __init__(self, id, repetitions, slice_ran_id):
        self.id = id
        self.repetitions = repetitions
        self.slice_ran_id = slice_ran_id
    def __repr__(self):
        return 'MTC {}'.format(self.id)

class SliceRANmMTC:
    '''
    Generates message arrivals at the mMTC devices
    according to the characteristics defined in MTC_description:
    - n_devices: total number of devices
    - repetition_set: possible repetitions
    - period_set: possible times between message arrivals
    '''
    def __init__(self, rng, id, SLA, MTCdescription, state_variables, norm_const, slots_per_step):
        self.type = 'mMTC'
        self.rng = rng
        self.id = id
        self.SLA = SLA
        self.state_variables = state_variables # ['devices', 'avg_rep', 'delay']
        self.norm_const = norm_const # 100 all
        self.slots_per_step = slots_per_step

        self.n_devices = MTCdescription['n_devices']
        self.repetition_set = MTCdescription['repetition_set']
        self.period_set = MTCdescription['period_set']

        self.reset()

    def reset(self):
        self.reset_state()
        self.reset_info()
        self.period = np.ones((self.n_devices), dtype=np.int64)
        self.t_to_arrival = np.zeros((self.n_devices), dtype=np.int64)
        self.devices = []
        for i in range(self.n_devices):
            repetitions = self.rng.choice(self.repetition_set)
            self.period[i] = self.rng.choice(self.period_set)
            self.t_to_arrival[i] = 1 + self.rng.choice(np.arange(self.period[i]))
            self.devices.append(MTCdevice(i, repetitions, self.id))

    def slot(self):
        self.slot_counter += 1

        # advance time
        self.t_to_arrival -= 1

        # arrivals
        arrival_list = []
        arrivals = self.t_to_arrival == 0
        indices = np.where(arrivals)

        for i in indices[0]:
            arrival_list.append(self.devices[i])

        # prepare for next arrival (deterministic inter arrival time)
        self.t_to_arrival[arrivals] = self.period[arrivals]

        # 新增：统计本 slot 新到达设备数（外生负载）
        self.info['new_devices'] += len(arrival_list)
        
        return arrival_list, []

    def reset_info(self):
        self.info = {
        'delay': 0,
        'avg_rep': 0,
        'devices': 0,
        'new_devices': 0,   # <-- 新增
        }
        self.slot_counter = 0
        ################################
        # 🔹补：优先级需要的键，初始化为 0/默认
        self.info['type'] = 'mmtc'
        self.info['viol'] = 0
        # 为了统一结构，mMTC 也放上 cbr/vbr 两个键（始终为 0）
        self.info['viol_cbr'] = 0
        self.info['viol_vbr'] = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = np.float32)

    def get_n_variables(self):
        return len(self.state_variables)

    def get_state(self):
        '''convert the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]        
        return self.state

    def update_info(self, delay, avg_rep, devices):
        self.info['delay'] += delay
        self.info['avg_rep'] += avg_rep
        self.info['devices'] += devices
        

    def compute_reward(self):
        '''assesses SLA violations'''
        viol = int(not (self.info['delay']/self.slots_per_step < self.SLA['delay']))
        self.info["viol"] = viol
        self.info["type"] = "mmtc"
        self.info['viol_cbr'] = 0
        self.info['viol_vbr'] = 0
        return viol


class SliceRANeMBB:
    '''
    Generates arrivals and departures of eMBB ues.
    There are two traffic types: CRB (GBR) and VBR (non-GBR)
    CBR traffic parameters are given in CBR_description
    VBR traffic parameters are given in VBR_description
    '''
    def __init__(self, rng, user_counter, id, SLA, CBR_description, VBR_description, state_variables, norm_const, slots_per_step, global_prbs, snr_generator, slot_length = 1e-3):
        self.type = 'eMBB'
        self.rng = rng
        self.user_counter = user_counter
        self.id = id
        self.slot_length = slot_length
        self.slots_per_step = slots_per_step
        self.observation_time = slots_per_step * slot_length
        self.SLA = SLA # service level agreement description
        self.state_variables = state_variables
        self.norm_const = norm_const
        self.global_prbs = global_prbs
        self.snr_generator = snr_generator

        # ===== MOD START =====
        # [MOD] Support two modes:
        # 1) Old mode (paper original): CBR_description['lambda'] exists -> constant Poisson arrivals.
        # 2) Bursty mode (TON-style): CBR_description has lambda_n/lambda_b/mean_normal/mean_burst -> MMPP arrivals.
        self.use_cbr_mmpp = ('lambda_n' in CBR_description and 'lambda_b' in CBR_description
                             and 'mean_normal' in CBR_description and 'mean_burst' in CBR_description)
        self.use_vbr_mmpp = ('lambda_n' in VBR_description and 'lambda_b' in VBR_description
                             and 'mean_normal' in VBR_description and 'mean_burst' in VBR_description)

        if self.use_cbr_mmpp:
            self.cbr_mmpp = TwoStateMMPP(
                rng=self.rng,
                lambda_n=CBR_description['lambda_n'],
                lambda_b=CBR_description['lambda_b'],
                mean_normal_sec=CBR_description['mean_normal'],
                mean_burst_sec=CBR_description['mean_burst'],
                slot_length=self.slot_length
            )
            # keep a fallback constant too (not used unless misconfigured)
            self.cbr_arrival_rate = float(CBR_description.get('lambda', CBR_description['lambda_n']))
        else:
            self.cbr_mmpp = None
            self.cbr_arrival_rate = float(CBR_description['lambda'])

        self.cbr_mean_time = float(CBR_description['t_mean'])
        self.cbr_bit_rate = float(CBR_description['bit_rate'])

        if self.use_vbr_mmpp:
            self.vbr_mmpp = TwoStateMMPP(
                rng=self.rng,
                lambda_n=VBR_description['lambda_n'],
                lambda_b=VBR_description['lambda_b'],
                mean_normal_sec=VBR_description['mean_normal'],
                mean_burst_sec=VBR_description['mean_burst'],
                slot_length=self.slot_length
            )
            self.vbr_arrival_rate = float(VBR_description.get('lambda', VBR_description['lambda_n']))
        else:
            self.vbr_mmpp = None
            self.vbr_arrival_rate = float(VBR_description['lambda'])

        self.vbr_mean_time = float(VBR_description['t_mean'])
        # ===== MOD END =====

        self.vbr_source_data = {
            'packet_size': VBR_description['p_size'],
            'burst_size': VBR_description['b_size'],
            'burst_rate':VBR_description['b_rate']
        }

        self.reset()

    def reset(self):
        self.slot_counter = 0
        self.remaining_time = {}
        self.cbr_steps_next_arrival = 0
        self.vbr_steps_next_arrival = 0
        self.vbr_ues = {}
        self.cbr_ues = {}
        self.reset_state()
        self.reset_info()

        # ===== MOD START =====
        # [MOD] Reset MMPP state to normal at episode start (optional but stable for reproducibility).
        if self.cbr_mmpp is not None:
            self.cbr_mmpp.state = 0
        if self.vbr_mmpp is not None:
            self.vbr_mmpp.state = 0
        # ===== MOD END =====

    def get_n_variables(self):
        return len(self.state_variables)

    def cbr_cac(self):
        '''Admission control for CBR users'''
        slots = max(self.slot_counter,1)
        time = slots * self.slot_length
        cbr_prb = self.info['cbr_prb'] / slots
        cbr_th = self.info['cbr_th'] / time
        if cbr_prb >= self.SLA['cbr_prb'] or cbr_th >= self.SLA['cbr_th']:
            return False
        return True

    # ===== MOD START =====
    # [MOD] helper: draw next inter-arrival steps given an arrival rate (1/s)
    def _draw_interarrival_steps(self, arrival_rate):
        # if arrival_rate is too small or zero, avoid inf
        if arrival_rate is None or arrival_rate <= 0:
            return np.int64(10**9)
        inter_arrival_time = self.rng.exponential(1.0 / arrival_rate)  # seconds
        inter_arrival_steps = np.rint(inter_arrival_time / self.slot_length)
        inter_arrival_steps = max(np.int64(inter_arrival_steps), np.int64(1))
        return inter_arrival_steps
    # ===== MOD END =====

    def cbr_arrivals(self):
        if self.cbr_steps_next_arrival == 0:
            # ===== MOD START =====
            # [MOD] Use MMPP current lambda if enabled; otherwise constant lambda.
            if self.cbr_mmpp is not None:
                arrival_rate = self.cbr_mmpp.current_lambda()
            else:
                arrival_rate = self.cbr_arrival_rate

            # generate next arrival
            self.cbr_steps_next_arrival = self._draw_interarrival_steps(arrival_rate)
            # ===== MOD END =====

            if self.cbr_cac(): # check admission control
                # generate new user
                ue_id = next(self.user_counter)
                cbr_source = CbrSource(bit_rate = self.cbr_bit_rate)
                ue = UE(ue_id, self.id, cbr_source, CBR, self.global_prbs, self.snr_generator)
                self.cbr_ues[ue_id] = ue

                # generate holding time
                holding_time = self.rng.exponential(self.cbr_mean_time)
                holding_time = np.rint(holding_time / self.slot_length)
                self.remaining_time[ue_id] = holding_time

                return [ue] # return user
        else:
            self.cbr_steps_next_arrival -= 1    
        return []

    def vbr_arrivals(self):
        if self.vbr_steps_next_arrival == 0:
            # create new vbr user
            ue_id = next(self.user_counter)
            vbr_source = VbrSource(**self.vbr_source_data)
            ue = UE(ue_id, self.id, vbr_source, VBR, self.global_prbs, self.snr_generator)
            self.vbr_ues[ue_id] = ue

            # generate holding time
            holding_time = self.rng.exponential(self.vbr_mean_time)
            holding_time = np.rint(holding_time / self.slot_length)
            self.remaining_time[ue_id] = holding_time

            # ===== MOD START =====
            # [MOD] Use MMPP current lambda if enabled; otherwise constant lambda.
            if self.vbr_mmpp is not None:
                arrival_rate = self.vbr_mmpp.current_lambda()
            else:
                arrival_rate = self.vbr_arrival_rate

            # generate next arrival
            self.vbr_steps_next_arrival = self._draw_interarrival_steps(arrival_rate)
            # ===== MOD END =====

            return [ue]
        else:
            self.vbr_steps_next_arrival -= 1
            return []

    def departures(self):
        departures = []
        current_ids = list(self.remaining_time.keys())
        for id in current_ids:
            self.remaining_time[id] -= 1
            if self.remaining_time[id] == 0:
                departures.append(id)
                del self.remaining_time[id] # delete timer
                self.vbr_ues.pop(id, None) # delete ue if here
                self.cbr_ues.pop(id, None) # or here    
        return departures   

    def slot(self):
        # ===== MOD START =====
        # [MOD] Advance MMPP state once per slot so arrivals show clear normal/burst segments (TON-style).
        if self.cbr_mmpp is not None:
            self.cbr_mmpp.step()
        if self.vbr_mmpp is not None:
            self.vbr_mmpp.step()
        # ===== MOD END =====

        self.slot_counter += 1
        arrivals = self.cbr_arrivals()
        arrivals.extend(self.vbr_arrivals())
        departures = self.departures()
        return arrivals, departures

    def reset_info(self):
        self.info = {'cbr_traffic': 0, 'cbr_th': 0, 'cbr_prb': 0, 'cbr_queue':0, 'cbr_snr': 0,\
                    'vbr_traffic': 0, 'vbr_th': 0, 'vbr_prb': 0, 'vbr_queue': 0, 'vbr_snr': 0}
        self.slot_counter = 0
        #####################
        # 🔹补：优先级需要的键，初始化为 0/默认
        self.info['type'] = 'embb'
        self.info['viol'] = 0
        self.info['viol_cbr'] = 0
        self.info['viol_vbr'] = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = np.float32)
    
    def update_info(self):
        queue = 0
        snr = 0
        n = 0
        for ue in self.cbr_ues.values():
            self.info['cbr_traffic'] += ue.new_bits
            self.info['cbr_th'] += ue.bits
            self.info['cbr_prb'] += ue.prbs
            queue += ue.queue
            snr += ue.e_snr
            n += 1
        n = max(n,1)
        self.info['cbr_queue'] += queue/n
        self.info['cbr_snr'] += snr/n

        queue = 0
        snr = 0
        n = 0
        for ue in self.vbr_ues.values():
            self.info['vbr_traffic'] += ue.new_bits
            self.info['vbr_th'] += ue.bits
            self.info['vbr_prb'] += ue.prbs
            queue += ue.queue
            snr += ue.e_snr
            n += 1
        n = max(n,1)
        self.info['vbr_queue'] += queue/n
        self.info['vbr_snr'] += snr/n

    def compute_reward(self):
        '''assesses SLA violations'''
        cbr_th = self.info['cbr_th']/self.observation_time > self.SLA['cbr_th']
        cbr_prb = self.info['cbr_prb']/self.slots_per_step > self.SLA['cbr_prb']
        cbr_queue = self.info['cbr_queue']/self.slots_per_step < self.SLA['cbr_queue']
        vbr_th = self.info['vbr_th']/self.observation_time > self.SLA['vbr_th']
        vbr_prb = self.info['vbr_prb']/self.slots_per_step > self.SLA['vbr_prb']
        vbr_queue = self.info['vbr_queue']/self.slots_per_step < self.SLA['vbr_queue']
        # the slice has to guarantee the objective delay for cbr and vbr if their traffics do not surpass the maximum
        cbr_fulfilled = cbr_th or cbr_prb or cbr_queue 
        vbr_fulfilled = vbr_th or vbr_prb or vbr_queue
        SLA_fulfilled = cbr_fulfilled and vbr_fulfilled
        viol = int(not SLA_fulfilled)
        self.info["viol"] = viol
        self.info["type"] = "embb"
        self.info["viol_cbr"] = int(not cbr_fulfilled)
        self.info["viol_vbr"] = int(not vbr_fulfilled)
        return viol

    def get_state(self):
        '''converts the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]        
        return self.state

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# @author: juanjosealcaraz

# Classes:

# UE
# SliceRANmMTC
# SliceRANeMBB

# """
# DEBUG = True
# CBR = 0
# VBR = 1

# import numpy as np
# from traffic_generators import VbrSource, CbrSource

# class UE:
#     '''
#     eMBB UE contains a traffic source that can be CRB (GBR) or VBR (non-GBR)
#     '''
#     def __init__(self, id, slice_ran_id, traffic_source, type, window = 50, slot_length = 1e-3):
#         self.id = id
#         self.slice_ran_id = slice_ran_id
#         self.traffic_source = traffic_source
#         self.type = type
#         self.th = 0
#         self.b = 1/window
#         self.a = 1 - self.b
#         self.queue = 0
#         self.slot_length = slot_length

#         # per subframe variables
#         self.snr = 0 # real error values per prb
#         self.e_snr = 0 # estimated error
#         self.new_bits = 0 # incoming bits
#         self.bits = 0 # assigned bits
#         self.prbs = 0 # assigned prbs
#         self.p = 0 # reception probability
    
#     def estimate_snr(self, snr):
#         self.snr = snr
#         self.e_snr = round(np.mean(snr))

#     def traffic_step(self):
#         self.new_bits = self.traffic_source.step()
#         self.queue += self.new_bits
    
#     def transmission_step(self, received):
#         if not received:
#             self.bits = 0
#         self.queue = max(self.queue - self.bits, 0)
#         self.th = self.a * self.th + self.b * self.bits / self.slot_length

#     def __repr__(self):
#         return 'UE {}'.format(self.id)

# class MTCdevice:
#     def __init__(self, id, repetitions, slice_ran_id):
#         self.id = id
#         self.repetitions = repetitions
#         self.slice_ran_id = slice_ran_id
#     def __repr__(self):
#         return 'MTC {}'.format(self.id)

# class SliceRANmMTC:
#     '''
#     Generates message arrivals at the mMTC devices
#     according to the characteristics defined in MTC_description:
#     - n_devices: total number of devices
#     - repetition_set: possible repetitions
#     - period_set: possible times between message arrivals
#     '''
#     def __init__(self, rng, id, SLA, MTCdescription, state_variables, norm_const, slots_per_step):
#         self.type = 'mMTC'
#         self.rng = rng
#         self.id = id
#         self.SLA = SLA
#         self.state_variables = state_variables # ['devices', 'avg_rep', 'delay']
#         self.norm_const = norm_const # 100 all
#         self.slots_per_step = slots_per_step

#         self.n_devices = MTCdescription['n_devices']
#         self.repetition_set = MTCdescription['repetition_set']
#         self.period_set = MTCdescription['period_set']

#         self.reset()

#     def reset(self):
#         self.reset_state()
#         self.reset_info()
#         self.period = np.ones((self.n_devices), dtype=np.int64)
#         self.t_to_arrival = np.zeros((self.n_devices), dtype=np.int64)
#         self.devices = []
#         for i in range(self.n_devices):
#             repetitions = self.rng.choice(self.repetition_set)
#             self.period[i] = self.rng.choice(self.period_set)
#             self.t_to_arrival[i] = 1 + self.rng.choice(np.arange(self.period[i]))
#             self.devices.append(MTCdevice(i, repetitions, self.id))

#     def slot(self):
#         self.slot_counter += 1

#         # advance time
#         self.t_to_arrival -= 1

#         # arrivals
#         arrival_list = []
#         arrivals = self.t_to_arrival == 0
#         indices = np.where(arrivals)

#         # print('indices = {}'.format(indices))
#         for i in indices[0]:
#             arrival_list.append(self.devices[i])

#         # prepare for next arrival (deterministic inter arrival time)
#         self.t_to_arrival[arrivals] = self.period[arrivals]

#         # 新增：统计本 slot 新到达设备数（外生负载）
#         self.info['new_devices'] += len(arrival_list)
        
#         return arrival_list, []

#     def reset_info(self):
#         # self.info = {'delay': 0, 'avg_rep': 0, 'devices': 0}
#         self.info = {
#         'delay': 0,
#         'avg_rep': 0,
#         'devices': 0,
#         'new_devices': 0,   # <-- 新增
#         }
#         self.slot_counter = 0
#         ################################
#         # 🔹补：优先级需要的键，初始化为 0/默认
#         self.info['type'] = 'mmtc'
#         self.info['viol'] = 0
#         # 为了统一结构，mMTC 也放上 cbr/vbr 两个键（始终为 0）
#         self.info['viol_cbr'] = 0
#         self.info['viol_vbr'] = 0

#     def reset_state(self):
#         self.state = np.full((len(self.state_variables)), 0, dtype = np.float32)

#     def get_n_variables(self):
#         return len(self.state_variables)

#     def get_state(self):
#         '''convert the info into a normalized vector'''
#         for i, var in enumerate(self.state_variables):
#             self.state[i] = self.info[var] / self.norm_const[var]        
#         return self.state

#     def update_info(self, delay, avg_rep, devices):
#         self.info['delay'] += delay
#         self.info['avg_rep'] += avg_rep
#         self.info['devices'] += devices
        

#     def compute_reward(self):
#         '''assesses SLA violations'''
#         #SLA_fulfilled = self.info['delay']/self.slots_per_step < self.SLA['delay']
#         #添加类型标注
#         viol = int(not (self.info['delay']/self.slots_per_step < self.SLA['delay']))
#         self.info["viol"] = viol
#         self.info["type"] = "mmtc"
#         self.info['viol_cbr'] = 0
#         self.info['viol_vbr'] = 0
#         return viol
#         #return not(SLA_fulfilled)

# class SliceRANeMBB:
#     '''
#     Generates arrivals and departures of eMBB ues.
#     There are two traffic types: CRB (GBR) and VBR (non-GBR)
#     CBR traffic parameters are given in CBR_description
#     VBR traffic parameters are given in VBR_description
#     '''
#     def __init__(self, rng, user_counter, id, SLA, CBR_description, VBR_description, state_variables, norm_const, slots_per_step, slot_length = 1e-3):
#         self.type = 'eMBB'
#         self.rng = rng
#         self.user_counter = user_counter
#         self.id = id
#         self.slot_length = slot_length
#         self.slots_per_step = slots_per_step
#         self.observation_time = slots_per_step * slot_length
#         self.SLA = SLA # service level agreement description
#         self.state_variables = state_variables
#         self.norm_const = norm_const

#         self.cbr_arrival_rate = CBR_description['lambda']
#         self.cbr_mean_time = CBR_description['t_mean']
#         self.cbr_bit_rate = CBR_description['bit_rate']

#         self.vbr_arrival_rate = VBR_description['lambda']
#         self.vbr_mean_time = VBR_description['t_mean']
#         self.vbr_source_data = {
#             'packet_size': VBR_description['p_size'],
#             'burst_size': VBR_description['b_size'],
#             'burst_rate':VBR_description['b_rate']
#         }
#         self.reset()

#     def reset(self):
#         self.slot_counter = 0
#         self.remaining_time = {}
#         self.cbr_steps_next_arrival = 0
#         self.vbr_steps_next_arrival = 0
#         self.vbr_ues = {}
#         self.cbr_ues = {}
#         self.reset_state()
#         self.reset_info()

#     def get_n_variables(self):
#         return len(self.state_variables)

#     def cbr_cac(self):
#         '''Admission control for CBR users'''
#         slots = max(self.slot_counter,1)
#         time = slots * self.slot_length
#         cbr_prb = self.info['cbr_prb'] / slots
#         cbr_th = self.info['cbr_th'] / time
#         if cbr_prb >= self.SLA['cbr_prb'] or cbr_th >= self.SLA['cbr_th']:
#             return False
#         return True

#     def cbr_arrivals(self):
#         if self.cbr_steps_next_arrival == 0:
#             # generate next arrival
#             inter_arrival_time = self.rng.exponential(1.0 / self.cbr_arrival_rate)
#             inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
#             self.cbr_steps_next_arrival = inter_arrival_time

#             if self.cbr_cac(): # check admission control
#                 # generate new user
#                 ue_id = next(self.user_counter)
#                 cbr_source = CbrSource(bit_rate = self.cbr_bit_rate)
#                 ue = UE(ue_id, self.id, cbr_source, CBR)
#                 self.cbr_ues[ue_id] = ue

#                 # generate holding time
#                 holding_time = self.rng.exponential(self.cbr_mean_time)
#                 holding_time = np.rint(holding_time / self.slot_length)
#                 self.remaining_time[ue_id] = holding_time

#                 return [ue] # return user
#         else:
#             self.cbr_steps_next_arrival -= 1    
#         return []

#     def vbr_arrivals(self):
#         if self.vbr_steps_next_arrival == 0:
#             # create new vbr user
#             ue_id = next(self.user_counter)
#             vbr_source = VbrSource(**self.vbr_source_data)
#             ue = UE(ue_id, self.id, vbr_source, VBR)
#             self.vbr_ues[ue_id] = ue

#             # generate holding time
#             holding_time = self.rng.exponential(self.vbr_mean_time)
#             holding_time = np.rint(holding_time / self.slot_length)
#             self.remaining_time[ue_id] = holding_time

#             # generate next arrival
#             inter_arrival_time = self.rng.exponential(1.0 / self.vbr_arrival_rate)
#             inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
#             self.vbr_steps_next_arrival = inter_arrival_time
#             return [ue]
#         else:
#             self.vbr_steps_next_arrival -= 1
#             return []

#     def departures(self):
#         departures = []
#         current_ids = list(self.remaining_time.keys())
#         for id in current_ids:
#             self.remaining_time[id] -= 1
#             if self.remaining_time[id] == 0:
#                 departures.append(id)
#                 del self.remaining_time[id] # delete timer
#                 self.vbr_ues.pop(id, None) # delete ue if here
#                 self.cbr_ues.pop(id, None) # or here    
#         return departures   

#     def slot(self):
#         self.slot_counter += 1
#         arrivals = self.cbr_arrivals()
#         arrivals.extend(self.vbr_arrivals())
#         departures = self.departures()
#         return arrivals, departures

#     def reset_info(self):
#         self.info = {'cbr_traffic': 0, 'cbr_th': 0, 'cbr_prb': 0, 'cbr_queue':0, 'cbr_snr': 0,\
#                     'vbr_traffic': 0, 'vbr_th': 0, 'vbr_prb': 0, 'vbr_queue': 0, 'vbr_snr': 0}
#         self.slot_counter = 0
#         #####################
#         # 🔹补：优先级需要的键，初始化为 0/默认
#         self.info['type'] = 'embb'
#         self.info['viol'] = 0
#         self.info['viol_cbr'] = 0
#         self.info['viol_vbr'] = 0

#     def reset_state(self):
#         self.state = np.full((len(self.state_variables)), 0, dtype = np.float32)
    
#     def update_info(self):
#         queue = 0
#         snr = 0
#         n = 0
#         for ue in self.cbr_ues.values():
#             self.info['cbr_traffic'] += ue.new_bits
#             self.info['cbr_th'] += ue.bits
#             self.info['cbr_prb'] += ue.prbs
#             queue += ue.queue
#             snr += ue.e_snr
#             n += 1
#         n = max(n,1)
#         self.info['cbr_queue'] += queue/n
#         self.info['cbr_snr'] += snr/n

#         queue = 0
#         snr = 0
#         n = 0
#         for ue in self.vbr_ues.values():
#             self.info['vbr_traffic'] += ue.new_bits
#             self.info['vbr_th'] += ue.bits
#             self.info['vbr_prb'] += ue.prbs
#             queue += ue.queue
#             snr += ue.e_snr
#             n += 1
#         n = max(n,1)
#         self.info['vbr_queue'] += queue/n
#         self.info['vbr_snr'] += snr/n

#     def compute_reward(self):
#         '''assesses SLA violations'''
#         cbr_th = self.info['cbr_th']/self.observation_time > self.SLA['cbr_th']
#         cbr_prb = self.info['cbr_prb']/self.slots_per_step > self.SLA['cbr_prb']
#         cbr_queue = self.info['cbr_queue']/self.slots_per_step < self.SLA['cbr_queue']
#         vbr_th = self.info['vbr_th']/self.observation_time > self.SLA['vbr_th']
#         vbr_prb = self.info['vbr_prb']/self.slots_per_step > self.SLA['vbr_prb']
#         vbr_queue = self.info['vbr_queue']/self.slots_per_step < self.SLA['vbr_queue']
#         # the slice has to guarantee the objective delay for cbr and vbr if their traffics do not surpass the maximum
#         cbr_fulfilled = cbr_th or cbr_prb or cbr_queue 
#         vbr_fulfilled = vbr_th or vbr_prb or vbr_queue
#         # cbr_fulfilled = cbr_th and cbr_prb and cbr_queue 
#         # vbr_fulfilled = vbr_th and vbr_prb and vbr_queue
#         SLA_fulfilled = cbr_fulfilled and vbr_fulfilled
#         #添加类型标注
#         viol = int(not SLA_fulfilled)
#         self.info["viol"] = viol
#         self.info["type"] = "embb"         # 标注类型
#         self.info["viol_cbr"] = int(not cbr_fulfilled)
#         self.info["viol_vbr"] = int(not vbr_fulfilled)
#         return viol
#         #return not(SLA_fulfilled)

#     def get_state(self):
#         '''converts the info into a normalized vector'''
#         for i, var in enumerate(self.state_variables):
#             self.state[i] = self.info[var] / self.norm_const[var]        
#         return self.state

