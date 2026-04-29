#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Defines two functions:

create_env
create_kbrl_agent

"""
import gym_ran_slice 
import gymnasium as gym
from gym_ran_slice.ran_slice import RanSlice
from itertools import count
from node_b import NodeB
from slice_l1 import SliceL1eMBB, SliceL1mMTC
from slice_ran import SliceRANmMTC, SliceRANeMBB
from schedulers import ProportionalFair
from channel_models import SINRSelectiveFading, MCSCodeset
from kbrl_control import KBRL_Control, Learner
from algorithms.kernel import GaussianKernel
from algorithms.projectron import SVvariable, Projectron
from wrappers.gym21_compat import Gym21Compat   # ← 新增导入

# ----------------- scenario parameters ------------------------

scenario_1 = {
    'n_prbs': 200,
    'n_embb': 5,
    'n_mmtc': 0
}

scenario_2 = {
    'n_prbs': 200,
    'n_embb': 3,
    'n_mmtc': 2
}

scenario_3 = {
    'n_prbs': 100,
    'n_embb': 1,
    'n_mmtc': 4
}

scenario_4 = {
    'n_prbs': 70,
    'n_embb': 1,
    'n_mmtc': 1
}

scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]


# -------------------- eMBB parameters -------------------------

# CBR_description = {
# #    'lambda': 1.0/60.0, # low traffic
#     'lambda': 2.0/60.0,
#     't_mean': 30.0,
#     'bit_rate': 500000
# }

# CBR_description = {
#     'lambda': 4.0/60.0,   # 原 2/60 → 4/60
#     't_mean': 20.0,       # 原 30 → 20
#     'bit_rate': 800000    # 原 500 kbps → 800 kbps
# }
# CBR_description = {
#     'lambda': 8.0/60.0,   # 原 2/60 → 4/60 CBR UE到达率 单位1/s
#     't_mean': 20.0,       # 原 30 → 20 CBR UE的“在线/连接”平均时长，单位s
#     'bit_rate': 300000    # 原 500 kbps → 800 kbps 单个CBR UE的恒定比特率，单位bit/s
# }
CBR_description = {
  'lambda_n': 2.0/60.0,       # normal 2 users/min
  'lambda_b': 8.0/60.0,       # burst  8 users/min (kappa=4)
  'mean_normal': 10.0,        # normal state avg duration (sec)
  'mean_burst':  2.0,         # burst state avg duration (sec)
  't_mean': 30.0,
  'bit_rate': 500000
}

# VBR_description = {
# #    'lambda': 1.0/60.0, # low traffic
#     'lambda': 5.0/60.0,
#     't_mean': 30.0,
#     'p_size': 1000,
#     'b_size': 500,
#     'b_rate': 1
# }

# VBR_description = {
#     'lambda': 10.0/60.0,   # 原来 5/60，提到 10/60（更频繁）
#     't_mean': 20.0,        # 原来 30，缩短保持时长以增替换频率
#     'p_size': 1500,        # 原来 1000，单包更大
#     'b_size': 1200,        # 原来 500，突发更长
#     'b_rate': 2            # 原来 1，突发内更快
# }

# VBR_description = {
#     'lambda': 20.0/60.0,   # 原来 5/60，提到 10/60（更频繁）VBR UE到达率，单位1/s
#     't_mean': 20.0,        # 原来 30，缩短保持时长以增替换频率 VBR UE在线时长均值，单位s
#     'p_size': 600,        # 原来 1000，单包更大 每个活跃突发源在每个 slot 产生的比特数，单位 bit/slot
#     'b_size': 500,        # 原来 500，突发更长 单个突发持续的平均长度，单位slot
#     'b_rate': 1            # 原来 1，突发内更快 突发到达强度，单位burst/s
# }

VBR_description = {
  'lambda_n': 5.0/60.0,       # normal 5 users/min
  'lambda_b': 25.0/60.0,      # burst  25 users/min (kappa=5)
  'mean_normal': 10.0,
  'mean_burst':  2.0,
  't_mean': 30.0,
  'p_size': 1000,             # 1 Mb/s when slot=1ms
  'b_size': 500,
  'b_rate': 1
}

SLA_embb = {
    'cbr_th': 10e6, 
    'cbr_prb': 20, # 30
    'cbr_queue': 10e4, # 5e4
    'vbr_th': 15e6, # 10e6 
    'vbr_prb': 30, # 40
    'vbr_queue': 15e4
    }

# SLA_embb = {
#     'cbr_th': 15e6,  # 原 10e6 → 15e6（要求更高）
#     'cbr_prb': 30,   # 原 20 → 30
#     'cbr_queue': 5e4,# 原 10e4 → 5e4（更严格）
#     'vbr_th': 20e6,  # 原 15e6 → 20e6
#     'vbr_prb': 40,   # 原 30 → 40
#     'vbr_queue': 8e4 # 原 15e4 → 8e4
# }

state_variables_embb = ['cbr_traffic','cbr_th', 'cbr_prb', \
                        'cbr_queue', 'cbr_snr', 'vbr_traffic', \
                        'vbr_th', 'vbr_prb', 'vbr_queue', 'vbr_snr']

# -------------------- mMTC parameters -------------------------

MTC_description = {
    'n_devices': 1000,
    'repetition_set': [2,4,8,16,32,64,128],
    'period_set': [1000, 50000, 10000, 15000, 20000, 25000, 50000, 100000]
}

state_variables_mmtc = ['devices', 'avg_rep', 'delay']

SLA_mmtc = {
    'delay': 300
}

# SLA_mmtc = { 'delay': 200 }   # 原 300 → 200

# -------------------- create environment -------------------------

def create_env(rng, n, slots_per_step = 50, propagation_type = 'macro_cell_urban_2GHz', L1_level = True, penalty = 100):
    '''
    Returns slice ran environment:
    - rng: for random number generation
    - n: selects the scenario (0, 1, 2)
    '''
    time_per_step = slots_per_step * 1e-3

    sc = scenarios[n]
    n_prbs = sc['n_prbs']
    n_embb = sc['n_embb']
    n_mmtc = sc['n_mmtc']

    # -------------------- eMBB normalization constants ----------------------

    norm_const_embb = {
        'cbr_traffic': 5e6 * time_per_step,
        'cbr_th': 10e6 * time_per_step,
        'cbr_prb': 25 * slots_per_step,
        'cbr_queue': 10e4 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,
        'vbr_traffic': 5e6 * time_per_step, 
        'vbr_th': 10e6 * time_per_step, 
        'vbr_prb': 35 * slots_per_step, 
        'vbr_queue': 10e4 * slots_per_step, 
        'vbr_snr': 35 * slots_per_step
    }

    # -------------------- mMTC normalization constants -----------------------

    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay': 100 * slots_per_step
    }

    # ------------------- auxiliary functions -----------------------

    def new_slice_mmtc(id, rng):
        return SliceRANmMTC(rng, id, SLA_mmtc, MTC_description, state_variables_mmtc, norm_const_mmtc, slots_per_step)

    def new_slice_embb(id, rng, user_counter):
        return SliceRANeMBB(rng, user_counter, id, SLA_embb, CBR_description, VBR_description, state_variables_embb, norm_const_embb, slots_per_step, n_prbs, snr_generator)

    # ------------------- environment creation ------------------------

    snr_generator = SINRSelectiveFading(rng, propagation_type, n_prbs = n_prbs)

    mcs_codeset = MCSCodeset()

    scheduler = ProportionalFair(mcs_codeset)

    user_counter = count()

    slices_l1 = []

    if L1_level: # each slice has its own L1 resources

        for id in range(n_embb):
            slices_ran_embb = [new_slice_embb(id, rng, user_counter)]
            slice_l1_embb = SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler)
            slices_l1.append(slice_l1_embb)

        for id in range(n_mmtc):
            slices_ran_mmtc = [new_slice_mmtc(id, rng)]
            slice_l1_mmtc = SliceL1mMTC(5, slices_ran_mmtc)
            slices_l1.append(slice_l1_mmtc)

    else: # slices are multiplexed in the L1 (the scheduler should handle ues from different slices) 

        slices_ran_embb = [new_slice_embb(id, rng, user_counter) for id in range(n_embb)]
        slice_l1_embb = SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler)
        slices_l1 = [slice_l1_embb]

        if n_mmtc > 0:
            slices_ran_mmtc = [new_slice_mmtc(id, rng) for id in range(n_mmtc)]
            slice_l1_mmtc = SliceL1mMTC(5, slices_ran_mmtc)
            slices_l1.append(slice_l1_mmtc)

    node = NodeB(slices_l1, slots_per_step, n_prbs)

    # # node_env = gym.make("gym_ran_slice/RanSlice-v1", node_b = node, penalty = penalty)
    # node_env = gym.make("gym_ran_slice/RanSlice-v1", node_b=node, penalty=penalty)
    # # 立刻套上旧→新接口适配（非常重要）
    # node_env = Gym21Compat(node_env)
        # 先直接实例化老版环境（不经 gym.make）
    raw_env = RanSlice(node_b=node, penalty=penalty)

    # 最内层立刻套上兼容器（拦截 seed / options）
    node_env = Gym21Compat(raw_env)

    return node_env

# ------------ KBRL Learner initialization values ------------------

alfa = 0.05 # learning parameter

# initial offset and initial action are initialized at random
embb_sec = (2, 8)
embb_a = (4, 20)
mmtc_sec = (1, 4)
mmtc_a = (2, 10)

# -------------------- create KBRL agent -------------------------

def create_kbrl_agent(rng, n, accuracy_range = [0.99, 0.999]):
    '''
    Returns kbrl agent:
    - rng: for random number generation
    - n: selects the scenario (0, 1, 2)
    - accuracy_range: for the learner
    - budget: number of support vectors in memory
    '''
    sc = scenarios[n]
    n_prbs = sc['n_prbs']
    n_embb = sc['n_embb']
    n_mmtc = sc['n_mmtc']
    embb_dim = len(state_variables_embb)
    mmtc_dim = len(state_variables_mmtc)

    learners = [] 
    i = 0

    # create one learner instance per slice
    for _ in range(n_embb):
        sv = SVvariable() # create support vector memory
        kernel = GaussianKernel(sv,1) # kernel
        algorithm = Projectron(kernel) # online classifier
        initial_action = rng.integers(embb_a[0], embb_a[1])
        sec = rng.integers(embb_sec[0], embb_sec[1])
        learner = Learner(algorithm, slice(i,i+embb_dim), initial_action, sec)
        learners.append(learner)
        i += embb_dim

    for _ in range(n_mmtc):
        sv = SVvariable()
        kernel = GaussianKernel(sv,1)
        algorithm = Projectron(kernel)
        initial_action = rng.integers(mmtc_a[0], mmtc_a[1])
        sec = rng.integers(mmtc_sec[0], mmtc_sec[1])
        learner = Learner(algorithm, slice(i,i+mmtc_dim), initial_action, sec)
        learners.append(learner)
        i += mmtc_dim

    kbrl_agent = KBRL_Control(learners, n_prbs, alfa = alfa, accuracy_range = accuracy_range)

    return kbrl_agent