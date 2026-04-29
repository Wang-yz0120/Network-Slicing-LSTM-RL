#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate RL baselines (PPO2, A2C) and RaRPPO (PPO2+LSTM+Risk+Inertia)
in the 3 network-slicing scenarios from the original repo.

Outputs:
  - Baselines:
      ./results/scenario_{N}/{ALG}/
      ./trained_models/scenario_{N}/{ALG}/{ALG}_agent_{run}
  - RaRPPO:
      ./results/scenario_{N}/RaRPPO/
      ./trained_models/scenario_{N}/RaRPPO/RaRPPO_agent_{run}
"""

# ========== Silence noisy logs (must be before any TF/SB import) ==========
import os, warnings, logging
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # TF C++ logs -> ERROR only
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # force CPU to avoid CUDA warnings
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# logging.getLogger("absl").setLevel(logging.ERROR)
# logging.getLogger("gym").setLevel(logging.ERROR)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# ==========================================================================

import numpy as np
import concurrent.futures as cf
from numpy.random import default_rng
from itertools import product

from scenario_creator import create_env
# 可选：如果你仓库里有 ReportWrapper，就会被使用；没有也能跑
try:
    from wrapper import ReportWrapper
    HAS_REPORT_WRAPPER = True
except Exception:
    HAS_REPORT_WRAPPER = False

# --- stable-baselines v2 (not SB3) ---
from stable_baselines import PPO2, A2C
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import MlpLstmPolicy
from tensorflow import set_random_seed

# TF python 侧日志也降为 ERROR
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    pass

# === 引入我们新加的两个 wrapper ===
from wrappers.risk_penalty_wrapper import RiskPenaltyWrapper
from wrappers.one_sided_inertia import OneSidedInertia

# -------------------- 实验参数（与原版保持一致，可按需改） --------------------
RUNS = 5
PROCESSES = 4
TRAIN_STEPS = 256   # 建议为 PPO2 n_steps 的整数倍
EVALUATION_STEPS = 50
CONTROL_STEPS = 320
PENALTY = 2000
VERBOSE = False

run_list = list(range(RUNS))
scenarios = [1]

# 仅保留稳定的算法；需要可自行加回 SAC/TD3（Windows+TF1 下易崩）
algorithms = {
    'PPO2': PPO2,
    'A2C':  A2C,
    # 'TD3':  TD3,
    # 'SAC':  SAC,
    # 新增的 RaRPPO：占位（真正逻辑在 evaluate 分支里）
    'RaRPPO': PPO2,
}

deterministic = {
    'PPO2':   False,
    'A2C':    False,
    'RaRPPO': True,   # 评估阶段用确定性
}

# ================================ Evaluator ================================

class RLEvaluator():
    def __init__(self, scenario, algo_name, algorithm):
        self.scenario = scenario
        self.algo_name = algo_name
        self.algorithm = algorithm
        self.path = './results/scenario_{}/{}/'.format(scenario, algo_name)
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print ("Creation of the directory %s failed" % self.path)
            else:
                print ("Successfully created the directory %s " % self.path)

        self.model_path = './trained_models/scenario_{}/{}/'.format(scenario, algo_name)
        if not os.path.isdir(self.model_path):
            try:
                os.makedirs(self.model_path)
            except OSError:
                print ("Creation of the directory %s failed" % self.model_path)
            else:
                print ("Successfully created the directory %s " % self.model_path)

    def _build_baseline_env_and_model(self, i):
        """
        原始基线（PPO2/A2C）路径：不带风险和惯性。
        """
        rng = default_rng(seed=i)
        set_random_seed(i)
        node_env = create_env(rng, self.scenario, penalty=PENALTY)
        print('environment created')

        # 若存在 ReportWrapper，并且原来就用它记日志/切分训练与评估
        if HAS_REPORT_WRAPPER:
            node_env = ReportWrapper(node_env, steps=TRAIN_STEPS,
                                     control_steps=CONTROL_STEPS,
                                     env_id=i, path=self.path,
                                     verbose=VERBOSE)
            print('wrapped environment created')

        env = make_vec_env(lambda: node_env, n_envs=1)
        print('vectorised environment created')

        # 非 LSTM 的原始策略
        model = self.algorithm('MlpPolicy', env, verbose=0)
        return env, node_env, model

    def _build_rarppo_env_and_model(self, i):
        """
        RaRPPO 路径：PPO2 + MlpLstmPolicy + RiskPenaltyWrapper + OneSidedInertia。
        """
        rng = default_rng(seed=i)
        set_random_seed(i)
        base_env = create_env(rng, self.scenario, penalty=PENALTY)
        print('environment created')

        # 把 ReportWrapper 放里层，这样我们外层做的奖励整形会被记录下来
        env_core = base_env
        if HAS_REPORT_WRAPPER:
            env_core = ReportWrapper(env_core, steps=TRAIN_STEPS,
                                     control_steps=CONTROL_STEPS,
                                     env_id=i, path='./results/scenario_{}/RaRPPO/'.format(self.scenario),
                                     verbose=VERBOSE)
            print('wrapped environment created')

        # 在外层套 风险惩罚 + 单边惯性（顺序很重要）
        env_core = RiskPenaltyWrapper(env_core, lam=0.2, tau=0.05, lr=5e-4, l2=1e-6)
        env_core = OneSidedInertia(env_core, mu=0.05)

        env = make_vec_env(lambda: env_core, n_envs=1)
        print('vectorised environment created')

        # 关键：使用 LSTM 策略；nminibatches=1 以维持时间顺序
        model = PPO2(MlpLstmPolicy, env,
                     n_steps=512, nminibatches=1,
                     gamma=0.99, lam=0.95,
                     learning_rate=3e-4, ent_coef=0.01, vf_coef=0.5,
                     verbose=0)
        return env, env_core, model

    def evaluate(self, i):
        print('start evaluation of scenario {} run {} algorithm {}'.format(self.scenario, i, self.algo_name))

        # ===================== RaRPPO 分支 =====================
        if self.algo_name == 'RaRPPO':
            env, env_core, model = self._build_rarppo_env_and_model(i)

            print('scenario {}: run {} of algorithm RaRPPO ... '.format(self.scenario, i))
            model.learn(total_timesteps=TRAIN_STEPS)
            print('trainning done!')

            # 保存模型
            model_path = './trained_models/scenario_{}/RaRPPO/RaRPPO_agent_{}'.format(self.scenario, i)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            print('model saved')

            # ===== 方案A：通用评估（不依赖 ReportWrapper.obs）=====
            # det = True
            # obs = env.reset()
            # state = None
            # for _ in range(EVALUATION_STEPS):
            #     action, state = model.predict(obs, state=state, deterministic=det)
            #     obs, _, _, _ = env.step(action)
            # print('evaluation done')
            # ===== 评估（正确 reset LSTM）=====
            det = True
            obs = env.reset()          # VecEnv: obs shape = (n_envs, obs_dim)
            state = None               # LSTM 初始状态
            dones = [False]            # VecEnv 的 done 掩码（n_envs=1）

            for _ in range(EVALUATION_STEPS):
                # 传入 mask，告知哪些 env 在上一步已经 done（需要重置 LSTM 状态）
                action, state = model.predict(obs, state=state, mask=dones, deterministic=det)
                obs, _, dones, _ = env.step(action)

                # 可选：当 episode 结束时也显式清空 state（n_envs=1 时更直观）
                if dones[0]:
                    state = None
            print('evaluation done')

            # 如果 ReportWrapper 在内层，尝试保存结果（可选）
            try:
                inner = env.envs[0]  # OneSidedInertia
                if hasattr(inner, 'env'): inner = inner.env  # RiskPenaltyWrapper
                if hasattr(inner, 'env'): inner = inner.env  # ReportWrapper 或 RanSlice
                if hasattr(inner, 'save_results'):
                    inner.save_results()
                    print('results saved')
            except Exception as e:
                print('save_results skipped:', e)

            return

        # ===================== 原始基线分支 =====================
        env, node_env, model = self._build_baseline_env_and_model(i)
        print('scenario {}: run {} of algorithm {} ... '.format(self.scenario, i, self.algo_name))
        model.learn(total_timesteps=TRAIN_STEPS)
        print('trainning done!')

        # 保存模型
        model_path = '{}{}_agent_{}'.format(self.model_path, self.algo_name, i)
        model.save(model_path)
        print('model saved')

        # 评估阶段
        if HAS_REPORT_WRAPPER and hasattr(node_env, "set_evaluation"):
            node_env.set_evaluation(EVALUATION_STEPS)
            obs = node_env.obs
            det = deterministic.get(self.algo_name, False)
            action, state = model.predict(obs, deterministic=det)
            for _ in range(EVALUATION_STEPS):
                action, state = model.predict(obs, state=state, deterministic=det)
                obs, _, _, _ = node_env.step(action)
            print('evaluation done')
            if hasattr(node_env, "save_results"):
                node_env.save_results()
                print('results saved')
        else:
            det = deterministic.get(self.algo_name, False)
            obs = env.reset()
            for _ in range(EVALUATION_STEPS):
                action, _ = model.predict(obs, deterministic=det)
                obs, _, _, _ = env.step(action)
            print('evaluation done')


# ================================== main ===================================

if __name__=='__main__':
    for scenario, (alg_name, alg) in product(scenarios, algorithms.items()):
        evaluator = RLEvaluator(scenario, alg_name, alg)

        # 顺序执行（也可改回并行）
        # with cf.ProcessPoolExecutor(PROCESSES) as E:
        #     results = E.map(evaluator.evaluate, run_list)

        for run in run_list:
            evaluator.evaluate(run)
