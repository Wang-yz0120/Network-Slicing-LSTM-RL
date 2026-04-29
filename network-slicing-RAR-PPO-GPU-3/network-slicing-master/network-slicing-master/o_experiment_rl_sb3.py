#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate baselines (PPO) and RaRPPO (RecurrentPPO + LSTM + Risk)
in the network-slicing scenarios (SB3/PyTorch).
增添lstm效果：将 rnn/* 与 env/* 指标写入 TensorBoard + CSV（供绘图脚本使用）

Outputs:
  - ./results/scenario_{N}/{ALG}/
  - ./trained_models/scenario_{N}/{ALG}/{ALG}_agent_{run}.zip
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import sys, os, atexit, datetime, re

# ====== SB3 / Gymnasium ======
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
import gymnasium as gym

# ====== Project code ======
from scenario_creator import create_env
from lstm_predict_wrapper import LSTMPredictWrapper
from wrapper import ReportWrapper
from wrappers.risk_penalty_wrapper import RiskPenaltyWrapper
from wrappers.sla_dense_reward_wrapper import SlaDenseRewardWrapper
from wrappers.priority_reward_wrapper import PriorityRewardWrapper

# -------------------- Config --------------------
RUNS = 1
# TRAIN_STEPS_A = 307200
# TRAIN_STEPS_B = 307200
# TRAIN_STEPS_C = 307200
TRAIN_STEPS_A = 20480
TRAIN_STEPS_B = 20480
TRAIN_STEPS_C = 20480
# TRAIN_STEPS_A = 10240
# TRAIN_STEPS_B = 10240
# TRAIN_STEPS_C = 10240
# TRAIN_STEPS_A = 512
# TRAIN_STEPS_B = 512
# TRAIN_STEPS_C = 512
# EVALUATION_STEPS = 10240
EVALUATION_STEPS = 256
CONTROL_STEPS = 1000
PENALTY = 1000
VERBOSE = False

# 向量化环境与训练批量
N_ENVS   = 1
N_STEPS  = 256
BATCH_SZ = N_ENVS * N_STEPS
N_EPOCHS = 10

# 学习率 / clip 的线性衰减（progress_remaining: 1->0）
def linear_schedule(start, end):
    def f(progress_remaining):
        return float(end + (start - end) * progress_remaining)
    return f

LR_SCHEDULE   = linear_schedule(3e-4, 1e-4)
CLIP_SCHEDULE = linear_schedule(0.2, 0.1)

# ent_coef 不能用 schedule，用常数
# ENT_COEF  = 0.003
ENT_COEF  = 0.01
TARGET_KL = 0.05

# 评估设置
EVAL_WARMUP_STEPS = 100
deterministic = {'PPO': False, 'RaRPPO': False}
# deterministic = {'PPO': True, 'RaRPPO': True}
run_list = list(range(RUNS))
scenarios = [1]
prio = {"mmtc": 3.0, "embb_cbr": 2.0, "embb_vbr": 1.0, "embb": 1.5}

algorithms = {
    'RaRPPO': PPO,
    'PPO': PPO,
}

# ====================== Utils ======================

def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _unwrap_chain(vec_env):
    chain = []
    cur = vec_env.envs[0] if hasattr(vec_env, "envs") and len(vec_env.envs) > 0 else vec_env
    while True:
        chain.append(cur)
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return chain

def _find_wrapper(outer_env, cls, max_depth: int = 30):
    cur = outer_env
    for _ in range(max_depth):
        if isinstance(cur, cls):
            return cur
        cur = getattr(cur, "env", None)
        if cur is None:
            break
    return None

def _force_save(vec_env, run_idx):
    chain = _unwrap_chain(vec_env)
    for obj in chain:
        if hasattr(obj, "save_results") and callable(getattr(obj, "save_results")):
            try:
                obj.save_results()
                print(f"[FORCE SAVE] results saved via {type(obj).__name__} for run {run_idx} | chain: " +
                      " -> ".join(type(x).__name__ for x in chain))
                return
            except Exception as e:
                print(f"[FORCE SAVE] save_results() on {type(obj).__name__} raised: {e}")
                return
    print("[FORCE SAVE] No save_results() found. Chain: " + " -> ".join(type(x).__name__ for x in chain))

def _find_lstm_wrapper(env):
    """
    从多层 wrapper 里往内找 LSTMPredictWrapper。
    env 可能是 ReportWrapper / DummyVecEnv 里的单个 env 等。
    """
    cur = env
    for _ in range(10):  # 最多往里剥 10 层，足够用
        if isinstance(cur, LSTMPredictWrapper):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return None

# ============== Build vec env (每个 worker 单独创建) ==============
def _make_wrapped_vec_env(scenario_id: int, run_idx: int, total_steps: int,
                          control_steps: int, save_dir: str, algo_name: str):
    def _make_single_env_thunk(worker_idx: int):
        def _thunk():
            rng = np.random.default_rng(seed=run_idx * 1000 + worker_idx)
            env = create_env(rng, scenario_id, penalty=PENALTY)

            n_prbs = getattr(getattr(env, "node_b", None), "n_prbs", None)
            n_prbs = int(n_prbs) if n_prbs is not None else None

            env = SlaDenseRewardWrapper(env, barrier=5.0, gamma=3.0, kappa=3.0, alpha=1.0, beta = 0.2, clip_abs=1.0)
            #env = PriorityRewardWrapper(env, weights=prio, lam=0.0, cap=10.0)
            #env = RiskPenaltyWrapper(env, lam=0.0, tau=0.9, lr=5e-4, l2=1e-6)

            #只有 RaRPPO 才接上 LSTM 预测 wrapper
            if algo_name == "RaRPPO":
                env = LSTMPredictWrapper(
                    env,
                    history_len=10,
                    hidden_size=128,
                    lr=1e-3,
                )

            total_with_eval = int(total_steps + EVALUATION_STEPS)
            env = ReportWrapper(env, steps=total_with_eval, control_steps=control_steps,
                                env_id=run_idx, path=save_dir, verbose=VERBOSE, n_prbs=n_prbs)
            return env
        return _thunk

    thunks = [_make_single_env_thunk(i) for i in range(N_ENVS)]
    vec_env = DummyVecEnv(thunks)
    outer_env0 = vec_env.envs[0]
    return vec_env, outer_env0

# -------- apply stage params to ALL sub-envs --------
def _set_stage_params(target_env_or_vec, stage: str):
    """
    Accepts either a single wrapped env or a VecEnv (with .envs),
    and applies stage parameters to ALL sub-envs.
    """
    env_list = getattr(target_env_or_vec, "envs", None)
    if env_list is None:
        env_list = [target_env_or_vec]

    for outer_report_env in env_list:
        prio = _find_wrapper(outer_report_env, PriorityRewardWrapper)
        dense = _find_wrapper(outer_report_env, SlaDenseRewardWrapper)
        risk  = _find_wrapper(outer_report_env, RiskPenaltyWrapper)

        if dense:
            dense.set_alpha(0.1)
            dense.set_clip(2.0)   # 放宽限幅，让 SLA 信号更明显

        if stage == "A":
            if prio: prio.set_lambda(5e-3); prio.set_cap(10.0)
            if risk: setattr(risk, "lam", 0.0)
        elif stage == "B":
            if prio: prio.set_lambda(1e-2); prio.set_cap(8.0)
            if risk:
                setattr(risk, "lam", 5e-4)
                setattr(risk, "tau", 0.9)
        else:  # C
            if prio: prio.set_lambda(2e-2); prio.set_cap(6.0)
            if risk:
                setattr(risk, "lam", 1e-3)
                setattr(risk, "tau", 0.95)

# ====================== Evaluator ======================
class RLEvaluator():
    def __init__(self, scenario: int, algo_name: str):
        self.scenario = scenario
        self.algo_name = algo_name
        self.res_path   = f'./results/scenario_{scenario}/{algo_name}/'
        self.model_path = f'./trained_models/scenario_{scenario}/{algo_name}/'
        _ensure_dir(self.res_path); _ensure_dir(self.model_path)

    # --- PPO ---
    def _build_ppo(self, vec_env):
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        return PPO(
            "MlpPolicy", vec_env,
            n_steps=N_STEPS, batch_size=BATCH_SZ, n_epochs=N_EPOCHS,
            gamma=0.99, gae_lambda=0.95,
            learning_rate=LR_SCHEDULE, ent_coef=ENT_COEF, vf_coef=0.5,
            clip_range=CLIP_SCHEDULE, target_kl=TARGET_KL, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tb", device="cuda", verbose=0
        )

    # --- RaRPPO ---
    def _build_rarppo(self, vec_env):
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        return PPO(
            "MlpPolicy", vec_env,
            n_steps=N_STEPS, batch_size=BATCH_SZ, n_epochs=N_EPOCHS,
            gamma=0.99, gae_lambda=0.95,
            learning_rate=LR_SCHEDULE, ent_coef=ENT_COEF, vf_coef=0.5,
            clip_range=CLIP_SCHEDULE, target_kl=TARGET_KL, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tb", device="cuda", verbose=0
        )

    def evaluate(self, run_idx: int):
        print(f'start evaluation of scenario {self.scenario} run {run_idx} algorithm {self.algo_name}')
        set_random_seed(run_idx)

        total_steps = TRAIN_STEPS_A + TRAIN_STEPS_B + TRAIN_STEPS_C
        vec_env, outer_env0 = _make_wrapped_vec_env(self.scenario, run_idx, total_steps,
                                                    CONTROL_STEPS, self.res_path, algo_name=self.algo_name)

        if self.algo_name == "PPO":
            model = self._build_ppo(vec_env)
        else:
            model = self._build_rarppo(vec_env)

        # -------- apply to the whole vec_env --------
        _set_stage_params(vec_env, "A")
        if TRAIN_STEPS_A > 0:
            model.learn(
                total_timesteps=TRAIN_STEPS_A,
                progress_bar=True,
                reset_num_timesteps=True
            )

        _set_stage_params(vec_env, "B")
        if TRAIN_STEPS_B > 0:
            model.learn(
                total_timesteps=TRAIN_STEPS_B,
                progress_bar=True,
                reset_num_timesteps=False  # 连续时间轴
            )

        if TRAIN_STEPS_C > 0:
            _set_stage_params(vec_env, "C")
            model.learn(
                total_timesteps=TRAIN_STEPS_C,
                progress_bar=True,
                reset_num_timesteps=False  # 连续时间轴
            )

        print('trainning done!')
        save_path = f'{self.model_path}{self.algo_name}_agent_{run_idx}'
        model.save(save_path + ".zip")
        print('model saved')

        # 训练结束后，如果是 RaRPPO，就尝试画一张 LSTM loss 曲线
        if self.algo_name == "RaRPPO":
            lstm_env = _find_lstm_wrapper(outer_env0)
            if lstm_env is not None and hasattr(lstm_env, "loss_history"):
                losses = lstm_env.loss_history
                if len(losses) > 0:
                    npz_path = os.path.join(self.res_path, f"lstm_loss_run{run_idx}.npz")
                    np.savez(npz_path, loss=np.asarray(losses, dtype=np.float32))
                    print(f"[INFO] LSTM loss history saved to: {npz_path}")

                    plt.figure()
                    plt.plot(range(1, len(losses) + 1), losses)
                    plt.xlabel("LSTM train step")
                    plt.ylabel("MSE loss")
                    plt.title(f"LSTM online training loss (run {run_idx})")
                    fig_path = os.path.join(self.res_path, f"lstm_loss_run{run_idx}.png")
                    plt.tight_layout()
                    plt.savefig(fig_path)
                    plt.close()
                    print(f"[INFO] LSTM loss curve saved to: {fig_path}")
                else:
                    print("[INFO] LSTM loss_history is empty, nothing to plot.")
            else:
                print("[INFO] LSTMPredictWrapper not found, skip LSTM loss plot.")

        det = deterministic.get(self.algo_name, False)
        obs = vec_env.reset()

        # ---- 可选热身（EVAL_WARMUP_STEPS=0 时不执行）----
        if EVAL_WARMUP_STEPS > 0:
            print(f"[EVAL] warmup {EVAL_WARMUP_STEPS} steps (det=False, not recorded)")
            for _ in range(EVAL_WARMUP_STEPS):
                action, _ = model.predict(obs, deterministic=False)
                obs, _, _, _ = vec_env.step(action)

        print(f"[EVAL] start evaluation (deterministic={det})")
        for _ in range(EVALUATION_STEPS):
            action, _ = model.predict(obs, deterministic=det)
            obs, _, _, _ = vec_env.step(action)

        print('evaluation done')

        _force_save(vec_env, run_idx)

# ================================== main ===================================
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    _buf_out, _buf_err = [], []
    _orig_write_out = sys.stdout.write
    _orig_write_err = sys.stderr.write

    def _tap_out(s):
        _buf_out.append(s)
        return _orig_write_out(s)

    def _tap_err(s):
        _buf_err.append(s)
        return _orig_write_err(s)

    sys.stdout.write = _tap_out
    sys.stderr.write = _tap_err

    def _ts():
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def _make_logfile():
        log_dir = os.path.join(".", "results", "console_logs")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"snapshot_{_ts()}.txt")

    _ANSI_CSI_PATTERN = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')

    def _strip_ansi(s: str) -> str:
        return _ANSI_CSI_PATTERN.sub('', s)

    def _collapse_carriage(s: str) -> str:
        out_lines = []
        cur_line = []
        for ch in s:
            if ch == '\r':
                cur_line = []
            elif ch == '\n':
                out_lines.append(''.join(cur_line))
                cur_line = []
            else:
                cur_line.append(ch)
        if cur_line:
            out_lines.append(''.join(cur_line))
        return '\n'.join(out_lines)

    def _final_snapshot(raw: str) -> str:
        s = _strip_ansi(raw)
        s = _collapse_carriage(s)
        s = '\n'.join([ln for ln in s.splitlines() if ln.strip() != '' or True])
        return s

    def _dump_console_snapshot():
        try:
            raw = ''.join(_buf_out) + ''.join(_buf_err)
            snap = _final_snapshot(raw)
            path = _make_logfile()
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(snap)
            _orig_write_out(f"\n[SNAPSHOT] console snapshot saved to: {path}\n")
        except Exception as e:
            _orig_write_err(f"\n[SNAPSHOT] failed to save console snapshot: {e}\n")

    atexit.register(_dump_console_snapshot)

    for scenario, alg_name in product(scenarios, algorithms.keys()):
        evaluator = RLEvaluator(scenario, alg_name)
        for run in run_list:
            evaluator.evaluate(run)



