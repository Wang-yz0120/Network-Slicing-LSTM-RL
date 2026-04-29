#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate KBRL in the current Gymnasium/SB3-compatible network-slicing setup.

Outputs:
  - ./results/scenario_{N}/KBRL_XX/results_{run}.npz
"""

import os
import warnings
from itertools import product

import numpy as np
from numpy import savez
from numpy.random import default_rng

from scenario_creator import create_env, create_kbrl_agent
from wrappers.delay_action_wrapper import DelayActionWrapper
from wrappers.sla_dense_reward_wrapper import SlaDenseRewardWrapper


RUNS = 1
TRAIN_STEPS_A = 51200
TRAIN_STEPS_B = 51200
TRAIN_STEPS_C = 51200
# TRAIN_STEPS_A = 2560
# TRAIN_STEPS_B = 2560
# TRAIN_STEPS_C = 2560
EVALUATION_STEPS = 2560
EVAL_WARMUP_STEPS = 100
PENALTY = 1000
DELAY_STEPS = 1
CONTROL_STEPS = 1000

run_list = list(range(RUNS))
scenarios = [1]
# accuracy_list = [[0.97, 0.99], [0.99, 0.999]]
accuracy_list = [[0.97, 0.99]]

def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


class KBRLEvaluator:
    def __init__(self, scenario: int, accuracy_range):
        self.scenario = scenario
        self.accuracy_range = list(accuracy_range)
        self.algo_name = f"KBRL_{int(self.accuracy_range[0] * 100)}"
        self.res_path = f"./results/scenario_{scenario}/{self.algo_name}/"
        _ensure_dir(self.res_path)

    def evaluate(self, run_idx: int):
        print(
            f"start evaluation of scenario {self.scenario} run {run_idx} "
            f"algorithm {self.algo_name}"
        )

        rng = default_rng(seed=run_idx)
        env = create_env(rng, self.scenario, penalty=PENALTY)
        env = SlaDenseRewardWrapper(
            env,
            barrier=10.0,
            gamma=3.0,
            kappa=3.0,
            alpha=0.1,
            beta=0.0,
            clip_abs=2.0,
        )
        n_slices = int(getattr(getattr(env, "node_b", None), "n_slices_l1", 0))
        n_prbs = int(getattr(getattr(env, "node_b", None), "n_prbs", 0))
        default_action = np.zeros((2 * n_slices,), dtype=np.int16)
        if n_slices > 0 and n_prbs > 0:
            base = n_prbs // n_slices
            default_action[n_slices:] = base
            default_action[n_slices:n_slices + (n_prbs - base * n_slices)] += 1
        env = DelayActionWrapper(
            env,
            delay_steps=DELAY_STEPS,
            default_action=default_action,
            debug=False,
        )
        print(f"run {run_idx}: Environment created")

        kbrl_agent = create_kbrl_agent(rng, self.scenario, accuracy_range=self.accuracy_range)
        print(f"run {run_idx}: KBRL agent created")

        total_train_steps = TRAIN_STEPS_A + TRAIN_STEPS_B + TRAIN_STEPS_C
        total_steps = total_train_steps + EVAL_WARMUP_STEPS + EVALUATION_STEPS
        file_path = os.path.join(self.res_path, f"results_{run_idx}.npz")
        save_extras = {
            "train_steps": np.array([total_train_steps], dtype=np.int32),
            "eval_warmup_steps": np.array([EVAL_WARMUP_STEPS], dtype=np.int32),
            "evaluation_steps": np.array([EVALUATION_STEPS], dtype=np.int32),
            "accuracy_low": np.array([self.accuracy_range[0]], dtype=np.float32),
            "accuracy_high": np.array([self.accuracy_range[1]], dtype=np.float32),
        }
        results = kbrl_agent.run(
            env,
            total_steps,
            learning_time=total_train_steps,
            save_every=CONTROL_STEPS,
            save_path=file_path,
            save_extras=save_extras,
        )
        print(f"run {run_idx}: KBRL run finished")

        print(f"run {run_idx}: Results saved to {file_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    for scenario, accuracy_range in product(scenarios, accuracy_list):
        evaluator = KBRLEvaluator(scenario, accuracy_range)
        for run in run_list:
            evaluator.evaluate(run)
