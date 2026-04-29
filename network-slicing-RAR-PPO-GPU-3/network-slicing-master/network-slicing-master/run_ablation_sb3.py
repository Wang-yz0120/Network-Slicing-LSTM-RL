#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

from experiment_rl_sb3 import ExperimentConfig, run_experiment


# =========================
# Editable config
# 直接修改这里，然后运行:
# python run_ablation_sb3.py
# =========================
SCENARIO_ID = 1
#VARIANT_NAME = "all" 
VARIANT_NAME = "full"  # "all", "full", "no_pred", "no_cross", "raw_reward"
SEEDS = [0, 1, 2]
ONLINE_FINETUNE = False
DEBUG = False
SMOKE = False
PRETRAINED_PATH = "./traffic_lstm_pretrained_3N_multiseed_all.pth"


VARIANTS = {
    "full": dict(
        algo_name="LSTMDDPG",
        use_prediction=True,
        use_cross_step=True,
        use_dense_reward=True,
    ),
    "no_pred": dict(
        algo_name="DDPG",
        use_prediction=False,
        use_cross_step=True,
        use_dense_reward=True,
    ),
    "no_cross": dict(
        algo_name="LSTMDDPG",
        use_prediction=True,
        use_cross_step=False,
        use_dense_reward=True,
    ),
    "raw_reward": dict(
        algo_name="LSTMDDPG",
        use_prediction=True,
        use_cross_step=True,
        use_dense_reward=False,
    ),
}


def build_cfg(variant_name: str, seed: int) -> ExperimentConfig:
    base = ExperimentConfig(
        scenario_id=SCENARIO_ID,
        exp_tag=variant_name,
        seed=seed,
        online_finetune=ONLINE_FINETUNE,
        debug=DEBUG,
    )
    if SMOKE:
        base.train_steps_a = 256
        base.train_steps_b = 0
        base.train_steps_c = 0
        base.eval_steps = 64
    if PRETRAINED_PATH:
        base.pretrained_path = PRETRAINED_PATH

    variant = deepcopy(VARIANTS[variant_name])
    for k, v in variant.items():
        setattr(base, k, v)
    return base


def main():
    variant_names = list(VARIANTS.keys()) if VARIANT_NAME == "all" else [VARIANT_NAME]
    for variant_name in variant_names:
        for run_idx, seed in enumerate(SEEDS):
            cfg = build_cfg(variant_name, seed)
            print(f"[ABLATION] variant={variant_name} run={run_idx} seed={seed}")
            run_experiment(cfg, run_idx=run_idx)


if __name__ == "__main__":
    main()
