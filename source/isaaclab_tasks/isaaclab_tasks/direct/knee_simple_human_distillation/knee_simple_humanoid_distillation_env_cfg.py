# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class KneeSimpleHumanoidDistillationEnvCfg(DirectRLEnvCfg):
    """Knee Lower Humanoid Distillation environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces
    observation_space = 66  # 81-12-3 减去线速度
    action_space = 28+2  # 添加两个膝关节动作
    state_space = 0
    # num_amp_observations = 2
    # amp_observation_space = 69

    is_distillation = True  # 启用蒸馏模式
    teacher_policy_path = "/home/hy/IsaacLab/logs/skrl/d_knee_simple_humanoid_amp/2025-11-15_14-46-05_amp_torch/checkpoints/agent_200000.pt"
    teacher_obs_dim = 69   # 教师原始观测维度（与SKRL训练一致）
    student_obs_dim = 30

    early_termination = True
    termination_height = 0.5  # 0.5

    # motion_file: str = MISSING
    reference_body = "torso"
    reset_strategy = "default"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,  # 60
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                velocity_limit_sim={
                    ".*": 100.0,
                },
            ),
        },
    )


# @configclass
# class KneeSimpleHumanoidDistillationDanceEnvCfg(KneeSimpleHumanoidDistillationEnvCfg):
#     motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


# @configclass
# class KneeSimpleHumanoidDistillationRunEnvCfg(KneeSimpleHumanoidDistillationEnvCfg):
#     motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


# @configclass
# class KneeSimpleHumanoidDistillationWalkEnvCfg(KneeSimpleHumanoidDistillationEnvCfg):
#     motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
