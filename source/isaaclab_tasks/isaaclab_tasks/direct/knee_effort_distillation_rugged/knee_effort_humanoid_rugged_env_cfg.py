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

from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field import (
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
)

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class KneeEffortHumanoidRuggedEnvCfg(DirectRLEnvCfg):
    """Knee Humanoid Distillation environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces
    observation_space = 65  # 课程式学习观测维度
    action_space = 2  # （外骨骼力矩修正网络）
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 77

    # 平地基础模型，输入人体77维观测，输出基础action
    curriculum_enabled = True 
    base_policy_path = "/home/hy/IsaacLab/logs/skrl/knee_effort_humanoid_distillation/2025-11-23_02-51-40_amp_torch/checkpoints/agent_200000.pt"  
    base_obs_dim = 77
    base_action_dim = 30

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "torso"
    reset_strategy = "random"  # default, random, random-start
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

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(64.0, 64.0),
            border_width=5.0,
            num_rows=64,  # 难度类型
            num_cols=64,  # 地形等级
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            curriculum=True,
            sub_terrains={
                # 全平地 (不同粗糙度作为难度)
                "flat": HfRandomUniformTerrainCfg(
                    proportion=0.9,
                    noise_range=(0.0, 0.02),      # 难度通过噪声范围体现
                    noise_step=0.005,
                    border_width=0.25,
                ),
                # 上斜坡
                "slope_up": HfPyramidSlopedTerrainCfg(
                    proportion=0.05,
                    slope_range=(0.01, 0.2),      # 难度：坡度从1%到20%
                    platform_width=2.0,
                    border_width=0.25,
                    inverted=False,               # 向上
                ),
                # 下斜坡
                "slope_down": HfPyramidSlopedTerrainCfg(
                    proportion=0.05,
                    slope_range=(0.01, 0.2),
                    platform_width=2.0,
                    border_width=0.25,
                    inverted=True,                # 向下（凹陷）
                ),
                # 随机上下斜坡组合
                "slope_mixed": HfPyramidSlopedTerrainCfg(
                    proportion=0.0,
                    slope_range=(0.1, 0.5),
                    platform_width=1.5,
                    border_width=0.25,
                    inverted=False,               # 会随机生成不同方向
                ),
                # 上楼梯
                "stairs_up": HfPyramidStairsTerrainCfg(
                    proportion=0.0,
                    step_height_range=(0.05, 0.25),  # 难度：台阶高度5cm到25cm
                    step_width=0.3,
                    platform_width=2.0,
                    border_width=0.25,
                    inverted=False,               # 向上
                ),
                # 下楼梯
                "stairs_down": HfPyramidStairsTerrainCfg(
                    proportion=0.0,
                    step_height_range=(0.05, 0.25),
                    step_width=0.3,
                    platform_width=2.0,
                    border_width=0.25,
                    inverted=True,                # 向下
                ),
                # 随机阶梯
                "stairs_random": HfPyramidStairsTerrainCfg(
                    proportion=0.0,
                    step_height_range=(0.08, 0.3),
                    step_width=0.25,
                    platform_width=1.5,
                    border_width=0.25,
                    inverted=False,
                ),
                # 起伏地形 (波浪)
                "undulating": HfWaveTerrainCfg(
                    proportion=0.125,
                    amplitude_range=(0.05, 0.25),   # 难度：波幅5cm到25cm
                    num_waves=1,
                    border_width=0.25,
                ),
            }
        ),
    )

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


@configclass
class KneeEffortHumanoidRuggedDanceEnvCfg(KneeEffortHumanoidRuggedEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class KneeEffortHumanoidRuggedRunEnvCfg(KneeEffortHumanoidRuggedEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class KneeEffortHumanoidRuggedWalkEnvCfg(KneeEffortHumanoidRuggedEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
