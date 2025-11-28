# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


HUMANOID_EXO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",  # 环境中的路径
    spawn=sim_utils.UsdFileCfg(usd_path="/home/hy/isaacsim_assets/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Classic/Humanoid28Exo/humanoid_32_exo.usd",
    # spawn=sim_utils.UsdFileCfg(usd_path="/home/hy/isaacsim_assets/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Classic/Humanoid28/humanoid_28.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),  # 初始高度，避免陷入地面
        joint_pos={".*": 0.0},  # 所有关节归零
    ),
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

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    # Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    # Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")
    robot = HUMANOID_EXO_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # ------------------------------------------------------------------
    # 1. 打印一次关节/刚体信息（调试用）
    # ------------------------------------------------------------------
    robot = scene["robot"]
    print("\n=== ROBOT INFO ===")
    print("joint_names :", robot.data.joint_names)
    print("body_names  :", robot.data.body_names)
    print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # 2. 运动参数
    # ------------------------------------------------------------------
    # 站立不动时间（秒）
    idle_time = 1.0
    # 一次蹲起周期（秒），包括下蹲 + 站起
    squat_cycle = 3.0
    # 膝关节最大弯曲角度（rad），约 60°
    knee_amp = 2

    # 找到膝关节索引（一次性查找，提升性能）
    knee_names = ["exo_left_knee", "exo_right_knee"]
    knee_idx = []
    num_dofs = len(robot.data.joint_names)
    for name in knee_names:
        if name in robot.data.joint_names:
            knee_idx.append(robot.data.joint_names.index(name))
        else:
            knee_idx.append(None)
            print(f"[WARN] Joint '{name}' not found!")

    # ------------------------------------------------------------------
    # 3. 主循环
    # ------------------------------------------------------------------
    while simulation_app.is_running():
        # ------------------- 重置 -------------------
        if count % int(5.0 / sim_dt) == 0:          # 每 5 s 重置一次
            count = 0
            scene.reset()
            print("[INFO]: Resetting robot to default pose...")

        # ------------------- 计算目标膝角 -------------------
        phase = sim_time % squat_cycle          # 0 ~ squat_cycle

        if phase < idle_time:
            # 站立不动
            target_knee = 0.0
        else:
            # 余弦波 → 平滑下蹲 & 站起
            t = (phase - idle_time) / (squat_cycle - idle_time)   # 0~1
            target_knee = knee_amp * (0.5 - np.cos(np.pi * t)) / 2.0

        # ------------------- 构造动作向量 -------------------
        action = torch.zeros(1, num_dofs, device=sim.device)

        for idx in knee_idx:
            if idx is not None:
                action[0, idx] = target_knee

        # 其它关节保持默认（0 rad），防止漂移
        # （如果 USD 中有非零默认角度，可在这里写 default_joint_pos）
        robot.set_joint_position_target(action)

        # ------------------- 仿真步进 -------------------
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        sim_time += sim_dt
        count += 1


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()