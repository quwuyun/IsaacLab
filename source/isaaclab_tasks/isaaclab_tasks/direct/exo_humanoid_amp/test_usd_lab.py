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

# JETBOT_CONFIG = ArticulationCfg(
#     # spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
#     spawn=sim_utils.UsdFileCfg(usd_path="/home/hy/isaacsim_assets/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Classic/Humanoid28Exo/humanoid28_exo.usd"),
#     actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
# )

# DOFBOT_CONFIG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             max_depenetration_velocity=5.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         joint_pos={
#             "joint1": 0.0,
#             "joint2": 0.0,
#             "joint3": 0.0,
#             "joint4": 0.0,
#         },
#         pos=(0.25, -0.25, 0.0),
#     ),
#     actuators={
#         "front_joints": ImplicitActuatorCfg(
#             joint_names_expr=["joint[1-2]"],
#             effort_limit_sim=100.0,
#             velocity_limit_sim=100.0,
#             stiffness=10000.0,
#             damping=100.0,
#         ),
#         "joint3_act": ImplicitActuatorCfg(
#             joint_names_expr=["joint3"],
#             effort_limit_sim=100.0,
#             velocity_limit_sim=100.0,
#             stiffness=10000.0,
#             damping=100.0,
#         ),
#         "joint4_act": ImplicitActuatorCfg(
#             joint_names_expr=["joint4"],
#             effort_limit_sim=100.0,
#             velocity_limit_sim=100.0,
#             stiffness=10000.0,
#             damping=100.0,
#         ),
#     },
# )
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
        # # 人体关节（驱动）
        # "humanoid_joints": ImplicitActuatorCfg(
        #     joint_names_expr=["^(?!exo_).*"],  # 匹配非 exo_ 开头的关节（如 right_hip）
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,      # 隐式执行器：用 PD 控制，不用 stiffness
        #     damping=0.0,
        # ),
        # # 外骨骼关节（驱动）
        # "exo_joints": ImplicitActuatorCfg(
        #     joint_names_expr=["^exo_.*"],      # 匹配 exo_ 开头的关节
        #     effort_limit=1000.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=0.0,
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

    print("Num DOFs:", len(scene['robot'].data.joint_names))
    print(f"Number of DOFs (joints): {scene['robot'].joint_names}")
    print("Num Bodies:", len(scene['robot'].data.body_names)) 
    print(f"Number of Bodies: {scene['robot'].body_names}")

    # 预计算正弦波用于摆动
    freq = 0.5  # Hz
    amp = 0.5   # 弧度

    while simulation_app.is_running():
        # --- 每 500 步重置一次 ---
        if count % 500 == 0:
            count = 0
            scene.reset()
            print("[INFO]: Resetting robot...")

        # --- 控制人体髋关节摆动（模拟走路）---
        if "right_thigh_y" in scene["robot"].data.joint_names:
            # 获取关节索引
            idx = scene["robot"].data.joint_names.index("right_thigh_y")
            target_pos = amp * np.sin(2 * np.pi * freq * sim_time)
            action = torch.zeros(1, scene["robot"].num_dofs, device=sim.device)
            action[0, idx] = target_pos
            scene["robot"].set_joint_position_target(action)

        # --- 外骨骼关节：不控制，靠 D6 弹簧跟随 ---
        # 什么都不写 → 靠 USD 中的 Drive 自动回中

        # --- 仿真步进 ---
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        sim_time += sim_dt
        count += 1


# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0

#     while simulation_app.is_running():
#         # reset
#         if count % 500 == 0:
#             # reset counters
#             count = 0
#             # reset the scene entities to their initial positions offset by the environment origins
#             root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
#             root_jetbot_state[:, :3] += scene.env_origins
#             root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
#             root_dofbot_state[:, :3] += scene.env_origins

#             # copy the default root state to the sim for the jetbot's orientation and velocity
#             scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
#             scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
#             scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
#             scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

#             # copy the default joint states to the sim
#             joint_pos, joint_vel = (
#                 scene["Jetbot"].data.default_joint_pos.clone(),
#                 scene["Jetbot"].data.default_joint_vel.clone(),
#             )
#             scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)
#             joint_pos, joint_vel = (
#                 scene["Dofbot"].data.default_joint_pos.clone(),
#                 scene["Dofbot"].data.default_joint_vel.clone(),
#             )
#             scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
#             # clear internal buffers
#             scene.reset()
#             print("[INFO]: Resetting Jetbot and Dofbot state...")

#         # drive around
#         if count % 100 < 75:
#             # Drive straight by setting equal wheel velocities
#             action = torch.Tensor([[10.0, 10.0]])
#         else:
#             # Turn by applying different velocities
#             action = torch.Tensor([[5.0, -5.0]])

#         scene["Jetbot"].set_joint_velocity_target(action)

#         # wave
#         wave_action = scene["Dofbot"].data.default_joint_pos
#         wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
#         scene["Dofbot"].set_joint_position_target(wave_action)

#         scene.write_data_to_sim()
#         sim.step()
#         sim_time += sim_dt
#         count += 1
#         scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
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