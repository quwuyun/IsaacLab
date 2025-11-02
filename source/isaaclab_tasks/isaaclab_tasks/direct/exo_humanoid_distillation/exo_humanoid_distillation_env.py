# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv  # isaaclab环境基类
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .exo_humanoid_distillation_env_cfg import ExoHumanoidDistillationEnvCfg  # 环境配置类
# from .motions import MotionLoader

import time
import os
import csv


# 手动实现轻量MLP（替代RSL-RL的MLP模块）
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str = "relu"):
        super().__init__()
        # 构建网络层
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, dim))
            if activation == "relu":
                layers.append(torch.nn.ReLU())
            prev_dim = dim
        # 输出层（无激活函数，与SKRL的高斯策略输出一致）
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ExoHumanoidDistillationEnv(DirectRLEnv):
    cfg: ExoHumanoidDistillationEnvCfg

    def __init__(self, cfg: ExoHumanoidDistillationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("Joint names:", self.robot.data.joint_names)
        print("Body names:", self.robot.data.body_names)
        print("Num DOFs:", len(self.robot.data.joint_names))
        print("Num Bodies:", len(self.robot.data.body_names))

        self.human_joint_names = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 
                                  'right_elbow', 'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_elbow', 'right_hip_x', 'right_hip_y', 'right_hip_z', 
                                  'right_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee', 
                                  'left_ankle_x', 'left_ankle_y', 'left_ankle_z']
        self.exo_joint_names = ["exo_D6Joint0:0", "exo_D6Joint0:1", "exo_D6Joint0:2", "exo_right_hip:0", "exo_right_hip:1", "exo_right_hip:2",
                                "exo_left_hip:0", "exo_left_hip:1", "exo_left_hip:2", "exo_right_knee", "exo_left_knee"]
        self.human_exo_joint_names = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee', 
                                      'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee']
        self.HUMAN_UPPER_JOINTS = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 
                                'right_elbow', 'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_elbow']
        self.HUMAN_LOWER_JOINTS = ['right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                                   'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z']

        # 动作缩放参数
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # # 加载参考运动（用于环境重置，与蒸馏无关）
        # self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)
        # print("motion加载成功")

        # 关键体和关节索引
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)  # 躯干torso索引
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        # self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        # self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # self.motion_human_dof_indexes = self._motion_loader.get_dof_index(self.human_joint_names)
        # print("Motion DOF indexes:", self.motion_human_dof_indexes)
        self.human_dof_indices = [self.robot.data.joint_names.index(name) for name in self.human_joint_names]
        self.exo_dof_indices = [self.robot.data.joint_names.index(name) for name in self.exo_joint_names]
        self.human_exo_dof_indices = [self.robot.data.joint_names.index(name) for name in self.human_exo_joint_names]
        self.human_upper_dof_indices = [self.robot.data.joint_names.index(name) for name in self.HUMAN_UPPER_JOINTS]
        self.human_lower_dof_indices = [self.robot.data.joint_names.index(name) for name in self.HUMAN_LOWER_JOINTS]
        print("Human DOF indexes:", self.human_dof_indices)
        print("Exo DOF indexes:", self.exo_dof_indices)
        print("Human-Exo DOF indexes:", self.human_exo_dof_indices)

        # 力矩日志（保持原有功能）
        self.log_torque = True
        self.torque_log_dir = "./source/isaaclab_tasks/isaaclab_tasks/direct/exo_humanoid_amp/torque_logs"
        self.torque_log_file = None
        self.torque_writer = None
        self.timestep = 0

        is_simulation = self.num_envs == 1 or render_mode is not None
        if self.log_torque and is_simulation:
            os.makedirs(self.torque_log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.torque_log_path = f"{self.torque_log_dir}/torque_{timestamp}.csv"
            try:
                self.torque_log_file = open(self.torque_log_path, "w", newline="", encoding="utf-8")
                self.torque_writer = csv.writer(self.torque_log_file)
                headers = ["timestamp", "timestep"] + self.robot.data.joint_names + [f"vel_{name}" for name in self.HUMAN_LOWER_JOINTS] + [f"vel_{name}" for name in self.exo_joint_names]
                self.torque_writer.writerow(headers)
                print(f"[INFO] 力矩日志启动成功！保存至：{self.torque_log_path}")
            except Exception as e:
                print(f"[ERROR] 日志文件创建失败：{e}")
                self.log_torque = False

        if self.log_torque:
            self.sim.add_physics_callback("torque_log_callback", self._record_data)
            print("[INFO] 仿真回调绑定成功，将每帧记录力矩数据")

        print("环境初始化成功")

        self.teacher_actor = None  # 手动构建的SKRL教师网络
        self.teacher_obs_normalizer = None  # SKRL的观测归一化器
        self.teacher_obs = None  # 存储补全后的103维教师观测（给教师模型用）
    
        # 加载教师模型（仅当启用蒸馏时）
        if self.cfg.is_distillation and self.cfg.teacher_policy_path:
            self._load_teacher_policy()

    def _load_teacher_policy(self):
        """加载SKRL训练的教师模型（跨框架适配：权重转换+维度对齐）"""
        try:
            # 1. 手动构建教师网络（与SKRL结构完全一致：103→1024→512→39）
            self.teacher_actor = SimpleMLP(
                input_dim=self.cfg.teacher_obs_dim,  # 103维（SKRL输入）
                output_dim=39,    # 39维（与学生动作一致）
                hidden_dims=[1024, 512],             # 对齐SKRL的网络结构（skrl_walk_amp_cfg.yaml）
                activation="relu"                    # 对齐SKRL的激活函数
            ).to(self.device)

            # 2. 加载SKRL权重并转换（解决跨框架键不匹配）
            checkpoint = torch.load(self.cfg.teacher_policy_path, map_location=self.device)
            skrl_weights = checkpoint["policy"]  # SKRL的权重存在"policy"键下
            
            # 权重键转换：SKRL的"net_container.x" → RSL-RL MLP的"x"，跳过高斯参数log_std_parameter
            converted_weights = {}
            for key, value in skrl_weights.items():
                if "log_std_parameter" in key:  # 跳过SKRL高斯策略的额外参数（学生用确定性动作）
                    continue
                if "net_container." in key:     # 转换键名：net_container.0.weight → layers.0.weight
                    converted_key = key.replace("net_container.", "layers.")
                    converted_weights[converted_key] = value

            # 3. 加载转换后的权重（strict=False忽略无关键）
            self.teacher_actor.load_state_dict(converted_weights, strict=False)
            
            # 4. 冻结教师网络（仅用于生成参考动作，不更新）
            for param in self.teacher_actor.parameters():
                param.requires_grad = False
            self.teacher_actor.eval()
            print(f"[INFO] SKRL教师模型加载成功！路径：{self.cfg.teacher_policy_path}")

            # 5. 加载SKRL的观测归一化器（保持观测分布一致）
            normalizer_path = os.path.join(
                os.path.dirname(self.cfg.teacher_policy_path),
                "../obs_normalizer.pth"  # SKRL默认归一化器路径（checkpoints同级目录）
            )
            if os.path.exists(normalizer_path):
                self.teacher_obs_normalizer = torch.load(normalizer_path, map_location=self.device)
                print(f"[INFO] SKRL观测归一化器加载成功：{normalizer_path}")

        except Exception as e:
            raise RuntimeError(f"教师模型加载失败：{str(e)}") from e


    # 设置模拟场景（保持原有逻辑）
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # 生成地面
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # 克隆多环境
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # 添加机器人和灯光
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        student_obs = compute_obs(
            self.robot.data.joint_pos,  # 39维关节位置
            self.robot.data.joint_vel,  # 39维关节速度
            self.robot.data.body_pos_w[:, self.ref_body_index],  # 躯干位置（3维）
            self.robot.data.body_quat_w[:, self.ref_body_index],  # 躯干四元数（4维）
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],  # 躯干线速度（3维）
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],  # 躯干角速度（3维）
            self.robot.data.body_pos_w[:, self.key_body_indexes],  # 关键体位置（4×3=12维）
        )
        if self.cfg.is_distillation and self.teacher_actor is not None:
            self.teacher_obs = compute_teacher_obs(
                self.robot.data.joint_pos,  # 39维关节位置
                self.robot.data.joint_vel,  # 39维关节速度
                self.robot.data.body_pos_w[:, self.ref_body_index],  # 躯干位置（3维）
                self.robot.data.body_quat_w[:, self.ref_body_index],  # 躯干四元数（4维）
                self.robot.data.body_lin_vel_w[:, self.ref_body_index],  # 躯干线速度（3维）
                self.robot.data.body_ang_vel_w[:, self.ref_body_index],  # 躯干角速度（3维）
                self.robot.data.body_pos_w[:, self.key_body_indexes],  # 关键体位置（4×3=12维）
            )

        # 存储关节力矩信息
        joint_torques = self.robot.data.applied_torque
        human_torques = joint_torques[:, self.human_dof_indices]
        exo_torques = joint_torques[:, self.exo_dof_indices]
        self.extras.update({
            "joint_torques": joint_torques,
            "human_torques": human_torques,
            "exo_torques": exo_torques,
        })

        return {"policy": student_obs}

    def _get_rewards(self) -> torch.Tensor:
        joint_torques = self.robot.data.applied_torque
        joint_vels = self.robot.data.joint_vel
        power_reward = compute_reward(
            joint_torques,
            joint_vels,
            self.human_upper_dof_indices,
            self.human_lower_dof_indices,
            self.exo_dof_indices
        )

        distill_reward = torch.zeros_like(power_reward)
        if self.cfg.is_distillation and self.teacher_actor is not None and self.teacher_obs is not None:
            # 生成教师参考动作（无梯度）
            with torch.no_grad():
                # 应用SKRL的观测归一化（保持分布一致）
                teacher_obs_norm = self.teacher_obs
                if self.teacher_obs_normalizer is not None:
                    teacher_obs_norm = self.teacher_obs_normalizer.normalize(teacher_obs_norm)
                # 教师输出参考动作
                teacher_actions = self.teacher_actor(teacher_obs_norm)
            
            # 计算MSE：学生动作与教师动作的差异（差异越小，奖励越大）
            action_mse = torch.mean(torch.square(self.actions - teacher_actions), dim=1)
            distill_reward = torch.exp(-5.0 * action_mse)

        reward = 0.1 * power_reward + 0.9 * distill_reward

        return reward

    # 终止条件（保持原有逻辑）
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    # 环境重置（保持原有逻辑）
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 2] += 0.15
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel


    # 力矩日志记录（保持原有逻辑）
    def _record_data(self, dt: float):
        if not self.log_torque or self.torque_writer is None:
            return
        try:
            if not hasattr(self.robot.data, "applied_torque") or self.robot.data.applied_torque is None:
                if self.timestep % 100 == 0:
                    print("[WARNING] applied_torque未初始化,暂不记录")
                self.timestep += 1
                return
            torque_data = self.robot.data.applied_torque[0].cpu().numpy()
            joint_vels = self.robot.data.joint_vel[0].cpu()
            human_lower_vels = joint_vels[self.human_lower_dof_indices].cpu().numpy()
            exo_vels = joint_vels[self.exo_dof_indices].cpu().numpy()
            timestamp = self.sim.current_time

            log_row = [timestamp, self.timestep] + torque_data.tolist() + human_lower_vels.tolist() + exo_vels.tolist()
            self.torque_writer.writerow(log_row)

            if self.timestep % 100 == 0:
                print(f"[INFO] 已记录{self.timestep}步力矩数据，当前时间：{timestamp:.2f}s")
            self.timestep += 1
        except Exception as e:
            print(f"[ERROR] 力矩记录失败：{e}")

    # 关闭资源（保持原有逻辑）
    def close(self):
        if self.torque_log_file is not None:
            self.torque_writer = None
            self.torque_log_file.close()
            print(f"[INFO] 力矩日志已关闭，保存至：{self.torque_log_path}，共记录{self.timestep}步")
        super().close()


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:

    obs = torch.cat(
        (
            dof_positions,  # 39维关节位置
            dof_velocities,  # 39维关节速度
            root_positions[:, 2:3],  # 1维躯干高度
            quaternion_to_tangent_and_normal(root_rotations),  # 6维四元数投影
            # root_linear_velocities,  # 3维躯干线速度（教师保留，学生剔除）
            root_angular_velocities,  # 3维躯干角速度
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),  # 12维关键体相对位置
        ),
        dim=-1,
    )
    return obs

@torch.jit.script
def compute_teacher_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:

    # root_linear_velocities = torch.tensor([[1.2, 0.0, 0.0]], device=dof_positions.device)  # 强制设置躯干线速度为1.2m/s，模拟行走状态
    num_envs = dof_positions.shape[0]
    root_linear_velocities = torch.full(
        (num_envs, 3),  # 维度：(环境数, 3)
        fill_value=1.2,  # x方向速度1.2m/s
        device=dof_positions.device
    )
    root_linear_velocities[:, 1:] = 0.0  # y、z方向速度设为0
    obs = torch.cat(
        (
            dof_positions,  # 39维关节位置
            dof_velocities,  # 39维关节速度
            root_positions[:, 2:3],  # 1维躯干高度
            quaternion_to_tangent_and_normal(root_rotations),  # 6维四元数投影
            root_linear_velocities,  # 3维躯干线速度（教师保留，学生剔除）
            root_angular_velocities,  # 3维躯干角速度
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),  # 12维关键体相对位置
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_reward(
    joint_torques: torch.Tensor,
    joint_vels: torch.Tensor,
    human_upper_dof_indices: list[int],
    human_lower_dof_indices: list[int],
    exo_dof_indices: list[int]
) -> torch.Tensor:
    """基于关节功率的奖励函数"""
    torque_upper = joint_torques[:, human_upper_dof_indices]
    vel_upper = joint_vels[:, human_upper_dof_indices]
    torque_human_lower = joint_torques[:, human_lower_dof_indices]
    vel_human_lower = joint_vels[:, human_lower_dof_indices]
    torque_exo_lower = joint_torques[:, exo_dof_indices]
    vel_exo_lower = joint_vels[:, exo_dof_indices]

    power_upper = torch.sum(torch.abs(torque_upper * vel_upper), dim=1)
    power_human_lower = torch.sum(torch.abs(torque_human_lower * vel_human_lower), dim=1)
    power_exo_lower = torch.sum(torch.abs(torque_exo_lower * vel_exo_lower), dim=1)

    total_power = 0.3 * power_upper + 0.4 * power_human_lower + 0.3 * power_exo_lower
    reward = 1.0 / (total_power / 1000.0 + 1.0)
    
    return reward

