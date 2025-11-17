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
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .knee_simple_humanoid_distillation_env_cfg import KneeSimpleHumanoidDistillationEnvCfg
from .motions import MotionLoader

import time
import os
import csv
from torch.utils.tensorboard import SummaryWriter
import datetime


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
    

class KneeSimpleHumanoidDistillationEnv(DirectRLEnv):
    cfg: KneeSimpleHumanoidDistillationEnvCfg

    def __init__(self, cfg: KneeSimpleHumanoidDistillationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.original_actions_dim = 28  # 原始动作维度
        self.exo_actions_dim = 2  # 外骨骼动作维度
        print("Joint names:", self.robot.data.joint_names)
        print("Body names:", self.robot.data.body_names)
        print("Num DOFs:", len(self.robot.data.joint_names))
        print("Num DOFs:", len(self.robot.data.body_names))
        self.HUMAN_JOINT = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 
                            'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'right_hip_x', 'right_hip_y', 'right_hip_z', 'left_hip_x', 'left_hip_y', 'left_hip_z', 
                            'right_elbow', 'left_elbow', 'right_knee', 'left_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z']
        self.HUMAN_BODY = ['torso', 'pelvis', 'head', 'right_upper_arm', 'left_upper_arm', 'right_thigh', 'left_thigh', 'right_lower_arm', 'left_lower_arm', 'right_shin', 'left_shin', 
                           'right_hand', 'left_hand', 'right_foot', 'left_foot']
        self.HUMAN_MOTION_JOINT = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_elbow',  
                            'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_elbow', 'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 
                            'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z']
        # simple人体模型顺序（exo）
        self.HUMAN_SIMPLE_JOINTS = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'right_hip_x', 'right_hip_y', 'right_hip_z', 'left_hip_x', 'left_hip_y', 'left_hip_z', 
                                   'right_knee', 'left_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z']

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        print(f"动作下限: {dof_lower_limits}")
        print(f"动作上限: {dof_upper_limits}")

        # load motion
        # self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # DOF and key body indexes
        # key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        # self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        # self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        # self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        # self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        # self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        # self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)  # 按motion的顺序
        # self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        # print("在motion中关节索引:", self.motion_dof_indexes)
        # print("在motion中body索引:", self._motion_loader.get_body_index(self.robot.data.body_names))
        

        # # reconfigure AMP observation space according to the number of observations and create the buffer
        # self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        # self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        # self.amp_observation_buffer = torch.zeros(
        #     (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        # )


        "“”额外关节索引"""
        self.HUMAN_UPPER_JOINTS = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 
                                'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'right_elbow', 'left_elbow']
        self.HUMAN_LOWER_JOINTS = ['right_hip_x', 'right_hip_y', 'right_hip_z', 'left_hip_x', 'left_hip_y', 'left_hip_z', 
                                   'right_knee', 'left_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z']
        self.human_upper_dof_indices = [self.robot.data.joint_names.index(name) for name in self.HUMAN_UPPER_JOINTS]
        self.human_lower_dof_indices = [self.robot.data.joint_names.index(name) for name in self.HUMAN_LOWER_JOINTS]
        self.human_simple_dof_indices = [self.robot.data.joint_names.index(name) for name in self.HUMAN_SIMPLE_JOINTS]
        # 关键关节
        self.human_knee_joint_names = ["right_knee", "left_knee"]
        self.human_hip_joint_names = ["right_hip_y", "left_hip_y"]
        self.human_knee_indices = [self.robot.data.joint_names.index(name) for name in self.human_knee_joint_names]
        self.human_hip_indices = [self.robot.data.joint_names.index(name) for name in self.human_hip_joint_names]

        # 外骨骼offset and scale
        self.exo_scale_ratio = 1.0  # 外骨骼动作缩放比例
        self.exo_action_scale = (dof_upper_limits[self.human_knee_indices] - dof_lower_limits[self.human_knee_indices]) * self.exo_scale_ratio
        self.exo_action_offset = (dof_upper_limits[self.human_knee_indices] + dof_lower_limits[self.human_knee_indices])*0.5
        print(self.exo_action_scale, self.exo_action_offset)


        """力矩日志"""
        self.log_torque = True  # 控制是否记录力矩（可在配置文件中设置）
        self.torque_log_dir = "./source/isaaclab_tasks/isaaclab_tasks/direct/knee_simple_humanoid_amp/d_torque_logs"
        self.torque_log_file = None  # 日志文件对象
        self.torque_writer = None
        self.timestep = 0
        # 仅在仿真模式（有渲染）时启动日志（不影响训练）
        is_simulation = self.num_envs == 1 or render_mode is not None
        if self.log_torque and is_simulation:
            os.makedirs(self.torque_log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.torque_log_path = f"{self.torque_log_dir}/energe_simple_torque_{timestamp}.csv"
            try:
                self.torque_log_file = open(self.torque_log_path, "w", newline="", encoding="utf-8")
                self.torque_writer = csv.writer(self.torque_log_file)
                headers = ["timestamp", "timestep"] + self.robot.data.joint_names + [f"pos_{name}" for name in self.HUMAN_LOWER_JOINTS] \
                + [f"vel_{name}" for name in self.HUMAN_LOWER_JOINTS]  + [f"action_{name}" for name in self.HUMAN_LOWER_JOINTS] + [f"action_exo_{name}" for name in self.human_knee_joint_names]
                self.torque_writer.writerow(headers)
                print(f"[INFO] 力矩日志启动成功！保存至：{self.torque_log_path}")
            except Exception as e:
                print(f"[ERROR] 日志文件创建失败：{e}")
                self.log_torque = False  # 创建失败则关闭日志

        if self.log_torque:
            self.sim.add_physics_callback("torque_log_callback", self._record_data)
            print("[INFO] 仿真回调绑定成功，将每帧记录力矩数据")
        

        """添加 TensorBoard writer"""
        log_dir = os.path.join("runs/d_kneesimplehumanamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.writer1 = SummaryWriter(log_dir=log_dir)
        self.max_episodes = 8000  # horizon_length*max_epochs（总步数）
        self.envs_episode_count = np.zeros(self.num_envs, dtype=np.int32)  # 每个环境各自的回合
        self.episode_count = 0  # 同步完成回合数
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)  # 回合总奖励(清零版)
        self.envs_episode_rewards = np.zeros((self.num_envs, int(self.max_episodes)), dtype=np.float32)  # 存储每个环境的奖励
        self.global_frame = 0  # 全局帧计数器

        print("环境初始化成功")
        self.teacher_actor = None  # 手动构建的SKRL教师网络
        self.teacher_obs_normalizer = None  # SKRL的观测归一化器
        self.teacher_obs = None  # 存储补全后的69维教师观测（给教师模型用）
        # 加载教师模型（仅当启用蒸馏时）
        if self.cfg.is_distillation and self.cfg.teacher_policy_path:
            self._load_teacher_policy()
    
    def _load_teacher_policy(self):
        """加载SKRL训练的教师模型（跨框架适配：权重转换+维度对齐）"""
        try:
            # 手动构建教师网络（与SKRL结构一致：69→1024→512→30）
            self.teacher_actor = SimpleMLP(
                input_dim=self.cfg.teacher_obs_dim,  # （SKRL输入）
                output_dim=self.cfg.student_obs_dim,    # 30维（与学生动作一致）
                hidden_dims=[1024, 512],             # 对齐SKRL的网络结构（skrl_walk_amp_cfg.yaml）
                activation="relu"                    # 对齐SKRL的激活函数
            ).to(self.device)

            # 加载SKRL权重并转换（解决跨框架键不匹配）
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

            # 加载转换后的权重（strict=False忽略无关键）
            self.teacher_actor.load_state_dict(converted_weights, strict=False)
            
            # 冻结教师网络（仅用于生成参考动作，不更新）
            for param in self.teacher_actor.parameters():
                param.requires_grad = False
            self.teacher_actor.eval()
            print(f"[INFO] SKRL教师模型加载成功！路径：{self.cfg.teacher_policy_path}")

            # 加载SKRL的观测归一化器（保持观测分布一致）
            normalizer_path = os.path.join(
                os.path.dirname(self.cfg.teacher_policy_path),
                "../obs_normalizer.pth"  # SKRL默认归一化器路径（checkpoints同级目录）
            )
            if os.path.exists(normalizer_path):
                self.teacher_obs_normalizer = torch.load(normalizer_path, map_location=self.device)
                print(f"[INFO] SKRL观测归一化器加载成功：{normalizer_path}")

        except Exception as e:
            raise RuntimeError(f"教师模型加载失败：{str(e)}") from e
        

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
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
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.original_actions = actions[:, :self.original_actions_dim].clone()
        self.exo_action = actions[:, self.original_actions_dim:].clone()

        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.original_actions
        exo_pos_offset = self.exo_action_offset + self.exo_action_scale * self.exo_action
        target[:, self.human_knee_indices] += exo_pos_offset

        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # build task observation
        student_obs = compute_student_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            # self.robot.data.body_pos_w[:, self.key_body_indexes],
        )
        if self.cfg.is_distillation and self.teacher_actor is not None:
            self.teacher_obs = compute_teacher_obs(
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                self.robot.data.body_pos_w[:, self.ref_body_index],
                self.robot.data.body_quat_w[:, self.ref_body_index],
                self.robot.data.body_lin_vel_w[:, self.ref_body_index],
                self.robot.data.body_ang_vel_w[:, self.ref_body_index],
                # self.robot.data.body_pos_w[:, self.key_body_indexes],
            )
        return {"policy": student_obs}

    def _get_rewards(self) -> torch.Tensor:
        # return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        joint_torques = self.robot.data.applied_torque  # (num_envs, num_dofs)
        joint_vels = self.robot.data.joint_vel          # (num_envs, num_dofs)
        key_lower_dof_indices = self.human_hip_indices + self.human_knee_indices

        human_reward = compute_reward(
            joint_torques,
            joint_vels,
            # self.human_simple_dof_indices,
            self.human_upper_dof_indices,
            self.human_lower_dof_indices,
            key_lower_dof_indices,
            self.original_actions[:, self.human_knee_indices]
        )

        distill_reward = torch.zeros_like(human_reward)
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
            distill_reward = torch.exp(-1 * action_mse)

        reward = 0.5 * human_reward + 0.5 * distill_reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    
    def _record_data(self, dt: float):
        """由仿真回调调用，每帧记录力矩数据"""
        if not self.log_torque or self.torque_writer is None:
            return
        try:
            if not hasattr(self.robot.data, "applied_torque") or self.robot.data.applied_torque is None:
                if self.timestep % 100 == 0:
                    print("[WARNING] applied_torque未初始化,暂不记录")
                self.timestep += 1
                return
            # 取第一个环境的力矩数据
            torque_data = self.robot.data.applied_torque[0].cpu().numpy()
            joint_pos = self.robot.data.joint_pos[0].cpu()
            joint_vels = self.robot.data.joint_vel[0].cpu()
            # human_simple_pos = joint_pos[self.human_simple_dof_indices].cpu().numpy()
            # human_lower_pos = joint_vels[self.human_simple_dof_indices].cpu().numpy()
            human_lower_pos = joint_pos[self.human_lower_dof_indices].cpu().numpy()
            human_lower_vel = joint_vels[self.human_lower_dof_indices].cpu().numpy()
            timestamp = self.sim.current_time
            actions_lower = self.actions[0, 14:].cpu().numpy()  # 所有关节

            log_row = [timestamp, self.timestep] + torque_data.tolist() + human_lower_pos.tolist() + human_lower_vel.tolist() + actions_lower.tolist()
            self.torque_writer.writerow(log_row)

            # 每100帧打印进度
            if self.timestep % 100 == 0:
                print(f"[INFO] 已记录{self.timestep}步力矩数据，当前时间：{timestamp:.2f}s")
            self.timestep += 1
        except Exception as e:
            print(f"[ERROR] 力矩记录失败：{e}")

    def step(self, action: torch.Tensor):
        observations, rewards, terminated, truncated, extras = super().step(action)

        rew_buf = rewards.detach().cpu().numpy() if rewards.requires_grad else rewards.cpu().numpy()

        reset_buf = terminated | truncated
        done_env_ids = reset_buf.nonzero(as_tuple=False).flatten()   # <-- 完全等价于你原来的 self.reset_buf

        self.tensorboard_rew(rew_buf, done_env_ids)

        if "amp_obs" in extras:
            amp_std = extras["amp_obs"].std().item()
            self.writer1.add_scalar("AMP/obs_std", amp_std, self.global_frame)

        return observations, rewards, terminated, truncated, extras
    

    def tensorboard_rew(self, rew_buf, done_env_ids):
        rewards_np = rew_buf.detach().cpu().numpy() if hasattr(rew_buf, "detach") else rew_buf
        self.episode_rewards += rewards_np

        mean_cumulative_reward = self.episode_rewards.mean()  # 所有环境当前帧平均回合奖励
        self.writer1.add_scalar("reward/frame", mean_cumulative_reward, self.global_frame)
        self.global_frame += 1

        for env_id in done_env_ids:
            ep_reward = self.episode_rewards[env_id]  # 完成回合的环境当前回合总奖励
            self.envs_episode_rewards[env_id, self.envs_episode_count[env_id]] = ep_reward

            if env_id < 10:
                self.writer1.add_scalar(f"reward/episode_env{env_id}", ep_reward, self.envs_episode_count[env_id])

            self.episode_rewards[env_id] = 0.0
            self.envs_episode_count[env_id] += 1

        # 完成某回合所有环境平均奖励
        min_episodes = self.envs_episode_count.min()
        while self.episode_count < min_episodes:
            mean_reward = self.envs_episode_rewards[:self.num_envs, self.episode_count].mean()
            self.writer1.add_scalar("reward/episode_mean", mean_reward, self.episode_count)
            self.episode_count += 1

    def close(self):
        # 关闭 TensorBoard writer
        if hasattr(self, 'writer'):
            self.writer1.close()
            print("[INFO] TensorBoard writer 已关闭")

        # 关闭日志文件（确保数据写入）
        if self.torque_log_file is not None:
            self.torque_writer = None
            self.torque_log_file.close()
            print(f"[INFO] 力矩日志已关闭，保存至：{self.torque_log_path}，共记录{self.timestep}步")
        # 调用父类关闭方法
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
def compute_student_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    # key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            # root_linear_velocities,
            root_angular_velocities,
            # (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
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
    # key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            # (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs

@torch.jit.script
def compute_reward(
    joint_torques: torch.Tensor,
    joint_vels: torch.Tensor,
    # human_simple_dof_indices: list[int],
    human_upper_dof_indices: list[int],
    human_lower_dof_indices: list[int],
    key_lower_dof_indices: list[int],
    knee_action: torch.Tensor
) -> torch.Tensor:
    """
    计算基于关节功率的奖励函数（功率 = 力矩 * 角速度，取绝对值）
    
    参数:
        joint_torques: 所有关节的力矩 (num_envs, num_dofs)
        joint_vels: 所有关节的角速度 (num_envs, num_dofs)
        human_upper_indices: 人体上肢关节的索引(list[int])
        human_lower_indices: 人体下肢关节的索引(list[int])
        knee_action: 外骨骼action
    """

    """e_distillation"""
    # 能量
    torque_upper = joint_torques[:, human_upper_dof_indices]
    vel_upper = joint_vels[:, human_upper_dof_indices]
    torque_lower = joint_torques[:, human_lower_dof_indices]
    vel_lower = joint_vels[:, human_lower_dof_indices]
    power_upper = torch.sum(torch.abs(torque_upper * vel_upper), dim=1)
    power_dof_upper = torch.abs(torque_upper * vel_upper)
    power_dof_lower = torch.abs(torque_lower * vel_lower)
    power_dof_lower_scale = torch.tensor([0.9, 1.2, 0.8, 0.9, 1.2, 0.8, 1.2, 1.2, 0.9, 1.2, 0.8, 0.9, 1.2, 0.8], 
                                         dtype=power_dof_upper.dtype, device="cuda").reshape(1, 14)
    power_lower = torch.sum(power_dof_lower_scale * power_dof_lower, dim=1)
    total_power = 0.3 * power_upper + 0.7 * power_lower
    knee_action_sum = torch.sum(torch.abs(knee_action), dim=1)
    reward_action = - torch.pow(0.5 * knee_action_sum, exponent=2)
    # 缩放功率，避免奖励过小(以力矩为100左右，具体需调整模型力矩限制)
    reward_power = 1.0 / (total_power / 1000.0 + 1.0)
    
    # 膝关节action惩罚
    knee_action_sum = torch.sum(torch.abs(knee_action), dim=1)
    reward_action = - torch.pow(0.5 * knee_action_sum, exponent=2)



    
    reward = reward_power + reward_action

    return reward
