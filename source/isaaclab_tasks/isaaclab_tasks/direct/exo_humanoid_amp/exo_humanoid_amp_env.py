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

from .exo_humanoid_amp_env_cfg import ExoHumanoidAmpEnvCfg  # 环境配置类
from .motions import MotionLoader

import time
import os
import csv
from torch.utils.tensorboard import SummaryWriter
import datetime


class ExoHumanoidAmpEnv(DirectRLEnv):
    cfg: ExoHumanoidAmpEnvCfg

    def __init__(self, cfg: ExoHumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("Joint names:", self.robot.data.joint_names)
        print("Body names:", self.robot.data.body_names)
        print("Num DOFs:", len(self.robot.data.joint_names))
        print("Num DOFs:", len(self.robot.data.body_names))

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

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)
        print("motion加载成功")

        # DOF and key body indexes
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)  # 躯干torso索引
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)  # 关节dof索引数组
        self.motion_human_dof_indexes = self._motion_loader.get_dof_index(self.human_joint_names)  # 关节dof索引数组
        print("Motion DOF indexes:", self.motion_human_dof_indexes)
        self.human_dof_indices = [self.robot.data.joint_names.index(name) for name in self.human_joint_names]
        self.exo_dof_indices = [self.robot.data.joint_names.index(name) for name in self.exo_joint_names]
        self.human_exo_dof_indices = [self.robot.data.joint_names.index(name) for name in self.human_exo_joint_names]
        self.human_upper_dof_indices = [self.robot.data.joint_names.index(name) for name in self.HUMAN_UPPER_JOINTS]
        self.human_lower_dof_indices = [self.robot.data.joint_names.index(name) for name in self.HUMAN_LOWER_JOINTS]
        print("Human DOF indexes:", self.human_dof_indices)
        print("Exo DOF indexes:", self.exo_dof_indices)
        print("Human-Exo DOF indexes:", self.human_exo_dof_indices)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space  # 历史长度*空间大小
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )
        # (num_envs, 2, 81)
        print("初始化成功")


        # --------------------------------------
        # 力矩日志
        # --------------------------------------
        self.log_torque = True  # 控制是否记录力矩（可在配置文件中设置）
        self.torque_log_dir = "./source/isaaclab_tasks/isaaclab_tasks/direct/exo_humanoid_amp/torque_logs"
        self.torque_log_file = None  # 日志文件对象
        self.torque_writer = None
        self.timestep = 0

        # 仅在仿真模式（有渲染）时启动日志（不影响训练）
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
                self.log_torque = False  # 创建失败则关闭日志

        if self.log_torque:
            self.sim.add_physics_callback("torque_log_callback", self._record_data)
            print("[INFO] 仿真回调绑定成功，将每帧记录力矩数据")


        # tensorboard创建一个新的e
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer1 = SummaryWriter(log_dir=os.path.join(
            self.cfg.get("log_dir", "/home/hy/文档/RL/IsaacGymEnvs-exo/isaacgymenvs/runs/tensorboard"),
            "env_" + str(os.getpid()) + str(time_str)))
        self.env_episode_count = np.zeros(self.num_envs, dtype=np.int32)  # 每个环境已完成的episode数
        self.env_current_ep_reward = np.zeros(self.num_envs, dtype=np.float32)  # 每个环境当前episode的累积奖励
        self.global_step = 0  # 全局训练步数（每步训练递增）
        self.total_episodes = 0  # 所有环境累计完成的episode总数


    # 设置模拟场景
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)  # 初始化机器人
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
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos,  # dof角度1*39
            self.robot.data.joint_vel,  # dof角速度1*39
            self.robot.data.body_pos_w[:, self.ref_body_index],  # torso位置1*3 --->1*1高度
            self.robot.data.body_quat_w[:, self.ref_body_index],  # torso四元数1*4 --->1*6投影基
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],  # torso线速度1*3
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],  # torso角速度1*3
            self.robot.data.body_pos_w[:, self.key_body_indexes],  # 关键刚体位置4*3
        )
        # print("速度", self.robot.data.body_lin_vel_w[:, self.ref_body_index])

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        # self.amp_observation_buffer[:, 0] = obs.clone()
        human_dof_indices_tensor = torch.tensor(self.human_dof_indices, device=obs.device)
        human_joint_vel_indices = 39 + human_dof_indices_tensor
        self.amp_observation_buffer[:, 0] = torch.cat([
            obs[:, self.human_dof_indices],  # 28dof在39dof的关节位置索引
            obs[:, human_joint_vel_indices],
            obs[:, 78:103]
            ], dim=-1)  # torso和关键刚体位置
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}  # 扁平化AMP观测值供判别器使用

        # 储存关节力矩信息
        joint_torques = self.robot.data.applied_torque
        human_torques = joint_torques[:, self.human_dof_indices]
        exo_torques = joint_torques[:, self.exo_dof_indices]
        self.extras.update({
            "joint_torques": joint_torques,
            "human_torques": human_torques,
            "exo_torques": exo_torques,
        })

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        
        joint_torques = self.robot.data.applied_torque  # (num_envs, num_dofs)
        joint_vels = self.robot.data.joint_vel          # (num_envs, num_dofs)
        
        reward = compute_reward(
            joint_torques,
            joint_vels,
            self.human_upper_dof_indices,
            self.human_lower_dof_indices,
            self.exo_dof_indices
        )
        return reward

    # 终止条件（倒地，超时）
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height  # 模型倒地判断0.5m
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
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)  # 重置根位姿3+4
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)  # 重置根速度3+3
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)  # 重置关节dof位置和速度1*

    # reset strategies
    # 默认重置
    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]  # 根据环境原点重置根位置
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    # 随即重置（利用采样数据对应时间）
    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]  # 环境数
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions 28dof
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.15  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]

        # get DOFs state
        # dof_pos = dof_positions[:, self.motion_dof_indexes]
        # dof_vel = dof_velocities[:, self.motion_dof_indexes]
        dof_pos = torch.zeros((num_samples, 39), device=self.device)
        dof_vel = torch.zeros((num_samples, 39), device=self.device)
        dof_pos[:, self.human_dof_indices] = dof_positions[:, self.motion_human_dof_indexes]
        dof_vel[:, self.human_dof_indices] = dof_velocities[:, self.motion_human_dof_indexes]
        dof_pos[:, self.exo_dof_indices] = dof_pos[:, self.human_exo_dof_indices]  # 重置外骨骼关节位置为人体对应刚体位置
        dof_vel[:, self.exo_dof_indices] = 0.0

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    # 读取：AMP参考运动数据
    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_human_dof_indexes],  # 只使用人体关节位置
            dof_velocities[:, self.motion_human_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)  # AMP参考数据amp_observation
    

    # tensorboard
    # --------------------------
    # 重写step方法：触发奖励记录（核心补充）
    # --------------------------
    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # 调用父类step获取核心数据（obs, rewards, terminated, truncated, info）
        obs, rewards, terminated, truncated, info = super().step(actions)
        
        # 合并终止条件（倒地=terminated，超时=truncated）
        done = terminated | truncated
        # 记录奖励并更新TensorBoard
        self._update_tensorboard(rewards, done)
        
        return obs, rewards, terminated, truncated, info

    # --------------------------
    # 完善TensorBoard更新逻辑（修正）
    # --------------------------
    def _update_tensorboard(self, rewards: torch.Tensor, done: torch.Tensor):
        # 1. 将torch奖励张量转换为numpy（适配累积逻辑）
        rewards_np = rewards.detach().cpu().numpy()
        
        # 2. 累积每个环境的当前episode奖励
        self.env_current_ep_reward += rewards_np
        
        # 3. 记录每步的平均奖励（实时监控训练趋势）
        step_mean_reward = rewards_np.mean()
        self.writer1.add_scalar("实时奖励/每步平均奖励", step_mean_reward, self.global_step)
        self.global_step += 1
        
        # 4. 处理已完成的episode（done=True的环境）
        done_env_ids = np.where(done.cpu().numpy())[0]  # 获取所有完成episode的环境索引
        if len(done_env_ids) > 0:
            for env_id in done_env_ids:
                # 记录单个环境的episode奖励
                ep_reward = self.env_current_ep_reward[env_id]
                self.writer1.add_scalar(f"单个环境奖励/环境{env_id}", ep_reward, self.env_episode_count[env_id])
                
                # 更新计数
                self.env_episode_count[env_id] += 1
                self.total_episodes += 1
                
                # 重置当前环境的累积奖励
                self.env_current_ep_reward[env_id] = 0.0
            
            # 记录所有环境的平均episode奖励（每完成一个episode更新）
            recent_ep_rewards = []
            for env_id in range(self.num_envs):
                if self.env_episode_count[env_id] > 0:
                    # 取每个环境最近1个episode的奖励（可调整为最近N个求平均）
                    recent_ep_rewards.append(self.env_current_ep_reward[env_id] if not done[env_id] else 0.0)
            if recent_ep_rewards:
                avg_ep_reward = np.mean(recent_ep_rewards)
                self.writer1.add_scalar("平均奖励/所有环境episode平均", avg_ep_reward, self.total_episodes)


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
            joint_vels = self.robot.data.joint_vel[0].cpu()
            human_lower_vels = joint_vels[self.human_lower_dof_indices].cpu().numpy()
            exo_vels = joint_vels[self.exo_dof_indices].cpu().numpy()
            timestamp = self.sim.current_time

            log_row = [timestamp, self.timestep] + torque_data.tolist() + human_lower_vels.tolist() + exo_vels.tolist()
            self.torque_writer.writerow(log_row)

            # 每100帧打印进度
            if self.timestep % 100 == 0:
                print(f"[INFO] 已记录{self.timestep}步力矩数据，当前时间：{timestamp:.2f}s")
            self.timestep += 1
        except Exception as e:
            print(f"[ERROR] 力矩记录失败：{e}")

    def close(self):
        self.writer1.close()
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
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),  # 6
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
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
    """
    计算基于关节功率的奖励函数（功率 = 力矩 * 角速度，取绝对值）
    
    参数:
        joint_torques: 所有关节的力矩 (num_envs, num_dofs)
        joint_vels: 所有关节的角速度 (num_envs, num_dofs)
        human_upper_indices: 人体上肢关节的索引(list[int])
        human_lower_indices: 人体下肢关节的索引(list[int])
        exo_lower_indices: 外骨骼下肢关节的索引(list[int])
    """
    torque_upper = joint_torques[:, human_upper_dof_indices]
    vel_upper = joint_vels[:, human_upper_dof_indices]
    torque_human_lower = joint_torques[:, human_lower_dof_indices]
    vel_human_lower = joint_vels[:, human_lower_dof_indices]
    torque_exo_lower = joint_torques[:, exo_dof_indices]
    vel_exo_lower = joint_vels[:, exo_dof_indices]

    power_upper = torch.sum(torch.abs(torque_upper * vel_upper), dim=1)  # (num_envs,)
    power_human_lower = torch.sum(torch.abs(torque_human_lower * vel_human_lower), dim=1)
    power_exo_lower = torch.sum(torch.abs(torque_exo_lower * vel_exo_lower), dim=1)

    total_power = 0.1 * power_upper + 0.6 * power_human_lower + 0.3 * power_exo_lower
    
    # 缩放功率，避免奖励过小(以力矩为100左右，具体需调整模型力矩限制)
    reward = 1.0 / (total_power / 1000.0 + 1.0)
    
    return reward

