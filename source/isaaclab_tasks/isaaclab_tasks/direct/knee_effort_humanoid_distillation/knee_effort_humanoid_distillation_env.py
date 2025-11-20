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

from .knee_effort_humanoid_distillation_env_cfg import KneeEffortHumanoidDistillationEnvCfg
# from .motions import MotionLoader

import time
import os
import csv
from torch.utils.tensorboard import SummaryWriter
import datetime

class KneeEffortHumanoidDistillationEnv(DirectRLEnv):
    cfg: KneeEffortHumanoidDistillationEnvCfg

    def __init__(self, cfg: KneeEffortHumanoidDistillationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.original_actions_dim = 28  # 原始动作维度（角度）
        self.exo_actions_dim = 2  # 外骨骼动作维度（力矩）

        current_stiffness = self.robot.data.joint_stiffness.clone()  # (num_envs, num_joints)
        current_damping = self.robot.data.joint_damping.clone()
        print(f"初始刚度: {current_stiffness[0]}")
        print(f"初始阻尼: {current_damping[0]}")
        stiffness_scale = 0.5
        damping_scale = 0.5
        new_stiffness = current_stiffness * stiffness_scale
        new_damping = current_damping * damping_scale
        self.robot.write_joint_stiffness_to_sim(new_stiffness)
        self.robot.write_joint_damping_to_sim(new_damping)
        print(f"最终刚度: {self.robot.data.joint_stiffness[0]}")
        print(f"最终阻尼: {self.robot.data.joint_damping[0]}")

        actuator = self.robot.actuators["body"]
        initial_kp = actuator.stiffness.clone()
        initial_kd = actuator.damping.clone()
        print(f"初始kp:{initial_kp[0]}")
        print(f"初始kd:{initial_kd[0]}")
        kp_scale = 0.05
        kd_scale = 0.04   # 0.05
        new_stiffness = initial_kp * kp_scale
        new_damping = initial_kd * kd_scale
        actuator.stiffness[:] = new_stiffness
        actuator.damping[:] = new_damping
        print(f"最终kp:{actuator.stiffness[0]}")
        print(f"最终kd:{actuator.damping[0]}")

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        print(f"动作下限: {dof_lower_limits}")
        print(f"动作上限: {dof_upper_limits}")

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # DOF and key body indexes
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )


        "“”额外关节索引"""
        print("Joint names:", self.robot.data.joint_names)
        print("Body names:", self.robot.data.body_names)
        print("Num DOFs:", len(self.robot.data.joint_names))
        print("Num DOFs:", len(self.robot.data.body_names))
        # 原始的人体模型顺序（exo）
        self.HUMAN_UPPER_JOINTS = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 
                                'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'right_elbow', 'left_elbow']
        self.HUMAN_LOWER_JOINTS = ['right_hip_x', 'right_hip_y', 'right_hip_z', 'left_hip_x', 'left_hip_y', 'left_hip_z', 
                                   'right_knee', 'left_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z']
        self.human_upper_dof_indices = [self.robot.data.joint_names.index(name) for name in self.HUMAN_UPPER_JOINTS]
        self.human_lower_dof_indices = [self.robot.data.joint_names.index(name) for name in self.HUMAN_LOWER_JOINTS]
        # 关键关节
        self.human_knee_joint_names = ["right_knee", "left_knee"]
        self.human_hip_joint_names = ["right_hip_y", "left_hip_y"]
        self.human_knee_indices = [self.robot.data.joint_names.index(name) for name in self.human_knee_joint_names]
        self.human_hip_indices = [self.robot.data.joint_names.index(name) for name in self.human_hip_joint_names]

        # 外骨骼offset and scale
        # self.exo_effort_scale = 5
        self.exo_effort_scale = 100
        self.exo_effort_offset = 0


        """力矩日志"""
        self.log_torque = True  # 控制是否记录力矩（可在配置文件中设置）
        self.torque_log_dir = "./source/isaaclab_tasks/isaaclab_tasks/direct/knee_effort_humanoid_distillation/torque-effort_logs"
        self.torque_log_file = None  # 日志文件对象
        self.torque_writer = None
        self.timestep = 0
        # 仅在仿真模式（有渲染）时启动日志（不影响训练）
        is_simulation = self.num_envs == 1 or render_mode is not None
        if self.log_torque and is_simulation:
            os.makedirs(self.torque_log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.torque_log_path = f"{self.torque_log_dir}/torque-effort_{timestamp}.csv"
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
        log_dir = os.path.join("runs/knee_effort_humanoid_distillation", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.writer1 = SummaryWriter(log_dir=log_dir)
        self.max_episodes = 800  # horizon_length*max_epochs（总步数）
        self.envs_episode_count = np.zeros(self.num_envs, dtype=np.int32)  # 每个环境各自的回合
        self.episode_count = 0  # 同步完成回合数
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)  # 回合总奖励(清零版)
        self.envs_episode_rewards = np.zeros((self.num_envs, int(self.max_episodes)), dtype=np.float32)  # 存储每个环境的奖励
        self.global_frame = 0  # 全局帧计数器

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
        human_target = self.action_offset + self.action_scale * self.original_actions
        exo_effort_target = self.exo_effort_offset + self.exo_effort_scale * self.exo_action
        # print(f"目标值类型", human_target.shape, exo_effort_target.shape)

        self.robot.set_joint_position_target(human_target)
        self.robot.set_joint_effort_target(exo_effort_target, joint_ids=self.human_knee_indices)  # 前馈力给到膝关节

    def _get_observations(self) -> dict:
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        joint_torques = self.robot.data.applied_torque  # (num_envs, num_dofs)
        joint_vels = self.robot.data.joint_vel          # (num_envs, num_dofs)
        key_lower_dof_indices = self.human_knee_indices
        exo_knee_torque = self.exo_effort_offset + self.exo_effort_scale * self.exo_action

        reward = compute_reward(
            joint_torques,
            joint_vels,
            exo_knee_torque,
            self.human_upper_dof_indices,
            self.human_lower_dof_indices,
            key_lower_dof_indices,
        )
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
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
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

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
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
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

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
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)
    

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
            timestamp = self.sim.current_time
            torque_data = self.robot.data.applied_torque[0].cpu().numpy()
            joint_pos = self.robot.data.joint_pos[0].cpu()
            human_lower_pos = joint_pos[self.human_lower_dof_indices].cpu().numpy()
            joint_vels = self.robot.data.joint_vel[0].cpu()
            human_lower_vels = joint_vels[self.human_lower_dof_indices].cpu().numpy()
            human_lower_actions = self.actions[0, self.human_lower_dof_indices].cpu().numpy()
            exo_efforts = self.actions[0, -2:].cpu().numpy()

            log_row = [timestamp, self.timestep] + torque_data.tolist() + human_lower_pos.tolist() + human_lower_vels.tolist() + human_lower_actions.tolist() + exo_efforts.tolist()
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
        done_env_ids = reset_buf.nonzero(as_tuple=False).flatten()

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

        mean_step_reward = np.mean(rewards_np)
        self.writer1.add_scalar(f"reward/step", mean_step_reward, self.global_frame)

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
            quaternion_to_tangent_and_normal(root_rotations),
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
    exo_knee_torque: torch.Tensor,
    human_upper_dof_indices: list[int],
    human_lower_dof_indices: list[int],
    key_lower_dof_insices: list[int],
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
    joint_torques[:, key_lower_dof_insices] -= exo_knee_torque
    torque_upper = joint_torques[:, human_upper_dof_indices]
    vel_upper = joint_vels[:, human_upper_dof_indices]
    torque_lower = joint_torques[:, human_lower_dof_indices]
    vel_lower = joint_vels[:, human_lower_dof_indices]
    # print("膝关节力矩：", torque_lower[key_lower_dof_insices])
    # print("外骨骼力矩：", exo_knee_torque)

    power_upper = torch.sum(torch.abs(torque_upper * vel_upper), dim=1)
    power_dof_upper = torch.abs(torque_upper * vel_upper)
    power_dof_lower = torch.abs(torque_lower * vel_lower)
    power_dof_lower_scale = torch.tensor([0.9, 1.2, 0.8, 0.9, 1.2, 0.8, 1.2, 1.2, 0.9, 1.2, 0.8, 0.9, 1.2, 0.8], 
                                         dtype=power_dof_upper.dtype, device="cuda").reshape(1, 14)
    power_lower = torch.sum(power_dof_lower_scale * power_dof_lower, dim=1)
    total_power = 0.3 * power_upper + 0.7 * power_lower
    
    # 缩放功率，避免奖励过小(以力矩为100左右，具体需调整模型力矩限制)
    reward_power = 1.0 / (total_power / 1000.0 + 1.0)
    

    reward = reward_power

    return reward
