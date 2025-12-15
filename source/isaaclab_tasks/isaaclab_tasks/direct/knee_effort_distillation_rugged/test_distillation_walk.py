"""
独立策略验证脚本 - 直接运行
用法: ./isaaclab.sh -p test.py
或者: python test.py (如果环境变量已配置)
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)  # headless=True 无GUI运行
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import time

# IsaacLab 导入
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.direct.knee_effort_humanoid_distillation.knee_effort_humanoid_distillation_env import KneeEffortHumanoidDistillationEnv
from isaaclab_tasks.direct.knee_effort_humanoid_distillation.knee_effort_humanoid_distillation_env_cfg import KneeEffortHumanoidDistillationEnvCfg

CONFIG = {
    # 检查点路径
    "checkpoint_path": "/home/hy/IsaacLab/logs/skrl/knee_effort_humanoid_distillation/2025-11-23_02-51-40_amp_torch/checkpoints/agent_200000.pt",
    "motion_file": "/home/hy/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/knee_effort_humanoid_distillation/motions/humanoid_walk.npz",

    # 环境配置
    "num_envs": 2,
    "device": "cuda:0",
    
    # 网络结构（必须与训练时一致）
    "obs_dim": 77,        # 观测维度
    "action_dim": 30,     # 动作维度
    "hidden_dims": [1024, 512],  # 隐藏层
    
    # 运行配置
    "max_steps": 0,       # 0 = 无限运行
    "real_time": False,   # 是否实时运行
    "dt": 1.0 / 60.0,     # 时间步长
}


class PolicyNetwork(nn.Module):
    """
    策略网络（与SKRL训练时结构一致）
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [1024, 512]):
        super().__init__()
        
        # 构建网络
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.net_container = nn.Sequential(*layers)
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向传播，返回动作均值"""
        return self.net_container(obs)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """获取动作"""
        mean = self.forward(obs)
        if deterministic:
            return mean
        else:
            std = torch.exp(self.log_std_parameter)
            return mean + std * torch.randn_like(mean)

class RunningStandardScaler:
    """SKRL 使用的观测归一化器"""
    
    def __init__(self, running_mean: torch.Tensor, running_variance: torch.Tensor, epsilon: float = 1e-8):
        self.running_mean = running_mean
        self.running_variance = running_variance
        self.epsilon = epsilon
    
    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """对观测进行归一化: (obs - mean) / sqrt(var + eps)"""
        return (obs - self.running_mean) / torch.sqrt(self.running_variance + self.epsilon)



def load_policy(checkpoint_path: str, obs_dim: int, action_dim: int, 
                hidden_dims: list, device: str):
    """加载策略网络"""
    print(f"[INFO] 加载检查点: {checkpoint_path}")
    
    # 创建网络
    policy = PolicyNetwork(obs_dim, action_dim, hidden_dims).to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"[INFO] 检查点包含: {list(checkpoint.keys())}")
    
    # SKRL权重在"policy"键下
    if "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"], strict=True)
        print("[INFO] 策略网络加载成功！")
    else:
        raise KeyError(f"检查点中未找到'policy'键")
    
    policy.eval()

    preprocessor = None
    if "state_preprocessor" in checkpoint:
        sp = checkpoint["state_preprocessor"]
        running_mean = sp["running_mean"].to(device)
        running_variance = sp["running_variance"].to(device)
        preprocessor = RunningStandardScaler(running_mean, running_variance)
        print("[INFO] 观测预处理器加载成功！")
        print(f"       - running_mean shape: {running_mean.shape}")
        print(f"       - running_variance shape: {running_variance.shape}")
    else:
        print("[WARN] 检查点中未找到 state_preprocessor, 将不进行观测归一化")

    return policy, preprocessor


# 观测处理
def process_observation(obs) -> torch.Tensor:
    """处理环境返回的观测"""
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        else:
            return list(obs.values())[0]
    return obs


# 动作修正函数
def modify_action(action: torch.Tensor) -> torch.Tensor:
    """
    动作修正函数
    Args:
        action: 原始动作 (num_envs, action_dim)
    
    Returns:
        修正后的动作 (num_envs, action_dim)
    """
    # TODO: 在这里添加修正逻辑
    # 例如:
    # action[:, 0] = action[:, 0] * 0.8  # 缩放第一个关节
    # action = torch.clamp(action, -1.0, 1.0)  # 限幅
    action = action.clone()
    action[:, 28:30] = action[:, 28:30] * 4  # 乘以系数后增加了速度
    return action


def main():
    print("=" * 70)
    print("独立策略验证脚本")
    print("=" * 70)
    
    device = CONFIG["device"]
    
    # ----- 创建环境 -----
    print("[INFO] 创建环境...")
    env_cfg = KneeEffortHumanoidDistillationEnvCfg()
    env_cfg.scene.num_envs = CONFIG["num_envs"]
    env_cfg.sim.device = device
    env_cfg.motion_file = CONFIG["motion_file"]

    env = KneeEffortHumanoidDistillationEnv(cfg=env_cfg)

    print(f"[INFO] 环境创建成功")
    print(f"[INFO]   - 环境数: {CONFIG['num_envs']}")
    print(f"[INFO]   - 观测维度: {CONFIG['obs_dim']}")
    print(f"[INFO]   - 动作维度: {CONFIG['action_dim']}")
    
    # ----- 加载策略 -----
    policy, preprocessor = load_policy(
        CONFIG["checkpoint_path"],
        CONFIG["obs_dim"],
        CONFIG["action_dim"],
        CONFIG["hidden_dims"],
        device
    )
    
    # ----- 重置环境 -----
    print("[INFO] 重置环境...")
    obs, info = env.reset()
    
    print("[INFO] 开始仿真循环...")
    print("-" * 70)
    
    # ----- 仿真主循环 -----
    step = 0
    total_reward = 0.0
    episode_count = 0
    episode_reward = 0.0
    
    while simulation_app.is_running():
        start_time = time.time()
        
        # 处理观测
        obs_tensor = process_observation(obs)
        if obs_tensor.device != torch.device(device):
            obs_tensor = obs_tensor.to(device)
        
        if preprocessor is not None:
            obs_normalized = preprocessor.normalize(obs_tensor)
        else:
            obs_normalized = obs_tensor
        obs_normalized = obs_normalized.float()

        # 策略推理
        with torch.inference_mode():
            action = policy.get_action(obs_normalized, deterministic=True)

        # 动作修正
        action = modify_action(action)
        
        # 环境step
        obs, reward, terminated, truncated, info = env.step(action)
        
        step += 1
        reward_val = reward.mean().item()
        total_reward += reward_val
        episode_reward += reward_val
        
        if terminated.any():
            episode_count += 1
            print(f"[INFO] 回合 {episode_count} 结束 | 回合奖励: {episode_reward:.4f}")
            episode_reward = 0.0

        if step % 100 == 0:
            avg_reward = total_reward / step
            print(f"[INFO] 步数: {step:6d} | 平均奖励: {avg_reward:.4f} | 回合: {episode_count}")

        if CONFIG["max_steps"] > 0 and step >= CONFIG["max_steps"]:
            print(f"[INFO] 达到最大步数 {CONFIG['max_steps']}")
            break

        if CONFIG["real_time"]:
            elapsed = time.time() - start_time
            if elapsed < CONFIG["dt"]:
                time.sleep(CONFIG["dt"] - elapsed)
    
    print("=" * 70)
    print("仿真结束")
    print(f"  总步数: {step}")
    print(f"  总回合: {episode_count}")
    print(f"  平均奖励: {total_reward / max(step, 1):.6f}")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()