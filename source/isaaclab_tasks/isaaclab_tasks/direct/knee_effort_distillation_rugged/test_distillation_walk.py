# test.py: Demo to load and run pre-trained model in Knee Effort Humanoid Rugged environment for walking simulation

"""Launch Isaac Sim Simulator first."""

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

# Hardcoded parameters
num_envs = 4
checkpoint = "/home/hy/IsaacLab/logs/skrl/knee_effort_humanoid_distillation/2025-11-23_02-51-40_amp_torch/checkpoints/agent_200000.pt"
headless = False
video = False
video_length = 200

# Launch Omniverse app
app_launcher = AppLauncher(headless=headless)
simulation_app = app_launcher.app

# Task name for Rugged environment
task_name = "Isaac-Knee-Effort-Humanoid-Rugged-Walk-Direct-v0"

# Create the environment, override base_policy_path in cfg
env = gym.make(task_name, num_envs=num_envs, render_mode="rgb_array" if video else None)
env.unwrapped.cfg.base_policy_path = checkpoint  # Override to load your pt as base_policy

# If video recording
if video:
    env = gym.wrappers.RecordVideo(env, video_folder="videos/demo_rugged", episode_trigger=lambda x: True, video_length=video_length)

# Reset environment
obs = env.reset()

# Simulation loop (use zero actions for no correction, rely on base_policy inference inside env)
timestep = 0
while simulation_app.is_running():
    # Zero actions (2-dim correction = 0, so only base_policy actions applied)
    actions = torch.zeros((num_envs, env.action_space.shape[0]), device=env.unwrapped.device)
    
    # Step environment
    obs, rewards, terminated, truncated, info = env.step(actions)
    timestep += 1
    
    if video and timestep >= video_length:
        break

# Close environment and sim
env.close()
simulation_app.close()