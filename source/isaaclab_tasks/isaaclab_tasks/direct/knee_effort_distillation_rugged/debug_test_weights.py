# debug_weights.py
"""
分析SKRL检查点的权重结构
用法: python debug_weights.py
"""

import torch

checkpoint_path = "/home/hy/IsaacLab/logs/skrl/knee_effort_humanoid_distillation/2025-11-23_02-51-40_amp_torch/checkpoints/agent_200000.pt"

print("=" * 70)
print("SKRL 检查点分析")
print("=" * 70)

ckpt = torch.load(checkpoint_path, map_location="cpu")

print(f"\n顶层键: {list(ckpt.keys())}")

if "policy" in ckpt:
    print("\n" + "=" * 50)
    print("Policy 网络权重:")
    print("=" * 50)
    for name, param in ckpt["policy"].items():
        print(f"  {name}: {param.shape}")
    
    # 推断网络结构
    print("\n推断的网络结构:")
    for name, param in ckpt["policy"].items():
        if "weight" in name and "log_std" not in name:
            print(f"  {name}: {param.shape[1]} -> {param.shape[0]}")

if "state_preprocessor" in ckpt:
    print("\n" + "=" * 50)
    print("State Preprocessor (观测归一化):")
    print("=" * 50)
    for name, param in ckpt["state_preprocessor"].items():
        print(f"  {name}: {param.shape if hasattr(param, 'shape') else param}")

print("\n" + "=" * 70)