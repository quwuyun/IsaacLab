import torch
import numpy as np
import time
from datetime import datetime

# 定义与训练一致的网络结构
class ActorNetwork(torch.nn.Module):
    """与RSL-RL训练时的Actor网络结构完全一致"""
    def __init__(self, input_dim=100, output_dim=39):
        super().__init__()
        # 匹配训练配置：actor_hidden_dims=[1024, 512]，activation="relu"
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


# 加载模型权重
def load_actor_model(model_path, device):
    """加载训练好的RSL-RL模型权重"""
    # 初始化网络和权重
    model = ActorNetwork(input_dim=100, output_dim=39).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    actor_weights = {}
    # 提取actor权重（RSL-RL的checkpoint中，actor权重通常在"model_state_dict"下）
    if "model_state_dict" in checkpoint:
        # 过滤出actor相关的权重（排除critic）
        for key, value in checkpoint["model_state_dict"].items():
            if key.startswith("actor."):
                # 原始键："actor.0.weight" → 去掉"actor."后为"0.weight"
                key_without_actor = key.replace("actor.", "")
                # 加上"layers."前缀 → "layers.0.weight"，匹配模型的层名
                new_key = f"layers.{key_without_actor}"
                actor_weights[new_key] = value
    else:
        for key, value in checkpoint.items():
            new_key = f"layers.{key}"
            actor_weights[new_key] = value
    
    # 加载调整后的权重
    model.load_state_dict(actor_weights, strict=True)
    model.eval()
    return model, checkpoint


# 应用观测归一化
def preprocess_input(random_data, obs_mean, obs_std, device):
    """对随机数据进行预处理（和训练时的观测归一化一致）"""
    # 转换为torch张量并添加批次维度 [1, 100]
    data = torch.tensor(random_data, dtype=torch.float32).unsqueeze(0)
    data = data.to(device)
    # 应用归一化：(x - mean) / std（和训练时的mean/std一致）
    normalized_data = (data - obs_mean) / (obs_std + 1e-6)
    return normalized_data


def main():
    MODEL_PATH = "/home/hy/IsaacLab/logs/rsl_rl/exo_humanoid_distillation/2025-11-02_20-31-58/model_0.pt"  # 你的训练模型路径
    
    USE_GPU = True  # 使用GPU
    device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")
    print(f"模型路径: {MODEL_PATH}")
    print("----------------------------------------")

    # 加载模型
    model, checkpoint = load_actor_model(MODEL_PATH, device)
    print("模型加载成功！开始推理...")
    for key in checkpoint.keys():
        print(f"- {key}")
    if "model_state_dict" in checkpoint:
        print("\nmodel_state_dict 中的子 key")
        for sub_key in checkpoint["model_state_dict"].keys():
            print(f"- {sub_key}")
    
    # 提取观测归一化参数
    obs_mean = checkpoint["model_state_dict"]["actor_obs_normalizer._mean"].to(device)
    obs_std = checkpoint["model_state_dict"]["actor_obs_normalizer._std"].to(device)
    print(f"obs_mean维度:{obs_mean.shape}")
    print(f"obs_std维度:{obs_std.shape}")
    print("\n观测归一化参数加载成功")

    # 推理
    step = 0
    try:
        while True:
            # 模拟传感器输入
            random_input = np.random.randn(100)  # 均值0、方差1的随机数据
            
            # 归一化
            processed_input = preprocess_input(random_input, obs_mean, obs_std, device)
            
            # 推理（禁用梯度计算，加速）
            with torch.no_grad():
                action = model(processed_input)  # 输出形状: [1, 39]
            # 后处理：转换为numpy数组并去除批次维度
            action_np = action.cpu().numpy().squeeze()  # 形状: [39]
            
            step += 1
            print(f"左膝关节action: {action_np[22].round(4)}")
            print(f"外骨骼左膝关节action: {action_np[38].round(4)}")
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
            print("----------------------------------------")

            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("推理终止")

# --------------------------
# 运行主函数
# --------------------------
if __name__ == "__main__":
    main()
