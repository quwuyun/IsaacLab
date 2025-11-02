import torch
import numpy as np
import time
from datetime import datetime

# --------------------------
# 1. 定义与训练时一致的网络结构
# --------------------------
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

# --------------------------
# 2. 加载模型权重
# --------------------------
def load_actor_model(model_path, device):
    """加载训练好的RSL-RL模型权重"""
    # 初始化网络
    model = ActorNetwork(input_dim=100, output_dim=39).to(device)
    
    # 加载权重（处理RSL-RL的checkpoint格式）
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
            # 原始键："0.weight" → 加上"layers." → "layers.0.weight"
            new_key = f"layers.{key}"
            actor_weights[new_key] = value
    
    # 加载调整后的权重
    model.load_state_dict(actor_weights, strict=True)
    model.eval()
    return model

# --------------------------
# 3. 数据预处理（模拟训练时的归一化）
# --------------------------
def preprocess_input(random_data, obs_mean, obs_std, device):
    """对随机数据进行预处理（和训练时的观测归一化一致）"""
    # 转换为torch张量并添加批次维度 [1, 100]
    data = torch.tensor(random_data, dtype=torch.float32).unsqueeze(0)
    data = data.to(device)
    # 应用归一化：(x - mean) / std（必须和训练时的mean/std一致）
    normalized_data = (data - obs_mean) / (obs_std + 1e-6)  # 加小值避免除零
    return normalized_data

# --------------------------
# 4. 主循环：生成随机数据→推理→打印输出
# --------------------------
def main():
    # --------------------------
    # 配置参数（请根据你的实际情况修改）
    # --------------------------
    MODEL_PATH = "/home/hy/IsaacLab/logs/rsl_rl/exo_humanoid_distillation/2025-11-02_20-31-58/model_0.pt"  # 你的训练模型路径
    USE_GPU = True  # 是否使用GPU（无GPU自动切换到CPU）
    PRINT_INTERVAL = 0  # 打印间隔（秒），控制输出频率
    # 训练时的观测归一化参数（必须从训练日志中提取！这里先用随机值示例，需替换）
    OBS_MEAN = torch.zeros(100)  # 实际应替换为训练时的obs_mean
    OBS_STD = torch.ones(100)   # 实际应替换为训练时的obs_std

    # 选择设备（GPU/CPU）
    device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")
    print(f"模型路径: {MODEL_PATH}")
    print("----------------------------------------")

    # 加载模型
    try:
        model = load_actor_model(MODEL_PATH, device)
        print("模型加载成功！开始推理...")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 将归一化参数移到对应设备
    obs_mean = OBS_MEAN.to(device)
    obs_std = OBS_STD.to(device)

    # 循环推理（持续生成随机数据并输出）
    step = 0
    try:
        while True:
            # 生成随机100维数据（模拟传感器输入）
            random_input = np.random.randn(100)  # 均值0、方差1的随机数据
            
            # 预处理
            processed_input = preprocess_input(random_input, obs_mean, obs_std, device)
            
            # 推理（禁用梯度计算，加速）
            with torch.no_grad():
                action = model(processed_input)  # 输出形状: [1, 39]
            
            # 后处理：转换为numpy数组并去除批次维度
            action_np = action.cpu().numpy().squeeze()  # 形状: [39]
            
            # 打印输出（每步打印前5维和后5维，避免过长）
            step += 1
            print(f"第 {step} 步 - 动作输出（39维）:")
            print(f"左膝关节action: {action_np[22].round(4)}")
            print(f"外骨骼左膝关节action: {action_np[38].round(4)}")

            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
            print("----------------------------------------")
            # 控制打印频率
            time.sleep(PRINT_INTERVAL)
    
    except KeyboardInterrupt:
        print("推理终止")

# --------------------------
# 运行主函数
# --------------------------
if __name__ == "__main__":
    main()
