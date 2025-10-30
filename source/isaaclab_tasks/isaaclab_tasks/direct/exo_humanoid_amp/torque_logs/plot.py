import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --------------------------
# 关节分组与样式配置（扩展版）
# --------------------------
# 人体关节（力矩）
HUMAN_TORQUE_JOINTS = [
    'abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z',
    'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_elbow',
    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_elbow',
    'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee',
    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
    'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee',
    'left_ankle_x', 'left_ankle_y', 'left_ankle_z'
]

# 外骨骼关节（力矩）
EXO_TORQUE_JOINTS = [
    "exo_D6Joint0:0", "exo_D6Joint0:1", "exo_D6Joint0:2", "exo_right_hip:0", "exo_right_hip:1", "exo_right_hip:2", 
    "exo_left_hip:0", "exo_left_hip:1", "exo_left_hip:2", "exo_right_knee", "exo_left_knee"
]

# 人体下肢关节（角速度，对应CSV中的前缀）
HUMAN_VEL_PREFIX = "vel_"
HUMAN_VEL_JOINTS = [
    'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
    'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z'
]
HUMAN_VEL_FULL_NAMES = [f"{HUMAN_VEL_PREFIX}{name}" for name in HUMAN_VEL_JOINTS]

# 外骨骼下肢关节（角速度，对应CSV中的前缀）
EXO_VEL_PREFIX = "vel_"
EXO_VEL_JOINTS = [
    "exo_D6Joint0:0", "exo_D6Joint0:1", "exo_D6Joint0:2", "exo_right_hip:0", "exo_right_hip:1", "exo_right_hip:2", 
    "exo_left_hip:0", "exo_left_hip:1", "exo_left_hip:2", "exo_right_knee", "exo_left_knee"
]
EXO_VEL_FULL_NAMES = [f"{EXO_VEL_PREFIX}{name}" for name in EXO_VEL_JOINTS]

# 样式配置（区分力矩/角速度，人体/外骨骼）
STYLES = {
    "human_torque": {'color': 'tab:blue', 'linestyle': '-', 'alpha': 0.8, 'label': '人体关节力矩'},
    "exo_torque": {'color': 'tab:red', 'linestyle': '-', 'alpha': 0.8, 'label': '外骨骼关节力矩'},
    "human_vel": {'color': 'tab:blue', 'linestyle': '--', 'alpha': 0.8, 'label': '人体下肢角速度'},
    "exo_vel": {'color': 'tab:red', 'linestyle': '--', 'alpha': 0.8, 'label': '外骨骼下肢角速度'}
}


def parse_args():
    parser = argparse.ArgumentParser(description="绘制力矩和角速度曲线（支持同时显示）")
    parser.add_argument("--log_file", type=str, required=True, help="日志CSV文件路径（含力矩和角速度）")
    parser.add_argument("--data_type", type=str, choices=['torque', 'velocity', 'both'], default='both',
                       help="数据类型：torque（仅力矩）、velocity（仅角速度）、both（两者都画）")
    parser.add_argument("--joints", type=str, nargs='*', default=None, 
                       help="指定关节名称（支持力矩关节名或角速度前缀+关节名，例如：right_hip_x 或 human_lower_vel_right_hip_x）")
    parser.add_argument("--title", type=str, default="关节力矩与角速度曲线", help="图表标题")
    parser.add_argument("--save", type=str, default=None, help="保存图片路径（如 torque_vel_plot.png）")
    parser.add_argument("--time_range", type=float, nargs=2, default=None, 
                       help="时间范围筛选（单位：秒），例如：--time_range 1.0 5.0")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 8], help="图表尺寸（宽, 高），默认12 8")
    return parser.parse_args()


def load_data(log_file):
    """加载CSV并区分力矩和角速度列"""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"日志文件不存在：{log_file}")
    
    df = pd.read_csv(log_file)
    required_cols = ['timestamp', 'timestep']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"日志文件缺少必要列 {required_cols}")
    
    all_vel_names = HUMAN_VEL_FULL_NAMES + EXO_VEL_FULL_NAMES
    
    # 分离力矩列和角速度列
    torque_cols = []
    vel_cols = []
    for col in df.columns:
        if col in required_cols:
            continue
        if col in all_vel_names:
            vel_cols.append(col)
        else:
            torque_cols.append(col)
    
    return df, torque_cols, vel_cols


def filter_joints_by_type(all_torque, all_vel, user_joints, data_type):
    """根据数据类型和用户输入筛选关节"""
    valid_joints = []
    
    # 确定需要处理的数据类型
    process_torque = (data_type in ['torque', 'both'])
    process_vel = (data_type in ['velocity', 'both'])
    
    if user_joints is None:
        # 默认选择所有对应类型的关节
        if process_torque:
            valid_joints.extend(all_torque)
        if process_vel:
            valid_joints.extend(all_vel)
        return valid_joints
    
    # 验证用户输入的关节
    all_valid = all_torque + all_vel
    invalid = [j for j in user_joints if j not in all_valid]
    if invalid:
        raise ValueError(f"无效关节名称：{invalid}，可用关节：\n力矩关节：{all_torque[:5]}...\n角速度关节：{all_vel[:5]}...")
    
    # 根据数据类型过滤用户输入
    for joint in user_joints:
        if process_torque and joint in all_torque:
            valid_joints.append(joint)
        if process_vel and joint in all_vel:
            valid_joints.append(joint)
    
    if not valid_joints:
        raise ValueError(f"没有符合数据类型 {data_type} 的关节")
    return valid_joints


def filter_time_range(df, time_range):
    """按时间范围筛选数据（复用原逻辑）"""
    if time_range is None:
        return df
    
    start, end = time_range
    if start >= end:
        raise ValueError(f"时间范围无效：start={start} >= end={end}")
    
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    filtered_df = df[mask]
    if len(filtered_df) == 0:
        raise ValueError(f"时间范围内无数据：{start} - {end}秒")
    return filtered_df


def plot_curves(df, joints, title, figsize, save_path, data_type):
    """绘制力矩和角速度曲线（支持子图分离）"""
    plt.style.use('seaborn-v0_8-notebook')
    
    # 根据数据类型决定是否使用子图
    if data_type == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'hspace': 0.1})
        axes = {'torque': ax1, 'velocity': ax2}
        ax2.set_xlabel("仿真时间 (秒)", fontsize=12)
        ax1.set_title(title, fontsize=14, pad=10)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = {data_type: ax}
        ax.set_xlabel("仿真时间 (秒)", fontsize=12)
        ax.set_title(title, fontsize=14, pad=10)
    
    # 记录已添加的图例（避免重复）
    added_legends = set()
    
    for joint in joints:
        # 判断关节类型（力矩/角速度，人体/外骨骼）
        if joint in HUMAN_TORQUE_JOINTS:
            style_key = "human_torque"
            data_type_joint = "torque"
            ylabel = "关节力矩 (N·m)"
        elif joint in EXO_TORQUE_JOINTS:
            style_key = "exo_torque"
            data_type_joint = "torque"
            ylabel = "关节力矩 (N·m)"
        elif joint in HUMAN_VEL_FULL_NAMES:
            style_key = "human_vel"
            data_type_joint = "velocity"
            ylabel = "关节角速度 (rad/s)"
        elif joint in EXO_VEL_FULL_NAMES:
            style_key = "exo_vel"
            data_type_joint = "velocity"
            ylabel = "关节角速度 (rad/s)"
        else:
            continue  # 跳过未识别的关节
        
        # 获取对应的坐标轴
        ax = axes[data_type_joint]
        
        # 绘制曲线
        style = STYLES[style_key]
        ax.plot(df['timestamp'], df[joint], 
                linewidth=1.5, 
                color=style['color'], 
                linestyle=style['linestyle'], 
                alpha=style['alpha'],
                label=style['label'] if style['label'] not in added_legends else "")
        
        # 添加图例（只加一次）
        if style['label'] not in added_legends:
            added_legends.add(style['label'])
        
        # 设置坐标轴标签
        ax.set_ylabel(ylabel, fontsize=12)
    
    # 美化所有坐标轴
    for ax in axes.values():
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至：{save_path}")
    else:
        plt.show()


def main():
    args = parse_args()
    
    try:
        # 加载数据并分离力矩和角速度列
        df, torque_cols, vel_cols = load_data(args.log_file)
        print(f"成功加载日志：{args.log_file}，包含 "
              f"{len(torque_cols)} 个力矩关节，{len(vel_cols)} 个角速度关节，共 {len(df)} 帧数据")
        
        # 筛选关节
        target_joints = filter_joints_by_type(torque_cols, vel_cols, args.joints, args.data_type)
        print(f"将绘制以下关节（{args.data_type}）：{target_joints[:5]}...")  # 只显示前5个避免过长
        
        # 筛选时间范围
        df_filtered = filter_time_range(df, args.time_range)
        
        # 绘制曲线
        plot_curves(
            df=df_filtered,
            joints=target_joints,
            title=args.title,
            figsize=args.figsize,
            save_path=args.save,
            data_type=args.data_type
        )
    
    except Exception as e:
        print(f"出错：{e}")
        exit(1)


if __name__ == "__main__":
    main()
