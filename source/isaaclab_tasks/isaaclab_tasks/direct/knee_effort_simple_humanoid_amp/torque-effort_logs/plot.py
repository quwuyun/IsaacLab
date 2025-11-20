import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --------------------------
# 关节分组与样式配置（最终优化版）
# --------------------------
# 人体关节（力矩列）
HUMAN_TORQUE_JOINTS = [
    'abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z',
    'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
    'right_hip_x', 'right_hip_y', 'right_hip_z',
    'left_hip_x', 'left_hip_y', 'left_hip_z',
    'right_elbow', 'left_elbow', 'right_knee', 'left_knee',
    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
    'left_ankle_x', 'left_ankle_y', 'left_ankle_z'
]

# 人体下肢关节（角速度列）
HUMAN_VEL_PREFIX = "vel_"
HUMAN_VEL_JOINTS = [
    'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee',
    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
    'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee',
    'left_ankle_x', 'left_ankle_y', 'left_ankle_z'
]
HUMAN_VEL_FULL_NAMES = [f"{HUMAN_VEL_PREFIX}{name}" for name in HUMAN_VEL_JOINTS]

# 动作列
HUMAN_ACTION_JOINTS = [
    'action_right_hip_x', 'action_right_hip_y', 'action_right_hip_z', 'action_right_knee',
    'action_right_ankle_x', 'action_right_ankle_y', 'action_right_ankle_z',
    'action_left_hip_x', 'action_left_hip_y', 'action_left_hip_z', 'action_left_knee',
    'action_left_ankle_x', 'action_left_ankle_y', 'action_left_ankle_z'
]
EXO_ACTION_JOINTS = ['action_exo_right_knee', 'action_exo_left_knee']
ACTION_JOINTS = HUMAN_ACTION_JOINTS + EXO_ACTION_JOINTS

# 基础样式（同类型关节统一风格，确保一致性）
STYLES = {
    "human_torque": {'color': 'tab:red', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 1.6, 'label': '人体关节力矩 (N·m)'},
    "human_vel": {'color': 'tab:blue', 'linestyle': '--', 'alpha': 0.8, 'linewidth': 1.6, 'label': '人体下肢角速度 (rad/s)'},
    "action_default": {'color': 'tab:orange', 'linestyle': '-.', 'alpha': 0.8, 'linewidth': 1.6, 'label': '关节动作（默认）'}
}

# 动作关节独立样式（重点关节单独配色，支持统一实线）
ACTION_STYLES = {
    'action_exo_right_knee': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.9, 'label': '外骨骼_右膝'},
    'action_exo_left_knee': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.9, 'label': '外骨骼_左膝'},
    'action_right_knee': {'color': '#d62728', 'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.9, 'label': '人体_右膝'},
    'action_left_knee': {'color': '#9467bd', 'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.9, 'label': '人体_左膝'},
    # 可扩展其他动作关节的独立样式
    'action_right_hip_x': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 1.8, 'alpha': 0.8, 'label': '人体_右髋X'},
    'action_left_hip_x': {'color': '#17becf', 'linestyle': '-', 'linewidth': 1.8, 'alpha': 0.8, 'label': '人体_左髋X'}
}


def parse_args():
    parser = argparse.ArgumentParser(description="绘制关节数据曲线（支持力矩/角速度/动作，稳定版）")
    parser.add_argument("--log_file", type=str, required=True, help="日志CSV文件路径")
    parser.add_argument("--data_type", type=str, choices=['torque', 'velocity', 'action', 'both'], default='both',
                       help="数据类型：torque（仅力矩）、velocity（仅角速度）、action（仅动作）、both（力矩+角速度）")
    parser.add_argument("--joints", type=str, nargs='*', default=None, 
                       help="指定关节名称（例：right_knee 或 vel_right_knee 或 action_exo_right_knee）")
    parser.add_argument("--title", type=str, default="关节数据曲线", help="图表标题")
    parser.add_argument("--save", type=str, default=None, help="保存图片路径（如：joint_curves.png）")
    parser.add_argument("--time_range", type=float, nargs=2, default=None, 
                       help="时间范围筛选（单位：秒），例：--time_range 1.0 5.0")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 8], help="图表尺寸（宽, 高）")
    parser.add_argument("--legend_loc", type=str, default='upper right', help="图例位置（如：upper right、lower left、best）")
    parser.add_argument("--grid_alpha", type=float, default=0.3, help="网格透明度（0-1，默认0.3，避免遮挡曲线）")
    return parser.parse_args()


def load_data(log_file):
    """加载数据并增强校验，处理异常值"""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"日志文件不存在：{log_file}")
    
    df = pd.read_csv(log_file)
    required_cols = ['timestamp', 'timestep']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"日志文件缺少必要列：{missing_required}")
    
    # 筛选有效数据列
    torque_cols = [col for col in HUMAN_TORQUE_JOINTS if col in df.columns]
    vel_cols = [col for col in HUMAN_VEL_FULL_NAMES if col in df.columns]
    action_cols = [col for col in ACTION_JOINTS if col in df.columns]
    
    # 打印列名校验信息
    print(f"[列名校验] CSV中实际存在的列数：")
    print(f"- 力矩列：{len(torque_cols)}/{len(HUMAN_TORQUE_JOINTS)} 个（例：{torque_cols[:3]}...）")
    print(f"- 角速度列：{len(vel_cols)}/{len(HUMAN_VEL_FULL_NAMES)} 个（例：{vel_cols[:3]}...）")
    print(f"- 动作列：{len(action_cols)}/{len(ACTION_JOINTS)} 个（例：{action_cols[:3]}...）")
    
    # 关键列缺失校验
    if not torque_cols and ('torque' in args.data_type or args.data_type == 'both'):
        raise ValueError("CSV中未找到任何力矩列，请检查列名是否匹配")
    if not vel_cols and ('velocity' in args.data_type or args.data_type == 'both'):
        raise ValueError("CSV中未找到任何角速度列，请检查列名是否匹配")
    if not action_cols and args.data_type == 'action':
        raise ValueError("CSV中未找到任何动作列，请检查列名是否匹配")
    
    # 数据清洗：删除timestamp或关节列中的NaN值
    all_data_cols = torque_cols + vel_cols + action_cols + required_cols
    df_clean = df[all_data_cols].dropna(subset=['timestamp']).reset_index(drop=True)
    print(f"[数据清洗] 原始数据：{len(df)} 帧 → 清洗后：{len(df_clean)} 帧（删除NaN值）")
    
    return df_clean, torque_cols, vel_cols, action_cols


def filter_joints_by_type(torque_cols, vel_cols, action_cols, user_joints, data_type):
    """精准筛选关节，优化错误提示"""
    valid_joints = []
    process_torque = (data_type in ['torque', 'both'])
    process_vel = (data_type in ['velocity', 'both'])
    process_action = (data_type == 'action')
    
    all_valid_cols = torque_cols + vel_cols + action_cols
    
    if user_joints is None:
        # 默认加载当前类型所有关节
        if process_torque:
            valid_joints.extend(torque_cols)
        if process_vel:
            valid_joints.extend(vel_cols)
        if process_action:
            valid_joints.extend(action_cols)
    else:
        # 校验用户输入关节
        invalid_joints = [j for j in user_joints if j not in all_valid_cols]
        if invalid_joints:
            raise ValueError(
                f"无效关节名称：{invalid_joints}\n"
                f"可用关节列表：\n"
                f"- 力矩：{torque_cols}\n"
                f"- 角速度：{vel_cols}\n"
                f"- 动作：{action_cols}"
            )
        
        # 筛选目标关节
        for joint in user_joints:
            if process_torque and joint in torque_cols:
                valid_joints.append(joint)
            elif process_vel and joint in vel_cols:
                valid_joints.append(joint)
            elif process_action and joint in action_cols:
                valid_joints.append(joint)
    
    if not valid_joints:
        raise ValueError(f"无符合数据类型「{data_type}」的关节，请检查关节名称或数据类型参数")
    
    return valid_joints


def filter_time_range(df, time_range):
    """优化时间范围筛选，增加范围提示"""
    if time_range is None:
        return df
    
    start, end = time_range
    if start >= end:
        raise ValueError(f"时间范围无效：起始时间 {start} ≥ 结束时间 {end}")
    
    # 确保timestamp为数值型
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)
    
    data_min = df['timestamp'].min()
    data_max = df['timestamp'].max()
    if start < data_min or end > data_max:
        print(f"[警告] 指定时间范围超出数据范围，数据实际范围：{data_min:.2f} - {data_max:.2f} 秒")
    
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    filtered_df = df[mask].reset_index(drop=True)
    
    if len(filtered_df) == 0:
        raise ValueError(f"时间范围 [{start}, {end}] 内无有效数据")
    
    return filtered_df


def plot_curves(df, joints, title, figsize, save_path, data_type, legend_loc, grid_alpha):
    """优化绘图逻辑，增强可视化效果"""
    plt.style.use('seaborn-v0_8-notebook')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
    plt.rcParams['legend.fontsize'] = 10  # 图例字体优化
    plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
    plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签大小
    
    # 初始化子图（按数据类型适配）
    if data_type == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'hspace': 0.15})
        axes_map = {"human_torque": ax1, "human_vel": ax2}
        ax2.set_xlabel("仿真时间 (秒)", fontsize=12)
        ax1.set_title(title, pad=15, fontweight='bold')
    elif data_type == 'action':
        fig, ax = plt.subplots(figsize=figsize)
        axes_map = {"action": ax}
        ax.set_xlabel("仿真时间 (秒)", fontsize=12)
        ax.set_title(title, pad=15, fontweight='bold')
    elif data_type == 'torque':
        fig, ax = plt.subplots(figsize=figsize)
        axes_map = {"human_torque": ax}
        ax.set_xlabel("仿真时间 (秒)", fontsize=12)
        ax.set_title(title, pad=15, fontweight='bold')
    elif data_type == 'velocity':
        fig, ax = plt.subplots(figsize=figsize)
        axes_map = {"human_vel": ax}
        ax.set_xlabel("仿真时间 (秒)", fontsize=12)
        ax.set_title(title, pad=15, fontweight='bold')
    
    added_legends = set()  # 图例去重
    
    for joint in joints:
        # 判定关节类型
        if joint in HUMAN_TORQUE_JOINTS:
            style_key = "human_torque"
            ylabel = "关节力矩 (N·m)"
        elif joint in HUMAN_VEL_FULL_NAMES:
            style_key = "human_vel"
            ylabel = "关节角速度 (rad/s)"
        elif joint in ACTION_JOINTS:
            style_key = "action"
            ylabel = "关节动作"
        else:
            continue
        
        # 获取对应坐标轴
        if style_key == "action":
            ax = axes_map["action"]
        else:
            ax = axes_map.get(style_key, None)
        if ax is None:
            continue
        
        # 选择样式（动作关节优先使用独立样式）
        if style_key == "action":
            if joint in ACTION_STYLES:
                style = ACTION_STYLES[joint]
                label = style['label']
            else:
                style = STYLES["action_default"]
                label = f"{style['label']}({joint.replace('action_', '')})"
        else:
            style = STYLES[style_key]
            label = style['label'] if style['label'] not in added_legends else ""
        
        # 绘制曲线
        ax.plot(
            df['timestamp'], df[joint],
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=style['linewidth'],
            alpha=style['alpha'],
            label=label
        )
        
        # 设置坐标轴标签（仅设置一次）
        if ylabel not in added_legends:
            added_legends.add(ylabel)
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        # 记录图例（避免重复）
        if label and label not in added_legends:
            added_legends.add(label)
    
    # 美化图表
    for ax in axes_map.values():
        # 坐标轴优化
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴仅显示整数秒
        ax.grid(True, linestyle='--', alpha=grid_alpha, linewidth=0.8)  # 网格不遮挡曲线
        # 图例优化（自动调整位置，避免遮挡）
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(dict.fromkeys(labels))
        unique_handles = [handles[labels.index(lbl)] for lbl in unique_labels if lbl]
        ax.legend(unique_handles, unique_labels, loc=legend_loc, framealpha=0.9, ncol=1 if len(unique_labels) <=5 else 2)
        # 刻度优化
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n图表已保存至：{save_path}")
    else:
        plt.show()


def main():
    global args
    args = parse_args()
    
    try:
        # 加载并清洗数据
        df, torque_cols, vel_cols, action_cols = load_data(args.log_file)
        print(f"\n成功加载日志文件：{args.log_file}")
        print(f"数据时间范围：{df['timestamp'].min():.2f} - {df['timestamp'].max():.2f} 秒")
        
        # 筛选目标关节
        target_joints = filter_joints_by_type(torque_cols, vel_cols, action_cols, args.joints, args.data_type)
        print(f"本次绘制关节数：{len(target_joints)} 个（{args.data_type}类型）→ {target_joints}")
        
        # 筛选时间范围
        df_filtered = filter_time_range(df, args.time_range)
        print(f"时间范围筛选后：{len(df_filtered)} 帧数据")
        
        # 绘制曲线
        plot_curves(
            df=df_filtered,
            joints=target_joints,
            title=args.title,
            figsize=args.figsize,
            save_path=args.save,
            data_type=args.data_type,
            legend_loc=args.legend_loc,
            grid_alpha=args.grid_alpha
        )
    
    except Exception as e:
        print(f"\n错误信息：{str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
