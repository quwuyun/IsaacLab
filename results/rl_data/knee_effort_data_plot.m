%% 读取CSV文件
% 若CSV含表头，将csvread改为readtable，后续用table.ColumnName调用列
base_path = 'C:\Users\30741\Desktop\rl_data';
% a_knee_amp(仅简单能量)
file1_path = fullfile(base_path, 'knee_human\a_torque_logs\torque_20251109_075622.csv');  % 1
file2_path = fullfile(base_path, 'knee_human\a_torque_logs\torque_20251109_140241.csv');  % 0.5
file3_path = fullfile(base_path, 'knee_human\a_torque_logs\torque_20251109_221926.csv');  % 0.6
file4_path = fullfile(base_path, 'knee_human\a_torque_logs\torque_20251110_132534.csv');  % 0.4
% human_amp=0.5
file5_path = fullfile(base_path, 'human\torque_20251109_004249.csv');
% b_knee_amp-action
file6_path = fullfile(base_path, 'knee_human\b_torque_logs\torque_20251111_011945.csv');
% c_energe
file7_path = fullfile(base_path, 'knee_human\c_torque_logs\torque_20251112_013321.csv');
file8_path = fullfile(base_path, 'knee_human\c_torque_logs\energe_torque_20251113_233826.csv');
% simple_knee_human
file9_path = fullfile(base_path, 'knee_human\d_torque_logs\energe_torque_20251115_163709.csv');

% effort_knee_human
file10_path = fullfile(base_path, 'knee_effort_human\torque-effort_logs\torque-effort_20251119_003147.csv');  % 外骨骼
file101_path = fullfile(base_path, 'knee_effort_human\torque-effort_logs\torque-effort_20251122_000919.csv');  % 外骨骼
file11_path = fullfile(base_path, 'human\pd_effort\torque_20251119_160219.csv');  % 人
% effort_knee_simple_human
file12_path = fullfile(base_path, 'knee_effort_simple_human\torque-effort_20251121_172748.csv');  % 外骨骼

% distillation
file13_path = fullfile(base_path, 'knee_distillation\torque-effort_20251123_142427.csv');  % 外骨骼

data1 = readtable(file1_path);
data2 = readtable(file2_path);
data3 = readtable(file3_path);
data4 = readtable(file4_path);
data5 = readtable(file5_path);
data6 = readtable(file6_path);
data7 = readtable(file7_path);
data8 = readtable(file8_path);
data9 = readtable(file9_path);
data10 = readtable(file10_path);
data101 = readtable(file101_path);
data11 = readtable(file11_path);
data12 = readtable(file12_path);
data13 = readtable(file13_path);

target_columns = [10, 23, 32, 34, 46, 48, 60, 62, 73];  % 髋t，膝t，髋pos，膝pos，髋vel，膝vel，action髋，action膝，r_exo_action
exo_target_colums = [73, 74];

fs = 60; % 采样频率
fc = 12;  % 截止频率
order = 4; % 滤波器阶数
% 创建巴特沃斯低通滤波器
[b, a] = butter(order, fc/(fs/2), 'low'); %fc/(fs/2)归一化截止频率

%% amp
% 提取当前组的列数据
col = target_columns(8);
y1 = data1{301:500, col};
y2 = data2{321:520, col};
y3 = data3{401:600, col};
y4 = data4{401:600, col};
x = [1:200]';

y1_filtered = filtfilt(b, a, y1);
y2_filtered = filtfilt(b, a, y2);
y3_filtered = filtfilt(b, a, y3);
y4_filtered = filtfilt(b, a, y4);

figure('Name', '髋关节数据对比', 'Position', [100, 100, 800, 600]);
grid on;
hold on;

% 绘制两条曲线（不同样式区分）
%plot(x, y1, 'b-', 'LineWidth', 1.2);  % 蓝色实线
%plot(x, y2, 'y-', 'LineWidth', 1.2);
%plot(x, y3, 'g-', 'LineWidth', 1.2);
%plot(x, y4, 'r-', 'LineWidth', 1.2);
plot(x, y1_filtered, 'b-', 'LineWidth', 1.2);
plot(x, y2_filtered, 'y-', 'LineWidth', 1.2);
plot(x, y3_filtered, 'g-', 'LineWidth', 1.2);
plot(x, y4_filtered, 'r-', 'LineWidth', 1.2);

% 添加图表标注（提升可读性）
xlabel('time', 'FontSize', 10);
ylabel('torque', 'FontSize', 10);
% title(['第', num2str(i), '张图：对应列', num2str(col), '对比'], 'FontSize', 11);
legend('Location', 'best');  % 自动放在最佳位置
hold off;

%% knee?
col = target_columns(8);
y5 = data5{301:500, col};
y6 = data6{320:519, col};
x = [1:200]';
y5_filtered = filtfilt(b, a, y5);
y6_filtered = filtfilt(b, a, y6);
figure('Name', '髋关节数据对比', 'Position', [100, 100, 800, 600]);
grid on;
hold on;
plot(x, y5, 'g-', 'LineWidth', 1.2);
plot(x, y6, 'r-', 'LineWidth', 1.2);
plot(x, y5_filtered, 'b-', 'LineWidth', 1.2);
plot(x, y6_filtered, 'r-', 'LineWidth', 1.2);
hold off;
%% energe准则

%% effort外骨骼膝关节力矩(膝关节能量权值变化)
col_exo_knee = target_columns(9);
col_exo_human = target_columns(2);
col_human = target_columns(2);
y_exo_knee = 100 * data101{101:500, col_exo_knee};
y_exo_human = data101{101:500, col_exo_human}-100 * data101{101:500, col_exo_knee};
y_knee = data101{101:500, col_exo_human};
y_human = data11{281:480, col_human};
x = [1:400]';
y_exo_knee_filtered = filtfilt(b, a, y_exo_knee);
y_exo_human_filtered = filtfilt(b, a, y_exo_human);
y_knee_filtered = filtfilt(b, a, y_knee);
y_human_filtered = filtfilt(b, a, y_human);

figure('Name', 'effort对比', 'Position', [100, 100, 800, 600]);
grid on;
hold on;

error_margin = 0.05 * y_exo_knee_filtered;  % 误差范围
y_upper = y_exo_knee_filtered + error_margin;  % 上边界
y_lower = y_exo_knee_filtered - error_margin;  % 下边界


%plot(x, y_exo_knee, 'b-', 'LineWidth', 1);
%plot(x, y_exo_human, 'y-', 'LineWidth', 0.7);
%plot(x, y_knee, 'r-', 'LineWidth', 1);
%plot(x, y_human, 'r-', 'LineWidth', 1.2);
plot(x, y_exo_knee_filtered, 'b-', 'LineWidth', 1.5);
plot(x, y_exo_human_filtered, 'y-', 'LineWidth', 1.2);
plot(x, y_knee_filtered, 'r-', 'LineWidth', 1.5);
%plot(x, y_human_filtered, 'r-', 'LineWidth', 1.2);
%errorbar(x, y_exo_knee_filtered, e, 'b-')

xlabel('timestep', 'FontSize', 10);
ylabel('torque/N', 'FontSize', 10);
legend('exo-torque','human-torque','applied-torque','Location', 'best');  % 自动放在最佳位置
hold off;

%% effort外骨骼膝关节力矩-无keypos
col_exo_knee_simple = target_columns(9);
col_exo_human_simple = target_columns(2);
y_exo_knee_simple = 100 * data12{301:500, col_exo_knee_simple};
y_applied_knee_simple = data12{301:500, col_exo_human_simple};
y_exo_human_simple = data12{301:500, col_exo_human_simple}-100 * data12{301:500, col_exo_knee_simple};
x = [1:200]';
y_exo_knee_simple_filtered = filtfilt(b, a, y_exo_knee_simple);
y_applied_knee_simple_filtered = filtfilt(b, a, y_applied_knee_simple);
y_exo_human_simple_filtered = filtfilt(b, a, y_exo_human_simple);
figure('Name', 'effort对比', 'Position', [100, 100, 800, 600]);
grid on;
hold on;
plot(x, y_exo_knee_simple, 'b-', 'LineWidth', 1.0);
plot(x, y_applied_knee_simple, 'r-', 'LineWidth', 1.0);
plot(x, y_exo_human_simple, 'y-', 'LineWidth', 0.8);
plot(x, y_exo_knee_simple_filtered, 'b-', 'LineWidth', 1.5);
plot(x, y_applied_knee_simple_filtered, 'r-', 'LineWidth', 1.5);
plot(x, y_exo_human_simple_filtered, 'y-', 'LineWidth', 1.2);

xlabel('timestep', 'FontSize', 10);
ylabel('torque/N', 'FontSize', 10);
legend('exo-torque','human-torque','applied-torque','Location', 'best');  % 自动放在最佳位置
hold off;

%% distillation
col_exo_knee_distillation = target_columns(9);
col_exo_human_distillation = target_columns(2);
y_exo_knee_distillation = 100 * data13{361:560, col_exo_knee_distillation};
y_applied_knee_distillation = data13{361:560, col_exo_human_distillation};
y_exo_human_distillation = y_applied_knee_distillation - y_exo_knee_distillation;
x = [1:200]';
y_exo_knee_distillation_filtered = filtfilt(b, a, y_exo_knee_distillation);
y_applied_knee_distillation_filtered = filtfilt(b, a, y_applied_knee_distillation);
y_exo_human_distillation_filtered = filtfilt(b, a, y_exo_human_distillation);
figure('Name', 'Distillation(77)', 'Position', [100, 100, 800, 600]);
grid on;
hold on;
%plot(x, y_exo_knee_distillation, 'b-', 'LineWidth', 1.0);
%plot(x, y_applied_knee_distillation, 'r-', 'LineWidth', 1.0);
%plot(x, y_exo_human_distillation, 'y-', 'LineWidth', 0.8);
plot(x, y_exo_knee_distillation_filtered, 'b-', 'LineWidth', 1.5);
plot(x, y_applied_knee_distillation_filtered, 'r-', 'LineWidth', 1.5);
plot(x, y_exo_human_distillation_filtered, 'y-', 'LineWidth', 1.2);

title('Distillation')
xlabel('timestep', 'FontSize', 10);
ylabel('torque/N', 'FontSize', 10);
legend('exo-torque','human-torque','applied-torque','Location', 'best');  % 自动放在最佳位置
hold off;