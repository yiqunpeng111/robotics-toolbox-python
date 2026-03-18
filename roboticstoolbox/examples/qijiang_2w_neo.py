#!/usr/bin/env python
"""
适配qijiang双臂移动机器人的NEO反应式规划算法
基于Jesse Haviland原版NEO代码修改，适配移动底盘+躯干+双臂全自由度
"""
import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp

# ====================== 1. 机器人与环境基础配置 ======================
# 【必须修改】你的qijiang机器人URDF文件绝对路径
URDF_PATH = "/home/user/2w_new/urdf/2w_new.urdf"

# 机器人核心配置
LEFT_END_LINK = "arml_7"   # 左臂末端连杆名（从你的ETS表获取）
RIGHT_END_LINK = "armr_7"  # 右臂末端连杆名（从你的ETS表获取）
n = 24  # 总主动关节数：底盘3+躯干/脖子6+左臂7+右臂7 = 24个自由度
SLACK_N = 15  # 松弛自由度数，取决于多少个变量约束
TORSO_4_LINK = "body-4-link"  # 新增：躯干第4连杆名称
DESIRED_Z_AXIS = np.array([0, 0, 1])  # 期望z轴方向（世界坐标系向上）

# 启动Swift仿真环境
env = swift.Swift()
env.launch(realtime=True)  # realtime=True开启实时仿真

# 加载qijiang机器人URDF模型
robot = rtb.ERobot.URDF(URDF_PATH)
# 设置机器人初始构型（置零，可根据需求修改）
# robot.q = robot.qr
print('robot.link_dict',robot.q)
print("="*50)
print(f"机器人加载成功！总关节数：{robot.n}")
# print(f"关节名称列表：{robot.joint_names()}")
print(f"左臂末端连杆：{LEFT_END_LINK}")
print(f"右臂末端连杆：{RIGHT_END_LINK}")
print("="*50)

# ====================== 2. 动态障碍物与目标配置 ======================
# --- 创建动态障碍物（可根据需求增减数量、修改速度/大小）
# 障碍物1：沿Y轴负方向运动
s0 = sg.Sphere(radius=0.08, pose=sm.SE3(1.2, 0.6, 0.6))
s0.v = [0, -0.3, 0, 0, 0, 0]  # 6维速度：[vx, vy, vz, wx, wy, wz]

# 障碍物2：沿Y轴负方向运动
s1 = sg.Sphere(radius=0.08, pose=sm.SE3(0.8, 0.5, 0.6))
s1.v = [0, -0.3, 0, 0, 0, 0]

collisions = [s0, s1]

# --- 创建双臂目标点（可根据需求修改位置）
# 左臂目标
target_left = sg.Sphere(radius=0.03, pose=sm.SE3(2.6, 0.3, 0.8), color=[0, 1, 0, 0.5])
# 右臂目标
target_right = sg.Sphere(radius=0.03, pose=sm.SE3(2.6, -0.8, 0.8), color=[1, 0, 0, 0.5])

# --- 将机器人、障碍物、目标添加到仿真环境
env.add(robot)
for obs in collisions:
    env.add(obs)
env.add(target_left)
env.add(target_right)

# --- 设置双臂期望末端位姿（保持初始姿态，仅平移到目标点）
# 左臂期望位姿
Tep_left = robot.fkine(robot.q, end=LEFT_END_LINK)
Tep_left.A[:3, 3] = target_left.T[:3, -1]  # 替换为目标点位置
print('target_left.T',target_left.T[:3, -1])

# 右臂期望位姿
Tep_right = robot.fkine(robot.q, end=RIGHT_END_LINK)
Tep_right.A[:3, 3] = target_right.T[:3, -1]  # 替换为目标点位置


J_body4 = robot.jacobe(robot.q, end=TORSO_4_LINK).astype(np.float64)
print('J_body4',J_body4)
print('J_body4.shape',J_body4.shape)

# ====================== 3. NEO核心控制步函数 ======================
def step():
    # ---------------------- 3.1 末端位姿与误差计算 ----------------------
    # 双臂位姿与误差
    Te_left = robot.fkine(robot.q, end=LEFT_END_LINK)
    Te_right = robot.fkine(robot.q, end=RIGHT_END_LINK)
    eTep_left = Te_left.inv() * Tep_left
    eTep_right = Te_right.inv() * Tep_right
    err_left = np.sum(np.abs(np.r_[eTep_left.t, eTep_left.rpy() * np.pi / 180]))
    err_right = np.sum(np.abs(np.r_[eTep_right.t, eTep_right.rpy() * np.pi / 180]))
    e_arm = err_left + err_right  # 双臂总误差

    # ---------------------- 核心修正：计算body-4-link z轴姿态误差 ----------------------
    # 1. 获取body-4-link当前位姿
    Te_torso4 = robot.fkine(robot.q, end=TORSO_4_LINK)
    # 2. 提取当前z轴方向向量（连杆坐标系z轴在世界坐标系中的方向）
    current_z_axis = Te_torso4.R[:, 2]  # R是旋转矩阵，第3列=z轴方向
    # 3. 计算z轴方向误差（叉乘：当前z轴 × 期望z轴 = 需修正的角速度方向）
    z_axis_error = np.cross(current_z_axis, DESIRED_Z_AXIS)
    # 4. 计算姿态误差的幅值（归一化后乘以增益，得到角速度指令）
    gain_torso = 1.0  # 躯干姿态伺服增益（可调整）
    torso_wz = gain_torso * z_axis_error  # 3维角速度指令（wx, wy, wz）
    # 5. 只保留旋转部分的速度（平移无约束），拼接为6维空间速度（平移0 + 旋转）
    v_torso4 = np.r_[np.zeros(3), torso_wz]  # [0,0,0, wx, wy, wz]
    # 6. 计算躯干z轴误差（用于打印）
    err_torso4 = np.linalg.norm(z_axis_error)

    # ---------------------- 3.2 期望速度合并 ----------------------
    # 双臂期望速度
    v_left, arrived_left = rtb.p_servo(Te_left, Tep_left, gain=0.5, threshold=0.02)
    v_right, arrived_right = rtb.p_servo(Te_right, Tep_right, gain=0.5, threshold=0.02)
    # 合并：双臂速度（12维） + 躯干z轴修正速度（仅旋转部分，1维核心）
    v_arm = np.r_[v_left.reshape(6,), v_right.reshape(6,)]
    arrived = arrived_left and arrived_right and (err_torso4 < 0.01)
    

    # ---------------------- 3.3 Q矩阵构建（分关节权重） ----------------------
    base_torso_idx = [0,1,2,3,4,5,6]  # 底盘+躯干关节索引
    Y_base = 1.0    # 底盘/躯干权重
    Y_arm = 0.01    # 双臂权重
    joint_weights = Y_arm * np.ones(n, dtype=np.float64)
    joint_weights[base_torso_idx] = Y_base

    # Q矩阵（36维：24关节+SLACK_N松弛）
    Q = np.eye(n + SLACK_N, dtype=np.float64)
    np.fill_diagonal(Q[:n, :n], joint_weights)
    Q[n:, n:] = (100 / (e_arm + err_torso4 + 1e-6)) * np.eye(SLACK_N)  # 避免除零

    # ---------------------- 3.4 QP等式约束构建（含躯干z轴约束） ----------------------
    # 1. 双臂雅可比矩阵
    J_total_l = np.zeros((6, n), dtype=np.float64)
    J_left = robot.jacobe(robot.q, end=LEFT_END_LINK).astype(np.float64)
    J_total_l[:,[0,1,2,3,4,5,6,10,11,12,13,14,15,16]] = J_left

    J_total_r = np.zeros((6, n), dtype=np.float64)
    J_right = robot.jacobe(robot.q, end=RIGHT_END_LINK).astype(np.float64)
    J_total_r[:,[0,1,2,3,4,5,6,17,18,19,20,21,22,23]] = J_right

    J_arm = np.r_[J_total_l, J_total_r]  # 12x24

    # 2. 躯干z轴约束（核心修正：beq为姿态误差对应的速度）
    # 躯干4连杆雅可比矩阵（6x7：前3平移，后3旋转）
    J_torso4 = robot.jacobe(robot.q, end=TORSO_4_LINK).astype(np.float64)
    # 只保留旋转部分雅可比（后3行），对应角速度指令
    J_torso4_rot = J_torso4[3:, :]  # 3x7
    # 补充0，使雅可比矩阵维度与关节数一致（3x24）
    J_torso4_rot = np.c_[J_torso4_rot, np.zeros((3, 24-7), dtype=np.float64)]  # 3x36
    print('J_torso4_rot.shape',J_torso4_rot.shape)

    # 3. 合并等式约束（12+3=15行）
    Jacob = np.r_[J_arm, J_torso4_rot]  # 15x24
    v = np.r_[v_arm.reshape(12,), torso_wz.reshape(3,)].astype(np.float64)
    Aeq = np.c_[Jacob, np.eye(SLACK_N)]  # 15x39
    beq = v.reshape(SLACK_N,)
    # -------------------------------------------------------------------------

    # ---------------------- 3.4 QP不等式约束构建（关节限位+避障） ----------------------
    # 初始化不等式约束矩阵（float64）
    Ain = np.zeros((n + SLACK_N, n + SLACK_N), dtype=np.float64)
    bin = np.zeros(n + SLACK_N, dtype=np.float64)

    # --- 关节限位速度阻尼器 ---
    # 旋转关节参数（单位：rad）
    ps_rot = 0.05    # 关节到限位的最小安全距离
    pi_rot = 0.7     # 阻尼器激活的影响距离
    # 平移关节参数（底盘x/y轴，单位：m）
    ps_pris = 0.01   # 平移关节最小安全距离
    pi_pris = 0.1    # 平移关节阻尼器激活距离

    # 生成全关节限位约束（函数自动适配平移/旋转关节，可手动替换参数）
    ain_joint, bin_joint = robot.joint_velocity_damper(ps=ps_rot, pi=pi_rot, n=n)
    Ain[:n, :n] = ain_joint.astype(np.float64)
    Ain[:3, :3] = 0.0
    # print('Ain',Ain[:7,:7])
    bin[:n] = bin_joint.astype(np.float64)
    bin[:3] = 0.0
    # print('bin',bin)

    # --- 动态障碍物避障速度阻尼器 ---
    # 避障参数
    d_i = 0.4    # 障碍物影响距离（单位：m）
    d_s = 0.09   # 最小安全停止距离（单位：m）
    xi = 1.0     # 避障增益
    # print('robot.q ',robot.q)
    for collision in collisions:
        # 生成全机器人碰撞约束（从底盘base_link到双臂末端，覆盖所有连杆）
        # 左臂运动链碰撞约束
        c_Ain_l, c_bin_l = robot.link_collision_damper(
            collision,
            robot.q[:n],
            d_i, d_s, xi,
            start=robot.link_dict["world"],
            end=robot.link_dict[LEFT_END_LINK],
        )
        # 右臂运动链碰撞约束
        c_Ain_r, c_bin_r = robot.link_collision_damper(
            collision,
            robot.q[:n],
            d_i, d_s, xi,
            start=robot.link_dict["world"],
            end=robot.link_dict[RIGHT_END_LINK],
        )

        # 合并左右臂碰撞约束
        c_Ain_list = []
        c_bin_list = []
        if c_Ain_l is not None and c_bin_l is not None:
            # print('c_Ain_l ',c_Ain_l.shape)
            # print('c_bin_l ',c_bin_l.shape)
            # 确保碰撞约束矩阵列数为24（关节数）
            c_Ain_l = c_Ain_l[:,[2,3,4,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
            c_Ain_l = np.hstack((c_Ain_l, np.zeros((c_Ain_l.shape[0], 7), dtype=np.float64)))

            # 拼接SLACK_N列0
            c_Ain_l = np.c_[c_Ain_l, np.zeros((c_Ain_l.shape[0], SLACK_N), dtype=np.float64)]
            c_Ain_list.append(c_Ain_l)
            c_bin_list.append(c_bin_l.astype(np.float64))
        
        if c_Ain_r is not None and c_bin_r is not None:
            # print('c_Ain_r ',c_Ain_r.shape)
            # print('c_bin_r ',c_bin_r.shape)
            # 确保碰撞约束矩阵列数为24（关节数）
            c_Ain_r = c_Ain_r[:,[2,3,4,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30]]
            # 拼接SLACK_N列0
            c_Ain_r = np.c_[c_Ain_r, np.zeros((c_Ain_r.shape[0], SLACK_N), dtype=np.float64)]
            c_Ain_list.append(c_Ain_r)
            c_bin_list.append(c_bin_r.astype(np.float64))
        
        if len(c_Ain_list) > 0:
            c_Ain = np.concatenate(c_Ain_list, axis=0)
            c_bin = np.concatenate(c_bin_list, axis=0)
            # print('c_Ain ',c_Ain.shape)
            # print('c_bin ',c_bin.shape)
            
            # 堆叠到总不等式约束中
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    # ---------------------- 3.5 QP目标函数线性项c（可操作度最大化） ----------------------
    # 分别计算左右臂的可操作度雅可比（梯度）
    Jm_total_l = np.zeros((24,1), dtype=np.float64)
    Jm_left = robot.jacobm(robot.q, end=LEFT_END_LINK).astype(np.float64)
    Jm_total_l[[3,4,5,6,10,11,12,13,14,15,16],:] = Jm_left[3:]
    Jm_total_r = np.zeros((24,1), dtype=np.float64)
    Jm_right = robot.jacobm(robot.q, end=RIGHT_END_LINK).astype(np.float64)
    Jm_total_r[[17,18,19,20,21,22,23],:] = Jm_right[7:]
    # 合并双臂可操作度梯度（最大化双臂整体灵巧性）
    # print('Jm_left.shape',Jm_left.shape)
    # print('Jm_right.shape',Jm_right.shape)
    Jm_total = (Jm_total_l + Jm_total_r).astype(np.float64)

    # 线性项：QP最小化 -Jm^T * qd 等价于最大化可操作度
    c = np.r_[-Jm_total.reshape((n,)), np.zeros(SLACK_N, dtype=np.float64)]  #

    # ---------------------- 3.6 优化变量上下界 ----------------------
    # 关节速度上下限（从URDF读取，shape(n,2)）
    qd_lim = robot.qlim.astype(np.float64)
    lb_q = qd_lim[0, :]  # 所有关节速度下限（n维）
    ub_q = qd_lim[1, :]  # 所有关节速度上限（n维）
    # print('lb_q.shape',lb_q)
    # print('ub_q.shape',ub_q)
    # 松弛变量上下界（限制末端最大轨迹偏离量）
    slack_lim = 5 * np.ones(SLACK_N, dtype=np.float64)

    # 拼接为36维上下界（float64）
    lb = np.r_[lb_q, -slack_lim]
    ub = np.r_[ub_q, slack_lim]

    # ---------------------- 3.7 求解QP并执行控制 ----------------------
    # 求解严格凸二次规划（确保所有输入为float64）
    qd_sol = qp.solve_qp(
        Q, c, Ain, bin, Aeq, beq, 
        lb=lb, ub=ub, 
        solver='cvxopt'  # 需提前安装：pip install cvxopt
    )

    # 提取关节速度（前24维），赋给机器人
    # print('qd_sol', qd_sol)
    # print(robot.q)
    if qd_sol is not None:
        robot.qd[:n] = qd_sol[:n]
    else:
        # QP无解时，停止运动，避免失控
        robot.qd[:n] = np.zeros(n)
        print("警告：QP无解，机器人停止运动！")

    # 仿真步进10ms（控制频率100Hz）
    env.step(0.01)

    # 打印关键状态
    print(f"左臂误差：{err_left:.3f} | 右臂误差：{err_right:.3f} | 到达状态：{arrived}")
    
    return arrived

# ====================== 4. 主运行循环 ======================
def run():
    arrived = False
    while not arrived:
        arrived = step()
    print("="*50)
    print("双臂已全部到达目标位姿！任务完成！")
    print("="*50)

# 先执行一次初始化步进，再启动主循环
if __name__ == "__main__":
    step()
    run()