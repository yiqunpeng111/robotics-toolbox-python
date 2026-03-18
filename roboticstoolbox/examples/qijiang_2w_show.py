import roboticstoolbox as rtb
import swift
import os
import time
import numpy as np
# 1. 设置 URDF 文件路径
# 假设你的模型文件夹结构如下：
# my_robot/
# ├── my_robot.urdf
# └── meshes/
#     ├── link1.stl
#     └── ...

urdf_path = "/home/user/2w_new/urdf/2w_new.urdf" 

# 确保工作目录正确，或者使用绝对路径
# os.chdir(os.path.dirname(urdf_path)) 

# 2. 加载模型
# 'ERobot' 是处理 URDF 的核心类
robot = rtb.ERobot.URDF(urdf_path)

# 打印模型信息，检查是否加载成功
print(robot)
print("关节数量：", robot.n)

# 3. (可选) 手动指定基坐标系或工具坐标系
# robot.base = sm.SE3(0,0,0)  # 设置基坐标系偏移
# robot.tool = sm.SE3(0,0,0.1) # 设置末端工具(TCP)偏移

# 4. 启动仿真环境查看
env = swift.Swift()
env.launch()
env.add(robot)


def set_joint(j, value):
    robot.q[j] = np.deg2rad(float(value))

print(robot.qlim)
Tep = robot.fkine(robot.q,end='armr_7')

print(Tep)
# Loop through each link in the Panda and if it is a variable joint,
# add a slider to Swift to control it
j = 0
for link in robot.links:
    if link.isjoint:

        # We use a lambda as the callback function from Swift
        # j=j is used to set the value of j rather than the variable j
        # We use the HTML unicode format for the degree sign in the unit arg
        print(link.name)
        print(link.qlim)
        env.add(
            swift.Slider(
                lambda x, j=j: set_joint(j, x),
                min=np.round(np.rad2deg(link.qlim[0]), 2),
                max=np.round(np.rad2deg(link.qlim[1]), 2),
                step=1,
                value=np.round(np.rad2deg(robot.q[j]), 2),
                desc="robot Joint " + str(j),
                unit="&#176;",
            )
        )

        j += 1


# 或者手动设置
# robot.q = [0, 0.5, -0.5, ...] 
while True:
    # Process the event queue from Swift, this invokes the callback functions
    # from the sliders if the slider value was changed
    # env.process_events()

    # Update the environment with the new robot pose
    env.step(0)

    time.sleep(0.01)