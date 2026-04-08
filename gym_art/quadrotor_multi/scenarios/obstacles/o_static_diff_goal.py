import numpy as np
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_o_static_diff_goal(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.free_space = None
        self.cell_centers = None
        self.obst_map = None

        # 手动设置一些必要的默认值，绕过父类的编队逻辑
        self.formation = "circle"  # 占位值
        self.num_agents_per_layer = 8
        self.formation_size = 1.0
        self.layer_dist = 1.0
        self.formation_center = np.array([0.0, 0.0, 2.0])

    def reset(self, obst_map=None, cell_centers=None):
        # 保存障碍物地图和单元格中心信息
        self.obst_map = obst_map
        self.cell_centers = cell_centers

        # 确保有障碍物信息
        if obst_map is None or cell_centers is None:
            # 如果没有障碍物信息，回退到父类的简单逻辑
            self.formation_center = np.array([0.0, 0.0, 2.0])
            self.goals = self.generate_goals(
                num_agents=self.num_agents,
                formation_center=self.formation_center,
                layer_dist=self.layer_dist
            )
            np.random.shuffle(self.goals)
            return

        # --- 障碍物场景的核心逻辑 ---
        # 获取自由空间索引
        self.free_space = np.where(obst_map.flatten() == 0)[0]

        if len(self.free_space) == 0:
            raise RuntimeError("No free space available in the obstacle map!")

        # 为每个智能体生成不同的目标点
        self.goals = []
        for i in range(self.num_agents):
            # 随机选择一个自由格子
            rand_idx = np.random.choice(len(self.free_space))
            cell_idx = self.free_space[rand_idx]

            # 安全地解包x, y坐标
            if self.cell_centers.ndim == 1:
                # 一维情况：假设格式为 [x1, y1, x2, y2, ...]
                x = self.cell_centers[cell_idx * 2]
                y = self.cell_centers[cell_idx * 2 + 1]
            else:
                # 二维情况：(N, 2) 格式
                cell_pos = self.cell_centers[cell_idx]
                x, y = cell_pos[0], cell_pos[1]

            # 生成3D目标点（z轴设为2.0）
            goal = np.array([x, y, 2.0])
            self.goals.append(goal)

        # 转换为numpy数组并打乱
        self.goals = np.array(self.goals)
        np.random.shuffle(self.goals)

    def step(self):
        """
        静态场景不需要在step中做特殊处理
        """
        pass
