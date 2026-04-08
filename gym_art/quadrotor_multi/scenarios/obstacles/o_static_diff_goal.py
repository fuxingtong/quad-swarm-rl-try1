import numpy as np
import copy
from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_static_diff_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # 无编队：每个无人机目标独立，无需编队中心/层距
        self.layer_dist = 0.0
        self.formation_size = -1.0  # 禁用编队大小更新

    def step(self):
        # 静态目标：step中不更新目标（目标全程固定）
        return

    def reset(self, obst_map=None, cell_centers=None):
        # 初始化障碍物地图（必须）
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError("必须传入障碍物地图和单元格中心")

        # 1. 计算自由空间（无障碍物区域）
        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # 2. 生成无人机起始位置（无障碍物区域）
        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.spawn_points = copy.deepcopy(self.start_point)

        # 3. 生成不同的静态目标（每个无人机一个独立目标，均在自由空间）
        self.goals = []
        for _ in range(self.num_agents):
            # 随机选自由空间中的位置作为目标（Z轴保证离地）
            rand_idx = np.random.choice(len(self.free_space))
            x, y = self.cell_centers[self.free_space[rand_idx]]
            z = np.random.uniform(low=1.0, high=3.0)  # 目标高度1~3m
            self.goals.append([x, y, z])
        self.goals = np.array(self.goals)

        # 4. 给每个环境（无人机）分配独立目标
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
