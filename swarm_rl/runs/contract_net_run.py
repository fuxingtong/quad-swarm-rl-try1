#!/usr/bin/env python
import sys
import time
import argparse
import numpy as np
import torch
from gymnasium import spaces
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.env_wrappers.task_allocation import DroneAgent, ContractNetAllocator, TaskGenerator, Task
from swarm_rl.env_wrappers.cbba_allocator import CBBADroneAgent, CBBAAllocator
from swarm_rl.train import register_swarm_components


def parse_contract_net_cfg(argv=None):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=True)
    add_quadrotors_env_args(partial_cfg.env, parser)
    quadrotors_override_defaults(partial_cfg.env, parser)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--total_tasks', type=int, default=30,
                        help='Total number of tasks to generate')
    parser.add_argument('--task_interval_sec', type=float, default=6.0,
                        help='Seconds between task generation')
    parser.add_argument('--arrival_threshold', type=float, default=1.0,
                        help='Distance threshold to consider task completed')
    parser.add_argument('--load_balance_weight', type=float, default=5.0,
                        help='Penalty weight for each task in queue for load balancing (ContractNet only)')
    parser.add_argument('--allocator', type=str, default='cbba', choices=['contract_net', 'cbba'],
                        help='Task allocation algorithm to use')
    cfg = parse_full_cfg(parser, argv)

    # 强制与训练时一致的配置
    cfg.quads_mode = "o_random"
    cfg.quads_render = True
    cfg.quads_use_obstacles = True
    cfg.quads_obst_density = 0.2
    cfg.quads_obst_size = 0.6
    cfg.quads_obst_spawn_area = [8.0, 8.0]
    cfg.quads_obs_repr = "xyz_vxyz_R_omega_floor"
    cfg.quads_neighbor_visible_num = 2
    cfg.quads_neighbor_obs_type = "pos_vel"
    cfg.quads_use_downwash = True
    cfg.quads_obstacle_obs_type = "octomap"
    cfg.quads_num_agents = 6
    cfg.env_gpu_observations = False
    cfg.device = "cpu"
    cfg.use_rnn = True
    return cfg


def load_policy(cfg, env, checkpoint_path):
    obs_space = env.observation_space
    if not isinstance(obs_space, spaces.Dict):
        obs_space = spaces.Dict({"obs": obs_space})
    actor_critic = create_actor_critic(cfg, obs_space, env.action_space)
    actor_critic.eval()
    device = torch.device(cfg.device)
    actor_critic.model_to_device(device)
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    actor_critic.load_state_dict(checkpoint_dict["model"])
    print(f"Loaded policy from {checkpoint_path}")
    return actor_critic, device


def main():
    register_swarm_components()
    cfg = parse_contract_net_cfg()

    env = make_quadrotor_env('quadrotor_multi', cfg, render_mode='human')
    num_agents = env.num_agents

    model, device = load_policy(cfg, env, cfg.checkpoint)

    # 根据分配器类型初始化不同的DroneAgent类
    if cfg.allocator == 'contract_net':
        drones = [DroneAgent(i) for i in range(num_agents)]
        print("Using ContractNet allocator")
    else:  # cbba
        drones = [CBBADroneAgent(i, pos_history=[]) for i in range(num_agents)]
        print("Using CBBA allocator")

    obs, _ = env.reset()

    # 禁用场景自动目标更新（防止 o_random 覆盖手动设置的目标）
    if hasattr(env.unwrapped, 'scenario') and hasattr(env.unwrapped.scenario, 'step'):
        env.unwrapped.scenario.step = lambda: None
        print("Disabled scenario auto-goal updates.")

    obs_dim = obs.shape[1]
    print(f"Observation dimension: {obs_dim}")
    for i, drone in enumerate(drones):
        drone.pos_history.append(env.envs[i].dynamics.pos.copy())
        drone.current_goal = None

    # 获取障碍物位置（用于生成无碰撞任务点）
    obstacles_pos = None
    if hasattr(env.unwrapped, 'obstacles') and env.unwrapped.obstacles is not None:
        obstacles_pos = env.unwrapped.obstacles.pos_arr  # list of [x,y,z]
        print(f"Obstacles count: {len(obstacles_pos)}")

    # 初始化分配器
    if cfg.allocator == 'contract_net':
        allocator = ContractNetAllocator(drones, load_balance_weight=cfg.load_balance_weight)
    else:
        allocator = CBBAAllocator(drones)

    task_gen = TaskGenerator(room_dims=cfg.quads_room_dims,
                             total_tasks=cfg.total_tasks,
                             interval_steps=int(cfg.task_interval_sec * env.envs[0].control_freq),
                             obstacle_radius=cfg.quads_obst_size / 2.0)

    total_steps = 0
    max_steps = 20000
    episode_done = False
    rnn_states = torch.zeros((num_agents, cfg.rnn_size), device=device) if cfg.use_rnn else None

    last_print_step = 0
    print_interval = 500

    while not episode_done and total_steps < max_steps:
        # 生成新任务（传递障碍物位置）
        new_tasks = task_gen.update(1, obstacles_pos=obstacles_pos)
        for task in new_tasks:
            winner_id = allocator.allocate_task(task)
            if total_steps - last_print_step >= print_interval or task.id % 5 == 0:
                print(f"[Step {total_steps}] Task {task.id} at {task.position.round(2)} -> drone {winner_id}")
                last_print_step = total_steps

        # 动态重规划：每个无人机选择队列中最近的任务作为当前目标
        for i, drone in enumerate(drones):
            current_pos = env.envs[i].dynamics.pos.copy()
            drone.update_nearest_goal(current_pos)
            if drone.current_goal is not None:
                env.envs[i].goal = drone.current_goal.copy()
            # 没有任务时，不改变目标

        # 获取动作
        obs_dict = {"obs": obs}
        normalized_obs = prepare_and_normalize_obs(model, obs_dict)
        with torch.no_grad():
            result = model.forward(normalized_obs, rnn_states, values_only=False)
        actions = result["actions"].cpu().numpy()
        if cfg.use_rnn:
            rnn_states = result["new_rnn_states"]

        # 执行环境步
        step_result = env.step(actions)
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, infos = step_result
            dones = np.logical_or(terminated, truncated)
        else:
            obs, rewards, dones, infos = step_result

        # 更新轨迹并检查任务完成
        for i, drone in enumerate(drones):
            new_pos = env.envs[i].dynamics.pos.copy()
            drone.pos_history.append(new_pos)
            if drone.current_goal is not None:
                dist = np.linalg.norm(new_pos - drone.current_goal)
                if dist < cfg.arrival_threshold:
                    drone.complete_current_task()
                    print(f"[Step {total_steps}] Drone {i} completed a task! Total completed: {drone.completed_tasks}")

        total_steps += 1

        # 终止条件
        if task_gen.remaining_tasks() == 0 and all(len(d.task_queue) == 0 for d in drones):
            episode_done = True
            print("All tasks completed!")

        if cfg.quads_render:
            time.sleep(0.01)

    # 最终统计
    total_distance = sum(drone.total_flight_distance() for drone in drones)
    total_completed = sum(drone.completed_tasks for drone in drones)
    print("\n" + "=" * 50)
    print(f"Total flight distance: {total_distance:.2f} m")
    print(f"Total tasks completed: {total_completed} / {cfg.total_tasks}")
    for i, drone in enumerate(drones):
        print(f"Drone {i}: distance {drone.total_flight_distance():.2f} m, completed {drone.completed_tasks} tasks")
    print("=" * 50)
    env.close()


if __name__ == "__main__":
    sys.exit(main())
