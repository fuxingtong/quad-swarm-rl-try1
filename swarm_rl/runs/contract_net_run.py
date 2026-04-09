#!/usr/bin/env python
import sys
import time
import argparse
import numpy as np
import torch
import random
from gymnasium import spaces
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.train import register_swarm_components

# ---------- Task ----------
class Task:
    def __init__(self, task_id, position, task_type, deadline, duration, timestamp):
        self.id = task_id
        self.position = position
        self.task_type = task_type
        self.deadline = deadline
        self.duration = duration
        self.timestamp = timestamp
        self.assigned_to = -1

# ---------- DroneAgent ----------
class DroneAgent:
    def __init__(self, drone_id, drone_type, max_range, speed, capacity):
        self.id = drone_id
        self.drone_type = drone_type
        self.max_range = max_range
        self.speed = speed
        self.capacity = capacity
        self.remaining_range = max_range
        self.flown_distance = 0.0
        self.task_queue = []
        self.current_goal = None
        self.current_task_id = None
        self.pos_history = []
        self.completed_tasks = 0

    def can_execute(self, task_type):
        if self.drone_type == 'attack':
            return task_type in ['attack', 'recon']
        elif self.drone_type == 'recon':
            return task_type in ['recon', 'transport']
        else:
            return self.drone_type == task_type

    def total_flight_distance(self):
        if len(self.pos_history) < 2:
            return 0.0
        return sum(np.linalg.norm(self.pos_history[i] - self.pos_history[i-1])
                   for i in range(1, len(self.pos_history)))

    def update_flown_distance(self, new_pos):
        if self.pos_history:
            self.flown_distance += np.linalg.norm(new_pos - self.pos_history[-1])
            self.remaining_range = max(0.0, self.max_range - self.flown_distance)
        self.pos_history.append(new_pos)

    def queue_total_distance_from_current(self):
        if not self.task_queue:
            return 0.0
        current_pos = self.pos_history[-1] if self.pos_history else np.zeros(3)
        total = 0.0
        prev = current_pos
        for pos, _, _, _ in self.task_queue:
            total += np.linalg.norm(pos - prev)
            prev = pos
        return total

    def queue_total_time_from_current(self):
        if not self.task_queue:
            return 0.0
        current_pos = self.pos_history[-1] if self.pos_history else np.zeros(3)
        total = 0.0
        prev = current_pos
        for pos, _, _, dur in self.task_queue:
            dist = np.linalg.norm(pos - prev)
            total += dist / self.speed + dur
            prev = pos
        return total

    def total_distance_with_new_task(self, new_pos):
        if not self.task_queue:
            current_pos = self.pos_history[-1] if self.pos_history else np.zeros(3)
            return np.linalg.norm(new_pos - current_pos)
        current_pos = self.pos_history[-1] if self.pos_history else np.zeros(3)
        seq = [current_pos] + [pos for pos, _, _, _ in self.task_queue]
        best_total = float('inf')
        for i in range(len(seq)):
            if i == len(seq) - 1:
                path = seq + [new_pos]
            else:
                path = seq[:i+1] + [new_pos] + seq[i+1:]
            total = sum(np.linalg.norm(path[j] - path[j-1]) for j in range(1, len(path)))
            if total < best_total:
                best_total = total
        return best_total

    def total_time_with_new_task(self, new_pos, task_duration):
        if not self.task_queue:
            current_pos = self.pos_history[-1] if self.pos_history else np.zeros(3)
            return np.linalg.norm(new_pos - current_pos) / self.speed + task_duration
        current_pos = self.pos_history[-1] if self.pos_history else np.zeros(3)
        seq = [current_pos] + [pos for pos, _, _, _ in self.task_queue]
        best_time = float('inf')
        for i in range(len(seq)):
            if i == len(seq) - 1:
                points = seq + [new_pos]
            else:
                points = seq[:i+1] + [new_pos] + seq[i+1:]
            total_time = 0.0
            prev = points[0]
            for j in range(1, len(points)):
                dist = np.linalg.norm(points[j] - prev)
                if points[j] is new_pos:
                    total_time += dist / self.speed + task_duration
                else:
                    dur = 0.0
                    for pos, _, _, d in self.task_queue:
                        if np.array_equal(pos, points[j]):
                            dur = d
                            break
                    total_time += dist / self.speed + dur
                prev = points[j]
            if total_time < best_time:
                best_time = total_time
        return best_time

    def update_nearest_goal(self, current_pos):
        if not self.task_queue:
            self.current_goal = None
            self.current_task_id = None
            return
        distances = [np.linalg.norm(pos - current_pos) for pos, _, _, _ in self.task_queue]
        nearest_idx = np.argmin(distances)
        if nearest_idx != 0:
            self.task_queue.insert(0, self.task_queue.pop(nearest_idx))
        self.current_goal = self.task_queue[0][0]
        self.current_task_id = self.task_queue[0][1]

    def complete_current_task(self):
        if self.task_queue:
            _, task_id, _, _ = self.task_queue.pop(0)
            self.completed_tasks += 1
            if self.task_queue:
                self.current_goal = self.task_queue[0][0]
                self.current_task_id = self.task_queue[0][1]
            else:
                self.current_goal = None
                self.current_task_id = None

# ---------- 任务生成器（可重复） ----------
class AdvancedTaskGenerator:
    def __init__(self, room_dims, total_tasks, interval_steps, obstacle_radius,
                 task_type_probs=None, deadline_offset_range=(10, 60), duration_range=(0, 2),
                 max_attempts=100):
        self.room_dims = room_dims
        self.total_tasks = total_tasks
        self.interval_steps = interval_steps
        self.obstacle_radius = obstacle_radius
        self.max_attempts = max_attempts
        self.next_task_id = 0
        self.steps_since_last = 0
        # 直接使用概率字典，包含三类任务
        if task_type_probs is None:
            self.task_type_probs = {'recon': 0.5, 'attack': 0.2, 'transport': 0.3}
        else:
            self.task_type_probs = task_type_probs
        self.types = list(self.task_type_probs.keys())
        self.probs = list(self.task_type_probs.values())
        self.deadline_offset_range = deadline_offset_range
        self.duration_range = duration_range

    def _generate_valid_position(self, obstacles_pos):
        for _ in range(self.max_attempts):
            x = np.random.uniform(-self.room_dims[0]/2, self.room_dims[0]/2)
            y = np.random.uniform(-self.room_dims[1]/2, self.room_dims[1]/2)
            z = np.random.uniform(0.5, self.room_dims[2])
            pos = np.array([x, y, z], dtype=np.float32)
            if obstacles_pos is None:
                return pos
            safe = True
            for obs_pos in obstacles_pos:
                dx = pos[0] - obs_pos[0]
                dy = pos[1] - obs_pos[1]
                dist_xy = np.sqrt(dx*dx + dy*dy)
                if dist_xy < (self.obstacle_radius + 0.2):
                    safe = False
                    break
            if safe:
                return pos
        print(f"Warning: Could not generate obstacle-free task after {self.max_attempts} attempts, using room center.")
        return np.array([0.0, 0.0, self.room_dims[2]/2], dtype=np.float32)

    def update(self, steps_elapsed, current_time_sec, obstacles_pos=None):
        self.steps_since_last += steps_elapsed
        new_tasks = []
        while self.steps_since_last >= self.interval_steps and self.next_task_id < self.total_tasks:
            self.steps_since_last -= self.interval_steps
            pos = self._generate_valid_position(obstacles_pos)
            task_type = np.random.choice(self.types, p=self.probs)
            duration = np.random.uniform(*self.duration_range)
            deadline_offset = np.random.uniform(*self.deadline_offset_range)
            deadline = current_time_sec + deadline_offset
            task = Task(
                task_id=self.next_task_id,
                position=pos,
                task_type=task_type,
                deadline=deadline,
                duration=duration,
                timestamp=current_time_sec
            )
            new_tasks.append(task)
            self.next_task_id += 1
        return new_tasks

    def remaining_tasks(self):
        return self.total_tasks - self.next_task_id

# ---------- 分配器 ----------
class ValueBasedContractNetAllocator:
    def __init__(self, drones, model, env, device,
                 w1=1.0, w2=0.5, w3=10.0):
        self.drones = drones
        self.model = model
        self.device = device
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.multi_env = env.unwrapped

    def _get_state_value(self, drone_idx, goal_position):
        multi_env = self.multi_env
        original_goal = multi_env.envs[drone_idx].goal.copy() if multi_env.envs[drone_idx].goal is not None else None
        if goal_position is None:
            goal_position = multi_env.envs[drone_idx].dynamics.pos.copy()
        multi_env.envs[drone_idx].goal = goal_position.copy()

        obs_self = [e.state_vector(e) for e in multi_env.envs]
        if multi_env.num_use_neighbor_obs > 0:
            obs_full = multi_env.add_neighborhood_obs(obs_self)
        else:
            obs_full = np.stack(obs_self)
        if multi_env.use_obstacles:
            obs_full = multi_env.obstacles.step(obs=obs_full, quads_pos=multi_env.pos)
        obs_i = obs_full[drone_idx:drone_idx+1]

        if original_goal is not None:
            multi_env.envs[drone_idx].goal = original_goal
        else:
            multi_env.envs[drone_idx].goal = None

        obs_tensor = torch.from_numpy(obs_i).float().to(self.device)
        obs_dict = {"obs": obs_tensor}
        normalized_obs = prepare_and_normalize_obs(self.model, obs_dict)
        with torch.no_grad():
            result = self.model.forward(normalized_obs, rnn_states=None, values_only=True)
        return result["values"].cpu().item()

    def _compute_cost_value_diff(self, drone_idx, task_pos):
        try:
            drone = self.drones[drone_idx]
            v_current = self._get_state_value(drone_idx, drone.current_goal)
            v_target = self._get_state_value(drone_idx, task_pos)
            diff = v_current - v_target
            return max(0.0, diff)
        except Exception as e:
            print(f"Value network error for drone {drone_idx}: {e}, using large cost")
            return 1e6

    def _check_constraints(self, drone, task, current_time_sec):
        if not drone.can_execute(task.task_type):
            return False, "type_mismatch"
        if len(drone.task_queue) >= drone.capacity:
            return False, "capacity_exceeded"
        if drone.remaining_range <= 0:
            return False, "range_exhausted"
        required_distance = drone.total_distance_with_new_task(task.position)
        if drone.remaining_range < required_distance:
            return False, "range_exceeded"
        required_time = drone.total_time_with_new_task(task.position, task.duration)
        if current_time_sec + required_time > task.deadline:
            return False, "deadline_missed"
        return True, "ok"

    def allocate_task(self, task, current_time_sec):
        best_drone_id = -1
        best_bid = float('inf')
        for drone in self.drones:
            feasible, _ = self._check_constraints(drone, task, current_time_sec)
            if not feasible:
                continue
            cost = self._compute_cost_value_diff(drone.id, task.position)
            load_penalty = len(drone.task_queue)
            bid = self.w1 * cost + self.w2 * load_penalty
            if bid < best_bid:
                best_bid = bid
                best_drone_id = drone.id
        if best_drone_id != -1:
            self._insert_into_queue(best_drone_id, task.position, task.id, task.duration)
            task.assigned_to = best_drone_id
        return best_drone_id

    def _insert_into_queue(self, drone_id, task_pos, task_id, duration):
        drone = self.drones[drone_id]
        current_pos = drone.pos_history[-1] if drone.pos_history else np.zeros(3)
        seq = [current_pos] + [pos for pos, _, _, _ in drone.task_queue]
        best_idx = 0
        best_delta = float('inf')
        for i in range(len(seq)):
            if i == len(seq) - 1:
                new_seq = seq + [task_pos]
            else:
                new_seq = seq[:i+1] + [task_pos] + seq[i+1:]
            total = sum(np.linalg.norm(new_seq[j] - new_seq[j-1]) for j in range(1, len(new_seq)))
            delta = total - (drone.queue_total_distance_from_current() + np.linalg.norm(task_pos - current_pos))
            if delta < best_delta:
                best_delta = delta
                best_idx = i
        if best_idx == len(seq) - 1:
            prev_point = seq[-1]
            dist_to_prev = np.linalg.norm(task_pos - prev_point)
            drone.task_queue.append((task_pos, task_id, dist_to_prev, duration))
        else:
            prev_point = seq[best_idx]
            dist_to_prev = np.linalg.norm(task_pos - prev_point)
            drone.task_queue.insert(best_idx, (task_pos, task_id, dist_to_prev, duration))
        if len(drone.task_queue) == 1:
            drone.current_goal = task_pos
            drone.current_task_id = task_id

# ---------- 配置解析 ----------
def parse_contract_net_cfg(argv=None):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=True)
    add_quadrotors_env_args(partial_cfg.env, parser)
    quadrotors_override_defaults(partial_cfg.env, parser)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--total_tasks', type=int, default=30)
    parser.add_argument('--task_interval_sec', type=float, default=8.0)
    parser.add_argument('--arrival_threshold', type=float, default=0.5)
    parser.add_argument('--w1', type=float, default=0.8)
    parser.add_argument('--w2', type=float, default=0.2)
    parser.add_argument('--w3', type=float, default=0.0)
    # 无人机参数：两类，顺序 [攻击, 侦察]
    parser.add_argument('--drone_types', nargs='+', default=['attack', 'recon'])
    parser.add_argument('--drone_max_ranges', nargs='+', type=float, default=[2000.0, 2500.0])
    parser.add_argument('--drone_speeds', nargs='+', type=float, default=[6.0, 5.0])
    parser.add_argument('--drone_capacities', nargs='+', type=int, default=[4, 4])
    # 任务概率：三类任务顺序 [recon, attack, transport]
    parser.add_argument('--task_type_probs', nargs='+', type=float, default=[0.5, 0.2, 0.3])
    parser.add_argument('--deadline_offset_min', type=float, default=10.0)
    parser.add_argument('--deadline_offset_max', type=float, default=60.0)
    parser.add_argument('--task_duration_min', type=float, default=0.0)
    parser.add_argument('--task_duration_max', type=float, default=2.0)

    cfg = parse_full_cfg(parser, argv)

    cfg.quads_mode = "o_static_diff_goal"
    cfg.quads_use_obstacles = True
    cfg.quads_obst_density = 0.2
    cfg.quads_obst_size = 0.6
    cfg.quads_obst_spawn_area = [8.0, 8.0]
    cfg.quads_obs_repr = "xyz_vxyz_R_omega_floor"
    cfg.quads_neighbor_visible_num = 5
    cfg.quads_neighbor_obs_type = "pos_vel"
    cfg.quads_neighbor_encoder_type = "attention"
    cfg.quads_use_downwash = True
    cfg.quads_obstacle_obs_type = "octomap"
    cfg.quads_num_agents = 6
    cfg.env_gpu_observations = False
    cfg.device = "cpu"
    cfg.use_rnn = False
    cfg.actor_critic_share_weights = False
    cfg.nonlinearity = "tanh"
    cfg.policy_initialization = "xavier_uniform"
    cfg.adaptive_stddev = False
    cfg.normalize_input = False
    cfg.normalize_returns = False
    cfg.rnn_size = 256
    return cfg

def load_policy(cfg, env, checkpoint_path):
    obs_space = env.observation_space
    if not isinstance(obs_space, spaces.Dict):
        obs_space = spaces.Dict({"obs": obs_space})
    actor_critic = create_actor_critic(cfg, obs_space, env.action_space)
    actor_critic.eval()
    device = torch.device(cfg.device)
    actor_critic.model_to_device(device)
    checkpoint_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_critic.load_state_dict(checkpoint_dict["model"], strict=True)
    print(f"Loaded policy from {checkpoint_path}")
    return actor_critic, device

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_single_experiment(exp_id, cfg, model, device):
    """运行单个实验，返回 (completed_tasks, total_distance, load_list)"""
    base_seed = 42
    exp_seed = base_seed + exp_id
    set_global_seed(exp_seed)

    env = make_quadrotor_env('quadrotor_multi', cfg, render_mode='human')

    # 固定无人机类型分配：3架攻击，3架侦察
    fixed_types = ['attack', 'attack', 'attack', 'recon', 'recon', 'recon']
    drone_types_list = cfg.drone_types
    max_ranges_dict = {t: r for t, r in zip(drone_types_list, cfg.drone_max_ranges)}
    speeds_dict = {t: s for t, s in zip(drone_types_list, cfg.drone_speeds)}
    capacities_dict = {t: c for t, c in zip(drone_types_list, cfg.drone_capacities)}

    drones = []
    for i, d_type in enumerate(fixed_types):
        drones.append(DroneAgent(
            drone_id=i,
            drone_type=d_type,
            max_range=max_ranges_dict[d_type],
            speed=speeds_dict[d_type],
            capacity=capacities_dict[d_type]
        ))

    obs, _ = env.reset()

    if hasattr(env.unwrapped, 'scenario') and hasattr(env.unwrapped.scenario, 'step'):
        env.unwrapped.scenario.step = lambda: None

    for i, drone in enumerate(drones):
        current_pos = env.unwrapped.envs[i].dynamics.pos.copy()
        drone.pos_history.append(current_pos)
        drone.update_flown_distance(current_pos)
        drone.current_goal = None
        env.unwrapped.envs[i].goal = current_pos.copy()

    obstacles_pos = None
    if hasattr(env.unwrapped, 'obstacles') and env.unwrapped.obstacles is not None:
        obstacles_pos = env.unwrapped.obstacles.pos_arr

    allocator = ValueBasedContractNetAllocator(
        drones, model, env, device,
        w1=cfg.w1, w2=cfg.w2, w3=cfg.w3
    )

    # 构造任务概率字典（三类任务）
    task_type_probs_dict = {
        'recon': cfg.task_type_probs[0],
        'attack': cfg.task_type_probs[1],
        'transport': cfg.task_type_probs[2]
    }
    task_gen = AdvancedTaskGenerator(
        room_dims=cfg.quads_room_dims,
        total_tasks=cfg.total_tasks,
        interval_steps=int(cfg.task_interval_sec * env.unwrapped.envs[0].control_freq),
        obstacle_radius=cfg.quads_obst_size / 2.0,
        task_type_probs=task_type_probs_dict,
        deadline_offset_range=(cfg.deadline_offset_min, cfg.deadline_offset_max),
        duration_range=(cfg.task_duration_min, cfg.task_duration_max)
    )

    total_steps = 0
    max_steps = 40000
    episode_done = False
    current_time_sec = 0.0

    while not episode_done and total_steps < max_steps:
        dt = 1.0 / env.unwrapped.envs[0].control_freq
        current_time_sec += dt

        new_tasks = task_gen.update(steps_elapsed=1, current_time_sec=current_time_sec, obstacles_pos=obstacles_pos)
        for task in new_tasks:
            allocator.allocate_task(task, current_time_sec)

        for i, drone in enumerate(drones):
            current_pos = env.unwrapped.envs[i].dynamics.pos.copy()
            drone.update_nearest_goal(current_pos)
            if drone.current_goal is not None:
                env.unwrapped.envs[i].goal = drone.current_goal.copy()
            else:
                env.unwrapped.envs[i].goal = current_pos.copy()

        obs_dict = {"obs": obs}
        normalized_obs = prepare_and_normalize_obs(model, obs_dict)
        with torch.no_grad():
            result = model.forward(normalized_obs, rnn_states=None, values_only=False)
        actions = result["actions"].cpu().numpy()

        step_result = env.step(actions)
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, infos = step_result
        else:
            obs, rewards, dones, infos = step_result

        for i, drone in enumerate(drones):
            new_pos = env.unwrapped.envs[i].dynamics.pos.copy()
            drone.update_flown_distance(new_pos)
            if drone.current_goal is not None:
                dist = np.linalg.norm(new_pos - drone.current_goal)
                if dist < cfg.arrival_threshold:
                    drone.complete_current_task()

        total_steps += 1

        if task_gen.remaining_tasks() == 0 and all(len(d.task_queue) == 0 for d in drones):
            episode_done = True

    total_distance = sum(d.total_flight_distance() for d in drones)
    total_completed = sum(d.completed_tasks for d in drones)
    load_list = [d.completed_tasks for d in drones]

    env.close()
    return total_completed, total_distance, load_list

def main():
    register_swarm_components()
    cfg = parse_contract_net_cfg()

    cfg.quads_render = False
    print("Rendering disabled for batch experiments")

    temp_env = make_quadrotor_env('quadrotor_multi', cfg, render_mode='human')
    model, device = load_policy(cfg, temp_env, cfg.checkpoint)
    temp_env.close()

    num_experiments = 20
    results_completed = []
    results_distance = []
    results_load_lists = []

    print(f"\n{'='*60}")
    print(f"Starting {num_experiments} experiments (experiment_id 0 to {num_experiments-1})")
    print(f"{'='*60}\n")

    for exp_id in range(num_experiments):
        print(f"Running experiment {exp_id+1}/{num_experiments} (seed={42+exp_id})...")
        completed, distance, load_list = run_single_experiment(exp_id, cfg, model, device)
        results_completed.append(completed)
        results_distance.append(distance)
        results_load_lists.append(load_list)
        print(f"  -> Completed: {completed}/{cfg.total_tasks}, Distance: {distance:.2f} m, Loads: {load_list}")

    completed_array = np.array(results_completed)
    distance_array = np.array(results_distance)
    load_array = np.array(results_load_lists)

    mean_completed = np.mean(completed_array)
    std_completed = np.std(completed_array)
    completion_rate_mean = mean_completed / cfg.total_tasks
    completion_rate_std = std_completed / cfg.total_tasks

    mean_distance = np.mean(distance_array)
    std_distance = np.std(distance_array)

    per_exp_load_std = np.std(load_array, axis=1)
    mean_load_std = np.mean(per_exp_load_std)
    std_load_std = np.std(per_exp_load_std)

    per_drone_mean = np.mean(load_array, axis=0)
    per_drone_std = np.std(load_array, axis=0)

    print("\n" + "="*60)
    print("FINAL RESULTS OVER 20 EXPERIMENTS")
    print("="*60)
    print(f"Total tasks per experiment: {cfg.total_tasks}")
    print(f"\nTask Completion:")
    print(f"  Mean completed tasks: {mean_completed:.2f} ± {std_completed:.2f}")
    print(f"  Mean completion rate: {completion_rate_mean:.3f} ± {completion_rate_std:.3f}")
    print(f"\nTotal Flight Distance (m):")
    print(f"  Mean: {mean_distance:.2f} ± {std_distance:.2f}")
    print(f"\nLoad Balancing (std of completed tasks per drone within an experiment):")
    print(f"  Mean intra-experiment std: {mean_load_std:.3f} ± {std_load_std:.3f}")
    print(f"\nPer-driver average completed tasks:")
    for i in range(6):
        drone_type = 'attack' if i < 3 else 'recon'
        print(f"  Drone {i} ({drone_type}): {per_drone_mean[i]:.2f} ± {per_drone_std[i]:.2f}")
    print("="*60)

if __name__ == "__main__":
    sys.exit(main())
