"""
Contract Net Protocol for multi-drone task allocation.
Minimizes total flight distance increment when assigning new tasks, with load balancing.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class Task:
    """A task with a 3D position."""
    id: int
    position: np.ndarray          # (x, y, z)
    timestamp: float              # simulation time or step count when created
    assigned_to: int = -1         # drone ID, -1 = unassigned


class DroneAgent:
    """Represents a drone with a queue of task positions to visit."""
    def __init__(self, drone_id: int):
        self.id = drone_id
        self.task_queue: List[np.ndarray] = []      # target positions in execution order
        self.current_goal: Optional[np.ndarray] = None
        self.pos_history: List[np.ndarray] = []     # recorded positions for path length
        self.completed_tasks: int = 0               # number of tasks completed

    def total_flight_distance(self) -> float:
        """Sum of Euclidean distances between consecutive recorded positions."""
        if len(self.pos_history) < 2:
            return 0.0
        return sum(np.linalg.norm(self.pos_history[i] - self.pos_history[i-1])
                   for i in range(1, len(self.pos_history)))

    def clear_history(self):
        self.pos_history = []

    def update_nearest_goal(self, current_pos: np.ndarray):
        """Re-order the task queue so that the nearest task becomes the current goal."""
        if not self.task_queue:
            self.current_goal = None
            return
        # Find index of nearest task in queue
        distances = [np.linalg.norm(t - current_pos) for t in self.task_queue]
        nearest_idx = np.argmin(distances)
        if nearest_idx != 0:
            # Move the nearest task to front
            self.task_queue.insert(0, self.task_queue.pop(nearest_idx))
        self.current_goal = self.task_queue[0]

    def complete_current_task(self):
        """Mark current task as completed and remove from queue."""
        if self.task_queue:
            self.task_queue.pop(0)
            self.completed_tasks += 1
            if self.task_queue:
                self.current_goal = self.task_queue[0]
            else:
                self.current_goal = None


class ContractNetAllocator:
    """Centralised contract net protocol: each new task is assigned to the drone
    that yields the smallest increase in total flight path, with load balancing."""
    def __init__(self, drones: List[DroneAgent], load_balance_weight: float = 10.0):
        self.drones = drones
        self.load_balance_weight = load_balance_weight  # penalty per task in queue

    @staticmethod
    def _insertion_cost(drone: DroneAgent, new_pos: np.ndarray) -> float:
        """Estimate the additional Euclidean distance if this drone takes the new task."""
        if not drone.task_queue:
            # No tasks yet: start from current position
            start = drone.pos_history[-1] if drone.pos_history else drone.current_goal
            if start is None:
                start = np.zeros(3)
            return np.linalg.norm(new_pos - start)

        # Build sequence: current_position -> task_queue[0] -> ... -> task_queue[-1]
        seq = [drone.pos_history[-1]] + drone.task_queue
        best_delta = float('inf')
        for i in range(len(seq)):
            prev = seq[i]
            nxt = seq[i+1] if i+1 < len(seq) else None
            if nxt is None:
                delta = np.linalg.norm(new_pos - prev)
            else:
                orig = np.linalg.norm(nxt - prev)
                new = np.linalg.norm(new_pos - prev) + np.linalg.norm(nxt - new_pos)
                delta = new - orig
            best_delta = min(best_delta, delta)
        return best_delta

    def allocate_task(self, task: Task) -> int:
        """Find best drone and insert task into its queue at optimal position,
        considering both path cost and current load (queue length)."""
        # 优先分配给空闲无人机（task_queue 为空）
        idle_drones = [d for d in self.drones if len(d.task_queue) == 0]
        if idle_drones:
            # 在空闲无人机中选择距离最近的
            best_idle = min(idle_drones, key=lambda d: np.linalg.norm(d.pos_history[-1] - task.position))
            self._insert_into_queue(best_idle.id, task.position)
            task.assigned_to = best_idle.id
            return best_idle.id

        # 否则正常按路径成本+负载惩罚分配
        best_drone_id = -1
        best_cost = float('inf')
        for drone in self.drones:
            path_cost = self._insertion_cost(drone, task.position)
            load_penalty = self.load_balance_weight * len(drone.task_queue)
            total_cost = path_cost + load_penalty
            if total_cost < best_cost:
                best_cost = total_cost
                best_drone_id = drone.id

        if best_drone_id != -1:
            self._insert_into_queue(best_drone_id, task.position)
            task.assigned_to = best_drone_id
        return best_drone_id

    def _insert_into_queue(self, drone_id: int, pos: np.ndarray):
        """Insert task into drone's queue at the position that minimises path increment."""
        drone = self.drones[drone_id]
        # Build sequence of fixed points (current pos + queue)
        fixed = [drone.pos_history[-1]] + drone.task_queue
        best_idx = 0
        best_delta = float('inf')
        for i in range(len(fixed)):
            prev = fixed[i]
            nxt = fixed[i+1] if i+1 < len(fixed) else None
            if nxt is None:
                delta = np.linalg.norm(pos - prev)
            else:
                orig = np.linalg.norm(nxt - prev)
                new = np.linalg.norm(pos - prev) + np.linalg.norm(nxt - pos)
                delta = new - orig
            if delta < best_delta:
                best_delta = delta
                best_idx = i
        # Insert into task_queue at position best_idx
        drone.task_queue.insert(best_idx, pos)
        # Note: current_goal will be updated later via update_nearest_goal


class TaskGenerator:
    """Spawn new tasks at fixed time intervals (in environment steps), avoiding obstacles."""
    def __init__(self, room_dims: Tuple[float, float, float],
                 total_tasks: int, interval_steps: int,
                 obstacle_radius: float = 0.6, max_attempts: int = 100):
        self.room_dims = room_dims      # (length, width, height)
        self.total_tasks = total_tasks
        self.interval_steps = interval_steps
        self.obstacle_radius = obstacle_radius
        self.max_attempts = max_attempts
        self.next_task_id = 0
        self.steps_since_last = 0

    def update(self, steps_elapsed: int, obstacles_pos: Optional[List[np.ndarray]] = None) -> List[Task]:
        """Return list of newly generated tasks, avoiding obstacles if positions provided."""
        self.steps_since_last += steps_elapsed
        new_tasks = []
        while self.steps_since_last >= self.interval_steps and self.next_task_id < self.total_tasks:
            self.steps_since_last -= self.interval_steps
            # Generate a valid position (not inside any obstacle)
            pos = self._generate_valid_position(obstacles_pos)
            task = Task(id=self.next_task_id,
                        position=pos,
                        timestamp=self.next_task_id)
            new_tasks.append(task)
            self.next_task_id += 1
        return new_tasks

    def _generate_valid_position(self, obstacles_pos: Optional[List[np.ndarray]]) -> np.ndarray:
        """Generate a random position inside the room, avoiding obstacles."""
        for _ in range(self.max_attempts):
            x = np.random.uniform(-self.room_dims[0]/2, self.room_dims[0]/2)
            y = np.random.uniform(-self.room_dims[1]/2, self.room_dims[1]/2)
            z = np.random.uniform(0.5, self.room_dims[2])   # above ground
            pos = np.array([x, y, z], dtype=np.float32)
            if obstacles_pos is None:
                return pos
            # Check collision with any obstacle (cylinder approximation)
            safe = True
            for obs_pos in obstacles_pos:
                dx = pos[0] - obs_pos[0]
                dy = pos[1] - obs_pos[1]
                dist_xy = np.sqrt(dx*dx + dy*dy)
                # Assume obstacles are cylinders of radius self.obstacle_radius/2?
                # Actually obstacle_size is diameter, so radius = obstacle_size/2
                # We'll use a safety margin of 0.2m
                if dist_xy < (self.obstacle_radius + 0.2):
                    safe = False
                    break
            if safe:
                return pos
        # Fallback: return center of room
        print(f"Warning: Could not generate obstacle-free task after {self.max_attempts} attempts, using room center.")
        return np.array([0.0, 0.0, self.room_dims[2]/2], dtype=np.float32)

    def remaining_tasks(self) -> int:
        return self.total_tasks - self.next_task_id
