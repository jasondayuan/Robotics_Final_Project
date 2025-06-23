import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from map_generation import MazeDataset
from corner_detector import CornerDetector

class RobotConfig:
    def __init__(self):
        # Robot physical parameters
        self.robot_radius = 0.089
        self.wheel_radius = 0.033
        self.wheel_base = 0.16
        self.ang_vel_limit = 2.84
        
        # LIDAR parameters
        self.num_scans = 180
        self.max_range = 3.5
        self.fov = np.pi * 2 
        
        # Simulation parameters
        self.dt = 0.1
        
        # Noise parameters (standard deviations)
        self.motion_pos_std = 0.01  
        self.motion_ang_std = np.deg2rad(0.5)  
        self.obs_noise_std = 0.02 
        
    @property
    def motion_noise_cov(self):
        return np.diag([self.motion_pos_std**2, self.motion_pos_std**2, self.motion_ang_std**2])
    
    @property
    def obs_noise_cov(self):
        return self.obs_noise_std**2

class Env:
    def __init__(self, maze, config):

        self.config = config

        self.maze = maze
        self.maze_size = np.array(maze['maze_size'])
        self.cell_size = np.array(maze['cell_size'])
        self.cell_state = np.array(maze['cell_state'])
        self.wall_thickness = maze['wall_thickness']
        
        # Robot state - [x, y, theta]
        self.robot_pose = np.array([self.cell_size[0] / 2, self.cell_size[1] / 2, 0.0])
        self.robot_radius = config.robot_radius

        # Robot parameters
        self.wheel_radius = config.wheel_radius
        self.wheel_base = config.wheel_base
        self.ang_vel_limit = config.ang_vel_limit

        # LIDAR parameters
        self.num_scans = config.num_scans
        self.max_range = config.max_range
        self.fov = config.fov

        # Noise parameters
        self.motion_noise_cov = config.motion_noise_cov
        self.obs_noise_cov = config.obs_noise_cov

        self.walls = self._get_walls()

    def reset(self):
        self.robot_pose = np.array([self.cell_size[0] / 2, self.cell_size[1] / 2, 0.0])
        return self._get_observation()

    def step(self, wheel_velocities):

        wheel_velocities = np.clip(wheel_velocities, -self.ang_vel_limit, self.ang_vel_limit)
        omega_l, omega_r = wheel_velocities
        dt = 0.1  # time step

        # Differential drive kinematics
        v = (self.wheel_radius / 2) * (omega_r + omega_l)
        w = (self.wheel_radius / self.wheel_base) * (omega_r - omega_l)

        # Current and potential new pose
        x, y, theta = self.robot_pose

        # Add motion noise
        motion_noise = np.random.multivariate_normal(np.zeros(3), self.motion_noise_cov)
        
        # Rotation update
        theta_new = theta + w * dt + motion_noise[2]

        # Intended translation path
        dx = v * np.cos(theta) * dt + motion_noise[0]
        dy = v * np.sin(theta) * dt + motion_noise[1]
        
        # Collision detection
        travel_fraction = 1.0
        
        # If already in collision, don't move
        if self._check_collision(np.array([x, y, theta])):
            travel_fraction = 0.0
            raise ValueError("Collision detected at initial pose.")
        else:
            # Check collision along the movement path
            for wall in self.walls:
                wall_x, wall_y, wall_w, wall_h = wall
                
                # Movement vector
                move_length = np.sqrt(dx*dx + dy*dy)
                if move_length < 1e-10:
                    continue
                    
                # Normalize movement direction
                move_dir_x = dx / move_length
                move_dir_y = dy / move_length
                
                # Check collision with expanded wall (wall + robot radius)
                expanded_x1 = wall_x - self.robot_radius
                expanded_y1 = wall_y - self.robot_radius
                expanded_x2 = wall_x + wall_w + self.robot_radius
                expanded_y2 = wall_y + wall_h + self.robot_radius
                
                # Check intersection with each edge of expanded wall
                for edge_x1, edge_y1, edge_x2, edge_y2 in [
                    (expanded_x1, expanded_y1, expanded_x2, expanded_y1),  # Bottom edge
                    (expanded_x2, expanded_y1, expanded_x2, expanded_y2),  # Right edge
                    (expanded_x2, expanded_y2, expanded_x1, expanded_y2),  # Top edge
                    (expanded_x1, expanded_y2, expanded_x1, expanded_y1),  # Left edge
                ]:
                    # Line-line intersection between movement path and wall edge
                    edge_dir_x = edge_x2 - edge_x1
                    edge_dir_y = edge_y2 - edge_y1
                    
                    denom = move_dir_y * edge_dir_x - move_dir_x * edge_dir_y
                    if abs(denom) < 1e-10:  # Parallel lines
                        continue
                    
                    t_edge = (move_dir_x * (edge_y1 - y) - move_dir_y * (edge_x1 - x)) / denom
                    t_move = (edge_dir_x * (edge_y1 - y) - edge_dir_y * (edge_x1 - x)) / denom
                    
                    # Check if intersection is valid
                    if 0 <= t_edge <= 1 and 0 < t_move <= move_length:
                        collision_fraction = t_move / move_length
                        if collision_fraction < travel_fraction:
                            travel_fraction = max(0, collision_fraction - 1e-6)

        # Update pose to the point just before collision
        final_x = x + dx * travel_fraction
        final_y = y + dy * travel_fraction
        
        # Normalize angle to [-pi, pi] for consistency with FastSLAM
        theta_normalized = self._normalize_angle(theta_new)

        self.robot_pose = np.array([final_x, final_y, theta_normalized])
        
        observation = self._get_observation()

        return observation

    def render(self, observation=None, filename=None):

        fig, ax = plt.subplots(figsize=(self.maze_size[0], self.maze_size[1]))
        ax.set_xlim(-0.5, self.maze_size[0] + 0.5)
        ax.set_ylim(-0.5, self.maze_size[1] + 0.5)
        ax.set_aspect('equal')

        # Draw walls
        for wall in self.walls:
            ax.add_patch(patches.Rectangle((wall[0], wall[1]), wall[2], wall[3], facecolor='blue'))

        # Draw robot
        robot_patch = patches.Circle((self.robot_pose[0], self.robot_pose[1]), self.robot_radius, facecolor='red')
        ax.add_patch(robot_patch)
        
        # Draw LIDAR scans if available
        if observation is not None:
            robot_x, robot_y, robot_theta = self.robot_pose
            lidar_points_x = []
            lidar_points_y = []
            for i, scan_dist in enumerate(observation):
                if scan_dist < self.max_range: 
                    angle_offset = -self.fov / 2 + (i / self.num_scans) * self.fov
                    scan_angle = robot_theta + angle_offset
                    
                    hit_x = robot_x + scan_dist * np.cos(scan_angle)
                    hit_y = robot_y + scan_dist * np.sin(scan_angle)
                    lidar_points_x.append(hit_x)
                    lidar_points_y.append(hit_y)
            
            ax.scatter(lidar_points_x, lidar_points_y, s=2, c='green', zorder=3)

        # Draw robot's direction
        arrow_len = self.robot_radius * 1.5
        ax.arrow(self.robot_pose[0], self.robot_pose[1],
                 arrow_len * np.cos(self.robot_pose[2]),
                 arrow_len * np.sin(self.robot_pose[2]),
                 head_width=0.1, head_length=0.1, fc='k', ec='k')
        
        # Save or show the plot
        if filename:
            plt.savefig(filename)
            print(f"Plot saved to {filename}")
        else:
            plt.show()
        
        plt.close(fig)

    def render_debug(self, observation=None, filename=None):

        fig, ax = plt.subplots(figsize=(self.maze_size[0], self.maze_size[1]))
        ax.set_xlim(-0.5, self.maze_size[0] + 0.5)
        ax.set_ylim(-0.5, self.maze_size[1] + 0.5)
        ax.set_aspect('equal')

        # Draw walls
        for wall in self.walls:
            ax.add_patch(patches.Rectangle((wall[0], wall[1]), wall[2], wall[3], facecolor='blue'))

        # Draw robot
        robot_patch = patches.Circle((self.robot_pose[0], self.robot_pose[1]), self.robot_radius, facecolor='red')
        ax.add_patch(robot_patch)
        
        # Draw LIDAR scans if available
        if observation is not None:
            robot_x, robot_y, robot_theta = self.robot_pose
            lidar_points_x = []
            lidar_points_y = []
            hit_points = []
            for i, scan_dist in enumerate(observation):
                if scan_dist < self.max_range: 
                    angle_offset = -self.fov / 2 + (i / self.num_scans) * self.fov
                    scan_angle = robot_theta + angle_offset
                    
                    hit_x = robot_x + scan_dist * np.cos(scan_angle)
                    hit_y = robot_y + scan_dist * np.sin(scan_angle)
                    lidar_points_x.append(hit_x)
                    lidar_points_y.append(hit_y)
                    hit_points.append([hit_x, hit_y])
            ax.scatter(lidar_points_x, lidar_points_y, s=2, c='green', zorder=3)

            hit_points = np.array(hit_points)
            detector = CornerDetector()
            corners = detector.detect_corners(hit_points) # (#corner, 2)
            if len(corners) > 0:
                ax.scatter(corners[:, 0], corners[:, 1], s=20, c='orange', zorder=4, label='Corners')

        # Draw robot's direction
        arrow_len = self.robot_radius * 1.5
        ax.arrow(self.robot_pose[0], self.robot_pose[1],
                 arrow_len * np.cos(self.robot_pose[2]),
                 arrow_len * np.sin(self.robot_pose[2]),
                 head_width=0.1, head_length=0.1, fc='k', ec='k')
        
        # Save or show the plot
        if filename:
            plt.savefig(filename)
            print(f"Plot saved to {filename}")
        else:
            plt.show()
        
        plt.close(fig)

    def _get_observation(self):
        perfect_scans = self._simulate_lidar(self.robot_pose)

        noise = np.random.normal(0, np.sqrt(self.obs_noise_cov), size=self.num_scans)

        noisy_scans = perfect_scans.copy()
        hit_mask = perfect_scans < self.max_range
        noisy_scans[hit_mask] += noise[hit_mask]
  
        noisy_scans = np.clip(noisy_scans, 0, self.max_range)
    
        return noisy_scans
        
    def _simulate_lidar(self, pose):

        scans = np.full(self.num_scans, self.max_range)
        robot_x, robot_y, robot_theta = pose

        for i in range(self.num_scans):
            
            angle_offset = - self.fov / 2 + (i / self.num_scans) * self.fov
            scan_angle = robot_theta + angle_offset
            ray_dir_x, ray_dir_y = np.cos(scan_angle), np.sin(scan_angle)

            for wall in self.walls:
                wall_x, wall_y, wall_w, wall_h = wall
                
                # Check intersection with the 4 sides of the wall rectangle
                for x1, y1, x2, y2 in [
                    (wall_x, wall_y, wall_x + wall_w, wall_y), # Bottom
                    (wall_x, wall_y, wall_x, wall_y + wall_h), # Left
                    (wall_x + wall_w, wall_y, wall_x + wall_w, wall_y + wall_h), # Right
                    (wall_x, wall_y + wall_h, wall_x + wall_w, wall_y + wall_h) # Top
                ]:
                    
                    seg_dir_x = x2 - x1
                    seg_dir_y = y2 - y1
                    
                    den = seg_dir_x * ray_dir_y - seg_dir_y * ray_dir_x
                    if abs(den) < 1e-10:  # Parallel, might have numerical issues
                        continue
                    
                    # Parameter t for line segment (0 <= t <= 1 means point is on segment)
                    t = ((robot_x - x1) * ray_dir_y - (robot_y - y1) * ray_dir_x) / den
                    if t < 0 or t > 1:
                        continue
                    
                    # Parameter u for ray (u > 0 means point is in front of robot)
                    u = ((robot_x - x1) * seg_dir_y - (robot_y - y1) * seg_dir_x) / den
                    
                    # Check if intersection is valid and closer than current scan
                    if 0 <= t <= 1 and u > 0 and u < scans[i]:
                        scans[i] = min(u, self.max_range)
        
        return scans

    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi] to match FastSLAM representation
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _get_walls(self):
        # Representation of wall - a (x, y, w, h) box

        walls = []
        maze_width = int(self.maze_size[0] / self.cell_size[0])
        maze_height = int(self.maze_size[1] / self.cell_size[1])

        for r in range(maze_height):
            for c in range(maze_width):
                state = self.cell_state[c, r]
                x, y = c * self.cell_size[0], r * self.cell_size[1]
                
                # Top wall
                if state & 1:
                    walls.append([x - self.wall_thickness/2, y + self.cell_size[1] - self.wall_thickness/2, self.cell_size[0] + self.wall_thickness, self.wall_thickness])
                # Right wall
                if state & 2:
                    walls.append([x + self.cell_size[0] - self.wall_thickness/2, y - self.wall_thickness/2, self.wall_thickness, self.cell_size[1] + self.wall_thickness])
                # Bottom wall
                if state & 4:
                    walls.append([x - self.wall_thickness/2, y - self.wall_thickness/2, self.cell_size[0] + self.wall_thickness, self.wall_thickness])
                # Left wall
                if state & 8:
                    walls.append([x - self.wall_thickness/2, y - self.wall_thickness/2, self.wall_thickness, self.cell_size[1] + self.wall_thickness])
                    
        return walls

    def _check_collision(self, pose):

        robot_x, robot_y, _ = pose

        for wall in self.walls:

            wall_x, wall_y, wall_w, wall_h = wall
            
            closest_x = np.clip(robot_x, wall_x, wall_x + wall_w)
            closest_y = np.clip(robot_y, wall_y, wall_y + wall_h)
            
            distance = np.sqrt((robot_x - closest_x)**2 + (robot_y - closest_y)**2)
            
            if distance < self.robot_radius:
                return True
            
        return False


if __name__ == "__main__":

    np.random.seed(42) 
    
    dataset = MazeDataset()
    dataset.load_dataset('dataset/development_dataset.json')
    maze = dataset[100]

    config = RobotConfig()
    env = Env(maze=maze, config=config)

    print("Initial robot pose:", env.robot_pose)
    initial_observation = env.reset()

    num_steps = 200
    action = np.array([5.0, 5.0])
    
    print(f"\nRunning simulation for {num_steps} steps...")
    for i in range(num_steps):
        observation = env.step(action)
        env.render_debug(observation=observation, filename=f"data/step_{i+1}.png")