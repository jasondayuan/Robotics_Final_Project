import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from map_generation import MazeDataset

class Env:
    def __init__(self, maze, motion_noise_cov, obs_noise_cov, num_scans=360, max_range=5.0, wheel_radius=0.05, wheel_base=0.3, fov=np.pi/2):

        self.maze = maze
        self.maze_size = np.array(maze['maze_size'])
        self.cell_size = np.array(maze['cell_size'])
        self.cell_state = np.array(maze['cell_state'])
        self.wall_thickness = maze['wall_thickness']
        
        # Robot state - [x, y, theta]
        self.robot_pose = np.array([self.cell_size[0] / 2, self.cell_size[1] / 2, 0.0])
        self.robot_radius = self.cell_size[0] / 4

        # Robot parameters
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base

        # LIDAR parameters
        self.num_scans = num_scans
        self.max_range = max_range
        self.fov = fov

        # Noise parameters
        self.motion_noise_cov = motion_noise_cov
        self.obs_noise_cov = obs_noise_cov

        self.walls = self._get_walls()

    def reset(self):
        self.robot_pose = np.array([self.cell_size[0] / 2, self.cell_size[1] / 2, 0.0])
        return self._get_observation()

    def step(self, wheel_velocities):

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
        travel_fraction = 1.0 # Represents the fraction of movement allowed
        
        # Check path against all walls
        for wall in self.walls:
            # Approximate continuous collision by checking collision with a wall expanded by the robot's radius
            expanded_wall_x = wall[0] - self.robot_radius
            expanded_wall_y = wall[1] - self.robot_radius
            expanded_wall_w = wall[2] + 2 * self.robot_radius
            expanded_wall_h = wall[3] + 2 * self.robot_radius

            # Check intersection with the 4 edges of the expanded wall
            for x1, y1, x2, y2 in [
                (expanded_wall_x, expanded_wall_y, expanded_wall_x + expanded_wall_w, expanded_wall_y),
                (expanded_wall_x, expanded_wall_y, expanded_wall_x, expanded_wall_y + expanded_wall_h),
                (expanded_wall_x + expanded_wall_w, expanded_wall_y, expanded_wall_x + expanded_wall_w, expanded_wall_y + expanded_wall_h),
                (expanded_wall_x, expanded_wall_y + expanded_wall_h, expanded_wall_x + expanded_wall_w, expanded_wall_y + expanded_wall_h)
            ]:
                den = (x1 - x2) * dy - (y1 - y2) * dx
                if den == 0:
                    continue
                
                t = ((x1 - x) * dy - (y1 - y) * dx) / den
                u = -((x1 - x2) * (y1 - y) - (y1 - y2) * (x1 - x)) / den

                # If intersection occurs within the path segment and wall edge
                if 0 < t < 1 and 0 < u < travel_fraction:
                    travel_fraction = u # We found a closer collision

        # Update pose to the point of collision
        final_x = x + dx * travel_fraction
        final_y = y + dy * travel_fraction

        self.robot_pose = np.array([final_x, final_y, theta_new])
        
        observation = self._get_observation()

        return observation

    def render(self, observation=None, filename=None):
        """
        Renders the environment, optionally showing LIDAR scans and saving to a file.
        """
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

    def _get_observation(self):
        perfect_scans = self._simulate_lidar(self.robot_pose)
        noise = np.random.normal(0, np.sqrt(self.obs_noise_cov))
        noisy_scans = perfect_scans + noise
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
    
    dataset = MazeDataset()
    dataset.load_dataset('dataset/development_dataset.json')
    first_maze = dataset[0]

    motion_cov = np.diag([0.01**2, 0.01**2, np.deg2rad(0.5)**2])
    obs_var = 0.02**2

    env = Env(
        maze=first_maze,
        motion_noise_cov=motion_cov,
        obs_noise_cov=obs_var,
        num_scans=90, 
        fov=np.pi * 2
    )

    print("Initial robot pose:", env.robot_pose)
    initial_observation = env.reset()
    env.render()

    num_steps = 50
    action = np.array([5.0, 4.5])
    
    print(f"\nRunning simulation for {num_steps} steps...")
    for i in range(num_steps):
        observation = env.step(action)
        env.render(observation=observation, filename=f"data/step_{i+1}.png")