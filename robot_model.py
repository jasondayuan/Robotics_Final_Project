import numpy as np
from env import RobotConfig, Env
from slam import FastSLAM
from corner_detector import CornerDetector
from map_generation import MazeDataset

class ModelConfig:
    def __init__(self):
        self.num_particles = 500
        self.cell_size = [0.75, 0.75]
        self.wall_thickness = 0.1

class RobotModel:

    def __init__(self, maze, config=ModelConfig()):

        self.maze = maze

        self.slam = FastSLAM(num_particles=config.num_particles)
        initial_pose = np.array([config.cell_size[0] / 2, config.cell_size[1] / 2, 0.0])
        for particle in self.slam.particles:
            particle.pose = initial_pose.copy()

        self.robot_config = RobotConfig()
        self.env = Env(maze, self.robot_config)

        self.corner_detector = CornerDetector()
    
    def get_control(self, cur_pose, target_waypoint):
        # Return one control command to move towards the target waypoint

        dx = target_waypoint[0] - cur_pose[0]
        dy = target_waypoint[1] - cur_pose[1]
        aim_angle = np.arctan2(dy, dx)

        # Align with aim angle first
        angle_diff = np.abs(aim_angle - cur_pose[2])
        if angle_diff > np.deg2rad(3):
            sgn = np.sign(np.sin(aim_angle - cur_pose[2]))
            return np.array([-1.2, 1.2]) * sgn
        
        # If aligned, move forward
        return np.array([2.84, 2.84])
        
    def run(self, render=False, debug=False):
        # Get the waypoints of the maze
        waypoints = MazeDataset().get_dfs_order(maze)
        current_waypoint_idx = 1
        counter = 0

        if debug:
            log_handler = open("log.txt", "w")

        while True:

            # Is the robot close enough to the current target waypoint?
            cur_pose = self._get_current_pose()
            target_waypoint = waypoints[current_waypoint_idx]
            distance_to_target = np.linalg.norm(np.array(target_waypoint) - np.array(cur_pose[:2]))

            if debug:
                print(f"Counter: {counter}, Current pose: {cur_pose}", file=log_handler, flush=True)
                # print(f"Differnece: {self.env.robot_pose - cur_pose}", file=log_handler, flush=True)

            # If so, change the target to the next waypoint
            if distance_to_target < 0.05:
                current_waypoint_idx += 1
                if current_waypoint_idx >= len(waypoints):
                    print("Reached the last waypoint!")
                    break
                target_waypoint = waypoints[current_waypoint_idx]
                print(f"Reached waypoint {current_waypoint_idx}, moving to next...")
                continue

            # Else, continue moving towards the current target waypoint and SLAMming
            control = self.get_control(cur_pose, target_waypoint)

            if debug:
                print(f"Counter: {counter}, Control: {control}", file=log_handler, flush=True)

            self.slam.predict(control, self.robot_config.dt)

            lidar_observations = self.env.step(control)
            hit_points = []
            for i, scan_dist in enumerate(lidar_observations):
                if scan_dist < self.robot_config.max_range and scan_dist > 0.05:
                    angle_offset = -self.robot_config.fov / 2 + (i / self.robot_config.num_scans) * self.robot_config.fov
                    obs_x = scan_dist * np.cos(angle_offset)
                    obs_y = scan_dist * np.sin(angle_offset) 
                    hit_points.append(np.array([obs_x, obs_y]))
            hit_points = np.array(hit_points)

            if render:
                self.env.render(observation=lidar_observations, filename=f"data/step_{counter}.png")

            corners = self.corner_detector.detect_corners(hit_points)

            self.slam.update(corners)

            counter += 1

            if counter > 100000:
                print("Too many iterations, stopping.")
                break
    
    def _get_current_pose(self):
        return self.slam.get_mean_estimate()

if __name__ == "__main__":

    np.random.seed(42)
    
    dataset = MazeDataset()
    dataset.load_dataset("dataset/maze_dataset.json")
    maze = dataset[0]

    model = RobotModel(maze)
    model.run(render=True, debug=True)