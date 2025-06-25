import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from env import RobotConfig

class Particle:
    def __init__(self):

        # Robot pose [x, y, theta]
        self.pose = np.zeros(3)
        
        # Particle weight
        self.weight = 1.0
        
        # Dictionary of landmarks: {landmark_id: {'mean': [x, y], 'cov': 2x2 matrix}}
        self.landmarks = {}
        
        # Path history for visualization
        self.path = []

class FastSLAM:

    def __init__(self, num_particles=100):

        self.config = RobotConfig()

        self.num_particles = num_particles
        self.particles = [Particle() for _ in range(num_particles)]
        
        # Motion model noise
        self.motion_noise_cov = self.config.motion_noise_cov 
        
        # Observation model noise
        self.obs_noise_cov = np.diag([self.config.obs_noise_std**2, self.config.obs_noise_std**2])
        
        # Data association parameters
        self.max_association_distance = 0.2
        self.new_landmark_threshold = 0.3
        
        # Resampling parameters
        self.effective_sample_size_threshold = self.num_particles / 3
        
    def predict(self, control_input, dt):
    
        for particle in self.particles:
            # Apply motion model with noise
            self._motion_model(particle, control_input, dt)
            
            # Store path for visualization
            particle.path.append(particle.pose.copy())
    
    def update(self, observations):

        for particle in self.particles:
     
            for obs in observations:

                # Data association
                associated_id = self._data_association(particle, obs)
                
                # Update landmark if associated, or initialize if new
                if associated_id is not None:
                    self._update_landmark(particle, associated_id, obs)
                else:
                    self._initialize_landmark(particle, obs)
            
            # Update particle weight
            self._update_weight(particle, observations)
        
        # Normalize weights
        self._normalize_weights()
        
        # Resample if particle degeneracy is detected
        if self._effective_sample_size() < self.effective_sample_size_threshold:
            self._resample()
    
    def _motion_model(self, particle, control_input, dt):

        omega_l, omega_r = control_input
        x, y, theta = particle.pose
        
        # Robot parameters
        wheel_radius = self.config.wheel_radius  
        wheel_base = self.config.wheel_base
        
        # Convert wheel velocities to robot velocities
        v = (wheel_radius / 2) * (omega_r + omega_l) 
        omega = (wheel_radius / wheel_base) * (omega_r - omega_l)
        
        # Differential drive kinematics
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)
        dtheta = omega * dt
        
        # Add motion noise
        motion_noise = np.random.multivariate_normal(np.zeros(3), self.motion_noise_cov)
        
        # Update pose
        particle.pose[0] += dx + motion_noise[0]
        particle.pose[1] += dy + motion_noise[1]
        particle.pose[2] += dtheta + motion_noise[2]
        
        # Normalize angle
        particle.pose[2] = self._normalize_angle(particle.pose[2])
    
    def _data_association(self, particle, observation):
 
        if len(particle.landmarks) == 0:
            return None
        
        min_distance = float('inf')
        best_id = None
        
        for landmark_id, landmark in particle.landmarks.items():

            predicted_obs = self._observation_model(particle.pose, landmark['mean'])
            
            distance = np.linalg.norm(observation - predicted_obs)
            
            if distance < min_distance and distance < self.max_association_distance:
                min_distance = distance
                best_id = landmark_id
        
        return best_id
    
    def _update_landmark(self, particle, landmark_id, observation):
        # Update existing landmark with new observation

        landmark = particle.landmarks[landmark_id]
        
        # Predicted observation
        predicted_obs = self._observation_model(particle.pose, landmark['mean'])
        
        # Innovation
        innovation = observation - predicted_obs
        
        # Jacobian
        H = self._observation_jacobian(particle.pose, landmark['mean'])
        
        # Measurement covariance
        S = H @ landmark['cov'] @ H.T + self.obs_noise_cov
        
        # Kalman gain
        try:
            K = landmark['cov'] @ H.T @ np.linalg.inv(S)
            
            # Update landmark mean and covariance
            landmark['mean'] = landmark['mean'] + K @ innovation
            landmark['cov'] = (np.eye(2) - K @ H) @ landmark['cov']
            
        except np.linalg.LinAlgError:
            pass  # Skip update if S is not invertible
    
    def _initialize_landmark(self, particle, observation):
        # Initialize a new landmark with the observation

        landmark_pos = self._inverse_observation_model(particle.pose, observation)
        
        H = self._observation_jacobian(particle.pose, landmark_pos)

        try:
            H_inv = np.linalg.inv(H)
            landmark_cov = H_inv @ self.obs_noise_cov @ H_inv.T
        except np.linalg.LinAlgError:
            landmark_cov = np.eye(2) * 0.1
        
        # Add to particle's landmark map
        new_id = len(particle.landmarks)
        particle.landmarks[new_id] = {
            'mean': landmark_pos,
            'cov': landmark_cov
        }
    
    def _update_weight(self, particle, observations):
        """
        Update particle weight based on observation
        """
        log_weight = 0.0
        
        for obs in observations:
            # Find associated landmark
            associated_id = self._data_association(particle, obs)
            
            if associated_id is not None:
                landmark = particle.landmarks[associated_id]
                predicted_obs = self._observation_model(particle.pose, landmark['mean'])
                
                # Compute likelihood
                innovation = obs - predicted_obs
                H = self._observation_jacobian(particle.pose, landmark['mean'])
                innovation_cov = self.obs_noise_cov + H @ landmark['cov'] @ H.T
                
                try:
                    det = np.linalg.det(innovation_cov)
                    if det > 0:
                        log_weight += -0.5 * (innovation.T @ np.linalg.inv(innovation_cov) @ innovation + 
                                             np.log(2 * np.pi * det))
                except np.linalg.LinAlgError:
                    pass
        
        particle.weight = np.exp(log_weight)
    
    def _observation_model(self, robot_pose, landmark_pos):
        """
        robot_pose: [x, y, theta]
        landmark_pos: [m_x, m_y], under the global frame
        Returns the observation in the robot frame [local_x, local_y]
        """

        dx = landmark_pos[0] - robot_pose[0]
        dy = landmark_pos[1] - robot_pose[1]
        
        cos_theta = np.cos(robot_pose[2])
        sin_theta = np.sin(robot_pose[2])
        
        local_x = cos_theta * dx + sin_theta * dy
        local_y = -sin_theta * dx + cos_theta * dy
        
        return np.array([local_x, local_y])
    
    def _inverse_observation_model(self, robot_pose, observation):
        # Inverse observation model: convert observation from robot frame to global frame

        local_x, local_y = observation
        
        # Rotate to global frame
        cos_theta = np.cos(robot_pose[2])
        sin_theta = np.sin(robot_pose[2])
        
        global_x = robot_pose[0] + cos_theta * local_x - sin_theta * local_y
        global_y = robot_pose[1] + sin_theta * local_x + cos_theta * local_y
        
        return np.array([global_x, global_y])
    
    def _observation_jacobian(self, robot_pose, landmark_pos):
        cos_theta = np.cos(robot_pose[2])
        sin_theta = np.sin(robot_pose[2])
    
        H = np.array([[cos_theta, sin_theta],
                      [-sin_theta,  cos_theta]])
        return H
    
    def _normalize_weights(self):
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
    
    def _effective_sample_size(self):
        # Estimates how many particles are actually contributing to the estimate
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights**2)
    
    def _resample(self):

        weights = np.array([p.weight for p in self.particles])
        cumsum = np.cumsum(weights)
        
        r = np.random.uniform(0, 1.0/self.num_particles)
        new_particles = []
        
        j = 0
        for i in range(self.num_particles):
            u = r + i / self.num_particles
            
            while u > cumsum[j]:
                j += 1
            
            new_particle = deepcopy(self.particles[j])
            new_particle.weight = 1.0 / self.num_particles
            new_particles.append(new_particle)
        
        self.particles = new_particles
    
    def get_best_estimate(self):
        """
        Get the pose estimation and landmarks of the best particle
        """
        best_particle = max(self.particles, key=lambda p: p.weight)
        return best_particle.pose.copy(), deepcopy(best_particle.landmarks)
    
    def get_mean_estimate(self):
        """
        Get weighted mean estimation of robot pose
        """
        weights = np.array([p.weight for p in self.particles])
        weight_sum = np.sum(weights)
        
        # Handle circular mean for angle
        x_mean = np.sum([p.pose[0] * p.weight for p in self.particles]) / weight_sum
        y_mean = np.sum([p.pose[1] * p.weight for p in self.particles]) / weight_sum
        
        # Circular mean for angle
        sin_sum = np.sum([np.sin(p.pose[2]) * p.weight for p in self.particles]) / weight_sum
        cos_sum = np.sum([np.cos(p.pose[2]) * p.weight for p in self.particles]) / weight_sum
        theta_mean = np.arctan2(sin_sum, cos_sum)
        
        return np.array([x_mean, y_mean, theta_mean])
    
    def _normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

if __name__ == "__main__":
    print("=== FastSLAM Validation ===")
    
    # Initialize FastSLAM
    slam = FastSLAM(num_particles=100)
    
    # Ground truth setup
    gt_pose = np.array([0.5, 0.5, 0.0])  # Starting pose [x, y, theta]
    
    # True landmark positions in global frame (simulated corner positions)
    true_landmarks = np.array([
        [2.0, 1.0],   # Corner 1
        [3.0, 3.0],   # Corner 2
        [1.0, 4.0],   # Corner 3
        [4.0, 2.0],   # Corner 4
        [0.5, 3.5]    # Corner 5
    ])
    
    # Initialize all particles with the same starting pose for comparison
    for particle in slam.particles:
        particle.pose = gt_pose.copy()
    
    # Simulation parameters
    dt = 0.1
    num_steps = 150
    
    # Storage for analysis
    gt_trajectory = [gt_pose.copy()]
    estimated_trajectory = []
    pose_errors = []
    
    print(f"Initial GT pose: {gt_pose}")
    print(f"True landmarks: {len(true_landmarks)} corners")
    print(f"Running simulation for {num_steps} steps...")
    
    # Simulation loop
    for step in range(num_steps):
        
        # Control input - create interesting trajectory
        if step < 50:
            control = np.array([2.0, 2.0])    # Move forward
        elif step < 100:
            control = np.array([1.5, 2.5])   # Turn right while moving
        else:
            control = np.array([2.2, 1.8])   # Turn left while moving
        
        # Update ground truth pose (same motion model as SLAM)
        config = slam.config
        omega_l, omega_r = control
        v = (config.wheel_radius / 2) * (omega_r + omega_l)
        omega = (config.wheel_radius / config.wheel_base) * (omega_r - omega_l)
        
        # Ground truth motion (no noise)
        dx = v * dt * np.cos(gt_pose[2])
        dy = v * dt * np.sin(gt_pose[2])
        dtheta = omega * dt
        
        gt_pose[0] += dx
        gt_pose[1] += dy
        gt_pose[2] += dtheta
        gt_pose[2] = np.arctan2(np.sin(gt_pose[2]), np.cos(gt_pose[2]))  # Normalize angle
        
        gt_trajectory.append(gt_pose.copy())
        
        # Generate observations: corners visible from current position
        observations = []
        max_range = 3.0  # Maximum sensor range
        
        for landmark in true_landmarks:
            # Transform landmark to robot frame
            dx = landmark[0] - gt_pose[0]
            dy = landmark[1] - gt_pose[1]
            
            # Check if landmark is within range
            distance = np.sqrt(dx*dx + dy*dy)
            if distance < max_range:
                # Transform to robot frame
                cos_theta = np.cos(gt_pose[2])
                sin_theta = np.sin(gt_pose[2])
                
                local_x = cos_theta * dx + sin_theta * dy
                local_y = -sin_theta * dx + cos_theta * dy
                
                # Add observation noise to simulate corner detection uncertainty
                noise = np.random.multivariate_normal([0, 0], slam.obs_noise_cov)
                observed_corner = np.array([local_x, local_y]) + noise
                
                # Only include observations in front of robot (positive x in robot frame)
                if observed_corner[0] > 0:
                    observations.append(observed_corner)
        
        # SLAM prediction step
        slam.predict(control, dt)
        
        # SLAM update step (only if we have observations)
        if len(observations) > 0:
            slam.update(observations)
        
        # Get estimates
        pose_estimate, landmarks_estimate = slam.get_best_estimate()
        mean_pose_estimate = slam.get_mean_estimate()
        estimated_trajectory.append(pose_estimate.copy())
        
        # Compute pose error
        position_error = np.linalg.norm(pose_estimate[:2] - gt_pose[:2])
        angle_error = abs(np.arctan2(np.sin(pose_estimate[2] - gt_pose[2]), 
                                   np.cos(pose_estimate[2] - gt_pose[2])))
        pose_errors.append([position_error, angle_error])
        
        # Print progress every 25 steps
        if step % 25 == 0:
            print(f"\nStep {step}:")
            print(f"  Control: [{control[0]:.1f}, {control[1]:.1f}] rad/s")
            print(f"  GT pose: [{gt_pose[0]:.2f}, {gt_pose[1]:.2f}, {gt_pose[2]:.2f}]")
            print(f"  Est pose: [{pose_estimate[0]:.2f}, {pose_estimate[1]:.2f}, {pose_estimate[2]:.2f}]")
            print(f"  Mean pose: [{mean_pose_estimate[0]:.2f}, {mean_pose_estimate[1]:.2f}, {mean_pose_estimate[2]:.2f}]")
            print(f"  Position error: {position_error:.3f} m")
            print(f"  Angle error: {np.rad2deg(angle_error):.1f} deg")
            print(f"  Observations: {len(observations)} corners")
            print(f"  Landmarks: {len(landmarks_estimate)} total")
            print(f"  ESS: {slam._effective_sample_size():.1f}")
    
    # Final analysis
    pose_errors = np.array(pose_errors)
    print(f"\n=== Final Results ===")
    print(f"Total trajectory length: {num_steps} steps")
    print(f"Average position error: {np.mean(pose_errors[:, 0]):.3f} ± {np.std(pose_errors[:, 0]):.3f} m")
    print(f"Average angle error: {np.rad2deg(np.mean(pose_errors[:, 1])):.1f} ± {np.rad2deg(np.std(pose_errors[:, 1])):.1f} deg")
    print(f"Final position error: {pose_errors[-1, 0]:.3f} m")
    print(f"Final angle error: {np.rad2deg(pose_errors[-1, 1]):.1f} deg")
    print(f"Final landmarks detected: {len(landmarks_estimate)} out of {len(true_landmarks)} true landmarks")
    
    # Simple visualization (optional - requires matplotlib)
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot trajectory
        gt_traj = np.array(gt_trajectory)
        est_traj = np.array(estimated_trajectory)
        
        ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', linewidth=2, label='Ground Truth')
        ax1.plot(est_traj[:, 0], est_traj[:, 1], 'r--', linewidth=2, label='SLAM Estimate')
        ax1.scatter(true_landmarks[:, 0], true_landmarks[:, 1], c='blue', s=100, marker='s', label='True Landmarks')
        
        # Plot estimated landmarks
        if len(landmarks_estimate) > 0:
            est_landmarks = np.array([lm['mean'] for lm in landmarks_estimate.values()])
            ax1.scatter(est_landmarks[:, 0], est_landmarks[:, 1], c='red', s=50, marker='x', label='Est. Landmarks')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('SLAM Trajectory Comparison')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot errors over time
        ax2.plot(pose_errors[:, 0], 'b-', label='Position Error')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(np.rad2deg(pose_errors[:, 1]), 'r-', label='Angle Error')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position Error (m)', color='b')
        ax2_twin.set_ylabel('Angle Error (deg)', color='r')
        ax2.set_title('SLAM Estimation Errors')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")
    
    print("\n=== Validation Complete ===")