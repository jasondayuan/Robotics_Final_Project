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
        self.obs_noise_cov = self.config.obs_noise_cov
        
        # Data association parameters
        self.max_association_distance = 0.5
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
            # Predicted observation for this landmark
            predicted_obs = self._observation_model(particle.pose, landmark['mean'])
            
            # Mahalanobis distance for data association
            innovation = observation - predicted_obs
            
            # Innovation covariance (simplified)
            innovation_cov = self.obs_noise_cov + landmark['cov']
            
            try:
                mahal_dist = np.sqrt(innovation.T @ np.linalg.inv(innovation_cov) @ innovation)
                
                if mahal_dist < min_distance and mahal_dist < self.max_association_distance:
                    min_distance = mahal_dist
                    best_id = landmark_id
            except np.linalg.LinAlgError:
                continue
        
        return best_id
    
    def _update_landmark(self, particle, landmark_id, observation):
        """
        Update landmark using EKF
        """
        landmark = particle.landmarks[landmark_id]
        
        # Predicted observation
        predicted_obs = self._observation_model(particle.pose, landmark['mean'])
        
        # Innovation
        innovation = observation - predicted_obs
        
        # Jacobian of observation model w.r.t. landmark position
        H = self._observation_jacobian(particle.pose, landmark['mean'])
        
        # Innovation covariance
        S = H @ landmark['cov'] @ H.T + self.obs_noise_cov
        
        # Kalman gain
        try:
            K = landmark['cov'] @ H.T @ np.linalg.inv(S)
            
            # Update landmark mean and covariance
            landmark['mean'] = landmark['mean'] + K @ innovation
            landmark['cov'] = (np.eye(2) - K @ H) @ landmark['cov']
            
        except np.linalg.LinAlgError:
            pass  # Skip update if matrix is singular
    
    def _initialize_landmark(self, particle, observation):
        """
        Initialize new landmark
        """
        # Convert observation to global coordinates
        landmark_pos = self._observation_to_global(particle.pose, observation)
        
        # Initialize with high uncertainty
        initial_cov = np.eye(2) * 1.0  # High initial uncertainty
        
        # Add to particle's landmark map
        new_id = len(particle.landmarks)
        particle.landmarks[new_id] = {
            'mean': landmark_pos,
            'cov': initial_cov
        }
    
    def _update_weight(self, particle, observations):
        """
        Update particle weight based on observation likelihood
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
                innovation_cov = self.obs_noise_cov + landmark['cov']
                
                try:
                    # Multivariate normal likelihood
                    det = np.linalg.det(innovation_cov)
                    if det > 0:
                        log_weight += -0.5 * (innovation.T @ np.linalg.inv(innovation_cov) @ innovation + 
                                             np.log(2 * np.pi * det))
                except np.linalg.LinAlgError:
                    pass
        
        particle.weight *= np.exp(log_weight)
    
    def _observation_model(self, robot_pose, landmark_pos):
        """
        Predict observation given robot pose and landmark position
        For corner landmarks, this could be direct (x, y) coordinates
        """
        # Transform landmark from global to robot frame
        dx = landmark_pos[0] - robot_pose[0]
        dy = landmark_pos[1] - robot_pose[1]
        
        # Rotate to robot frame
        cos_theta = np.cos(-robot_pose[2])
        sin_theta = np.sin(-robot_pose[2])
        
        local_x = cos_theta * dx - sin_theta * dy
        local_y = sin_theta * dx + cos_theta * dy
        
        return np.array([local_x, local_y])
    
    def _observation_to_global(self, robot_pose, observation):
        """
        Convert observation from robot frame to global frame
        """
        local_x, local_y = observation
        
        # Rotate to global frame
        cos_theta = np.cos(robot_pose[2])
        sin_theta = np.sin(robot_pose[2])
        
        global_x = robot_pose[0] + cos_theta * local_x - sin_theta * local_y
        global_y = robot_pose[1] + sin_theta * local_x + cos_theta * local_y
        
        return np.array([global_x, global_y])
    
    def _observation_jacobian(self, robot_pose, landmark_pos):
        """
        Jacobian of observation model w.r.t. landmark position
        """
        # For direct (x, y) observations, this is the rotation matrix
        cos_theta = np.cos(-robot_pose[2])
        sin_theta = np.sin(-robot_pose[2])
        
        H = np.array([[cos_theta, -sin_theta],
                      [sin_theta,  cos_theta]])
        
        return H
    
    def _normalize_weights(self):
        """
        Normalize particle weights
        """
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
    
    def _effective_sample_size(self):
        """
        Compute effective sample size
        """
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights**2)
    
    def _resample(self):
        """
        Resample particles using systematic resampling
        """
        weights = np.array([p.weight for p in self.particles])
        cumsum = np.cumsum(weights)
        
        # Systematic resampling
        r = np.random.uniform(0, 1.0/self.num_particles)
        new_particles = []
        
        j = 0
        for i in range(self.num_particles):
            u = r + i / self.num_particles
            
            while u > cumsum[j]:
                j += 1
            
            # Deep copy the selected particle
            new_particle = deepcopy(self.particles[j])
            new_particle.weight = 1.0 / self.num_particles
            new_particles.append(new_particle)
        
        self.particles = new_particles
    
    def get_best_estimate(self):
        """
        Get the best pose estimate (highest weight particle)
        """
        best_particle = max(self.particles, key=lambda p: p.weight)
        return best_particle.pose.copy(), deepcopy(best_particle.landmarks)
    
    def get_mean_estimate(self):
        """
        Get weighted mean pose estimate
        """
        weights = np.array([p.weight for p in self.particles])
        
        # Handle circular mean for angle
        x_mean = np.sum([p.pose[0] * p.weight for p in self.particles])
        y_mean = np.sum([p.pose[1] * p.weight for p in self.particles])
        
        # Circular mean for angle
        sin_sum = np.sum([np.sin(p.pose[2]) * p.weight for p in self.particles])
        cos_sum = np.sum([np.cos(p.pose[2]) * p.weight for p in self.particles])
        theta_mean = np.arctan2(sin_sum, cos_sum)
        
        return np.array([x_mean, y_mean, theta_mean])
    
    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi]
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

# Example usage
if __name__ == "__main__":
    # Initialize FastSLAM
    slam = FastSLAM(num_particles=50)
    
    # Example simulation loop
    for step in range(100):
        # Control input [omega_left, omega_right] - wheel angular velocities
        control = np.array([1.0, 1.2])  # Example: slight turn to the right
        dt = 0.1
        
        # Prediction step
        slam.predict(control, dt)
        
        # Simulated observations (corners detected by your corner detector)
        observations = [
            np.array([1.0, 0.5]),  # Example corner observation
            np.array([2.0, 1.0])   # Another corner
        ]
        
        # Update step
        slam.update(observations)
        
        # Get current estimate
        pose_estimate, landmarks = slam.get_best_estimate()
        print(f"Step {step}: Pose = {pose_estimate}, Landmarks = {len(landmarks)}")