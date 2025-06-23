import numpy as np

class CornerDetector:

    def __init__(self):
        self.epsilon = 0.015
        self.inlier_ratio = 0.1
        self.min_points_for_ransac = 10

    def detect_corners(self, points):
        # points - (N, 2) numpy array of points

        if len(points) < self.min_points_for_ransac:
            return np.array([])

        # Extract lines with iterative RANSAC
        lines = self.iterative_ransac(points)
        if len(lines) < 2:
            return np.array([])

        # Take all intersections
        intersections = self.find_line_intersections(lines)
        if len(intersections) == 0:
            return np.array([])

        # Filter intersections: only keep those with at least 5 points within 2*epsilon
        corners = self.filter_intersections_by_point_density(intersections, points)

        return corners

    def iterative_ransac(self, points):

        lines = []
        N = points.shape[0]
        remaining_points = points.copy()
        min_inliers = int(self.inlier_ratio * N)
        
        while len(remaining_points) >= self.inlier_ratio * N:

            line_params, inliers = self.ransac(remaining_points, min_inliers)
            
            if line_params is None:
                break
                
            lines.append(line_params)
            
            # Remove inliers for next iteration
            remaining_points = remaining_points[~inliers]
        
        return lines

    def ransac(self, points, min_inliers):

        if len(points) < 2:
            return None, None

        N = len(points)
        max_iterations = 200
        best_line = None
        best_inliers = None
        best_inlier_count = 0

        for _ in range(max_iterations):
            # Randomly sample 2 points
            sample_indices = np.random.choice(N, 2, replace=False)
            p1, p2 = points[sample_indices]

            # Fit line through these 2 points
            line_params = self.fit_line(p1, p2)
            if line_params is None:
                continue

            # Count inliers
            distances = self.point_to_line_distance(points, line_params)
            inliers = distances < self.epsilon
            inlier_count = np.sum(inliers)

            # Update best if this is better
            if inlier_count > best_inlier_count and inlier_count >= min_inliers:
                best_inlier_count = inlier_count
                best_line = line_params
                best_inliers = inliers

        return best_line, best_inliers

    def fit_line(self, p1, p2):

        x1, y1 = p1
        x2, y2 = p2
        
        if np.allclose(p1, p2):
            return None

        a = y2 - y1
        b = -(x2 - x1)
        c = (x2 - x1) * y1 - (y2 - y1) * x1

        norm = np.sqrt(a*a + b*b)
        if norm < 1e-10:
            return None
            
        return (a/norm, b/norm, c/norm)

    def point_to_line_distance(self, points, line_params):
        a, b, c = line_params
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        return distances

    def find_line_intersections(self, lines):

        intersections = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                intersection = self.line_intersection(lines[i], lines[j])
                if intersection is not None:
                    intersections.append(intersection)
        
        return np.array(intersections) if intersections else np.array([])

    def line_intersection(self, line1, line2):

        a1, b1, c1 = line1
        a2, b2, c2 = line2
        
        det = a1 * b2 - a2 * b1
        
        if abs(det) < 1e-10:  # Lines are parallel
            return None
            
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        
        return np.array([x, y])

    def filter_intersections_by_point_density(self, intersections, points):

        valid_corners = []
        min_points_required = 2
        search_radius = 2 * self.epsilon
        
        for intersection in intersections:
            # Calculate distances from intersection to all points
            distances = np.linalg.norm(points - intersection, axis=1)
            
            # Count points within 2*epsilon radius
            nearby_points = np.sum(distances <= search_radius)
            
            # Accept intersection if it has enough nearby points
            if nearby_points >= min_points_required:
                valid_corners.append(intersection)
        
        return np.array(valid_corners) if valid_corners else np.array([])
