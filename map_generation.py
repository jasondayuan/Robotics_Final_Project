import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json
import os
from tqdm import tqdm

class _DSU:

    def __init__(self, cells):
        self._parent = {cell: cell for cell in cells}
        self._rank = {cell: 0 for cell in cells}

    def find(self, cell):
        if self._parent[cell] == cell:
            return cell
        self._parent[cell] = self.find(self._parent[cell])
        return self._parent[cell]

    def union(self, cell1, cell2):
        root1 = self.find(cell1)
        root2 = self.find(cell2)

        if root1 != root2:
            if self._rank[root1] < self._rank[root2]:
                self._parent[root1] = root2
            elif self._rank[root1] > self._rank[root2]:
                self._parent[root2] = root1
            else:
                self._parent[root2] = root1
                self._rank[root1] += 1
            return True
        
        return False

def generate_random_kruskal(maze_width, maze_height):
    cells = [(c, r) for r in range(maze_height) for c in range(maze_width)]
    dsu = _DSU(cells)

    edges = []
    for r in range(maze_height):
        for c in range(maze_width):
            if c < maze_width - 1:
                edges.append(((c, r), (c + 1, r)))
            if r < maze_height - 1:
                edges.append(((c, r), (c, r + 1)))

    random.shuffle(edges)

    resulting_edges = set()
    for cell1, cell2 in edges:
        if dsu.union(cell1, cell2):
            resulting_edges.add((cell1, cell2))

    return resulting_edges

class MazeDataset:
    def __init__(self):
        self.mazes = []

    def save_dataset(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        json_data = []
        for maze in self.mazes:
            maze_dict = {}
            for key, value in maze.items():
                if isinstance(value, np.ndarray):
                    maze_dict[key] = value.tolist()
                else:
                    maze_dict[key] = value
            json_data.append(maze_dict)
    
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)

    def show_maze(self, maze):
        cell_size = maze['cell_size']
        maze_width =int(maze['maze_size'][0] / cell_size[0])
        maze_height = int(maze['maze_size'][1] / cell_size[1])
        cell_state = np.array(maze['cell_state'])
        wall_thickness = maze['wall_thickness']

        fig, ax = plt.subplots(figsize=(maze['maze_size'][0], maze['maze_size'][1]))
        ax.set_xlim(-0.5, maze['maze_size'][0] + 0.5)
        ax.set_ylim(-0.5, maze['maze_size'][1] + 0.5)
        ax.set_aspect('equal')

        for r in range(maze_height):
            for c in range(maze_width):
                state = cell_state[c, r]
                # Top wall
                if state & 1:
                    ax.add_patch(patches.Rectangle((c * cell_size[0] - wall_thickness/2, (r + 1) * cell_size[1] - wall_thickness/2), 
                                                   cell_size[0] + wall_thickness, 
                                                   wall_thickness, 
                                                   facecolor='blue'))
                # Right wall
                if state & 2:
                    ax.add_patch(patches.Rectangle(((c + 1) * cell_size[0] - wall_thickness/2, r * cell_size[1] - wall_thickness/2), 
                                                   wall_thickness, 
                                                   cell_size[1] + wall_thickness, 
                                                   facecolor='blue'))
                # Bottom wall
                if state & 4:
                    ax.add_patch(patches.Rectangle((c * cell_size[0] - wall_thickness/2, r * cell_size[1] - wall_thickness/2), 
                                                   cell_size[0] + wall_thickness, 
                                                   wall_thickness, 
                                                   facecolor='blue'))
                # Left wall
                if state & 8:
                    ax.add_patch(patches.Rectangle((c * cell_size[0] - wall_thickness/2, r * cell_size[1] - wall_thickness/2), 
                                                   wall_thickness, 
                                                   cell_size[1] + wall_thickness, 
                                                   facecolor='blue'))
        
        plt.show()

    def __len__(self):
        return len(self.mazes)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.mazes):
            raise IndexError("Index out of bounds")
        return self.mazes[idx]
    
    def load_dataset(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")
        
        with open(filename, 'r') as f:
            json_data = json.load(f)
            self.mazes = []
            for maze_dict in json_data:
                maze = {}
                for key, value in maze_dict.items():
                    if isinstance(value, list):
                        maze[key] = np.array(value)
                    else:
                        maze[key] = value
                self.mazes.append(maze)

    def generate_mazes(self, maze_size, cell_size, num_mazes, wall_thickness):
        for _ in tqdm(range(num_mazes)):
            cell_state = self._generate_maze(maze_size, cell_size)
            maze = {'maze_size': maze_size, 'cell_size': cell_size, 'cell_state': cell_state, 'wall_thickness': wall_thickness}
            self.mazes.append(maze)

    def _generate_maze(self, maze_size, cell_size):
        maze_width = int(np.floor(maze_size[0] / cell_size[0]))
        maze_height = int(np.floor(maze_size[1] / cell_size[1]))
        maze = np.zeros((maze_width, maze_height), dtype=np.uint8)
        edges = generate_random_kruskal(maze_width, maze_height)

        for (c1, r1), (c2, r2) in edges: 
            if c1 == c2: 
                if r1 < r2: 
                    maze[c1, r1] |= 1
                    maze[c2, r2] |= 4
                else: 
                    maze[c1, r1] |= 4
                    maze[c2, r2] |= 1 
            elif r1 == r2: 
                if c1 < c2: 
                    maze[c1, r1] |= 2 
                    maze[c2, r2] |= 8 
                else: 
                    maze[c1, r1] |= 8 
                    maze[c2, r2] |= 2 
            else:
                raise ValueError("Cells are not adjacent")
            
        for r in range(maze_height):
            for c in range(maze_width):
                maze[c, r] =( ~maze[c, r]) & 15
            
        return maze
    
    def get_dfs_order(self, maze, start_cell=None):

        cell_size = maze['cell_size']
        maze_size = maze['maze_size']
        cell_state = np.array(maze['cell_state'])
        
        maze_width = int(maze_size[0] / cell_size[0])
        maze_height = int(maze_size[1] / cell_size[1])
        
        # Determine starting cell
        if start_cell is None:
            start_cell = (0, 0) 
        
        if not (0 <= start_cell[0] < maze_width and 0 <= start_cell[1] < maze_height):
            raise ValueError(f"Start cell {start_cell} is out of bounds for maze size {maze_width}x{maze_height}")
        
        def get_neighbors(c, r):

            neighbors = []
            state = cell_state[c, r]
            
            # Check each direction - if there's NO wall, we can move there
            # Top (r+1) - bit 0
            if not (state & 1) and r + 1 < maze_height:
                neighbors.append((c, r + 1))
            
            # Right (c+1) - bit 1  
            if not (state & 2) and c + 1 < maze_width:
                neighbors.append((c + 1, r))
            
            # Bottom (r-1) - bit 2
            if not (state & 4) and r - 1 >= 0:
                neighbors.append((c, r - 1))
            
            # Left (c-1) - bit 3
            if not (state & 8) and c - 1 >= 0:
                neighbors.append((c - 1, r))
            
            return neighbors
        
        visited = set()
        robot_path = []
        
        def dfs_recursive(cell):

            nonlocal visited, robot_path

            current_path = []
            
            if cell in visited:
                return
            
            visited.add(cell)
            current_path.append(cell)

            neighbors = get_neighbors(cell[0], cell[1])
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            for i, neighbor in enumerate(unvisited_neighbors):
                current_path.extend(dfs_recursive(neighbor))
                current_path.append(cell)
            
            return current_path
        
        robot_path = dfs_recursive(start_cell)

        world_coordinates = []
        for c, r in robot_path:
            x = (c + 0.5) * cell_size[0]
            y = (r + 0.5) * cell_size[1]
            world_coordinates.append((x, y))
        
        return world_coordinates

    def visualize_dfs_path(self, maze, dfs_path):

        cell_size = maze['cell_size']
        maze_width = int(maze['maze_size'][0] / cell_size[0])
        maze_height = int(maze['maze_size'][1] / cell_size[1])
        cell_state = np.array(maze['cell_state'])
        wall_thickness = maze['wall_thickness']

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-0.5, maze['maze_size'][0] + 0.5)
        ax.set_ylim(-0.5, maze['maze_size'][1] + 0.5)
        ax.set_aspect('equal')

        # Draw maze walls
        for r in range(maze_height):
            for c in range(maze_width):
                state = cell_state[c, r]
                # Top wall
                if state & 1:
                    ax.add_patch(patches.Rectangle((c * cell_size[0] - wall_thickness/2, (r + 1) * cell_size[1] - wall_thickness/2), 
                                                   cell_size[0] + wall_thickness, 
                                                   wall_thickness, 
                                                   facecolor='blue'))
                # Right wall
                if state & 2:
                    ax.add_patch(patches.Rectangle(((c + 1) * cell_size[0] - wall_thickness/2, r * cell_size[1] - wall_thickness/2), 
                                                   wall_thickness, 
                                                   cell_size[1] + wall_thickness, 
                                                   facecolor='blue'))
                # Bottom wall
                if state & 4:
                    ax.add_patch(patches.Rectangle((c * cell_size[0] - wall_thickness/2, r * cell_size[1] - wall_thickness/2), 
                                                   cell_size[0] + wall_thickness, 
                                                   wall_thickness, 
                                                   facecolor='blue'))
                # Left wall
                if state & 8:
                    ax.add_patch(patches.Rectangle((c * cell_size[0] - wall_thickness/2, r * cell_size[1] - wall_thickness/2), 
                                                   wall_thickness, 
                                                   cell_size[1] + wall_thickness, 
                                                   facecolor='blue'))

        # Draw DFS path with arrows and numbers
        if len(dfs_path) > 1:
            path_x = [point[0] for point in dfs_path]
            path_y = [point[1] for point in dfs_path]
            ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7, label='Robot Path')
            
            # Add arrows to show direction
            for i in range(0, len(dfs_path) - 1, max(1, len(dfs_path) // 20)):  # Show arrows at intervals
                dx = dfs_path[i+1][0] - dfs_path[i][0]
                dy = dfs_path[i+1][1] - dfs_path[i][1]
                if dx != 0 or dy != 0:  # Only draw arrow if there's movement
                    ax.arrow(dfs_path[i][0], dfs_path[i][1], dx*0.3, dy*0.3, 
                            head_width=0.1, head_length=0.05, fc='darkred', ec='darkred', alpha=0.8)
            
            # Add step numbers at key points (start, every 10th step, and end)
            step_interval = max(1, len(dfs_path) // 10)  # Show about 10 numbers total
            for i in range(0, len(dfs_path), step_interval):
                ax.annotate(str(i), (dfs_path[i][0], dfs_path[i][1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='black', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Always show the final step number
            if len(dfs_path) > 1:
                final_idx = len(dfs_path) - 1
                ax.annotate(str(final_idx), (dfs_path[final_idx][0], dfs_path[final_idx][1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='black', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Mark start and end points
        if len(dfs_path) > 0:
            ax.scatter(dfs_path[0][0], dfs_path[0][1], c='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(dfs_path[-1][0], dfs_path[-1][1], c='red', s=100, marker='s', label='End', zorder=5)

        ax.legend()
        ax.set_title(f'DFS Robot Path ({len(dfs_path)} waypoints)')
        ax.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)

    maze_size = [3, 3]
    num_mazes = 1000

    dataset = MazeDataset()
    cell_size = [0.75, 0.75]
    wall_thickness = 0.1
    dataset.generate_mazes(maze_size, cell_size, num_mazes, wall_thickness)
    
    dataset.save_dataset('dataset/maze_dataset.json')

    world_coordinates = dataset.get_dfs_order(dataset[0], start_cell=(0, 0))
    dataset.visualize_dfs_path(dataset[0], world_coordinates)