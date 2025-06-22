import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Env:
    def __init__(self, maze):
        self.maze = maze
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def _get_observation(self):
        pass
        
    def _simulate_lidar(self, pose):
        pass

    def _get_walls(self):
        pass

    def _check_collision(self, pose):
        pass


if __name__ == "__main__":
    pass