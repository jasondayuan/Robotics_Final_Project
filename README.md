# Robotics SLAM Project: Maze Navigation and Mapping

This project is a full-stack implementation of the Simultaneous Localization and Mapping (SLAM) pipeline, developed entirely from scratch. It features a robot that can autonomously navigate and map an unknown, randomly generated maze environment.

## Features

* **Random Maze Generation**: Creates complex and unique mazes for the robot to explore using a randomized version of Kruskal's algorithm.
* **Simulated Environment**: A custom 2D environment that models the robot's differential drive kinematics, simulates LIDAR sensor data, and handles collision.
* **Corner Detection**: A RANSAC-based algorithm that extracts corner features from noisy LIDAR scans to use as landmarks.
* **FastSLAM Algorithm**: Implements the FastSLAM algorithm to enable the robot to build a map and localize itself within the map. The robot follows a path determined by a Depth-First Search (DFS) for systematic exploration.

## Code Structure

* `robot_model.py`: The main script to run the full simulation.
* `slam.py`: The FastSLAM algorithm implementation.
* `env.py`: Defines the simulated robot and maze environment.
* `map_generation.py`: Contains the maze generation logic.
* `corner_detector.py`: The RANSAC-based corner detection algorithm.
* `create_video.py`: A utility script to generate a video from simulation frames.
* `Robotics_Final_Project.pdf`: The detailed project report.

## Environment Setup

```
conda create -n robotics python=3.9
conda activate robotics
conda install numpy matplotlib tqdm
pip install opencv-python
```