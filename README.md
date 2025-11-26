# cudaPRM: GPU-Accelerated Perception-Aware Probabilistic Roadmap Planner

A high-performance CUDA-based Probabilistic Roadmap (PRM) motion planner with integrated perception scoring for mobile robot navigation. This library enables real-time path planning that optimizes both motion efficiency and camera visibility of objects of interest.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Experiments](#experiments)
- [ROS Integration](#ros-integration)
- [Citation](#citation)

## Overview

cudaPRM (CUDA Parallel PRM) is a GPU-accelerated motion planning framework designed for perception-aware robot navigation. It constructs probabilistic roadmaps using massively parallel CUDA kernels and integrates a neural network-based perception scoring model to generate paths that:

1. **Avoid collisions** with obstacles in 2D environments
2. **Maximize visibility** of objects of interest through learned perception scores
3. **Generate smooth trajectories** using Reeds-Shepp curves for non-holonomic robots

The planner is specifically designed for the **Hello Robot Stretch** mobile manipulation platform but can be adapted for other robots with pan-tilt camera heads.

## Features

- **GPU-Accelerated PRM Construction**: Parallel node sampling, neighbor finding, and collision checking on CUDA
- **Perception-Aware Planning**: Neural network model predicts camera perception quality scores
- **Reeds-Shepp Local Planning**: Supports non-holonomic motion constraints for car-like robots
- **Real-time Performance**: Sub-100ms planning times for replanning scenarios
- **ROS2 Integration**: WebSocket bridge for real-time robot control via ROSBridge
- **Isaac Sim Support**: Integration with NVIDIA Isaac Sim for simulation experiments
- **Flexible Environment Loading**: YAML-based scene configuration with mesh support

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Python API                               │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │   PSPRM     │  │   Solution   │  │   EnvironmentLoader    │  │
│  │  (prm.py)   │  │  (prm.py)    │  │   (EnvLoader.py)       │  │
│  └──────┬──────┘  └──────┬───────┘  └────────────────────────┘  │
├─────────┼────────────────┼───────────────────────────────────────┤
│         │                │              Neural Network            │
│         │                │    ┌─────────────────────────────┐    │
│         │                │    │  PercScoreProxyNet          │    │
│         │                │    │  (percscorenn.py)           │    │
│         │                │    └─────────────────────────────┘    │
├─────────┼────────────────┼───────────────────────────────────────┤
│         │    CUDA Kernels (cuPRM Extension)                      │
│  ┌──────┴──────────────────────────────────────────────────────┐ │
│  │  ┌────────────────┐  ┌───────────────┐  ┌────────────────┐  │ │
│  │  │  construction  │  │  collision    │  │  reedsshepp    │  │ │
│  │  │  .cu/.cuh      │  │  cc_2D.cu     │  │  .cu/.cuh      │  │ │
│  │  │  - Sampling    │  │  - Circle CC  │  │  - Path types  │  │ │
│  │  │  - K-NN        │  │  - Rect CC    │  │  - Interpolate │  │ │
│  │  │  - Edges       │  │  - Edge CC    │  │                │  │ │
│  │  └────────────────┘  └───────────────┘  └────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- CUDA Toolkit 11.8+ (tested with CUDA 12.x)
- Python 3.8+
- PyTorch with CUDA support
- C++17 compatible compiler

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/l-bashaw/cudaPRM.git
   cd cudaPRM
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the CUDA extension:**
   ```bash
   cd planning
   python setup.py build_ext --inplace
   ```

   This compiles the `cuPRM` CUDA extension module with:
   - Device-linked CUDA kernels for cross-file device calls
   - Auto-detected GPU compute capabilities
   - Optimized compilation flags (`-O3`, `--use_fast_math`)

### Requirements

```
trimesh
numpy
torch
matplotlib
rsplan
pytorch-kinematics
lightning
pybind11
packaging
```

For simulation experiments:
- NVIDIA Isaac Sim (optional)
- ROSBridge server (for real robot deployment)

## Quick Start

```python
import torch
import numpy as np
from planning.prm import PSPRM, Solution
from planning.nn.inference import ModelLoader
from planning.utils.EnvLoader import EnvironmentLoader

# Configuration
device = 'cuda'
env_config_file = "planning/resources/scenes/environment/multigoal_demo.yaml"
model_path = "planning/resources/models/percscore-nov12-50k.pt"

# Define start and goal (x, y, theta, pan, tilt)
start = np.array([1, -3, np.pi/2, 0.0, 0.0])
goal = np.array([1, 2.5, np.pi/2, 0.0, 0.0])

# Load environment from YAML
env_loader = EnvironmentLoader(device=device)
env = env_loader.load_world(env_config_file)

# Configure environment bounds and object of interest
env['bounds'] = torch.concat([
    env['bounds'], 
    torch.tensor([[-3.14159, 0.0, 0.0], [3.14159, 0.0, 0.0]], device=device)
], dim=1)
env['object_pose'] = torch.tensor([0.2, -0.2, 1.65, 0, 0, 0, 1], device=device)
env['object_label'] = torch.tensor([1.0, 0.0, 0.0], device=device)  # "human" label

# Load perception scoring model
model_loader = ModelLoader(label_size=3, max_batch_size=10000, use_trt=False)
model = model_loader.load_model(model_path)

# Build PRM and plan path
prm = PSPRM(model, env)
prm.build_prm(seed=2387)

# Add start/goal and search
start_id, goal_id = prm.addStartAndGoal(start, goal)
path = prm.a_star_search(start_id=start_id, goal_id=goal_id, alpha=0.5, beta=0.2)

# Post-process solution
sol = Solution(path)
sol.simplify(prm, env, max_skip_dist=1.5)
trajectory = sol.generate_trajectory_rsplan(prm, turning_radius=1)
trajectory = sol.project_trajectory(env['object_pose'])

sol.print_path(prm.graph)
```

## API Reference

### PSPRM Class

The main planning class that builds and queries the probabilistic roadmap.

```python
class PSPRM:
    def __init__(self, model, env)
    def build_prm(self, seed: int) -> None
    def addStartAndGoal(self, start: np.ndarray, goal: np.ndarray) -> Tuple[int, int]
    def a_star_search(self, start_id: int, goal_id: int, alpha: float, beta: float) -> List[int]
```

**Parameters:**
- `model`: Trained perception scoring neural network
- `env`: Environment dictionary containing:
  - `bounds`: Tensor [2, 5] with lower/upper bounds for (x, y, theta, pan, tilt)
  - `circles`: Tensor [N, 3] with circular obstacles (x, y, radius)
  - `rectangles`: Tensor [M, 4] with rectangular obstacles (x, y, height, width)
  - `object_pose`: Tensor [7] with object of interest pose (x, y, z, qx, qy, qz, qw)
  - `object_label`: Tensor [3] one-hot label for object class

**A* Search Parameters:**
- `alpha`: Weight for perception score term (higher = prioritize visibility)
- `beta`: Weight for motion cost term (higher = shorter paths)

### Solution Class

Post-processing class for path simplification and trajectory generation.

```python
class Solution:
    def __init__(self, path: List[int])
    def simplify(self, prm: PSPRM, env: dict, max_skip_dist: float) -> None
    def generate_trajectory_rsplan(self, prm: PSPRM, turning_radius: float) -> np.ndarray
    def project_trajectory(self, object_pose_world: torch.Tensor) -> np.ndarray
```

### EnvironmentLoader Class

Utility for loading environments from YAML configuration files.

```python
class EnvironmentLoader:
    def __init__(self, device: str = 'cuda')
    def load_world(self, config_file: str) -> dict
    def visualize_environment(self, env: dict) -> None
```

### ModelLoader Class

Loads and optionally converts perception scoring models to TensorRT.

```python
class ModelLoader:
    def __init__(self, label_size: int = 3, max_batch_size: int = 10000, use_trt: bool = True)
    def load_model(self, model_path: str) -> nn.Module
```

## Project Structure

```
cudaPRM/
├── example_usage.py              # Basic usage example
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
└── planning/
    ├── prm.py                    # Main PSPRM and Solution classes
    ├── setup.py                  # CUDA extension build script
    │
    ├── nn/                       # Neural network modules
    │   ├── inference.py          # Model loading utilities
    │   └── percscorenn.py        # PercScoreProxyNet architecture
    │
    ├── utils/                    # Utility modules
    │   ├── EnvLoader.py          # YAML environment loader
    │   ├── BatchFK.py            # Batched forward kinematics
    │   └── SimUtils.py           # Isaac Sim utilities
    │
    ├── src/                      # CUDA source code
    │   ├── bindings/
    │   │   └── py_bind.cpp       # PyBind11 Python bindings
    │   │
    │   ├── planning/
    │   │   ├── pprm.cu           # Roadmap allocation/management
    │   │   └── construction.cu   # Node sampling, K-NN, edge generation
    │   │
    │   ├── collision/
    │   │   ├── cc_2D.cu          # 2D collision checking kernels
    │   │   └── env_2D.cuh        # Environment data structures
    │   │
    │   ├── local_planning/
    │   │   └── reedsshepp.cu     # Reeds-Shepp path computation
    │   │
    │   └── params/
    │       └── hyperparameters.cuh  # PRM hyperparameters
    │
    ├── ros_bridge/               # ROS2 integration
    │   ├── pose_receiver.py      # Subscribe to pose topics
    │   └── trajectory_publisher.py  # Publish trajectories
    │
    ├── experiments/              # Experiment scripts
    │   ├── sim/
    │   │   ├── comparison_exp.py # Benchmarking experiments
    │   │   ├── isaac_sim_static.py
    │   │   └── isaac_sim_dynamic.py
    │   └── real/
    │       └── replanning_real_robot.py
    │
    └── resources/                # Configuration files
        ├── models/               # Trained neural network weights
        │   ├── percscore-nov12-50k.pt
        │   └── percscore-nov12-50k.onnx
        │
        ├── robots/               # Robot configurations
        │   ├── stretch/          # Hello Robot Stretch
        │   ├── fetch/            # Fetch Mobile Manipulator
        │   └── ur5/              # UR5 variants
        │
        └── scenes/               # Environment configurations
            └── environment/      # YAML scene files
```

## Configuration

### Hyperparameters (hyperparameters.cuh)

Key PRM parameters that can be tuned:

```cpp
constexpr unsigned int K = 10;              // Number of nearest neighbors
constexpr unsigned int NUM_STATES = 1560;   // Number of roadmap nodes
constexpr unsigned int DIM = 5;             // State dimension (x, y, θ, pan, tilt)
constexpr unsigned int INTERP_STEPS = 10;   // Edge interpolation steps
constexpr float R_TURNING = 1;              // Turning radius for Reeds-Shepp
constexpr float R_ROBOT = 0.1f;             // Robot collision radius
```

### Environment YAML Format

```yaml
world:
  collision_objects:
    - id: obstacle_name
      mesh_poses:
        - orientation: [qx, qy, qz, qw]
          position: [x, y, z]
      meshes:
        - dimensions: [scale_x, scale_y, scale_z]
          resource: path/to/mesh.obj
```

### Object Labels

The perception model supports object classification with one-hot encoded labels:

```python
label_map = {
    "human":   [1, 0, 0],
    "monitor": [0, 1, 0],
    "cup":     [0, 0, 1]
}
```

## Experiments

### Simulation Experiments

Run benchmark comparisons in Isaac Sim:

```bash
cd planning/experiments/sim
python comparison_exp.py
```

### Real Robot Deployment

For real robot experiments with ROS2:

1. Start ROSBridge server:
   ```bash
   ros2 launch rosbridge_server rosbridge_websocket_launch.xml
   ```

2. Run replanning controller:
   ```bash
   cd planning/experiments/real
   python replanning_real_robot.py
   ```

## ROS Integration

The planner communicates with ROS2 via WebSocket using ROSBridge:

- **Pose Receiver** (`ros_bridge/pose_receiver.py`): Subscribes to `/poses` and `/pose_names` topics for dynamic obstacle tracking
- **Trajectory Publisher** (`ros_bridge/trajectory_publisher.py`): Publishes planned trajectories to `/trajectory` as `Float32MultiArray`

## Performance

Typical performance on NVIDIA RTX 3090:

| Phase | Time |
|-------|------|
| PRM Construction | ~15-30 ms |
| Path Search + Simplification | ~5-10 ms |
| Trajectory Generation | ~2-5 ms |
| **Total** | **~25-50 ms** |

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{cudaprm2025,
  title={cudaPRM: GPU-Accelerated Perception-Aware Probabilistic Roadmap Planning},
  author={Bashaw, L.},
  year={2025},
  howpublished={\url{https://github.com/l-bashaw/cudaPRM}}
}
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

- Hello Robot for the Stretch platform
- NVIDIA for CUDA and Isaac Sim
- Rice University ELEC594 Capstone Project