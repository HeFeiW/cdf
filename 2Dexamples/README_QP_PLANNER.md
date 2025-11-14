# QP Motion Planning for 2D Robots with CDF

## Overview

This module implements Quadratic Programming (QP) based motion planning for 2D robots using Configuration Distance Fields (CDF). The planner enables obstacle avoidance and target reaching in configuration space.

## Files

- **qp_mp_tao_2d.py**: Main QP planner implementation for 2D robots
- **test_qp_planner.py**: Test script with examples
- **example.py**: Integrated visualization examples (updated)

## Key Features

- **QP-based planning**: Formulates motion planning as a quadratic program
- **CDF integration**: Uses trained neural network CDF models for distance computation
- **Obstacle avoidance**: Maintains safety distance from obstacles
- **Target reaching**: Attractive forces toward goal configurations
- **Real-time capable**: Efficient computation for online planning

## Architecture

### QPPlanner2D Class

Main planner class that implements the QP-based motion planning algorithm.

```python
planner = QPPlanner2D(robot, cdf_model, dt=0.01, cons_u=2.7, 
                      solver='ipopt', safety_buffer=0.01, device=device)
```

**Parameters:**
- `robot`: Robot2D instance defining the robot kinematics
- `cdf_model`: Trained MLP model for CDF inference
- `dt`: Time step for integration (default: 0.01)
- `cons_u`: Control input constraint limit (default: 2.7)
- `solver`: QP solver type ('ipopt', 'osqp', 'qpOASES', 'qrqp')
- `safety_buffer`: Safety distance buffer from obstacles (default: 0.01)
- `device`: PyTorch device (cuda/cpu)

**Main Method:**
```python
q_next = planner.step(q_current, obs_objs, targ_objs)
```

### Optimization Problem

The QP planner solves the following optimization at each step:

**Objective Function:**
```
min  1/2 u^T H u + h^T u
```

Where:
- `H = (B^T * grad_targ^T * grad_targ * B) * dt^2 + R`
- `h = 2 * B^T * grad_targ^T * dist_targ * dt`
- `R`: Control cost matrix (diagonal)

**Constraints:**
```
-grad_obs * B * u * dt <= dist_obs - safety_buffer
-cons_u <= u <= cons_u
```

**System Dynamics:**
```
q_{k+1} = A * q_k + B * u_k
```

For single integrator: `A = I`, `B = dt * I`

## Usage Examples

### Basic Usage

```python
import torch
from robot2D_torch import Robot2D
from primitives2D_torch import Circle, Box
from qp_mp_tao_2d import build_planner_2d
from mlp import MLPRegression

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create robot
link_lengths = torch.tensor([[2.0, 2.0]]).float()
robot = Robot2D(num_links=2, init_states=torch.zeros((1, 2)),
                link_lengths=link_lengths, device=device)

# Load or create CDF model
cdf_model = MLPRegression(input_dims=4, output_dims=1, 
                          mlp_layers=[128, 64, 32],
                          skips=[], act_fn=torch.nn.ReLU, nerf=True).to(device)

# Create planner
planner = build_planner_2d(robot, cdf_model, device, dt=0.05, cons_u=1.5)

# Define obstacles and targets
obs_objs = [Circle(center=torch.tensor([1.5, 1.0]).to(device), 
                   radius=0.3, attract=False, device=device)]
targ_objs = [Circle(center=torch.tensor([2.5, 0.0]).to(device), 
                    radius=0.2, attract=True, device=device)]

# Plan trajectory
q_current = torch.tensor([0.0, 0.0]).to(device)
trajectory = []

for step in range(100):
    q_next = planner.step(q_current, obs_objs, targ_objs)
    trajectory.append(q_next.cpu().numpy())
    
    # Check convergence
    if torch.norm(q_next - q_current) < 1e-3:
        break
    
    q_current = q_next
```

### Integration with example.py

The planner is integrated into `example.py` with visualization:

```python
from cdf import CDF2D
from qp_mp_tao_2d import build_planner_2d

# Create CDF instance
cdf = CDF2D(device)

# Load trained CDF model
cdf_model = torch.load('model/model.pth').to(device)

# Create planner
planner = build_planner_2d(cdf.robot, cdf_model, device)

# Plan and visualize
q_trajectory = plot_qp_planning(obs_objs + targ_objs, 'test_qp')
```

## Running Tests

### Standalone Test:
```bash
cd /home/hefei/cdf/2Dexamples
python test_qp_planner.py
```

This runs two tests:
1. Simple test with basic functionality
2. Full integration test with CDF visualization

### Integrated Test in example.py:
```bash
python example.py
```

The main script now includes QP planning test at the end.

## Visualization

The planner generates three types of visualizations:

1. **Configuration Space**: Shows the planned trajectory overlaid on the CDF
2. **Task Space**: Shows the robot motion in Cartesian space with obstacles
3. **Joint Angles**: Time series plot of joint angles

Output images are saved to `image/` directory:
- `test_qp_simple.png`: Basic test results
- `test_qp_with_cdf.png`: Full integration test results
- `test_qp_qp_planning.png`: Main example results

## Dependencies

- PyTorch
- CasADi (for QP solving)
- NumPy
- Matplotlib

Required files from the repository:
- `robot2D_torch.py`: 2D robot kinematics
- `primitives2D_torch.py`: Geometric primitives (Circle, Box)
- `mlp.py`: MLP model definition
- `cdf.py`: CDF class (for integration)

## CDF Model Requirements

The CDF model should be a trained MLP that takes input:
- `[p_x, p_y, q_1, q_2, ..., q_n]` where `p` is a point in task space, `q` is joint configuration

And outputs:
- `distance`: CDF distance value (scalar)

The gradient is computed automatically via PyTorch autograd.

## Solver Options

The planner supports multiple QP solvers:

- **ipopt** (default): Interior point optimizer, robust for nonlinear problems
- **osqp**: Fast for convex QP, good for real-time
- **qpOASES**: Efficient for small to medium problems
- **qrqp**: QR-based QP solver

Choose solver based on your speed/accuracy requirements.

## Parameters Tuning

### Key parameters to tune:

- **dt**: Smaller values → smoother trajectories, more steps
- **cons_u**: Larger values → faster motion, less smooth
- **safety_buffer**: Larger values → more conservative, safer
- **cost_mat_R**: Diagonal values control smoothness vs. speed

### Typical values:

```python
# Conservative (safe, slow)
planner = build_planner_2d(robot, model, device, 
                           dt=0.01, cons_u=0.5, safety_buffer=0.1)

# Balanced (default)
planner = build_planner_2d(robot, model, device, 
                           dt=0.05, cons_u=1.5, safety_buffer=0.05)

# Aggressive (fast, less safe)
planner = build_planner_2d(robot, model, device, 
                           dt=0.1, cons_u=3.0, safety_buffer=0.01)
```

## Comparison with Original 3D Version

This 2D implementation is adapted from `qp_mp_tao.py` for 3D robots:

| Feature | 3D Version (qp_mp_tao.py) | 2D Version (qp_mp_tao_2d.py) |
|---------|---------------------------|------------------------------|
| Robot | ParallelRobotLayer (multi-finger) | Robot2D (single chain) |
| CDF | CDF class with robot layers | Direct MLP inference |
| Input | Point clouds (N, 3) | Object surfaces (N, 2) |
| Model | Per-finger MLP models | Single MLP model |
| Output | Full robot joint state | Single chain joint state |

## Future Improvements

Possible enhancements:
- [ ] Multi-finger/multi-chain support for 2D
- [ ] Adaptive safety buffer based on velocity
- [ ] Time-varying obstacles
- [ ] Path optimization (smoothing)
- [ ] Online CDF model updates
- [ ] MPC-style receding horizon

## Troubleshooting

### Common Issues:

1. **Solver fails to converge**: 
   - Try different solver (ipopt is most robust)
   - Increase `safety_buffer`
   - Decrease `cons_u`

2. **Robot moves too slowly**:
   - Increase `dt` or `cons_u`
   - Check target gradient magnitude

3. **Oscillating behavior**:
   - Increase control cost (diagonal of `cost_mat_R`)
   - Decrease `dt`

4. **Collision with obstacles**:
   - Increase `safety_buffer`
   - Check CDF model accuracy
   - Verify obstacle definitions

## References

Based on the QP motion planning approach from `qp_mp_tao.py` for multi-finger robots, adapted for 2D planar manipulators with CDF-based distance fields.

## Contact

For issues or questions, please refer to the main CDF project documentation.
