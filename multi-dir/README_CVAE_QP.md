# CVAE-based QP Motion Planning

Complete implementation of motion planning using CVAE-sampled goals with QP optimization and CDF-based obstacle avoidance.

## Files

- **`qp_mp_cvae.py`**: Main CVAE-based QP planner implementation
- **`plot_utils.py`**: Plotting utilities for visualization
- **`CVAE.py`**: CVAE model
- **`cdf_dataloader.py`**: Data loader for CVAE training
- **`q_CVAE.py`**: CVAE training script

## Workflow

### 1. Train CVAE Model

First, generate training data and train the CVAE:

```bash
cd /home/hefei/cdf/multi-dir

# Train CVAE (make sure data22.pt exists in ../2Dexamples/)
python q_CVAE.py \
    --data_path ../2Dexamples/data22.pt \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-3 \
    --save_path ./checkpoints/cvae_model.pth
```

### 2. Run Motion Planning

Use the trained CVAE model for motion planning:

```bash
# Test with scene 4 (default)
python qp_mp_cvae.py \
    --cvae_model ./checkpoints/cvae_model.pth \
    --cdf_model ../2Dexamples/model_dict/siren_model22.pth \
    --scene scene4 \
    --max_steps 300 \
    --rounds 1

# Test with scene 5
python qp_mp_cvae.py \
    --cvae_model ./checkpoints/cvae_model.pth \
    --cdf_model ../2Dexamples/model_dict/siren_model22.pth \
    --scene scene5 \
    --max_steps 300

# With custom parameters
python qp_mp_cvae.py \
    --cvae_model ./checkpoints/cvae_model.pth \
    --cdf_model ../2Dexamples/model_dict/siren_model22.pth \
    --scene scene4 \
    --max_steps 300 \
    --dt 0.05 \
    --cons_u 1.0 \
    --safety_buffer 0.1 \
    --n_cvae_samples 5 \
    --resample_interval 5
```

## Command-line Arguments

### Motion Planning (`qp_mp_cvae.py`)

- `--cvae_model`: Path to trained CVAE model (default: `./checkpoints/cvae_model.pth`)
- `--cdf_model`: Path to trained CDF model (default: `../2Dexamples/model_dict/siren_model22.pth`)
- `--scene`: Scene to test, choices: ['scene4', 'scene5'] (default: 'scene4')
- `--max_steps`: Maximum planning steps (default: 300)
- `--rounds`: Number of planning rounds (default: 1)
- `--dt`: Time step for dynamics (default: 0.05)
- `--cons_u`: Control input constraint (default: 1.0)
- `--safety_buffer`: Safety buffer distance for obstacles (default: 0.1)
- `--n_cvae_samples`: Number of samples from CVAE per step (default: 5)
- `--resample_interval`: How often to resample goals from CVAE (default: 5)

## Test Scenes

### Scene 4
- **Obstacles**: Two boxes at positions [0.75, -1.5] and [0.75, -2.5]
- **Targets**: Two boxes at positions [1.25, -1.5] and [1.25, -2.5]
- **Start config**: [-π + 0.1, 2.2]
- **Target point**: [1.25, -2.0]

### Scene 5
- **Obstacles**: One box at position [2.0, 2.0]
- **Targets**: One circle at position [0.0, -2.25]
- **Start config**: [0.5, 0.5]
- **Target point**: [0.0, -2.25]

## Output

The planner generates the following visualizations in `../2Dexamples/image/`:

1. **`{scene}_cvae_qp_full.png`**: Complete planning results with 3 subplots:
   - Configuration space trajectory with CDF
   - Task space trajectory with robot
   - Joint angles over time

2. **`{scene}_cvae_qp_cvae_goals.png`**: Visualization showing:
   - CVAE-sampled goals (blue X markers)
   - Arrows from trajectory to goals
   - Actual trajectory followed

## Algorithm Overview

### CVAE-based Goal Sampling

At each time step (or every `resample_interval` steps):

1. **Sample from CVAE**: Given current config `q_current` and target point `x_goal`, sample multiple contact configurations:
   ```
   condition = [q_current, x_goal]
   delta_q ~ CVAE(condition)
   q_goal = q_current + delta_q
   ```

2. **Select best goal**: Choose the closest sampled goal to current configuration

### QP Optimization

Solve the following QP at each step:

```
min  1/2 (q_{k+1} - q_goal)^T Q (q_{k+1} - q_goal) + 1/2 u^T R u

s.t. q_{k+1} = A q_k + B u_k
     -∇_q f_c(p_obs, q_k) B u_k Δt ≤ ln(f_c(p_obs, q_k) + 1 - γ)
     -cons_u ≤ u ≤ cons_u
     cons_x[0] ≤ q_{k+1} ≤ cons_x[1]
```

where:
- `q_goal`: sampled from CVAE
- `f_c`: CDF distance to obstacles
- `γ`: safety buffer

## Comparison with Baseline

| Method | Goal Generation | Optimization | Adaptability |
|--------|----------------|--------------|--------------|
| **Original (qp_mp_tao_2d.py)** | Fixed workspace point | Minimize CDF distance | Limited to pre-specified targets |
| **CVAE-based (qp_mp_cvae.py)** | CVAE sampling | Track sampled config | Learns from data, handles complex distributions |

## Troubleshooting

### Model not found
Make sure you've trained the CVAE model first:
```bash
python q_CVAE.py --data_path ../2Dexamples/data22.pt --epochs 100
```

### Poor planning results
Try adjusting:
- `--n_cvae_samples`: Increase for better goal quality
- `--resample_interval`: Decrease for more adaptive behavior
- `--safety_buffer`: Increase if colliding with obstacles
- `--cons_u`: Increase for faster motion

### Solver failures
- Check that CDF model is properly trained
- Verify obstacle and target positions are valid
- Try different start configurations
