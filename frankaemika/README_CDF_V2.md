# CDF V2 - Configuration Distance Field with Base DoF Support

This is an enhanced version of the CDF model that supports:
1. **Base DoF**: 6DoF base pose (3 translation + 3 rotation) in addition to joint angles
2. **Multiple Network Types**: Choice between MLP and SIREN networks
3. **Multi-finger Training**: Support for training multiple serial chains (e.g., all 4 fingers of LeapHand)

## Key Features

### Base DoF Support
- Each finger can have a 6DoF base pose: `[tx, ty, tz, rx, ry, rz]`
- Total configuration space: `(joint_dof + 6)` dimensions
- Input to network: `[x, y, z, q1, q2, ..., qn, tx, ty, tz, rx, ry, rz]`

### Network Types
- **MLP**: Multi-layer perceptron with ReLU activations
  - Layers: [1024, 512, 256, 128, 128]
  - Good for general-purpose learning
  
- **SIREN**: Sinusoidal representation networks
  - Layers: [256, 256, 256, 256]
  - Better for smooth implicit functions
  - Periodic activation functions

## Usage

### Training a Single Finger

#### With MLP (default):
```bash
python para_nn_cdf_v2.py \
    --train \
    --use_base \
    --robot leaphand \
    --serial_idx 0 \
    --network_type mlp \
    --data_path data_with_base_dof_0.pt \
    --model_dict leaphand_finger0_mlp_base.pt \
    --epoches 50000 \
    --with_writer
```

#### With SIREN:
```bash
python para_nn_cdf_v2.py \
    --train \
    --use_base \
    --robot leaphand \
    --serial_idx 0 \
    --network_type siren \
    --data_path data_with_base_dof_0.pt \
    --model_dict leaphand_finger0_siren_base.pt \
    --epoches 50000 \
    --with_writer
```

### Training All 4 Fingers

Use the provided bash script:

```bash
# Train all fingers with MLP
bash train_leaphand_all_fingers.sh

# Train all fingers with SIREN
NETWORK_TYPE=siren bash train_leaphand_all_fingers.sh

# Customize training parameters
NETWORK_TYPE=mlp EPOCHS=30000 BATCH_X=20 BATCH_Q=200 bash train_leaphand_all_fingers.sh
```

### Evaluation

```bash
python para_nn_cdf_v2.py \
    --eval \
    --use_base \
    --robot leaphand \
    --serial_idx 0 \
    --network_type mlp \
    --model_dict leaphand_finger0_mlp_base.pt
```

## Arguments

### Required Arguments
- `--robot`: Robot type (`panda`, `leaphand`, `dexhand`)
- `--serial_idx`: Index of the serial chain (finger) to train (0-3 for leaphand)

### Optional Arguments
- `--network_type`: Network architecture (`mlp` or `siren`, default: `mlp`)
- `--use_base`: Enable 6DoF base support (flag)
- `--train`: Enable training mode (flag)
- `--eval`: Enable evaluation mode (flag)
- `--epoches`: Number of training epochs (default: 50000)
- `--batch_x`: Batch size for spatial points (default: 10)
- `--batch_q`: Batch size for configurations (default: 100)
- `--max_q_per_link`: Max q samples per link (default: 100)
- `--signed_distance`: Use signed distance field (flag)
- `--with_writer`: Enable TensorBoard logging (flag)
- `--data_path`: Path to training data file
- `--model_dict`: Path to save/load model weights

## Data Format

### Without Base DoF
Data structure:
```python
{
    'x': (G, 3),           # Grid points in Cartesian space
    'q': (G, M, DoF, DoF), # Configuration manifolds
    'k': (G,)              # Grid indices
}
```

### With Base DoF
Data structure:
```python
{
    'x': (G, 3),                # Grid points in Cartesian space
    'q': (G, M, DoF+6, DoF+6),  # Configuration manifolds with base
    'k': (G,)                   # Grid indices
}
```

Where:
- `G`: Number of grid points
- `M`: Max samples per link (max_q_per_link)
- `DoF`: Joint degrees of freedom

## LeapHand Configuration

For LeapHand robot:
- **4 fingers** (serial chains)
- **Each finger**:
  - 4 joint DoF
  - 6 base DoF (optional)
  - Total: 10 DoF per finger when base is enabled

### Finger Indices
- 0: Index finger
- 1: Middle finger
- 2: Ring finger  
- 3: Thumb

## Data Generation

Before training, generate data with base DoF:

```python
from parallel_data_generator import DataGenerator

# Generate data for finger 0 with base DoF
gen = DataGenerator(device, parallel_robot, paths, serial_idx=0)
gen.generate_offline_data(serial_idx=0)
# This creates: data_with_base_dof_0.npy
```

Process the raw data:
```python
from para_nn_cdf_v2 import CDF_V2

cdf = CDF_V2(device, paths, robot='leaphand', use_base=True, serial_idx=0)
# Will automatically process .npy to .pt format
```

## Network Input/Output

### Input Dimensions
- Without base: `3 + DoF` (e.g., 3 + 4 = 7 for leaphand finger)
- With base: `3 + DoF + 6` (e.g., 3 + 4 + 6 = 13 for leaphand finger)

### Output
- Single scalar value: signed distance to configuration manifold

### Forward Pass
```python
# x: (B, 3) - Cartesian positions
# q: (B, DoF+6) - Configurations with base
# output: (B, 1) - Distance values

inputs = torch.cat([x, q], dim=-1)  # (B, 13)
if network_type == 'siren':
    distance, _ = model(inputs)
else:
    distance = model(inputs)
```

## Loss Functions

The model is trained with multiple loss components:

1. **Distance Loss** (weight: 5.0)
   - MSE between predicted and ground truth distances

2. **Eikonal Loss** (weight: 0.01)
   - Enforces unit gradient norm: `|∇_q d| = 1`

3. **Tension Loss** (weight: 0.01)
   - Smoothness regularization: `||∇²_q d||²`

4. **Gradient Loss** (weight: 0.1)
   - Cosine similarity with ground truth gradients

## Differences from Original para_nn_cdf.py

1. **Base DoF Support**
   - Added `use_base` parameter
   - Extended configuration space by 6 dimensions
   - Modified `sample_q()`, `compute_sdf()`, `decode_distance()`

2. **Network Type Selection**
   - Added `network_type` parameter
   - Support for both MLP and SIREN
   - Factory method `create_network()`

3. **Pose Handling**
   - Integration with `utils.q_to_poseMatrix()` for base pose
   - Proper splitting of joint angles and base parameters

4. **Enhanced Training**
   - Better logging and visualization
   - More flexible hyperparameters
   - TensorBoard integration with run naming

## TensorBoard Monitoring

When training with `--with_writer`:
```bash
tensorboard --logdir=runs/
```

Logged metrics:
- `loss/d_loss`: Distance prediction loss
- `loss/eikonal_loss`: Gradient norm constraint
- `loss/tension_loss`: Smoothness regularization
- `loss/gradient_loss`: Gradient direction accuracy
- `loss/total_loss`: Combined weighted loss

## Example Workflow

1. **Generate data for all fingers**:
```bash
python parallel_data_generator.py --robot leaphand
```

2. **Train all fingers**:
```bash
# With MLP
bash train_leaphand_all_fingers.sh

# With SIREN
NETWORK_TYPE=siren bash train_leaphand_all_fingers.sh
```

3. **Monitor training**:
```bash
tensorboard --logdir=runs/
```

4. **Evaluate models**:
```bash
for i in {0..3}; do
    python para_nn_cdf_v2.py \
        --eval \
        --use_base \
        --robot leaphand \
        --serial_idx $i \
        --network_type mlp \
        --model_dict leaphand_finger${i}_mlp_base.pt
done
```

## Tips

1. **SIREN vs MLP**:
   - Use SIREN for smoother distance fields
   - Use MLP for faster training and inference
   - SIREN requires careful initialization (already handled)

2. **Base DoF Limits**:
   - Define `theta_min_base` and `theta_max_base` in your robot model
   - Default: `[-0.5, -0.5, -0.5, -π, -π, -π]` to `[0.5, 0.5, 0.5, π, π, π]`

3. **Training Time**:
   - With base: ~2x longer due to higher dimensionality
   - Recommended epochs: 30000-50000
   - Use GPU for faster training

4. **Memory**:
   - Higher batch sizes may cause OOM
   - Reduce `batch_x` and `batch_q` if needed
   - Enable mixed precision (already enabled via AMP)

## Citation

Based on the CDF framework from:
- Original CDF paper and code
- RDF (Robot Distance Field) implementation
- SIREN: Implicit Neural Representations with Periodic Activation Functions

## License

MIT License - See LICENSE file for details
