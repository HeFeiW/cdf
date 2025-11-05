# CDF V2 Quick Reference Card

## 🚀 Quick Start

### Training Single Finger
```bash
# MLP (faster, good general purpose)
python para_nn_cdf_v2.py --train --use_base --robot leaphand --serial_idx 0 --network_type mlp

# SIREN (smoother fields, better gradients)
python para_nn_cdf_v2.py --train --use_base --robot leaphand --serial_idx 0 --network_type siren
```

### Training All Fingers
```bash
# MLP for all fingers
bash train_leaphand_all_fingers.sh

# SIREN for all fingers
NETWORK_TYPE=siren bash train_leaphand_all_fingers.sh
```

## 📊 Dimensions

| Robot | DoF/Finger | With Base | Input Dims | Fingers |
|-------|-----------|-----------|------------|---------|
| Panda | 7 | 13 | 3+7 or 3+13 | 1 |
| LeapHand | 4 | 10 | 3+4 or 3+10 | 4 |
| DexHand | 4 | 10 | 3+4 or 3+10 | 5 |

**Input Format**: `[x, y, z, q1, q2, ..., qn, tx, ty, tz, rx, ry, rz]`
- First 3: Cartesian position
- Next n: Joint angles
- Last 6 (optional): Base pose

## 🎛️ Command Line Arguments

### Essential
```bash
--robot leaphand          # Robot type
--serial_idx 0            # Finger index (0-3)
--network_type mlp        # Network: mlp or siren
--use_base                # Enable base DoF
```

### Training
```bash
--train                   # Enable training
--epoches 50000          # Training epochs
--batch_x 10             # Spatial batch size
--batch_q 100            # Config batch size
```

### Data
```bash
--data_path data_with_base_dof_0.pt
--model_dict leaphand_finger0_mlp_base.pt
```

### Optional
```bash
--with_writer            # Enable TensorBoard
--signed_distance        # Use signed distance
--eval                   # Evaluation mode
```

## 🏗️ Network Architectures

### MLP
- **Layers**: [1024, 512, 256, 128, 128]
- **Activation**: ReLU
- **Parameters**: ~1.5M
- **Speed**: Fast ⚡
- **Use for**: General purpose, fast inference

### SIREN
- **Layers**: [256, 256, 256, 256]
- **Activation**: Sine (ω₀=30)
- **Parameters**: ~350K
- **Speed**: Medium
- **Use for**: Smooth fields, better gradients

## 📁 File Structure

```
cdf/frankaemika/
├── para_nn_cdf_v2.py                    # Main implementation
├── train_leaphand_all_fingers.sh        # Training script
├── README_CDF_V2.md                     # Full documentation
├── IMPLEMENTATION_SUMMARY.md            # Technical details
├── compare_versions.py                  # Testing script
├── data/
│   └── leaphand/
│       ├── data_with_base_dof_0.pt      # Finger 0 data
│       ├── data_with_base_dof_1.pt      # Finger 1 data
│       ├── data_with_base_dof_2.pt      # Finger 2 data
│       └── data_with_base_dof_3.pt      # Finger 3 data
└── model_dict/
    └── leaphand/
        ├── leaphand_finger0_mlp_base.pt
        ├── leaphand_finger0_siren_base.pt
        └── ...
```

## 🔄 Workflow

1. **Generate Data** (if needed)
   ```bash
   cd /home/hefei/cdf/frankaemika
   python parallel_data_generator.py --robot leaphand
   ```

2. **Train Models**
   ```bash
   # Option A: All fingers at once
   bash train_leaphand_all_fingers.sh
   
   # Option B: One finger at a time
   for i in {0..3}; do
       python para_nn_cdf_v2.py --train --use_base --robot leaphand --serial_idx $i
   done
   ```

3. **Monitor Training**
   ```bash
   tensorboard --logdir=runs/
   # Open: http://localhost:6006
   ```

4. **Evaluate**
   ```bash
   python para_nn_cdf_v2.py --eval --use_base --robot leaphand --serial_idx 0
   ```

## 📈 Loss Components

| Loss | Weight | Purpose |
|------|--------|---------|
| Distance | 5.0 | MSE between pred and GT |
| Eikonal | 0.01 | ‖∇d‖ = 1 constraint |
| Tension | 0.01 | Smoothness (‖∇²d‖²) |
| Gradient | 0.1 | Direction accuracy |

**Total**: `L = 5.0·L_d + 0.01·L_eik + 0.01·L_ten + 0.1·L_grad`

## 🎯 Expected Performance

### Training Time (per finger)
- **50k epochs**: 2-4 hours (with GPU)
- **MLP**: Slightly faster
- **SIREN**: Slightly slower

### Evaluation Metrics
- **MAE**: < 0.01 (good), < 0.005 (excellent)
- **RMSE**: < 0.02 (good), < 0.01 (excellent)
- **Success Rate** (SR): > 90% at 3cm threshold

## 🐛 Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch sizes
python para_nn_cdf_v2.py --train --batch_x 5 --batch_q 50
```

### Data not found
```bash
# Check data directory
ls data/leaphand/

# Regenerate if needed
python parallel_data_generator.py --robot leaphand
```

### Training not converging
- Check data quality
- Try different network type
- Adjust learning rate (edit code)
- Increase epochs

### SIREN NaN loss
- Usually auto-corrects
- Check data range
- Reduce learning rate

## 🔬 Code Snippets

### Load and Use Model
```python
from para_nn_cdf_v2 import CDF_V2

# Initialize
cdf = CDF_V2(device, paths, robot='leaphand', 
             network_type='mlp', use_base=True, serial_idx=0)

# Create and load model
model = cdf.create_network(input_dims=13)
model.load_state_dict(torch.load('model.pt')[epoch])

# Inference
x = torch.rand(1, 3)  # Query point
q = cdf.sample_q(100)  # Sample configs
d, grad = cdf.inference_d_wrt_q(x, q, model)
```

### Custom Training Loop
```python
# Sample data
x_batch, q_batch, d, grad = cdf.select_data()

# Forward pass
inputs = torch.cat([x_batch.expand(...), q_batch], dim=-1)
d_pred = model(inputs)

# Compute loss
loss = ((d_pred - d)**2).mean()
```

## 📚 Key Functions

| Function | Purpose |
|----------|---------|
| `create_network()` | MLP or SIREN factory |
| `sample_q()` | Random config sampling |
| `compute_sdf()` | Ground truth SDF |
| `decode_distance()` | CDF from q library |
| `inference()` | Model prediction |
| `train_nn()` | Training loop |

## 🎓 Best Practices

1. **Start Small**: Test with 1000 epochs first
2. **Monitor**: Always use `--with_writer`
3. **Save Often**: Models saved every 10 epochs
4. **Validate**: Check data files before training
5. **GPU**: Essential for reasonable training time
6. **Batch Sizes**: Start conservative, increase if stable

## 📞 Quick Debug

```bash
# Test installation
python compare_versions.py

# Check single forward pass
python -c "from para_nn_cdf_v2 import CDF_V2; print('OK')"

# Verify data
python -c "import torch; d=torch.load('data/leaphand/data_with_base_dof_0.pt'); print(d.keys())"

# Quick 100 epoch test
python para_nn_cdf_v2.py --train --use_base --robot leaphand --serial_idx 0 --epoches 100
```

## 🔗 Related Files

- **Original**: `para_nn_cdf.py` (without base support)
- **Data Gen**: `parallel_data_generator.py`
- **Robot Layer**: `../../RDF/panda_layers/parallel_robot_layer.py`
- **SDF**: `../../RDF/parallel_bf_sdf.py`
- **Utils**: `../../RDF/utils.py`

## 📋 Environment

```bash
# Required packages
torch >= 1.9
numpy
trimesh
tensorboard  # optional, for monitoring
```

---

**Created**: 2025
**Version**: 2.0
**License**: MIT
