#!/usr/bin/env python
"""
Quick comparison and testing script for para_nn_cdf_v2.py
"""

import torch
import sys
import os

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_PATH)

def compare_dimensions():
    """Compare input dimensions with/without base"""
    print("=" * 60)
    print("Dimension Comparison")
    print("=" * 60)
    
    robot_configs = {
        'panda': {'dof': 7, 'fingers': 1},
        'leaphand': {'dof': 4, 'fingers': 4},
        'dexhand': {'dof': 4, 'fingers': 5}
    }
    
    for robot, config in robot_configs.items():
        dof = config['dof']
        fingers = config['fingers']
        
        print(f"\n{robot.upper()}:")
        print(f"  Fingers: {fingers}")
        print(f"  Joint DoF per finger: {dof}")
        print(f"  Input dims without base: 3 + {dof} = {3 + dof}")
        print(f"  Input dims with base: 3 + {dof} + 6 = {3 + dof + 6}")
        if fingers > 1:
            print(f"  Total configs to train: {fingers}")

def test_network_types():
    """Test both MLP and SIREN network creation"""
    print("\n" + "=" * 60)
    print("Network Architecture Comparison")
    print("=" * 60)
    
    from mlp import MLPRegression
    sys.path.append(os.path.join(CUR_PATH, '../../RDF'))
    from Siren import Siren
    
    input_dims = 13  # 3 + 4 + 6 for leaphand with base
    
    # MLP
    print("\nMLP Network:")
    mlp = MLPRegression(
        input_dims=input_dims,
        output_dims=1,
        mlp_layers=[1024, 512, 256, 128, 128],
        skips=[],
        act_fn=torch.nn.ReLU,
        nerf=True
    )
    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"  Layers: [1024, 512, 256, 128, 128]")
    print(f"  Activation: ReLU")
    print(f"  Total parameters: {mlp_params:,}")
    
    # SIREN
    print("\nSIREN Network:")
    siren = Siren(
        in_features=input_dims,
        out_features=1,
        hidden_features=256,
        hidden_layers=3,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30
    )
    siren_params = sum(p.numel() for p in siren.parameters())
    print(f"  Layers: [256, 256, 256, 256]")
    print(f"  Activation: Sine")
    print(f"  Total parameters: {siren_params:,}")
    
    print(f"\nParameter ratio: {siren_params / mlp_params:.2f}x")

def test_base_pose_conversion():
    """Test base pose to matrix conversion"""
    print("\n" + "=" * 60)
    print("Base Pose Conversion Test")
    print("=" * 60)
    
    sys.path.append(os.path.join(CUR_PATH, '../../RDF'))
    import utils
    
    # Create sample base pose
    q_base = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.5]])
    print(f"\nBase pose (6DoF): {q_base}")
    print(f"  Translation: [{q_base[0, 0]:.2f}, {q_base[0, 1]:.2f}, {q_base[0, 2]:.2f}]")
    print(f"  Rotation (Euler): [{q_base[0, 3]:.2f}, {q_base[0, 4]:.2f}, {q_base[0, 5]:.2f}]")
    
    # Convert to 4x4 matrix
    pose_matrix = utils.q_to_poseMatrix(None, q_base)
    print(f"\n4x4 Transformation Matrix:")
    print(pose_matrix[0])

def show_training_commands():
    """Show example training commands"""
    print("\n" + "=" * 60)
    print("Example Training Commands")
    print("=" * 60)
    
    commands = [
        ("Single finger with MLP", 
         "python para_nn_cdf_v2.py --train --use_base --robot leaphand --serial_idx 0 --network_type mlp"),
        
        ("Single finger with SIREN",
         "python para_nn_cdf_v2.py --train --use_base --robot leaphand --serial_idx 0 --network_type siren"),
        
        ("All fingers with script",
         "bash train_leaphand_all_fingers.sh"),
        
        ("Evaluation",
         "python para_nn_cdf_v2.py --eval --use_base --robot leaphand --serial_idx 0 --network_type mlp"),
    ]
    
    for i, (desc, cmd) in enumerate(commands, 1):
        print(f"\n{i}. {desc}:")
        print(f"   {cmd}")

def check_data_files():
    """Check if required data files exist"""
    print("\n" + "=" * 60)
    print("Data File Check")
    print("=" * 60)
    
    data_dir = os.path.join(CUR_PATH, 'data', 'leaphand')
    
    required_files = [
        f'data_with_base_dof_{i}.npy' for i in range(4)
    ] + [
        f'data_with_base_dof_{i}.pt' for i in range(4)
    ]
    
    print(f"\nChecking directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"  ❌ Directory does not exist")
        print(f"  Create it with: mkdir -p {data_dir}")
    else:
        print(f"  ✅ Directory exists")
        
        for filename in required_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"  ✅ {filename} ({size:.2f} MB)")
            else:
                print(f"  ❌ {filename} (missing)")

def show_differences():
    """Show key differences from original version"""
    print("\n" + "=" * 60)
    print("Key Differences from para_nn_cdf.py")
    print("=" * 60)
    
    differences = [
        ("Base DoF Support", 
         "Original: Fixed identity pose",
         "V2: Optional 6DoF base pose"),
        
        ("Network Types",
         "Original: MLP only",
         "V2: MLP or SIREN"),
        
        ("Input Dimensions",
         "Original: 3 + DoF",
         "V2: 3 + DoF + (6 if use_base)"),
        
        ("Configuration Space",
         "Original: Joint space only",
         "V2: Joint + base space"),
        
        ("Training Scripts",
         "Original: Manual per-finger",
         "V2: Automated bash script"),
        
        ("Documentation",
         "Original: Minimal",
         "V2: Complete with examples"),
    ]
    
    for feature, original, v2 in differences:
        print(f"\n{feature}:")
        print(f"  Original: {original}")
        print(f"  V2:       {v2}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CDF V2 - Comparison and Testing")
    print("=" * 60)
    
    compare_dimensions()
    test_network_types()
    test_base_pose_conversion()
    check_data_files()
    show_differences()
    show_training_commands()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60 + "\n")
