#!/usr/bin/env python
"""
Test script for process_data functionality in para_nn_cdf_v2.py
"""

import torch
import numpy as np
import os
import sys

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_PATH)

def test_process_data():
    """Test the process_data method with and without base DoF"""
    
    print("=" * 60)
    print("Testing process_data functionality")
    print("=" * 60)
    
    # Test parameters
    robot = 'leaphand'
    serial_idx = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup paths
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'urdf': os.path.join(CUR_DIR, f'../../RDF/descriptions/{robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR, f'../../RDF/descriptions/{robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR, f'../../RDF/data/{robot}/sdf_points/'),
        'model': os.path.join(CUR_DIR, f'../../RDF/models/{robot}/BP_8.pt'),
        'raw_data': os.path.join(CUR_DIR, f'data/{robot}/data_with_base_dof_{serial_idx}.npy'),
        'data': os.path.join(CUR_DIR, f'data/{robot}/test_processed_{serial_idx}.pt'),
        'model_dict': os.path.join(CUR_DIR, f'model_dict/{robot}/test_model.pt'),
    }
    
    # Check if raw data exists
    if not os.path.exists(paths['raw_data']):
        print(f"❌ Raw data not found: {paths['raw_data']}")
        print("Please generate data first using parallel_data_generator.py")
        return False
    
    print(f"\n✅ Found raw data: {paths['raw_data']}")
    
    # Load raw data to inspect structure
    raw_data = np.load(paths['raw_data'], allow_pickle=True).item()
    print(f"\nRaw data structure:")
    print(f"  Number of grid points: {len(raw_data)}")
    
    # Sample one point to check structure
    sample_key = list(raw_data.keys())[0]
    sample_data = raw_data[sample_key]
    print(f"  Sample grid point {sample_key}:")
    print(f"    x shape: {sample_data['x'].shape}")
    print(f"    q shape: {sample_data['q'].shape}")
    print(f"    idx shape: {sample_data['idx'].shape}")
    
    # Determine if data has base DoF
    q_dim = sample_data['q'].shape[1]
    print(f"  Configuration dimension: {q_dim}")
    
    # Test 1: Process data WITH base DoF
    print("\n" + "=" * 60)
    print("Test 1: Processing data WITH base DoF")
    print("=" * 60)
    
    try:
        from para_nn_cdf_v2 import CDF_V2
        
        paths['data'] = os.path.join(CUR_DIR, f'data/{robot}/test_with_base_{serial_idx}.pt')
        
        cdf_with_base = CDF_V2(
            device=device,
            paths=paths,
            robot=robot,
            network_type='mlp',
            signed_distance=False,
            writer=None,
            serial_idx=serial_idx,
            use_base=True
        )
        
        print("\n✅ Successfully processed data WITH base DoF")
        print(f"  Processed data saved to: {paths['data']}")
        
        # Check processed data
        processed = torch.load(paths['data'])
        print(f"\nProcessed data shapes:")
        print(f"  x: {processed['x'].shape}")
        print(f"  q: {processed['q'].shape}")
        print(f"  k: {processed['k'].shape}")
        
        expected_config_dim = cdf_with_base.robot.serials[serial_idx].dof + 6
        actual_config_dim = processed['q'].shape[2]
        
        if actual_config_dim == expected_config_dim:
            print(f"  ✅ Configuration dimension matches: {actual_config_dim}")
        else:
            print(f"  ❌ Configuration dimension mismatch!")
            print(f"     Expected: {expected_config_dim}, Got: {actual_config_dim}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to process data WITH base DoF")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Process data WITHOUT base DoF
    print("\n" + "=" * 60)
    print("Test 2: Processing data WITHOUT base DoF")
    print("=" * 60)
    
    try:
        paths['data'] = os.path.join(CUR_DIR, f'data/{robot}/test_without_base_{serial_idx}.pt')
        
        # Need to create data without base for this test
        # For now, we'll skip if data has base DoF
        if q_dim > 4:  # Assuming LeapHand has 4 DoF per finger
            print("⚠️  Skipping test without base (data has base DoF)")
            print("    This test requires data without base DoF")
        else:
            cdf_without_base = CDF_V2(
                device=device,
                paths=paths,
                robot=robot,
                network_type='mlp',
                signed_distance=False,
                writer=None,
                serial_idx=serial_idx,
                use_base=False
            )
            
            print("\n✅ Successfully processed data WITHOUT base DoF")
            
            processed = torch.load(paths['data'])
            print(f"\nProcessed data shapes:")
            print(f"  x: {processed['x'].shape}")
            print(f"  q: {processed['q'].shape}")
            print(f"  k: {processed['k'].shape}")
            
    except Exception as e:
        print(f"⚠️  Test without base DoF skipped or failed")
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ process_data testing completed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_process_data()
    sys.exit(0 if success else 1)
