#!/usr/bin/env python3
"""
Quick test to validate single-peak optimization works.
"""

import sys
import pathlib
import numpy as np
import torch

# Add src directory to path
current_dir = pathlib.Path(__file__).parent
src_path = current_dir / "hd_ring_attractor" / "src"
sys.path.insert(0, str(src_path))

from models import RingAttractorNetwork
from optimized_training import (apply_single_peak_constraints, initialize_for_single_peak, 
                               create_single_peak_config, monitor_peak_count)
from utils import angle_to_input

def quick_peak_test():
    """Quick test of peak optimization."""
    
    print("🧠 QUICK SINGLE-PEAK OPTIMIZATION TEST")
    print("="*50)
    
    device = torch.device('cpu')
    
    # Test 1: Problematic parameters (should cause multiple peaks)
    print("\n1. Testing PROBLEMATIC parameters:")
    model_bad = RingAttractorNetwork(n_exc=100, n_inh=25, device=device)
    
    # Set bad parameters
    with torch.no_grad():
        model_bad.g_ee.data.fill_(1.5)      # High excitation
        model_bad.g_ie.data.fill_(1.0)      # Low inhibition  
        model_bad.noise_rate_e.data.fill_(0.3)  # High noise
        model_bad.sigma_ee.data.fill_(0.8)  # Wide connections
    
    print(f"   g_ee: {model_bad.g_ee.item():.3f}")
    print(f"   g_ie: {model_bad.g_ie.item():.3f}")
    print(f"   ratio: {model_bad.g_ie.item()/model_bad.g_ee.item():.2f}")
    print(f"   noise: {model_bad.noise_rate_e.item():.3f}")
    
    # Test peak count with problematic parameters
    peak_stats_bad = monitor_peak_count(model_bad, input_directions=np.linspace(0, 2*np.pi, 4, endpoint=False))
    print(f"   Single peak ratio: {peak_stats_bad['single_peak_ratio']:.1%}")
    print(f"   Mean peaks: {peak_stats_bad['mean_peaks']:.1f}")
    
    # Test 2: Apply optimization
    print("\n2. Applying OPTIMIZATION:")
    model_good = RingAttractorNetwork(n_exc=100, n_inh=25, device=device)
    
    # Apply optimized initialization
    config = create_single_peak_config('standard')
    initialize_for_single_peak(model_good, config)
    apply_single_peak_constraints(model_good, strength='strong')
    
    print(f"   g_ee: {model_good.g_ee.item():.3f}")
    print(f"   g_ie: {model_good.g_ie.item():.3f}")
    print(f"   ratio: {model_good.g_ie.item()/model_good.g_ee.item():.2f}")
    print(f"   noise: {model_good.noise_rate_e.item():.3f}")
    
    # Test peak count with optimized parameters  
    peak_stats_good = monitor_peak_count(model_good, input_directions=np.linspace(0, 2*np.pi, 4, endpoint=False))
    print(f"   Single peak ratio: {peak_stats_good['single_peak_ratio']:.1%}")
    print(f"   Mean peaks: {peak_stats_good['mean_peaks']:.1f}")
    
    # Test 3: Detailed analysis
    print("\n3. DETAILED ANALYSIS:")
    
    test_direction = np.pi/2  # 90 degrees
    
    # Test problematic model
    model_bad.reset_state()
    model_bad.initialize_bump(test_direction, width=0.3, amplitude=0.1)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model_bad(external_input=None, steps=1)
    
    activity_bad = model_bad.r_e.cpu().numpy()
    
    # Test optimized model
    model_good.reset_state()
    model_good.initialize_bump(test_direction, width=0.3, amplitude=0.1)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model_good(external_input=None, steps=1)
    
    activity_good = model_good.r_e.cpu().numpy()
    
    # Analyze activities
    try:
        from scipy.signal import find_peaks
        
        # Count peaks in bad model
        peaks_bad, _ = find_peaks(activity_bad, height=0.05 * np.max(activity_bad), distance=3)
        peaks_good, _ = find_peaks(activity_good, height=0.05 * np.max(activity_good), distance=3)
        
        print(f"   Problematic model peaks: {len(peaks_bad)}")
        print(f"   Optimized model peaks: {len(peaks_good)}")
        print(f"   Max activity (bad): {np.max(activity_bad):.4f}")
        print(f"   Max activity (good): {np.max(activity_good):.4f}")
        
    except ImportError:
        print("   (scipy not available for peak detection)")
        print(f"   Max activity (bad): {np.max(activity_bad):.4f}")
        print(f"   Max activity (good): {np.max(activity_good):.4f}")
    
    # Test 4: Tracking performance
    print("\n4. TRACKING PERFORMANCE:")
    
    test_directions = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    errors_bad = []
    errors_good = []
    
    for direction in test_directions:
        # Test bad model
        model_bad.reset_state()
        input_pattern = angle_to_input(torch.tensor(direction, device=device), n_exc=model_bad.n_exc, device=device)
        
        with torch.no_grad():
            activity = model_bad(input_pattern, steps=5)
            decoded = model_bad.decode_angle(activity).item()
            error = abs(decoded - direction)
            error = min(error, 2*np.pi - error)
            errors_bad.append(np.degrees(error))
        
        # Test good model
        model_good.reset_state()
        
        with torch.no_grad():
            activity = model_good(input_pattern, steps=5)
            decoded = model_good.decode_angle(activity).item()
            error = abs(decoded - direction)
            error = min(error, 2*np.pi - error)
            errors_good.append(np.degrees(error))
    
    print(f"   Mean tracking error (bad): {np.mean(errors_bad):.1f}°")
    print(f"   Mean tracking error (good): {np.mean(errors_good):.1f}°")
    
    # Summary
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY:")
    print("="*50)
    
    improvement = peak_stats_good['single_peak_ratio'] - peak_stats_bad['single_peak_ratio']
    
    print(f"Single Peak Ratio:")
    print(f"  Before optimization: {peak_stats_bad['single_peak_ratio']:.1%}")
    print(f"  After optimization:  {peak_stats_good['single_peak_ratio']:.1%}")
    print(f"  Improvement:         {improvement:+.1%}")
    
    print(f"\nTracking Performance:")
    print(f"  Before optimization: {np.mean(errors_bad):.1f}° error")
    print(f"  After optimization:  {np.mean(errors_good):.1f}° error")
    
    if peak_stats_good['single_peak_ratio'] >= 0.8:
        print("\n✅ SUCCESS: Optimization eliminates multiple peaks!")
    elif improvement > 0.3:
        print("\n✅ GOOD: Significant improvement achieved!")
    else:
        print("\n⚠️  PARTIAL: Some improvement, may need further tuning")
    
    return {
        'before': peak_stats_bad,
        'after': peak_stats_good,
        'improvement': improvement
    }

if __name__ == "__main__":
    try:
        result = quick_peak_test()
        print(f"\nTest completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()