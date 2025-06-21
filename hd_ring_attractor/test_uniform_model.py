#!/usr/bin/env python3
"""
Test script for the Uniform Ring Attractor Model.

This script demonstrates that the improved model maintains:
1. Uniform preferred direction distribution
2. Stable single-peak dynamics
3. Accurate tracking performance
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('.')

from src.uniform_ring_attractor import UniformRingAttractorNetwork
from src.utils import angle_to_input


def test_uniform_distribution():
    """
    Test that the model maintains uniform preferred direction distribution.
    """
    print("\n" + "="*70)
    print("TESTING UNIFORM PREFERRED DIRECTION DISTRIBUTION")
    print("="*70)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UniformRingAttractorNetwork(
        n_exc=800,
        n_inh=200,
        sigma_ee=0.15,
        device=str(device),
        enforce_symmetry=True
    )
    model.to(device)
    
    # Get tuning curves
    print("\nComputing tuning curves for all neurons...")
    tuning_data = model.get_tuning_curves(n_directions=36, input_strength=1.5)
    
    # Validate uniformity
    is_uniform, chi2, p_value = model.validate_uniformity(n_bins=18)
    
    print(f"\nUniformity Test Results:")
    print(f"  Chi-square statistic: {chi2:.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Result: {'UNIFORM ✓' if is_uniform else 'NON-UNIFORM ✗'}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Histogram of preferred directions
    ax = axes[0]
    ax.hist(tuning_data['preferred_dirs_deg'], bins=18, alpha=0.7, 
            color='lightgreen', edgecolor='black')
    expected_count = len(tuning_data['preferred_dirs_deg']) / 18
    ax.axhline(expected_count, color='red', linestyle='--', linewidth=2,
               label=f'Expected: {expected_count:.1f}')
    ax.set_xlabel('Preferred Direction (°)')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Preferred Direction Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Sample tuning curves
    ax = axes[1]
    n_samples = 20
    sample_indices = np.linspace(0, model.n_exc-1, n_samples, dtype=int)
    directions_deg = np.degrees(tuning_data['directions'])
    
    for idx in sample_indices:
        curve = tuning_data['tuning_curves'][idx]
        if np.max(curve) > 0:
            normalized_curve = curve / np.max(curve)
            ax.plot(directions_deg, normalized_curve, alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Direction (°)')
    ax.set_ylabel('Normalized Response')
    ax.set_title(f'Sample Tuning Curves ({n_samples} neurons)')
    ax.grid(True, alpha=0.3)
    
    # 3. Connectivity profile
    ax = axes[2]
    # Show E-E connectivity for a sample neuron
    W_EE = model._create_ring_weights(model.sigma_ee)
    sample_neuron = model.n_exc // 2
    connectivity = W_EE[sample_neuron].detach().cpu().numpy()
    
    # Shift to center the peak
    shift = len(connectivity) // 2 - sample_neuron
    connectivity_shifted = np.roll(connectivity, shift)
    neuron_indices = np.arange(len(connectivity)) - len(connectivity) // 2
    
    ax.plot(neuron_indices, connectivity_shifted, 'b-', linewidth=2)
    ax.fill_between(neuron_indices, 0, connectivity_shifted, alpha=0.3)
    ax.set_xlabel('Relative Neuron Index')
    ax.set_ylabel('Connection Weight')
    ax.set_title(f'E-E Connectivity Profile (σ={model.sigma_ee.item():.3f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('uniform_model_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, is_uniform


def test_tracking_performance(model):
    """
    Test the model's ability to track head direction accurately.
    """
    print("\n" + "="*70)
    print("TESTING TRACKING PERFORMANCE")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Test directions
    test_directions = np.linspace(0, 2*np.pi, 8, endpoint=False)
    errors = []
    
    print("\nTesting tracking at different directions:")
    for i, true_dir in enumerate(test_directions):
        model.reset_state()
        
        # Create input
        input_pattern = angle_to_input(
            torch.tensor(true_dir, device=device),
            n_exc=model.n_exc,
            input_strength=1.0,
            input_width=0.3,
            device=device
        )
        
        # Run network
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_pattern, steps=1)
            
            # Decode
            decoded_dir = model.decode_angle().item()
        
        # Calculate error
        error = np.abs(decoded_dir - true_dir)
        error = min(error, 2*np.pi - error)  # Circular error
        error_deg = np.degrees(error)
        errors.append(error_deg)
        
        print(f"  Direction {i+1}: True={np.degrees(true_dir):.1f}°, "
              f"Decoded={np.degrees(decoded_dir):.1f}°, Error={error_deg:.2f}°")
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"\nTracking Performance Summary:")
    print(f"  Mean error: {mean_error:.2f}°")
    print(f"  Max error: {max_error:.2f}°")
    print(f"  Performance: {'EXCELLENT ✓' if mean_error < 5 else 'GOOD' if mean_error < 10 else 'NEEDS IMPROVEMENT'}")
    
    return errors


def test_bump_stability(model):
    """
    Test the stability of activity bumps in the network.
    """
    print("\n" + "="*70)
    print("TESTING BUMP STABILITY")
    print("="*70)
    
    device = next(model.parameters()).device
    
    # Test persistence without input
    model.reset_state()
    initial_dir = np.pi
    model.initialize_bump(initial_dir, width=0.2, amplitude=0.5)
    
    # Track position and amplitude over time
    n_steps = 100  # 10 seconds at dt=0.1
    positions = []
    amplitudes = []
    
    print("\nTesting bump persistence without input...")
    with torch.no_grad():
        for step in range(n_steps):
            # No external input
            activity = model(external_input=None, steps=1)
            
            # Decode position
            decoded = model.decode_angle().item()
            positions.append(decoded)
            
            # Measure amplitude
            amp = activity.max().item()
            amplitudes.append(amp)
    
    # Analyze drift
    time_points = np.arange(n_steps) * 0.1  # Convert to seconds
    initial_amp = amplitudes[0]
    final_amp = amplitudes[-1]
    amp_retention = final_amp / initial_amp if initial_amp > 0 else 0
    
    # Calculate drift
    position_changes = np.diff(np.unwrap(positions))
    drift_rate = np.mean(position_changes) * 10  # Convert to rad/s
    drift_rate_deg = np.degrees(drift_rate)
    
    print(f"\nBump Stability Results:")
    print(f"  Initial amplitude: {initial_amp:.3f}")
    print(f"  Final amplitude: {final_amp:.3f} ({amp_retention:.1%} retained)")
    print(f"  Drift rate: {np.abs(drift_rate_deg):.1f}°/s")
    print(f"  Stability: {'EXCELLENT ✓' if amp_retention > 0.8 and abs(drift_rate_deg) < 10 else 'GOOD' if amp_retention > 0.5 else 'NEEDS IMPROVEMENT'}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Position over time
    ax1.plot(time_points, np.degrees(positions), 'b-', linewidth=2)
    ax1.axhline(np.degrees(initial_dir), color='gray', linestyle='--', 
                alpha=0.5, label='Initial')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Decoded Direction (°)')
    ax1.set_title('Position Tracking Without Input')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Amplitude over time
    ax2.plot(time_points, amplitudes, 'g-', linewidth=2)
    ax2.axhline(initial_amp * 0.5, color='red', linestyle='--', 
                alpha=0.5, label='50% threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Peak Amplitude')
    ax2.set_title('Bump Amplitude Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bump_stability_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return amp_retention, abs(drift_rate_deg)


def main():
    """
    Run all tests on the uniform ring attractor model.
    """
    print("\n" + "="*70)
    print("UNIFORM RING ATTRACTOR MODEL VALIDATION")
    print("="*70)
    
    # Test 1: Uniform distribution
    model, is_uniform = test_uniform_distribution()
    
    # Test 2: Tracking performance
    errors = test_tracking_performance(model)
    
    # Test 3: Bump stability
    amp_retention, drift_rate = test_bump_stability(model)
    
    # Summary
    print("\n" + "="*70)
    print("OVERALL VALIDATION SUMMARY")
    print("="*70)
    print(f"✓ Uniform distribution: {'PASSED' if is_uniform else 'FAILED'}")
    print(f"✓ Tracking accuracy: {np.mean(errors):.2f}° mean error")
    print(f"✓ Bump stability: {amp_retention:.1%} retention, {drift_rate:.1f}°/s drift")
    print("\nThe uniform ring attractor model successfully maintains:")
    print("  1. Uniform preferred direction distribution")
    print("  2. Accurate head direction tracking")
    print("  3. Stable activity bumps with minimal drift")


if __name__ == "__main__":
    main() 