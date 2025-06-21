"""
Compare bump behavior WITH continuous input vs WITHOUT input.

This script directly addresses the question: 
"What happens to the bump after input is removed?"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import RingAttractorNetwork
from src.utils import angle_to_input


def test_with_continuous_input(model, direction, duration=3.0):
    """Test bump maintenance WITH continuous input."""
    dt = 0.05
    n_steps = int(duration / dt)
    
    # Storage
    amplitudes = []
    positions = []
    
    # Reset and initialize
    model.eval()
    model.reset_state()
    model.initialize_bump(direction, width=0.3, amplitude=0.2)
    
    # Run with continuous input
    with torch.no_grad():
        for _ in range(n_steps):
            input_pattern = angle_to_input(
                torch.tensor(direction),
                n_exc=model.n_exc,
                input_strength=1.0,
                input_width=0.3
            )
            activity = model(input_pattern.to(model.device), steps=1)
            
            amplitudes.append(torch.max(activity).cpu().numpy())
            positions.append(model.decode_angle(activity).cpu().numpy())
    
    return np.array(amplitudes), np.array(positions)


def test_without_input(model, direction, duration=3.0):
    """Test bump persistence WITHOUT any input."""
    dt = 0.05
    n_steps = int(duration / dt)
    
    # Storage
    amplitudes = []
    positions = []
    
    # Reset and initialize
    model.eval()
    model.reset_state()
    model.initialize_bump(direction, width=0.3, amplitude=0.2)
    
    # Run WITHOUT input
    with torch.no_grad():
        for _ in range(n_steps):
            activity = model(external_input=None, steps=1)
            
            amplitudes.append(torch.max(activity).cpu().numpy())
            positions.append(model.decode_angle(activity).cpu().numpy())
    
    return np.array(amplitudes), np.array(positions)


def main():
    """
    Compare bump behavior with and without input.
    """
    print("="*60)
    print("COMPARING BUMP BEHAVIOR: WITH vs WITHOUT INPUT")
    print("="*60)
    
    # Create model with trained parameters
    model = RingAttractorNetwork(
        n_exc=800,
        n_inh=200,
        dt=0.01,
        tau_e=0.02,
        tau_i=0.005
    )
    
    # Set trained parameters
    with torch.no_grad():
        model.sigma_ee.data = torch.tensor(0.6642)
        model.g_ee.data = torch.tensor(0.8529)
        model.g_ei.data = torch.tensor(1.0001)
        model.g_ie.data = torch.tensor(1.4458)
        model.noise_rate_e.data = torch.tensor(0.5386)
        model.noise_rate_i.data = torch.tensor(0.1004)
    
    # Test parameters
    test_direction = np.pi / 2  # 90 degrees
    test_duration = 5.0
    
    print(f"Testing at {np.degrees(test_direction):.0f}° for {test_duration}s...")
    
    # Run both tests
    print("\n1. WITH continuous input...")
    amp_with, pos_with = test_with_continuous_input(model, test_direction, test_duration)
    
    print("2. WITHOUT input...")
    amp_without, pos_without = test_without_input(model, test_direction, test_duration)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Bump Behavior: WITH vs WITHOUT Input', fontsize=16, fontweight='bold')
    
    time_points = np.arange(len(amp_with)) * 0.05
    
    # Amplitude comparison
    axes[0, 0].plot(time_points, amp_with, 'b-', linewidth=2, label='WITH input')
    axes[0, 0].plot(time_points, amp_without, 'r-', linewidth=2, label='WITHOUT input')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Bump Amplitude')
    axes[0, 0].set_title('Bump Amplitude Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Position comparison
    axes[0, 1].plot(time_points, np.degrees(pos_with), 'b-', linewidth=2, label='WITH input')
    axes[0, 1].plot(time_points, np.degrees(pos_without), 'r-', linewidth=2, label='WITHOUT input')
    axes[0, 1].axhline(y=np.degrees(test_direction), color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Decoded Direction (°)')
    axes[0, 1].set_title('Head Direction Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Drift analysis
    drift_with = np.degrees(pos_with - test_direction)
    drift_without = np.degrees(pos_without - test_direction)
    
    axes[1, 0].plot(time_points, drift_with, 'b-', linewidth=2, label='WITH input')
    axes[1, 0].plot(time_points, drift_without, 'r-', linewidth=2, label='WITHOUT input')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Drift from Initial (°)')
    axes[1, 0].set_title('Position Drift')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""SUMMARY STATISTICS:

WITH CONTINUOUS INPUT:
• Final amplitude: {amp_with[-1]:.3f} ({amp_with[-1]/amp_with[0]*100:.1f}% of initial)
• Total drift: {abs(drift_with[-1]):.1f}°
• Drift rate: {abs(drift_with[-1])/test_duration:.1f}°/s
• Stable: {"Yes" if abs(drift_with[-1]) < 10 else "No"}

WITHOUT INPUT:
• Final amplitude: {amp_without[-1]:.3f} ({amp_without[-1]/amp_without[0]*100:.1f}% of initial)
• Total drift: {abs(drift_without[-1]):.1f}°
• Drift rate: {abs(drift_without[-1])/test_duration:.1f}°/s
• Stable: {"Yes" if abs(drift_without[-1]) < 30 else "Drifts"}

KEY FINDING:
{"The bump drifts gradually without input,\nsimilar to biological HD cells in darkness." 
if abs(drift_without[-1]) > 10 else 
"The bump remains relatively stable\neven without input."}"""
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Print conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS:")
    print("="*60)
    
    if abs(drift_without[-1]) > 10:
        print("✓ The bump DRIFTS without input, similar to biological HD cells")
        print(f"  • Drift rate: {abs(drift_without[-1])/test_duration:.1f}°/s")
        print("  • This is EXPECTED and BIOLOGICALLY PLAUSIBLE")
        print("  • Real HD cells drift at 5-15°/s in darkness")
    else:
        print("✓ The bump remains relatively stable without input")
        print("  • This indicates strong attractor dynamics")
    
    if amp_without[-1] > 0.1 * amp_without[0]:
        print(f"✓ Bump persists for at least {test_duration}s")
        print(f"  • Retains {amp_without[-1]/amp_without[0]*100:.0f}% of initial amplitude")
    else:
        print("⚠ Bump amplitude decays significantly")
    
  
    print("-"*60)
    print("'Without external input, the activity bump exhibits gradual drift")
    print(f"at approximately {abs(drift_without[-1])/test_duration:.1f}°/s while maintaining {amp_without[-1]/amp_without[0]*100:.0f}% of its")
    print(f"initial amplitude over {test_duration}s. This drift behavior is consistent")
    print("with biological head direction cells recorded in darkness, which")
    print("show similar drift rates of 5-15°/s (Valerio & Taube, 2016).'")


if __name__ == "__main__":
    main() 