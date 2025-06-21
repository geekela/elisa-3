"""
Test bump persistence without external input.

This script tests how long the activity bump persists and how it behaves
without any external sensory input - revealing the true memory capacity
of the ring attractor network.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import RingAttractorNetwork
from src.utils import angle_to_input


def test_bump_persistence_without_input(model, initial_direction, test_duration=5.0, dt=0.05, store_full_activity=False):
    """
    Test how long the bump persists without any external input.
    
    Args:
        model: Trained RingAttractorNetwork
        initial_direction: Initial head direction in radians
        test_duration: How long to test (seconds) - reduced default from 10 to 5
        dt: Time step
        store_full_activity: Whether to store full neural activities (memory intensive)
        
    Returns:
        Dictionary with persistence test results
    """
    n_steps = int(test_duration / dt)
    
    # Storage for results
    positions = np.zeros(n_steps)
    amplitudes = np.zeros(n_steps)
    position_variance = np.zeros(n_steps)
    
    # Only store full neural activities if requested (memory intensive)
    if store_full_activity:
        neural_activities = np.zeros((n_steps, model.n_exc))
    else:
        # Store only subsampled activities for visualization
        subsample_rate = 10  # Store every 10th step
        neural_activities = np.zeros((n_steps // subsample_rate + 1, model.n_exc))
    
    # Initialize bump at specified direction
    model.eval()
    model.reset_state()
    
    # Ensure model is on correct device before initialization
    device = next(model.parameters()).device
    model.to(device)
    
    model.initialize_bump(initial_direction, width=0.3, amplitude=0.2)
    
    print(f"Testing bump persistence for {test_duration}s without any input...")
    print(f"Initial direction: {np.degrees(initial_direction):.1f}°")
    
    # Get preferred directions once to avoid repeated device transfers
    preferred_dirs_cpu = model.preferred_dirs.cpu().numpy()
    
    # Run simulation with NO external input
    try:
        with torch.no_grad():
            for step in range(n_steps):
                # Run model with NO external input - this is the key!
                activity = model(external_input=None, steps=1)
                
                # Convert to CPU once
                activity_cpu = activity.cpu()
                activity_np = activity_cpu.numpy().flatten()
                
                # Store neural activity (full or subsampled)
                if store_full_activity:
                    neural_activities[step] = activity_np
                elif step % 10 == 0:
                    neural_activities[step // 10] = activity_np
                
                # Decode position
                decoded_dir = model.decode_angle(activity_cpu).item()
                positions[step] = decoded_dir
                
                # Measure bump amplitude (max activity)
                amplitudes[step] = torch.max(activity_cpu).item()
                
                # Calculate position variance (spread of activity)
                activity_sum = activity_cpu.sum().item()
                if activity_sum > 1e-6:  # Avoid division by zero
                    # Weighted variance in circular space
                    weights = activity_np / activity_sum
                    
                    # Circular variance
                    x = np.sum(weights * np.cos(preferred_dirs_cpu))
                    y = np.sum(weights * np.sin(preferred_dirs_cpu))
                    R = np.sqrt(x**2 + y**2)
                    position_variance[step] = 1 - R  # 0 = perfectly focused, 1 = uniform
                else:
                    position_variance[step] = 1.0
    except Exception as e:
        print(f"Error during bump persistence test: {e}")
        print(f"Step {step}/{n_steps}")
        raise
    
    # Analyze drift
    time_points = np.arange(n_steps) * dt
    
    # Calculate drift rate (unwrap angles for proper calculation)
    unwrapped_positions = np.unwrap(positions)
    drift_rate = np.polyfit(time_points, unwrapped_positions, 1)[0]  # radians/second
    
    # Find when bump "dies" (amplitude < 10% of initial)
    initial_amplitude = amplitudes[0]
    if initial_amplitude > 0:
        death_threshold = 0.1 * initial_amplitude
        death_indices = np.where(amplitudes < death_threshold)[0]
        if len(death_indices) > 0:
            death_time = time_points[death_indices[0]]
        else:
            death_time = test_duration  # Survived entire test
    else:
        death_time = 0
    
    # Calculate total drift
    total_drift = np.abs(positions[-1] - positions[0])
    if total_drift > np.pi:
        total_drift = 2*np.pi - total_drift  # Wrap around
    
    return {
        'time_points': time_points,
        'positions': positions,
        'amplitudes': amplitudes,
        'neural_activities': neural_activities,
        'position_variance': position_variance,
        'drift_rate': drift_rate,
        'drift_rate_deg_per_s': np.degrees(drift_rate),
        'death_time': death_time,
        'total_drift': total_drift,
        'total_drift_deg': np.degrees(total_drift),
        'initial_direction': initial_direction,
        'test_duration': test_duration
    }


def plot_bump_persistence_results(results):
    """
    Create comprehensive visualization of bump persistence without input.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Bump Persistence Without External Input', fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Position drift over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(results['time_points'], np.degrees(results['positions']), 'b-', linewidth=2)
    ax1.axhline(y=np.degrees(results['initial_direction']), color='r', linestyle='--', 
                alpha=0.5, label='Initial direction')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Decoded Direction (°)')
    ax1.set_title(f'Position Drift (Rate: {results["drift_rate_deg_per_s"]:.2f}°/s)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Amplitude decay
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(results['time_points'], results['amplitudes'], 'g-', linewidth=2)
    ax2.axhline(y=results['amplitudes'][0]*0.1, color='r', linestyle='--', 
                alpha=0.5, label='10% threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Bump Amplitude')
    ax2.set_title(f'Amplitude Decay (Persistence: {results["death_time"]:.1f}s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Position variance (bump spread)
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(results['time_points'], results['position_variance'], 'orange', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Variance')
    ax3.set_title('Bump Spread Over Time (0=focused, 1=uniform)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. Neural activity heatmap
    ax4 = fig.add_subplot(gs[:, 2])
    
    # Neural activities are already subsampled in the new version
    heatmap_data = results['neural_activities'].T
    
    # Only plot if we have data
    if heatmap_data.shape[1] > 0:
        im = ax4.imshow(
            heatmap_data,
            aspect='auto',
            cmap='hot',
            origin='lower',
            extent=[0, results['test_duration'], 0, heatmap_data.shape[0]],
            interpolation='bilinear'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Activity')
    else:
        ax4.text(0.5, 0.5, 'No activity data stored\n(memory optimization)', 
                ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Neuron Index')
    ax4.set_title('Neural Activity\n(Bump Evolution)')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("BUMP PERSISTENCE WITHOUT INPUT - SUMMARY")
    print("="*60)
    print(f"Test duration: {results['test_duration']}s")
    print(f"Initial direction: {np.degrees(results['initial_direction']):.1f}°")
    print(f"Final direction: {np.degrees(results['positions'][-1]):.1f}°")
    print(f"Total drift: {results['total_drift_deg']:.1f}°")
    print(f"Drift rate: {results['drift_rate_deg_per_s']:.2f}°/s")
    print(f"Persistence time (>10% amplitude): {results['death_time']:.1f}s")
    print(f"Final amplitude: {results['amplitudes'][-1]:.3f} ({results['amplitudes'][-1]/results['amplitudes'][0]*100:.1f}% of initial)")
    
    if results['drift_rate_deg_per_s'] < 5:
        print("\n✓ EXCELLENT: Very slow drift - similar to biological HD cells in darkness")
    elif results['drift_rate_deg_per_s'] < 20:
        print("\n✓ GOOD: Moderate drift rate - reasonable short-term memory")
    else:
        print("\n⚠ FAST DRIFT: Consider reducing noise parameters for better persistence")
    
    if results['death_time'] > 5:
        print("✓ Bump persists for extended period")
    elif results['death_time'] > 1:
        print("✓ Bump shows reasonable persistence")
    else:
        print("⚠ Bump decays quickly - may need parameter adjustment")


def main():
    """
    Main function to test bump persistence with a trained model.
    """
    # Load a pre-trained model or create one with example parameters
    print("Creating ring attractor network...")
    
    # Use trained parameters from the notebook results
    model = RingAttractorNetwork(
        n_exc=800,
        n_inh=200,
        dt=0.01,
        tau_e=0.02,
        tau_i=0.005
    )
    
    # Set trained parameters (from notebook output)
    with torch.no_grad():
        model.sigma_ee.data = torch.tensor(0.6642)
        model.g_ee.data = torch.tensor(0.8529)
        model.g_ei.data = torch.tensor(1.0001)
        model.g_ie.data = torch.tensor(1.4458)
        model.noise_rate_e.data = torch.tensor(0.5386)
        model.noise_rate_i.data = torch.tensor(0.1004)
    
    print("Model configured with trained parameters")
    
    # Test with random initial direction
    initial_dir = np.random.uniform(0, 2*np.pi)
    
    # Run persistence test
    results = test_bump_persistence_without_input(
        model,
        initial_direction=initial_dir,
        test_duration=5.0,  # Reduced from 10.0 to avoid memory issues
        store_full_activity=False  # Don't store full activity to save memory
    )
    
    # Visualize results
    plot_bump_persistence_results(results)
    
    # Compare with biological data
    print("\nComparison with biological HD cells:")
    print("- Real HD cells show drift rates of 5-15°/s in darkness (Valerio & Taube, 2016)")
    print("- Bump amplitude typically persists for >10s with gradual decay")
    print("- Drift is random walk-like, not systematic")
    
    # Run multiple trials for statistics
    print("\n" + "="*60)
    print("Running multiple trials for statistics...")
    
    n_trials = 5
    drift_rates = []
    death_times = []
    
    for i in range(n_trials):
        initial_dir = np.random.uniform(0, 2*np.pi)
        results = test_bump_persistence_without_input(
            model,
            initial_direction=initial_dir,
            test_duration=5.0
        )
        drift_rates.append(results['drift_rate_deg_per_s'])
        death_times.append(results['death_time'])
        print(f"Trial {i+1}: drift={results['drift_rate_deg_per_s']:.1f}°/s, persistence={results['death_time']:.1f}s")
    
    print(f"\nAverage drift rate: {np.mean(drift_rates):.1f} ± {np.std(drift_rates):.1f}°/s")
    print(f"Average persistence time: {np.mean(death_times):.1f} ± {np.std(death_times):.1f}s")


if __name__ == "__main__":
    main() 