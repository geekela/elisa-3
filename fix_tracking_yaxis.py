#!/usr/bin/env python3
"""
Fix y-axis scaling for Head Direction Tracking Performance visualization.
This script ensures all first row plots (direction tracking) have the same y-axis scale.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_tracking_performance_fixed(results, scenarios):
    """
    Create comprehensive visualization of tracking performance across scenarios.
    Fixed version with consistent y-axis scale for first row plots.
    """
    if not results:
        print("No tracking results available. Please run the tracking evaluation first!")
        return
        
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Head Direction Tracking Performance with Trained Parameters', 
                 fontsize=16, fontweight='bold')
    
    n_scenarios = len(results)
    
    # Create subplots
    gs = fig.add_gridspec(3, n_scenarios, hspace=0.4, wspace=0.3)
    
    scenario_names = list(results.keys())
    
    # First pass: find global min/max for direction data across all scenarios
    all_min_direction = float('inf')
    all_max_direction = float('-inf')
    
    for scenario_name in scenario_names:
        result = results[scenario_name]
        # Get min/max for both actual and decoded directions
        actual_deg = np.degrees(result['head_directions'])
        decoded_deg = np.degrees(result['decoded_directions'])
        
        scenario_min = min(np.min(actual_deg), np.min(decoded_deg))
        scenario_max = max(np.max(actual_deg), np.max(decoded_deg))
        
        all_min_direction = min(all_min_direction, scenario_min)
        all_max_direction = max(all_max_direction, scenario_max)
    
    # Add some padding (5% of range)
    direction_range = all_max_direction - all_min_direction
    padding = direction_range * 0.05
    all_min_direction -= padding
    all_max_direction += padding
    
    # Create all subplots
    ax1_list = []  # Store first row axes for setting ylim later
    
    for i, scenario_name in enumerate(scenario_names):
        result = results[scenario_name]
        description = scenarios[scenario_name][1]
        
        dt = 0.05
        time_points = np.arange(len(result['head_directions'])) * dt
        
        # Row 1: Direction tracking
        ax1 = fig.add_subplot(gs[0, i])
        ax1.plot(time_points, np.degrees(result['head_directions']), 
                'r-', linewidth=2, label='Actual', alpha=0.8)
        ax1.plot(time_points, np.degrees(result['decoded_directions']), 
                'b--', linewidth=2, label='Decoded', alpha=0.8)
        ax1.set_title(f'{scenario_name.replace("_", " ").title()}\n{description}', fontsize=10)
        ax1.set_ylabel('Direction (°)')
        ax1.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits for all first row plots
        ax1.set_ylim(all_min_direction, all_max_direction)
        
        if i == 0:
            ax1.legend()
        
        ax1_list.append(ax1)
        
        # Row 2: Tracking error
        ax2 = fig.add_subplot(gs[1, i])
        ax2.plot(time_points, np.degrees(result['tracking_errors']), 
                'purple', linewidth=2)
        ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='30° threshold')
        ax2.set_ylabel('Error (°)')
        ax2.grid(True, alpha=0.3)
        if i == 0:
            ax2.legend()
        
        # Row 3: Bump amplitude
        ax3 = fig.add_subplot(gs[2, i])
        ax3.plot(time_points, result['bump_amplitudes'], 
                'green', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Bump Amplitude')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def demo_fixed_visualization():
    """
    Demonstrate the fixed visualization with sample data.
    """
    # Generate sample data for demonstration
    print("Generating sample tracking data for demonstration...")
    
    # Sample scenario data
    scenarios = {
        'static_direction': (None, 'Maintain bump at fixed direction'),
        'slow_turn': (None, 'Slow 180° turn over 4s'),
        'fast_turn': (None, 'Fast 180° turn over 1s'),
        'oscillation': (None, 'Oscillation ±45° around 90°'),
        'random_walk': (None, 'Random walk with occasional turns')
    }
    
    # Generate sample results
    results = {}
    np.random.seed(42)
    
    # Static direction
    n_steps = 60
    results['static_direction'] = {
        'head_directions': np.ones(n_steps) * np.radians(90) + np.random.normal(0, 0.01, n_steps),
        'decoded_directions': np.ones(n_steps) * np.radians(90) + np.random.normal(0, 0.05, n_steps),
        'tracking_errors': np.abs(np.random.normal(0, 0.1, n_steps)),
        'bump_amplitudes': 0.4 + np.random.normal(0, 0.02, n_steps),
        'mean_error': 0.05,
        'tracking_accuracy': 95.0,
        'bump_stability': 0.95
    }
    
    # Slow turn
    n_steps = 80
    angles = np.linspace(0, np.pi, n_steps)
    results['slow_turn'] = {
        'head_directions': angles,
        'decoded_directions': angles + np.random.normal(0, 0.1, n_steps),
        'tracking_errors': np.abs(np.random.normal(0, 0.15, n_steps)),
        'bump_amplitudes': 0.35 + np.random.normal(0, 0.03, n_steps),
        'mean_error': 0.1,
        'tracking_accuracy': 85.0,
        'bump_stability': 0.85
    }
    
    # Fast turn
    n_steps = 20
    angles = np.linspace(0, np.pi, n_steps)
    results['fast_turn'] = {
        'head_directions': angles,
        'decoded_directions': angles + np.random.normal(0, 0.2, n_steps),
        'tracking_errors': np.abs(np.random.normal(0.1, 0.2, n_steps)),
        'bump_amplitudes': 0.3 + np.random.normal(0, 0.05, n_steps),
        'mean_error': 0.2,
        'tracking_accuracy': 75.0,
        'bump_stability': 0.75
    }
    
    # Oscillation
    n_steps = 120
    t = np.linspace(0, 6, n_steps)
    center = np.pi/2
    angles = center + np.radians(45) * np.sin(2 * np.pi * 0.5 * t)
    results['oscillation'] = {
        'head_directions': angles,
        'decoded_directions': angles + np.random.normal(0, 0.1, n_steps),
        'tracking_errors': np.abs(np.random.normal(0, 0.12, n_steps)),
        'bump_amplitudes': 0.38 + np.random.normal(0, 0.02, n_steps),
        'mean_error': 0.08,
        'tracking_accuracy': 88.0,
        'bump_stability': 0.9
    }
    
    # Random walk
    n_steps = 100
    angles = np.cumsum(np.random.normal(0, 0.1, n_steps))
    angles = np.mod(angles, 2*np.pi)
    results['random_walk'] = {
        'head_directions': angles,
        'decoded_directions': angles + np.random.normal(0, 0.15, n_steps),
        'tracking_errors': np.abs(np.random.normal(0, 0.18, n_steps)),
        'bump_amplitudes': 0.32 + np.random.normal(0, 0.04, n_steps),
        'mean_error': 0.15,
        'tracking_accuracy': 80.0,
        'bump_stability': 0.8
    }
    
    # Create the fixed visualization
    print("\nCreating fixed visualization with consistent y-axis scaling...")
    fig = plot_tracking_performance_fixed(results, scenarios)
    
    if fig is not None:
        # Save the figure
        fig.savefig('head_direction_tracking_fixed_yaxis.png', dpi=150, bbox_inches='tight')
        print("✓ Fixed visualization saved as 'head_direction_tracking_fixed_yaxis.png'")
        
        # Show the plot
        plt.show()
        
        # Calculate y-axis range info for display
        all_min_direction = float('inf')
        all_max_direction = float('-inf')
        
        for result in results.values():
            actual_deg = np.degrees(result['head_directions'])
            decoded_deg = np.degrees(result['decoded_directions'])
            
            scenario_min = min(np.min(actual_deg), np.min(decoded_deg))
            scenario_max = max(np.max(actual_deg), np.max(decoded_deg))
            
            all_min_direction = min(all_min_direction, scenario_min)
            all_max_direction = max(all_max_direction, scenario_max)
        
        # Add padding
        direction_range = all_max_direction - all_min_direction
        padding = direction_range * 0.05
        all_min_direction -= padding
        all_max_direction += padding
        
        # Print y-axis range info
        print(f"\nY-axis range for all first row plots:")
        print(f"  Min: {all_min_direction:.1f}°")
        print(f"  Max: {all_max_direction:.1f}°")
        print("\nAll direction tracking plots now share the same y-axis scale!")


if __name__ == "__main__":
    demo_fixed_visualization() 