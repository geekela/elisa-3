#!/usr/bin/env python3
"""
Utility functions for improved visualization of head direction tracking performance.
This module provides fixed versions of visualization functions with consistent scaling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_tracking_performance_fixed_yaxis(results, scenarios, figsize=(16, 12)):
    """
    Create comprehensive visualization of tracking performance across scenarios
    with consistent y-axis scale for first row (direction tracking) plots.
    
    Args:
        results: Dictionary of tracking results for each scenario
        scenarios: Dictionary of scenario descriptions
        figsize: Figure size tuple (width, height)
    
    Returns:
        fig: Matplotlib figure object
    """
    if not results:
        print("No tracking results available. Please run the tracking evaluation first!")
        return None
        
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Head Direction Tracking Performance with Trained Parameters', 
                 fontsize=16, fontweight='bold')
    
    n_scenarios = len(results)
    
    # Create subplots
    gs = GridSpec(3, n_scenarios, hspace=0.4, wspace=0.3, figure=fig)
    
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
    
    print(f"Setting consistent y-axis range for direction plots: [{all_min_direction:.1f}°, {all_max_direction:.1f}°]")
    
    # Create all subplots
    axes = {'direction': [], 'error': [], 'amplitude': []}
    
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
        
        axes['direction'].append(ax1)
        
        # Row 2: Tracking error
        ax2 = fig.add_subplot(gs[1, i])
        ax2.plot(time_points, np.degrees(result['tracking_errors']), 
                'purple', linewidth=2)
        ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='30° threshold')
        ax2.set_ylabel('Error (°)')
        ax2.grid(True, alpha=0.3)
        if i == 0:
            ax2.legend()
        
        axes['error'].append(ax2)
        
        # Row 3: Bump amplitude
        ax3 = fig.add_subplot(gs[2, i])
        ax3.plot(time_points, result['bump_amplitudes'], 
                'green', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Bump Amplitude')
        ax3.grid(True, alpha=0.3)
        
        axes['amplitude'].append(ax3)
    
    plt.tight_layout()
    
    return fig, axes


def plot_summary_statistics_enhanced(results, figsize=(15, 5)):
    """
    Enhanced plot of summary statistics across all tracking scenarios.
    
    Args:
        results: Dictionary of tracking results
        figsize: Figure size tuple
    
    Returns:
        fig: Matplotlib figure object
    """
    if not results:
        print("No tracking results available. Please run the tracking evaluation first!")
        return None
        
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Tracking Performance Summary', fontsize=14, fontweight='bold')
    
    scenario_names = [name.replace('_', ' ').title() for name in results.keys()]
    
    # Mean tracking errors
    mean_errors = [np.degrees(result['mean_error']) for result in results.values()]
    bars1 = axes[0].bar(scenario_names, mean_errors, color='skyblue', alpha=0.7)
    axes[0].set_ylabel('Mean Error (°)')
    axes[0].set_title('Mean Tracking Error')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, mean_errors):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}°', ha='center', va='bottom', fontsize=9)
    
    # Tracking accuracy
    accuracies = [result['tracking_accuracy'] for result in results.values()]
    bars2 = axes[1].bar(scenario_names, accuracies, color='lightgreen', alpha=0.7)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Tracking Accuracy (<30° error)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 110)
    
    # Add value labels on bars
    for bar, value in zip(bars2, accuracies):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Bump stability
    stabilities = [result['bump_stability'] for result in results.values()]
    bars3 = axes[2].bar(scenario_names, stabilities, color='orange', alpha=0.7)
    axes[2].set_ylabel('Stability Index')
    axes[2].set_title('Bump Stability')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars3, stabilities):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    return fig


# Export the main functions
__all__ = ['plot_tracking_performance_fixed_yaxis', 'plot_summary_statistics_enhanced'] 