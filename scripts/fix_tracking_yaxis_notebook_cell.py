"""
Fixed plot_tracking_performance function with consistent y-axis scaling.
This code can be copied into a Jupyter notebook cell to replace the existing function.

Make sure you have the following imports in your notebook:
import numpy as np
import matplotlib.pyplot as plt
"""

# Note: In Jupyter notebooks, numpy and matplotlib are typically already imported as:
# import numpy as np
# import matplotlib.pyplot as plt
# If not, please add these imports before running this cell.

# Fixed visualization function with consistent y-axis scaling for first row plots
def plot_tracking_performance(results, scenarios):
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
    
    print(f"Setting consistent y-axis range for direction plots: [{all_min_direction:.1f}°, {all_max_direction:.1f}°]")
    
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
    plt.show()


# If you want to re-run the visualization with the fixed function:
if 'results' in locals() and results:
    print("Re-creating tracking performance visualizations with fixed y-axis...")
    
    # Main tracking performance plot with fixed y-axis
    plot_tracking_performance(results, scenarios)
    
    # The summary statistics plot can remain the same
    plot_summary_statistics(results)
else:
    print("No tracking results found. Please run the tracking evaluation first!")
    print("After running the evaluation, execute this cell to see the fixed visualization.") 