# Head Direction Tracking Visualization Y-Axis Fix

## Problem
In the "Head Direction Tracking Performance with Trained Parameters" visualization, the y-axis scales for the first row (direction tracking) plots were inconsistent across different scenarios, making it difficult to compare tracking performance.

## Solution
I've created fixed versions of the visualization functions that ensure all first row plots share the same y-axis scale.

## Files Created

1. **`fix_tracking_yaxis.py`** - Standalone demonstration script
2. **`fix_tracking_yaxis_notebook_cell.py`** - Code to paste into Jupyter notebooks
3. **`hd_ring_attractor/src/visualization_utils.py`** - Reusable utility module

## How to Use

### Option 1: In Jupyter Notebooks

If you're running the tracking performance tests in a Jupyter notebook, simply copy the function from `fix_tracking_yaxis_notebook_cell.py` and paste it into a cell after your imports but before running the visualization:

```python
# Copy the plot_tracking_performance function from fix_tracking_yaxis_notebook_cell.py
# Then run your tracking tests and call the fixed function
```

### Option 2: In Python Scripts

Import the fixed function from the utility module:

```python
from hd_ring_attractor.src.visualization_utils import plot_tracking_performance_fixed_yaxis

# Run your tracking tests to get results and scenarios
# ...

# Create the fixed visualization
fig, axes = plot_tracking_performance_fixed_yaxis(results, scenarios)
plt.show()
```

### Option 3: Run the Demo

To see a demonstration with sample data:

```bash
cd /workspace/elisa-3
python fix_tracking_yaxis.py
```

This creates `head_direction_tracking_fixed_yaxis.png` showing the fixed visualization.

## What Changed

The fixed visualization:
1. Calculates the global min/max across all direction data (both actual and decoded)
2. Adds 5% padding for better visualization
3. Sets the same y-axis limits for all first row plots using `ax.set_ylim()`

This makes it much easier to compare direction tracking performance across different test scenarios.

## Example Output

The fixed visualization will show:
- **First row**: Direction tracking with consistent y-axis scale across all scenarios
- **Second row**: Tracking errors (unchanged)  
- **Third row**: Bump amplitudes (unchanged)

All scenarios can now be directly compared visually in the first row. 