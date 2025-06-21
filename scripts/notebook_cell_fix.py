"""
This file contains code to be added as a cell in your Jupyter notebook.
Copy the code below into a cell at the beginning of your notebook and run it
to ensure all tracking errors are displayed in degrees.
"""

# =============================================================================
# BEGIN NOTEBOOK CELL CODE - COPY BELOW THIS LINE
# =============================================================================

# Monkey patch print function to convert radians to degrees
import builtins
import numpy as np
import re

# Store the original print function
_original_print = builtins.print

def _new_print(*args, **kwargs):
    """
    Custom print function that automatically converts radians to degrees
    when printing tracking errors.
    """
    # Process each argument
    new_args = []
    for arg in args:
        if isinstance(arg, str):
            # Convert tracking error in radians to degrees
            if "tracking error" in arg.lower() and "radians" in arg:
                # Try to extract the error value using regex
                match = re.search(r'([Mm]ean|[Rr][Mm][Ss]|[Mm]ax)?\s*[Tt]racking\s*[Ee]rror:?\s*([\d\.]+)\s*radians', arg)
                if match:
                    error_type = match.group(1) or "Tracking"
                    error_value = float(match.group(2))
                    # Convert to degrees
                    error_deg = np.degrees(error_value)
                    # Format the new string
                    new_arg = f"{error_type} Tracking Error: {error_deg:.2f}°"
                    new_args.append(new_arg)
                    continue
            # Convert any error with "radians" to degrees
            elif "error" in arg.lower() and "radians" in arg:
                # Try to extract the error value using regex
                match = re.search(r'([^:]+):\s*([\d\.]+)\s*radians', arg)
                if match:
                    error_type = match.group(1)
                    error_value = float(match.group(2))
                    # Convert to degrees
                    error_deg = np.degrees(error_value)
                    # Format the new string
                    new_arg = f"{error_type}: {error_deg:.2f}°"
                    new_args.append(new_arg)
                    continue
        
        # If no conversion was done, use the original argument
        new_args.append(arg)
    
    # Call the original print function with the modified arguments
    _original_print(*new_args, **kwargs)

# Replace the built-in print function with our custom one
builtins.print = _new_print

print("Print function patched to display angles in degrees instead of radians")

# Function to display tracking errors in degrees
def display_tracking_errors(tracking_errors):
    """
    Display tracking errors in degrees instead of radians.
    
    Args:
        tracking_errors: Array of tracking errors in radians
    """
    # Convert all errors to degrees for display
    mean_error_deg = np.degrees(np.mean(tracking_errors))
    rms_error_deg = np.degrees(np.sqrt(np.mean(np.array(tracking_errors)**2)))
    max_error_deg = np.degrees(np.max(tracking_errors))
    
    print(f"Mean tracking error: {mean_error_deg:.2f}°")
    print(f"RMS tracking error: {rms_error_deg:.2f}°")
    print(f"Max tracking error: {max_error_deg:.2f}°")
    
    # Additional statistics if needed
    accuracy_15deg = np.mean(np.degrees(tracking_errors) < 15) * 100
    accuracy_30deg = np.mean(np.degrees(tracking_errors) < 30) * 100
    
    print(f"\nTracking accuracy:")
    print(f"Within 15°: {accuracy_15deg:.1f}%")
    print(f"Within 30°: {accuracy_30deg:.1f}%")
    
    return {
        'mean_error_deg': mean_error_deg,
        'rms_error_deg': rms_error_deg,
        'max_error_deg': max_error_deg,
        'accuracy_15deg': accuracy_15deg,
        'accuracy_30deg': accuracy_30deg
    }

# =============================================================================
# END NOTEBOOK CELL CODE - COPY ABOVE THIS LINE
# ============================================================================= 