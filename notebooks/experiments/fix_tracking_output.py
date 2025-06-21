"""
Script to run in the notebook to fix tracking error displays.
This can be imported or copied directly into a cell to convert tracking errors to degrees.
"""

import numpy as np

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

# Optional code to patch all prints to use degrees
def patch_print_functions():
    """
    Monkey patch the print functions to always show angles in degrees.
    Call this at the start of your notebook.
    """
    import builtins
    _original_print = builtins.print
    
    def _new_print(*args, **kwargs):
        # Convert any arguments with "radians" to degrees
        new_args = []
        for arg in args:
            if isinstance(arg, str) and "radians" in arg:
                # Try to extract the value and convert to degrees
                parts = arg.split(":")
                if len(parts) >= 2:
                    try:
                        prefix = parts[0]
                        value_part = parts[1].strip()
                        if "radians" in value_part:
                            value_str = value_part.split()[0]
                            value = float(value_str)
                            degrees = np.degrees(value)
                            new_arg = f"{prefix}: {degrees:.2f}°"
                            new_args.append(new_arg)
                            continue
                    except:
                        pass
            new_args.append(arg)
        
        _original_print(*new_args, **kwargs)
    
    builtins.print = _new_print
    print("Print function patched to show angles in degrees") 