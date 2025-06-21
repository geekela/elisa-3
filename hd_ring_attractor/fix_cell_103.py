#!/usr/bin/env python3
"""
Script to fix cell 103 in enhanced_training_demo.ipynb to display tracking errors in degrees.
"""

import json
import re

# Path to the notebook
notebook_path = "notebooks/enhanced_training_demo.ipynb"

# Define the pattern to search for
radians_pattern = r'print\(f"([^"]*tracking error[^"]*): \{([^}]*)\}[ ]*radians([^"]*)"\)'
degrees_pattern = r'print\(f"Mean error in degrees:'

def fix_cell():
    """Fix cell 103 to display tracking errors in degrees."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return False
    
    if 'cells' not in notebook:
        print("No cells found in notebook")
        return False
    
    # Find the cell that prints tracking errors in radians
    target_cells = []
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        cell_code = ''.join(cell['source'])
        
        # Check if cell contains tracking error in radians
        if 'Mean tracking error:' in cell_code and 'radians' in cell_code:
            print(f"Found target cell at index {i}")
            target_cells.append((i, cell))
    
    if not target_cells:
        print("No target cells found")
        return False
    
    modifications_made = False
    
    # Process each target cell
    for cell_idx, cell in target_cells:
        cell_modified = False
        new_source = []
        
        # First, check if we need to add a conversion block
        conversion_block_needed = False
        
        for line in cell['source']:
            if re.search(radians_pattern, line):
                conversion_block_needed = True
        
        # If needed, add conversion block at the beginning
        if conversion_block_needed:
            conversion_added = False
            
            for i, line in enumerate(cell['source']):
                # Process each line
                if re.search(radians_pattern, line) and not conversion_added:
                    # Add conversion block before first tracking error print
                    new_source.append("# Convert all errors to degrees for display\n")
                    new_source.append("mean_error_deg = np.degrees(np.mean(tracking_errors))\n")
                    new_source.append("rms_error_deg = np.degrees(np.sqrt(np.mean(np.array(tracking_errors)**2)))\n")
                    new_source.append("max_error_deg = np.degrees(np.max(tracking_errors))\n")
                    new_source.append("\n")
                    
                    # Replace the line with degree version
                    if "Mean tracking error:" in line:
                        new_source.append('print(f"Mean tracking error: {mean_error_deg:.2f}°")\n')
                    elif "RMS tracking error:" in line:
                        new_source.append('print(f"RMS tracking error: {rms_error_deg:.2f}°")\n')
                    elif "Max tracking error:" in line:
                        new_source.append('print(f"Max tracking error: {max_error_deg:.2f}°")\n')
                    else:
                        # Generic replacement
                        new_line = re.sub(radians_pattern, 
                                         lambda m: f'print(f"{m.group(1)}: {{np.degrees({m.group(2)}):.2f}}°")', 
                                         line)
                        new_source.append(new_line)
                    
                    conversion_added = True
                    cell_modified = True
                
                elif re.search(radians_pattern, line) and conversion_added:
                    # Replace subsequent tracking error lines
                    if "Mean tracking error:" in line:
                        new_source.append('print(f"Mean tracking error: {mean_error_deg:.2f}°")\n')
                    elif "RMS tracking error:" in line:
                        new_source.append('print(f"RMS tracking error: {rms_error_deg:.2f}°")\n')
                    elif "Max tracking error:" in line:
                        new_source.append('print(f"Max tracking error: {max_error_deg:.2f}°")\n')
                    else:
                        # Generic replacement
                        new_line = re.sub(radians_pattern, 
                                         lambda m: f'print(f"{m.group(1)}: {{np.degrees({m.group(2)}):.2f}}°")', 
                                         line)
                        new_source.append(new_line)
                    
                    cell_modified = True
                
                elif re.search(degrees_pattern, line):
                    # Remove redundant "Mean error in degrees" line
                    cell_modified = True
                    # Skip this line (don't add to new_source)
                
                else:
                    # Keep unchanged
                    new_source.append(line)
        
        # Update cell source if modified
        if cell_modified:
            notebook['cells'][cell_idx]['source'] = new_source
            modifications_made = True
            print(f"Modified cell {cell_idx}")
    
    if not modifications_made:
        print("No modifications needed")
        return False
    
    # Save the updated notebook
    try:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Successfully updated {notebook_path}")
        return True
    except Exception as e:
        print(f"Error saving notebook: {e}")
        return False

if __name__ == "__main__":
    print("Fixing cell 103 in enhanced_training_demo.ipynb...")
    success = fix_cell()
    if success:
        print("Fix complete!")
    else:
        print("Fix failed.") 