#!/usr/bin/env python3
"""
Script to update angle display in tracking output from radians to degrees.
This updates any remaining files that still display angles in radians.
"""

import os
import re
import glob
import json

def process_python_file(filepath):
    """Process a single Python file to convert radians to degrees in tracking error outputs."""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern for tracking error in radians with degrees in parentheses
    pattern1 = r'(["\'])([^"\']*tracking error[^"\']*):[ ]*\{[^}]*\}[ ]*radians([^"\']*)\1'
    
    # Pattern for error in radians without degree conversion
    pattern2 = r'(["\'])([^"\']*error[^"\']*):[ ]*\{[^}]*\}[ ]*radians\1'
    
    # Count original matches
    matches1 = re.findall(pattern1, content, re.IGNORECASE)
    matches2 = re.findall(pattern2, content, re.IGNORECASE)
    
    total_matches = len(matches1) + len(matches2)
    
    if total_matches == 0:
        print(f"  No matches found in {filepath}")
        return 0
    
    # Replace pattern1 - convert to show only degrees
    content = re.sub(
        pattern1,
        lambda m: m.group(1) + m.group(2) + ": {np.degrees(" + m.group(3).split('(')[0].strip() + "):.2f}°" + m.group(1),
        content
    )
    
    # Replace pattern2 - add degrees conversion
    content = re.sub(
        pattern2,
        lambda m: m.group(1) + m.group(2) + ": {np.degrees(" + m.group(0).split('{')[1].split('}')[0] + "):.2f}°" + m.group(1),
        content
    )
    
    # Update Python print statements for tracking error
    pattern3 = r'print\(f"([^"]*tracking error[^"]*): \{([^}]*)\}[ ]*radians([^"]*)"\)'
    content = re.sub(
        pattern3,
        lambda m: f'print(f"{m.group(1)}: {{np.degrees({m.group(2)}):.2f}}°")',
        content
    )
    
    # Replace print statements with "Mean error in degrees"
    pattern4 = r'print\(f"Mean error in degrees: \{([^}]*) \* 180 / np\.pi:[^}]*\}°"\)'
    content = re.sub(
        pattern4,
        lambda m: f'print(f"Mean error: {{np.degrees({m.group(1)}):.2f}}°")',
        content
    )
    
    # Save the updated content
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  Updated {total_matches} instances in {filepath}")
    return total_matches

def process_notebook(filepath):
    """Process a Jupyter notebook to convert radians to degrees in tracking error outputs."""
    print(f"Processing notebook {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except json.JSONDecodeError:
        print(f"  Error: Cannot parse {filepath} as JSON")
        return 0
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return 0
    
    if 'cells' not in notebook:
        print(f"  No cells found in {filepath}")
        return 0
    
    total_matches = 0
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        # Process source lines in each cell
        for j, line in enumerate(cell['source']):
            # Pattern for tracking error in radians
            if re.search(r'tracking error.*radians', line, re.IGNORECASE) or \
               re.search(r'error.*radians', line, re.IGNORECASE):
                
                # Update print statements with tracking error in radians
                if 'print(f"' in line and 'radians' in line:
                    # Pattern for print statements with tracking error
                    if 'tracking error' in line.lower():
                        new_line = re.sub(
                            r'print\(f"([^"]*tracking error[^"]*): \{([^}]*)\}[ ]*radians([^"]*)"\)',
                            lambda m: f'print(f"{m.group(1)}: {{np.degrees({m.group(2)}):.2f}}°")',
                            line
                        )
                    # Pattern for general error in radians
                    elif 'error' in line.lower() and 'radians' in line:
                        new_line = re.sub(
                            r'print\(f"([^"]*error[^"]*): \{([^}]*)\}[ ]*radians([^"]*)"\)',
                            lambda m: f'print(f"{m.group(1)}: {{np.degrees({m.group(2)}):.2f}}°")',
                            line
                        )
                    else:
                        new_line = line
                        
                    # Pattern for error in degrees with conversion formula
                    if 'degrees' in line and '180 / np.pi' in line:
                        new_line = re.sub(
                            r'print\(f"([^"]*degrees[^"]*): \{([^}]*) \* 180 / np\.pi:[^}]*\}°"\)',
                            lambda m: f'print(f"{m.group(1)}: {{np.degrees({m.group(2)}):.2f}}°")',
                            new_line
                        )
                    
                    if new_line != line:
                        cell['source'][j] = new_line
                        total_matches += 1
                        print(f"  Updated cell {i}, line {j}")
    
    if total_matches > 0:
        # Save the updated notebook
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"  Updated {total_matches} instances in notebook {filepath}")
    else:
        print(f"  No matches found in notebook {filepath}")
    
    return total_matches

def main():
    """Main function to process all relevant files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Files to process
    python_files = glob.glob(f"{base_dir}/*.py")
    test_files = glob.glob(f"{base_dir}/test_*.py")
    notebook_files = glob.glob(f"{base_dir}/notebooks/*.ipynb")
    
    total_updates = 0
    
    # Process Python files
    for filepath in python_files + test_files:
        if os.path.basename(filepath) == 'update_angle_display.py':
            continue  # Skip this script
        updates = process_python_file(filepath)
        total_updates += updates
    
    # Process Jupyter notebooks
    for filepath in notebook_files:
        updates = process_notebook(filepath)
        total_updates += updates
    
    print(f"\nCompleted with {total_updates} total updates")

if __name__ == "__main__":
    main() 