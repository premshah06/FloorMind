#!/usr/bin/env python3
"""
Fix the FloorMind notebook by adding missing time import
"""

import json

def fix_notebook():
    """Add missing time import to the notebook"""
    
    # Read the notebook
    with open('notebooks/FloorMind_Training_and_Analysis.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Find the first code cell and add time import
    for cell in notebook['cells']:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            
            # Check if this is the import cell
            if any('import torch' in line for line in source):
                # Find the line with 'import os' and add time import after it
                for i, line in enumerate(source):
                    if 'import os' in line and 'import time' not in ''.join(source):
                        # Insert time import after os import
                        source.insert(i + 1, 'import time\n')
                        break
                break
    
    # Save the fixed notebook
    with open('notebooks/FloorMind_Training_and_Analysis.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("✅ Fixed notebook - added missing time import")

if __name__ == "__main__":
    fix_notebook()