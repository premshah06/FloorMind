#!/usr/bin/env python3
"""
FloorMind Setup Test Script
Verifies that all components are properly installed and configured
"""

import sys
import os
import json
from pathlib import Path

def test_directory_structure():
    """Test that all required directories exist"""
    print("🔍 Testing directory structure...")
    
    required_dirs = [
        'backend',
        'backend/routes',
        'backend/services', 
        'backend/utils',
        'backend/models',
        'data',
        'data/raw',
        'data/processed',
        'notebooks',
        'outputs',
        'outputs/sample_generations',
        'outputs/metrics'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True

def test_required_files():
    """Test that all required files exist"""
    print("\n🔍 Testing required files...")
    
    required_files = [
        'requirements.txt',
        'README.md',
        'backend/app.py',
        'backend/routes/generate.py',
        'backend/routes/evaluate.py',
        'backend/services/model_service.py',
        'backend/utils/helpers.py',
        'notebooks/FloorMind_Training_and_Analysis.ipynb',
        'data/metadata.csv',
        'outputs/metrics/results.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_python_imports():
    """Test that key Python packages can be imported"""
    print("\n🔍 Testing Python package imports...")
    
    required_packages = [
        'torch',
        'numpy', 
        'pandas',
        'matplotlib',
        'seaborn',
        'PIL',
        'flask',
        'json',
        'tqdm'
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            failed_imports.append(package)
            print(f"  ❌ {package}")
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages can be imported")
        return True

def test_json_files():
    """Test that JSON files are valid"""
    print("\n🔍 Testing JSON file validity...")
    
    json_files = [
        'outputs/metrics/results.json'
    ]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                json.load(f)
            print(f"  ✅ {json_file}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  ❌ {json_file}: {e}")
            return False
    
    print("✅ All JSON files are valid")
    return True

def test_notebook_structure():
    """Test notebook file structure"""
    print("\n🔍 Testing notebook structure...")
    
    notebook_path = 'notebooks/FloorMind_Training_and_Analysis.ipynb'
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Check for required sections
        cells = notebook.get('cells', [])
        markdown_cells = [cell for cell in cells if cell.get('cell_type') == 'markdown']
        code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']
        
        print(f"  📄 Total cells: {len(cells)}")
        print(f"  📝 Markdown cells: {len(markdown_cells)}")
        print(f"  💻 Code cells: {len(code_cells)}")
        
        if len(code_cells) >= 5:
            print("✅ Notebook has sufficient code cells")
            return True
        else:
            print("❌ Notebook needs more code cells")
            return False
            
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FloorMind Setup Test")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_required_files,
        test_json_files,
        test_notebook_structure,
        test_python_imports
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ FloorMind is ready to use!")
        print("\n🚀 Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run notebook: jupyter notebook notebooks/FloorMind_Training_and_Analysis.ipynb")
        print("  3. Start API: cd backend && python app.py")
        return True
    else:
        print(f"❌ {total - passed} TESTS FAILED ({passed}/{total} passed)")
        print("\n🔧 Please fix the issues above before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)