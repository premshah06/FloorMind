#!/usr/bin/env python3
"""
Dataset Structure Checker
Analyzes your CubiCasa5K dataset structure and provides setup guidance
"""

import os
from pathlib import Path
import json

def analyze_dataset_structure():
    """Analyze the CubiCasa5K dataset structure"""
    
    print("🔍 FloorMind Dataset Structure Analyzer")
    print("=" * 60)
    
    dataset_path = Path("data/cubicasa5k")
    
    print(f"📁 Checking: {dataset_path.absolute()}")
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found!")
        print(f"💡 Expected location: {dataset_path.absolute()}")
        print(f"💡 Your dataset is at: /DeepLearning/FloorMind/FloorMind/data/cubicasa5k")
        print(f"💡 Make sure you're running this from the FloorMind directory")
        return False
    
    print(f"✅ Dataset directory exists")
    
    # Analyze structure
    total_dirs = 0
    total_files = 0
    image_files = 0
    json_files = 0
    valid_floor_plans = 0
    
    structure_info = {
        'directories': [],
        'sample_files': [],
        'floor_plans': []
    }
    
    print(f"\n📊 Analyzing structure...")
    
    # Walk through directory
    for root, dirs, files in os.walk(dataset_path):
        total_dirs += len(dirs)
        total_files += len(files)
        
        rel_root = Path(root).relative_to(dataset_path)
        
        # Track directory structure
        if len(str(rel_root).split('/')) <= 3:  # Don't go too deep
            structure_info['directories'].append(str(rel_root))
        
        # Count file types
        for file in files:
            if file.endswith('.png'):
                image_files += 1
                if len(structure_info['sample_files']) < 5:
                    structure_info['sample_files'].append(f"{rel_root}/{file}")
            elif file.endswith('.json'):
                json_files += 1
        
        # Check for valid floor plan structure
        if 'model.json' in files and any(d == 'colorful' for d in dirs):
            valid_floor_plans += 1
            if len(structure_info['floor_plans']) < 5:
                structure_info['floor_plans'].append(str(rel_root))
    
    # Display results
    print(f"\n📈 Dataset Statistics:")
    print(f"  📁 Total directories: {total_dirs:,}")
    print(f"  📄 Total files: {total_files:,}")
    print(f"  🖼️  Image files (.png): {image_files:,}")
    print(f"  📋 JSON files (.json): {json_files:,}")
    print(f"  🏠 Valid floor plans: {valid_floor_plans:,}")
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
    size_gb = total_size / (1024**3)
    print(f"  💾 Total size: {size_gb:.2f} GB")
    
    print(f"\n🗂️  Directory Structure (sample):")
    for directory in sorted(structure_info['directories'][:10]):
        print(f"  📁 {directory}")
    if len(structure_info['directories']) > 10:
        print(f"  ... and {len(structure_info['directories']) - 10} more")
    
    print(f"\n🖼️  Sample Image Files:")
    for file_path in structure_info['sample_files']:
        print(f"  📸 {file_path}")
    
    print(f"\n🏠 Sample Floor Plan Directories:")
    for plan_dir in structure_info['floor_plans']:
        print(f"  🏗️  {plan_dir}")
    
    # Check specific structures
    print(f"\n🔍 Structure Analysis:")
    
    high_quality_dir = dataset_path / "high_quality"
    if high_quality_dir.exists():
        hq_plans = sum(1 for item in high_quality_dir.iterdir() 
                      if item.is_dir() and (item / "model.json").exists())
        print(f"  ✅ High quality directory found with {hq_plans} floor plans")
    else:
        print(f"  ⚠️  No 'high_quality' directory found")
    
    # Check for common CubiCasa5K structure
    expected_structure = [
        "high_quality",
        "high_quality_architectural", 
        "colorful",
        "model.json"
    ]
    
    found_structure = []
    for item in expected_structure:
        if (dataset_path / item).exists():
            found_structure.append(item)
    
    if found_structure:
        print(f"  ✅ Found expected CubiCasa5K elements: {', '.join(found_structure)}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if valid_floor_plans == 0:
        print(f"  ❌ No valid floor plans found!")
        print(f"  💡 Expected structure:")
        print(f"     cubicasa5k/")
        print(f"     ├── high_quality/")
        print(f"     │   ├── plan_001/")
        print(f"     │   │   ├── model.json")
        print(f"     │   │   └── colorful/")
        print(f"     │   │       └── *.png")
        return False
    elif valid_floor_plans < 50:
        print(f"  ⚠️  Only {valid_floor_plans} floor plans found - consider getting more data")
        print(f"  💡 Minimum 50+ recommended for training")
    elif valid_floor_plans < 500:
        print(f"  ✅ {valid_floor_plans} floor plans found - good for initial training")
        print(f"  💡 500+ recommended for best results")
    else:
        print(f"  🎉 {valid_floor_plans} floor plans found - excellent for training!")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Process dataset:")
    print(f"     python setup_training.py")
    print(f"  2. Or process manually:")
    print(f"     python data/process_datasets.py --dataset cubicasa5k --dataset-path data/cubicasa5k")
    print(f"  3. Start training:")
    print(f"     python train_models.py")
    
    return valid_floor_plans > 0

def main():
    """Main function"""
    
    try:
        success = analyze_dataset_structure()
        
        if success:
            print(f"\n✅ Dataset analysis complete - ready for processing!")
        else:
            print(f"\n❌ Dataset issues found - please check structure")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)