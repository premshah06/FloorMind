#!/usr/bin/env python3
"""
Full CubiCasa5K Dataset Processing Script
Processes the complete CubiCasa5K dataset with 60/40 train/test split
"""

import sys
from pathlib import Path

# Add data directory to path
sys.path.append(str(Path(__file__).parent / 'data'))

from process_cubicasa5k_improved import CubiCasa5KProcessor

def main():
    """Process the full CubiCasa5K dataset"""
    
    print("🚀 Starting Full CubiCasa5K Dataset Processing")
    print("=" * 50)
    
    # Initialize processor
    processor = CubiCasa5KProcessor()
    
    try:
        # Process the full dataset with 60/40 split
        df = processor.process_dataset(train_ratio=0.6, max_samples=None)
        
        print(f"\n🎉 Full dataset processing completed successfully!")
        print(f"📊 Processed {len(df)} floor plans")
        print(f"📊 Train/Test split: 60%/40%")
        print(f"🖼️  All images resized to 512×512")
        
        # Show dataset breakdown
        print(f"\n📂 Dataset breakdown:")
        for category in df['category'].unique():
            count = len(df[df['category'] == category])
            print(f"   - {category}: {count} samples")
        
        print(f"\n📁 Generated files:")
        print(f"   - Full dataset: data/processed/cubicasa5k_full.csv")
        print(f"   - Train dataset: data/processed/cubicasa5k_train.csv")
        print(f"   - Test dataset: data/processed/cubicasa5k_test.csv")
        print(f"   - Training metadata: data/metadata.csv")
        print(f"   - Processed images: data/processed/images/")
        print(f"   - Statistics: data/processed/cubicasa5k_enhanced_statistics.json")
        print(f"   - Analysis plot: outputs/cubicasa5k_enhanced_analysis.png")
        
        print(f"\n💡 Next steps:")
        print("1. Train FloorMind models:")
        print("   python simple_training.py")
        print("2. Or use the training pipeline:")
        print("   python training/train_model.py")
        print("3. Test the results:")
        print("   python test_training.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)