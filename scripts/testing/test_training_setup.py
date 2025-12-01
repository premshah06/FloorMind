#!/usr/bin/env python3
"""
Test Training Setup
Quick test to verify the training environment and dataset are ready
"""

import sys
import torch
from pathlib import Path
import pandas as pd
from PIL import Image

def test_environment():
    """Test the training environment"""
    print("🧪 Testing Training Environment")
    print("=" * 40)
    
    # Test PyTorch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ MPS Available: {torch.backends.mps.is_available()}")
    
    # Test device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"🚀 Using Apple Silicon GPU")
    else:
        device = "cpu"
        print(f"⚠️ Using CPU (training will be slow)")
    
    return device

def test_dataset():
    """Test dataset availability"""
    print("\n📊 Testing Dataset")
    print("=" * 40)
    
    # Check metadata
    metadata_path = Path("data/metadata.csv")
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        print(f"✅ Metadata found: {len(df)} samples")
        print(f"📋 Columns: {list(df.columns)}")
    else:
        print("❌ Metadata file not found")
        return False
    
    # Check processed images
    images_dir = Path("data/processed/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png"))
        print(f"✅ Images found: {len(image_files)}")
        
        # Test loading an image
        if image_files:
            try:
                test_image = Image.open(image_files[0])
                print(f"✅ Test image loaded: {test_image.size}")
            except Exception as e:
                print(f"❌ Error loading test image: {e}")
                return False
    else:
        print("❌ Images directory not found")
        return False
    
    return True

def test_model_loading():
    """Test model component loading"""
    print("\n🤖 Testing Model Loading")
    print("=" * 40)
    
    try:
        from diffusers import StableDiffusionPipeline
        from transformers import CLIPTokenizer
        
        # Test loading tokenizer (lightweight test)
        tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="tokenizer"
        )
        print("✅ Tokenizer loaded successfully")
        
        # Test tokenization
        test_text = "A detailed architectural floor plan"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Text tokenization works: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 FloorMind Training Setup Test")
    print("=" * 50)
    
    # Test environment
    device = test_environment()
    
    # Test dataset
    dataset_ok = test_dataset()
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 40)
    print(f"Environment: {'✅' if device else '❌'}")
    print(f"Dataset: {'✅' if dataset_ok else '❌'}")
    print(f"Model Loading: {'✅' if model_ok else '❌'}")
    
    if all([device, dataset_ok, model_ok]):
        print("\n🎉 All tests passed! Ready for training.")
        print("\n💡 To start training:")
        print("   python training/simple_training.py")
        return True
    else:
        print("\n❌ Some tests failed. Please fix issues before training.")
        
        if not dataset_ok:
            print("\n🔧 To fix dataset issues:")
            print("   python process_full_dataset.py")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)