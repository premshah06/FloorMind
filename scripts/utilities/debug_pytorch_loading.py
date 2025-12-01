#!/usr/bin/env python3
"""
Debug PyTorch and Diffusers Loading
Test script to identify the segmentation fault issue
"""

import os
import sys
import traceback

def test_basic_imports():
    """Test basic library imports"""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device count: {torch.cuda.device_count()}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import diffusers
        print(f"✅ Diffusers imported successfully: {diffusers.__version__}")
    except Exception as e:
        print(f"❌ Diffusers import failed: {e}")
        return False
    
    return True

def test_model_path():
    """Test model path and files"""
    print("\n🔍 Testing model path...")
    
    model_path = "./google"
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False
    
    print(f"✅ Model path exists: {model_path}")
    
    # Check required files
    required_files = ["model.safetensors", "tokenizer_config.json", "scheduler_config.json"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file}: {size:,} bytes")
        else:
            missing_files.append(file)
            print(f"❌ {file}: missing")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def test_simple_tensor():
    """Test simple tensor operations"""
    print("\n🔍 Testing simple tensor operations...")
    
    try:
        import torch
        
        # Create a simple tensor
        x = torch.randn(2, 3)
        print(f"✅ Created tensor: {x.shape}")
        
        # Test device movement
        if torch.cuda.is_available():
            try:
                x_cuda = x.cuda()
                print(f"✅ Moved tensor to CUDA: {x_cuda.device}")
                x_cpu = x_cuda.cpu()
                print(f"✅ Moved tensor back to CPU: {x_cpu.device}")
            except Exception as e:
                print(f"⚠️  CUDA operations failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        traceback.print_exc()
        return False

def test_diffusers_components():
    """Test individual diffusers components"""
    print("\n🔍 Testing diffusers components...")
    
    try:
        from diffusers import UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        
        print("✅ Diffusers components imported successfully")
        
        # Test loading individual components
        model_path = "./google"
        
        try:
            print("🔄 Loading tokenizer...")
            tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
            print("✅ Tokenizer loaded")
        except Exception as e:
            print(f"❌ Tokenizer loading failed: {e}")
            return False
        
        try:
            print("🔄 Loading scheduler...")
            scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", local_files_only=True)
            print("✅ Scheduler loaded")
        except Exception as e:
            print(f"❌ Scheduler loading failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Diffusers components test failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_loading():
    """Test pipeline loading with safe settings"""
    print("\n🔍 Testing pipeline loading...")
    
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        model_path = "./google"
        
        print("🔄 Loading pipeline with safe settings...")
        
        # Try with minimal settings first
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for stability
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
            device_map=None  # Don't auto-assign devices
        )
        
        print("✅ Pipeline loaded successfully")
        
        # Test moving to CPU explicitly
        pipeline = pipeline.to("cpu")
        print("✅ Pipeline moved to CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline loading failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🏠 FloorMind PyTorch Loading Debug")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Path", test_model_path),
        ("Simple Tensor", test_simple_tensor),
        ("Diffusers Components", test_diffusers_components),
        ("Pipeline Loading", test_pipeline_loading)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Model loading should work.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()