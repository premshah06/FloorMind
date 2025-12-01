#!/usr/bin/env python3
"""
Test script for FloorMind Backend
Tests all API endpoints and functionality
"""

import requests
import json
import time
import base64
from io import BytesIO
from PIL import Image

BASE_URL = "http://localhost:5001"

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health():
    """Test health endpoint"""
    print_section("Testing Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print_section("Testing Model Info")
    
    try:
        response = requests.get(f"{BASE_URL}/model/info", timeout=5)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if data.get('model', {}).get('is_loaded'):
            print("\n✅ Model is loaded!")
            return True
        else:
            print("\n⚠️  Model not loaded yet")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_load():
    """Test model loading endpoint"""
    print_section("Testing Model Load")
    
    try:
        print("Sending load request (this may take a minute)...")
        response = requests.post(
            f"{BASE_URL}/model/load",
            json={"force_cpu": True},
            timeout=300  # 5 minutes timeout
        )
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if response.status_code == 200:
            print("\n✅ Model loaded successfully!")
            return True
        else:
            print(f"\n❌ Failed to load model: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_presets():
    """Test presets endpoint"""
    print_section("Testing Presets")
    
    try:
        response = requests.get(f"{BASE_URL}/presets", timeout=5)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        
        if response.status_code == 200:
            print("\n✅ Presets retrieved!")
            print(f"\nCategories: {list(data.get('presets', {}).keys())}")
            
            # Show first preset from each category
            for category, prompts in data.get('presets', {}).items():
                if prompts:
                    print(f"\n{category.title()}:")
                    print(f"  - {prompts[0]}")
            return True
        else:
            print(f"❌ Failed: {data}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_generation():
    """Test floor plan generation"""
    print_section("Testing Floor Plan Generation")
    
    prompt = "Modern 2-bedroom apartment with open kitchen and living room"
    
    try:
        print(f"Prompt: {prompt}")
        print("Generating (this will take 30-60 seconds on CPU)...")
        
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/generate",
            json={
                "description": prompt,
                "width": 512,
                "height": 512,
                "steps": 20,
                "guidance": 7.5,
                "seed": 42,
                "save": True
            },
            timeout=300
        )
        
        elapsed = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Generation successful!")
            print(f"Time taken: {elapsed:.1f} seconds")
            print(f"Saved to: {data.get('saved_path', 'Not saved')}")
            
            # Try to decode and display image info
            if 'image' in data:
                img_data = data['image'].split(',')[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes))
                print(f"Image size: {img.size}")
                print(f"Image mode: {img.mode}")
            
            return True
        else:
            data = response.json()
            print(f"❌ Failed: {data.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_generation():
    """Test batch generation"""
    print_section("Testing Batch Generation")
    
    prompt = "Cozy studio apartment with efficient layout"
    
    try:
        print(f"Prompt: {prompt}")
        print("Generating 2 variations (this will take 1-2 minutes on CPU)...")
        
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/generate/batch",
            json={
                "description": prompt,
                "count": 2,
                "steps": 15,  # Fewer steps for faster testing
                "guidance": 7.5,
                "seed": 100
            },
            timeout=600
        )
        
        elapsed = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Batch generation successful!")
            print(f"Time taken: {elapsed:.1f} seconds")
            print(f"Variations generated: {data.get('count')}")
            print(f"Average time per image: {elapsed/data.get('count', 1):.1f}s")
            return True
        else:
            data = response.json()
            print(f"❌ Failed: {data.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🧪 FloorMind Backend Test Suite".center(60))
    print("="*60)
    
    results = {}
    
    # Test 1: Health check
    results['health'] = test_health()
    
    if not results['health']:
        print("\n❌ Server is not running!")
        print("\nPlease start the server first:")
        print("  ./start_backend.sh")
        print("  OR")
        print("  cd backend && python app_clean.py")
        return
    
    # Test 2: Model info
    results['model_info'] = test_model_info()
    
    # Test 3: Load model if not loaded
    if not results['model_info']:
        results['model_load'] = test_model_load()
        if not results['model_load']:
            print("\n❌ Cannot proceed without model loaded")
            return
    
    # Test 4: Presets
    results['presets'] = test_presets()
    
    # Test 5: Single generation
    print("\n⚠️  The next tests will take several minutes...")
    input("Press Enter to continue with generation tests (or Ctrl+C to stop)...")
    
    results['generation'] = test_generation()
    
    # Test 6: Batch generation (optional)
    if results['generation']:
        print("\n⚠️  Batch generation will take even longer...")
        response = input("Run batch generation test? (y/N): ")
        if response.lower() == 'y':
            results['batch'] = test_batch_generation()
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test.ljust(20)}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
