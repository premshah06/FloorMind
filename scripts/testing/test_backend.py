#!/usr/bin/env python3
"""
Test FloorMind Backend
Quick test to verify the backend is working with your trained model
"""

import requests
import json
import base64
from PIL import Image
import io
import os

def test_backend_health():
    """Test backend health endpoint"""
    
    try:
        response = requests.get("http://localhost:5000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is healthy!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    
    try:
        response = requests.get("http://localhost:5000/model/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved!")
            print(f"   Model loaded: {data['model_info'].get('is_loaded')}")
            print(f"   Device: {data['model_info'].get('device')}")
            print(f"   Resolution: {data['model_info'].get('resolution')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_generation():
    """Test floor plan generation"""
    
    try:
        # Test generation request
        test_request = {
            "description": "Modern 2-bedroom apartment with open kitchen",
            "width": 512,
            "height": 512,
            "steps": 15,  # Faster for testing
            "guidance": 7.5,
            "seed": 42,
            "save": True
        }
        
        print(f"🎨 Testing generation: {test_request['description']}")
        print("⏳ This may take 10-30 seconds...")
        
        response = requests.post(
            "http://localhost:5000/generate", 
            json=test_request,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Generation successful!")
            print(f"   Description: {data.get('description')}")
            print(f"   Timestamp: {data.get('timestamp')}")
            print(f"   Saved path: {data.get('saved_path')}")
            
            # Save the generated image locally for verification
            if 'image' in data:
                try:
                    # Decode base64 image
                    image_data = data['image'].split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    
                    # Save image
                    with open("test_generated_floor_plan.png", "wb") as f:
                        f.write(image_bytes)
                    
                    print("💾 Test image saved as: test_generated_floor_plan.png")
                    
                    # Verify image
                    img = Image.open(io.BytesIO(image_bytes))
                    print(f"   Image size: {img.size}")
                    print(f"   Image mode: {img.mode}")
                    
                except Exception as e:
                    print(f"⚠️ Could not save test image: {e}")
            
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Generation timed out (>60 seconds)")
        return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 FloorMind Backend Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_backend_health),
        ("Model Info", test_model_info),
        ("Generation Test", test_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n🔬 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\\n" + "=" * 50)
    print(f"🏁 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your FloorMind backend is working perfectly!")
        print("\\n🚀 Next steps:")
        print("1. Start the frontend: cd frontend && npm start")
        print("2. Open http://localhost:3000 in your browser")
        print("3. Go to the Generator page and test floor plan generation")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\\n🔧 Troubleshooting:")
        print("1. Make sure the backend server is running: python backend/app.py")
        print("2. Check that your model files are in the google/ directory")
        print("3. Verify PyTorch and diffusers are installed")

if __name__ == "__main__":
    main()