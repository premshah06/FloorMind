#!/usr/bin/env python3
"""
FloorMind Integration Test
Tests the complete integration between frontend and backend
"""

import requests
import json
import time
import subprocess
import sys
import os

def test_backend_endpoints():
    """Test all backend endpoints"""
    
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Backend Endpoints")
    print("=" * 40)
    
    # Test 1: Health Check
    print("\n1. Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"   Service: {data['service']}")
            print(f"   Model loaded: {data['model_loaded']}")
        else:
            print(f"❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Model Info
    print("\n2. Model Info...")
    try:
        response = requests.get(f"{base_url}/model/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            model_info = data['model_info']
            print(f"✅ Model loaded: {model_info.get('is_loaded', False)}")
            print(f"   Device: {model_info.get('device', 'unknown')}")
            if 'error' in model_info:
                print(f"   Error: {model_info['error']}")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Presets
    print("\n3. Presets...")
    try:
        response = requests.get(f"{base_url}/presets", timeout=5)
        if response.status_code == 200:
            data = response.json()
            presets = data['presets']
            print(f"✅ Residential: {len(presets.get('residential', []))} presets")
            print(f"   Commercial: {len(presets.get('commercial', []))} presets")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Model Loading
    print("\n4. Model Loading...")
    try:
        print("   Attempting to load model...")
        response = requests.post(f"{base_url}/model/load", timeout=120)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            if data['status'] == 'success':
                print("   Model loaded successfully!")
                return True
            else:
                print(f"   Message: {data.get('message', 'Unknown')}")
        else:
            print(f"❌ Failed: {response.status_code}")
            if response.text:
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text}")
    except requests.exceptions.Timeout:
        print("⏰ Timeout (normal for large models)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return True

def test_generation():
    """Test floor plan generation"""
    
    base_url = "http://localhost:5001"
    
    print("\n🎨 Testing Generation")
    print("=" * 40)
    
    # Test generation
    print("\n1. Testing generation...")
    try:
        test_prompt = "Modern 2-bedroom apartment with open kitchen"
        
        payload = {
            "description": test_prompt,
            "width": 512,
            "height": 512,
            "steps": 10,  # Reduced for faster testing
            "guidance": 7.5,
            "seed": 42,
            "save": True
        }
        
        print(f"   Prompt: {test_prompt}")
        print("   Generating (this may take a while)...")
        
        response = requests.post(f"{base_url}/generate", json=payload, timeout=300)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Generation successful!")
            print(f"   Status: {data['status']}")
            print(f"   Description: {data['description']}")
            print(f"   Image size: {len(data['image'])} characters")
            if 'saved_path' in data and data['saved_path']:
                print(f"   Saved to: {data['saved_path']}")
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            if response.text:
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Generation timeout (model may be loading)")
        return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def check_frontend():
    """Check if frontend is accessible"""
    
    print("\n🎨 Testing Frontend")
    print("=" * 40)
    
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible at http://localhost:3000")
            return True
        else:
            print(f"❌ Frontend returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend not accessible: {e}")
        print("💡 Make sure to start the frontend with: npm start (in frontend directory)")
        return False

def main():
    """Main test function"""
    
    print("🧪 FloorMind Integration Test")
    print("=" * 50)
    
    # Test backend
    if not test_backend_endpoints():
        print("\n❌ Backend tests failed!")
        print("💡 Make sure to start the backend with: python backend/app.py")
        return
    
    # Test generation (optional)
    print("\n" + "=" * 50)
    user_input = input("🎨 Test generation? (y/n, may take several minutes): ").lower()
    if user_input == 'y':
        test_generation()
    
    # Test frontend
    print("\n" + "=" * 50)
    check_frontend()
    
    print("\n" + "=" * 50)
    print("✅ Integration test completed!")
    print("\n💡 Next steps:")
    print("   1. Start backend: python start_backend.py")
    print("   2. Start frontend: python start_frontend.py")
    print("   3. Or start both: python start_floormind_fixed.py")
    print("   4. Open http://localhost:3000 in your browser")

if __name__ == "__main__":
    main()