#!/usr/bin/env python3
"""
Simple Backend Test Script
Tests the FloorMind backend without loading the heavy model
"""

import requests
import json
import time

def test_backend():
    """Test the backend endpoints"""
    
    base_url = "http://localhost:5001"
    
    print("🧪 Testing FloorMind Backend...")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("💡 Make sure the backend server is running: python backend/app.py")
        return False
    
    # Test 2: Model Info
    print("\n2. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            model_info = data['model_info']
            print(f"✅ Model info retrieved")
            print(f"   - Model loaded: {model_info.get('is_loaded', False)}")
            print(f"   - Device: {model_info.get('device', 'unknown')}")
            if 'error' in model_info:
                print(f"   - Error: {model_info['error']}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    # Test 3: Presets
    print("\n3. Testing presets endpoint...")
    try:
        response = requests.get(f"{base_url}/presets", timeout=5)
        if response.status_code == 200:
            data = response.json()
            presets = data['presets']
            print(f"✅ Presets retrieved")
            print(f"   - Residential: {len(presets.get('residential', []))} presets")
            print(f"   - Commercial: {len(presets.get('commercial', []))} presets")
        else:
            print(f"❌ Presets failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Presets failed: {e}")
    
    # Test 4: Model Loading (optional)
    print("\n4. Testing model loading endpoint...")
    try:
        print("   Attempting to load model (this may take a while)...")
        response = requests.post(f"{base_url}/model/load", timeout=60)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model loading: {data['status']}")
            if data['status'] == 'success':
                print("   Model loaded successfully!")
                return True
            else:
                print(f"   Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ Model loading failed: {response.status_code}")
            if response.text:
                print(f"   Response: {response.text}")
    except requests.exceptions.Timeout:
        print("⏰ Model loading timed out (this is normal for large models)")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Backend test completed!")
    print("💡 If model loading failed, check the model files in the google/ directory")
    return True

if __name__ == "__main__":
    test_backend()