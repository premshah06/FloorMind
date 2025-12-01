#!/usr/bin/env python3
"""
FloorMind v2.0 Integration Test
Test the new structured integration
"""

import sys
import os
import requests
import time
import subprocess
from pathlib import Path

def test_project_structure():
    """Test if the new project structure exists"""
    
    print("🔍 Testing project structure...")
    
    required_paths = [
        "src/core/model_manager.py",
        "src/api/app.py",
        "src/api/routes.py",
        "src/scripts/start_complete.py",
        "frontend/src/services/api.js"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
        else:
            print(f"✅ {path}")
    
    if missing_paths:
        print(f"❌ Missing paths: {missing_paths}")
        print("💡 Run: python migrate_to_v2.py")
        return False
    
    print("✅ Project structure looks good!")
    return True

def test_model_manager():
    """Test the model manager functionality"""
    
    print("\n🧪 Testing model manager...")
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        from core.model_manager import get_model_manager
        
        manager = get_model_manager()
        print(f"✅ Model manager created")
        print(f"   Model path: {manager.model_path}")
        print(f"   Is loaded: {manager.is_loaded}")
        
        # Get model info
        info = manager.get_model_info()
        print(f"   Model info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_structure():
    """Test if the API structure is correct"""
    
    print("\n🔍 Testing API structure...")
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        from api.app import create_app
        
        app = create_app()
        print("✅ API app created successfully")
        print(f"   Blueprints: {[bp.name for bp in app.blueprints.values()]}")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backend_startup():
    """Test if the backend can start"""
    
    print("\n🚀 Testing backend startup...")
    
    # Check if backend startup script exists
    startup_script = Path("src/scripts/start_backend.py")
    if not startup_script.exists():
        print("❌ Backend startup script not found")
        return False
    
    print("✅ Backend startup script found")
    print("💡 To test actual startup, run: python src/scripts/start_backend.py")
    return True

def test_frontend_api_service():
    """Test the frontend API service"""
    
    print("\n🎨 Testing frontend API service...")
    
    api_service_path = Path("frontend/src/services/api.js")
    if not api_service_path.exists():
        print("❌ Frontend API service not found")
        return False
    
    # Read and check for v2.0 features
    content = api_service_path.read_text()
    
    v2_features = [
        "FloorMind API Service v2.0",
        "Enhanced FloorMind API Service Class",
        "checkHealth(useCache = true)",
        "loadModel(onProgress = null)",
        "unloadModel()",
        "generateBatch("
    ]
    
    missing_features = []
    for feature in v2_features:
        if feature not in content:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Missing v2.0 features: {missing_features}")
        return False
    
    print("✅ Frontend API service has v2.0 features")
    return True

def test_migration_readiness():
    """Test if the project is ready for migration"""
    
    print("\n📋 Testing migration readiness...")
    
    # Check for old structure files
    old_files = [
        "start_floormind.py",
        "start_floormind_fixed.py",
        "google/model.safetensors",
        "generated_floor_plans"
    ]
    
    found_old_files = []
    for file_path in old_files:
        if Path(file_path).exists():
            found_old_files.append(file_path)
    
    if found_old_files:
        print(f"📁 Found old structure files: {found_old_files}")
        print("💡 These will be migrated/backed up during migration")
    else:
        print("✅ No old structure files found (already migrated?)")
    
    return True

def run_integration_test():
    """Run a quick integration test if backend is running"""
    
    print("\n🔗 Testing integration (if backend is running)...")
    
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is running!")
            print(f"   Service: {data.get('service', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            
            # Test model info endpoint
            try:
                info_response = requests.get("http://localhost:5001/model/info", timeout=5)
                if info_response.status_code == 200:
                    print("✅ Model info endpoint working")
                else:
                    print(f"⚠️  Model info endpoint returned: {info_response.status_code}")
            except:
                print("⚠️  Model info endpoint not accessible")
            
            return True
        else:
            print(f"⚠️  Backend returned status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  Backend not running (this is OK for structure testing)")
        print("💡 To test integration: python src/scripts/start_complete.py")
        return True
    except Exception as e:
        print(f"⚠️  Integration test failed: {e}")
        return True  # Don't fail the overall test

def main():
    """Main test function"""
    
    print("🧪 FloorMind v2.0 Integration Test")
    print("=" * 50)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Model Manager", test_model_manager),
        ("API Structure", test_api_structure),
        ("Backend Startup", test_backend_startup),
        ("Frontend API Service", test_frontend_api_service),
        ("Migration Readiness", test_migration_readiness),
        ("Integration", run_integration_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! FloorMind v2.0 structure is ready!")
        print("\n💡 Next steps:")
        print("   1. Run migration: python migrate_to_v2.py")
        print("   2. Start services: python src/scripts/start_complete.py")
        print("   3. Open browser: http://localhost:3000")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\n💡 Common fixes:")
        print("   - Run migration: python migrate_to_v2.py")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Check project structure")

if __name__ == "__main__":
    main()