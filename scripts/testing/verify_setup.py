#!/usr/bin/env python3
"""
FloorMind Setup Verification Script
Checks that everything is properly configured and working
"""

import os
import sys
import requests
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_mark(condition):
    return "✅" if condition else "❌"

def check_files():
    """Check that all required files exist"""
    print_header("Checking Required Files")
    
    required_files = {
        "Backend": [
            "backend/app_clean.py",
            "backend/model_loader.py",
        ],
        "Scripts": [
            "start_backend.sh",
            "test_backend.py",
            "check_model_quality.py",
        ],
        "Documentation": [
            "README.md",
            "QUICK_START.md",
            "docs/BACKEND_GUIDE.md",
            "docs/FRONTEND_INTEGRATION_GUIDE.md",
        ],
        "Models": [
            "models/floormind_pipeline/model_index.json",
            "models/floormind_pipeline/unet/diffusion_pytorch_model.safetensors",
        ]
    }
    
    all_good = True
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            exists = os.path.exists(file)
            print(f"  {check_mark(exists)} {file}")
            if not exists:
                all_good = False
    
    return all_good

def check_model():
    """Check model integrity"""
    print_header("Checking Model Integrity")
    
    model_path = "models/floormind_pipeline"
    
    # Check structure
    required_components = ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]
    all_good = True
    
    for component in required_components:
        component_path = os.path.join(model_path, component)
        exists = os.path.exists(component_path)
        print(f"  {check_mark(exists)} {component}/")
        if not exists:
            all_good = False
    
    # Check UNet weights
    unet_weights = os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
    if os.path.exists(unet_weights):
        size_mb = os.path.getsize(unet_weights) / (1024 * 1024)
        print(f"  ✅ UNet weights: {size_mb:.1f} MB")
    else:
        print(f"  ❌ UNet weights missing!")
        all_good = False
    
    return all_good

def check_server():
    """Check if server is running"""
    print_header("Checking Server Status")
    
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Server is running")
            print(f"  ✅ Status: {data.get('status')}")
            print(f"  ✅ Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"  ❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ⚠️  Server is not running")
        print(f"  💡 Start it with: ./start_backend.sh")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def check_api():
    """Check API endpoints"""
    print_header("Checking API Endpoints")
    
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/model/info", None),
        ("GET", "/presets", None),
    ]
    
    all_good = True
    for method, endpoint, data in endpoints:
        try:
            url = f"http://localhost:5001{endpoint}"
            if method == "GET":
                response = requests.get(url, timeout=5)
            else:
                response = requests.post(url, json=data, timeout=5)
            
            success = response.status_code == 200
            print(f"  {check_mark(success)} {method} {endpoint} - {response.status_code}")
            if not success:
                all_good = False
        except Exception as e:
            print(f"  ❌ {method} {endpoint} - Error: {e}")
            all_good = False
    
    return all_good

def check_dependencies():
    """Check Python dependencies"""
    print_header("Checking Python Dependencies")
    
    required_packages = [
        "flask",
        "flask_cors",
        "torch",
        "diffusers",
        "transformers",
        "PIL",
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Not installed")
            all_good = False
    
    return all_good

def check_cleanup():
    """Check that cleanup was successful"""
    print_header("Checking Cleanup Status")
    
    # Check that old files are archived
    archived = os.path.exists(".archive")
    print(f"  {check_mark(archived)} Old files archived")
    
    # Check that docs are organized
    docs_exist = os.path.exists("docs")
    print(f"  {check_mark(docs_exist)} Documentation organized")
    
    # Check that incomplete model is removed
    incomplete_removed = not os.path.exists("models/final_model_production")
    print(f"  {check_mark(incomplete_removed)} Incomplete model removed")
    
    return archived and docs_exist and incomplete_removed

def main():
    """Run all checks"""
    print("\n" + "🔍 FloorMind Setup Verification".center(60))
    print("="*60)
    
    results = {
        "Files": check_files(),
        "Model": check_model(),
        "Dependencies": check_dependencies(),
        "Cleanup": check_cleanup(),
        "Server": check_server(),
    }
    
    # Only check API if server is running
    if results["Server"]:
        results["API"] = check_api()
    
    # Summary
    print_header("Summary")
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check.ljust(20)}: {status}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\n  Total: {passed_count}/{total_count} checks passed")
    
    if passed_count == total_count:
        print("\n🎉 All checks passed! Your setup is perfect!")
        print("\n📝 Next steps:")
        print("   1. Generate test images: python test_backend.py")
        print("   2. Integrate with frontend: See docs/FRONTEND_INTEGRATION_GUIDE.md")
        print("   3. Deploy to production: See docs/BACKEND_GUIDE.md")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the issues above.")
        
        if not results.get("Server"):
            print("\n💡 Quick fix: Start the server with ./start_backend.sh")
        
        if not results.get("Dependencies"):
            print("\n💡 Quick fix: pip install -r requirements.txt")
        
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
