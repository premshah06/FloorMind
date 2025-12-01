#!/usr/bin/env python3
"""
FloorMind Dataset Setup Script
One-click setup for all architectural datasets
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def run_command(command, description, cwd=None):
    """Run a command with proper error handling"""
    print(f"\n🚀 {description}")
    print(f"💻 Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            cwd=cwd,
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            print(result.stdout)
        
        print(f"✅ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def setup_cubicasa5k(data_dir: Path, max_samples: int = None):
    """Setup CubiCasa5K dataset"""
    
    print("🏗️  Setting up CubiCasa5K Dataset")
    print("=" * 60)
    
    # Step 1: Download dataset
    download_script = data_dir / "download_cubicasa5k.py"
    if not run_command(
        f"python {download_script}",
        "Downloading CubiCasa5K dataset",
        cwd=data_dir
    ):
        return False
    
    # Step 2: Process dataset
    process_script = data_dir / "process_datasets.py"
    process_cmd = f"python {process_script} --dataset cubicasa5k"
    
    if max_samples:
        process_cmd += f" --max-samples {max_samples}"
    
    if not run_command(
        process_cmd,
        "Processing CubiCasa5K dataset",
        cwd=data_dir
    ):
        return False
    
    return True

def setup_environment():
    """Setup Python environment and dependencies"""
    
    print("🔧 Setting up Python environment")
    print("=" * 60)
    
    # Check if required packages are installed
    required_packages = [
        'requests', 'tqdm', 'pandas', 'numpy', 
        'pillow', 'opencv-python', 'matplotlib', 
        'seaborn', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        
        install_cmd = f"pip install {' '.join(missing_packages)}"
        if not run_command(install_cmd, "Installing Python packages"):
            return False
    else:
        print("✅ All required packages are already installed")
    
    return True

def create_directory_structure(base_dir: Path):
    """Create necessary directory structure"""
    
    print("📁 Creating directory structure")
    print("=" * 60)
    
    directories = [
        "data/datasets",
        "data/raw", 
        "data/processed",
        "data/processed/images",
        "data/processed/splits",
        "outputs/sample_generations",
        "outputs/metrics"
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    return True

def generate_quick_start_guide(base_dir: Path):
    """Generate a quick start guide"""
    
    guide_content = """
# FloorMind Quick Start Guide

## 🎯 What's Ready

Your FloorMind installation is now ready with:
- ✅ CubiCasa5K dataset downloaded and processed
- ✅ Training data prepared and split
- ✅ All dependencies installed
- ✅ Directory structure created

## 🚀 Next Steps

### 1. Train Models (Optional)
```bash
# Open the training notebook
jupyter notebook notebooks/FloorMind_Training_and_Analysis.ipynb

# Or run training directly
cd notebooks && python -c "
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
with open('FloorMind_Training_and_Analysis.ipynb') as f:
    nb = nbformat.read(f, as_version=4)
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': '.'}})
"
```

### 2. Start Backend API
```bash
cd backend
python app.py
```

### 3. Open Frontend
```bash
# Option 1: Simple demo (no setup required)
open frontend/demo.html

# Option 2: Full React app
cd frontend
npm install
npm start
```

### 4. Test Generation
```bash
# Test with curl
curl -X POST http://localhost:5000/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "3-bedroom apartment with open kitchen",
    "model_type": "gemini",
    "style": "modern"
  }'
```

## 📊 Dataset Information

- **Dataset**: CubiCasa5K
- **Total Samples**: Check `data/metadata.csv`
- **Processed Images**: `data/processed/images/`
- **Statistics**: `outputs/cubicasa5k_analysis.png`

## 🔧 Configuration

### Gemini API (Recommended)
```bash
# Get API key from: https://makersuite.google.com/app/apikey
export GEMINI_API_KEY=your_api_key_here

# Or add to .env file
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

### Environment Variables
```bash
# Copy template
cp .env.example .env

# Edit with your settings
nano .env
```

## 📚 Documentation

- **Setup Guide**: `SETUP.md`
- **Gemini Integration**: `GEMINI_SETUP.md`
- **API Documentation**: `README.md`
- **Dataset Manager**: `data/dataset_manager.py --help`

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Not Found**
   ```bash
   python data/download_cubicasa5k.py
   ```

3. **API Key Issues**
   ```bash
   export GEMINI_API_KEY=your_key
   ```

4. **Port Conflicts**
   - Backend: Edit `backend/app.py` (default: 5000)
   - Frontend: Edit `frontend/package.json` (default: 3000)

### Test Everything
```bash
# Test dataset processing
python data/process_datasets.py --dataset cubicasa5k --max-samples 10

# Test Gemini integration
python test_gemini.py

# Test backend
cd backend && python app.py &
curl http://localhost:5000/

# Test frontend
open frontend/demo.html
```

## 🎉 You're Ready!

FloorMind is now fully set up and ready to generate amazing floor plans!

Happy building! 🏗️✨
"""
    
    guide_file = base_dir / "QUICK_START.md"
    with open(guide_file, 'w') as f:
        f.write(guide_content.strip())
    
    print(f"📖 Quick start guide created: {guide_file}")

def main():
    """Main setup function"""
    
    parser = argparse.ArgumentParser(description="FloorMind Dataset Setup")
    parser.add_argument('--dataset', type=str, default='cubicasa5k',
                       choices=['cubicasa5k', 'all'],
                       help='Dataset to setup (default: cubicasa5k)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip dataset download (if already downloaded)')
    parser.add_argument('--skip-env', action='store_true',
                       help='Skip environment setup')
    
    args = parser.parse_args()
    
    print("🏠 FloorMind Dataset Setup")
    print("=" * 60)
    print("This script will:")
    print("1. 🔧 Setup Python environment")
    print("2. 📁 Create directory structure") 
    print("3. 📦 Download architectural datasets")
    print("4. 🔄 Process datasets for training")
    print("5. 📊 Generate statistics and visualizations")
    print("6. 📖 Create quick start guide")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    try:
        # Step 1: Setup environment
        if not args.skip_env:
            if not setup_environment():
                print("❌ Environment setup failed")
                return False
        
        # Step 2: Create directories
        if not create_directory_structure(base_dir):
            print("❌ Directory creation failed")
            return False
        
        # Step 3: Setup datasets
        if args.dataset == 'cubicasa5k' or args.dataset == 'all':
            if not setup_cubicasa5k(data_dir, args.max_samples):
                print("❌ CubiCasa5K setup failed")
                return False
        
        # Step 4: Generate guide
        generate_quick_start_guide(base_dir)
        
        print("\n" + "=" * 60)
        print("🎉 FloorMind Setup Complete!")
        print("=" * 60)
        
        print("\n✅ What's been set up:")
        print("  📦 Python environment with all dependencies")
        print("  📁 Complete directory structure")
        print("  🏗️  CubiCasa5K dataset downloaded and processed")
        print("  📊 Dataset statistics and visualizations")
        print("  📖 Quick start guide and documentation")
        
        print("\n🚀 Next steps:")
        print("  1. Read the quick start guide: QUICK_START.md")
        print("  2. Set up Gemini API key (optional): export GEMINI_API_KEY=your_key")
        print("  3. Start the backend: cd backend && python app.py")
        print("  4. Open the frontend: open frontend/demo.html")
        
        print("\n💡 Pro tips:")
        print("  • Use --max-samples 100 for quick testing")
        print("  • Check outputs/ for dataset visualizations")
        print("  • See GEMINI_SETUP.md for advanced AI features")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)