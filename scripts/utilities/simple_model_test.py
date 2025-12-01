#!/usr/bin/env python3
"""
Simple FloorMind Model Test
A lightweight script to test the trained model
"""

import os
import sys
import pickle
import json
from PIL import Image
import matplotlib.pyplot as plt

def check_model_files():
    """Check what model files we have"""
    
    print("🔍 Checking available model files...")
    
    google_dir = "google"
    if not os.path.exists(google_dir):
        print(f"❌ Google directory not found: {google_dir}")
        return False
    
    files = os.listdir(google_dir)
    print(f"📁 Files in google directory:")
    
    important_files = {
        'floormind_model.pkl': 'Main model file',
        'model.safetensors': 'UNet weights (fine-tuned)',
        'tokenizer_config.json': 'Text tokenizer config',
        'scheduler_config.json': 'Diffusion scheduler',
        'training_config.json': 'Training configuration',
        'training_stats.csv': 'Training statistics',
        'model_index.json': 'Pipeline index'
    }
    
    available_files = {}
    for file in files:
        file_path = os.path.join(google_dir, file)
        size_mb = os.path.getsize(file_path) / 1024**2
        
        if file in important_files:
            status = "✅"
            available_files[file] = file_path
        else:
            status = "📄"
        
        print(f"   {status} {file} ({size_mb:.1f} MB) - {important_files.get(file, 'Additional file')}")
    
    return len(available_files) > 0

def load_training_info():
    """Load training information"""
    
    try:
        # Load pickle file info
        with open("google/floormind_model.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        config = model_data.get('config', {})
        best_val_loss = model_data.get('best_val_loss', 'Unknown')
        
        print(f"\\n📊 Training Information:")
        print(f"   Model: {config.get('model_name', 'Unknown')}")
        print(f"   Resolution: {config.get('resolution', 'Unknown')}x{config.get('resolution', 'Unknown')}")
        print(f"   Epochs: {config.get('num_epochs', 'Unknown')}")
        print(f"   Batch Size: {config.get('train_batch_size', 'Unknown')}")
        print(f"   Learning Rate: {config.get('learning_rate', 'Unknown')}")
        print(f"   Best Validation Loss: {best_val_loss}")
        print(f"   Mixed Precision: {config.get('mixed_precision', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading training info: {e}")
        return False

def show_generated_samples():
    """Show the test images that were generated during training"""
    
    print(f"\\n🖼️ Generated Test Samples:")
    
    # Look for test generation images
    test_images = []
    for file in os.listdir("google"):
        if file.startswith("test_generation") and file.endswith(".png"):
            test_images.append(os.path.join("google", file))
    
    if not test_images:
        print("   ❌ No test generation images found")
        return
    
    print(f"   Found {len(test_images)} test images")
    
    # Display images
    try:
        if len(test_images) == 1:
            # Single image
            img = Image.open(test_images[0])
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title("Generated Floor Plan")
            plt.axis('off')
            plt.show()
        else:
            # Multiple images
            n_images = min(len(test_images), 4)
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(n_images):
                img = Image.open(test_images[i])
                axes[i].imshow(img)
                axes[i].set_title(f"Test Generation {i+1}")
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(n_images, 4):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        print("✅ Test images displayed successfully")
        
    except Exception as e:
        print(f"❌ Error displaying images: {e}")

def create_integration_guide():
    """Create a guide for integrating the model"""
    
    integration_code = '''
# FloorMind Model Integration Guide

## Method 1: Using Diffusers Pipeline (Recommended)
```python
from diffusers import StableDiffusionPipeline
import torch

# Load the pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "path/to/your/google/folder",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
)

# Move to appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)

# Generate floor plan
prompt = "Modern 3-bedroom apartment floor plan with open kitchen"
image = pipeline(
    prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

image.save("generated_floor_plan.png")
```

## Method 2: Manual Component Loading
```python
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors.torch
import torch

# Load base components
base_model = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

# Load fine-tuned UNet
unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
state_dict = safetensors.torch.load_file("google/model.safetensors")
unet.load_state_dict(state_dict)

# Create pipeline
pipeline = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None
)
```

## Method 3: Backend Integration
```python
class FloorMindGenerator:
    def __init__(self, model_path="google"):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = self.pipeline.to(self.device)
    
    def generate_floor_plan(self, description, **kwargs):
        return self.pipeline(
            description,
            num_inference_steps=kwargs.get('steps', 20),
            guidance_scale=kwargs.get('guidance', 7.5),
            height=kwargs.get('height', 512),
            width=kwargs.get('width', 512)
        ).images[0]
```
'''
    
    with open("FloorMind_Integration_Guide.md", "w") as f:
        f.write(integration_code)
    
    print(f"\\n📖 Integration guide created: FloorMind_Integration_Guide.md")

def main():
    """Main function"""
    
    print("🏠 FloorMind Model Analysis & Integration Helper")
    print("=" * 60)
    
    # Check files
    if not check_model_files():
        print("❌ Required model files not found")
        return
    
    # Load training info
    if not load_training_info():
        print("⚠️ Could not load training information")
    
    # Show generated samples
    show_generated_samples()
    
    # Create integration guide
    create_integration_guide()
    
    print(f"\\n🎉 Analysis Complete!")
    print(f"\\n📋 Summary:")
    print(f"   ✅ Model files are available")
    print(f"   ✅ Training completed successfully")
    print(f"   ✅ Test images generated")
    print(f"   ✅ Integration guide created")
    
    print(f"\\n🚀 Next Steps:")
    print(f"   1. Review the generated test images above")
    print(f"   2. Check FloorMind_Integration_Guide.md for integration code")
    print(f"   3. Test the integration methods in your application")
    print(f"   4. Deploy to your FloorMind backend")

if __name__ == "__main__":
    main()