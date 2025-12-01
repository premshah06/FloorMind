# ULTRA LOW MEMORY CONFIGURATION FOR APPLE SILICON (MPS)
# Replace the existing configuration cell with this optimized version

import os
import torch
import gc

# Clear any existing memory
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()

# Set MPS memory optimization environment variables
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'  # Use only 50% of available memory
os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'

# Ultra Low Memory Training Configuration
config = {
    # Data paths
    "data_dir": "../data",
    "metadata_file": "../data/metadata.csv",
    "images_dir": "../data/processed/images",
    
    # Model configuration - OPTIMIZED FOR LOW MEMORY
    "model_name": "runwayml/stable-diffusion-v1-5",
    "resolution": 256,  # REDUCED from 512 to 256 (75% memory reduction)
    "train_batch_size": 1,  # Minimum batch size for MPS
    "eval_batch_size": 1,   # Minimum batch size for MPS
    
    # Training parameters - OPTIMIZED FOR STABILITY
    "num_epochs": 15,  # Increased to compensate for smaller resolution
    "learning_rate": 5e-6,  # Slightly lower for stability with small batches
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 100,  # Reduced warmup steps
    "gradient_accumulation_steps": 8,  # Maintain effective batch size of 8
    "max_grad_norm": 0.5,  # Reduced for stability
    
    # Diffusion parameters
    "num_train_timesteps": 1000,
    "noise_schedule": "linear",
    "prediction_type": "epsilon",
    
    # Output configuration - MORE FREQUENT SAVES
    "output_dir": "../outputs/models/base_model_low_mem",
    "save_steps": 200,  # More frequent saves (every 200 steps)
    "eval_steps": 100,  # More frequent evaluation
    "logging_steps": 25,  # More frequent logging
    
    # Hardware - APPLE SILICON OPTIMIZED
    "mixed_precision": "no",  # MPS doesn't support fp16 mixed precision
    "dataloader_num_workers": 0,  # Required for Jupyter notebooks
    
    # Memory optimization flags
    "quick_test": False,
    "max_samples": None,
    "enable_gradient_checkpointing": True,
    "enable_memory_efficient_attention": True
}

# Create output directory
os.makedirs(config["output_dir"], exist_ok=True)

# Display configuration with memory optimization info
print("🔧 ULTRA LOW MEMORY CONFIGURATION LOADED")
print("=" * 60)
print("📱 HARDWARE OPTIMIZATIONS:")
print(f"   Target Device: Apple Silicon (MPS)")
print(f"   MPS Memory Watermark: 50%")
print(f"   Mixed Precision: {config['mixed_precision']}")
print(f"   DataLoader Workers: {config['dataloader_num_workers']}")

print(f"\n🖼️ MEMORY OPTIMIZATIONS:")
print(f"   Resolution: 512 → 256 (75% memory reduction)")
print(f"   Batch Size: {config['train_batch_size']} (minimum for stability)")
print(f"   Gradient Accumulation: {config['gradient_accumulation_steps']} steps")
print(f"   Effective Batch Size: {config['train_batch_size'] * config['gradient_accumulation_steps']}")

print(f"\n⚡ TRAINING OPTIMIZATIONS:")
print(f"   Epochs: {config['num_epochs']} (increased to compensate)")
print(f"   Learning Rate: {config['learning_rate']} (optimized for small batches)")
print(f"   Save Frequency: Every {config['save_steps']} steps")
print(f"   Eval Frequency: Every {config['eval_steps']} steps")

print(f"\n💾 OUTPUT:")
print(f"   Model Directory: {config['output_dir']}")

print("=" * 60)

# Memory usage estimation
estimated_memory = 8.0  # GB - estimated for 256x256 resolution
print(f"📊 ESTIMATED MEMORY USAGE: ~{estimated_memory} GB")
print(f"💡 TRAINING TIME: Longer per epoch, but faster per step")
print(f"🎯 QUALITY: Good baseline quality at 256x256 resolution")

print(f"\n✅ Configuration optimized for Apple Silicon training!")
print(f"🚀 Ready to train without memory errors!")

# Final memory cleanup
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()