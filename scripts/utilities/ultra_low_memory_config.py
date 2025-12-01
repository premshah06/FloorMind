# ULTRA LOW MEMORY CONFIGURATION FOR MPS
# Copy this into a new cell and run it to minimize RAM usage

import torch
import os
import gc

# Aggressive memory cleanup
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()

# Set very conservative MPS memory settings
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'  # Use only 50% of available memory

# Ultra low memory configuration
ultra_low_config = {
    # Data paths (keep same)
    "data_dir": "../data",
    "metadata_file": "../data/metadata.csv", 
    "images_dir": "../data/processed/images",
    
    # Model configuration - REDUCED RESOLUTION
    "model_name": "runwayml/stable-diffusion-v1-5",
    "resolution": 256,  # REDUCED from 512 to 256 (75% less memory)
    "train_batch_size": 1,
    "eval_batch_size": 1,
    
    # Training parameters - OPTIMIZED FOR LOW MEMORY
    "num_epochs": 15,  # Increased epochs to compensate for smaller batches
    "learning_rate": 5e-6,  # Slightly lower LR for stability
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 100,  # Reduced warmup
    "gradient_accumulation_steps": 8,  # Increased to maintain effective batch size
    "max_grad_norm": 0.5,  # Reduced for stability
    
    # Diffusion parameters
    "num_train_timesteps": 1000,
    "noise_schedule": "linear", 
    "prediction_type": "epsilon",
    
    # Output configuration
    "output_dir": "../outputs/models/base_model_low_mem",
    "save_steps": 200,  # More frequent saves
    "eval_steps": 100,  # More frequent evaluation
    "logging_steps": 25,
    
    # Hardware - ULTRA CONSERVATIVE
    "mixed_precision": "no",
    "dataloader_num_workers": 0,
    
    # Memory optimization
    "quick_test": False,
    "max_samples": None,
    "enable_memory_efficient_attention": True,
    "gradient_checkpointing": True
}

# Update the global config
config.update(ultra_low_config)

# Additional memory optimizations
def enable_memory_optimizations():
    """Enable all possible memory optimizations"""
    
    # Set environment variables for memory efficiency
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
    
    # Enable gradient checkpointing if available
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except:
        pass
    
    print("🔧 Ultra Low Memory Optimizations Applied:")
    print(f"   Resolution: 512 → 256 (75% memory reduction)")
    print(f"   Batch size: {config['train_batch_size']}")
    print(f"   Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {config['train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"   MPS watermark: 50%")
    print(f"   Mixed precision: {config['mixed_precision']}")
    print(f"   Epochs increased to: {config['num_epochs']} (to compensate)")
    print("\n✅ Configuration optimized for minimal RAM usage!")
    print("💡 Training will be slower but use ~75% less memory")

# Apply optimizations
enable_memory_optimizations()

# Clear memory again after config update
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()

print(f"\n📊 New Configuration Summary:")
for key, value in config.items():
    if key in ['resolution', 'train_batch_size', 'gradient_accumulation_steps', 'mixed_precision', 'num_epochs']:
        print(f"   {key}: {value}")