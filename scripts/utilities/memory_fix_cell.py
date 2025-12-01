# EMERGENCY MEMORY FIX FOR MPS
# Copy and paste this into a new notebook cell and run it before training

import torch
import os
import gc

# Clear any existing memory
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()

# Set MPS memory optimization
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'

# Update configuration for MPS compatibility
config.update({
    "train_batch_size": 1,
    "eval_batch_size": 1, 
    "gradient_accumulation_steps": 4,
    "mixed_precision": "no"
})

print("🔧 MPS Memory Fix Applied:")
print(f"   Batch size: {config['train_batch_size']}")
print(f"   Gradient accumulation: {config['gradient_accumulation_steps']}")
print(f"   Mixed precision: {config['mixed_precision']}")
print(f"   MPS watermark: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'not set')}")
print("\n✅ Ready to restart training with optimized settings!")