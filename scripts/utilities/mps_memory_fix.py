# MPS Memory Optimization for Apple Silicon
# Run this before training to optimize memory usage

import torch
import os

def optimize_mps_memory():
    """Optimize memory settings for Apple Silicon (MPS) training"""
    
    if torch.backends.mps.is_available():
        print("🍎 Apple Silicon (MPS) detected - Applying memory optimizations...")
        
        # Set MPS memory fraction to avoid OOM
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'
        print("   ✅ MPS memory watermark set to 70%")
        
        # Enable memory-efficient attention if available
        try:
            torch.backends.mps.empty_cache()
            print("   ✅ MPS cache cleared")
        except:
            pass
        
        # Recommended settings for MPS
        recommended_config = {
            "train_batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "no",
            "resolution": 512  # Keep at 512, don't reduce further
        }
        
        print("   📋 Recommended MPS settings:")
        for key, value in recommended_config.items():
            print(f"      {key}: {value}")
        
        return recommended_config
    
    else:
        print("❌ MPS not available")
        return {}

if __name__ == "__main__":
    optimize_mps_memory()