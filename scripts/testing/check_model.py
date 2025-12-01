#!/usr/bin/env python3
"""
Check FloorMind Model Quality
Analyzes the fine-tuned model to verify training quality
"""

import os
import json
import torch
from safetensors.torch import load_file

def check_model_structure(model_path):
    """Check if model has all required components"""
    print(f"\n📁 Checking model structure: {model_path}")
    print("-" * 60)
    
    required_components = {
        "scheduler": ["scheduler_config.json"],
        "text_encoder": ["config.json", "model.safetensors"],
        "tokenizer": ["tokenizer_config.json", "vocab.json"],
        "unet": ["config.json", "diffusion_pytorch_model.safetensors"],
        "vae": ["config.json", "diffusion_pytorch_model.safetensors"]
    }
    
    all_good = True
    
    for component, files in required_components.items():
        component_path = os.path.join(model_path, component)
        
        if not os.path.exists(component_path):
            print(f"❌ Missing: {component}/")
            all_good = False
            continue
        
        print(f"✅ Found: {component}/")
        
        for file in files:
            file_path = os.path.join(component_path, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                size_mb = size / (1024 * 1024)
                print(f"   ✓ {file} ({size_mb:.1f} MB)")
            else:
                print(f"   ✗ Missing: {file}")
                all_good = False
    
    return all_good

def check_unet_weights(model_path):
    """Check UNet weights to verify fine-tuning"""
    print(f"\n🔍 Analyzing UNet weights...")
    print("-" * 60)
    
    unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
    
    if not os.path.exists(unet_path):
        print("❌ UNet weights not found!")
        return False
    
    try:
        # Load weights
        weights = load_file(unet_path)
        
        print(f"✅ UNet weights loaded successfully")
        print(f"\nWeight Statistics:")
        print(f"  Total parameters: {len(weights)}")
        
        # Analyze a few key layers
        sample_layers = [
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
            "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight",
            "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q.weight"
        ]
        
        print(f"\nSample Layer Analysis:")
        for layer_name in sample_layers:
            if layer_name in weights:
                tensor = weights[layer_name]
                print(f"\n  {layer_name}:")
                print(f"    Shape: {tensor.shape}")
                print(f"    Mean: {tensor.mean().item():.6f}")
                print(f"    Std: {tensor.std().item():.6f}")
                print(f"    Min: {tensor.min().item():.6f}")
                print(f"    Max: {tensor.max().item():.6f}")
        
        # Check if weights look reasonable (not all zeros, not NaN)
        first_weight = next(iter(weights.values()))
        if torch.isnan(first_weight).any():
            print("\n❌ WARNING: Found NaN values in weights!")
            return False
        
        if torch.all(first_weight == 0):
            print("\n❌ WARNING: Weights are all zeros!")
            return False
        
        print("\n✅ Weights look healthy!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False

def check_training_config(model_path):
    """Check training configuration"""
    print(f"\n⚙️  Training Configuration:")
    print("-" * 60)
    
    config_path = os.path.join(model_path, "training_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(json.dumps(config, indent=2))
        
        # Analyze config
        print("\n📊 Training Analysis:")
        
        epochs = config.get('num_epochs', config.get('epochs', 'Unknown'))
        print(f"  Epochs: {epochs}")
        
        lr = config.get('learning_rate', 'Unknown')
        print(f"  Learning Rate: {lr}")
        
        batch_size = config.get('batch_size', config.get('train_batch_size', 'Unknown'))
        print(f"  Batch Size: {batch_size}")
        
        # Check if LoRA was used
        if 'lora_rank' in config:
            print(f"  LoRA Rank: {config['lora_rank']}")
            print(f"  LoRA Alpha: {config.get('lora_alpha', 'N/A')}")
            print("  ✅ Using LoRA fine-tuning")
        else:
            print("  ⚠️  Full fine-tuning (not LoRA)")
        
        return True
    else:
        print("⚠️  No training config found")
        return False

def check_model_info(model_path):
    """Check model info file"""
    print(f"\n📄 Model Info:")
    print("-" * 60)
    
    info_path = os.path.join(model_path, "model_info.json")
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(json.dumps(info, indent=2))
        return True
    else:
        print("⚠️  No model info found")
        return False

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🔬 FloorMind Model Quality Check".center(60))
    print("="*60)
    
    # Check all available models
    model_paths = [
        ("Final Model (4 epochs)", "models/final_model"),
        ("Production Model (20 epochs)", "models/final_model_production"),
        ("FloorMind Pipeline", "models/floormind_pipeline")
    ]
    
    results = {}
    
    for name, path in model_paths:
        print(f"\n\n{'='*60}")
        print(f"Checking: {name}")
        print(f"Path: {path}")
        print('='*60)
        
        if not os.path.exists(path):
            print(f"❌ Model not found at: {path}")
            results[name] = False
            continue
        
        # Run checks
        structure_ok = check_model_structure(path)
        weights_ok = check_unet_weights(path) if structure_ok else False
        config_ok = check_training_config(path)
        info_ok = check_model_info(path)
        
        results[name] = structure_ok and weights_ok
    
    # Summary
    print("\n\n" + "="*60)
    print("📊 Summary".center(60))
    print("="*60)
    
    for name, status in results.items():
        status_str = "✅ COMPLETE" if status else "❌ INCOMPLETE"
        print(f"{name.ljust(40)}: {status_str}")
    
    # Recommendations
    print("\n" + "="*60)
    print("💡 Recommendations".center(60))
    print("="*60)
    
    complete_models = [name for name, status in results.items() if status]
    
    if complete_models:
        print(f"\n✅ You have {len(complete_models)} complete model(s):")
        for model in complete_models:
            print(f"   - {model}")
        
        print("\n📝 Next Steps:")
        print("   1. Start the backend: ./start_backend.sh")
        print("   2. Test generation: python test_backend.py")
        print("   3. Evaluate quality of generated floor plans")
        print("   4. If quality is poor, consider retraining with:")
        print("      - More epochs (10-20)")
        print("      - More training data")
        print("      - Higher LoRA rank (32-64)")
    else:
        print("\n❌ No complete models found!")
        print("\n📝 You need to:")
        print("   1. Check your training script")
        print("   2. Ensure model saves properly")
        print("   3. Retrain if necessary")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Check interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
