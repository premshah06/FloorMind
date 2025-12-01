#!/usr/bin/env python3
"""
FloorMind Trained Model Usage Script
Load and use your trained FloorMind model for floor plan generation
"""

import pickle
import torch
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_trained_model(model_path="google/floormind_model.pkl"):
    """Load the trained FloorMind model"""
    
    print("🔄 Loading trained FloorMind model...")
    
    try:
        # First try to load the pickle file
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        config = model_data['config']
        training_stats = model_data['training_stats']
        best_val_loss = model_data['best_val_loss']
        
        # Check if pipeline is directly in the data or we need to load from directory
        if 'pipeline' in model_data:
            pipeline = model_data['pipeline']
            print("✅ Pipeline loaded from pickle file")
        elif 'pipeline_dir' in model_data:
            # Load pipeline from directory
            from diffusers import StableDiffusionPipeline
            pipeline_dir = model_data['pipeline_dir']
            
            # Try to find the pipeline directory in the google folder
            local_pipeline_dirs = [
                "google",  # Check if components are in google folder
                "google/floormind_pipeline",  # Standard pipeline directory
                pipeline_dir  # Original path (might not exist locally)
            ]
            
            pipeline = None
            for pipeline_path in local_pipeline_dirs:
                try:
                    if os.path.exists(pipeline_path):
                        # Check if this directory has model components
                        required_files = ['model_index.json', 'tokenizer_config.json']
                        if all(os.path.exists(os.path.join(pipeline_path, f)) for f in required_files):
                            print(f"🔄 Loading pipeline from: {pipeline_path}")
                            pipeline = StableDiffusionPipeline.from_pretrained(
                                pipeline_path,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                safety_checker=None
                            )
                            break
                        else:
                            # Try to construct pipeline from individual components
                            print(f"🔄 Constructing pipeline from components in: {pipeline_path}")
                            pipeline = construct_pipeline_from_components(pipeline_path)
                            if pipeline:
                                break
                except Exception as e:
                    print(f"⚠️ Failed to load from {pipeline_path}: {e}")
                    continue
            
            if pipeline is None:
                raise ValueError("Could not load pipeline from any available path")
                
            print("✅ Pipeline loaded from components")
        else:
            raise ValueError("No pipeline or pipeline_dir found in model data")
        
        print("✅ Model loaded successfully!")
        print(f"📊 Training info:")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Training epochs: {config.get('num_epochs', 'Unknown')}")
        print(f"   Resolution: {config.get('resolution', 'Unknown')}x{config.get('resolution', 'Unknown')}")
        
        return pipeline, config, training_stats
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def construct_pipeline_from_components(components_dir):
    """Construct pipeline from individual component files"""
    
    try:
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        
        print("🔧 Constructing pipeline from individual components...")
        
        # Check what files we have
        available_files = os.listdir(components_dir)
        print(f"📁 Available files: {available_files}")
        
        # Load components if they exist as separate files
        tokenizer = None
        text_encoder = None
        vae = None
        unet = None
        scheduler = None
        
        # Try to load from standard diffusers format first
        try:
            return StableDiffusionPipeline.from_pretrained(
                components_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None
            )
        except:
            pass
        
        # If that fails, try to construct from individual files
        base_model = "runwayml/stable-diffusion-v1-5"
        
        # Load tokenizer
        if 'tokenizer_config.json' in available_files:
            tokenizer = CLIPTokenizer.from_pretrained(components_dir)
        else:
            tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        
        # Load text encoder
        if 'config.json' in available_files:
            try:
                text_encoder = CLIPTextModel.from_pretrained(components_dir)
            except:
                text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
        else:
            text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        
        # Load UNet (this should be the fine-tuned one)
        if 'model.safetensors' in available_files:
            # This is likely our fine-tuned UNet
            unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
            # Load the fine-tuned weights
            import safetensors.torch
            state_dict = safetensors.torch.load_file(os.path.join(components_dir, 'model.safetensors'))
            unet.load_state_dict(state_dict)
            print("✅ Loaded fine-tuned UNet weights")
        else:
            unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
        
        # Load scheduler
        if 'scheduler_config.json' in available_files:
            scheduler = DDPMScheduler.from_pretrained(components_dir)
        else:
            scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
        
        # Construct pipeline
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        
        print("✅ Pipeline constructed successfully from components")
        return pipeline
        
    except Exception as e:
        print(f"❌ Error constructing pipeline: {e}")
        return None

def generate_floor_plan(pipeline, prompt, output_path=None, seed=42):
    """Generate a floor plan from text prompt"""
    
    print(f"🎨 Generating floor plan: '{prompt}'")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Generate image
    generator = torch.Generator(device=device).manual_seed(seed)
    
    with torch.no_grad():
        image = pipeline(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=generator
        ).images[0]
    
    # Save image if path provided
    if output_path:
        image.save(output_path)
        print(f"💾 Image saved to: {output_path}")
    
    return image

def test_model_generation():
    """Test the trained model with various prompts"""
    
    # Load model
    pipeline, config, training_stats = load_trained_model()
    
    if pipeline is None:
        print("❌ Failed to load model")
        return
    
    # Test prompts
    test_prompts = [
        "Modern 3-bedroom apartment floor plan with open kitchen",
        "Traditional house floor plan with separate dining room",
        "Studio apartment floor plan with efficient layout",
        "Two-story house floor plan with garage",
        "Luxury penthouse floor plan with balcony"
    ]
    
    print(f"\n🖼️ Generating {len(test_prompts)} test floor plans...")
    
    # Create output directory
    output_dir = "generated_floor_plans"
    os.makedirs(output_dir, exist_ok=True)
    
    generated_images = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n[{i+1}/{len(test_prompts)}] {prompt}")
        
        # Generate image
        output_path = f"{output_dir}/floor_plan_{i+1:02d}.png"
        image = generate_floor_plan(pipeline, prompt, output_path, seed=42+i)
        generated_images.append((image, prompt))
    
    # Create comparison grid
    print(f"\n📊 Creating comparison grid...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (image, prompt) in enumerate(generated_images):
        if i < len(axes):
            axes[i].imshow(image)
            axes[i].set_title(f"{prompt[:40]}...", fontsize=10, pad=10)
            axes[i].axis('off')
    
    # Hide unused subplot
    if len(generated_images) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    comparison_path = f"{output_dir}/floor_plans_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Generation complete!")
    print(f"📁 Individual images: {output_dir}/floor_plan_*.png")
    print(f"📊 Comparison grid: {comparison_path}")

def analyze_training_results():
    """Analyze the training results"""
    
    print("📊 Analyzing training results...")
    
    try:
        # Load training stats
        import pandas as pd
        stats_df = pd.read_csv("google/training_stats.csv")
        
        print(f"📈 Training Statistics:")
        print(f"   Total training steps: {len(stats_df)}")
        print(f"   Final training loss: {stats_df['train_loss'].iloc[-1]:.4f}")
        print(f"   Average training loss: {stats_df['train_loss'].mean():.4f}")
        print(f"   Best training loss: {stats_df['train_loss'].min():.4f}")
        
        # Plot training curve
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(stats_df['step'], stats_df['train_loss'], alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(stats_df['step'], stats_df['lr'], color='orange', alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig("training_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Training analysis saved to: training_analysis.png")
        
    except Exception as e:
        print(f"⚠️ Could not analyze training results: {e}")

def main():
    """Main function"""
    
    print("🏠 FloorMind Trained Model Usage")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists("google/floormind_model.pkl"):
        print("❌ Trained model not found at google/floormind_model.pkl")
        print("Please make sure you have the trained model files in the google/ directory")
        return
    
    print("Choose an option:")
    print("1. Test model generation (recommended)")
    print("2. Generate single floor plan")
    print("3. Analyze training results")
    print("4. All of the above")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            test_model_generation()
        elif choice == "2":
            prompt = input("Enter floor plan description: ").strip()
            if prompt:
                pipeline, _, _ = load_trained_model()
                if pipeline:
                    image = generate_floor_plan(pipeline, prompt, "custom_floor_plan.png")
                    print("✅ Custom floor plan generated: custom_floor_plan.png")
        elif choice == "3":
            analyze_training_results()
        elif choice == "4":
            test_model_generation()
            analyze_training_results()
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()