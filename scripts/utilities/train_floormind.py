#!/usr/bin/env python3
"""
FloorMind Training Script
Complete training pipeline for FloorMind floor plan generation model
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

class FloorPlanDataset(Dataset):
    """Optimized dataset class for floor plan images"""
    
    def __init__(self, metadata_file: str, image_dir: str, split: str = 'train', transform=None, max_samples=None):
        """Initialize dataset"""
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.split = split
        
        # Load metadata
        if Path(metadata_file).exists():
            df = pd.read_csv(metadata_file)
            
            # Filter by split if available
            if 'split' in df.columns:
                self.metadata = df[df['split'] == split].copy().reset_index(drop=True)
            else:
                # Simple split
                if split == 'train':
                    self.metadata = df[:int(len(df) * 0.6)].copy()
                else:
                    self.metadata = df[int(len(df) * 0.6):].copy()
            
            if max_samples:
                self.metadata = self.metadata.head(max_samples)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        print(f"📊 {split.title()} dataset loaded: {len(self.metadata)} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Get image path
        if 'image_path' in row:
            image_path = Path(row['image_path'])
            if not image_path.is_absolute():
                image_path = self.image_dir.parent / image_path
        else:
            # Fallback
            image_files = list(self.image_dir.glob("*.png"))
            image_path = image_files[idx % len(image_files)] if image_files else None
        
        # Load image
        try:
            if image_path and image_path.exists():
                image = Image.open(image_path).convert('RGB')
            else:
                # Create white background image
                image = Image.new('RGB', (512, 512), color='white')
        except Exception:
            image = Image.new('RGB', (512, 512), color='white')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get description
        description = row.get('description', 'A floor plan layout')
        
        return {
            'image': image,
            'text': description,
            'idx': idx
        }

class FloorMindTrainer:
    """FloorMind training class with simplified approach"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"🚀 FloorMind Trainer Initialized")
        print(f"📱 Device: {self.device}")
        
        # Create directories
        os.makedirs(config["output_dir"], exist_ok=True)
        
        # Setup dataset
        self._setup_dataset()
        
        # Initialize training stats
        self.training_stats = {
            'epoch': [],
            'step': [],
            'loss': [],
            'lr': [],
            'timestamp': []
        }
    
    def _setup_dataset(self):
        """Setup dataset and dataloader"""
        print("📊 Setting up dataset...")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.config["resolution"], self.config["resolution"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Create datasets
        self.train_dataset = FloorPlanDataset(
            metadata_file=self.config["metadata_file"],
            image_dir=self.config["images_dir"],
            split='train',
            transform=self.transform,
            max_samples=self.config.get("max_samples")
        )
        
        self.test_dataset = FloorPlanDataset(
            metadata_file=self.config["metadata_file"],
            image_dir=self.config["images_dir"],
            split='test',
            transform=self.transform,
            max_samples=self.config.get("max_samples") // 2 if self.config.get("max_samples") else None
        )
        
        # Create dataloaders (no multiprocessing to avoid issues)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config["eval_batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"✅ Datasets ready:")
        print(f"   Train: {len(self.train_dataset)} samples, {len(self.train_dataloader)} batches")
        print(f"   Test: {len(self.test_dataset)} samples, {len(self.test_dataloader)} batches")
    
    def load_model_components(self):
        """Load Stable Diffusion components with error handling"""
        print("🔄 Loading model components...")
        
        try:
            from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer
            
            model_name = self.config["model_name"]
            
            # Load components one by one with error handling
            print("   Loading tokenizer...")
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
            
            print("   Loading text encoder...")
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
            
            print("   Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
            
            print("   Loading UNet...")
            self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
            
            print("   Loading scheduler...")
            self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
            
            # Move to device
            self.text_encoder = self.text_encoder.to(self.device)
            self.vae = self.vae.to(self.device)
            self.unet = self.unet.to(self.device)
            
            # Freeze components
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.unet.train()
            
            print("✅ All model components loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            return False
    
    def setup_training(self):
        """Setup optimizer and scheduler"""
        print("⚙️ Setting up training components...")
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Setup scheduler
        num_training_steps = len(self.train_dataloader) * self.config["num_epochs"]
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_training_steps,
            eta_min=self.config["learning_rate"] * 0.1
        )
        
        print(f"✅ Training setup complete")
        print(f"🔄 Total training steps: {num_training_steps}")
    
    def encode_text(self, text_batch):
        """Encode text prompts to embeddings"""
        text_inputs = self.tokenizer(
            text_batch,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        return text_embeddings
    
    def training_step(self, batch):
        """Single training step"""
        images = batch['image'].to(self.device)
        texts = batch['text']
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (latents.shape[0],), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text
        text_embeddings = self.encode_text(texts)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting training...")
        
        global_step = 0
        start_time = datetime.now()
        
        for epoch in range(self.config["num_epochs"]):
            epoch_losses = []
            
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
                leave=True
            )
            
            for step, batch in enumerate(progress_bar):
                try:
                    # Forward pass
                    loss = self.training_step(batch)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.config["max_grad_norm"])
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Track metrics
                    current_loss = loss.detach().item()
                    epoch_losses.append(current_loss)
                    
                    # Log progress
                    if global_step % self.config["logging_steps"] == 0:
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        
                        self.training_stats['epoch'].append(epoch)
                        self.training_stats['step'].append(global_step)
                        self.training_stats['loss'].append(current_loss)
                        self.training_stats['lr'].append(current_lr)
                        self.training_stats['timestamp'].append(datetime.now())
                        
                        progress_bar.set_postfix({
                            'loss': f'{current_loss:.4f}',
                            'lr': f'{current_lr:.2e}'
                        })
                    
                    global_step += 1
                    
                    # Save checkpoint
                    if global_step % self.config["save_steps"] == 0:
                        self.save_checkpoint(global_step)
                        
                except Exception as e:
                    print(f"⚠️ Error in training step: {e}")
                    continue
            
            # Epoch summary
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"\n📊 Epoch {epoch+1} Summary:")
                print(f"   Average Loss: {avg_loss:.4f}")
                print(f"   Steps: {len(epoch_losses)}")
        
        total_time = datetime.now() - start_time
        print(f"\n🎉 Training completed!")
        print(f"⏱️ Total time: {total_time}")
        print(f"📈 Total steps: {global_step}")
        
        # Save final model
        self.save_final_model()
        self.save_training_stats()
        
        return global_step
    
    def save_checkpoint(self, step):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config["output_dir"]) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model states
        torch.save(self.unet.state_dict(), checkpoint_dir / "unet.pth")
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pth")
        torch.save(self.lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pth")
        
        print(f"💾 Checkpoint saved at step {step}")
    
    def save_final_model(self):
        """Save final trained model"""
        final_model_dir = Path(self.config["output_dir"]) / "final_model"
        final_model_dir.mkdir(exist_ok=True)
        
        try:
            # Save the fine-tuned UNet
            self.unet.save_pretrained(final_model_dir / "unet")
            
            # Save other components
            self.tokenizer.save_pretrained(final_model_dir / "tokenizer")
            self.text_encoder.save_pretrained(final_model_dir / "text_encoder")
            self.vae.save_pretrained(final_model_dir / "vae")
            self.noise_scheduler.save_pretrained(final_model_dir / "scheduler")
            
            # Save config
            with open(final_model_dir / "training_config.json", 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            
            print(f"💾 Final model saved to: {final_model_dir}")
            
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
            # Fallback: save as PyTorch state dict
            torch.save(self.unet.state_dict(), final_model_dir / "unet_state_dict.pth")
            print(f"💾 Model state dict saved as fallback")
    
    def save_training_stats(self):
        """Save training statistics"""
        if not self.training_stats['step']:
            print("⚠️ No training statistics to save")
            return
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(self.training_stats)
        
        # Save CSV
        output_dir = Path(self.config["output_dir"])
        stats_df.to_csv(output_dir / "training_stats.csv", index=False)
        
        # Create visualization
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss curve
            ax1.plot(stats_df['step'], stats_df['loss'], alpha=0.7)
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss Over Time')
            ax1.grid(True, alpha=0.3)
            
            # Learning rate curve
            ax2.plot(stats_df['step'], stats_df['lr'], color='orange', alpha=0.7)
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            plt.tight_layout()
            plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Training curves saved")
            
        except Exception as e:
            print(f"⚠️ Error creating training curves: {e}")
        
        # Print summary
        print(f"📊 Training Summary:")
        print(f"   Final Loss: {stats_df['loss'].iloc[-1]:.4f}")
        print(f"   Average Loss: {stats_df['loss'].mean():.4f}")
        print(f"   Min Loss: {stats_df['loss'].min():.4f}")

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="FloorMind Training")
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples for quick testing')
    parser.add_argument('--full-dataset', action='store_true', help='Use full dataset')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        # Data paths
        "metadata_file": "data/metadata.csv",
        "images_dir": "data/processed/images",
        
        # Model configuration
        "model_name": "runwayml/stable-diffusion-v1-5",
        "resolution": 512,
        "train_batch_size": args.batch_size,
        "eval_batch_size": 1,
        
        # Training parameters
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "max_grad_norm": 1.0,
        
        # Output configuration
        "output_dir": "outputs/models/floormind_base",
        "save_steps": 50,
        "logging_steps": 10,
        
        # Sample limitation
        "max_samples": None if args.full_dataset else args.max_samples,
    }
    
    print("🚀 FloorMind Training Pipeline")
    print("=" * 50)
    print(f"📊 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # Initialize trainer
        trainer = FloorMindTrainer(config)
        
        # Load model components
        if not trainer.load_model_components():
            print("❌ Failed to load model components")
            print("💡 This might be due to memory constraints or missing dependencies")
            print("💡 Try running with smaller batch size or on a machine with more RAM")
            return False
        
        # Setup training
        trainer.setup_training()
        
        # Start training
        total_steps = trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📈 Total steps: {total_steps}")
        print(f"📁 Model saved to: {config['output_dir']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure you have enough RAM (8GB+ recommended)")
        print("2. Try reducing batch size: --batch-size 1")
        print("3. Try fewer samples: --max-samples 50")
        print("4. Check dataset preparation: python process_full_dataset.py")
        
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)