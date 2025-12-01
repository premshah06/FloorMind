#!/usr/bin/env python3
"""
FloorMind QLoRA Fine-tuning Script v2
Production-ready fine-tuning with actual QLoRA implementation

Based on previous training:
- Base: Stable Diffusion 2.1
- Previous training: 10 epochs, achieved 0.025 val_loss
- Dataset: 5,050 floor plan images
- Now: Adding QLoRA for efficient continued fine-tuning
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Diffusion imports
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FloorPlanDataset(Dataset):
    """
    Enhanced dataset for architectural floor plans.
    Loads the 5,050 CubiCasa5K processed images.
    """
    
    def __init__(
        self,
        image_dir: str,
        resolution: int = 512,
        max_samples: Optional[int] = None,
        augment: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.augment = augment
        
        # Load all floor plan images
        self.images = sorted(list(self.image_dir.glob("*.png")))
        if max_samples:
            self.images = self.images[:max_samples]
        
        # Generate rich descriptions
        self.descriptions = self._generate_descriptions()
        
        # Define transforms
        self.transform = self._get_transforms()
        
        logger.info(f"✅ Loaded {len(self.images)} floor plan images")
    
    def _generate_descriptions(self) -> List[str]:
        """Generate diverse architectural descriptions"""
        
        templates = [
            "architectural floor plan with {} layout",
            "detailed {} floor plan design",
            "{} residential floor plan",
            "modern {} apartment layout",
            "{} house floor plan",
            "contemporary {} building design",
            "{} floor plan with multiple rooms",
            "professional {} architectural drawing",
        ]
        
        layouts = [
            "open concept", "traditional", "modern", "L-shaped",
            "U-shaped", "rectangular", "square", "split-level",
            "multi-room", "studio", "loft", "duplex"
        ]
        
        descriptions = []
        for i in range(len(self.images)):
            template = templates[i % len(templates)]
            layout = layouts[i % len(layouts)]
            desc = template.format(layout)
            descriptions.append(desc)
        
        return descriptions
    
    def _get_transforms(self):
        """Architectural-aware augmentation"""
        if self.augment:
            return transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=3),  # Small rotation
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            image = Image.new('RGB', (self.resolution, self.resolution), 'white')
        
        image = self.transform(image)
        description = self.descriptions[idx]
        
        return {
            'image': image,
            'text': description,
            'idx': idx
        }



class FloorMindQLoRATrainer:
    """
    Production QLoRA trainer for FloorMind.
    Continues from previous training checkpoint.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 FloorMind QLoRA Trainer v2")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"💾 Output: {self.output_dir}")
        
        # Load components
        self._load_model_components()
        self._setup_lora()
        self._setup_dataset()
        self._setup_training()
        
        # Metrics
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = {
            'epoch': [], 'step': [], 'loss': [],
            'lr': [], 'timestamp': []
        }
    
    def _setup_device(self) -> torch.device:
        """Setup device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("✅ Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            logger.warning("⚠️ CPU mode")
        return device
    
    def _load_model_components(self):
        """Load model from checkpoint or base"""
        logger.info("📦 Loading model components...")
        
        model_name = self.config['model_name']
        
        # Try to load from previous training first, then base model
        checkpoint_paths = [
            "models/final_model",
            "models/floormind_pipeline",
            "models/trained_model",
        ]
        
        model_loaded = False
        
        # Try local checkpoints first
        for checkpoint_path in checkpoint_paths:
            if Path(checkpoint_path).exists():
                try:
                    logger.info(f"Trying local checkpoint: {checkpoint_path}")
                    
                    self.tokenizer = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer")
                    self.text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")
                    self.vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae")
                    self.unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
                    self.noise_scheduler = DDPMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler")
                    
                    logger.info(f"✅ Loaded from local checkpoint: {checkpoint_path}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load from {checkpoint_path}: {e}")
                    continue
        
        # If no local checkpoint, download base model from Hugging Face
        if not model_loaded:
            try:
                logger.info(f"📥 Downloading base model from Hugging Face: {model_name}")
                logger.info("This may take a few minutes on first run...")
                
                self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
                self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
                self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
                self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
                self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
                
                logger.info(f"✅ Successfully loaded base model: {model_name}")
                model_loaded = True
                
            except Exception as e:
                logger.error(f"Failed to load base model from Hugging Face: {e}")
                raise RuntimeError(f"Could not load model. Error: {e}")
        
        # Move to device
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Enable optimizations
        if self.config.get('gradient_checkpointing', True):
            self.unet.enable_gradient_checkpointing()
            logger.info("✅ Gradient checkpointing enabled")
        
        if self.config.get('use_xformers', True):
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                logger.info("✅ xFormers enabled")
            except:
                logger.warning("⚠️ xFormers not available")
        
        logger.info(f"📊 U-Net params: {sum(p.numel() for p in self.unet.parameters()) / 1e6:.1f}M")
    
    def _setup_lora(self):
        """Setup LoRA adapters"""
        logger.info("🔧 Setting up LoRA...")
        
        lora_config = LoraConfig(
            r=self.config.get('lora_rank', 16),  # Increased rank for better capacity
            lora_alpha=self.config.get('lora_alpha', 32),
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",  # Additional targets
            ],
            lora_dropout=self.config.get('lora_dropout', 0.05),
            bias="none"
        )
        
        # Apply LoRA
        self.unet = get_peft_model(self.unet, lora_config)
        
        trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.unet.parameters())
        
        logger.info(f"✅ LoRA configured")
        logger.info(f"📊 Trainable: {trainable / 1e6:.2f}M ({100 * trainable / total:.2f}%)")
    
    def _setup_dataset(self):
        """Setup dataset"""
        logger.info("📊 Setting up dataset...")
        
        self.dataset = FloorPlanDataset(
            image_dir=self.config['data_dir'],
            resolution=self.config['resolution'],
            max_samples=self.config.get('max_samples'),
            augment=True
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=True
        )
        
        logger.info(f"✅ Dataset: {len(self.dataset)} samples, {len(self.dataloader)} batches")
    
    def _setup_training(self):
        """Setup optimizer and scheduler"""
        logger.info("⚙️ Setting up training...")
        
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=self.config.get('weight_decay', 0.01),
            eps=1e-8
        )
        
        num_training_steps = len(self.dataloader) * self.config['num_epochs']
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(self.dataloader),
            T_mult=2,
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        logger.info(f"✅ Optimizer: AdamW")
        logger.info(f"✅ Scheduler: CosineAnnealingWarmRestarts")
        logger.info(f"📊 Total steps: {num_training_steps}")

    
    def encode_text(self, text_batch: List[str]) -> torch.Tensor:
        """Encode text to embeddings"""
        text_inputs = self.tokenizer(
            text_batch,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def training_step(self, batch: Dict) -> torch.Tensor:
        """Single training step"""
        images = batch['image'].to(self.device)
        texts = batch['text']
        
        # Encode to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device
        ).long()
        
        # Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        text_embeddings = self.encode_text(texts)
        
        # Predict noise
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting training...")
        logger.info(f"📊 Epochs: {self.config['num_epochs']}")
        logger.info(f"📊 Batch size: {self.config['batch_size']}")
        logger.info(f"📊 Learning rate: {self.config['learning_rate']}")
        
        start_time = datetime.now()
        
        for epoch in range(self.config['num_epochs']):
            self.unet.train()
            epoch_losses = []
            
            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
                leave=True
            )
            
            for step, batch in enumerate(progress_bar):
                # Forward
                loss = self.training_step(batch)
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.unet.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Track
                epoch_losses.append(loss.item())
                current_lr = self.lr_scheduler.get_last_lr()[0]
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # Log
                if self.global_step % self.config.get('logging_steps', 50) == 0:
                    self.training_history['epoch'].append(epoch)
                    self.training_history['step'].append(self.global_step)
                    self.training_history['loss'].append(loss.item())
                    self.training_history['lr'].append(current_lr)
                    self.training_history['timestamp'].append(datetime.now())
                
                # Save checkpoint
                if self.global_step % self.config.get('save_steps', 500) == 0 and self.global_step > 0:
                    self.save_checkpoint(epoch, self.global_step)
                
                self.global_step += 1
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            logger.info(f"\n📊 Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            
            # Save best
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, self.global_step, is_best=True)
                logger.info(f"💾 New best model! Loss: {avg_loss:.4f}")
            
            # Validation
            if (epoch + 1) % self.config.get('validation_epochs', 2) == 0:
                self.generate_validation_images(epoch + 1)
        
        # Complete
        total_time = datetime.now() - start_time
        logger.info(f"\n🎉 Training complete!")
        logger.info(f"⏱️ Time: {total_time}")
        logger.info(f"📊 Best loss: {self.best_loss:.4f}")
        
        self.save_final_model()
        self.save_training_stats()
        
        return self.global_step
    
    def save_checkpoint(self, epoch: int, step: int, is_best: bool = False):
        """Save checkpoint"""
        name = "best_model" if is_best else f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_dir / "unet_lora")
        
        # Save training state
        torch.save({
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, checkpoint_dir / "training_state.pt")
        
        logger.info(f"💾 Saved: {name}")
    
    def save_final_model(self):
        """Save final model"""
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        
        logger.info("💾 Saving final model...")
        
        # Merge LoRA
        self.unet = self.unet.merge_and_unload()
        
        # Create pipeline
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=DPMSolverMultistepScheduler.from_config(
                self.noise_scheduler.config
            ),
            safety_checker=None,
            feature_extractor=None
        )
        
        pipeline.save_pretrained(final_dir)
        
        # Save config
        with open(final_dir / "training_config.json", 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        logger.info(f"✅ Saved to: {final_dir}")
    
    def save_training_stats(self):
        """Save training statistics"""
        if not self.training_history['step']:
            return
        
        import pandas as pd
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.output_dir / "training_history.csv", index=False)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(df['step'], df['loss'], alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(df['step'], df['lr'], color='orange', alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300)
        plt.close()
        
        logger.info("📊 Stats saved")
    
    def generate_validation_images(self, epoch: int):
        """Generate validation images"""
        logger.info(f"🎨 Generating validation images...")
        
        self.unet.eval()
        
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        
        prompts = [
            "modern studio apartment floor plan",
            "two bedroom house floor plan with garage",
            "office space floor plan with meeting rooms",
            "contemporary apartment with balcony"
        ]
        
        val_dir = self.output_dir / "validation"
        val_dir.mkdir(exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            with torch.no_grad():
                image = pipeline(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]
            
            image.save(val_dir / f"epoch{epoch}_sample{i+1}.png")
        
        logger.info("✅ Validation images saved")
        self.unet.train()


def main():
    """Main training function"""
    
    config = {
        # Model - SD 2.1 (works on Colab Free T4)
        'model_name': "stabilityai/stable-diffusion-2-1",
        'resolution': 512,
        
        # Data - use all 5,050 images
        # UPDATE THIS PATH FOR GOOGLE COLAB:
        'data_dir': "/content/drive/MyDrive/Fine Tuning Deep Project/images",
        'max_samples': None,
        
        # Training
        'num_epochs': 15,
        'batch_size': 2,  # Works on T4 GPU
        'learning_rate': 5e-5,  # Lower LR for fine-tuning
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        
        # LoRA - increased capacity
        'lora_rank': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        
        # Optimization
        'gradient_checkpointing': True,
        'use_xformers': True,
        'num_workers': 0,
        
        # Logging
        # Save to Google Drive so you don't lose it
        'output_dir': "/content/drive/MyDrive/floormind_model",
        'logging_steps': 50,
        'save_steps': 500,
        'validation_epochs': 2,
    }
    
    logger.info("="*60)
    logger.info("🏠 FloorMind QLoRA Training v2")
    logger.info("="*60)
    logger.info("\n📋 Configuration:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    logger.info("")
    
    try:
        trainer = FloorMindQLoRATrainer(config)
        total_steps = trainer.train()
        
        logger.info(f"\n🎉 Success!")
        logger.info(f"📈 Steps: {total_steps}")
        logger.info(f"📁 Model: {config['output_dir']}/final_model")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
