#!/usr/bin/env python3
"""
FloorMind SDXL Fine-tuning Script
Production-grade fine-tuning for Stable Diffusion XL 1.0

SDXL Advantages:
- 2.6B parameters (3x larger than SD 2.1)
- 1024×1024 native resolution
- Better text understanding (dual text encoders)
- Superior image quality
- Better architectural details
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional, Tuple

# SDXL imports
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_sdxl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SDXLFloorPlanDataset(Dataset):
    """Dataset optimized for SDXL training"""
    
    def __init__(self, image_dir: str, resolution: int = 1024, max_samples: Optional[int] = None):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        
        # Load images
        self.images = sorted(list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg")))
        if max_samples:
            self.images = self.images[:max_samples]
        
        # Rich architectural prompts
        self.prompts = self._generate_prompts()
        
        # SDXL-optimized transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        logger.info(f"✅ Dataset: {len(self.images)} images at {resolution}×{resolution}")
    
    def _generate_prompts(self) -> List[str]:
        """Generate high-quality architectural prompts"""
        
        styles = ["modern", "contemporary", "traditional", "minimalist", "industrial", "luxury"]
        types = ["apartment", "house", "condo", "loft", "studio", "penthouse"]
        features = [
            "open kitchen", "master suite", "walk-in closet", "balcony",
            "home office", "laundry room", "dining area", "living room",
            "multiple bedrooms", "ensuite bathroom", "storage space"
        ]
        layouts = ["open concept", "split level", "single story", "L-shaped", "rectangular"]
        
        prompts = []
        for i in range(len(self.images)):
            style = styles[i % len(styles)]
            type_ = types[i % len(types)]
            feature = features[i % len(features)]
            layout = layouts[i % len(layouts)]
            
            prompt = f"architectural floor plan, {style} {type_} with {layout} layout, featuring {feature}, detailed blueprint"
            prompts.append(prompt)
        
        return prompts
    
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
        prompt = self.prompts[idx]
        
        return {'image': image, 'text': prompt}


class SDXLLoRATrainer:
    """SDXL LoRA Trainer with production optimizations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info("🏠 FloorMind SDXL LoRA Training")
        logger.info("="*60)
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"💾 Output: {self.output_dir}")
        
        self._load_sdxl_components()
        self._setup_lora()
        self._setup_dataset()
        self._setup_training()
        
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = {'epoch': [], 'step': [], 'loss': [], 'lr': []}
    
    def _setup_device(self) -> torch.device:
        """Setup device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("✅ Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            logger.warning("⚠️ CPU mode (not recommended for SDXL)")
        return device
    
    def _load_sdxl_components(self):
        """Load SDXL model components"""
        logger.info("📦 Loading SDXL components...")
        
        model_name = self.config['model_name']
        
        # SDXL has TWO text encoders
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
        
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2")
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        
        # U-Net (2.6B parameters!)
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        
        # Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        # Move to device
        self.text_encoder.to(self.device)
        self.text_encoder_2.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)
        
        # Freeze encoders and VAE
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)
        
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
        
        logger.info(f"✅ SDXL loaded - U-Net: {sum(p.numel() for p in self.unet.parameters()) / 1e9:.2f}B params")
    
    def _setup_lora(self):
        """Setup LoRA for SDXL"""
        logger.info("🔧 Setting up LoRA...")
        
        lora_config = LoraConfig(
            r=self.config.get('lora_rank', 32),  # Higher rank for SDXL
            lora_alpha=self.config.get('lora_alpha', 64),
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2",  # Feed-forward layers
            ],
            lora_dropout=0.05,
            bias="none"
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        
        trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.unet.parameters())
        
        logger.info(f"✅ LoRA configured")
        logger.info(f"📊 Trainable: {trainable / 1e6:.1f}M ({100 * trainable / total:.2f}%)")
        logger.info(f"📊 Total: {total / 1e9:.2f}B")

    
    def _setup_dataset(self):
        """Setup dataset"""
        logger.info("📊 Setting up dataset...")
        
        self.dataset = SDXLFloorPlanDataset(
            image_dir=self.config['data_dir'],
            resolution=self.config['resolution'],
            max_samples=self.config.get('max_samples')
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=True
        )
        
        logger.info(f"✅ Dataset ready: {len(self.dataset)} samples")
    
    def _setup_training(self):
        """Setup optimizer and scheduler"""
        logger.info("⚙️ Setting up training...")
        
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        num_steps = len(self.dataloader) * self.config['num_epochs']
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_steps,
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        logger.info(f"✅ Training setup complete - {num_steps} total steps")
    
    def encode_text_sdxl(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text with SDXL's dual text encoders"""
        
        # Encoder 1 (CLIP ViT-L)
        text_inputs_1 = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encoder 2 (CLIP ViT-G with projection)
        text_inputs_2 = self.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            # Get embeddings from both encoders
            prompt_embeds_1 = self.text_encoder(
                text_inputs_1.input_ids.to(self.device),
                output_hidden_states=True
            )
            prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]  # Penultimate layer
            
            prompt_embeds_2 = self.text_encoder_2(
                text_inputs_2.input_ids.to(self.device),
                output_hidden_states=True
            )
            prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
            
            # Concatenate embeddings
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
            
            # Get pooled embeddings for add_time_ids
            pooled_prompt_embeds = prompt_embeds_2[
                torch.arange(prompt_embeds_2.shape[0]),
                text_inputs_2.input_ids.to(self.device).argmax(dim=-1)
            ]
        
        return prompt_embeds, pooled_prompt_embeds
    
    def get_add_time_ids(self, original_size, crops_coords_top_left, target_size, batch_size):
        """Get time embeddings for SDXL"""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids] * batch_size, dtype=torch.long)
        return add_time_ids.to(self.device)
    
    def training_step(self, batch: Dict) -> torch.Tensor:
        """SDXL training step"""
        images = batch['image'].to(self.device)
        texts = batch['text']
        bsz = images.shape[0]
        
        # Encode to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device
        ).long()
        
        # Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings (dual encoders)
        prompt_embeds, pooled_prompt_embeds = self.encode_text_sdxl(texts)
        
        # Get time embeddings for SDXL
        original_size = (self.config['resolution'], self.config['resolution'])
        crops_coords = (0, 0)
        target_size = (self.config['resolution'], self.config['resolution'])
        add_time_ids = self.get_add_time_ids(original_size, crops_coords, target_size, bsz)
        
        # Prepare added conditioning
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids
        }
        
        # Predict noise with SDXL U-Net
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting SDXL training...")
        logger.info(f"📊 Epochs: {self.config['num_epochs']}")
        logger.info(f"📊 Batch size: {self.config['batch_size']}")
        logger.info(f"📊 Learning rate: {self.config['learning_rate']}")
        logger.info(f"📊 Resolution: {self.config['resolution']}×{self.config['resolution']}")
        
        start_time = datetime.now()
        
        for epoch in range(self.config['num_epochs']):
            self.unet.train()
            epoch_losses = []
            
            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
                leave=True
            )
            
            for batch in progress_bar:
                # Forward
                loss = self.training_step(batch)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                
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
                if self.global_step % 50 == 0:
                    self.history['epoch'].append(epoch)
                    self.history['step'].append(self.global_step)
                    self.history['loss'].append(loss.item())
                    self.history['lr'].append(current_lr)
                
                # Save checkpoint
                if self.global_step % 500 == 0 and self.global_step > 0:
                    self.save_checkpoint(epoch, self.global_step)
                
                self.global_step += 1
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            logger.info(f"\n📊 Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            
            # Save best
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, self.global_step, is_best=True)
                logger.info(f"💾 New best! Loss: {avg_loss:.4f}")
            
            # Validation
            if (epoch + 1) % 3 == 0:
                self.generate_validation_images(epoch + 1)
        
        total_time = datetime.now() - start_time
        logger.info(f"\n🎉 Training complete!")
        logger.info(f"⏱️ Time: {total_time}")
        logger.info(f"📊 Best loss: {self.best_loss:.4f}")
        
        self.save_final_model()
        self.save_stats()
        
        return self.global_step

    def save_checkpoint(self, epoch: int, step: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint_name = "best_model" if is_best else f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save LoRA weights only
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
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_name}")
    
    def save_final_model(self):
        """Save final trained model"""
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        
        logger.info("💾 Saving final SDXL model...")
        
        # Merge LoRA weights
        self.unet = self.unet.merge_and_unload()
        
        # Create SDXL pipeline
        pipeline = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.noise_scheduler
        )
        
        pipeline.save_pretrained(final_dir)
        
        # Save config
        with open(final_dir / "training_config.json", 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        logger.info(f"✅ Final model saved to: {final_dir}")
    
    def save_stats(self):
        """Save training statistics"""
        if not self.history['step']:
            return
        
        import pandas as pd
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(self.history)
        df.to_csv(self.output_dir / "training_history.csv", index=False)
        
        # Create plots
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
        
        logger.info("📊 Training statistics saved")
    
    def generate_validation_images(self, epoch: int):
        """Generate validation images"""
        logger.info(f"🎨 Generating validation images (epoch {epoch})...")
        
        self.unet.eval()
        
        # Create pipeline
        pipeline = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.noise_scheduler
        )
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        
        # Validation prompts
        val_prompts = [
            "modern studio apartment floor plan with open kitchen",
            "two bedroom house floor plan with garage",
            "office space floor plan with meeting rooms",
            "contemporary apartment with balcony and living room"
        ]
        
        # Generate
        val_dir = self.output_dir / "validation"
        val_dir.mkdir(exist_ok=True)
        
        for i, prompt in enumerate(val_prompts):
            with torch.no_grad():
                image = pipeline(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=1024,
                    width=1024
                ).images[0]
            
            image.save(val_dir / f"epoch{epoch}_sample{i+1}.png")
        
        logger.info("✅ Validation images saved")
        self.unet.train()


def main():
    """Main training function"""
    
    config = {
        # Model
        'model_name': "stabilityai/stable-diffusion-xl-base-1.0",
        'resolution': 1024,
        
        # Data - Google Colab paths
        'data_dir': "/content/drive/MyDrive/Fine Tuning Deep Project/images",
        'max_samples': None,
        
        # Training
        'num_epochs': 15,
        'batch_size': 1,  # SDXL requires more memory
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        
        # LoRA
        'lora_rank': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.05,
        
        # Optimization
        'gradient_checkpointing': True,
        'use_xformers': True,
        'num_workers': 0,
        
        # Logging - Save to Google Drive
        'output_dir': "/content/drive/MyDrive/floormind_sdxl_model",
        'logging_steps': 50,
        'save_steps': 500,
        'validation_epochs': 3,
    }
    
    logger.info("="*60)
    logger.info("🏠 FloorMind SDXL Training")
    logger.info("="*60)
    logger.info("\n📋 Configuration:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    logger.info("")
    
    try:
        trainer = SDXLLoRATrainer(config)
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
