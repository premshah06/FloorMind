# MEMORY EFFICIENT TRAINING MODIFICATIONS
# Add this to optimize the training loop for minimal memory usage

import torch
import gc

def memory_efficient_training_step(batch, unet, vae, text_encoder, tokenizer, noise_scheduler, device):
    """Memory-optimized training step with aggressive cleanup"""
    
    try:
        # Move batch to device with explicit dtype
        images = batch['image'].to(device, dtype=torch.float32, non_blocking=False)
        texts = batch['text']
        
        # Encode images with memory cleanup
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # Clear image memory immediately
            del images
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents, device=device, dtype=latents.dtype)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device, dtype=torch.long
        )
        
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Clear original latents
        del latents
        
        # Encode text with memory management
        with torch.no_grad():
            text_inputs = tokenizer(
                texts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
            # Clear text inputs
            del text_inputs
        
        # Forward pass through UNet
        noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), noise.float(), reduction="mean"
        )
        
        # Cleanup
        del noise_pred, noise, noisy_latents, text_embeddings, timesteps
        
        # Aggressive memory cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        return loss
        
    except Exception as e:
        print(f"Error in training step: {e}")
        # Emergency cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        return torch.tensor(0.0, device=device, requires_grad=True)

def setup_memory_efficient_models(unet, vae, text_encoder):
    """Setup models for memory efficiency"""
    
    # Enable gradient checkpointing if available
    if hasattr(unet, 'enable_gradient_checkpointing'):
        unet.enable_gradient_checkpointing()
        print("✅ UNet gradient checkpointing enabled")
    
    # Set models to appropriate precision
    unet = unet.to(dtype=torch.float32)
    vae = vae.to(dtype=torch.float32) 
    text_encoder = text_encoder.to(dtype=torch.float32)
    
    # Ensure models are in correct mode
    unet.train()
    vae.eval()
    text_encoder.eval()
    
    print("✅ Models configured for memory efficiency")
    return unet, vae, text_encoder

# Memory monitoring function
def check_memory_usage():
    """Check current memory usage"""
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        print(f"📊 MPS Memory: {allocated:.2f} GB allocated")
        return allocated
    return 0

print("🔧 Memory efficient training functions loaded!")
print("💡 Use memory_efficient_training_step() instead of regular training_step()")
print("💡 Call setup_memory_efficient_models() before training")
print("💡 Use check_memory_usage() to monitor memory")