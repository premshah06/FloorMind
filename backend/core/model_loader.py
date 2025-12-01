#!/usr/bin/env python3
"""
Unified Model Loader for FloorMind
Handles loading and managing the Stable Diffusion XL pipeline
Supports both full fine-tuned models and LoRA adapters
"""

import os
import logging
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
    logger.info("✅ PEFT available - LoRA support enabled")
except ImportError:
    PEFT_AVAILABLE = False
    logger.info("ℹ️  PEFT not available - LoRA support disabled")

class ModelLoader:
    """Manages the Stable Diffusion XL model pipeline with LoRA support"""
    
    def __init__(self):
        self.pipeline = None
        self.model_info = {}
        self.device = "cpu"  # Default to CPU for stability
        self.is_lora = False  # Track if using LoRA
        
    def find_model_path(self) -> Optional[Dict[str, str]]:
        """
        Find the fine-tuned SDXL model or LoRA adapter in priority order
        Returns dict with 'type' ('full' or 'lora') and 'path'
        """
        
        # Check environment variable first
        env_model_path = os.getenv('FLOORMIND_MODEL_DIR')
        env_lora_path = os.getenv('FLOORMIND_LORA_DIR')
        
        # Priority 1: LoRA adapter (if PEFT available and path specified)
        if PEFT_AVAILABLE and env_lora_path:
            lora_path = os.path.abspath(env_lora_path)
            if os.path.exists(lora_path):
                logger.info(f"✅ Found LoRA adapter at: {lora_path}")
                return {'type': 'lora', 'path': lora_path}
        
        # Priority 2: Full fine-tuned model
        if env_model_path:
            logger.info(f"Using model path from environment: {env_model_path}")
            model_paths = [env_model_path]
        else:
            # Priority order: floormind_sdxl_finetuned > floormind_pipeline > final_model
            model_paths = [
                "models/floormind_sdxl_finetuned",   # Your fine-tuned SDXL model
                "./models/floormind_sdxl_finetuned",
                "models/floormind_sdxl_lora",        # LoRA adapter directory
                "./models/floormind_sdxl_lora",
                "models/active/floormind_pipeline",  # Fallback to SD 1.5
                "models/active/final_model",
                "../models/floormind_sdxl_finetuned",
                "../models/floormind_sdxl_lora",
                "../models/active/floormind_pipeline",
                "../models/active/final_model",
            ]
        
        for path in model_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                # Check if it's a LoRA adapter directory
                if PEFT_AVAILABLE and os.path.exists(os.path.join(abs_path, "adapter_config.json")):
                    logger.info(f"✅ Found LoRA adapter at: {abs_path}")
                    return {'type': 'lora', 'path': abs_path}
                
                # Verify it's a valid Diffusers model
                # SDXL has text_encoder_2 and tokenizer_2
                required_dirs = ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]
                if all(os.path.exists(os.path.join(abs_path, d)) for d in required_dirs):
                    # Check if it's SDXL (has text_encoder_2)
                    is_sdxl = os.path.exists(os.path.join(abs_path, "text_encoder_2"))
                    
                    # CRITICAL: Check if UNet weights actually exist
                    unet_weights_safetensors = os.path.join(abs_path, "unet", "diffusion_pytorch_model.safetensors")
                    unet_weights_bin = os.path.join(abs_path, "unet", "diffusion_pytorch_model.bin")
                    
                    # For SDXL, UNet is often sharded
                    unet_weights_sharded = os.path.join(abs_path, "unet", "diffusion_pytorch_model-00001-of-00002.safetensors")
                    
                    if not (os.path.exists(unet_weights_safetensors) or 
                           os.path.exists(unet_weights_bin) or
                           os.path.exists(unet_weights_sharded)):
                        logger.warning(f"⚠️ Model at {abs_path} missing UNet weights, skipping")
                        continue
                    
                    model_type = "SDXL" if is_sdxl else "SD 1.5"
                    logger.info(f"✅ Found valid {model_type} model at: {abs_path}")
                    return {'type': 'full', 'path': abs_path}
                else:
                    logger.warning(f"⚠️ Path exists but missing components: {abs_path}")
        
        logger.error("❌ No valid fine-tuned model or LoRA adapter found")
        return None
    
    def load(self, force_cpu: bool = True) -> bool:
        """
        Load the Stable Diffusion XL pipeline
        
        Args:
            force_cpu: Force CPU mode for stability (recommended: False for GPU)
            
        Returns:
            bool: True if loaded successfully
        """
        if self.pipeline is not None:
            logger.info("Model already loaded")
            return True
        
        try:
            # Find the model or LoRA adapter
            model_info = self.find_model_path()
            if not model_info:
                self.model_info = {
                    "is_loaded": False,
                    "error": "Fine-tuned model or LoRA adapter not found",
                    "device": "none"
                }
                return False
            
            model_type = model_info['type']
            model_path = model_info['path']
            self.is_lora = (model_type == 'lora')
            
            # Check if it's SDXL (for full models)
            if model_type == 'full':
                is_sdxl = os.path.exists(os.path.join(model_path, "text_encoder_2"))
            else:
                # LoRA adapters are always for SDXL base
                is_sdxl = True
            
            # Determine device (CUDA, MPS, or CPU)
            if force_cpu:
                self.device = "cpu"
                dtype = torch.float32
                logger.info("🖥️ Using CPU mode (stable, slower)")
            else:
                # Check for available accelerators in priority order
                if torch.cuda.is_available():
                    self.device = "cuda"
                    dtype = torch.float16
                    logger.info("🚀 Using CUDA GPU acceleration")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                    dtype = torch.float32  # MPS works better with float32
                    logger.info("🍎 Using MPS (Apple Silicon) GPU acceleration")
                else:
                    self.device = "cpu"
                    dtype = torch.float32
                    logger.info("🖥️ Using CPU (no GPU detected)")
                
                logger.info(f"   Device: {self.device} | Dtype: {dtype}")
            
            # Load pipeline
            if self.is_lora:
                logger.info(f"📦 Loading SDXL base model with LoRA adapter from: {model_path}")
                
                # Load base SDXL model
                base_model = "stabilityai/stable-diffusion-xl-base-1.0"
                logger.info(f"   Loading base model: {base_model}")
                
                # Load base UNet
                base_unet = UNet2DConditionModel.from_pretrained(
                    base_model,
                    subfolder="unet",
                    torch_dtype=dtype
                )
                
                # Attach LoRA adapter
                logger.info(f"   Attaching LoRA adapter...")
                unet_lora = PeftModel.from_pretrained(base_unet, model_path)
                unet_lora.eval()
                
                # Load full pipeline
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    base_model,
                    torch_dtype=dtype,
                    use_safetensors=True
                )
                
                # Replace UNet with LoRA version
                self.pipeline.unet = unet_lora
                logger.info("✅ LoRA adapter attached successfully")
                
            elif is_sdxl:
                # Load full fine-tuned SDXL pipeline
                logger.info(f"📦 Loading fine-tuned SDXL model from: {model_path}")
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    local_files_only=True,
                    variant="fp16" if dtype == torch.float16 else None
                )
            else:
                # Fallback to SD 1.5
                logger.info(f"📦 Loading SD 1.5 model from: {model_path}")
                from diffusers import StableDiffusionPipeline
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Use faster DPM++ scheduler (same quality, fewer steps needed)
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            logger.info("✅ Using DPM++ scheduler for faster generation")
            
            # Apply optimizations
            self._optimize_pipeline()
            
            # Store model info
            self.model_info = {
                "is_loaded": True,
                "device": self.device,
                "dtype": str(dtype),
                "model_path": model_path,
                "model_type": "SDXL" if is_sdxl else "SD 1.5",
                "is_lora": self.is_lora,
                "adapter_type": "LoRA" if self.is_lora else "Full Fine-tuned",
                "resolution": 1024 if is_sdxl else 512,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            }
            
            logger.info("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            self.model_info = {
                "is_loaded": False,
                "error": str(e),
                "device": "none"
            }
            return False
    
    def _optimize_pipeline(self):
        """Apply memory and performance optimizations"""
        try:
            # Enable attention slicing for memory efficiency
            self.pipeline.enable_attention_slicing()
            logger.info("✅ Enabled attention slicing")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
        
        # GPU-specific optimizations
        if self.device == "cuda":
            try:
                # Try xformers if available
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ Enabled xformers")
            except:
                logger.info("ℹ️  Xformers not available, using default attention")
            
            try:
                # Enable TF32 for better performance on Ampere+ GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✅ Enabled TF32")
            except:
                pass
        elif self.device == "mps":
            # MPS-specific optimizations
            logger.info("✅ MPS optimizations enabled")
            # MPS doesn't support xformers, but attention slicing is already enabled
    
    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a floor plan image
        
        Args:
            prompt: Text description of the floor plan
            width: Image width (default 512, SDXL supports up to 1024)
            height: Image height (default 512, SDXL supports up to 1024)
            num_inference_steps: Number of denoising steps (default 30)
            guidance_scale: How closely to follow the prompt (default 7.5)
            seed: Random seed for reproducibility
            
        Returns:
            dict: Contains 'image' (PIL Image) and 'metadata'
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Auto-adjust resolution for SDXL if needed
        is_sdxl = self.model_info.get("model_type") == "SDXL"
        if is_sdxl and width == 512 and height == 512:
            # SDXL works better at 1024x1024, but 512x512 is also supported
            logger.info("Using 512x512 for SDXL (faster, lower quality than 1024x1024)")
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate with autocast for mixed precision on GPU
        if self.device == "cuda":
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    result = self.pipeline(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )
        elif self.device == "mps":
            # MPS doesn't support autocast, use regular inference
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
        else:
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
        
        return {
            "image": result.images[0],
            "metadata": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "seed": seed,
                "model_type": self.model_info.get("model_type", "Unknown")
            }
        }
    
    def unload(self):
        """Unload the model to free memory"""
        if self.pipeline is not None:
            logger.info("Unloading model...")
            self.pipeline = self.pipeline.to("cpu")
            del self.pipeline
            self.pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import gc
            gc.collect()
            
            logger.info("✅ Model unloaded")
    
    def get_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return self.model_info.copy()
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.pipeline is not None


# Global instance
_model_loader = ModelLoader()

def get_model_loader() -> ModelLoader:
    """Get the global model loader instance"""
    return _model_loader
