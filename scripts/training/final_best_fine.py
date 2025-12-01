# Fine-Tuning Stable Diffusion XL 1.0 with LoRA (Colab-ready)

import os
import random
import itertools
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
)
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from peft import LoraConfig, get_peft_model, PeftModel

# ---------------------------
# 1. Reproducibility & device
# ---------------------------
seed = 1337
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# 2. Config
# ---------------------------
@dataclass
class FineTuneConfig:
    # Model
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # DATA PATHS (UPDATED FOR YOUR COLAB)
    # This folder must contain images + matching .txt captions:
    # e.g., floor_001.png + floor_001.txt
    dataset_dir: str = "/content/drive/MyDrive/Fine Tuning Deep Project/images"

    # Where to save LoRA weights
    output_dir: str = "/content/drive/MyDrive/floormind_sdxl_lora"

    # Training
    train_resolution: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_train_steps: int = 1000
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"  # "linear" or "cosine"
    lr_warmup_steps: int = 0

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    train_text_encoder_1: bool = False  # set True if you want LoRA on TE1
    train_text_encoder_2: bool = False  # set True if you want LoRA on TE2

    # Precision & memory
    mixed_precision: str = "bf16"  # "bf16" or "fp16" or "fp32"
    gradient_checkpointing: bool = True
    use_xformers: bool = True
    use_8bit_adam: bool = True
    allow_tf32: bool = True


config = FineTuneConfig()
os.makedirs(config.output_dir, exist_ok=True)

# Precision / TF32
if torch.cuda.is_available() and config.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if torch.cuda.is_available():
    if config.mixed_precision == "bf16" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    elif config.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
else:
    dtype = torch.float32

print("Using dtype:", dtype)

# ---------------------------
# 3. Load SDXL base components
# ---------------------------
print("Loading SDXL base model components...")
vae = AutoencoderKL.from_pretrained(config.model_name, subfolder="vae", torch_dtype=dtype)
tokenizer1 = AutoTokenizer.from_pretrained(config.model_name, subfolder="tokenizer", use_fast=False)
tokenizer2 = AutoTokenizer.from_pretrained(config.model_name, subfolder="tokenizer_2", use_fast=False)
text_encoder1 = CLIPTextModel.from_pretrained(
    config.model_name, subfolder="text_encoder", torch_dtype=dtype
)
text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
    config.model_name, subfolder="text_encoder_2", torch_dtype=dtype
)
unet = UNet2DConditionModel.from_pretrained(
    config.model_name, subfolder="unet", torch_dtype=dtype
)

# Freeze base weights
vae.requires_grad_(False)
text_encoder1.requires_grad_(False)
text_encoder2.requires_grad_(False)
unet.requires_grad_(False)

# Move to device
vae.to(device)
text_encoder1.to(device)
text_encoder2.to(device)
unet.to(device)

# xFormers
if config.use_xformers:
    try:
        import xformers  # noqa
        unet.enable_xformers_memory_efficient_attention()
        print("Enabled xFormers memory-efficient attention.")
    except Exception as e:
        print("xFormers not available, proceeding without it.", e)

# Gradient checkpointing
if config.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    if config.train_text_encoder_1:
        text_encoder1.gradient_checkpointing_enable()
    if config.train_text_encoder_2:
        text_encoder2.gradient_checkpointing_enable()

# ---------------------------
# 4. LoRA setup
# ---------------------------
# LoRA for UNet (main image generator)
lora_unet_config = LoraConfig(
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    lora_dropout=config.lora_dropout,
    bias="none",
)
unet = get_peft_model(unet, lora_unet_config)

# Optional: LoRA for text encoders
if config.train_text_encoder_1:
    lora_te1_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    text_encoder1 = get_peft_model(text_encoder1, lora_te1_config)

if config.train_text_encoder_2:
    lora_te2_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "proj_out"],
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    text_encoder2 = get_peft_model(text_encoder2, lora_te2_config)

# Ensure LoRA-wrapped models are on device
unet.to(device)
if config.train_text_encoder_1:
    text_encoder1.to(device)
if config.train_text_encoder_2:
    text_encoder2.to(device)

# Cast LoRA params to fp32 for numerical stability in mixed precision
if config.mixed_precision in {"fp16", "bf16"}:
    for model in [
        unet,
        text_encoder1 if config.train_text_encoder_1 else None,
        text_encoder2 if config.train_text_encoder_2 else None,
    ]:
        if model is None:
            continue
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

# Count params
trainable_params = [p for p in unet.parameters() if p.requires_grad]
if config.train_text_encoder_1:
    trainable_params += [p for p in text_encoder1.parameters() if p.requires_grad]
if config.train_text_encoder_2:
    trainable_params += [p for p in text_encoder2.parameters() if p.requires_grad]

total_params = sum(
    p.numel()
    for p in itertools.chain(unet.parameters(), text_encoder1.parameters(), text_encoder2.parameters())
)
print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
print(f"Total parameters (UNet + TE1 + TE2): {total_params}")

# ---------------------------
# 5. Dataset
# ---------------------------
class ImageCaptionDataset(Dataset):
    """
    Expects:
      dataset_dir/
        img_001.png
        img_001.txt   (caption)
        img_002.jpg
        img_002.txt
      ...
    """

    def __init__(self, image_dir, tokenizer1, tokenizer2, resolution):
        self.image_dir = image_dir
        self.resolution = resolution
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

        exts = (".png", ".jpg", ".jpeg")
        candidate_files = []
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(exts):
                img_path = os.path.join(image_dir, fname)
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                if os.path.isfile(txt_path):
                    candidate_files.append((img_path, txt_path))

        # Filter out corrupted/unreadable images (like plan_12480.png)
        self.image_files = []
        print(f"Scanning {len(candidate_files)} image-caption candidates for validity...")
        for img_path, txt_path in candidate_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                self.image_files.append((img_path, txt_path))
            except Exception as e:
                print(f"Skipping corrupted image: {img_path} ({e})")

        print(f"Found {len(self.image_files)} valid image-caption pairs in {image_dir}.")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, txt_path = self.image_files[idx]

        # Image
        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)

        # Caption
        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # Tokenize for both text encoders
        tokens1 = self.tokenizer1(
            caption,
            max_length=self.tokenizer1.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens2 = self.tokenizer2(
            caption,
            max_length=self.tokenizer2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids_1": tokens1.input_ids[0],
            "attention_mask_1": tokens1.attention_mask[0],
            "input_ids_2": tokens2.input_ids[0],
            "attention_mask_2": tokens2.attention_mask[0],
        }


train_dataset = ImageCaptionDataset(config.dataset_dir, tokenizer1, tokenizer2, config.train_resolution)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

# ---------------------------
# 6. Noise scheduler + optimizer
# ---------------------------
noise_scheduler = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler")

optimizer_cls = torch.optim.AdamW
if config.use_8bit_adam:
    try:
        import bitsandbytes as bnb  # noqa

        optimizer_cls = bnb.optim.AdamW8bit
        print("Using 8-bit Adam optimizer for lower memory usage.")
    except ImportError:
        print("bitsandbytes not installed, using standard AdamW.")

optimizer = optimizer_cls(trainable_params, lr=config.learning_rate, weight_decay=1e-2)

if config.lr_scheduler == "linear":
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=config.max_train_steps,
    )
elif config.lr_scheduler == "cosine":
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_train_steps,
    )
else:
    lr_scheduler = None

unet.train()
if config.train_text_encoder_1:
    text_encoder1.train()
if config.train_text_encoder_2:
    text_encoder2.train()

# ---------------------------
# 7. Training loop
# ---------------------------
global_step = 0
print("Starting training...")

for epoch in range(config.num_epochs):
    for batch in train_dataloader:
        # Move batch to GPU
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        ids1 = batch["input_ids_1"].to(device)
        mask1 = batch["attention_mask_1"].to(device)
        ids2 = batch["input_ids_2"].to(device)
        mask2 = batch["attention_mask_2"].to(device)

        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Text encoder 1 (hidden states)
        if config.train_text_encoder_1:
            enc1_outputs = text_encoder1(ids1, attention_mask=mask1)
        else:
            with torch.no_grad():
                enc1_outputs = text_encoder1(ids1, attention_mask=mask1)
        hidden_states = enc1_outputs.last_hidden_state

        # Text encoder 2 (pooled embedding)
        if config.train_text_encoder_2:
            enc2_outputs = text_encoder2(ids2, attention_mask=mask2)
        else:
            with torch.no_grad():
                enc2_outputs = text_encoder2(ids2, attention_mask=mask2)
        text_embeds = enc2_outputs.text_embeds

        # SDXL extra time conditioning
        add_time_ids = torch.tensor(
            [
                config.train_resolution,
                config.train_resolution,
                0,
                0,
                config.train_resolution,
                config.train_resolution,
            ],
            device=device,
            dtype=text_embeds.dtype,
        ).unsqueeze(0).repeat(bsz, 1)

        # UNet forward
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": add_time_ids},
        )
        noise_pred = model_pred.sample

        loss = nn.functional.mse_loss(noise_pred, noise)

        # Backprop
        loss.backward()

        # Gradient accumulation
        if (global_step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        global_step += 1

        if global_step % 100 == 0:
            print(f"Step {global_step}: loss={loss.item():.4f}")

        if global_step >= config.max_train_steps:
            break

    print(f"Epoch {epoch + 1} completed.")
    if global_step >= config.max_train_steps:
        break

print("Training completed.")

# ---------------------------
# 8. Save LoRA weights (PEFT)
# ---------------------------
print("Saving LoRA weights to disk...")

os.makedirs(config.output_dir, exist_ok=True)

# Save UNet LoRA
if isinstance(unet, PeftModel):
    unet_lora_dir = os.path.join(config.output_dir, "unet_lora")
    unet.save_pretrained(unet_lora_dir)
    print(f"✅ Saved UNet LoRA weights to: {unet_lora_dir}")
else:
    print("⚠️ UNet is not a PeftModel, nothing to save for LoRA.")

# Optional: save TE1 / TE2 LoRA if enabled later
if getattr(config, "train_text_encoder_1", False) and isinstance(text_encoder1, PeftModel):
    te1_lora_dir = os.path.join(config.output_dir, "text_encoder_1_lora")
    text_encoder1.save_pretrained(te1_lora_dir)
    print(f"✅ Saved Text Encoder 1 LoRA to: {te1_lora_dir}")

if getattr(config, "train_text_encoder_2", False) and isinstance(text_encoder2, PeftModel):
    te2_lora_dir = os.path.join(config.output_dir, "text_encoder_2_lora")
    text_encoder2.save_pretrained(te2_lora_dir)
    print(f"✅ Saved Text Encoder 2 LoRA to: {te2_lora_dir}")

print("All done ✅")
