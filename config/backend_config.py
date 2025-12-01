# Backend Configuration
HOST = "0.0.0.0"
PORT = 5001
DEBUG = False

# Model Configuration
# Use environment variable FLOORMIND_MODEL_DIR or default to SDXL model
import os
MODEL_PATH = os.getenv('FLOORMIND_MODEL_DIR', './models/floormind_sdxl_finetuned')
DEVICE = "cuda"  # Use "cuda" for GPU (10-20x faster), "cpu" for CPU
DTYPE = "float16"  # Use "float16" for GPU, "float32" for CPU

# Generation Defaults
DEFAULT_STEPS = 10  # 10 steps for ultra-fast generation (~3-5s on GPU, good quality)
DEFAULT_GUIDANCE = 7.5
DEFAULT_WIDTH = 512  # SDXL supports 512-1024
DEFAULT_HEIGHT = 512

# Output
OUTPUT_DIR = "output/generated"
