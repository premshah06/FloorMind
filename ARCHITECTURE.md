# FloorMind SDXL Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FloorMind System                         │
│                  AI Floor Plan Generation Platform               │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│                  │         │                  │         │                  │
│    Frontend      │◄───────►│     Backend      │◄───────►│   SDXL Model     │
│   React App      │  HTTP   │   Flask API      │  Load   │   Fine-tuned     │
│                  │         │                  │         │                  │
└──────────────────┘         └──────────────────┘         └──────────────────┘
  localhost:3000              localhost:5001              ./models/floormind_
                                                          sdxl_finetuned
```

## Component Details

### 1. Frontend (React)

```
frontend/
├── src/
│   ├── pages/
│   │   └── GeneratorPage.js      ← Main UI for generation
│   ├── services/
│   │   └── api.js                ← API client
│   └── App.jsx                   ← Root component
```

**Key Features:**
- User input for floor plan descriptions
- Real-time generation status
- Image display and download
- Model status indicator
- Preset prompts

**API Calls:**
```javascript
// Check health
floorMindAPI.checkHealth()

// Load model
floorMindAPI.loadModel()

// Generate floor plan
floorMindAPI.generateFloorPlan({
  description: "modern 3 bedroom apartment",
  steps: 30,
  guidance: 7.5
})
```

### 2. Backend (Flask)

```
backend/
├── api/
│   └── app.py                    ← Main Flask application
├── core/
│   └── model_loader.py           ← SDXL model loader
└── routes/
    └── generate.py               ← Legacy routes
```

**Key Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/model/info` | GET | Get model information |
| `/model/load` | POST | Load SDXL model |
| `/generate` | POST | Generate floor plan (legacy) |
| `/api/generate-floorplan` | POST | Generate floor plan (new) |
| `/generate/batch` | POST | Generate variations |
| `/presets` | GET | Get example prompts |

**Request Flow:**

```
1. Client sends POST /api/generate-floorplan
   ↓
2. Flask receives request
   ↓
3. Validates prompt and parameters
   ↓
4. Calls model_loader.generate()
   ↓
5. SDXL pipeline generates image
   ↓
6. Converts image to base64
   ↓
7. Returns JSON response with image
```

### 3. Model Loader

```python
class ModelLoader:
    def __init__(self):
        self.pipeline = None
        self.device = "cpu"
    
    def find_model_path(self):
        # Finds SDXL model in priority order
        # 1. Environment variable FLOORMIND_MODEL_DIR
        # 2. ./models/floormind_sdxl_finetuned
        # 3. Fallback paths
    
    def load(self, force_cpu=False):
        # Loads StableDiffusionXLPipeline
        # - Auto-detects SDXL vs SD 1.5
        # - Configures GPU/CPU
        # - Applies optimizations
    
    def generate(self, prompt, width, height, steps, guidance, seed):
        # Generates floor plan image
        # - Uses mixed precision on GPU
        # - Returns PIL Image + metadata
```

**Model Loading Process:**

```
1. Check if model already loaded
   ↓
2. Find model path (env var or default)
   ↓
3. Detect model type (SDXL or SD 1.5)
   ↓
4. Determine device (CUDA or CPU)
   ↓
5. Load StableDiffusionXLPipeline
   ↓
6. Move to device (GPU/CPU)
   ↓
7. Configure DPM++ scheduler
   ↓
8. Apply optimizations:
   - Attention slicing
   - xformers (if available)
   - TF32 (on Ampere+ GPUs)
   ↓
9. Store model info
   ↓
10. Ready for generation
```

### 4. SDXL Model

```
models/floormind_sdxl_finetuned/
├── scheduler/              ← Noise scheduler config
├── text_encoder/           ← CLIP text encoder
├── text_encoder_2/         ← OpenCLIP text encoder (SDXL)
├── tokenizer/              ← CLIP tokenizer
├── tokenizer_2/            ← OpenCLIP tokenizer (SDXL)
├── unet/                   ← Denoising U-Net (fine-tuned)
├── vae/                    ← VAE encoder/decoder
└── model_index.json        ← Model metadata
```

**Generation Pipeline:**

```
Text Prompt
    ↓
Tokenizer → Text Encoder → Text Embeddings
    ↓
Random Noise (latent space)
    ↓
U-Net Denoising (30 steps)
    ├─ Step 1: noise → slightly less noise
    ├─ Step 2: ...
    ├─ ...
    └─ Step 30: clean latent
    ↓
VAE Decoder
    ↓
Floor Plan Image (512x512 or 1024x1024)
```

## Data Flow

### Generation Request

```
┌─────────────┐
│   User      │
│   Types     │
│   Prompt    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  Frontend (GeneratorPage.js)                            │
│  ┌────────────────────────────────────────────────┐    │
│  │ floorMindAPI.generateFloorPlan({               │    │
│  │   description: "modern 3 bedroom apartment",   │    │
│  │   steps: 30,                                   │    │
│  │   guidance: 7.5,                               │    │
│  │   width: 512,                                  │    │
│  │   height: 512                                  │    │
│  │ })                                             │    │
│  └────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP POST
                        │ /api/generate-floorplan
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Backend (app.py)                                       │
│  ┌────────────────────────────────────────────────┐    │
│  │ @app.route('/api/generate-floorplan')         │    │
│  │ def generate_floorplan_api():                  │    │
│  │     data = request.get_json()                  │    │
│  │     result = model_loader.generate(            │    │
│  │         prompt=data['prompt'],                 │    │
│  │         width=512,                             │    │
│  │         height=512,                            │    │
│  │         num_inference_steps=30,                │    │
│  │         guidance_scale=7.5                     │    │
│  │     )                                          │    │
│  │     return jsonify({                           │    │
│  │         "status": "success",                   │    │
│  │         "image_base64": image_to_base64(...)   │    │
│  │     })                                         │    │
│  └────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Model Loader (model_loader.py)                         │
│  ┌────────────────────────────────────────────────┐    │
│  │ def generate(prompt, width, height, ...):      │    │
│  │     generator = torch.Generator(...)           │    │
│  │     with torch.cuda.amp.autocast():            │    │
│  │         result = self.pipeline(                │    │
│  │             prompt=prompt,                     │    │
│  │             width=width,                       │    │
│  │             height=height,                     │    │
│  │             num_inference_steps=steps,         │    │
│  │             guidance_scale=guidance,           │    │
│  │             generator=generator                │    │
│  │         )                                      │    │
│  │     return {"image": result.images[0], ...}    │    │
│  └────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  SDXL Pipeline (Diffusers)                              │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. Tokenize prompt                             │    │
│  │ 2. Encode text → embeddings                    │    │
│  │ 3. Generate random noise                       │    │
│  │ 4. Denoise for 30 steps:                       │    │
│  │    - U-Net predicts noise                      │    │
│  │    - Scheduler removes noise                   │    │
│  │    - Repeat                                    │    │
│  │ 5. VAE decode latent → image                   │    │
│  │ 6. Return PIL Image                            │    │
│  └────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Response                                               │
│  ┌────────────────────────────────────────────────┐    │
│  │ {                                              │    │
│  │   "status": "success",                         │    │
│  │   "image_base64": "data:image/png;base64,...", │    │
│  │   "metadata": {                                │    │
│  │     "model_type": "SDXL",                      │    │
│  │     "steps": 30,                               │    │
│  │     "guidance": 7.5                            │    │
│  │   }                                            │    │
│  │ }                                              │    │
│  └────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Frontend displays image                                │
│  ┌────────────────────────────────────────────────┐    │
│  │ <img src={generatedImage} />                   │    │
│  │ <button onClick={handleDownload}>              │    │
│  │   Download Floor Plan                          │    │
│  │ </button>                                      │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# .env file
FLOORMIND_MODEL_DIR=./models/floormind_sdxl_finetuned
USE_GPU=true
```

### Backend Config

```python
# config/backend_config.py
MODEL_PATH = os.getenv('FLOORMIND_MODEL_DIR', './models/floormind_sdxl_finetuned')
DEVICE = "cuda"  # or "cpu"
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
```

## Performance Optimization

### GPU Acceleration

```
CPU Mode:
- Device: cpu
- Dtype: float32
- Speed: 2-5 minutes per image

GPU Mode:
- Device: cuda
- Dtype: float16 (mixed precision)
- Speed: 5-10 seconds per image
- Speedup: 10-20x faster
```

### Memory Optimization

```python
# Applied automatically in model_loader.py

1. Attention Slicing
   - Reduces memory usage
   - Slight speed tradeoff

2. xformers (if available)
   - Memory-efficient attention
   - Faster on GPU

3. TF32 (Ampere+ GPUs)
   - Better performance
   - No accuracy loss
```

### Scheduler Optimization

```python
# DPM++ Multistep Scheduler
- Faster convergence
- Same quality with fewer steps
- 30 steps ≈ 50 steps with other schedulers
```

## Error Handling

```
┌─────────────────┐
│  Request        │
└────────┬────────┘
         │
         ▼
    ┌────────────┐
    │ Validation │
    └─────┬──────┘
          │
          ├─ Missing prompt? → 400 Bad Request
          ├─ Model not loaded? → 503 Service Unavailable
          ├─ Invalid params? → 400 Bad Request
          │
          ▼
    ┌────────────┐
    │ Generation │
    └─────┬──────┘
          │
          ├─ CUDA OOM? → 500 Internal Error (with message)
          ├─ Model error? → 500 Internal Error (with traceback)
          │
          ▼
    ┌────────────┐
    │  Success   │
    └────────────┘
```

## Deployment Architecture

### Development

```
┌──────────────┐         ┌──────────────┐
│  Frontend    │         │   Backend    │
│  npm start   │◄───────►│  python app  │
│  :3000       │         │  :5001       │
└──────────────┘         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  SDXL Model  │
                         │  Local GPU   │
                         └──────────────┘
```

### Production

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Nginx      │         │   Gunicorn   │         │  GPU Server  │
│   Reverse    │◄───────►│   Workers    │◄───────►│  SDXL Model  │
│   Proxy      │         │   (Flask)    │         │  CUDA        │
└──────────────┘         └──────────────┘         └──────────────┘
     :80/:443                 :5001                  Local/Remote
```

## Technology Stack

### Frontend
- **React** 18.x - UI framework
- **Axios** - HTTP client
- **Framer Motion** - Animations
- **Tailwind CSS** - Styling

### Backend
- **Flask** 3.0 - Web framework
- **Flask-CORS** - CORS handling
- **Python-dotenv** - Environment variables

### AI/ML
- **PyTorch** 2.0 - Deep learning framework
- **Diffusers** 0.25 - Stable Diffusion pipeline
- **Transformers** 4.36 - Text encoders
- **Accelerate** 0.25 - GPU optimization
- **Safetensors** 0.4 - Model weights format

### Model
- **Stable Diffusion XL** - Base architecture
- **Fine-tuned** - On floor plan dataset
- **DPM++ Scheduler** - Fast sampling
- **Mixed Precision** - FP16 on GPU

## Summary

The FloorMind SDXL integration provides:

✅ **Clean Architecture**: Separation of concerns (Frontend, Backend, Model)
✅ **RESTful API**: Standard HTTP endpoints
✅ **GPU Acceleration**: 10-20x faster generation
✅ **Error Handling**: Comprehensive validation and error messages
✅ **Flexibility**: Environment-based configuration
✅ **Production Ready**: Logging, health checks, optimizations
✅ **Easy to Use**: Simple API, clear documentation

The system is designed to be:
- **Scalable**: Can handle multiple concurrent requests
- **Maintainable**: Clear code structure and documentation
- **Extensible**: Easy to add new features
- **Reliable**: Robust error handling and logging
