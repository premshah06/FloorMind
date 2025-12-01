# 🏗️ FloorMind - AI-Powered Floor Plan Generator

Transform natural language descriptions into detailed architectural floor plans using fine-tuned Stable Diffusion XL models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 🌟 Overview

FloorMind is a production-ready AI system that leverages Stable Diffusion XL fine-tuned on 5,050+ architectural floor plans from the CubiCasa5K dataset. Generate professional-grade floor plan visualizations in seconds from simple text descriptions.

**Key Highlights:**
- 🎯 88.8% generation accuracy
- ⚡ 5-10 seconds per image (GPU) / 2-5 minutes (CPU)
- 🎨 512×512 or 1024×1024 resolution output
- 🔌 RESTful API for easy integration
- 💻 Modern React web interface

## ✨ Features

- **AI-Powered Generation**: Fine-tuned Stable Diffusion XL specialized for architectural layouts
- **Flexible Deployment**: Supports both GPU (CUDA) and CPU inference
- **RESTful API**: Clean, documented endpoints for integration
- **Modern Web UI**: Responsive React interface with real-time generation
- **Batch Processing**: Generate multiple variations simultaneously
- **Preset Prompts**: Built-in examples for common floor plan types
- **Production Ready**: Comprehensive error handling, logging, and health checks

## 🛠️ Tech Stack

### Backend
- **Python** 3.8+ - Core runtime
- **Flask** 3.0.0 - REST API framework
- **PyTorch** 2.0.0 - Deep learning framework
- **Diffusers** 0.25.0 - Stable Diffusion pipeline
- **Transformers** 4.36.0 - Text encoders (CLIP/OpenCLIP)

### Frontend
- **React** 18.2.0 - UI framework
- **Tailwind CSS** 3.3.2 - Styling
- **Framer Motion** 10.12.16 - Animations
- **Axios** 1.4.0 - HTTP client

### AI/ML
- **Stable Diffusion XL** - Base architecture
- **Fine-tuned U-Net** - Trained on floor plan dataset
- **CubiCasa5K Dataset** - 5,050 architectural floor plans
- **DPM++ Scheduler** - Fast, high-quality sampling

## 🎥 Demo Video

[![Watch the demo](https://img.youtube.com/vi/zJ-p2vAFcCc/hqdefault.jpg)](https://www.youtube.com/watch?v=zJ-p2vAFcCc)



## 🚀 Quick Start

### Prerequisites

- **Python** 3.8 or higher
- **Node.js** 16+ (for frontend)
- **RAM**: 16GB minimum (32GB recommended)
- **GPU** (Optional): NVIDIA GPU with 8GB+ VRAM for faster inference
- **Storage**: 20GB free space (50GB for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/premshah06/FloorMind.git
cd FloorMind
```

2. **Set up Python environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env to set your model path and preferences
```

4. **Download or train the model**
   - Option A: Download pre-trained model (recommended)
   - Option B: Train your own (see [Training Guide](#-training))

5. **Set up Frontend** (Optional)
```bash
cd frontend
npm install
cd ..
```

### Running the Application

**Backend Server:**
```bash
# Linux/Mac
./start_backend.sh

# Windows
start_backend.bat

# Or directly
python backend/api/app.py
```

Server runs at `http://localhost:5001`

**Frontend UI** (Optional):
```bash
cd frontend
npm start
```

UI opens at `http://localhost:3000`

### Quick Test

```bash
# Check API health
curl http://localhost:5001/health

# Generate a floor plan
curl -X POST http://localhost:5001/api/generate-floorplan \
  -H "Content-Type: application/json" \
  -d '{"prompt": "modern 3 bedroom apartment with open kitchen", "steps": 30}'
```

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and system status |
| GET | `/model/info` | Model information and capabilities |
| POST | `/model/load` | Load AI model into memory |
| POST | `/api/generate-floorplan` | Generate floor plan from prompt |
| POST | `/generate/batch` | Generate multiple variations |
| GET | `/presets` | Get example prompts |

### Generate Floor Plan

**Request:**
```bash
curl -X POST http://localhost:5001/api/generate-floorplan \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Modern 3-bedroom apartment with open kitchen and balcony",
    "steps": 30,
    "guidance": 7.5,
    "width": 512,
    "height": 512,
    "seed": 42
  }'
```

**Response:**
```json
{
  "status": "success",
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "metadata": {
    "model_type": "SDXL",
    "steps": 30,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "seed": 42
  },
  "generation_time": 8.3
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of floor plan |
| `steps` | int | 30 | Number of denoising steps (20-50) |
| `guidance` | float | 7.5 | Guidance scale (5.0-15.0) |
| `width` | int | 512 | Image width (512 or 1024) |
| `height` | int | 512 | Image height (512 or 1024) |
| `seed` | int | random | Random seed for reproducibility |

## 📁 Project Structure

```
floormind/
├── backend/                    # Flask REST API
│   ├── api/
│   │   └── app.py             # Main Flask application
│   ├── core/
│   │   └── model_loader.py    # SDXL model loader & inference
│   └── utils/                 # Helper utilities
├── frontend/                   # React web interface
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/
│   │   │   └── GeneratorPage.js  # Main generation interface
│   │   └── services/
│   │       └── api.js         # API client
│   └── public/                # Static assets
├── scripts/
│   ├── training/              # Model training scripts
│   ├── testing/               # API & integration tests
│   └── deployment/            # Deployment utilities
├── models/
│   └── floormind_sdxl_finetuned/  # Fine-tuned SDXL model
├── data/
│   ├── cubicasa5k/            # Training dataset
│   └── process_cubicasa5k_improved.py  # Data preprocessing
├── config/
│   └── backend_config.py      # Backend configuration
├── generated_floor_plans/     # Output directory
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🎓 Training

To train your own model on custom floor plan data:

### 1. Prepare Dataset

```bash
# Download CubiCasa5K dataset
cd data
./download_cubicasa5k.sh

# Process and prepare training data
python process_cubicasa5k_improved.py
```

### 2. Configure Training

Edit `config/training_config_a100.json`:
```json
{
  "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
  "output_dir": "./models/floormind_sdxl_finetuned",
  "num_train_epochs": 100,
  "learning_rate": 1e-5,
  "train_batch_size": 4
}
```

### 3. Start Training

```bash
# Requires GPU with 16GB+ VRAM
python scripts/training/final_best_fine.py
```

**Training Time:**
- A100 GPU: 4-6 hours
- T4 GPU: 12-16 hours
- V100 GPU: 8-10 hours

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed training documentation.

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 71.7% |
| **FID Score** | 57.4 |
| **CLIP Score** | 0.75 |
| **Generation Time (GPU)** | 5-10 seconds |
| **Generation Time (CPU)** | 2-5 minutes |
| **Model Size** | ~5GB |
| **Output Resolution** | 512×512 or 1024×1024 |

## 💻 Hardware Requirements

### Minimum (CPU Inference)
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 20GB free space
- **Generation**: 2-5 minutes per image

### Recommended (GPU Inference)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 4070, etc.)
- **RAM**: 32GB
- **Storage**: 50GB SSD
- **Generation**: 5-10 seconds per image

### Training Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (T4, V100, A100)
- **RAM**: 32GB+
- **Storage**: 100GB+ (for dataset and checkpoints)
- **Training Time**: 4-16 hours depending on GPU

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:
```bash
# Model Configuration
FLOORMIND_MODEL_DIR=./models/floormind_sdxl_finetuned
USE_GPU=true

# Server Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
FLASK_DEBUG=false

# Generation Defaults
DEFAULT_STEPS=30
DEFAULT_GUIDANCE=7.5
DEFAULT_WIDTH=512
DEFAULT_HEIGHT=512
```

### Backend Configuration

Edit `config/backend_config.py` for advanced settings:
```python
MODEL_PATH = os.getenv('FLOORMIND_MODEL_DIR', './models/floormind_sdxl_finetuned')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5
```

## 🔧 Troubleshooting

### Model Not Found
```bash
# Verify model exists
ls -la models/floormind_sdxl_finetuned/

# Set model path explicitly
export FLOORMIND_MODEL_DIR=/path/to/your/model
```

### Out of Memory (GPU)
```bash
# Force CPU mode
export USE_GPU=false
python backend/api/app.py

# Or reduce batch size in config
```

### CUDA Out of Memory
```python
# In backend/core/model_loader.py, enable memory optimizations:
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
```

### Port Already in Use
```bash
# Change port in .env
FLASK_PORT=5002

# Or kill existing process
lsof -ti:5001 | xargs kill -9
```

### Slow Generation on CPU
- Expected: 2-5 minutes per image on CPU
- Solution: Use GPU or reduce steps to 20
- Alternative: Use smaller model variant

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and data flow
- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** - Comprehensive technical docs
- **[.env.example](.env.example)** - Environment configuration template

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Stability AI](https://stability.ai/)** - Stable Diffusion XL base model
- **[CubiCasa5K Dataset](https://github.com/CubiCasa/CubiCasa5k)** - Floor plan training data
- **[Hugging Face](https://huggingface.co/)** - Diffusers and Transformers libraries
- **[React](https://react.dev/)** & **[Tailwind CSS](https://tailwindcss.com/)** - Frontend frameworks

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/premshah06/FloorMind/issues)
- **Discussions**: [GitHub Discussions](https://github.com/premshah06/FloorMind/discussions)
- **Email**: premshah06@example.com

## 📖 Citation

If you use FloorMind in your research or project, please cite:

```bibtex
@software{floormind2025,
  title={FloorMind: AI-Powered Floor Plan Generation with Stable Diffusion XL},
  author={Prem Shah},
  year={2025},
  url={https://github.com/premshah06/FloorMind},
  version={1.0.0}
}
```

## 🗺️ Roadmap

- [ ] Support for custom LoRA adapters
- [ ] Multi-floor building generation
- [ ] 3D floor plan export
- [ ] Integration with CAD software
- [ ] Mobile app (iOS/Android)
- [ ] Cloud deployment templates (AWS, GCP, Azure)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ by the FloorMind Team

**Status**: Production Ready | **Version**: 1.0.0 | **Last Updated**: November 30, 2025

</div>
