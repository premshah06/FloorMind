#!/usr/bin/env python3
"""
FloorMind Backend - Clean Production Version
Unified, optimized backend for floor plan generation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import base64
import io
import logging
from datetime import datetime
from PIL import Image
import traceback

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required, but recommended

# Add backend to path
backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_root)

from core.model_loader import get_model_loader

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get model loader
model_loader = get_model_loader()


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URL"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "FloorMind AI",
        "version": "1.0.0",
        "model_loaded": model_loader.is_loaded(),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information and status"""
    info = model_loader.get_info()
    return jsonify({
        "status": "success",
        "model": info
    })


@app.route('/model/load', methods=['POST'])
def load_model():
    """Load the model (if not already loaded)"""
    try:
        if model_loader.is_loaded():
            return jsonify({
                "status": "success",
                "message": "Model already loaded",
                "model": model_loader.get_info()
            })
        
        # Get force_cpu from request (default True)
        data = request.get_json() or {}
        force_cpu = data.get('force_cpu', True)
        
        logger.info("Loading model...")
        success = model_loader.load(force_cpu=force_cpu)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Model loaded successfully",
                "model": model_loader.get_info()
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to load model",
                "model": model_loader.get_info()
            }), 500
            
    except Exception as e:
        logger.error(f"Load endpoint error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/generate', methods=['POST'])
def generate_floor_plan():
    """Generate a single floor plan"""
    
    # Check if model is loaded
    if not model_loader.is_loaded():
        return jsonify({
            "status": "error",
            "error": "Model not loaded. Call /model/load first."
        }), 503
    
    try:
        # Parse request
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({
                "status": "error",
                "error": "Missing 'description' field"
            }), 400
        
        description = data['description']
        
        # Get parameters with defaults (optimized for ultra-fast speed)
        params = {
            'width': data.get('width', 512),
            'height': data.get('height', 512),
            'num_inference_steps': data.get('steps', 10),  # 10 steps for ultra-fast generation (~3-5s on GPU)
            'guidance_scale': data.get('guidance', 7.5),
            'seed': data.get('seed', None)
        }
        
        logger.info(f"Generating: {description[:60]}...")
        
        # Generate
        result = model_loader.generate(
            prompt=description,
            **params
        )
        
        # Convert to base64
        image_b64 = image_to_base64(result['image'])
        
        # Save if requested
        saved_path = None
        if data.get('save', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "generated_floor_plans"
            os.makedirs(save_dir, exist_ok=True)
            saved_path = os.path.join(save_dir, f"floor_plan_{timestamp}.png")
            result['image'].save(saved_path)
            logger.info(f"Saved to: {saved_path}")
        
        return jsonify({
            "status": "success",
            "description": description,
            "image": image_b64,
            "parameters": params,
            "saved_path": saved_path,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/generate-floorplan', methods=['POST'])
def generate_floorplan_api():
    """
    Generate a floor plan using the fine-tuned SDXL model
    
    Request body:
    {
        "prompt": "3 bedroom apartment with open kitchen",
        "num_inference_steps": 30 (optional, default 30),
        "guidance_scale": 7.5 (optional, default 7.5),
        "height": 512 (optional, default 512),
        "width": 512 (optional, default 512),
        "seed": null (optional, for reproducibility)
    }
    
    Response:
    {
        "status": "success",
        "image_base64": "data:image/png;base64,...",
        "metadata": {...}
    }
    """
    if not model_loader.is_loaded():
        return jsonify({
            "status": "error",
            "error": "Model not loaded. Call /model/load first."
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({
                "status": "error",
                "error": "Missing 'prompt' field in request body"
            }), 400
        
        prompt = data['prompt']
        
        # Get parameters with defaults
        params = {
            'width': data.get('width', 512),
            'height': data.get('height', 512),
            'num_inference_steps': data.get('num_inference_steps', 30),
            'guidance_scale': data.get('guidance_scale', 7.5),
            'seed': data.get('seed', None)
        }
        
        logger.info(f"Generating floor plan: {prompt[:60]}...")
        
        # Generate
        result = model_loader.generate(
            prompt=prompt,
            **params
        )
        
        # Convert to base64
        image_b64 = image_to_base64(result['image'])
        
        return jsonify({
            "status": "success",
            "image_base64": image_b64,
            "metadata": result['metadata'],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Floor plan generation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/generate/batch', methods=['POST'])
def generate_batch():
    """Generate multiple variations of a floor plan"""
    
    if not model_loader.is_loaded():
        return jsonify({
            "status": "error",
            "error": "Model not loaded"
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({
                "status": "error",
                "error": "Missing 'description' field"
            }), 400
        
        description = data['description']
        num_variations = min(data.get('count', 4), 8)  # Max 8 variations
        
        params = {
            'width': data.get('width', 512),
            'height': data.get('height', 512),
            'num_inference_steps': data.get('steps', 10),  # 10 steps for ultra-fast generation
            'guidance_scale': data.get('guidance', 7.5),
            'seed': data.get('seed', 42)
        }
        
        logger.info(f"Generating {num_variations} variations...")
        
        variations = []
        for i in range(num_variations):
            # Use different seed for each variation
            seed = params['seed'] + i
            
            result = model_loader.generate(
                prompt=description,
                width=params['width'],
                height=params['height'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                seed=seed
            )
            
            variations.append({
                "index": i + 1,
                "image": image_to_base64(result['image']),
                "seed": seed
            })
        
        return jsonify({
            "status": "success",
            "description": description,
            "count": num_variations,
            "variations": variations,
            "parameters": params,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/presets', methods=['GET'])
def get_presets():
    """Get example prompts for floor plans"""
    
    presets = {
        "residential": [
            "Modern 3-bedroom apartment with open kitchen and living room",
            "Cozy 2-bedroom house with separate dining area",
            "Spacious studio apartment with efficient layout",
            "Luxury 4-bedroom penthouse with master suite",
            "Traditional family home with garage",
            "Contemporary loft with industrial design"
        ],
        "commercial": [
            "Small office space with reception and meeting rooms",
            "Open-plan coworking space with flexible seating",
            "Retail store with customer area and storage",
            "Restaurant with dining area and kitchen",
            "Medical clinic with waiting room and exam rooms",
            "Gym with equipment area and changing rooms"
        ],
        "styles": [
            "Minimalist apartment with clean lines",
            "Victorian house with traditional layout",
            "Scandinavian-style compact living",
            "Industrial loft conversion",
            "Mediterranean villa with courtyard",
            "Japanese-inspired zen apartment"
        ]
    }
    
    return jsonify({
        "status": "success",
        "presets": presets
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({
        "status": "error",
        "error": "Internal server error"
    }), 500


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🏗️  FloorMind AI Backend Server")
    print("="*60)
    
    # Auto-load model on startup
    print("\n📦 Loading AI model...")
    # Set force_cpu=False to use GPU if available (10-20x faster for SDXL)
    # For CPU-only systems, set force_cpu=True
    use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
    success = model_loader.load(force_cpu=not use_gpu)
    
    if success:
        print("✅ Model loaded successfully!")
        info = model_loader.get_info()
        print(f"   Device: {info.get('device')}")
        print(f"   Model: {info.get('model_path')}")
    else:
        print("⚠️  Model not loaded - will load on first request")
    
    print("\n📡 Available endpoints:")
    print("   GET  /health                      - Health check")
    print("   GET  /model/info                  - Model information")
    print("   POST /model/load                  - Load model")
    print("   POST /generate                    - Generate floor plan")
    print("   POST /api/generate-floorplan      - Generate floor plan (SDXL)")
    print("   POST /generate/batch              - Generate variations")
    print("   GET  /presets                     - Get example prompts")
    
    print(f"\n🚀 Server starting on http://localhost:5001")
    print("="*60 + "\n")
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        model_loader.unload()
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
