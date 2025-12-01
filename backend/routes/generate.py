"""
Floor plan generation API routes
"""

from flask import Blueprint, request, jsonify
import uuid
import os
import json
from datetime import datetime
import sys

# Add services to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.model_service import ModelService

generate_bp = Blueprint('generate', __name__)
model_service = ModelService()

@generate_bp.route('/generate', methods=['POST'])
def generate_floorplan():
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Missing prompt in request'}), 400
        
        prompt = data['prompt']
        model_type = 'sdxl'  # Using SDXL model
        seed = data.get('seed', None)
        guidance_scale = data.get('guidance_scale', 7.5)
        include_3d = data.get('include_3d', False)
        style = data.get('style', 'modern')
        
        # Validate inputs
        if not prompt.strip():
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        # Using SDXL model - no validation needed
        
        # Generate floor plan
        result = model_service.generate_floorplan(
            prompt=prompt,
            model_type=model_type,
            seed=seed,
            guidance_scale=guidance_scale
        )
        
        if result['success']:
            # Log generation request
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'model_type': model_type,
                'seed': seed,
                'guidance_scale': guidance_scale,
                'output_path': result['image_path'],
                'generation_time': result.get('generation_time', 0)
            }
            
            # Save to generation log
            log_file = '../outputs/generation_log.json'
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            return jsonify({
                'success': True,
                'prompt': prompt,
                'model_type': model_type,
                'image_path': result['image_path'],
                'generation_time': result.get('generation_time', 0),
                'metadata': result.get('metadata', {}),
                'include_3d': include_3d,
                'style': style
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Generation failed: {str(e)}'
        }), 500

@generate_bp.route('/generate/batch', methods=['POST'])
def generate_batch():
    """
    Generate multiple floor plans from a list of prompts
    
    Expected JSON payload:
    {
        "prompts": ["prompt1", "prompt2", ...],
        "model_type": "constraint_aware",
        "seed": 42
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'prompts' not in data:
            return jsonify({'error': 'Missing prompts in request'}), 400
        
        prompts = data['prompts']
        model_type = data.get('model_type', 'constraint_aware')
        seed = data.get('seed', None)
        
        if not isinstance(prompts, list) or len(prompts) == 0:
            return jsonify({'error': 'Prompts must be a non-empty list'}), 400
        
        if len(prompts) > 10:
            return jsonify({'error': 'Maximum 10 prompts per batch'}), 400
        
        results = []
        
        for i, prompt in enumerate(prompts):
            if not prompt.strip():
                results.append({
                    'success': False,
                    'prompt': prompt,
                    'error': 'Empty prompt'
                })
                continue
            
            # Use different seed for each generation if seed provided
            current_seed = seed + i if seed is not None else None
            
            result = model_service.generate_floorplan(
                prompt=prompt,
                model_type=model_type,
                seed=current_seed
            )
            
            results.append({
                'success': result['success'],
                'prompt': prompt,
                'image_path': result.get('image_path'),
                'error': result.get('error'),
                'generation_time': result.get('generation_time', 0)
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_prompts': len(prompts),
            'successful_generations': sum(1 for r in results if r['success'])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Batch generation failed: {str(e)}'
        }), 500
@generate_bp.route('/analyze', methods=['POST'])
def analyze_floorplan():
    """
    Analyze an existing floor plan image using Gemini Vision
    
    Expected form data:
    - image: floor plan image file
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            file.save(tmp_file.name)
            
            # Analyze with model service
            result = model_service.analyze_floorplan(tmp_file.name)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return jsonify(result)
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500

@generate_bp.route('/suggestions', methods=['POST'])
def get_design_suggestions():
    """
    Get design suggestions using Gemini
    
    Expected JSON payload:
    {
        "prompt": "design requirements",
        "style_preferences": ["modern", "minimalist"]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Missing prompt in request'}), 400
        
        prompt = data['prompt']
        style_preferences = data.get('style_preferences', ['modern'])
        
        result = model_service.get_design_suggestions(prompt, style_preferences)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Suggestion generation failed: {str(e)}'
        }), 500

@generate_bp.route('/3d-ready', methods=['POST'])
def generate_3d_ready():
    """
    Generate a floor plan with 3D visualization data
    
    Expected JSON payload:
    {
        "prompt": "floor plan description",
        "style": "modern",
        "seed": 42
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Missing prompt in request'}), 400
        
        prompt = data['prompt']
        style = data.get('style', 'modern')
        seed = data.get('seed', None)
        
        result = model_service.generate_3d_ready_floorplan(prompt, style, seed)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'3D-ready generation failed: {str(e)}'
        }), 500

@generate_bp.route('/models', methods=['GET'])
def get_available_models():
    """
    Get list of available generation models and their capabilities
    """
    try:
        models = model_service.get_available_models()
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get models: {str(e)}'
        }), 500