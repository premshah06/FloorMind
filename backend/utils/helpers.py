"""
Utility functions for FloorMind backend
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from datetime import datetime

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        '../data/raw',
        '../data/processed',
        '../outputs/sample_generations',
        '../outputs/metrics',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_metrics(metrics, filename='results.json'):
    """Save metrics to JSON file"""
    metrics_path = f'../outputs/metrics/{filename}'
    
    # Add timestamp
    metrics['last_updated'] = datetime.now().isoformat()
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filename='results.json'):
    """Load metrics from JSON file"""
    metrics_path = f'../outputs/metrics/{filename}'
    
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def calculate_fid_score(real_images, generated_images):
    """
    Calculate Fréchet Inception Distance (FID) score
    Simplified implementation for demonstration
    """
    # This is a placeholder implementation
    # In practice, you would use a proper FID calculation
    # with Inception features
    
    if len(real_images) == 0 or len(generated_images) == 0:
        return float('inf')
    
    # Simulate FID calculation
    # Lower is better
    base_fid = 50.0
    noise = np.random.normal(0, 10, 1)[0]
    return max(0, base_fid + noise)

def calculate_adjacency_score(floorplan_image, room_adjacencies):
    """
    Calculate adjacency consistency score
    
    Args:
        floorplan_image: PIL Image of the floor plan
        room_adjacencies: Expected adjacency relationships
    
    Returns:
        float: Adjacency score between 0 and 1
    """
    # This is a simplified implementation
    # In practice, you would analyze the image to detect rooms
    # and verify adjacency relationships
    
    if room_adjacencies is None or len(room_adjacencies) == 0:
        return 0.5  # Default score
    
    # Simulate adjacency analysis
    correct_adjacencies = np.random.randint(0, len(room_adjacencies) + 1)
    return correct_adjacencies / len(room_adjacencies)

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to image file
        target_size: Target image size (width, height)
    
    Returns:
        PIL.Image: Preprocessed image
    """
    try:
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def extract_room_metadata(image_path):
    """
    Extract room metadata from floor plan image
    This is a placeholder for actual room detection
    
    Args:
        image_path: Path to floor plan image
    
    Returns:
        dict: Room metadata
    """
    # In a real implementation, this would use computer vision
    # to detect and classify rooms in the floor plan
    
    room_types = ['bedroom', 'bathroom', 'kitchen', 'living room', 'dining room']
    
    # Simulate room detection
    detected_rooms = np.random.choice(room_types, size=np.random.randint(2, 6), replace=False)
    
    return {
        'room_count': len(detected_rooms),
        'room_types': detected_rooms.tolist(),
        'total_area': np.random.randint(500, 2000),  # sq ft
        'has_balcony': np.random.choice([True, False]),
        'has_garage': np.random.choice([True, False])
    }

def validate_prompt(prompt):
    """
    Validate and enhance user prompt for floor plan generation
    
    Args:
        prompt: User input prompt
    
    Returns:
        dict: Validation result and enhanced prompt
    """
    if not prompt or not prompt.strip():
        return {
            'valid': False,
            'error': 'Prompt cannot be empty'
        }
    
    # Check for minimum length
    if len(prompt.strip()) < 5:
        return {
            'valid': False,
            'error': 'Prompt too short. Please provide more details.'
        }
    
    # Enhance prompt with architectural keywords
    architectural_keywords = [
        'floor plan', 'blueprint', 'architectural', 'layout', 
        'room', 'bedroom', 'bathroom', 'kitchen', 'living'
    ]
    
    has_architectural_context = any(keyword in prompt.lower() for keyword in architectural_keywords)
    
    enhanced_prompt = prompt
    if not has_architectural_context:
        enhanced_prompt = f"architectural floor plan of {prompt}"
    
    return {
        'valid': True,
        'original_prompt': prompt,
        'enhanced_prompt': enhanced_prompt,
        'has_architectural_context': has_architectural_context
    }

def format_generation_time(seconds):
    """Format generation time in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"