"""
Model evaluation API routes
"""

from flask import Blueprint, jsonify
import json
import os
from datetime import datetime

evaluate_bp = Blueprint('evaluate', __name__)

@evaluate_bp.route('/evaluate', methods=['GET'])
def get_model_metrics():
    """
    Get current model performance metrics
    
    Returns metrics for both baseline and constraint-aware models
    """
    try:
        metrics_file = '../outputs/metrics/results.json'
        
        if not os.path.exists(metrics_file):
            # Return default metrics if file doesn't exist
            default_metrics = {
                'last_updated': datetime.now().isoformat(),
                'models': {
                    'baseline': {
                        'fid_score': 85.2,
                        'clip_score': 0.62,
                        'adjacency_score': 0.41,
                        'accuracy': 71.3,
                        'training_epochs': 0,
                        'status': 'not_trained'
                    },
                },
                'dataset_stats': {
                    'total_samples': 0,
                    'training_samples': 0,
                    'validation_samples': 0,
                    'avg_room_count': 0,
                    'room_types': []
                }
            }
            
            return jsonify({
                'success': True,
                'metrics': default_metrics,
                'note': 'Default metrics - models not yet trained'
            })
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to load metrics: {str(e)}'
        }), 500

@evaluate_bp.route('/evaluate/comparison', methods=['GET'])
def get_model_comparison():
    """
    Get detailed comparison between baseline and constraint-aware models
    """
    try:
        metrics_file = '../outputs/metrics/results.json'
        
        if not os.path.exists(metrics_file):
            return jsonify({
                'success': False,
                'error': 'Metrics file not found. Train models first.'
            }), 404
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        models = metrics.get('models', {})
        baseline = models.get('baseline', {})
        constraint_aware = models.get('constraint_aware', {})
        
        comparison = {
            'fid_improvement': baseline.get('fid_score', 0) - constraint_aware.get('fid_score', 0),
            'clip_improvement': constraint_aware.get('clip_score', 0) - baseline.get('clip_score', 0),
            'adjacency_improvement': constraint_aware.get('adjacency_score', 0) - baseline.get('adjacency_score', 0),
            'accuracy_improvement': constraint_aware.get('accuracy', 0) - baseline.get('accuracy', 0),
            'better_model': 'constraint_aware' if constraint_aware.get('accuracy', 0) > baseline.get('accuracy', 0) else 'baseline'
        }
        
        return jsonify({
            'success': True,
            'comparison': comparison,
            'baseline_metrics': baseline,
            'constraint_aware_metrics': constraint_aware
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to generate comparison: {str(e)}'
        }), 500

@evaluate_bp.route('/evaluate/history', methods=['GET'])
def get_training_history():
    """
    Get training history and loss curves
    """
    try:
        history_file = '../outputs/metrics/training_history.json'
        
        if not os.path.exists(history_file):
            return jsonify({
                'success': False,
                'error': 'Training history not found. Train models first.'
            }), 404
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        return jsonify({
            'success': True,
            'training_history': history
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to load training history: {str(e)}'
        }), 500