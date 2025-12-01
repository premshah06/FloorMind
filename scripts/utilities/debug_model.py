#!/usr/bin/env python3
"""
Debug script to check the trained model structure
"""

import pickle
import os

def debug_model_file():
    """Debug the model pickle file"""
    
    model_path = "google/floormind_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"🔍 Debugging model file: {model_path}")
    print(f"📊 File size: {os.path.getsize(model_path) / 1024**2:.1f} MB")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Model loaded successfully!")
        print(f"📋 Model data type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print(f"🔑 Keys in model data: {list(model_data.keys())}")
            
            for key, value in model_data.items():
                print(f"   {key}: {type(value)}")
                
                if key == 'config' and isinstance(value, dict):
                    print(f"      Config keys: {list(value.keys())}")
                elif key == 'training_stats' and hasattr(value, '__len__'):
                    print(f"      Training stats length: {len(value)}")
        else:
            print(f"📦 Model data is not a dictionary, it's: {type(model_data)}")
            
            # Check if it's directly a pipeline
            if hasattr(model_data, 'to'):
                print("🎯 Looks like it might be a pipeline directly")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_file()