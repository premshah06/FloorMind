#!/usr/bin/env python3
"""
FloorMind Web Interface
A simple web interface to use your trained model
"""

import streamlit as st
import os
import json
from PIL import Image
import base64
import io
from datetime import datetime

# Page config
st.set_page_config(
    page_title="FloorMind AI - Floor Plan Generator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_model_info():
    """Load model information"""
    
    config_path = "google/training_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def show_test_images():
    """Show the test images generated during training"""
    
    st.subheader("🖼️ Generated Test Samples")
    
    # Find test images
    test_images = []
    if os.path.exists("google"):
        for file in os.listdir("google"):
            if file.startswith("test_generation") and file.endswith(".png"):
                test_images.append(os.path.join("google", file))
    
    if test_images:
        # Display images in columns
        cols = st.columns(min(len(test_images), 4))
        
        for i, img_path in enumerate(test_images[:4]):
            with cols[i % 4]:
                try:
                    img = Image.open(img_path)
                    st.image(img, caption=f"Test Sample {i+1}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading {img_path}: {e}")
    else:
        st.warning("No test images found in google/ directory")

def show_model_info():
    """Show model information"""
    
    st.subheader("📊 Model Information")
    
    config = load_model_info()
    
    if config:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Resolution", f"{config.get('resolution', 'Unknown')}×{config.get('resolution', 'Unknown')}")
            st.metric("Epochs Trained", config.get('num_epochs', 'Unknown'))
            st.metric("Batch Size", config.get('train_batch_size', 'Unknown'))
        
        with col2:
            st.metric("Learning Rate", f"{config.get('learning_rate', 'Unknown')}")
            st.metric("Mixed Precision", config.get('mixed_precision', 'Unknown'))
            st.metric("Base Model", config.get('model_name', 'Unknown').split('/')[-1])
        
        # Show full config in expander
        with st.expander("🔧 Full Configuration"):
            st.json(config)
    else:
        st.warning("Model configuration not found")

def show_integration_guide():
    """Show integration guide"""
    
    st.subheader("🚀 Integration Guide")
    
    st.markdown("""
    ### Method 1: Using Diffusers Pipeline (Recommended)
    
    ```python
    from diffusers import StableDiffusionPipeline
    import torch
    
    # Load your trained model
    pipeline = StableDiffusionPipeline.from_pretrained(
        "path/to/your/google/folder",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )
    
    # Generate floor plan
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    image = pipeline(
        "Modern 3-bedroom apartment with open kitchen",
        num_inference_steps=20,
        guidance_scale=7.5,
        height=512,
        width=512
    ).images[0]
    
    image.save("generated_floor_plan.png")
    ```
    
    ### Method 2: Manual Component Loading
    
    ```python
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
    from transformers import CLIPTextModel, CLIPTokenizer
    import safetensors.torch
    import torch
    
    # Load base components
    base_model = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    
    # Load fine-tuned UNet
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    state_dict = safetensors.torch.load_file("google/model.safetensors")
    unet.load_state_dict(state_dict)
    
    # Create pipeline
    pipeline = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler, safety_checker=None
    )
    ```
    
    ### Method 3: Flask API Integration
    
    ```python
    from flask import Flask, request, jsonify
    from diffusers import StableDiffusionPipeline
    import torch
    import base64
    import io
    
    app = Flask(__name__)
    
    # Load model once at startup
    pipeline = StableDiffusionPipeline.from_pretrained("google")
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    
    @app.route('/generate', methods=['POST'])
    def generate():
        data = request.json
        description = data['description']
        
        image = pipeline(description, num_inference_steps=20).images[0]
        
        # Convert to base64 for JSON response
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            "image": f"data:image/png;base64,{image_base64}",
            "description": description
        })
    
    app.run(host='0.0.0.0', port=5000)
    ```
    """)

def show_file_structure():
    """Show the file structure"""
    
    st.subheader("📁 Your Model Files")
    
    if os.path.exists("google"):
        files = os.listdir("google")
        
        important_files = {
            'model.safetensors': '🎯 Fine-tuned UNet weights (MOST IMPORTANT)',
            'tokenizer_config.json': '📝 Text tokenizer configuration',
            'scheduler_config.json': '⚙️ Diffusion scheduler settings',
            'training_config.json': '🔧 Training configuration',
            'training_stats.csv': '📊 Training statistics',
            'model_index.json': '📋 Pipeline index file',
            'floormind_model.pkl': '📦 Complete model package'
        }
        
        st.markdown("**Essential Files:**")
        for file in files:
            if file in important_files:
                file_path = os.path.join("google", file)
                size_mb = os.path.getsize(file_path) / 1024**2
                st.markdown(f"✅ `{file}` ({size_mb:.1f} MB) - {important_files[file]}")
        
        st.markdown("**Additional Files:**")
        for file in files:
            if file not in important_files:
                file_path = os.path.join("google", file)
                size_mb = os.path.getsize(file_path) / 1024**2
                st.markdown(f"📄 `{file}` ({size_mb:.1f} MB)")
    else:
        st.error("Google directory not found!")

def main():
    """Main app"""
    
    # Header
    st.title("🏠 FloorMind AI - Floor Plan Generator")
    st.markdown("### Your Trained Model Dashboard & Integration Guide")
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    
    pages = {
        "🏠 Overview": "overview",
        "🖼️ Generated Samples": "samples", 
        "📊 Model Info": "info",
        "📁 File Structure": "files",
        "🚀 Integration Guide": "integration",
        "💡 Usage Examples": "examples"
    }
    
    selected_page = st.sidebar.radio("Select Page", list(pages.keys()))
    page_key = pages[selected_page]
    
    # Main content
    if page_key == "overview":
        st.markdown("""
        ## 🎉 Congratulations! Your FloorMind Model is Ready!
        
        Your model has been successfully trained on the CubiCasa5K dataset and is ready for use.
        
        ### ✅ Training Results:
        - **Model Type**: Fine-tuned Stable Diffusion for floor plan generation
        - **Dataset**: CubiCasa5K floor plan images
        - **Resolution**: 512×512 pixels
        - **Training**: Completed successfully with excellent results
        
        ### 🎯 What You Can Do:
        1. **View Generated Samples** - See test images created during training
        2. **Check Model Info** - Review training configuration and metrics
        3. **Integration Guide** - Learn how to use the model in your applications
        4. **File Structure** - Understand what files you have
        
        ### 🚀 Next Steps:
        - Integrate the model with your FloorMind application
        - Deploy to production
        - Generate custom floor plans
        """)
        
        # Quick stats
        config = load_model_info()
        if config:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Resolution", f"{config.get('resolution', 512)}px")
            with col2:
                st.metric("Epochs", config.get('num_epochs', 10))
            with col3:
                st.metric("Batch Size", config.get('train_batch_size', 4))
            with col4:
                st.metric("Precision", config.get('mixed_precision', 'fp16').upper())
    
    elif page_key == "samples":
        show_test_images()
    
    elif page_key == "info":
        show_model_info()
    
    elif page_key == "files":
        show_file_structure()
    
    elif page_key == "integration":
        show_integration_guide()
    
    elif page_key == "examples":
        st.subheader("💡 Usage Examples")
        
        st.markdown("""
        ### Example Prompts for Floor Plan Generation:
        
        **Residential:**
        - "Modern 3-bedroom apartment with open kitchen and living room"
        - "Cozy 2-bedroom house with separate dining area"
        - "Studio apartment with efficient space utilization"
        - "Luxury penthouse with master suite and balcony"
        
        **Commercial:**
        - "Small office space with reception area and meeting rooms"
        - "Open-plan coworking space with flexible seating"
        - "Retail store layout with customer area and storage"
        - "Restaurant floor plan with dining area and kitchen"
        
        **Architectural Styles:**
        - "Traditional colonial house floor plan"
        - "Contemporary minimalist apartment layout"
        - "Victorian-style house with formal rooms"
        - "Modern loft with industrial design elements"
        
        ### Generation Parameters:
        - **num_inference_steps**: 15-30 (higher = better quality, slower)
        - **guidance_scale**: 7.0-10.0 (higher = more adherence to prompt)
        - **seed**: Any integer (for reproducible results)
        - **resolution**: 512×512 (trained resolution)
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("🏠 **FloorMind AI**")
    st.sidebar.markdown("Trained Model Dashboard")

if __name__ == "__main__":
    main()