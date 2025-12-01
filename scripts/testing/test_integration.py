#!/usr/bin/env python3
"""
FloorMind Integration Test
Test the trained model integration
"""

import sys
import os
sys.path.append('backend')

from backend.floormind_generator import FloorMindGenerator
import matplotlib.pyplot as plt

def test_generator():
    """Test the FloorMind generator"""
    
    print("🧪 Testing FloorMind Generator Integration...")
    print("=" * 50)
    
    try:
        # Initialize generator
        print("🔄 Initializing generator...")
        generator = FloorMindGenerator(model_path="google")
        
        if not generator.is_loaded:
            print("❌ Generator failed to load")
            return False
        
        print("✅ Generator loaded successfully!")
        
        # Test single generation
        print("\\n🎨 Testing single generation...")
        description = "Modern 3-bedroom apartment with open kitchen and living room"
        
        image = generator.generate_floor_plan(
            description=description,
            seed=42,
            num_inference_steps=10  # Faster for testing
        )
        
        print("✅ Single generation successful!")
        
        # Save the result
        output_path = generator.save_generation(image, description)
        print(f"💾 Saved to: {output_path}")
        
        # Test model info
        print("\\n📊 Model Information:")
        info = generator.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Display the generated image
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Generated: {description}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("test_generation.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\\n🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_simulation():
    """Simulate API calls"""
    
    print("\\n🌐 Testing API Integration Simulation...")
    print("=" * 50)
    
    try:
        generator = FloorMindGenerator(model_path="google")
        
        # Simulate API requests
        test_requests = [
            {
                "description": "2-bedroom apartment with balcony",
                "width": 512,
                "height": 512,
                "steps": 15,
                "guidance": 7.5,
                "seed": 123
            },
            {
                "description": "Studio apartment with efficient layout",
                "width": 512,
                "height": 512,
                "steps": 15,
                "guidance": 8.0,
                "seed": 456
            }
        ]
        
        results = []
        for i, request in enumerate(test_requests):
            print(f"\\n📝 Processing request {i+1}: {request['description']}")
            
            image = generator.generate_floor_plan(
                description=request['description'],
                width=request['width'],
                height=request['height'],
                num_inference_steps=request['steps'],
                guidance_scale=request['guidance'],
                seed=request['seed']
            )
            
            # Save result
            output_path = generator.save_generation(
                image, 
                request['description'],
                metadata={"api_simulation": True, "request_id": i+1}
            )
            
            results.append({
                "request": request,
                "output_path": output_path,
                "success": True
            })
            
            print(f"✅ Request {i+1} completed: {output_path}")
        
        print(f"\\n🎉 API simulation completed! Generated {len(results)} floor plans")
        return True
        
    except Exception as e:
        print(f"❌ API simulation failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🏠 FloorMind Integration Testing Suite")
    print("=" * 60)
    
    # Check if model files exist
    if not os.path.exists("google/model.safetensors"):
        print("❌ Model files not found in google/ directory")
        print("Please make sure you have:")
        print("   - google/model.safetensors")
        print("   - google/tokenizer_config.json")
        print("   - google/scheduler_config.json")
        return
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Generator Integration
    if test_generator():
        success_count += 1
    
    # Test 2: API Simulation
    if test_api_simulation():
        success_count += 1
    
    # Summary
    print("\\n" + "=" * 60)
    print(f"🏁 INTEGRATION TEST SUMMARY")
    print(f"   Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("   🎉 ALL TESTS PASSED! Your model is ready for production!")
        print("\\n🚀 Next steps:")
        print("   1. Start the API server: python backend/api_endpoints.py")
        print("   2. Test API endpoints with curl or Postman")
        print("   3. Integrate with your frontend")
        print("   4. Deploy to production")
    else:
        print("   ⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()