#!/usr/bin/env python3
"""
Comprehensive Prompt Testing for FloorMind
Tests 15 different prompts and evaluates results
"""

import requests
import json
import time
import base64
from pathlib import Path
from datetime import datetime
import sys

# Configuration
API_URL = "http://localhost:5001"
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test prompts - diverse range of floor plan types
TEST_PROMPTS = [
    # Simple residential
    {
        "id": 1,
        "category": "Simple Residential",
        "prompt": "2 bedroom apartment with living room and kitchen",
        "steps": 30,
        "guidance": 7.5
    },
    {
        "id": 2,
        "category": "Simple Residential",
        "prompt": "studio apartment with open floor plan",
        "steps": 30,
        "guidance": 7.5
    },
    
    # Detailed residential
    {
        "id": 3,
        "category": "Detailed Residential",
        "prompt": "modern 3 bedroom apartment with open kitchen, living room, 2 bathrooms, and balcony",
        "steps": 30,
        "guidance": 7.5
    },
    {
        "id": 4,
        "category": "Detailed Residential",
        "prompt": "luxury 4 bedroom house with master suite, walk-in closet, dining room, and garage",
        "steps": 30,
        "guidance": 7.5
    },
    {
        "id": 5,
        "category": "Detailed Residential",
        "prompt": "compact 1 bedroom apartment with efficient layout, bathroom, and kitchenette",
        "steps": 30,
        "guidance": 7.5
    },
    
    # Architectural style specific
    {
        "id": 6,
        "category": "Architectural Style",
        "prompt": "minimalist modern apartment with open concept living space",
        "steps": 30,
        "guidance": 7.5
    },
    {
        "id": 7,
        "category": "Architectural Style",
        "prompt": "traditional family home with separate dining room and den",
        "steps": 30,
        "guidance": 7.5
    },
    
    # Commercial/Office
    {
        "id": 8,
        "category": "Commercial",
        "prompt": "small office space with reception area, 3 offices, and conference room",
        "steps": 30,
        "guidance": 7.5
    },
    {
        "id": 9,
        "category": "Commercial",
        "prompt": "retail store with open floor plan and storage room",
        "steps": 30,
        "guidance": 7.5
    },
    
    # Specific features
    {
        "id": 10,
        "category": "Feature Focused",
        "prompt": "apartment with large windows, open kitchen, and spacious living area",
        "steps": 30,
        "guidance": 7.5
    },
    {
        "id": 11,
        "category": "Feature Focused",
        "prompt": "house with central hallway connecting all rooms",
        "steps": 30,
        "guidance": 7.5
    },
    {
        "id": 12,
        "category": "Feature Focused",
        "prompt": "apartment with L-shaped layout and corner balcony",
        "steps": 30,
        "guidance": 7.5
    },
    
    # Size variations
    {
        "id": 13,
        "category": "Size Variation",
        "prompt": "small efficient 1 bedroom apartment",
        "steps": 30,
        "guidance": 7.5
    },
    {
        "id": 14,
        "category": "Size Variation",
        "prompt": "spacious 5 bedroom family home with multiple bathrooms",
        "steps": 30,
        "guidance": 7.5
    },
    
    # Complex layouts
    {
        "id": 15,
        "category": "Complex Layout",
        "prompt": "multi-level apartment with split bedroom layout and open living area",
        "steps": 30,
        "guidance": 7.5
    }
]

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_model_loaded():
    """Check if model is loaded"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('loaded', False)
        return False
    except:
        return False

def load_model():
    """Load the model"""
    print(f"{YELLOW}Loading model...{RESET}")
    try:
        response = requests.post(f"{API_URL}/model/load", timeout=300)
        if response.status_code == 200:
            print(f"{GREEN}✓ Model loaded successfully{RESET}")
            return True
        else:
            print(f"{RED}✗ Failed to load model: {response.text}{RESET}")
            return False
    except Exception as e:
        print(f"{RED}✗ Error loading model: {e}{RESET}")
        return False

def generate_floor_plan(prompt_data):
    """Generate floor plan from prompt"""
    try:
        payload = {
            "prompt": prompt_data["prompt"],
            "steps": prompt_data["steps"],
            "guidance": prompt_data["guidance"],
            "width": 512,
            "height": 512
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/api/generate-floorplan",
            json=payload,
            timeout=300
        )
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "image_base64": data.get("image_base64"),
                "generation_time": generation_time,
                "metadata": data.get("metadata", {})
            }
        else:
            return {
                "success": False,
                "error": response.text,
                "generation_time": generation_time
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "generation_time": 0
        }

def save_image(image_base64, filename):
    """Save base64 image to file"""
    try:
        # Remove data URL prefix if present
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
        
        image_data = base64.b64decode(image_base64)
        filepath = OUTPUT_DIR / filename
        
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        return str(filepath)
    except Exception as e:
        print(f"{RED}Error saving image: {e}{RESET}")
        return None

def evaluate_result(prompt_data, result):
    """Simple evaluation of result quality"""
    score = 0
    notes = []
    
    # Generation success
    if result["success"]:
        score += 30
        notes.append("✓ Generated successfully")
    else:
        notes.append("✗ Generation failed")
        return score, notes
    
    # Generation time (faster is better)
    gen_time = result["generation_time"]
    if gen_time < 10:
        score += 20
        notes.append(f"✓ Fast generation ({gen_time:.1f}s)")
    elif gen_time < 30:
        score += 15
        notes.append(f"○ Moderate speed ({gen_time:.1f}s)")
    else:
        score += 10
        notes.append(f"○ Slow generation ({gen_time:.1f}s)")
    
    # Prompt complexity (more complex = potentially better if successful)
    prompt_length = len(prompt_data["prompt"].split())
    if prompt_length > 10:
        score += 15
        notes.append("✓ Detailed prompt")
    elif prompt_length > 5:
        score += 10
        notes.append("○ Moderate detail")
    else:
        score += 5
        notes.append("○ Simple prompt")
    
    # Metadata check
    if result.get("metadata"):
        score += 10
        notes.append("✓ Complete metadata")
    
    # Image saved successfully
    if result.get("image_path"):
        score += 25
        notes.append("✓ Image saved")
    
    return score, notes

def main():
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}FloorMind Comprehensive Prompt Testing{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    # Check backend
    print(f"{CYAN}Checking backend...{RESET}")
    if not check_backend():
        print(f"{RED}✗ Backend is not running!{RESET}")
        print(f"\n{YELLOW}Please start the backend first:{RESET}")
        print(f"  python backend/api/app.py")
        print(f"  or: ./start_backend.sh\n")
        return 1
    print(f"{GREEN}✓ Backend is running{RESET}\n")
    
    # Check/load model
    print(f"{CYAN}Checking model...{RESET}")
    if not check_model_loaded():
        if not load_model():
            print(f"{RED}✗ Failed to load model{RESET}\n")
            return 1
    else:
        print(f"{GREEN}✓ Model is loaded{RESET}\n")
    
    # Run tests
    results = []
    total_time = 0
    successful = 0
    
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Running {len(TEST_PROMPTS)} test prompts...{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    for i, prompt_data in enumerate(TEST_PROMPTS, 1):
        print(f"{CYAN}Test {i}/{len(TEST_PROMPTS)}: {prompt_data['category']}{RESET}")
        print(f"Prompt: \"{prompt_data['prompt']}\"")
        
        # Generate
        result = generate_floor_plan(prompt_data)
        
        # Save image if successful
        if result["success"] and result.get("image_base64"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_{prompt_data['id']:02d}_{timestamp}.png"
            image_path = save_image(result["image_base64"], filename)
            result["image_path"] = image_path
        
        # Evaluate
        score, notes = evaluate_result(prompt_data, result)
        
        # Store result
        results.append({
            "prompt_data": prompt_data,
            "result": result,
            "score": score,
            "notes": notes
        })
        
        # Display result
        if result["success"]:
            print(f"{GREEN}✓ Success{RESET} - Time: {result['generation_time']:.1f}s - Score: {score}/100")
            successful += 1
            total_time += result["generation_time"]
        else:
            print(f"{RED}✗ Failed{RESET} - Error: {result.get('error', 'Unknown')}")
        
        for note in notes:
            print(f"  {note}")
        print()
    
    # Summary
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Test Summary{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    print(f"Total Tests: {len(TEST_PROMPTS)}")
    print(f"Successful: {GREEN}{successful}{RESET}")
    print(f"Failed: {RED}{len(TEST_PROMPTS) - successful}{RESET}")
    print(f"Success Rate: {successful/len(TEST_PROMPTS)*100:.1f}%")
    if successful > 0:
        print(f"Average Generation Time: {total_time/successful:.1f}s")
    print()
    
    # Top performers
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Top 5 Best Performing Prompts{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    for i, item in enumerate(sorted_results[:5], 1):
        prompt_data = item["prompt_data"]
        score = item["score"]
        result = item["result"]
        
        print(f"{GREEN}#{i} - Score: {score}/100{RESET}")
        print(f"Category: {prompt_data['category']}")
        print(f"Prompt: \"{prompt_data['prompt']}\"")
        if result["success"]:
            print(f"Time: {result['generation_time']:.1f}s")
            if result.get("image_path"):
                print(f"Image: {result['image_path']}")
        print()
    
    # Category analysis
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Performance by Category{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    categories = {}
    for item in results:
        category = item["prompt_data"]["category"]
        if category not in categories:
            categories[category] = {"scores": [], "times": [], "success": 0}
        
        categories[category]["scores"].append(item["score"])
        if item["result"]["success"]:
            categories[category]["success"] += 1
            categories[category]["times"].append(item["result"]["generation_time"])
    
    for category, data in sorted(categories.items(), key=lambda x: sum(x[1]["scores"])/len(x[1]["scores"]), reverse=True):
        avg_score = sum(data["scores"]) / len(data["scores"])
        avg_time = sum(data["times"]) / len(data["times"]) if data["times"] else 0
        success_rate = data["success"] / len(data["scores"]) * 100
        
        print(f"{CYAN}{category}{RESET}")
        print(f"  Avg Score: {avg_score:.1f}/100")
        print(f"  Success Rate: {success_rate:.0f}%")
        if avg_time > 0:
            print(f"  Avg Time: {avg_time:.1f}s")
        print()
    
    # Recommendations
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Recommendations for Best Results{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    best_prompts = sorted_results[:3]
    print(f"{GREEN}✓ Use these prompt patterns for best results:{RESET}\n")
    
    for i, item in enumerate(best_prompts, 1):
        prompt = item["prompt_data"]["prompt"]
        print(f"{i}. \"{prompt}\"")
    
    print(f"\n{YELLOW}Tips:{RESET}")
    print("• Include specific room counts (e.g., '3 bedroom')")
    print("• Mention key features (e.g., 'open kitchen', 'balcony')")
    print("• Use architectural terms (e.g., 'modern', 'traditional')")
    print("• Be specific but not overly complex")
    print("• Optimal prompt length: 8-15 words")
    
    # Save report
    report_file = OUTPUT_DIR / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(TEST_PROMPTS),
            "successful": successful,
            "results": [{
                "id": r["prompt_data"]["id"],
                "category": r["prompt_data"]["category"],
                "prompt": r["prompt_data"]["prompt"],
                "score": r["score"],
                "success": r["result"]["success"],
                "generation_time": r["result"].get("generation_time", 0),
                "image_path": r["result"].get("image_path")
            } for r in results]
        }, f, indent=2)
    
    print(f"\n{GREEN}✓ Report saved to: {report_file}{RESET}")
    print(f"{GREEN}✓ Images saved to: {OUTPUT_DIR}/{RESET}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
