#!/usr/bin/env python3
"""
FloorMind Dataset Processor
Processes architectural datasets and integrates them with FloorMind training pipeline
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class FloorPlanProcessor:
    """Processes floor plan datasets for FloorMind training"""
    
    def __init__(self, data_dir: str = None):
        """Initialize processor"""
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent
        self.datasets_dir = self.data_dir / "datasets"
        self.processed_dir = self.data_dir / "processed"
        self.output_dir = Path(__file__).parent.parent / "outputs"
        
        # Create directories
        for directory in [self.processed_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_cubicasa5k(self, max_samples: int = None, dataset_path: str = None) -> pd.DataFrame:
        """Process CubiCasa5K dataset"""
        
        print("🏗️  Processing CubiCasa5K Dataset")
        print("=" * 50)
        
        # Use provided path or default
        if dataset_path:
            cubicasa_dir = Path(dataset_path)
        else:
            cubicasa_dir = self.data_dir / "cubicasa5k"
        
        if not cubicasa_dir.exists():
            raise FileNotFoundError(f"CubiCasa5K dataset not found at {cubicasa_dir}")
        
        print(f"📁 Dataset location: {cubicasa_dir}")
        
        # Find all floor plan directories
        floor_plan_dirs = []
        
        # Look for different possible structures
        print("🔍 Scanning dataset structure...")
        
        # Check for direct structure: cubicasa5k/high_quality/*/
        high_quality_dir = cubicasa_dir / "high_quality"
        if high_quality_dir.exists():
            print("📂 Found high_quality directory")
            for item in high_quality_dir.iterdir():
                if item.is_dir():
                    model_json = item / "model.json"
                    colorful_dir = item / "colorful"
                    if model_json.exists() and colorful_dir.exists():
                        floor_plan_dirs.append(item)
        
        # Check for direct structure in root
        if not floor_plan_dirs:
            print("📂 Checking root directory structure")
            for root, dirs, files in os.walk(cubicasa_dir):
                if 'model.json' in files and any(d == 'colorful' for d in dirs):
                    floor_plan_dirs.append(Path(root))
        
        print(f"📊 Found {len(floor_plan_dirs)} floor plans")
        
        if len(floor_plan_dirs) == 0:
            print("❌ No valid floor plans found!")
            print("💡 Expected structure:")
            print("   cubicasa5k/")
            print("   ├── high_quality/")
            print("   │   ├── plan_001/")
            print("   │   │   ├── model.json")
            print("   │   │   └── colorful/")
            print("   │   │       └── *.png")
            return pd.DataFrame()
        
        if max_samples:
            floor_plan_dirs = floor_plan_dirs[:max_samples]
            print(f"🔢 Processing first {max_samples} samples")
        
        processed_data = []
        
        for i, plan_dir in enumerate(tqdm(floor_plan_dirs, desc="Processing floor plans")):
            try:
                metadata = self._process_single_cubicasa_plan(plan_dir, i)
                if metadata:
                    processed_data.append(metadata)
                    
            except Exception as e:
                print(f"⚠️  Error processing {plan_dir.name}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        if len(df) > 0:
            # Save processed data
            output_file = self.processed_dir / 'cubicasa5k_processed.csv'
            df.to_csv(output_file, index=False)
            
            print(f"✅ Processed {len(df)} floor plans")
            print(f"📄 Data saved to {output_file}")
            
            # Generate statistics
            self._generate_dataset_statistics(df, 'CubiCasa5K')
            
            return df
        else:
            print("❌ No valid floor plans processed")
            return pd.DataFrame()
    
    def _process_single_cubicasa_plan(self, plan_dir: Path, index: int) -> Optional[Dict]:
        """Process a single CubiCasa5K floor plan"""
        
        # Load annotation
        annotation_file = plan_dir / 'model.json'
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)
        
        # Find image file
        colorful_dir = plan_dir / 'colorful'
        image_files = list(colorful_dir.glob('*.png'))
        
        if not image_files:
            return None
        
        image_file = image_files[0]
        
        # Load and process image
        image = Image.open(image_file)
        width, height = image.size
        
        # Extract room information
        rooms_info = self._extract_room_info(annotation)
        
        # Generate text description
        description = self._generate_description(rooms_info, annotation)
        
        # Copy image to processed directory for easy access
        processed_image_dir = self.processed_dir / 'images'
        processed_image_dir.mkdir(exist_ok=True)
        
        processed_image_name = f"cubicasa5k_{index:05d}.png"
        processed_image_path = processed_image_dir / processed_image_name
        
        # Resize image to standard size if needed
        if width != 512 or height != 512:
            image_resized = image.resize((512, 512), Image.Resampling.LANCZOS)
            image_resized.save(processed_image_path)
        else:
            image.save(processed_image_path)
        
        return {
            'id': f'cubicasa5k_{index:05d}',
            'dataset': 'cubicasa5k',
            'image_path': str(processed_image_path.relative_to(self.data_dir.parent)),
            'original_path': str(image_file),
            'description': description,
            'room_count': rooms_info['room_count'],
            'width': 512,  # Standardized
            'height': 512,  # Standardized
            'original_width': width,
            'original_height': height,
            'room_types': ','.join(rooms_info['room_types']),
            'adjacencies': json.dumps(rooms_info['adjacencies']),
            'area_sqft': rooms_info.get('area_sqft', 0),
            'floors': 1,  # CubiCasa5K is mostly single floor
            'has_balcony': 'balcony' in rooms_info['room_types'],
            'has_garage': 'garage' in rooms_info['room_types'],
            'architectural_style': self._infer_style(rooms_info, annotation)
        }
    
    def _extract_room_info(self, annotation: Dict) -> Dict:
        """Extract room information from CubiCasa5K annotation"""
        
        rooms = []
        room_types = set()
        adjacencies = []
        
        # Extract rooms
        if 'rooms' in annotation:
            for room in annotation['rooms']:
                room_type = room.get('type', 'unknown').lower()
                
                # Normalize room types
                room_type = self._normalize_room_type(room_type)
                
                rooms.append(room_type)
                room_types.add(room_type)
        
        # Extract connections/adjacencies
        if 'connections' in annotation:
            for connection in annotation['connections']:
                if 'rooms' in connection and len(connection['rooms']) >= 2:
                    room1 = self._normalize_room_type(connection['rooms'][0])
                    room2 = self._normalize_room_type(connection['rooms'][1])
                    adjacencies.append((room1, room2))
        
        # Calculate area
        area_sqft = 0
        if 'scale' in annotation and 'area' in annotation:
            area_sqft = annotation.get('area', 0)
        elif len(rooms) > 0:
            # Estimate based on room count
            area_sqft = len(rooms) * 150  # Rough estimate
        
        return {
            'room_count': len(rooms),
            'room_types': sorted(list(room_types)),
            'adjacencies': adjacencies,
            'area_sqft': area_sqft
        }
    
    def _normalize_room_type(self, room_type: str) -> str:
        """Normalize room type names"""
        
        room_type = room_type.lower().strip()
        
        # Mapping for common variations
        mappings = {
            'living': 'living_room',
            'lounge': 'living_room',
            'family': 'living_room',
            'sitting': 'living_room',
            'bed': 'bedroom',
            'master': 'bedroom',
            'guest': 'bedroom',
            'bath': 'bathroom',
            'toilet': 'bathroom',
            'wc': 'bathroom',
            'restroom': 'bathroom',
            'cook': 'kitchen',
            'culinary': 'kitchen',
            'eat': 'dining_room',
            'dining': 'dining_room',
            'breakfast': 'dining_room',
            'hall': 'hallway',
            'corridor': 'hallway',
            'entry': 'hallway',
            'entrance': 'hallway',
            'closet': 'storage',
            'pantry': 'storage',
            'laundry': 'utility',
            'work': 'office',
            'study': 'office',
            'den': 'office'
        }
        
        for key, value in mappings.items():
            if key in room_type:
                return value
        
        return room_type
    
    def _generate_description(self, rooms_info: Dict, annotation: Dict) -> str:
        """Generate natural language description of floor plan"""
        
        room_count = rooms_info['room_count']
        room_types = rooms_info['room_types']
        
        # Base description
        if room_count == 1:
            description = "Studio apartment"
        elif room_count <= 3:
            description = f"{room_count}-room apartment"
        else:
            description = f"{room_count}-room house"
        
        # Add specific room mentions
        important_rooms = ['bedroom', 'bathroom', 'kitchen', 'living_room']
        mentioned_rooms = []
        
        for room in important_rooms:
            if room in room_types:
                count = room_types.count(room)
                if count > 1:
                    mentioned_rooms.append(f"{count} {room}s")
                else:
                    mentioned_rooms.append(room.replace('_', ' '))
        
        if mentioned_rooms:
            description += f" with {', '.join(mentioned_rooms[:3])}"
        
        # Add special features
        special_features = []
        if 'balcony' in room_types:
            special_features.append('balcony')
        if 'garage' in room_types:
            special_features.append('garage')
        if 'office' in room_types:
            special_features.append('home office')
        
        if special_features:
            description += f" and {', '.join(special_features)}"
        
        return description
    
    def _infer_style(self, rooms_info: Dict, annotation: Dict) -> str:
        """Infer architectural style from floor plan"""
        
        room_count = rooms_info['room_count']
        room_types = rooms_info['room_types']
        
        # Simple heuristics for style inference
        if room_count <= 2:
            return 'modern'
        elif 'office' in room_types and 'garage' in room_types:
            return 'contemporary'
        elif room_count >= 5:
            return 'traditional'
        else:
            return 'modern'
    
    def _generate_dataset_statistics(self, df: pd.DataFrame, dataset_name: str):
        """Generate and save dataset statistics"""
        
        print(f"\n📊 Generating {dataset_name} Statistics...")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{dataset_name} Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Room count distribution
        df['room_count'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Room Count Distribution')
        axes[0,0].set_xlabel('Number of Rooms')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # 2. Area distribution
        if 'area_sqft' in df.columns and df['area_sqft'].sum() > 0:
            axes[0,1].hist(df['area_sqft'], bins=30, color='lightcoral', alpha=0.7)
            axes[0,1].set_title('Area Distribution')
            axes[0,1].set_xlabel('Area (sq ft)')
            axes[0,1].set_ylabel('Frequency')
        
        # 3. Room types frequency
        all_room_types = []
        for room_types_str in df['room_types'].dropna():
            all_room_types.extend(room_types_str.split(','))
        
        room_type_counts = pd.Series(all_room_types).value_counts().head(10)
        room_type_counts.plot(kind='bar', ax=axes[0,2], color='lightgreen')
        axes[0,2].set_title('Top 10 Room Types')
        axes[0,2].set_xlabel('Room Type')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Architectural styles
        if 'architectural_style' in df.columns:
            df['architectural_style'].value_counts().plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
            axes[1,0].set_title('Architectural Styles')
            axes[1,0].set_ylabel('')
        
        # 5. Special features
        special_features = ['has_balcony', 'has_garage']
        feature_counts = []
        feature_names = []
        
        for feature in special_features:
            if feature in df.columns:
                count = df[feature].sum()
                feature_counts.append(count)
                feature_names.append(feature.replace('has_', '').title())
        
        if feature_counts:
            axes[1,1].bar(feature_names, feature_counts, color=['gold', 'orange'])
            axes[1,1].set_title('Special Features')
            axes[1,1].set_ylabel('Count')
        
        # 6. Dataset summary stats
        axes[1,2].axis('off')
        summary_text = f"""
Dataset Summary:
• Total samples: {len(df):,}
• Avg room count: {df['room_count'].mean():.1f}
• Room count range: {df['room_count'].min()}-{df['room_count'].max()}
• Most common rooms: {', '.join(room_type_counts.head(3).index)}
• Unique room types: {len(room_type_counts)}
        """
        axes[1,2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / f'{dataset_name.lower()}_analysis.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Statistics visualization saved to {viz_file}")
        
        # Save detailed statistics
        stats = {
            'dataset_name': dataset_name,
            'total_samples': len(df),
            'room_count_stats': {
                'mean': float(df['room_count'].mean()),
                'std': float(df['room_count'].std()),
                'min': int(df['room_count'].min()),
                'max': int(df['room_count'].max()),
                'distribution': df['room_count'].value_counts().to_dict()
            },
            'room_types': {
                'unique_count': len(room_type_counts),
                'most_common': room_type_counts.head(10).to_dict()
            },
            'generation_date': datetime.now().isoformat()
        }
        
        stats_file = self.processed_dir / f'{dataset_name.lower()}_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📄 Detailed statistics saved to {stats_file}")
    
    def create_training_dataset(self, datasets: List[str] = None) -> pd.DataFrame:
        """Create combined training dataset from processed datasets"""
        
        print("🔄 Creating combined training dataset...")
        
        if datasets is None:
            datasets = ['cubicasa5k']  # Default to CubiCasa5K
        
        combined_df = pd.DataFrame()
        
        for dataset in datasets:
            dataset_file = self.processed_dir / f'{dataset}_processed.csv'
            
            if dataset_file.exists():
                df = pd.read_csv(dataset_file)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                print(f"✅ Added {len(df)} samples from {dataset}")
            else:
                print(f"⚠️  Dataset file not found: {dataset_file}")
        
        if len(combined_df) > 0:
            # Save combined dataset
            output_file = self.data_dir / 'metadata.csv'
            combined_df.to_csv(output_file, index=False)
            
            print(f"✅ Created combined dataset with {len(combined_df)} samples")
            print(f"📄 Saved to {output_file}")
            
            return combined_df
        else:
            print("❌ No datasets found to combine")
            return pd.DataFrame()

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description="FloorMind Dataset Processor")
    parser.add_argument('--dataset', type=str, default='cubicasa5k',
                       help='Dataset to process (default: cubicasa5k)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory path')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Specific path to dataset (e.g., /path/to/cubicasa5k)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = FloorPlanProcessor(args.data_dir)
    
    try:
        if args.dataset == 'cubicasa5k':
            df = processor.process_cubicasa5k(args.max_samples, args.dataset_path)
        else:
            print(f"❌ Unknown dataset: {args.dataset}")
            return
        
        if len(df) > 0:
            # Create combined training dataset
            combined_df = processor.create_training_dataset([args.dataset])
            
            print("\n🎉 Dataset processing completed successfully!")
            print(f"📊 Processed {len(df)} floor plans")
            print("\n💡 Next steps:")
            print("1. Train FloorMind models:")
            print("   python train_models.py")
            print("2. Or train step by step:")
            print("   python train_models.py --stage base")
            print("   python train_models.py --stage constraint")
            print("3. Test results:")
            print("   python test_models.py")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()