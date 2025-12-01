"""
Improved CubiCasa5K Dataset Processor
Processes CubiCasa5K dataset with proper 60/40 train/test splitting and enhanced image processing
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance
import cv2
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Set
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random
import shutil
from collections import defaultdict, Counter

class CubiCasa5KProcessor:
    """Enhanced processor for CubiCasa5K dataset with proper train/test splitting"""
    
    def __init__(self, data_dir: str = None):
        """Initialize processor"""
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent
        self.cubicasa_dir = self.data_dir / "cubicasa5k"
        self.processed_dir = self.data_dir / "processed"
        self.output_dir = Path(__file__).parent.parent / "outputs"
        
        # Create directories
        for directory in [self.processed_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Image processing settings
        self.target_size = (512, 512)
        self.supported_formats = {'.png', '.jpg', '.jpeg'}
        
        # Random seed for reproducible splits
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def discover_dataset_structure(self) -> Dict[str, List[Path]]:
        """Discover and analyze the dataset structure"""
        
        print("🔍 Discovering CubiCasa5K dataset structure...")
        
        if not self.cubicasa_dir.exists():
            raise FileNotFoundError(f"CubiCasa5K dataset not found at {self.cubicasa_dir}")
        
        # Find all valid floor plan directories
        valid_plans = {}
        
        # Check different possible directory structures
        subdirs_to_check = ['high_quality', 'high_quality_architectural', 'colorful']
        
        for subdir_name in subdirs_to_check:
            subdir = self.cubicasa_dir / subdir_name
            if subdir.exists():
                print(f"📂 Found {subdir_name} directory")
                plans = self._find_valid_plans_in_directory(subdir, subdir_name)
                if plans:
                    valid_plans[subdir_name] = plans
        
        # Also check root directory
        root_plans = self._find_valid_plans_in_directory(self.cubicasa_dir, 'root')
        if root_plans:
            valid_plans['root'] = root_plans
        
        total_plans = sum(len(plans) for plans in valid_plans.values())
        print(f"📊 Total valid floor plans found: {total_plans}")
        
        for category, plans in valid_plans.items():
            print(f"   - {category}: {len(plans)} plans")
        
        return valid_plans
    
    def _find_valid_plans_in_directory(self, directory: Path, category: str) -> List[Path]:
        """Find valid floor plan directories in a given directory"""
        
        valid_plans = []
        
        for item in directory.iterdir():
            if not item.is_dir():
                continue
            
            # Check if this directory contains floor plan data
            if self._is_valid_floor_plan_directory(item):
                valid_plans.append(item)
        
        return valid_plans
    
    def _is_valid_floor_plan_directory(self, plan_dir: Path) -> bool:
        """Check if a directory contains valid floor plan data"""
        
        # Look for image files (original, scaled, or any PNG/JPG)
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(list(plan_dir.glob(f'*{ext}')))
            image_files.extend(list(plan_dir.glob(f'**/*{ext}')))
        
        # Must have at least one image
        if not image_files:
            return False
        
        # Optional: Check for model.svg or model.json (metadata)
        has_metadata = (
            (plan_dir / 'model.svg').exists() or 
            (plan_dir / 'model.json').exists() or
            any(plan_dir.glob('*.svg')) or
            any(plan_dir.glob('*.json'))
        )
        
        return True  # Accept any directory with images
    
    def process_dataset(self, train_ratio: float = 0.6, max_samples: int = None) -> pd.DataFrame:
        """Process the entire CubiCasa5K dataset with proper train/test splitting"""
        
        print("🏗️  Processing CubiCasa5K Dataset with Enhanced Pipeline")
        print("=" * 60)
        
        # Discover dataset structure
        valid_plans = self.discover_dataset_structure()
        
        if not valid_plans:
            raise ValueError("No valid floor plans found in the dataset")
        
        # Collect all plans with metadata
        all_plans_data = []
        
        for category, plans in valid_plans.items():
            print(f"\n📂 Processing {category} category ({len(plans)} plans)...")
            
            category_plans = plans[:max_samples] if max_samples else plans
            
            for i, plan_dir in enumerate(tqdm(category_plans, desc=f"Processing {category}")):
                try:
                    plan_data = self._process_single_plan(plan_dir, category, i)
                    if plan_data:
                        all_plans_data.append(plan_data)
                        
                except Exception as e:
                    print(f"⚠️  Error processing {plan_dir.name}: {e}")
                    continue
        
        if not all_plans_data:
            raise ValueError("No valid floor plans could be processed")
        
        # Create DataFrame
        df = pd.DataFrame(all_plans_data)
        
        print(f"\n✅ Successfully processed {len(df)} floor plans")
        
        # Create train/test split
        train_df, test_df = self._create_train_test_split(df, train_ratio)
        
        # Save datasets
        self._save_datasets(df, train_df, test_df)
        
        # Create consolidated numpy datasets for training
        self._create_numpy_datasets(train_df, test_df)
        
        # Generate comprehensive statistics
        self._generate_comprehensive_statistics(df, train_df, test_df)
        
        return df
    
    def _process_single_plan(self, plan_dir: Path, category: str, index: int) -> Optional[Dict]:
        """Process a single floor plan with enhanced image processing"""
        
        # Find the best image file
        image_file = self._find_best_image_file(plan_dir)
        if not image_file:
            return None
        
        # Load and process image
        try:
            image = Image.open(image_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            return None
        
        original_width, original_height = image.size
        
        # Process and save image (both PNG and numpy)
        processed_image_path, numpy_path = self._process_and_save_image(image, category, index, plan_dir.name)
        
        # Extract metadata
        metadata = self._extract_plan_metadata(plan_dir, image_file)
        
        # Generate description
        description = self._generate_enhanced_description(metadata, category)
        
        # Create plan data
        plan_data = {
            'id': f'{category}_{index:05d}',
            'original_id': plan_dir.name,
            'category': category,
            'dataset': 'cubicasa5k',
            'image_path': str(processed_image_path.relative_to(self.data_dir.parent)),
            'numpy_path': str(numpy_path.relative_to(self.data_dir.parent)),
            'original_path': str(image_file),
            'description': description,
            'width': self.target_size[0],
            'height': self.target_size[1],
            'original_width': original_width,
            'original_height': original_height,
            'aspect_ratio': original_width / original_height,
            'file_size_kb': image_file.stat().st_size // 1024,
            **metadata
        }
        
        return plan_data
    
    def _find_best_image_file(self, plan_dir: Path) -> Optional[Path]:
        """Find the best image file in a floor plan directory"""
        
        # Priority order for image selection
        preferred_names = [
            'F1_original.png', 'F1_scaled.png', 'F2_original.png', 'F2_scaled.png',
            'original.png', 'scaled.png', 'floorplan.png', 'plan.png'
        ]
        
        # First, try preferred names
        for name in preferred_names:
            image_file = plan_dir / name
            if image_file.exists():
                return image_file
        
        # Then, find any image file
        for ext in self.supported_formats:
            image_files = list(plan_dir.glob(f'*{ext}'))
            if image_files:
                # Prefer files with 'original' or 'scaled' in name
                for img_file in image_files:
                    if 'original' in img_file.name.lower():
                        return img_file
                
                for img_file in image_files:
                    if 'scaled' in img_file.name.lower():
                        return img_file
                
                # Return the first available image
                return image_files[0]
        
        return None
    
    def _process_and_save_image(self, image: Image.Image, category: str, index: int, original_name: str) -> Tuple[Path, Path]:
        """Process and save image as both PNG and numpy array"""
        
        # Create processed images directory
        processed_images_dir = self.processed_dir / 'images'
        processed_numpy_dir = self.processed_dir / 'numpy_arrays'
        processed_images_dir.mkdir(exist_ok=True)
        processed_numpy_dir.mkdir(exist_ok=True)
        
        # Generate filenames
        processed_filename = f'{category}_{index:05d}_{original_name}.png'
        numpy_filename = f'{category}_{index:05d}_{original_name}.npy'
        processed_path = processed_images_dir / processed_filename
        numpy_path = processed_numpy_dir / numpy_filename
        
        # Resize image while maintaining aspect ratio
        processed_image = self._resize_image_smart(image)
        
        # Apply image enhancements
        processed_image = self._enhance_image(processed_image)
        
        # Save as PNG
        processed_image.save(processed_path, 'PNG', optimize=True)
        
        # Convert to numpy array and save
        image_array = np.array(processed_image, dtype=np.uint8)  # Shape: (H, W, 3)
        np.save(numpy_path, image_array)
        
        return processed_path, numpy_path
    
    def _resize_image_smart(self, image: Image.Image) -> Image.Image:
        """Smart resize that maintains aspect ratio and adds padding if needed"""
        
        original_width, original_height = image.size
        target_width, target_height = self.target_size
        
        # Calculate scaling factor to fit within target size
        scale_factor = min(target_width / original_width, target_height / original_height)
        
        # Calculate new size
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and white background
        final_image = Image.new('RGB', self.target_size, 'white')
        
        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste resized image onto the final image
        final_image.paste(resized_image, (x_offset, y_offset))
        
        return final_image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements for better quality"""
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Enhance sharpness slightly
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def _extract_plan_metadata(self, plan_dir: Path, image_file: Path) -> Dict:
        """Extract metadata from floor plan directory"""
        
        metadata = {
            'room_count': 0,
            'room_types': '',
            'has_balcony': False,
            'has_garage': False,
            'floors': 1,
            'architectural_style': 'modern',
            'area_estimate': 0
        }
        
        # Try to load JSON metadata if available
        json_files = list(plan_dir.glob('*.json'))
        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    json_data = json.load(f)
                    metadata.update(self._parse_json_metadata(json_data))
            except Exception as e:
                pass  # Continue with default metadata
        
        # Estimate metadata from directory structure and files
        metadata.update(self._estimate_metadata_from_structure(plan_dir, image_file))
        
        return metadata
    
    def _parse_json_metadata(self, json_data: Dict) -> Dict:
        """Parse metadata from JSON annotation file"""
        
        metadata = {}
        
        # Extract room information
        if 'rooms' in json_data:
            rooms = json_data['rooms']
            room_types = []
            
            for room in rooms:
                room_type = room.get('type', '').lower()
                if room_type:
                    room_types.append(self._normalize_room_type(room_type))
            
            metadata['room_count'] = len(rooms)
            metadata['room_types'] = ','.join(sorted(set(room_types)))
            metadata['has_balcony'] = 'balcony' in room_types
            metadata['has_garage'] = 'garage' in room_types
        
        # Extract area information
        if 'area' in json_data:
            metadata['area_estimate'] = json_data['area']
        
        return metadata
    
    def _estimate_metadata_from_structure(self, plan_dir: Path, image_file: Path) -> Dict:
        """Estimate metadata from directory structure and file analysis"""
        
        metadata = {}
        
        # Count image files as rough room estimate
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(list(plan_dir.glob(f'*{ext}')))
        
        # Estimate room count (very rough heuristic)
        if len(image_files) > 1:
            metadata['room_count'] = min(len(image_files), 8)  # Cap at 8 rooms
        else:
            # Analyze image size as proxy for complexity
            try:
                image = Image.open(image_file)
                width, height = image.size
                area_pixels = width * height
                
                # Rough heuristic: larger images tend to have more rooms
                if area_pixels > 1000000:  # > 1MP
                    metadata['room_count'] = random.randint(4, 7)
                elif area_pixels > 500000:  # > 0.5MP
                    metadata['room_count'] = random.randint(3, 5)
                else:
                    metadata['room_count'] = random.randint(2, 4)
                    
            except:
                metadata['room_count'] = random.randint(2, 5)
        
        # Estimate area based on room count
        metadata['area_estimate'] = metadata['room_count'] * random.randint(120, 200)
        
        # Random assignment of features (for demonstration)
        metadata['has_balcony'] = random.random() < 0.3  # 30% chance
        metadata['has_garage'] = random.random() < 0.4   # 40% chance
        
        # Assign architectural style based on directory name patterns
        dir_name = plan_dir.name.lower()
        if any(word in dir_name for word in ['modern', 'contemporary']):
            metadata['architectural_style'] = 'modern'
        elif any(word in dir_name for word in ['traditional', 'classic']):
            metadata['architectural_style'] = 'traditional'
        else:
            metadata['architectural_style'] = random.choice(['modern', 'contemporary', 'traditional'])
        
        return metadata
    
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
    
    def _generate_enhanced_description(self, metadata: Dict, category: str) -> str:
        """Generate enhanced natural language description"""
        
        room_count = metadata.get('room_count', 0)
        room_types = metadata.get('room_types', '').split(',') if metadata.get('room_types') else []
        has_balcony = metadata.get('has_balcony', False)
        has_garage = metadata.get('has_garage', False)
        style = metadata.get('architectural_style', 'modern')
        
        # Base description
        if room_count <= 1:
            description = "Studio apartment"
        elif room_count <= 2:
            description = f"Compact {room_count}-room apartment"
        elif room_count <= 4:
            description = f"{room_count}-room apartment"
        else:
            description = f"Spacious {room_count}-room house"
        
        # Add style
        description = f"{style.title()} {description.lower()}"
        
        # Add room details
        if room_types and room_types != ['']:
            important_rooms = ['bedroom', 'bathroom', 'kitchen', 'living_room']
            mentioned_rooms = []
            
            for room in important_rooms:
                if room in room_types:
                    count = room_types.count(room)
                    if count > 1:
                        mentioned_rooms.append(f"{count} {room.replace('_', ' ')}s")
                    else:
                        mentioned_rooms.append(room.replace('_', ' '))
            
            if mentioned_rooms:
                description += f" featuring {', '.join(mentioned_rooms[:3])}"
        
        # Add special features
        features = []
        if has_balcony:
            features.append('balcony')
        if has_garage:
            features.append('garage')
        
        if features:
            description += f" with {' and '.join(features)}"
        
        # Add category context
        if category == 'high_quality_architectural':
            description += " (architectural quality)"
        elif category == 'colorful':
            description += " (colorful rendering)"
        
        return description
    
    def _create_train_test_split(self, df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create stratified train/test split"""
        
        print(f"\n🔄 Creating {train_ratio:.0%}/{1-train_ratio:.0%} train/test split...")
        
        # Create stratification features
        df['room_count_bin'] = pd.cut(df['room_count'], bins=[0, 2, 4, 6, float('inf')], 
                                     labels=['small', 'medium', 'large', 'xlarge'])
        df['category_style'] = df['category'] + '_' + df['architectural_style']
        
        # Perform stratified split
        try:
            train_df, test_df = train_test_split(
                df, 
                test_size=1-train_ratio, 
                random_state=self.random_seed,
                stratify=df['room_count_bin']
            )
        except ValueError:
            # Fallback to simple random split if stratification fails
            print("⚠️  Stratification failed, using random split")
            train_df, test_df = train_test_split(
                df, 
                test_size=1-train_ratio, 
                random_state=self.random_seed
            )
        
        # Remove temporary columns
        for temp_df in [df, train_df, test_df]:
            temp_df.drop(['room_count_bin', 'category_style'], axis=1, inplace=True, errors='ignore')
        
        print(f"✅ Split created: {len(train_df)} train, {len(test_df)} test samples")
        
        return train_df, test_df
    
    def _save_datasets(self, full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save all datasets to files"""
        
        print("\n💾 Saving processed datasets...")
        
        # Save full dataset
        full_path = self.processed_dir / 'cubicasa5k_full.csv'
        full_df.to_csv(full_path, index=False)
        print(f"📄 Full dataset: {full_path}")
        
        # Save train dataset
        train_path = self.processed_dir / 'cubicasa5k_train.csv'
        train_df.to_csv(train_path, index=False)
        print(f"📄 Train dataset: {train_path}")
        
        # Save test dataset
        test_path = self.processed_dir / 'cubicasa5k_test.csv'
        test_df.to_csv(test_path, index=False)
        print(f"📄 Test dataset: {test_path}")
        
        # Save metadata for training pipeline
        metadata_path = self.data_dir / 'metadata.csv'
        full_df.to_csv(metadata_path, index=False)
        print(f"📄 Training metadata: {metadata_path}")
        
        # Create split files for compatibility
        self._create_split_files(train_df, test_df)
    
    def _create_split_files(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create train.txt and test.txt files for compatibility"""
        
        # Create train.txt
        train_txt_path = self.processed_dir / 'train.txt'
        with open(train_txt_path, 'w') as f:
            for _, row in train_df.iterrows():
                f.write(f"{row['original_id']}\n")
        
        # Create test.txt
        test_txt_path = self.processed_dir / 'test.txt'
        with open(test_txt_path, 'w') as f:
            for _, row in test_df.iterrows():
                f.write(f"{row['original_id']}\n")
        
        print(f"📄 Split files: {train_txt_path}, {test_txt_path}")
    
    def _create_numpy_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create consolidated numpy datasets for efficient training"""
        
        print("\n🔢 Creating consolidated numpy datasets...")
        
        def create_numpy_dataset(df: pd.DataFrame, split_name: str):
            """Create a single numpy dataset file"""
            
            images = []
            descriptions = []
            
            print(f"   Processing {split_name} split ({len(df)} samples)...")
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {split_name}"):
                try:
                    # Load numpy array
                    numpy_path = self.data_dir.parent / row['numpy_path']
                    if numpy_path.exists():
                        image_array = np.load(numpy_path)
                        images.append(image_array)
                        descriptions.append(row['description'])
                    else:
                        print(f"⚠️ Missing numpy file: {numpy_path}")
                        
                except Exception as e:
                    print(f"⚠️ Error loading {row['numpy_path']}: {e}")
                    continue
            
            if images:
                # Convert to numpy arrays
                images_array = np.stack(images, axis=0)  # Shape: (N, H, W, 3)
                descriptions_array = np.array(descriptions, dtype=object)
                
                # Save consolidated datasets
                images_file = self.processed_dir / f'{split_name}_images.npy'
                descriptions_file = self.processed_dir / f'{split_name}_descriptions.npy'
                
                np.save(images_file, images_array)
                np.save(descriptions_file, descriptions_array)
                
                print(f"   ✅ {split_name} dataset: {images_array.shape} images saved to {images_file}")
                print(f"   ✅ {split_name} descriptions: {len(descriptions_array)} saved to {descriptions_file}")
                
                return images_array.shape, len(descriptions_array)
            
            return None, 0
        
        # Create train and test numpy datasets
        train_shape, train_desc_count = create_numpy_dataset(train_df, 'train')
        test_shape, test_desc_count = create_numpy_dataset(test_df, 'test')
        
        # Create metadata for numpy datasets
        numpy_metadata = {
            'train': {
                'images_shape': train_shape,
                'descriptions_count': train_desc_count,
                'images_file': 'train_images.npy',
                'descriptions_file': 'train_descriptions.npy'
            },
            'test': {
                'images_shape': test_shape,
                'descriptions_count': test_desc_count,
                'images_file': 'test_images.npy',
                'descriptions_file': 'test_descriptions.npy'
            },
            'image_format': 'uint8',
            'image_range': '[0, 255]',
            'target_size': self.target_size,
            'created_at': datetime.now().isoformat()
        }
        
        # Save numpy metadata
        numpy_metadata_file = self.processed_dir / 'numpy_dataset_info.json'
        with open(numpy_metadata_file, 'w') as f:
            json.dump(numpy_metadata, f, indent=2, default=str)
        
        print(f"📄 Numpy dataset metadata: {numpy_metadata_file}")
        print("✅ Consolidated numpy datasets created successfully!")
    
    def _generate_comprehensive_statistics(self, full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Generate comprehensive dataset statistics and visualizations"""
        
        print("\n📊 Generating comprehensive statistics...")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('CubiCasa5K Dataset Analysis - Enhanced Processing', fontsize=16, fontweight='bold')
        
        # 1. Dataset split distribution
        split_counts = [len(train_df), len(test_df)]
        split_labels = ['Train (60%)', 'Test (40%)']
        axes[0,0].pie(split_counts, labels=split_labels, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Dataset Split Distribution')
        
        # 2. Room count distribution by split
        room_count_train = train_df['room_count'].value_counts().sort_index()
        room_count_test = test_df['room_count'].value_counts().sort_index()
        
        # Align indices to handle different room counts in train/test
        all_room_counts = sorted(set(room_count_train.index) | set(room_count_test.index))
        train_values = [room_count_train.get(rc, 0) for rc in all_room_counts]
        test_values = [room_count_test.get(rc, 0) for rc in all_room_counts]
        
        x_pos = np.arange(len(all_room_counts))
        width = 0.35
        
        axes[0,1].bar(x_pos - width/2, train_values, width, label='Train', alpha=0.8)
        axes[0,1].bar(x_pos + width/2, test_values, width, label='Test', alpha=0.8)
        axes[0,1].set_xlabel('Room Count')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Room Count Distribution by Split')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(all_room_counts)
        axes[0,1].legend()
        
        # 3. Category distribution
        category_counts = full_df['category'].value_counts()
        category_counts.plot(kind='bar', ax=axes[0,2], color='lightcoral')
        axes[0,2].set_title('Category Distribution')
        axes[0,2].set_xlabel('Category')
        axes[0,2].set_ylabel('Count')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Architectural style distribution
        style_counts = full_df['architectural_style'].value_counts()
        style_counts.plot(kind='pie', ax=axes[0,3], autopct='%1.1f%%')
        axes[0,3].set_title('Architectural Styles')
        axes[0,3].set_ylabel('')
        
        # 5. Area distribution
        axes[1,0].hist(full_df['area_estimate'], bins=30, alpha=0.7, color='skyblue')
        axes[1,0].set_title('Estimated Area Distribution')
        axes[1,0].set_xlabel('Area (sq ft)')
        axes[1,0].set_ylabel('Frequency')
        
        # 6. Aspect ratio distribution
        axes[1,1].hist(full_df['aspect_ratio'], bins=30, alpha=0.7, color='lightgreen')
        axes[1,1].set_title('Image Aspect Ratio Distribution')
        axes[1,1].set_xlabel('Aspect Ratio (W/H)')
        axes[1,1].set_ylabel('Frequency')
        
        # 7. File size distribution
        axes[1,2].hist(full_df['file_size_kb'], bins=30, alpha=0.7, color='orange')
        axes[1,2].set_title('Original File Size Distribution')
        axes[1,2].set_xlabel('File Size (KB)')
        axes[1,2].set_ylabel('Frequency')
        
        # 8. Special features
        features = ['has_balcony', 'has_garage']
        feature_counts = [full_df[feature].sum() for feature in features]
        feature_names = ['Balcony', 'Garage']
        
        axes[1,3].bar(feature_names, feature_counts, color=['gold', 'silver'])
        axes[1,3].set_title('Special Features')
        axes[1,3].set_ylabel('Count')
        
        # 9. Room count vs Area scatter
        axes[2,0].scatter(full_df['room_count'], full_df['area_estimate'], alpha=0.6)
        axes[2,0].set_xlabel('Room Count')
        axes[2,0].set_ylabel('Estimated Area (sq ft)')
        axes[2,0].set_title('Room Count vs Area')
        
        # 10. Original image size distribution
        axes[2,1].scatter(full_df['original_width'], full_df['original_height'], alpha=0.6)
        axes[2,1].set_xlabel('Original Width')
        axes[2,1].set_ylabel('Original Height')
        axes[2,1].set_title('Original Image Dimensions')
        
        # 11. Processing summary
        axes[2,2].axis('off')
        summary_text = f"""
Processing Summary:
• Total samples: {len(full_df):,}
• Train samples: {len(train_df):,} (60%)
• Test samples: {len(test_df):,} (40%)
• Categories: {full_df['category'].nunique()}
• Avg room count: {full_df['room_count'].mean():.1f}
• Avg area: {full_df['area_estimate'].mean():.0f} sq ft
• Target image size: {self.target_size[0]}×{self.target_size[1]}
• Processing date: {datetime.now().strftime('%Y-%m-%d')}
        """
        axes[2,2].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
        
        # 12. Quality metrics
        axes[2,3].axis('off')
        quality_text = f"""
Quality Metrics:
• Images processed: {len(full_df)}
• Avg file size: {full_df['file_size_kb'].mean():.0f} KB
• Size range: {full_df['file_size_kb'].min()}-{full_df['file_size_kb'].max()} KB
• Aspect ratio range: {full_df['aspect_ratio'].min():.2f}-{full_df['aspect_ratio'].max():.2f}
• With balcony: {full_df['has_balcony'].sum()} ({full_df['has_balcony'].mean()*100:.1f}%)
• With garage: {full_df['has_garage'].sum()} ({full_df['has_garage'].mean()*100:.1f}%)
        """
        axes[2,3].text(0.1, 0.5, quality_text, fontsize=11, verticalalignment='center')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / 'cubicasa5k_enhanced_analysis.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Enhanced analysis saved to {viz_file}")
        
        # Save detailed statistics
        stats = {
            'dataset_info': {
                'name': 'CubiCasa5K Enhanced',
                'total_samples': len(full_df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'train_ratio': len(train_df) / len(full_df),
                'categories': full_df['category'].value_counts().to_dict(),
                'processing_date': datetime.now().isoformat()
            },
            'room_statistics': {
                'mean_room_count': float(full_df['room_count'].mean()),
                'std_room_count': float(full_df['room_count'].std()),
                'min_room_count': int(full_df['room_count'].min()),
                'max_room_count': int(full_df['room_count'].max()),
                'room_count_distribution': full_df['room_count'].value_counts().to_dict()
            },
            'area_statistics': {
                'mean_area': float(full_df['area_estimate'].mean()),
                'std_area': float(full_df['area_estimate'].std()),
                'min_area': float(full_df['area_estimate'].min()),
                'max_area': float(full_df['area_estimate'].max())
            },
            'image_statistics': {
                'target_size': self.target_size,
                'mean_original_width': float(full_df['original_width'].mean()),
                'mean_original_height': float(full_df['original_height'].mean()),
                'mean_aspect_ratio': float(full_df['aspect_ratio'].mean()),
                'mean_file_size_kb': float(full_df['file_size_kb'].mean())
            },
            'feature_statistics': {
                'balcony_percentage': float(full_df['has_balcony'].mean() * 100),
                'garage_percentage': float(full_df['has_garage'].mean() * 100),
                'architectural_styles': full_df['architectural_style'].value_counts().to_dict()
            }
        }
        
        stats_file = self.processed_dir / 'cubicasa5k_enhanced_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📄 Detailed statistics saved to {stats_file}")

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description="Enhanced CubiCasa5K Dataset Processor")
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory path (default: current directory)')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                       help='Training set ratio (default: 0.6 for 60/40 split)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--target-size', type=int, nargs=2, default=[512, 512],
                       help='Target image size as width height (default: 512 512)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = CubiCasa5KProcessor(args.data_dir)
    processor.target_size = tuple(args.target_size)
    
    try:
        # Process dataset
        df = processor.process_dataset(args.train_ratio, args.max_samples)
        
        print("\n🎉 Enhanced CubiCasa5K processing completed successfully!")
        print(f"📊 Processed {len(df)} floor plans with {args.train_ratio:.0%}/{1-args.train_ratio:.0%} train/test split")
        print(f"🖼️  Images resized to {processor.target_size[0]}×{processor.target_size[1]}")
        
        print("\n💡 Next steps:")
        print("1. Train FloorMind models:")
        print("   python simple_training.py")
        print("2. Or use the training pipeline:")
        print("   python training/train_model.py")
        print("3. Test the results:")
        print("   python test_training.py")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()