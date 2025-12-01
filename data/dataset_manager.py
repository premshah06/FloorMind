#!/usr/bin/env python3
"""
FloorMind Dataset Manager
Handles downloading, processing, and managing architectural datasets
"""

import os
import sys
import json
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

class DatasetManager:
    """Manages architectural datasets for FloorMind"""
    
    def __init__(self, base_dir: str = None):
        """Initialize dataset manager"""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.datasets_dir = self.base_dir / "datasets"
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories
        for directory in [self.datasets_dir, self.raw_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            'cubicasa5k': {
                'name': 'CubiCasa5K',
                'description': 'Large-scale floor plan dataset with 5000+ floor plans',
                'urls': {
                    'images': 'https://zenodo.org/record/2613548/files/cubicasa5k_images.zip',
                    'annotations': 'https://zenodo.org/record/2613548/files/cubicasa5k_annotations.zip'
                },
                'size': '~2.5GB',
                'license': 'CC BY 4.0',
                'citation': 'Kalervo et al. CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis. 2019.'
            },
            'rplan': {
                'name': 'RPLAN',
                'description': 'Residential floor plan dataset with spatial relationships',
                'urls': {
                    'data': 'http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/data/rplan_dataset.zip'
                },
                'size': '~500MB',
                'license': 'Academic Use',
                'citation': 'Wu et al. Data-driven Interior Plan Generation via Tree-structured Representation Learning. 2019.'
            },
            'lifull': {
                'name': 'LIFULL HOME\'S Dataset',
                'description': 'Japanese residential floor plans',
                'urls': {
                    'data': 'https://www.nii.ac.jp/dsc/idr/lifull/lifull_dataset.zip'
                },
                'size': '~1GB',
                'license': 'Research Use',
                'citation': 'Nauata et al. House-GAN: Relational Generative Adversarial Networks for Graph-constrained House Layout Generation. 2020.'
            }
        }
    
    def list_available_datasets(self) -> Dict:
        """List all available datasets"""
        return self.dataset_configs
    
    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
        """Download a specific dataset"""
        
        if dataset_name not in self.dataset_configs:
            print(f"❌ Unknown dataset: {dataset_name}")
            print(f"Available datasets: {list(self.dataset_configs.keys())}")
            return False
        
        config = self.dataset_configs[dataset_name]
        dataset_dir = self.datasets_dir / dataset_name
        
        print(f"📦 Downloading {config['name']} dataset...")
        print(f"📄 Description: {config['description']}")
        print(f"📏 Size: {config['size']}")
        print(f"📜 License: {config['license']}")
        
        # Check if already downloaded
        if dataset_dir.exists() and not force_redownload:
            print(f"✅ Dataset already exists at {dataset_dir}")
            print("💡 Use --force to redownload")
            return True
        
        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download files
        success = True
        for file_type, url in config['urls'].items():
            success &= self._download_file(url, dataset_dir, file_type, dataset_name)
        
        if success:
            # Save dataset info
            info_file = dataset_dir / 'dataset_info.json'
            with open(info_file, 'w') as f:
                json.dump({
                    'name': config['name'],
                    'description': config['description'],
                    'download_date': datetime.now().isoformat(),
                    'citation': config['citation'],
                    'license': config['license']
                }, f, indent=2)
            
            print(f"✅ Successfully downloaded {config['name']}")
            return True
        else:
            print(f"❌ Failed to download {config['name']}")
            return False
    
    def _download_file(self, url: str, dataset_dir: Path, file_type: str, dataset_name: str) -> bool:
        """Download a single file with progress bar"""
        
        try:
            filename = f"{dataset_name}_{file_type}.zip"
            filepath = dataset_dir / filename
            
            print(f"⬇️  Downloading {file_type} from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=f"Downloading {filename}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Extract if it's a zip file
            if filepath.suffix == '.zip':
                print(f"📂 Extracting {filename}...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                
                # Remove zip file after extraction
                filepath.unlink()
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {file_type}: {e}")
            return False
    
    def process_cubicasa5k(self) -> bool:
        """Process CubiCasa5K dataset for FloorMind"""
        
        dataset_dir = self.datasets_dir / 'cubicasa5k'
        if not dataset_dir.exists():
            print("❌ CubiCasa5K dataset not found. Download it first.")
            return False
        
        print("🔄 Processing CubiCasa5K dataset...")
        
        # Find image and annotation directories
        image_dirs = list(dataset_dir.glob("**/colorful"))
        annotation_dirs = list(dataset_dir.glob("**/model.json"))
        
        if not image_dirs:
            print("❌ Could not find image directory (colorful)")
            return False
        
        processed_data = []
        image_count = 0
        
        # Process each floor plan
        for image_dir in image_dirs:
            floor_plan_id = image_dir.parent.name
            
            # Find corresponding annotation
            annotation_file = image_dir.parent / "model.json"
            
            if not annotation_file.exists():
                continue
            
            try:
                # Load annotation
                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)
                
                # Find floor plan image
                image_files = list(image_dir.glob("*.png"))
                if not image_files:
                    continue
                
                image_file = image_files[0]  # Take first image
                
                # Extract metadata
                metadata = self._extract_cubicasa_metadata(annotation, image_file)
                metadata['id'] = floor_plan_id
                metadata['dataset'] = 'cubicasa5k'
                metadata['image_path'] = str(image_file.relative_to(self.base_dir))
                
                processed_data.append(metadata)
                image_count += 1
                
                if image_count % 100 == 0:
                    print(f"   Processed {image_count} floor plans...")
                
            except Exception as e:
                print(f"⚠️  Error processing {floor_plan_id}: {e}")
                continue
        
        # Save processed metadata
        if processed_data:
            df = pd.DataFrame(processed_data)
            output_file = self.processed_dir / 'cubicasa5k_metadata.csv'
            df.to_csv(output_file, index=False)
            
            print(f"✅ Processed {len(processed_data)} CubiCasa5K floor plans")
            print(f"📄 Metadata saved to {output_file}")
            
            # Update main metadata file
            self._update_main_metadata(df)
            
            return True
        else:
            print("❌ No valid floor plans found in CubiCasa5K dataset")
            return False
    
    def _extract_cubicasa_metadata(self, annotation: Dict, image_file: Path) -> Dict:
        """Extract metadata from CubiCasa5K annotation"""
        
        # Get image dimensions
        try:
            with Image.open(image_file) as img:
                width, height = img.size
        except:
            width, height = 0, 0
        
        # Extract room information
        rooms = []
        room_types = set()
        
        if 'rooms' in annotation:
            for room in annotation['rooms']:
                room_type = room.get('type', 'unknown')
                rooms.append(room_type)
                room_types.add(room_type)
        
        # Calculate area (if available)
        area_sqft = 0
        if 'scale' in annotation and 'area' in annotation:
            area_sqft = annotation.get('area', 0)
        
        # Extract adjacency information
        adjacencies = []
        if 'connections' in annotation:
            for connection in annotation['connections']:
                if 'rooms' in connection and len(connection['rooms']) >= 2:
                    adjacencies.append((connection['rooms'][0], connection['rooms'][1]))
        
        return {
            'room_count': len(rooms),
            'width': width,
            'height': height,
            'room_types': ','.join(sorted(room_types)),
            'adjacencies': json.dumps(adjacencies),
            'area_sqft': area_sqft,
            'floors': 1,  # Most CubiCasa5K are single floor
            'description': f"Floor plan with {len(rooms)} rooms: {', '.join(sorted(room_types)[:3])}"
        }
    
    def process_rplan(self) -> bool:
        """Process RPLAN dataset for FloorMind"""
        
        dataset_dir = self.datasets_dir / 'rplan'
        if not dataset_dir.exists():
            print("❌ RPLAN dataset not found. Download it first.")
            return False
        
        print("🔄 Processing RPLAN dataset...")
        
        # Find data files
        data_files = list(dataset_dir.glob("**/*.npz"))
        
        if not data_files:
            print("❌ Could not find RPLAN data files")
            return False
        
        processed_data = []
        
        for i, data_file in enumerate(tqdm(data_files, desc="Processing RPLAN")):
            try:
                # Load RPLAN data
                data = np.load(data_file)
                
                # Extract floor plan information
                metadata = self._extract_rplan_metadata(data, data_file)
                metadata['id'] = f"rplan_{i:05d}"
                metadata['dataset'] = 'rplan'
                
                processed_data.append(metadata)
                
            except Exception as e:
                print(f"⚠️  Error processing {data_file}: {e}")
                continue
        
        # Save processed metadata
        if processed_data:
            df = pd.DataFrame(processed_data)
            output_file = self.processed_dir / 'rplan_metadata.csv'
            df.to_csv(output_file, index=False)
            
            print(f"✅ Processed {len(processed_data)} RPLAN floor plans")
            print(f"📄 Metadata saved to {output_file}")
            
            # Update main metadata file
            self._update_main_metadata(df)
            
            return True
        else:
            print("❌ No valid floor plans found in RPLAN dataset")
            return False
    
    def _extract_rplan_metadata(self, data: np.ndarray, data_file: Path) -> Dict:
        """Extract metadata from RPLAN data"""
        
        # RPLAN specific processing
        # This is a simplified version - actual implementation would depend on RPLAN format
        
        room_count = len(np.unique(data)) - 1  # Subtract background
        width, height = data.shape if len(data.shape) == 2 else (256, 256)
        
        return {
            'room_count': room_count,
            'width': width,
            'height': height,
            'room_types': 'bedroom,bathroom,kitchen,living_room',  # Default types
            'adjacencies': '[]',
            'area_sqft': width * height // 100,  # Rough estimate
            'floors': 1,
            'description': f"RPLAN floor plan with {room_count} rooms",
            'image_path': str(data_file.relative_to(self.base_dir))
        }
    
    def _update_main_metadata(self, new_df: pd.DataFrame):
        """Update the main metadata.csv file"""
        
        main_metadata_file = self.base_dir / 'metadata.csv'
        
        if main_metadata_file.exists():
            # Load existing metadata
            existing_df = pd.read_csv(main_metadata_file)
            
            # Remove existing entries from the same dataset
            dataset_name = new_df['dataset'].iloc[0]
            existing_df = existing_df[existing_df['dataset'] != dataset_name]
            
            # Combine with new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Save updated metadata
        combined_df.to_csv(main_metadata_file, index=False)
        print(f"📄 Updated main metadata file: {main_metadata_file}")
    
    def create_train_test_split(self, test_size: float = 0.2, val_size: float = 0.1) -> bool:
        """Create train/validation/test splits"""
        
        metadata_file = self.base_dir / 'metadata.csv'
        if not metadata_file.exists():
            print("❌ No metadata file found. Process datasets first.")
            return False
        
        print("🔄 Creating train/validation/test splits...")
        
        df = pd.read_csv(metadata_file)
        
        # Stratify by dataset and room count
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['dataset'], 
            random_state=42
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size / (1 - test_size),
            stratify=train_val_df['dataset'],
            random_state=42
        )
        
        # Save splits
        splits_dir = self.processed_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        train_df.to_csv(splits_dir / 'train.csv', index=False)
        val_df.to_csv(splits_dir / 'val.csv', index=False)
        test_df.to_csv(splits_dir / 'test.csv', index=False)
        
        print(f"✅ Created data splits:")
        print(f"   📚 Train: {len(train_df)} samples")
        print(f"   🔍 Validation: {len(val_df)} samples")
        print(f"   🧪 Test: {len(test_df)} samples")
        
        return True
    
    def generate_dataset_report(self) -> str:
        """Generate a comprehensive dataset report"""
        
        metadata_file = self.base_dir / 'metadata.csv'
        if not metadata_file.exists():
            return "❌ No metadata file found. Process datasets first."
        
        df = pd.read_csv(metadata_file)
        
        report = []
        report.append("📊 FloorMind Dataset Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        report.append("📈 Overall Statistics")
        report.append("-" * 30)
        report.append(f"Total floor plans: {len(df):,}")
        report.append(f"Datasets: {df['dataset'].nunique()}")
        report.append(f"Average room count: {df['room_count'].mean():.1f}")
        report.append(f"Room count range: {df['room_count'].min()} - {df['room_count'].max()}")
        report.append("")
        
        # Dataset breakdown
        report.append("📦 Dataset Breakdown")
        report.append("-" * 30)
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            report.append(f"{dataset.upper()}:")
            report.append(f"  • Samples: {len(dataset_df):,}")
            report.append(f"  • Avg rooms: {dataset_df['room_count'].mean():.1f}")
            report.append(f"  • Avg area: {dataset_df['area_sqft'].mean():.0f} sq ft")
        report.append("")
        
        # Room type analysis
        report.append("🏠 Room Type Analysis")
        report.append("-" * 30)
        all_room_types = []
        for room_types_str in df['room_types'].dropna():
            all_room_types.extend(room_types_str.split(','))
        
        room_type_counts = pd.Series(all_room_types).value_counts().head(10)
        for room_type, count in room_type_counts.items():
            report.append(f"  • {room_type}: {count:,}")
        report.append("")
        
        # Data quality
        report.append("✅ Data Quality")
        report.append("-" * 30)
        report.append(f"Complete records: {len(df.dropna()):,} ({len(df.dropna())/len(df)*100:.1f}%)")
        report.append(f"Missing descriptions: {df['description'].isna().sum():,}")
        report.append(f"Missing adjacencies: {df['adjacencies'].isna().sum():,}")
        report.append("")
        
        return "\n".join(report)
    
    def cleanup_datasets(self, dataset_name: str = None):
        """Clean up downloaded datasets"""
        
        if dataset_name:
            dataset_dir = self.datasets_dir / dataset_name
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
                print(f"🗑️  Removed {dataset_name} dataset")
            else:
                print(f"❌ Dataset {dataset_name} not found")
        else:
            # Clean all datasets
            if self.datasets_dir.exists():
                shutil.rmtree(self.datasets_dir)
                self.datasets_dir.mkdir(parents=True, exist_ok=True)
                print("🗑️  Removed all datasets")

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description="FloorMind Dataset Manager")
    parser.add_argument('command', choices=['list', 'download', 'process', 'split', 'report', 'cleanup'],
                       help='Command to execute')
    parser.add_argument('--dataset', type=str, help='Dataset name (for download/process/cleanup)')
    parser.add_argument('--force', action='store_true', help='Force redownload/reprocess')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size (default: 0.1)')
    
    args = parser.parse_args()
    
    # Initialize dataset manager
    manager = DatasetManager()
    
    if args.command == 'list':
        print("📦 Available Datasets")
        print("=" * 50)
        for name, config in manager.list_available_datasets().items():
            print(f"\n🏗️  {config['name']} ({name})")
            print(f"   📄 {config['description']}")
            print(f"   📏 Size: {config['size']}")
            print(f"   📜 License: {config['license']}")
    
    elif args.command == 'download':
        if not args.dataset:
            print("❌ Please specify a dataset name with --dataset")
            return
        
        success = manager.download_dataset(args.dataset, args.force)
        if success:
            print(f"\n🎉 {args.dataset} downloaded successfully!")
            print("💡 Next steps:")
            print(f"   python dataset_manager.py process --dataset {args.dataset}")
        
    elif args.command == 'process':
        if not args.dataset:
            print("❌ Please specify a dataset name with --dataset")
            return
        
        if args.dataset == 'cubicasa5k':
            success = manager.process_cubicasa5k()
        elif args.dataset == 'rplan':
            success = manager.process_rplan()
        else:
            print(f"❌ Processing not implemented for {args.dataset}")
            return
        
        if success:
            print(f"\n🎉 {args.dataset} processed successfully!")
            print("💡 Next step:")
            print("   python dataset_manager.py split")
    
    elif args.command == 'split':
        success = manager.create_train_test_split(args.test_size, args.val_size)
        if success:
            print("\n🎉 Data splits created successfully!")
            print("💡 Ready for training!")
    
    elif args.command == 'report':
        report = manager.generate_dataset_report()
        print(report)
        
        # Save report to file
        report_file = manager.processed_dir / 'dataset_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to {report_file}")
    
    elif args.command == 'cleanup':
        manager.cleanup_datasets(args.dataset)

if __name__ == "__main__":
    main()