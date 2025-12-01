#!/bin/bash

# FloorMind CubiCasa5K Dataset Download Script
# Equivalent to the commands you provided, but organized and robust

set -e  # Exit on any error

echo "🏗️  FloorMind CubiCasa5K Dataset Downloader"
echo "=============================================="

# Configuration
DATASET_DIR="$(pwd)/cubicasa5k"
IMAGES_URL="https://zenodo.org/record/2613548/files/cubicasa5k_images.zip"
ANNOTATIONS_URL="https://zenodo.org/record/2613548/files/cubicasa5k_annotations.zip"

# Create data folder
echo "📁 Creating data folder..."
mkdir -p "$DATASET_DIR"
echo "✅ Created directory: $DATASET_DIR"

# Download both archives
echo ""
echo "📦 Downloading CubiCasa5K dataset files..."

echo "⬇️  Downloading images archive..."
if [ ! -f "$DATASET_DIR/cubicasa5k_images.zip" ]; then
    wget "$IMAGES_URL" -O "$DATASET_DIR/cubicasa5k_images.zip"
    echo "✅ Images archive downloaded"
else
    echo "⚠️  Images archive already exists"
fi

echo "⬇️  Downloading annotations archive..."
if [ ! -f "$DATASET_DIR/cubicasa5k_annotations.zip" ]; then
    wget "$ANNOTATIONS_URL" -O "$DATASET_DIR/cubicasa5k_annotations.zip"
    echo "✅ Annotations archive downloaded"
else
    echo "⚠️  Annotations archive already exists"
fi

# Unzip both archives
echo ""
echo "📂 Extracting archives..."

echo "📂 Extracting images..."
if [ -f "$DATASET_DIR/cubicasa5k_images.zip" ]; then
    unzip -q "$DATASET_DIR/cubicasa5k_images.zip" -d "$DATASET_DIR/"
    echo "✅ Images extracted"
    
    # Remove zip file to save space
    rm "$DATASET_DIR/cubicasa5k_images.zip"
    echo "🗑️  Removed images zip file"
else
    echo "❌ Images zip file not found"
fi

echo "📂 Extracting annotations..."
if [ -f "$DATASET_DIR/cubicasa5k_annotations.zip" ]; then
    unzip -q "$DATASET_DIR/cubicasa5k_annotations.zip" -d "$DATASET_DIR/"
    echo "✅ Annotations extracted"
    
    # Remove zip file to save space
    rm "$DATASET_DIR/cubicasa5k_annotations.zip"
    echo "🗑️  Removed annotations zip file"
else
    echo "❌ Annotations zip file not found"
fi

# Analyze dataset structure
echo ""
echo "🔍 Analyzing dataset structure..."

TOTAL_DIRS=$(find "$DATASET_DIR" -type d | wc -l)
TOTAL_FILES=$(find "$DATASET_DIR" -type f | wc -l)
IMAGE_FILES=$(find "$DATASET_DIR" -name "*.png" | wc -l)
JSON_FILES=$(find "$DATASET_DIR" -name "*.json" | wc -l)

echo "📊 Dataset Structure:"
echo "   📁 Total directories: $TOTAL_DIRS"
echo "   📄 Total files: $TOTAL_FILES"
echo "   🖼️  Image files (.png): $IMAGE_FILES"
echo "   📋 JSON files (.json): $JSON_FILES"

# Calculate total size
TOTAL_SIZE=$(du -sh "$DATASET_DIR" | cut -f1)
echo "   💾 Total size: $TOTAL_SIZE"

# Create dataset info file
cat > "$DATASET_DIR/dataset_info.json" << EOF
{
  "name": "CubiCasa5K",
  "description": "Large-scale floor plan dataset with 5000+ annotated floor plans",
  "download_date": "$(date -Iseconds)",
  "source": "https://zenodo.org/record/2613548",
  "citation": "Kalervo, A., Kämppi, M., Lehtiniemi, T., & Rantanen, T. (2019). CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis.",
  "license": "CC BY 4.0",
  "files": {
    "images": "Floor plan images in PNG format",
    "annotations": "JSON annotations with room segmentation and metadata"
  },
  "statistics": {
    "total_directories": $TOTAL_DIRS,
    "total_files": $TOTAL_FILES,
    "image_files": $IMAGE_FILES,
    "json_files": $JSON_FILES,
    "total_size": "$TOTAL_SIZE"
  }
}
EOF

echo "📄 Dataset info saved to: $DATASET_DIR/dataset_info.json"

echo ""
echo "=============================================="
echo "🎉 CubiCasa5K Dataset Download Complete!"
echo "=============================================="
echo "📁 Dataset location: $DATASET_DIR"
echo "📊 Total images: $IMAGE_FILES"
echo "📋 Total annotations: $JSON_FILES"
echo "💾 Total size: $TOTAL_SIZE"

echo ""
echo "💡 Next steps:"
echo "1. Process the dataset:"
echo "   python process_datasets.py --dataset cubicasa5k"
echo "2. Or use the full setup script:"
echo "   python ../setup_datasets.py"
echo "3. Start training:"
echo "   jupyter notebook ../notebooks/FloorMind_Training_and_Analysis.ipynb"

echo ""
echo "✅ Ready for FloorMind training!"