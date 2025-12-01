#!/bin/bash

# FloorMind Codebase Cleanup Script
# Removes unnecessary files and organizes the project

echo "🧹 Cleaning up FloorMind codebase..."
echo ""

# Create archive directory for old files
mkdir -p .archive

# Move unnecessary status/temp files to archive
echo "📦 Archiving temporary status files..."
mv COMPLETE_INTEGRATION_NOW.txt .archive/ 2>/dev/null
mv COMPLETE_SUMMARY.txt .archive/ 2>/dev/null
mv ENVIRONMENT_ISSUE.txt .archive/ 2>/dev/null
mv FINAL_ANSWER.txt .archive/ 2>/dev/null
mv FINAL_STEPS.txt .archive/ 2>/dev/null
mv FIXED_SUCCESS.txt .archive/ 2>/dev/null
mv FRONTEND_INTEGRATION.txt .archive/ 2>/dev/null
mv FRONTEND_UPDATED.txt .archive/ 2>/dev/null
mv LAUNCH_APP.txt .archive/ 2>/dev/null
mv MODEL_STATUS.txt .archive/ 2>/dev/null
mv READY_TO_UPLOAD.txt .archive/ 2>/dev/null
mv RUN_NOW.txt .archive/ 2>/dev/null
mv START_APP_NOW.txt .archive/ 2>/dev/null
mv SUCCESS.txt .archive/ 2>/dev/null
mv TODAY_PLAN.txt .archive/ 2>/dev/null

# Move old backend files to archive
echo "📦 Archiving old backend files..."
mkdir -p .archive/backend_old
mv backend/app.py .archive/backend_old/ 2>/dev/null
mv backend/app_production.py .archive/backend_old/ 2>/dev/null
mv backend/app_simple.py .archive/backend_old/ 2>/dev/null

# Move old scripts
echo "📦 Archiving old scripts..."
mv quick_start.sh .archive/ 2>/dev/null
mv start_floormind.sh .archive/ 2>/dev/null
mv test_model.py .archive/ 2>/dev/null

# Remove incomplete model
echo "🗑️  Removing incomplete production model..."
rm -rf models/final_model_production

# Clean Python cache
echo "🧹 Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# Clean macOS files
echo "🧹 Cleaning macOS files..."
find . -name ".DS_Store" -delete 2>/dev/null

# Create docs directory and organize documentation
echo "📚 Organizing documentation..."
mkdir -p docs
mv BACKEND_GUIDE.md docs/ 2>/dev/null
mv BEFORE_AFTER_COMPARISON.md docs/ 2>/dev/null
mv FINAL_BACKEND_SUMMARY.md docs/ 2>/dev/null
mv FRONTEND_INTEGRATION_GUIDE.md docs/ 2>/dev/null

# Keep QUICK_START.md and README.md in root

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📁 Project structure:"
echo "   ├── backend/           - Clean backend code"
echo "   ├── docs/              - Documentation"
echo "   ├── models/            - AI models"
echo "   ├── generated_floor_plans/ - Generated images"
echo "   ├── .archive/          - Old/temporary files"
echo "   ├── QUICK_START.md     - Quick start guide"
echo "   ├── README.md          - Main readme"
echo "   └── *.py               - Utility scripts"
echo ""
echo "🗑️  Archived files are in .archive/ (can be deleted later)"
