#!/bin/bash

# FloorMind - Clean Push to GitHub Script
# This script removes the old git history with large files and pushes cleanly

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║   FloorMind - Clean Push to GitHub                            ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Confirm with user
echo "⚠️  WARNING: This will remove your git history and create a fresh commit."
echo "   Your files will be preserved, but commit history will be lost."
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Aborted."
    exit 0
fi

echo ""
echo "📦 Step 1: Backing up current .git folder..."
if [ -d .git.backup ]; then
    rm -rf .git.backup
fi
cp -r .git .git.backup
echo "✅ Backup created at .git.backup"

echo ""
echo "🗑️  Step 2: Removing old git history..."
rm -rf .git

echo ""
echo "🆕 Step 3: Initializing fresh repository..."
git init
git branch -M main

echo ""
echo "📝 Step 4: Adding files..."
git add .

echo ""
echo "💾 Step 5: Creating initial commit..."
git commit -m "Initial commit: FloorMind v1.0.0 - AI Floor Plan Generator

- Fine-tuned Stable Diffusion XL for architectural floor plans
- RESTful API with Flask backend
- React web interface
- 71.7% generation accuracy
- Supports GPU and CPU inference
- Complete documentation and examples"

echo ""
echo "🔗 Step 6: Adding remote repository..."
git remote add origin https://github.com/premshah06/FloorMind.git

echo ""
echo "🚀 Step 7: Pushing to GitHub..."
echo "   This may take a moment..."
git push -u origin main --force

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║   ✅ SUCCESS! Repository pushed to GitHub                     ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 Your repository is now live at:"
echo "   https://github.com/premshah06/FloorMind"
echo ""
echo "📋 Next steps:"
echo "   1. Visit your repository on GitHub"
echo "   2. Add repository description and topics"
echo "   3. Enable Issues and Discussions"
echo "   4. Upload model to Hugging Face Hub (see FIX_LARGE_FILES.md)"
echo "   5. Create your first release (v1.0.0)"
echo ""
echo "🗑️  Cleanup: Remove backup with: rm -rf .git.backup"
echo ""
