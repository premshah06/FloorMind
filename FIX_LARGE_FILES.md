# 🔧 Fix Large Files in Git History

## Problem Detected

Your git repository contains a **9.1GB file** (`floormind_sd21_20251027_220748.zip`) in the commit history, which is preventing you from pushing to GitHub.

**Error you saw:**
```
error: RPC failed; HTTP 500
Writing objects: 100% (120/120), 8.93 GiB | 30.74 MiB/s, done.
fatal: the remote end hung up unexpectedly
```

## Solution: Remove Large Files from Git History

### Option 1: Using BFG Repo-Cleaner (Recommended - Fastest)

```bash
# 1. Install BFG (Mac)
brew install bfg

# 2. Create a fresh clone
cd ..
git clone --mirror floormind floormind-cleanup.git
cd floormind-cleanup.git

# 3. Remove files larger than 100MB
bfg --strip-blobs-bigger-than 100M

# 4. Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. Force push
git push --force

# 6. Go back to your working directory
cd ../floormind
git pull --rebase
```

### Option 2: Using git filter-repo (More Control)

```bash
# 1. Install git-filter-repo
pip install git-filter-repo

# 2. Backup your repo
cd ..
cp -r floormind floormind-backup

# 3. Remove the specific file
cd floormind
git filter-repo --path floormind_sd21_20251027_220748.zip --invert-paths

# 4. Force push
git remote add origin https://github.com/YOUR_USERNAME/floormind.git
git push --force origin main
```

### Option 3: Start Fresh (Easiest)

If you don't need the old commit history:

```bash
# 1. Remove git history
rm -rf .git

# 2. Initialize new repo
git init
git add .
git commit -m "Initial commit: FloorMind v1.0.0"

# 3. Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/floormind.git
git branch -M main
git push -u origin main --force
```

## Prevent This in the Future

### Update .gitignore (Already Done)

The `.gitignore` has been updated to exclude:
```
*.zip
*.safetensors
*.bin
*.pth
models/active/
models/floormind_sdxl_finetuned/
data/cubicasa5k/
```

### Use Git LFS for Large Files

If you need to track large files:

```bash
# Install Git LFS
brew install git-lfs  # Mac
# or: apt-get install git-lfs  # Linux

# Initialize
git lfs install

# Track large file types
git lfs track "*.zip"
git lfs track "*.safetensors"
git lfs track "*.bin"

# Commit
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## Recommended Approach for FloorMind

**For your model files (5-10GB):**

1. **Upload to Hugging Face Hub** (Best for ML models)
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload YOUR_USERNAME/floormind-sdxl ./models/floormind_sdxl_finetuned
   ```

2. **Update README with download instructions**
   ```markdown
   ## Model Download
   
   Download the fine-tuned model:
   ```bash
   pip install huggingface_hub
   huggingface-cli download YOUR_USERNAME/floormind-sdxl --local-dir ./models/floormind_sdxl_finetuned
   ```
   ```

3. **Keep git repo clean** - Only code, configs, and documentation

## Quick Fix (Recommended)

Since you want to push to GitHub quickly:

```bash
# 1. Start fresh (preserves your current files)
rm -rf .git
git init
git add .
git commit -m "Initial commit: FloorMind v1.0.0 - Clean repository"

# 2. Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/floormind.git
git branch -M main
git push -u origin main

# 3. Upload model separately to Hugging Face
# (See instructions above)
```

## Verify Before Pushing

```bash
# Check repository size
du -sh .git

# Should be < 100MB for code only

# Check for large files
git ls-files | xargs ls -lh | awk '{if ($5 ~ /M$/) print $5, $9}' | sort -hr

# Verify .gitignore is working
git status --ignored
```

## After Fixing

Your repository should be:
- **Size**: < 100MB (code, configs, docs only)
- **Models**: Hosted on Hugging Face Hub
- **Datasets**: Downloaded separately by users
- **Clean**: No large files in git history

---

**Need help?** Run: `python verify_github_ready.py`
