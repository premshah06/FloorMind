#!/usr/bin/env python3
"""
Verify FloorMind repository is ready for GitHub push.
Checks for common issues before pushing to GitHub.
"""

import os
import sys
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_file_exists(filepath, required=True):
    """Check if a file exists."""
    exists = Path(filepath).exists()
    status = f"{GREEN}✓{RESET}" if exists else f"{RED}✗{RESET}"
    req_text = "(required)" if required else "(optional)"
    print(f"{status} {filepath} {req_text}")
    return exists if required else True

def check_env_file():
    """Check .env file is not tracked."""
    if Path('.env').exists():
        print(f"{YELLOW}⚠{RESET}  .env file exists (make sure it's in .gitignore)")
        with open('.gitignore', 'r') as f:
            if '.env' in f.read():
                print(f"{GREEN}✓{RESET} .env is in .gitignore")
                return True
            else:
                print(f"{RED}✗{RESET} .env is NOT in .gitignore!")
                return False
    return True

def check_large_files():
    """Check for large files that shouldn't be committed."""
    large_files = []
    exclude_dirs = {'venv', 'node_modules', '.git', '__pycache__', 'models', 'data'}
    
    for root, dirs, files in os.walk('.'):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            filepath = Path(root) / file
            try:
                size = filepath.stat().st_size
                if size > 10 * 1024 * 1024:  # 10MB
                    large_files.append((str(filepath), size / (1024 * 1024)))
            except:
                pass
    
    if large_files:
        print(f"\n{YELLOW}⚠{RESET}  Large files found (>10MB):")
        for filepath, size in large_files:
            print(f"   {filepath}: {size:.1f}MB")
        return False
    else:
        print(f"{GREEN}✓{RESET} No large files found")
        return True

def check_secrets():
    """Check for potential secrets in code."""
    patterns = ['api_key', 'secret', 'password', 'token', 'aws_access']
    found_secrets = []
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in {'venv', 'node_modules', '.git', '__pycache__'}]
        
        for file in files:
            if file.endswith(('.py', '.js', '.jsx', '.json', '.env.example')):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        for pattern in patterns:
                            if pattern in content and 'example' not in str(filepath).lower():
                                found_secrets.append((str(filepath), pattern))
                except:
                    pass
    
    if found_secrets:
        print(f"\n{YELLOW}⚠{RESET}  Potential secrets found (review manually):")
        for filepath, pattern in found_secrets[:10]:  # Show first 10
            print(f"   {filepath}: contains '{pattern}'")
        return False
    else:
        print(f"{GREEN}✓{RESET} No obvious secrets found")
        return True

def main():
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}FloorMind GitHub Readiness Check{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    all_checks_passed = True
    
    # Essential files
    print(f"{BLUE}Essential Files:{RESET}")
    all_checks_passed &= check_file_exists('README.md')
    all_checks_passed &= check_file_exists('LICENSE')
    all_checks_passed &= check_file_exists('.gitignore')
    all_checks_passed &= check_file_exists('requirements.txt')
    all_checks_passed &= check_file_exists('.env.example')
    
    # Documentation
    print(f"\n{BLUE}Documentation:{RESET}")
    check_file_exists('ARCHITECTURE.md', required=False)
    check_file_exists('CONTRIBUTING.md', required=False)
    check_file_exists('CHANGELOG.md', required=False)
    
    # Code structure
    print(f"\n{BLUE}Code Structure:{RESET}")
    all_checks_passed &= check_file_exists('backend/api/app.py')
    all_checks_passed &= check_file_exists('backend/core/model_loader.py')
    check_file_exists('frontend/package.json', required=False)
    
    # Security checks
    print(f"\n{BLUE}Security Checks:{RESET}")
    all_checks_passed &= check_env_file()
    all_checks_passed &= check_large_files()
    check_secrets()  # Warning only, doesn't fail
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_checks_passed:
        print(f"{GREEN}✓ All critical checks passed!{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print("1. Review the checklist: .github/PULL_REQUEST_CHECKLIST.md")
        print("2. Update README.md with your GitHub username")
        print("3. git add . && git commit -m 'Initial commit'")
        print("4. git remote add origin <your-repo-url>")
        print("5. git push -u origin main")
        return 0
    else:
        print(f"{RED}✗ Some checks failed. Please review above.{RESET}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
