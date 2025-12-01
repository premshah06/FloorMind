#!/usr/bin/env python3
"""
FloorMind Migration Script
Migrate from flat structure to organized v2.0 structure
"""

import os
import shutil
import sys
from pathlib import Path

class FloorMindMigrator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "backup_flat_structure"
        self.dry_run = False
    
    def log(self, message, level="INFO"):
        """Log migration messages"""
        print(f"[{level}] {message}")
    
    def create_new_structure(self):
        """Create the new directory structure"""
        
        self.log("Creating new directory structure...")
        
        new_dirs = [
            "src",
            "src/core",
            "src/api", 
            "src/frontend",
            "src/frontend/services",
            "src/scripts",
            "models",
            "models/trained",
            "models/checkpoints",
            "outputs",
            "outputs/generated",
            "outputs/logs",
            "outputs/exports",
            "docs",
            "tests",
            "tests/api",
            "tests/core",
            "legacy"
        ]
        
        for dir_path in new_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                if not self.dry_run:
                    full_path.mkdir(parents=True, exist_ok=True)
                self.log(f"Created directory: {dir_path}")
            else:
                self.log(f"Directory already exists: {dir_path}")
    
    def backup_existing_files(self):
        """Backup existing flat structure files"""
        
        self.log("Creating backup of existing files...")
        
        if not self.backup_dir.exists() and not self.dry_run:
            self.backup_dir.mkdir()
        
        # Files to backup (flat structure files)
        backup_files = [
            "start_floormind.py",
            "start_floormind_fixed.py", 
            "start_backend.py",
            "start_frontend.py",
            "test_backend.py",
            "test_integration.py",
            "test_integration_fixed.py",
            "web_interface.py",
            "simple_model_test.py",
            "use_trained_model.py"
        ]
        
        for file_name in backup_files:
            src_file = self.project_root / file_name
            if src_file.exists():
                dst_file = self.backup_dir / file_name
                if not self.dry_run:
                    shutil.copy2(src_file, dst_file)
                self.log(f"Backed up: {file_name}")
    
    def migrate_model_files(self):
        """Migrate model files to new location"""
        
        self.log("Migrating model files...")
        
        # Move google/ to models/trained/ if it exists
        google_dir = self.project_root / "google"
        trained_dir = self.project_root / "models" / "trained"
        
        if google_dir.exists():
            if not trained_dir.exists():
                if not self.dry_run:
                    shutil.move(str(google_dir), str(trained_dir))
                self.log(f"Moved google/ → models/trained/")
            else:
                self.log("models/trained/ already exists, skipping google/ migration")
                self.log("Please manually merge google/ contents if needed")
        else:
            self.log("google/ directory not found, skipping model migration")
    
    def migrate_generated_outputs(self):
        """Migrate generated outputs to new location"""
        
        self.log("Migrating generated outputs...")
        
        # Move generated_floor_plans/ to outputs/generated/
        old_generated = self.project_root / "generated_floor_plans"
        new_generated = self.project_root / "outputs" / "generated"
        
        if old_generated.exists():
            if not new_generated.exists():
                if not self.dry_run:
                    shutil.move(str(old_generated), str(new_generated))
                self.log(f"Moved generated_floor_plans/ → outputs/generated/")
            else:
                self.log("outputs/generated/ already exists, skipping migration")
                # Copy files instead
                if not self.dry_run:
                    for file in old_generated.glob("*"):
                        if file.is_file():
                            shutil.copy2(file, new_generated / file.name)
                self.log("Copied files from generated_floor_plans/ to outputs/generated/")
        else:
            self.log("generated_floor_plans/ not found, skipping output migration")
    
    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        
        self.log("Creating __init__.py files...")
        
        init_dirs = [
            "src",
            "src/core",
            "src/api",
            "src/frontend", 
            "src/scripts",
            "tests",
            "tests/api",
            "tests/core"
        ]
        
        for dir_path in init_dirs:
            init_file = self.project_root / dir_path / "__init__.py"
            if not init_file.exists() and not self.dry_run:
                init_file.write_text("# FloorMind package\n")
                self.log(f"Created __init__.py in {dir_path}")
    
    def update_gitignore(self):
        """Update .gitignore for new structure"""
        
        self.log("Updating .gitignore...")
        
        gitignore_path = self.project_root / ".gitignore"
        
        new_entries = [
            "\n# New structure directories",
            "outputs/generated/*.png",
            "outputs/logs/*.log", 
            "models/trained/",
            "models/checkpoints/",
            "backup_flat_structure/",
            "legacy/",
            "\n# Python cache",
            "src/**/__pycache__/",
            "*.pyc",
            "\n# Logs",
            "*.log"
        ]
        
        if gitignore_path.exists():
            existing_content = gitignore_path.read_text()
            
            # Check if entries already exist
            needs_update = False
            for entry in new_entries:
                if entry.strip() and entry.strip() not in existing_content:
                    needs_update = True
                    break
            
            if needs_update and not self.dry_run:
                with open(gitignore_path, 'a') as f:
                    f.write('\n'.join(new_entries))
                self.log("Updated .gitignore with new structure entries")
            else:
                self.log(".gitignore already up to date")
        else:
            if not self.dry_run:
                gitignore_path.write_text('\n'.join(new_entries))
            self.log("Created new .gitignore")
    
    def create_migration_summary(self):
        """Create a summary of the migration"""
        
        summary_path = self.project_root / "MIGRATION_SUMMARY.md"
        
        summary_content = f"""# FloorMind Migration Summary

## Migration Completed

This project has been migrated from a flat structure to the organized v2.0 structure.

### Changes Made:

1. **Directory Structure:**
   - Created organized src/ directory with core/, api/, scripts/ subdirectories
   - Created models/trained/ for model storage
   - Created outputs/ for generated content and logs

2. **File Migrations:**
   - Model files: google/ → models/trained/
   - Generated outputs: generated_floor_plans/ → outputs/generated/
   - Backup created in: backup_flat_structure/

3. **New Files Created:**
   - src/core/model_manager.py - Centralized model management
   - src/api/app.py - Enhanced Flask application
   - src/api/routes.py - Organized API routes
   - src/scripts/start_complete.py - Complete launcher

### Next Steps:

1. **Test the new structure:**
   ```bash
   python src/scripts/start_complete.py
   ```

2. **Update your workflow:**
   - Use new startup scripts in src/scripts/
   - Place models in models/trained/
   - Generated outputs go to outputs/generated/

3. **Clean up (optional):**
   - Review files in backup_flat_structure/
   - Remove old flat structure files when confident
   - Update any custom scripts to use new paths

### Rollback (if needed):

If you need to rollback:
1. Stop all services
2. Restore files from backup_flat_structure/
3. Remove new src/ and models/ directories
4. Restore original structure

### Documentation:

- See PROJECT_STRUCTURE_V2.md for detailed structure info
- See INTEGRATION_FIX_README.md for integration details

Migration completed on: {self.get_timestamp()}
"""
        
        if not self.dry_run:
            summary_path.write_text(summary_content)
        self.log("Created migration summary: MIGRATION_SUMMARY.md")
    
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def run_migration(self, dry_run=False):
        """Run the complete migration"""
        
        self.dry_run = dry_run
        
        if dry_run:
            self.log("=== DRY RUN MODE - No files will be modified ===", "WARN")
        
        self.log("Starting FloorMind v2.0 migration...")
        self.log("=" * 50)
        
        try:
            # Step 1: Create new structure
            self.create_new_structure()
            
            # Step 2: Backup existing files
            self.backup_existing_files()
            
            # Step 3: Migrate model files
            self.migrate_model_files()
            
            # Step 4: Migrate generated outputs
            self.migrate_generated_outputs()
            
            # Step 5: Create __init__.py files
            self.create_init_files()
            
            # Step 6: Update .gitignore
            self.update_gitignore()
            
            # Step 7: Create migration summary
            if not dry_run:
                self.create_migration_summary()
            
            self.log("=" * 50)
            if dry_run:
                self.log("DRY RUN COMPLETED - Run without --dry-run to apply changes")
            else:
                self.log("MIGRATION COMPLETED SUCCESSFULLY!")
                self.log("Next step: python src/scripts/start_complete.py")
            
        except Exception as e:
            self.log(f"Migration failed: {e}", "ERROR")
            raise

def main():
    """Main migration function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate FloorMind to v2.0 structure")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    migrator = FloorMindMigrator()
    migrator.run_migration(dry_run=args.dry_run)

if __name__ == "__main__":
    main()