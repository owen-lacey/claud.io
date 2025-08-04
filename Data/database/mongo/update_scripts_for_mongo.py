#!/usr/bin/env python3
"""
Update Python Scripts for MongoDB
=================================

Updates existing Python scripts to use MongoDB instead of JSON files.
This script creates backup files and updates imports and function calls.

Usage:
    python update_scripts_for_mongo.py [--dry-run] [--backup]
    
Options:
    --dry-run: Show what would be changed without making changes
    --backup: Create backup files before making changes

Author: FPL Team Picker
Version: 1.0.0
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PythonScriptUpdater:
    """Updates Python scripts to use MongoDB compatibility layer"""
    
    def __init__(self, root_dir: str, dry_run: bool = False, backup: bool = True):
        """
        Initialize the script updater
        
        Args:
            root_dir: Root directory to search for Python files
            dry_run: If True, show changes without applying them
            backup: If True, create backup files before changes
        """
        self.root_dir = Path(root_dir)
        self.dry_run = dry_run
        self.backup = backup
        
        # Patterns to find and replace
        self.patterns = [
            # JSON file loading patterns
            {
                'pattern': r'with open\([\'\"](.*?/database/players\.json)[\'\"]\s*,\s*[\'\"r\'\"]\s*\)\s*as\s*f:\s*\n\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*json\.load\(f\)',
                'replacement': r'# MongoDB: \2 = load_players_data()\nfrom Data.database.mongo.mongo_data_loader import load_players_data\n\2 = load_players_data()',
                'description': 'Replace players.json loading with MongoDB loader'
            },
            {
                'pattern': r'with open\([\'\"](.*?/database/teams\.json)[\'\"]\s*,\s*[\'\"r\'\"]\s*\)\s*as\s*f:\s*\n\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*json\.load\(f\)',
                'replacement': r'# MongoDB: \2 = load_teams_data()\nfrom Data.database.mongo.mongo_data_loader import load_teams_data\n\2 = load_teams_data()',
                'description': 'Replace teams.json loading with MongoDB loader'
            },
            {
                'pattern': r'with open\([\'\"](.*?/database/fixtures\.json)[\'\"]\s*,\s*[\'\"r\'\"]\s*\)\s*as\s*f:\s*\n\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*json\.load\(f\)',
                'replacement': r'# MongoDB: \2 = load_fixtures_data()\nfrom Data.database.mongo.mongo_data_loader import load_fixtures_data\n\2 = load_fixtures_data()',
                'description': 'Replace fixtures.json loading with MongoDB loader'
            },
            
            # Function calls that load data
            {
                'pattern': r'load_teams_data\([\'\"](.*?teams\.json)[\'\"]\)',
                'replacement': r'load_teams_data()',
                'description': 'Update load_teams_data() calls to use MongoDB'
            },
            
            # Specific existing patterns found in the codebase
            {
                'pattern': r'teams_data = load_teams_data\([\'\"](.*?teams\.json)[\'\"]\)',
                'replacement': r'from Data.database.mongo.mongo_data_loader import load_teams_data\nteams_data = load_teams_data()',
                'description': 'Update existing load_teams_data calls'
            }
        ]
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files that might need updating"""
        python_files = []
        
        # Search for Python files
        for pattern in ['**/*.py']:
            python_files.extend(self.root_dir.glob(pattern))
        
        # Filter out files we don't want to modify
        exclude_patterns = [
            'mongo',  # Our MongoDB files
            '__pycache__',
            '.git',
            'venv',
            'env'
        ]
        
        filtered_files = []
        for file_path in python_files:
            if not any(pattern in str(file_path) for pattern in exclude_patterns):
                filtered_files.append(file_path)
        
        logger.info(f"🔍 Found {len(filtered_files)} Python files to check")
        return filtered_files
    
    def analyze_file(self, file_path: Path) -> List[Dict]:
        """Analyze a file to find patterns that need updating"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"⚠️ Could not read {file_path}: {e}")
            return []
        
        matches = []
        
        # Check for JSON file references
        json_refs = [
            'players.json',
            'teams.json', 
            'fixtures.json'
        ]
        
        for json_ref in json_refs:
            if json_ref in content:
                matches.append({
                    'type': 'json_reference',
                    'reference': json_ref,
                    'file': file_path
                })
        
        # Check for specific patterns
        for pattern_info in self.patterns:
            pattern_matches = re.finditer(pattern_info['pattern'], content, re.MULTILINE | re.DOTALL)
            for match in pattern_matches:
                matches.append({
                    'type': 'pattern_match',
                    'pattern': pattern_info,
                    'match': match,
                    'file': file_path,
                    'text': match.group(0)
                })
        
        return matches
    
    def update_file(self, file_path: Path, matches: List[Dict]) -> bool:
        """Update a file based on found matches"""
        if not matches:
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"❌ Could not read {file_path}: {e}")
            return False
        
        original_content = content
        
        # Apply pattern replacements
        pattern_matches = [m for m in matches if m['type'] == 'pattern_match']
        for match_info in pattern_matches:
            pattern_info = match_info['pattern']
            content = re.sub(
                pattern_info['pattern'], 
                pattern_info['replacement'], 
                content, 
                flags=re.MULTILINE | re.DOTALL
            )
        
        # Check if content actually changed
        if content == original_content:
            return False
        
        if self.dry_run:
            logger.info(f"🔍 Would update {file_path}")
            logger.info(f"   Changes: {len(pattern_matches)} pattern replacements")
            return True
        
        # Create backup if requested
        if self.backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            shutil.copy2(file_path, backup_path)
            logger.debug(f"📁 Created backup: {backup_path}")
        
        # Write updated content
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✅ Updated {file_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Could not write {file_path}: {e}")
            return False
    
    def add_mongo_import(self, file_path: Path) -> bool:
        """Add MongoDB import to a file if it needs it"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"⚠️ Could not read {file_path}: {e}")
            return False
        
        # Check if file uses any of our data loading functions
        needs_import = any([
            'load_players_data(' in content,
            'load_teams_data(' in content,
            'load_fixtures_data(' in content
        ])
        
        if not needs_import:
            return False
        
        # Check if import already exists
        if 'from Data.database.mongo.mongo_data_loader import' in content:
            return False
        
        if self.dry_run:
            logger.info(f"🔍 Would add MongoDB import to {file_path}")
            return True
        
        # Add import at the top after existing imports
        lines = content.split('\n')
        import_line = 'from Data.database.mongo.mongo_data_loader import load_players_data, load_teams_data, load_fixtures_data'
        
        # Find the best place to insert the import
        insert_position = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_position = i + 1
            elif line.strip() == '':
                continue
            else:
                break
        
        lines.insert(insert_position, import_line)
        content = '\n'.join(lines)
        
        # Create backup if requested
        if self.backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            shutil.copy2(file_path, backup_path)
        
        # Write updated content
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✅ Added MongoDB import to {file_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Could not write {file_path}: {e}")
            return False
    
    def process_all_files(self) -> Dict[str, int]:
        """Process all Python files"""
        python_files = self.find_python_files()
        
        stats = {
            'total_files': len(python_files),
            'files_with_matches': 0,
            'files_updated': 0,
            'total_matches': 0
        }
        
        for file_path in python_files:
            matches = self.analyze_file(file_path)
            
            if matches:
                stats['files_with_matches'] += 1
                stats['total_matches'] += len(matches)
                
                logger.info(f"📄 {file_path}")
                for match in matches:
                    if match['type'] == 'json_reference':
                        logger.info(f"   📎 References: {match['reference']}")
                    elif match['type'] == 'pattern_match':
                        logger.info(f"   🔧 Pattern: {match['pattern']['description']}")
                
                # Update the file
                if self.update_file(file_path, matches):
                    stats['files_updated'] += 1
        
        return stats
    
    def run(self) -> None:
        """Run the script updater"""
        logger.info("🚀 Starting Python script updates for MongoDB...")
        
        if self.dry_run:
            logger.info("🧪 DRY RUN MODE - No changes will be made")
        
        stats = self.process_all_files()
        
        # Print summary
        logger.info("📊 Update Summary:")
        logger.info(f"   Total files checked: {stats['total_files']}")
        logger.info(f"   Files with matches: {stats['files_with_matches']}")
        logger.info(f"   Files updated: {stats['files_updated']}")
        logger.info(f"   Total matches: {stats['total_matches']}")
        
        if stats['files_updated'] > 0:
            logger.info("🎉 Script update completed successfully!")
            logger.info("💡 Next steps:")
            logger.info("   1. Install MongoDB: brew install mongodb-community")
            logger.info("   2. Start MongoDB: brew services start mongodb-community")
            logger.info("   3. Install Python packages: pip install -r requirements-mongo.txt")
            logger.info("   4. Run migration: python Data/database/mongo/migrate_json_to_mongo.py")
        else:
            logger.info("ℹ️ No files needed updating")


def main():
    """Main script entry point"""
    parser = argparse.ArgumentParser(description='Update Python scripts to use MongoDB')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without making changes')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup files before making changes')
    parser.add_argument('--root-dir', default='.',
                       help='Root directory to search for Python files')
    
    args = parser.parse_args()
    
    updater = PythonScriptUpdater(
        root_dir=args.root_dir,
        dry_run=args.dry_run,
        backup=args.backup
    )
    
    try:
        updater.run()
    except Exception as e:
        logger.error(f"❌ Script update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
