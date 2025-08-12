#!/usr/bin/env python3
"""
MongoDB Predictions Migration Script

Migrates from the old dual prediction system:
- predictions (old FPL-based predictions)  
- predictions_fbref (new FBRef-based predictions)

To a single prediction system:
- predictions_legacy (backup of old predictions)
- predictions (migrated from predictions_fbref, now the primary)

This ensures all pipelines use the same 'predictions' field going forward.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add Data directory to path for imports
DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from database.mongo.fpl_mongo_client import FPLMongoClient


class PredictionsMigration:
    """Handles the migration from dual predictions to single predictions system"""
    
    def __init__(self):
        self.client = FPLMongoClient()
        self.players_collection = self.client.get_collection('players')
        
    def backup_old_predictions(self) -> Dict[str, int]:
        """Backup old predictions to predictions_legacy field"""
        print("🔄 Step 1: Backing up old predictions to predictions_legacy...")
        
        # Find all players with old predictions
        players_with_old = list(self.players_collection.find(
            {'predictions': {'$exists': True}},
            {'code': 1, 'web_name': 1, 'predictions': 1}
        ))
        
        if not players_with_old:
            print("   ✅ No old predictions found to backup")
            return {'backed_up': 0}
        
        # Backup each player's old predictions
        operations = []
        for player in players_with_old:
            operations.append({
                'filter': {'code': player['code']},
                'update': {
                    '$set': {'predictions_legacy': player['predictions']},
                    '$unset': {'predictions': 1}
                }
            })
        
        # Bulk write the backup operations
        if operations:
            from pymongo.operations import UpdateOne
            bulk_operations = []
            for op in operations:
                bulk_operations.append(UpdateOne(
                    op['filter'],
                    {'$set': op['update']['$set'], '$unset': op['update']['$unset']}
                ))
            self.players_collection.bulk_write(bulk_operations)
        
        print(f"   ✅ Backed up predictions for {len(players_with_old)} players")
        return {'backed_up': len(players_with_old)}
    
    def migrate_fbref_to_predictions(self) -> Dict[str, int]:
        """Migrate FBRef predictions to the main predictions field"""
        print("🔄 Step 2: Migrating FBRef predictions to main predictions field...")
        
        # Find all players with FBRef predictions
        players_with_fbref = list(self.players_collection.find(
            {'predictions_fbref': {'$exists': True}},
            {'code': 1, 'web_name': 1, 'predictions_fbref': 1}
        ))
        
        if not players_with_fbref:
            print("   ✅ No FBRef predictions found to migrate")
            return {'migrated': 0}
        
        # Migrate each player's FBRef predictions
        operations = []
        for player in players_with_fbref:
            operations.append({
                'filter': {'code': player['code']},
                'update': {
                    '$set': {'predictions': player['predictions_fbref']},
                    '$unset': {'predictions_fbref': 1}
                }
            })
        
        # Bulk write the migration operations
        if operations:
            from pymongo.operations import UpdateOne
            bulk_operations = []
            for op in operations:
                bulk_operations.append(UpdateOne(
                    op['filter'],
                    {'$set': op['update']['$set'], '$unset': op['update']['$unset']}
                ))
            self.players_collection.bulk_write(bulk_operations)
        
        print(f"   ✅ Migrated FBRef predictions for {len(players_with_fbref)} players")
        return {'migrated': len(players_with_fbref)}
    
    def update_field_prefix_in_client(self):
        """Update the FPL client to use 'predictions' instead of 'predictions_fbref'"""
        print("🔄 Step 3: Updating MongoDB client to use unified predictions field...")
        
        # This will be done by modifying the fpl_mongo_client.py file
        # The field_prefix logic should default to 'predictions' for all sources
        print("   ✅ Client update instructions provided (manual step required)")
        
    def verify_migration(self) -> Dict[str, Any]:
        """Verify the migration was successful"""
        print("🔍 Step 4: Verifying migration...")
        
        # Count different prediction types
        legacy_count = self.players_collection.count_documents({'predictions_legacy': {'$exists': True}})
        current_count = self.players_collection.count_documents({'predictions': {'$exists': True}})
        fbref_remaining = self.players_collection.count_documents({'predictions_fbref': {'$exists': True}})
        
        # Get sample data
        sample = self.players_collection.find_one(
            {'predictions': {'$exists': True}},
            {'code': 1, 'web_name': 1, 'predictions': 1, 'predictions_legacy': 1}
        )
        
        results = {
            'legacy_predictions_count': legacy_count,
            'current_predictions_count': current_count,
            'fbref_predictions_remaining': fbref_remaining,
            'sample_player': sample.get('web_name') if sample else None,
            'migration_successful': fbref_remaining == 0 and current_count > 0
        }
        
        print(f"   📊 Legacy predictions (backed up): {legacy_count}")
        print(f"   📊 Current predictions (active): {current_count}")
        print(f"   📊 FBRef predictions remaining: {fbref_remaining}")
        print(f"   🎯 Sample migrated player: {results['sample_player']}")
        
        if results['migration_successful']:
            print("   ✅ Migration successful!")
        else:
            print("   ❌ Migration may have issues - check counts")
            
        return results
    
    def run_migration(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run the complete migration process"""
        print("🚀 MONGODB PREDICTIONS MIGRATION")
        print("=" * 50)
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
            print("=" * 50)
        
        try:
            results = {}
            
            if not dry_run:
                # Step 1: Backup old predictions
                backup_result = self.backup_old_predictions()
                results.update(backup_result)
                
                # Step 2: Migrate FBRef predictions  
                migration_result = self.migrate_fbref_to_predictions()
                results.update(migration_result)
                
                # Step 3: Verify migration
                verification_result = self.verify_migration()
                results.update(verification_result)
            else:
                # Dry run - just show what would happen
                old_count = self.players_collection.count_documents({'predictions': {'$exists': True}})
                fbref_count = self.players_collection.count_documents({'predictions_fbref': {'$exists': True}})
                
                print(f"   📊 Would backup {old_count} old predictions")
                print(f"   📊 Would migrate {fbref_count} FBRef predictions")
                
                results = {
                    'dry_run': True,
                    'old_predictions_to_backup': old_count,
                    'fbref_predictions_to_migrate': fbref_count
                }
            
            # Step 4: Show next steps
            self.show_next_steps()
            
            return results
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            raise
        finally:
            self.client.disconnect()
    
    def show_next_steps(self):
        """Show manual steps required after migration"""
        print("\n🔧 MANUAL STEPS REQUIRED:")
        print("=" * 50)
        print("1. Update fpl_mongo_client.py:")
        print("   - Change field_prefix logic to always use 'predictions'")
        print("   - Remove 'predictions_fbref' branching logic")
        print()
        print("2. Update generate_predictions.py:")
        print("   - Remove source='fbref' parameter from bulk_upsert_players()")
        print("   - Ensure all predictions save to unified 'predictions' field")
        print()
        print("3. Test the updated pipeline:")
        print("   - Run generate_predictions.py 1")
        print("   - Verify predictions save to 'predictions' field")


def main():
    """Main entry point for the migration script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate MongoDB predictions to unified structure')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated without making changes')
    parser.add_argument('--backup-only', action='store_true', help='Only backup old predictions, do not migrate')
    args = parser.parse_args()
    
    migration = PredictionsMigration()
    
    if args.backup_only:
        print("🔄 BACKUP ONLY MODE")
        print("=" * 50)
        migration.backup_old_predictions()
        migration.client.disconnect()
    else:
        results = migration.run_migration(dry_run=args.dry_run)
        print(f"\n✅ Migration completed: {results}")


if __name__ == '__main__':
    main()
