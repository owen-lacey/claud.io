#!/usr/bin/env python3
"""
MongoDB Legacy Predictions Cleanup Script

Removes the predictions_legacy fields from MongoDB after successful migration
to the unified predictions system.

This script should only be run after verifying that:
1. The unified predictions system is working correctly
2. All required predictions have been migrated successfully
3. No rollback to legacy predictions will be needed
"""

import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add Data directory to path for imports
DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from database.mongo.fpl_mongo_client import FPLMongoClient


class LegacyPredictionsCleanup:
    """Handles cleanup of legacy prediction fields"""
    
    def __init__(self):
        self.client = FPLMongoClient()
        self.players_collection = self.client.get_collection('players')
        
    def verify_current_predictions(self) -> Dict[str, Any]:
        """Verify current predictions are working before cleanup"""
        print("🔍 Step 1: Verifying current predictions system...")
        
        # Count players with current predictions
        current_count = len(list(self.players_collection.aggregate([
            {'$match': {'predictions': {'$exists': True, '$ne': None}}},
            {'$project': {'web_name': 1, 'predictions': 1}},
            {'$match': {'predictions.1': {'$exists': True}}}
        ])))
        
        # Check if Gyökeres is still there (our canary)
        gyokeres = self.players_collection.find_one({'web_name': 'Gyökeres'})
        gyokeres_has_predictions = bool(gyokeres and gyokeres.get('predictions'))
        
        # Count legacy predictions to be removed
        legacy_count = len(list(self.players_collection.find({'predictions_legacy': {'$exists': True}})))
        
        results = {
            'current_predictions_count': current_count,
            'gyokeres_has_predictions': gyokeres_has_predictions,
            'legacy_predictions_count': legacy_count
        }
        
        print(f"   📊 Current predictions: {current_count} players")
        print(f"   🎯 Gyökeres has predictions: {'✅' if gyokeres_has_predictions else '❌'}")
        print(f"   🗑️  Legacy predictions to remove: {legacy_count} players")
        
        # Safety checks
        if current_count < 400:
            print("   ⚠️  WARNING: Low number of current predictions")
        if not gyokeres_has_predictions:
            print("   ⚠️  WARNING: Gyökeres missing from current predictions")
        
        return results
    
    def remove_legacy_predictions(self, dry_run: bool = False) -> Dict[str, int]:
        """Remove legacy prediction fields"""
        print(f"🗑️  Step 2: {'[DRY RUN] ' if dry_run else ''}Removing legacy predictions...")
        
        # Find all players with legacy predictions
        players_with_legacy = list(self.players_collection.find(
            {'predictions_legacy': {'$exists': True}},
            {'code': 1, 'web_name': 1}
        ))
        
        if not players_with_legacy:
            print("   ✅ No legacy predictions found to remove")
            return {'removed': 0}
        
        print(f"   📋 Found {len(players_with_legacy)} players with legacy predictions")
        
        if not dry_run:
            # Remove legacy predictions using bulk write
            from pymongo.operations import UpdateMany
            
            result = self.players_collection.bulk_write([
                UpdateMany(
                    {'predictions_legacy': {'$exists': True}},
                    {'$unset': {'predictions_legacy': 1}}
                )
            ])
            
            print(f"   ✅ Removed legacy predictions from {result.modified_count} players")
            return {'removed': result.modified_count}
        else:
            print(f"   🔍 Would remove legacy predictions from {len(players_with_legacy)} players")
            return {'would_remove': len(players_with_legacy)}
    
    def verify_cleanup(self) -> Dict[str, Any]:
        """Verify cleanup was successful"""
        print("🔍 Step 3: Verifying cleanup...")
        
        # Check that legacy predictions are gone
        legacy_remaining = self.players_collection.count_documents({'predictions_legacy': {'$exists': True}})
        
        # Check that current predictions are still there
        current_count = len(list(self.players_collection.aggregate([
            {'$match': {'predictions': {'$exists': True, '$ne': None}}},
            {'$project': {'web_name': 1, 'predictions': 1}},
            {'$match': {'predictions.1': {'$exists': True}}}
        ])))
        
        # Check Gyökeres is still there
        gyokeres = self.players_collection.find_one({'web_name': 'Gyökeres'})
        gyokeres_ok = bool(gyokeres and gyokeres.get('predictions'))
        
        results = {
            'legacy_remaining': legacy_remaining,
            'current_predictions_count': current_count,
            'gyokeres_ok': gyokeres_ok,
            'cleanup_successful': legacy_remaining == 0 and current_count > 400 and gyokeres_ok
        }
        
        print(f"   📊 Legacy predictions remaining: {legacy_remaining}")
        print(f"   📊 Current predictions: {current_count}")
        print(f"   🎯 Gyökeres still has predictions: {'✅' if gyokeres_ok else '❌'}")
        
        if results['cleanup_successful']:
            print("   ✅ Cleanup successful!")
        else:
            print("   ❌ Cleanup may have issues")
            
        return results
    
    def run_cleanup(self, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
        """Run the complete cleanup process"""
        print("🧹 MONGODB LEGACY PREDICTIONS CLEANUP")
        print("=" * 50)
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
            print("=" * 50)
        
        try:
            results = {}
            
            # Step 1: Verify current system is working
            verification_result = self.verify_current_predictions()
            results.update(verification_result)
            
            # Safety check
            if not force and (verification_result['current_predictions_count'] < 400 or not verification_result['gyokeres_has_predictions']):
                print("\n❌ SAFETY CHECK FAILED")
                print("Current predictions system appears to have issues.")
                print("Use --force flag to proceed anyway, or fix the issues first.")
                return results
            
            # Step 2: Remove legacy predictions
            cleanup_result = self.remove_legacy_predictions(dry_run=dry_run)
            results.update(cleanup_result)
            
            # Step 3: Verify cleanup (only if not dry run)
            if not dry_run:
                verification_result = self.verify_cleanup()
                results.update(verification_result)
            
            return results
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            raise
        finally:
            self.client.disconnect()


def main():
    """Main entry point for the cleanup script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove legacy predictions from MongoDB')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without making changes')
    parser.add_argument('--force', action='store_true', help='Force cleanup even if safety checks fail')
    args = parser.parse_args()
    
    cleanup = LegacyPredictionsCleanup()
    results = cleanup.run_cleanup(dry_run=args.dry_run, force=args.force)
    
    print(f"\n✅ Cleanup completed: {results}")


if __name__ == '__main__':
    main()
