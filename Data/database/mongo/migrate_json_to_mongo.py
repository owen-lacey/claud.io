#!/usr/bin/env python3
"""
JSON to MongoDB Migration Script
===============================

Migrates FPL data from JSON files to MongoDB collections.
This script handles the initial data migration and can be run multiple times safely.

Usage:
    python migrate_json_to_mongo.py [--reset]
    
Options:
    --reset: Drop existing collections before migration (use with caution!)

Author: FPL Team Picker
Version: 1.0.0
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add the mongo directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from fpl_mongo_client import FPLMongoClient
except ImportError as e:
    print(f"❌ Error importing MongoDB client: {e}")
    print("💡 Please install required packages: pip install -r requirements-mongo.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JSONToMongoMigrator:
    """Handles migration of JSON data to MongoDB"""
    
    def __init__(self, json_dir: str = "../"):
        """
        Initialize the migrator
        
        Args:
            json_dir: Directory containing JSON files (relative to this script)
        """
        self.json_dir = Path(__file__).parent / json_dir
        self.mongo_client = FPLMongoClient()
        
        # JSON file paths
        self.json_files = {
            'players': self.json_dir / 'players.json',
            'teams': self.json_dir / 'teams.json', 
            'fixtures': self.json_dir / 'fixtures.json'
        }
        
        logger.info(f"📁 JSON directory: {self.json_dir.absolute()}")
    
    def validate_json_files(self) -> bool:
        """Validate that all required JSON files exist"""
        logger.info("🔍 Validating JSON files...")
        
        all_exist = True
        for name, path in self.json_files.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                logger.info(f"   ✅ {name}.json found ({size_mb:.2f} MB)")
            else:
                logger.error(f"   ❌ {name}.json not found at {path}")
                all_exist = False
        
        return all_exist
    
    def load_json_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data)}")
            
            logger.info(f"📊 Loaded {len(data)} records from {file_path.name}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            raise
    
    def migrate_players(self, reset: bool = False) -> None:
        """Migrate players data"""
        logger.info("👥 Migrating players...")
        
        if reset:
            collection = self.mongo_client.get_collection(self.mongo_client.players_collection_name)
            collection.drop()
            logger.info("🗑️ Dropped existing players collection")
        
        # Load and validate players data
        players_data = self.load_json_data(self.json_files['players'])
        
        # Validate data structure
        required_fields = ['id', 'web_name', 'element_type', 'team']
        sample_player = players_data[0] if players_data else {}
        
        for field in required_fields:
            if field not in sample_player:
                raise ValueError(f"Required field '{field}' missing from players data")
        
        # Convert numeric fields to proper types
        for player in players_data:
            # Ensure numeric fields are numbers, not strings
            numeric_fields = ['id', 'code', 'element_type', 'team', 'now_cost', 'total_points', 
                            'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
                            'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 
                            'red_cards', 'saves', 'bonus', 'bps']
            
            for field in numeric_fields:
                if field in player and isinstance(player[field], str):
                    try:
                        player[field] = int(player[field])
                    except (ValueError, TypeError):
                        player[field] = 0
            
            # Convert float fields
            float_fields = ['expected_goals', 'expected_assists', 'expected_goal_involvements', 
                          'expected_goals_conceded', 'form', 'points_per_game', 'selected_by_percent']
            
            for field in float_fields:
                if field in player and isinstance(player[field], str):
                    try:
                        player[field] = float(player[field])
                    except (ValueError, TypeError):
                        player[field] = 0.0
        
        # Bulk insert players
        self.mongo_client.bulk_upsert_players(players_data)
        logger.info(f"✅ Migrated {len(players_data)} players")
    
    def migrate_teams(self, reset: bool = False) -> None:
        """Migrate teams data"""
        logger.info("⚽ Migrating teams...")
        
        if reset:
            collection = self.mongo_client.get_collection(self.mongo_client.teams_collection_name)
            collection.drop()
            logger.info("🗑️ Dropped existing teams collection")
        
        # Load and validate teams data
        teams_data = self.load_json_data(self.json_files['teams'])
        
        # Validate data structure
        required_fields = ['id', 'name', 'short_name']
        sample_team = teams_data[0] if teams_data else {}
        
        for field in required_fields:
            if field not in sample_team:
                raise ValueError(f"Required field '{field}' missing from teams data")
        
        # Convert numeric fields to proper types
        for team in teams_data:
            numeric_fields = ['id', 'code', 'strength', 'strength_overall_home', 'strength_overall_away',
                            'strength_attack_home', 'strength_attack_away', 'strength_defence_home',
                            'strength_defence_away', 'played', 'win', 'draw', 'loss', 'points', 'position']
            
            for field in numeric_fields:
                if field in team and isinstance(team[field], str):
                    try:
                        team[field] = int(team[field])
                    except (ValueError, TypeError):
                        team[field] = 0
        
        # Bulk insert teams
        self.mongo_client.bulk_upsert_teams(teams_data)
        logger.info(f"✅ Migrated {len(teams_data)} teams")
    
    def migrate_fixtures(self, reset: bool = False) -> None:
        """Migrate fixtures data"""
        logger.info("🏟️ Migrating fixtures...")
        
        if reset:
            collection = self.mongo_client.get_collection(self.mongo_client.fixtures_collection_name)
            collection.drop()
            logger.info("🗑️ Dropped existing fixtures collection")
        
        # Load and validate fixtures data
        fixtures_data = self.load_json_data(self.json_files['fixtures'])
        
        # Validate data structure
        required_fields = ['id', 'gameweek', 'team_h', 'team_a']
        sample_fixture = fixtures_data[0] if fixtures_data else {}
        
        for field in required_fields:
            if field not in sample_fixture:
                raise ValueError(f"Required field '{field}' missing from fixtures data")
        
        # Convert numeric fields to proper types
        for fixture in fixtures_data:
            numeric_fields = ['id', 'gameweek', 'team_h', 'team_a', 'team_h_difficulty', 
                            'team_a_difficulty', 'team_h_score', 'team_a_score']
            
            for field in numeric_fields:
                if field in fixture and fixture[field] is not None:
                    if isinstance(fixture[field], str):
                        try:
                            fixture[field] = int(fixture[field])
                        except (ValueError, TypeError):
                            fixture[field] = None
        
        # Bulk insert fixtures
        self.mongo_client.bulk_upsert_fixtures(fixtures_data)
        logger.info(f"✅ Migrated {len(fixtures_data)} fixtures")
    
    def migrate_all(self, reset: bool = False) -> None:
        """Migrate all data from JSON to MongoDB"""
        logger.info("🚀 Starting complete migration...")
        
        try:
            # Validate JSON files exist
            if not self.validate_json_files():
                raise RuntimeError("❌ JSON file validation failed")
            
            # Create indexes for better performance
            self.mongo_client.create_indexes()
            
            # Migrate each collection
            self.migrate_teams(reset)
            self.migrate_players(reset)
            self.migrate_fixtures(reset)
            
            # Display final statistics
            stats = self.mongo_client.get_database_stats()
            logger.info("📊 Migration Complete! Database Statistics:")
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise
        finally:
            self.mongo_client.disconnect()
    
    def verify_migration(self) -> None:
        """Verify the migration was successful"""
        logger.info("🔍 Verifying migration...")
        
        # Create a new client connection for verification
        verification_client = FPLMongoClient()
        
        try:
            # Test data retrieval
            players = verification_client.get_all_players()
            teams = verification_client.get_all_teams()
            fixtures = verification_client.get_all_fixtures()
            
            logger.info(f"✅ Players in DB: {len(players)}")
            logger.info(f"✅ Teams in DB: {len(teams)}")
            logger.info(f"✅ Fixtures in DB: {len(fixtures)}")
            
            # Test specific queries
            gk_players = verification_client.get_players_by_position(1)  # Goalkeepers
            arsenal_players = verification_client.get_players_by_team(1)  # Arsenal (assuming team_id=1)
            gw1_fixtures = verification_client.get_fixtures_by_gameweek(1)
            
            logger.info(f"✅ Goalkeepers: {len(gk_players)}")
            logger.info(f"✅ Arsenal players: {len(arsenal_players)}")
            logger.info(f"✅ GW1 fixtures: {len(gw1_fixtures)}")
            
            logger.info("🎉 Migration verification successful!")
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            raise
        finally:
            verification_client.disconnect()


def main():
    """Main migration script"""
    parser = argparse.ArgumentParser(description='Migrate FPL JSON data to MongoDB')
    parser.add_argument('--reset', action='store_true', 
                       help='Drop existing collections before migration')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing data, skip migration')
    
    args = parser.parse_args()
    
    # Create migrator
    migrator = JSONToMongoMigrator()
    
    try:
        if args.verify_only:
            migrator.verify_migration()
        else:
            # Run migration
            migrator.migrate_all(reset=args.reset)
            
            # Verify migration
            migrator.verify_migration()
            
        logger.info("🎉 All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
