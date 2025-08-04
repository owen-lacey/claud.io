"""
MongoDB Database Client for FPL Team Picker
==========================================

Provides connection and CRUD operations for MongoDB collections.
Replaces JSON file-based data storage with proper database functionality.

Collections:
- players: FPL player data with stats and attributes
- teams: Team information with strength ratings
- fixtures: Match fixtures with results and difficulties

Author: FPL Team Picker
Version: 1.0.0
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.operations import ReplaceOne, UpdateOne
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FPLMongoClient:
    """MongoDB client for FPL Team Picker data management"""
    
    def __init__(self, connection_string: Optional[str] = None, database_name: Optional[str] = None):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB URI (defaults to env MONGODB_URI)
            database_name: Database name (defaults to env MONGODB_DATABASE)
        """
        self.connection_string = connection_string or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/fpl_team_picker')
        self.database_name = database_name or os.getenv('MONGODB_DATABASE', 'fpl_team_picker')
        
        # Collection names
        self.players_collection_name = os.getenv('MONGODB_PLAYERS_COLLECTION', 'players')
        self.teams_collection_name = os.getenv('MONGODB_TEAMS_COLLECTION', 'teams')
        self.fixtures_collection_name = os.getenv('MONGODB_FIXTURES_COLLECTION', 'fixtures')
        
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        
        self.connect()
    
    def connect(self) -> None:
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string)
            self.database = self.client[self.database_name]
            
            # Test connection
            self.client.admin.command('ismaster')
            logger.info(f"✅ Connected to MongoDB: {self.database_name}")
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a MongoDB collection"""
        if self.database is None:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]
    
    # === PLAYERS OPERATIONS ===
    
    def get_all_players(self) -> List[Dict[str, Any]]:
        """Get all players from the database"""
        collection = self.get_collection(self.players_collection_name)
        players = list(collection.find({}, {'_id': 0}).sort('id', 1))
        logger.info(f"📊 Retrieved {len(players)} players")
        return players
    
    def get_player_by_id(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific player by current season ID"""
        collection = self.get_collection(self.players_collection_name)
        player = collection.find_one({'id': player_id}, {'_id': 0})
        return player
    
    def get_player_by_code(self, player_code: int) -> Optional[Dict[str, Any]]:
        """Get a specific player by their unique code (used for historical matching)"""
        collection = self.get_collection(self.players_collection_name)
        player = collection.find_one({'code': player_code}, {'_id': 0})
        return player
    
    def get_players_by_team(self, team_id: int) -> List[Dict[str, Any]]:
        """Get all players from a specific team"""
        collection = self.get_collection(self.players_collection_name)
        players = list(collection.find({'team': team_id}, {'_id': 0}).sort('id', 1))
        return players
    
    def get_players_by_position(self, position_type: int) -> List[Dict[str, Any]]:
        """Get all players by position type (1=GK, 2=DEF, 3=MID, 4=FWD)"""
        collection = self.get_collection(self.players_collection_name)
        players = list(collection.find({'element_type': position_type}, {'_id': 0}).sort('id', 1))
        return players
    
    def upsert_player(self, player_data: Dict[str, Any]) -> None:
        """Insert or update a player using 'code' field for matching"""
        collection = self.get_collection(self.players_collection_name)
        collection.replace_one(
            {'code': player_data['code']}, 
            player_data, 
            upsert=True
        )
    
    def bulk_upsert_players(self, players_data: List[Dict[str, Any]]) -> None:
        """Bulk insert or update players using 'code' field for matching"""
        if not players_data:
            return
            
        collection = self.get_collection(self.players_collection_name)
        operations = []
        
        for player in players_data:
            # Use 'code' field for matching as it's consistent across seasons
            # and matches with historical data player_code field
            
            # Check if this is prediction data (has expected_points) or base player data (has element_type)
            if 'expected_points' in player and 'element_type' not in player:
                # This is prediction data - store in nested predictions object by gameweek
                gameweek = player.get('gameweek', 1)  # Default to GW1 if not specified
                
                # Build the prediction data for this gameweek
                prediction_data = {}
                for k, v in player.items():
                    if k not in ['code', 'player_id', 'name', 'gameweek']:
                        prediction_data[k] = v
                
                # Store prediction data nested by gameweek: predictions.{gw}.{field}
                update_fields = {f'predictions.{gameweek}': prediction_data}
                
                operations.append(
                    UpdateOne(
                        {'code': int(player['code'])},
                        {'$set': update_fields},
                        upsert=False  # Don't create new players from predictions
                    )
                )
            else:
                # This is base player data - replace entire document
                operations.append(
                    ReplaceOne(
                        {'code': player['code']},
                        player,
                        upsert=True
                    )
                )
        
        if operations:
            result = collection.bulk_write(operations)
            logger.info(f"✅ Bulk upserted {result.upserted_count + result.modified_count} players")
    
    # === TEAMS OPERATIONS ===
    
    def get_all_teams(self) -> List[Dict[str, Any]]:
        """Get all teams from the database"""
        collection = self.get_collection(self.teams_collection_name)
        teams = list(collection.find({}, {'_id': 0}).sort('id', 1))
        logger.info(f"⚽ Retrieved {len(teams)} teams")
        return teams
    
    def get_team_by_id(self, team_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific team by ID"""
        collection = self.get_collection(self.teams_collection_name)
        team = collection.find_one({'id': team_id}, {'_id': 0})
        return team
    
    def get_teams_lookup(self) -> Dict[int, Dict[str, Any]]:
        """Get teams as a lookup dictionary by ID"""
        teams = self.get_all_teams()
        return {team['id']: team for team in teams}
    
    def upsert_team(self, team_data: Dict[str, Any]) -> None:
        """Insert or update a team"""
        collection = self.get_collection(self.teams_collection_name)
        collection.replace_one(
            {'id': team_data['id']}, 
            team_data, 
            upsert=True
        )
    
    def bulk_upsert_teams(self, teams_data: List[Dict[str, Any]]) -> None:
        """Bulk insert or update teams"""
        if not teams_data:
            return
            
        collection = self.get_collection(self.teams_collection_name)
        operations = []
        
        for team in teams_data:
            operations.append(
                ReplaceOne(
                    {'id': team['id']},
                    team,
                    upsert=True
                )
            )
        
        if operations:
            result = collection.bulk_write(operations)
            logger.info(f"✅ Bulk upserted {result.upserted_count + result.modified_count} teams")
    
    # === FIXTURES OPERATIONS ===
    
    def get_all_fixtures(self) -> List[Dict[str, Any]]:
        """Get all fixtures from the database"""
        collection = self.get_collection(self.fixtures_collection_name)
        fixtures = list(collection.find({}, {'_id': 0}).sort([('gameweek', 1), ('id', 1)]))
        logger.info(f"🏟️ Retrieved {len(fixtures)} fixtures")
        return fixtures
    
    def get_fixtures_by_gameweek(self, gameweek: int, season: str = "2025-26") -> List[Dict[str, Any]]:
        """Get fixtures for a specific gameweek"""
        collection = self.get_collection(self.fixtures_collection_name)
        fixtures = list(collection.find(
            {'gameweek': gameweek, 'season': season}, 
            {'_id': 0}
        ).sort('id', 1))
        return fixtures
    
    def get_fixtures_by_team(self, team_id: int, season: str = "2025-26") -> List[Dict[str, Any]]:
        """Get all fixtures for a specific team"""
        collection = self.get_collection(self.fixtures_collection_name)
        fixtures = list(collection.find(
            {
                '$or': [
                    {'team_h': team_id, 'season': season},
                    {'team_a': team_id, 'season': season}
                ]
            }, 
            {'_id': 0}
        ).sort([('gameweek', 1), ('id', 1)]))
        return fixtures
    
    def get_next_fixtures(self, team_id: int, gameweek: int, count: int = 5, season: str = "2025-26") -> List[Dict[str, Any]]:
        """Get next N fixtures for a team from a specific gameweek"""
        collection = self.get_collection(self.fixtures_collection_name)
        fixtures = list(collection.find(
            {
                '$or': [
                    {'team_h': team_id, 'season': season},
                    {'team_a': team_id, 'season': season}
                ],
                'gameweek': {'$gte': gameweek}
            }, 
            {'_id': 0}
        ).sort([('gameweek', 1), ('id', 1)]).limit(count))
        return fixtures
    
    def upsert_fixture(self, fixture_data: Dict[str, Any]) -> None:
        """Insert or update a fixture"""
        collection = self.get_collection(self.fixtures_collection_name)
        collection.replace_one(
            {'id': fixture_data['id']}, 
            fixture_data, 
            upsert=True
        )
    
    def bulk_upsert_fixtures(self, fixtures_data: List[Dict[str, Any]]) -> None:
        """Bulk insert or update fixtures"""
        if not fixtures_data:
            return
            
        collection = self.get_collection(self.fixtures_collection_name)
        operations = []
        
        for fixture in fixtures_data:
            operations.append(
                ReplaceOne(
                    {'id': fixture['id']},
                    fixture,
                    upsert=True
                )
            )
        
        if operations:
            result = collection.bulk_write(operations)
            logger.info(f"✅ Bulk upserted {result.upserted_count + result.modified_count} fixtures")
    
    # === DATABASE MANAGEMENT ===
    
    def create_indexes(self) -> None:
        """Create indexes for better query performance"""
        logger.info("📈 Creating database indexes...")
        
        # Players indexes
        players_collection = self.get_collection(self.players_collection_name)
        players_collection.create_index('id', unique=True)
        players_collection.create_index('element_type')  # Position
        players_collection.create_index('team')
        players_collection.create_index('web_name')
        
        # Teams indexes
        teams_collection = self.get_collection(self.teams_collection_name)
        teams_collection.create_index('id', unique=True)
        teams_collection.create_index('short_name')
        
        # Fixtures indexes
        fixtures_collection = self.get_collection(self.fixtures_collection_name)
        fixtures_collection.create_index('id', unique=True)
        fixtures_collection.create_index([('gameweek', 1), ('season', 1)])
        fixtures_collection.create_index('team_h')
        fixtures_collection.create_index('team_a')
        fixtures_collection.create_index('season')
        
        logger.info("✅ Database indexes created")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        if self.database is None:
            return stats
        
        # Collection counts
        stats['players_count'] = self.get_collection(self.players_collection_name).count_documents({})
        stats['teams_count'] = self.get_collection(self.teams_collection_name).count_documents({})
        stats['fixtures_count'] = self.get_collection(self.fixtures_collection_name).count_documents({})
        
        # Database size
        db_stats = self.database.command("dbstats")
        stats['database_size_mb'] = round(db_stats['dataSize'] / (1024 * 1024), 2)
        stats['index_size_mb'] = round(db_stats['indexSize'] / (1024 * 1024), 2)
        
        return stats


# === UTILITY FUNCTIONS ===

def load_teams_data(mongo_client: Optional[FPLMongoClient] = None) -> List[Dict[str, Any]]:
    """
    Load teams data - compatible replacement for JSON file loading
    
    Args:
        mongo_client: Optional MongoDB client instance
        
    Returns:
        List of team dictionaries
    """
    if mongo_client:
        return mongo_client.get_all_teams()
    else:
        # Create temporary client for one-off operations
        client = FPLMongoClient()
        try:
            return client.get_all_teams()
        finally:
            client.disconnect()


def get_fpl_mongo_client() -> FPLMongoClient:
    """Get a configured FPL MongoDB client instance"""
    return FPLMongoClient()


if __name__ == "__main__":
    # Test the MongoDB client
    client = FPLMongoClient()
    
    try:
        # Test connection and get stats
        stats = client.get_database_stats()
        print("📊 Database Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.disconnect()
