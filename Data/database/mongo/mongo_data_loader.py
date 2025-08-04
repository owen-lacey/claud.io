"""
MongoDB Data Loading Utilities
==============================

Drop-in replacement functions for JSON file loading.
These functions maintain compatibility with existing code while using MongoDB.

Key functions:
- load_teams_data(): Replaces JSON file loading for teams
- load_players_data(): Replaces JSON file loading for players  
- load_fixtures_data(): Replaces JSON file loading for fixtures

Author: FPL Team Picker
Version: 1.0.0
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add the mongo directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from fpl_mongo_client import FPLMongoClient
except ImportError:
    # Fallback to JSON files if MongoDB is not available
    FPLMongoClient = None

logger = logging.getLogger(__name__)


# Global MongoDB client instance for reuse
_mongo_client: Optional[FPLMongoClient] = None


def get_mongo_client() -> Optional[FPLMongoClient]:
    """Get or create a MongoDB client instance"""
    global _mongo_client
    
    if FPLMongoClient is None:
        return None
    
    if _mongo_client is None:
        try:
            _mongo_client = FPLMongoClient()
            logger.info("✅ MongoDB client initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to MongoDB: {e}")
            _mongo_client = None
    
    return _mongo_client


def load_teams_data(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load teams data from MongoDB or fallback to JSON file
    
    Args:
        file_path: Path to JSON file (used as fallback if MongoDB unavailable)
        
    Returns:
        List of team dictionaries
        
    Note:
        This function maintains compatibility with existing code that expects
        teams data as a list of dictionaries.
    """
    
    # Try MongoDB first
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            teams = mongo_client.get_all_teams()
            logger.debug(f"📊 Loaded {len(teams)} teams from MongoDB")
            return teams
        except Exception as e:
            logger.warning(f"⚠️ MongoDB query failed, falling back to JSON: {e}")
    
    # Fallback to JSON file
    if file_path is None:
        # Default path relative to this file
        file_path = str(Path(__file__).parent.parent / 'teams.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            teams = json.load(f)
        logger.debug(f"📊 Loaded {len(teams)} teams from JSON file")
        return teams
    except Exception as e:
        logger.error(f"❌ Could not load teams data: {e}")
        raise


def load_players_data(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load players data from MongoDB or fallback to JSON file
    
    Args:
        file_path: Path to JSON file (used as fallback if MongoDB unavailable)
        
    Returns:
        List of player dictionaries
    """
    
    # Try MongoDB first
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            players = mongo_client.get_all_players()
            logger.debug(f"📊 Loaded {len(players)} players from MongoDB")
            return players
        except Exception as e:
            logger.warning(f"⚠️ MongoDB query failed, falling back to JSON: {e}")
    
    # Fallback to JSON file
    if file_path is None:
        # Default path relative to this file
        file_path = str(Path(__file__).parent.parent / 'players.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            players = json.load(f)
        logger.debug(f"📊 Loaded {len(players)} players from JSON file")
        return players
    except Exception as e:
        logger.error(f"❌ Could not load players data: {e}")
        raise


def load_fixtures_data(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load fixtures data from MongoDB or fallback to JSON file
    
    Args:
        file_path: Path to JSON file (used as fallback if MongoDB unavailable)
        
    Returns:
        List of fixture dictionaries
    """
    
    # Try MongoDB first
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            fixtures = mongo_client.get_all_fixtures()
            logger.debug(f"📊 Loaded {len(fixtures)} fixtures from MongoDB")
            return fixtures
        except Exception as e:
            logger.warning(f"⚠️ MongoDB query failed, falling back to JSON: {e}")
    
    # Fallback to JSON file
    if file_path is None:
        # Default path relative to this file
        file_path = str(Path(__file__).parent.parent / 'fixtures.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            fixtures = json.load(f)
        logger.debug(f"📊 Loaded {len(fixtures)} fixtures from JSON file")
        return fixtures
    except Exception as e:
        logger.error(f"❌ Could not load fixtures data: {e}")
        raise


def get_players_by_team(team_id: int) -> List[Dict[str, Any]]:
    """
    Get all players from a specific team
    
    Args:
        team_id: FPL team ID
        
    Returns:
        List of player dictionaries for the team
    """
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            return mongo_client.get_players_by_team(team_id)
        except Exception as e:
            logger.warning(f"⚠️ MongoDB query failed: {e}")
    
    # Fallback to loading all players and filtering
    all_players = load_players_data()
    return [player for player in all_players if player.get('team') == team_id]


def get_players_by_position(position_type: int) -> List[Dict[str, Any]]:
    """
    Get all players by position type
    
    Args:
        position_type: 1=GK, 2=DEF, 3=MID, 4=FWD
        
    Returns:
        List of player dictionaries for the position
    """
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            return mongo_client.get_players_by_position(position_type)
        except Exception as e:
            logger.warning(f"⚠️ MongoDB query failed: {e}")
    
    # Fallback to loading all players and filtering
    all_players = load_players_data()
    return [player for player in all_players if player.get('element_type') == position_type]


def get_fixtures_by_gameweek(gameweek: int, season: str = "2025-26") -> List[Dict[str, Any]]:
    """
    Get fixtures for a specific gameweek
    
    Args:
        gameweek: Gameweek number
        season: Season string (default: "2025-26")
        
    Returns:
        List of fixture dictionaries for the gameweek
    """
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            return mongo_client.get_fixtures_by_gameweek(gameweek, season)
        except Exception as e:
            logger.warning(f"⚠️ MongoDB query failed: {e}")
    
    # Fallback to loading all fixtures and filtering
    all_fixtures = load_fixtures_data()
    return [
        fixture for fixture in all_fixtures 
        if fixture.get('gameweek') == gameweek and fixture.get('season') == season
    ]


def get_fixtures_by_team(team_id: int, season: str = "2025-26") -> List[Dict[str, Any]]:
    """
    Get all fixtures for a specific team
    
    Args:
        team_id: FPL team ID
        season: Season string (default: "2025-26")
        
    Returns:
        List of fixture dictionaries for the team
    """
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            return mongo_client.get_fixtures_by_team(team_id, season)
        except Exception as e:
            logger.warning(f"⚠️ MongoDB query failed: {e}")
    
    # Fallback to loading all fixtures and filtering
    all_fixtures = load_fixtures_data()
    return [
        fixture for fixture in all_fixtures 
        if (fixture.get('team_h') == team_id or fixture.get('team_a') == team_id) 
        and fixture.get('season') == season
    ]


# === LEGACY COMPATIBILITY FUNCTIONS ===

def get_teams_lookup() -> Dict[int, Dict[str, Any]]:
    """
    Get teams as a lookup dictionary by ID
    Maintains compatibility with existing code patterns.
    
    Returns:
        Dictionary mapping team_id -> team_data
    """
    teams = load_teams_data()
    return {team['id']: team for team in teams}


def create_teams_lookup(teams_data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Create teams lookup from teams data
    Legacy compatibility function.
    
    Args:
        teams_data: List of team dictionaries
        
    Returns:
        Dictionary mapping team_id -> team_data
    """
    return {team['id']: team for team in teams_data}


# === DATABASE UPDATE FUNCTIONS ===

def update_players_data(players_data: List[Dict[str, Any]]) -> None:
    """
    Update players data in MongoDB
    
    Args:
        players_data: List of player dictionaries to update
    """
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            mongo_client.bulk_upsert_players(players_data)
            logger.info(f"✅ Updated {len(players_data)} players in MongoDB")
        except Exception as e:
            logger.error(f"❌ Failed to update players: {e}")
            raise
    else:
        logger.warning("⚠️ MongoDB not available, cannot update players data")


def update_teams_data(teams_data: List[Dict[str, Any]]) -> None:
    """
    Update teams data in MongoDB
    
    Args:
        teams_data: List of team dictionaries to update
    """
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            mongo_client.bulk_upsert_teams(teams_data)
            logger.info(f"✅ Updated {len(teams_data)} teams in MongoDB")
        except Exception as e:
            logger.error(f"❌ Failed to update teams: {e}")
            raise
    else:
        logger.warning("⚠️ MongoDB not available, cannot update teams data")


def update_fixtures_data(fixtures_data: List[Dict[str, Any]]) -> None:
    """
    Update fixtures data in MongoDB
    
    Args:
        fixtures_data: List of fixture dictionaries to update
    """
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            mongo_client.bulk_upsert_fixtures(fixtures_data)
            logger.info(f"✅ Updated {len(fixtures_data)} fixtures in MongoDB")
        except Exception as e:
            logger.error(f"❌ Failed to update fixtures: {e}")
            raise
    else:
        logger.warning("⚠️ MongoDB not available, cannot update fixtures data")


if __name__ == "__main__":
    # Test the data loading functions
    try:
        print("🧪 Testing MongoDB data loading...")
        
        teams = load_teams_data()
        print(f"✅ Loaded {len(teams)} teams")
        
        players = load_players_data()
        print(f"✅ Loaded {len(players)} players")
        
        fixtures = load_fixtures_data()
        print(f"✅ Loaded {len(fixtures)} fixtures")
        
        # Test specific queries
        gk_players = get_players_by_position(1)
        print(f"✅ Found {len(gk_players)} goalkeepers")
        
        gw1_fixtures = get_fixtures_by_gameweek(1)
        print(f"✅ Found {len(gw1_fixtures)} GW1 fixtures")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Close MongoDB connection
        if _mongo_client:
            _mongo_client.disconnect()
