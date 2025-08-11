#!/usr/bin/env python3
"""
Add Team Strength Features to Canonical Features

Enhances the canonical features with team/opponent strength indices and fixture context
for model retraining. These features replace inference-time multipliers with learnable
team strength relationships.

New features added:
- team_attack_index: Team's rolling xG for per match (last N games)
- team_defense_index: Team's rolling xG allowed per match (last N games) 
- opponent_attack_index: Opponent's attack strength for this fixture
- opponent_defense_index: Opponent's defense strength for this fixture
- venue: Home/Away/Neutral (H/A/N)
- opponent_team_id: FBRef team ID of opponent

Usage:
  python add_team_strength_features.py \
    --features-file Data/fbref_ingest/canonical/features_gw1.csv \
    --output-file Data/fbref_ingest/canonical/features_gw1_with_strength.csv \
    --gameweek 1

Environment variables:
  FBREF_STRENGTH_ROLLING_N=6    # Number of matches for rolling averages
  FBREF_STRENGTH_MIN_MATCHES=3  # Minimum matches required for index
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add Data directory to path for imports
DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

# Configuration from environment
ROLLING_N = int(os.getenv('FBREF_STRENGTH_ROLLING_N', '6'))
MIN_MATCHES = int(os.getenv('FBREF_STRENGTH_MIN_MATCHES', '3'))

def load_canonical_data():
    """Load canonical matches and team_game_stats data."""
    base_dir = DATA_DIR / "fbref_ingest" / "canonical"
    
    # Load matches
    matches_parquet = base_dir / "matches.parquet"
    matches_csv = base_dir / "matches.csv"
    if matches_parquet.exists():
        matches = pd.read_parquet(matches_parquet)
    elif matches_csv.exists():
        matches = pd.read_csv(matches_csv)
    else:
        raise FileNotFoundError(f"Matches data not found at {base_dir}")
    
    # Load team game stats  
    team_stats_parquet = base_dir / "team_game_stats.parquet"
    team_stats_csv = base_dir / "team_game_stats.csv"
    if team_stats_parquet.exists():
        team_stats = pd.read_parquet(team_stats_parquet)
    elif team_stats_csv.exists():
        team_stats = pd.read_csv(team_stats_csv)
    else:
        raise FileNotFoundError(f"Team stats data not found at {base_dir}")
    
    return matches, team_stats

def compute_team_strength_indices(matches: pd.DataFrame, team_stats: pd.DataFrame, rolling_n: int = ROLLING_N) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute rolling team strength indices.
    
    Returns:
        attack_indices: Dict[team_id, rolling_avg_xg_for]
        defense_indices: Dict[team_id, rolling_avg_xg_allowed]
    """
    # Filter to latest season to avoid mixing eras
    if 'season' in matches.columns and not matches.empty:
        latest_season = sorted(matches['season'].unique())[-1]
        matches = matches[matches['season'] == latest_season].copy()
    
    if 'season' in team_stats.columns and not team_stats.empty:
        latest_season_ts = sorted(team_stats['season'].unique())[-1]
        team_stats = team_stats[team_stats['season'] == latest_season_ts].copy()
    
    # Sort by date for rolling calculations
    if 'match_date' in matches.columns:
        matches = matches.sort_values('match_date')
    if 'match_date' in team_stats.columns:
        team_stats = team_stats.sort_values('match_date')
    
    # Compute attack indices from matches (xG for)
    attack_data = []
    for _, match in matches.iterrows():
        # Home team attack
        attack_data.append({
            'team_id': match['home_team_id'],
            'date': match['match_date'],
            'xg_for': float(match.get('home_xg', 0) or 0)
        })
        # Away team attack  
        attack_data.append({
            'team_id': match['away_team_id'],
            'date': match['match_date'],
            'xg_for': float(match.get('away_xg', 0) or 0)
        })
    
    attack_df = pd.DataFrame(attack_data)
    if 'date' in attack_df.columns:
        attack_df = attack_df.sort_values(['team_id', 'date'])
    
    # Rolling attack averages
    attack_indices = {}
    for team_id in attack_df['team_id'].unique():
        if not isinstance(team_id, str):
            continue
        team_data = attack_df[attack_df['team_id'] == team_id]
        if len(team_data) >= MIN_MATCHES:
            rolling_avg = team_data['xg_for'].tail(rolling_n).mean()
            attack_indices[team_id] = rolling_avg
    
    # Compute defense indices from team_stats (xG allowed)
    defense_indices = {}
    for team_id in team_stats['fbref_team_id'].unique():
        if not isinstance(team_id, str):
            continue
        team_data = team_stats[team_stats['fbref_team_id'] == team_id]
        if len(team_data) >= MIN_MATCHES:
            rolling_avg = team_data['xg_allowed'].tail(rolling_n).mean()
            defense_indices[team_id] = rolling_avg
    
    return attack_indices, defense_indices

def get_fixtures_for_gameweek(gameweek: int) -> pd.DataFrame:
    """Load fixtures for the specified gameweek from Mongo."""
    try:
        from database.mongo.mongo_data_loader import get_fixtures_by_gameweek
        fixtures = get_fixtures_by_gameweek(gameweek)
        if fixtures:
            return pd.DataFrame(fixtures)
        else:
            print(f"Warning: No fixtures found for GW{gameweek}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not load fixtures for GW{gameweek}: {e}")
        return pd.DataFrame()

def get_team_fbref_mapping() -> Dict[int, str]:
    """Get mapping from FPL team ID to FBRef team name."""
    try:
        from database.mongo.mongo_data_loader import load_teams_data
        teams = load_teams_data()
        mapping = {}
        for team in teams:
            fpl_id = team.get('id')
            fbref_id = team.get('fbref_id')
            if fpl_id is not None and fbref_id:
                mapping[int(fpl_id)] = fbref_id
        return mapping
    except Exception as e:
        print(f"Warning: Could not load team mapping: {e}")
        return {}

def add_team_strength_features(features_df: pd.DataFrame, gameweek: int, attack_indices: Dict[str, float], defense_indices: Dict[str, float]) -> pd.DataFrame:
    """
    Add team strength features to the features DataFrame.
    
    Features added:
    - team_attack_index: Team's attack strength
    - team_defense_index: Team's defense strength  
    - opponent_attack_index: Opponent's attack strength
    - opponent_defense_index: Opponent's defense strength
    - venue: H/A/N for home/away/neutral
    - opponent_team_id: FBRef team ID of opponent
    """
    df = features_df.copy()
    
    # Initialize new columns
    df['team_attack_index'] = 0.0
    df['team_defense_index'] = 0.0
    df['opponent_attack_index'] = 0.0
    df['opponent_defense_index'] = 0.0
    df['venue'] = 'N'  # Default to neutral
    df['opponent_team_id'] = ''
    
    # Add team strength indices
    if 'fbref_team_id' in df.columns:
        df['team_attack_index'] = df['fbref_team_id'].map(attack_indices).fillna(0.0)
        df['team_defense_index'] = df['fbref_team_id'].map(defense_indices).fillna(0.0)
    
    # Load fixtures and team mapping for opponent features
    fixtures_df = get_fixtures_for_gameweek(gameweek)
    team_mapping = get_team_fbref_mapping()
    
    if not fixtures_df.empty and team_mapping:
        # Create fixture lookup: FBRef team -> (opponent, venue)
        fixture_lookup = {}
        
        for _, fixture in fixtures_df.iterrows():
            team_h_id = fixture.get('team_h')
            team_a_id = fixture.get('team_a')
            
            if team_h_id in team_mapping and team_a_id in team_mapping:
                team_h_fbref = team_mapping[team_h_id]
                team_a_fbref = team_mapping[team_a_id]
                
                # Home team perspective
                fixture_lookup[team_h_fbref] = {
                    'opponent': team_a_fbref,
                    'venue': 'H'
                }
                
                # Away team perspective  
                fixture_lookup[team_a_fbref] = {
                    'opponent': team_h_fbref,
                    'venue': 'A'
                }
        
        # Apply fixture-based opponent features
        for idx, row in df.iterrows():
            team_id = row.get('fbref_team_id')
            if team_id in fixture_lookup:
                fixture_info = fixture_lookup[team_id]
                opponent = fixture_info['opponent']
                venue = fixture_info['venue']
                
                df.at[idx, 'opponent_team_id'] = opponent
                df.at[idx, 'venue'] = venue
                df.at[idx, 'opponent_attack_index'] = attack_indices.get(opponent, 0.0)
                df.at[idx, 'opponent_defense_index'] = defense_indices.get(opponent, 0.0)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Add team strength features to canonical features')
    parser.add_argument('--features-file', required=True, help='Path to canonical features file')
    parser.add_argument('--output-file', help='Output file path (defaults to input with _with_strength suffix)')
    parser.add_argument('--gameweek', type=int, required=True, help='Gameweek number for fixture lookup')
    parser.add_argument('--rolling-n', type=int, default=ROLLING_N, help='Number of matches for rolling averages')
    parser.add_argument('--dry-run', action='store_true', help='Print feature summary without saving')
    
    args = parser.parse_args()
    
    print(f"Loading canonical data...")
    matches, team_stats = load_canonical_data()
    
    print(f"Computing team strength indices (rolling {args.rolling_n} matches)...")
    attack_indices, defense_indices = compute_team_strength_indices(matches, team_stats, args.rolling_n)
    
    print(f"Found strength indices for {len(attack_indices)} teams (attack), {len(defense_indices)} teams (defense)")
    
    # Load features
    features_df = pd.read_csv(args.features_file)
    print(f"Loaded {len(features_df)} player features from {args.features_file}")
    
    # Add strength features
    enhanced_df = add_team_strength_features(features_df, args.gameweek, attack_indices, defense_indices)
    
    # Show feature summary
    new_cols = ['team_attack_index', 'team_defense_index', 'opponent_attack_index', 'opponent_defense_index', 'venue', 'opponent_team_id']
    print(f"\nAdded features summary:")
    for col in new_cols:
        if col in enhanced_df.columns:
            if enhanced_df[col].dtype in ['float64', 'int64']:
                print(f"  {col}: mean={enhanced_df[col].mean():.3f}, std={enhanced_df[col].std():.3f}")
            else:
                value_counts = enhanced_df[col].value_counts()
                print(f"  {col}: {dict(value_counts.head(3))}")
    
    if args.dry_run:
        print("Dry run - no files written")
        return
    
    # Save output
    if args.output_file:
        output_path = args.output_file
    else:
        input_path = Path(args.features_file)
        output_path = input_path.parent / f"{input_path.stem}_with_strength{input_path.suffix}"
    
    enhanced_df.to_csv(output_path, index=False)
    print(f"Saved enhanced features to {output_path}")
    print(f"Feature count: {len(features_df.columns)} -> {len(enhanced_df.columns)} (+{len(new_cols)})")

if __name__ == '__main__':
    main()
