#!/usr/bin/env python3
"""
Enhance Training Data with Team Strength Features

Adds team strength features to canonical player_game_stats.parquet for model retraining.
This version works on historical training data without requiring gameweek fixtures.

Features added:
- team_attack_index: Team's rolling attack strength (xG for)
- team_defense_index: Team's rolling defense strength (xG allowed)
- opponent_attack_index: Set to 0 for training (no opponent context)
- opponent_defense_index: Set to 0 for training (no opponent context)
- venue: Set to 'N' for training (neutral)
- opponent_team_id: Set to '' for training (no opponent)

Usage:
  python enhance_training_data.py \
    --input-file Data/fbref_ingest/canonical/player_game_stats.parquet \
    --output-file Data/fbref_ingest/canonical/player_game_stats_enhanced.parquet
"""

import os
import sys
import argparse
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path

# Add Data directory to path for imports
DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

# Configuration
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
    Compute rolling team strength indices for training data.
    
    Returns:
        attack_indices: Dict[team_id, avg_xg_for]
        defense_indices: Dict[team_id, avg_xg_allowed]
    """
    # Sort by date for rolling calculations
    if 'match_date' in matches.columns:
        matches = matches.sort_values('match_date')
    if 'match_date' in team_stats.columns:
        team_stats = team_stats.sort_values('match_date')
    
    # Compute attack indices from matches (xG for)
    attack_data = []
    for _, match in matches.iterrows():
        # Home team attack
        if pd.notna(match.get('home_team_id')):
            attack_data.append({
                'team_id': match['home_team_id'],
                'date': match['match_date'],
                'xg_for': float(match.get('home_xg', 0) or 0)
            })
        # Away team attack  
        if pd.notna(match.get('away_team_id')):
            attack_data.append({
                'team_id': match['away_team_id'],
                'date': match['match_date'],
                'xg_for': float(match.get('away_xg', 0) or 0)
            })
    
    attack_df = pd.DataFrame(attack_data)
    if not attack_df.empty and 'date' in attack_df.columns:
        attack_df = attack_df.sort_values(['team_id', 'date'])
    
    # Rolling attack averages (use overall average for training)
    attack_indices = {}
    if not attack_df.empty:
        for team_id in attack_df['team_id'].unique():
            if not isinstance(team_id, str) or not team_id:
                continue
            team_data = attack_df[attack_df['team_id'] == team_id]
            if len(team_data) >= MIN_MATCHES:
                # Use overall average rather than rolling for training stability
                overall_avg = team_data['xg_for'].mean()
                attack_indices[team_id] = overall_avg
    
    # Compute defense indices from team_stats (xG allowed)
    defense_indices = {}
    if not team_stats.empty:
        for team_id in team_stats['fbref_team_id'].unique():
            if not isinstance(team_id, str) or not team_id:
                continue
            team_data = team_stats[team_stats['fbref_team_id'] == team_id]
            if len(team_data) >= MIN_MATCHES:
                # Use overall average rather than rolling for training stability
                overall_avg = team_data['xg_allowed'].mean()
                defense_indices[team_id] = overall_avg
    
    return attack_indices, defense_indices

def add_team_strength_features(features_df: pd.DataFrame, attack_indices: Dict[str, float], defense_indices: Dict[str, float]) -> pd.DataFrame:
    """
    Add team strength features to the features DataFrame for training data.
    
    For training data, we only add team-level features, not opponent-specific ones.
    """
    df = features_df.copy()
    
    # Initialize new columns
    df['team_attack_index'] = 0.0
    df['team_defense_index'] = 0.0
    df['opponent_attack_index'] = 0.0  # Not used in training
    df['opponent_defense_index'] = 0.0  # Not used in training  
    df['venue'] = 'N'  # Neutral for training
    df['opponent_team_id'] = ''  # No opponent for training
    
    # Add team strength indices
    if 'fbref_team_id' in df.columns:
        df['team_attack_index'] = df['fbref_team_id'].map(attack_indices).fillna(0.0)
        df['team_defense_index'] = df['fbref_team_id'].map(defense_indices).fillna(0.0)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Enhance training data with team strength features')
    parser.add_argument('--input-file', required=True, help='Path to input training data file')
    parser.add_argument('--output-file', help='Output file path (defaults to input with _enhanced suffix)')
    parser.add_argument('--rolling-n', type=int, default=ROLLING_N, help='Number of matches for rolling averages')
    parser.add_argument('--dry-run', action='store_true', help='Print feature summary without saving')
    
    args = parser.parse_args()
    
    # Set output file if not provided
    if not args.output_file:
        input_path = Path(args.input_file)
        if input_path.suffix == '.parquet':
            args.output_file = str(input_path.with_suffix('')) + '_enhanced.parquet'
        else:
            args.output_file = str(input_path.with_suffix('')) + '_enhanced' + input_path.suffix
    
    print(f"Loading canonical data...")
    matches, team_stats = load_canonical_data()
    
    print(f"Computing team strength indices (overall averages)...")
    attack_indices, defense_indices = compute_team_strength_indices(matches, team_stats, args.rolling_n)
    
    print(f"Found strength indices for {len(attack_indices)} teams (attack), {len(defense_indices)} teams (defense)")
    
    # Load training features
    input_path = Path(args.input_file)
    if input_path.suffix == '.parquet':
        features_df = pd.read_parquet(args.input_file)
    else:
        features_df = pd.read_csv(args.input_file)
    
    print(f"Loaded {len(features_df)} player features from {args.input_file}")
    
    # Add strength features
    enhanced_df = add_team_strength_features(features_df, attack_indices, defense_indices)
    
    # Print summary
    print(f"\nAdded features summary:")
    new_features = ['team_attack_index', 'team_defense_index', 'opponent_attack_index', 'opponent_defense_index', 'venue', 'opponent_team_id']
    for feat in new_features:
        if feat in enhanced_df.columns:
            if enhanced_df[feat].dtype in ['int64', 'float64']:
                print(f"  {feat}: mean={enhanced_df[feat].mean():.3f}, std={enhanced_df[feat].std():.3f}")
            else:
                print(f"  {feat}: {dict(enhanced_df[feat].value_counts())}")
    
    print(f"Feature count: {len(features_df.columns)} -> {len(enhanced_df.columns)} (+{len(enhanced_df.columns) - len(features_df.columns)})")
    
    if args.dry_run:
        print("Dry run - no files written")
    else:
        # Save enhanced features
        if Path(args.output_file).suffix == '.parquet':
            enhanced_df.to_parquet(args.output_file, index=False)
        else:
            enhanced_df.to_csv(args.output_file, index=False)
        print(f"Saved enhanced training data to {args.output_file}")

if __name__ == "__main__":
    main()
