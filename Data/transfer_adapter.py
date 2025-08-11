#!/usr/bin/env python3
"""
Transfer Adapter for FBRef Features

Patches canonical features for players with no EPL history by using their
previous league performance scaled by league multipliers.

Usage:
  python transfer_adapter.py \
    --features-file Data/fbref_ingest/canonical/features_gw1.csv \
    --sqlite-path /Users/owen/src/Personal/FBRef_DB/master.db \
    --output-file Data/fbref_ingest/canonical/features_gw1_with_transfers.csv

Environment variables:
  FBREF_TRANSFER_LAST_N=6          # Number of recent matches to use
  FBREF_TRANSFER_MIN_MINUTES=180   # Minimum minutes to consider
  FBREF_TRANSFER_MAX_DAYS=90       # Maximum days since last match

Process:
1. Load canonical features and identify players with 0 EPL minutes
2. Query SQLite for their last N matches in any league  
3. Apply league multipliers to per90 rates
4. Update features with scaled previous league performance
5. Save patched features file
"""

import os
import sys
import json
import sqlite3
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add Data directory to path for imports
DATA_DIR = Path(__file__).resolve().parent.parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

# Configuration from environment
LAST_N_MATCHES = int(os.getenv('FBREF_TRANSFER_LAST_N', '6'))
MIN_MINUTES = int(os.getenv('FBREF_TRANSFER_MIN_MINUTES', '180'))
MAX_DAYS = int(os.getenv('FBREF_TRANSFER_MAX_DAYS', '90'))

def load_league_multipliers(multipliers_path: str) -> Dict[str, Dict[str, float]]:
    """Load league multipliers from JSON file."""
    with open(multipliers_path, 'r') as f:
        data = json.load(f)
    # Remove metadata key if present
    return {k: v for k, v in data.items() if not k.startswith('_')}

def get_player_previous_league_stats(sqlite_path: str, player_id: str) -> Optional[Dict[str, float]]:
    """
    Get player's last N matches performance from SQLite across all leagues.
    Returns aggregated per90 stats if sufficient data, None otherwise.
    """
    try:
        con = sqlite3.connect(sqlite_path)
        cur = con.cursor()
        
        # Get last N matches with match dates
        query = """
        SELECT s.*, m.competition, m.date, p.minutes
        FROM Summary s
        JOIN Match m ON s.match_id = m.match_id  
        JOIN Player_Info p ON s.match_id = p.match_id AND s.player_id = p.player_id
        WHERE s.player_id = ?
        ORDER BY m.date DESC
        LIMIT ?
        """
        
        cur.execute(query, (player_id, LAST_N_MATCHES))
        rows = cur.fetchall()
        
        if not rows:
            return None
            
        # Get column names
        columns = [description[0] for description in cur.description]
        matches = [dict(zip(columns, row)) for row in rows]
        
        # Check if data is recent enough
        latest_date_str = matches[0]['date']
        if latest_date_str:
            try:
                latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
                days_ago = (datetime.now() - latest_date).days
                if days_ago > MAX_DAYS:
                    return None
            except ValueError:
                pass  # If date parsing fails, continue anyway
        
        # Aggregate stats
        total_minutes = sum(m['minutes'] or 0 for m in matches)
        if total_minutes < MIN_MINUTES:
            return None
            
        # Calculate per90 rates
        minutes_90 = total_minutes / 90.0
        
        # Get most common competition for league multiplier
        competitions = [m['competition'] for m in matches if m['competition']]
        if not competitions:
            return None
        most_common_league = max(set(competitions), key=competitions.count)
        
        stats = {
            'total_minutes': total_minutes,
            'minutes_90': minutes_90,
            'league': most_common_league,
            'xg_per90': sum(m['xG'] or 0 for m in matches) / minutes_90 if minutes_90 > 0 else 0,
            'xa_per90': sum(m['xA'] or 0 for m in matches) / minutes_90 if minutes_90 > 0 else 0,
            'shots_per90': sum(m['shots'] or 0 for m in matches) / minutes_90 if minutes_90 > 0 else 0,
            'key_passes_per90': sum(m['goal_creating_actions'] or 0 for m in matches) / minutes_90 if minutes_90 > 0 else 0,  # Proxy for key passes
            'goals': sum(m['goals'] or 0 for m in matches),
            'assists': sum(m['assists'] or 0 for m in matches),
            'minutes_last_3': sum(m['minutes'] or 0 for m in matches[:3]),
            'minutes_last_5': sum(m['minutes'] or 0 for m in matches[:5]),
            'starts_last_3': sum(1 for m in matches[:3] if (m['minutes'] or 0) > 0),
            'starts_last_5': sum(1 for m in matches[:5] if (m['minutes'] or 0) > 0),
        }
        
        return stats
        
    except Exception as e:
        print(f"Error querying player {player_id}: {e}")
        return None
    finally:
        con.close()

def apply_league_multipliers(stats: Dict[str, float], league: str, multipliers: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Apply league multipliers to per90 stats."""
    if league not in multipliers:
        print(f"Warning: No multipliers found for league '{league}', using 1.0")
        return stats
    
    league_mults = multipliers[league]
    adjusted_stats = stats.copy()
    
    # Apply multipliers to per90 rates only
    per90_fields = ['xg_per90', 'xa_per90', 'shots_per90', 'key_passes_per90']
    mult_fields = ['xg', 'xa', 'shots', 'key_passes']
    
    for per90_field, mult_field in zip(per90_fields, mult_fields):
        if per90_field in adjusted_stats and mult_field in league_mults:
            adjusted_stats[per90_field] *= league_mults[mult_field]
    
    return adjusted_stats

def patch_features_with_transfers(features_df: pd.DataFrame, sqlite_path: str, multipliers: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Patch features DataFrame with transfer player data.
    Identifies players with 0 EPL minutes and fills with previous league performance.
    """
    df = features_df.copy()
    patched_count = 0
    
    # Identify transfer candidates (players with very low EPL minutes)
    transfer_candidates = df[
        (df.get('minutes_last_5', 0) == 0) | 
        (df.get('minutes_last_3', 0) == 0)
    ].copy()
    
    print(f"Found {len(transfer_candidates)} potential transfer players")
    
    for idx, row in transfer_candidates.iterrows():
        player_id = row.get('fbref_player_id')
        if not player_id:
            continue
            
        # Get previous league stats
        prev_stats = get_player_previous_league_stats(sqlite_path, str(player_id))
        if not prev_stats:
            continue
            
        # Apply league multipliers
        adjusted_stats = apply_league_multipliers(prev_stats, prev_stats['league'], multipliers)
        
        # Update features
        update_fields = {
            'xg_per90': adjusted_stats.get('xg_per90', 0),
            'xa_per90': adjusted_stats.get('xa_per90', 0), 
            'shots_per90': adjusted_stats.get('shots_per90', 0),
            'key_passes_per90': adjusted_stats.get('key_passes_per90', 0),
            'goals': adjusted_stats.get('goals', 0),
            'assists': adjusted_stats.get('assists', 0),
            'minutes_last_3': adjusted_stats.get('minutes_last_3', 0),
            'minutes_last_5': adjusted_stats.get('minutes_last_5', 0),
            'starts_last_3': adjusted_stats.get('starts_last_3', 0),
            'starts_last_5': adjusted_stats.get('starts_last_5', 0),
        }
        
        # Only update fields that exist in the DataFrame
        for field, value in update_fields.items():
            if field in df.columns:
                df.at[idx, field] = value
        
        patched_count += 1
        print(f"Patched {row.get('player_name', player_id)} with {prev_stats['league']} data (xG90: {adjusted_stats['xg_per90']:.3f})")
    
    print(f"Successfully patched {patched_count} transfer players")
    return df

def main():
    parser = argparse.ArgumentParser(description='Patch FBRef features with transfer player data')
    parser.add_argument('--features-file', required=True, help='Path to canonical features file')
    parser.add_argument('--sqlite-path', required=True, help='Path to FBRef SQLite database')
    parser.add_argument('--output-file', help='Output file path (defaults to input with _with_transfers suffix)')
    parser.add_argument('--multipliers-file', default='Data/crosswalk/league_multipliers.json', help='League multipliers JSON file')
    parser.add_argument('--dry-run', action='store_true', help='Print changes without saving')
    
    args = parser.parse_args()
    
    # Load data
    features_df = pd.read_csv(args.features_file)
    multipliers = load_league_multipliers(args.multipliers_file)
    
    print(f"Loaded {len(features_df)} players from {args.features_file}")
    print(f"Loaded multipliers for {len(multipliers)} leagues")
    
    # Patch features
    patched_df = patch_features_with_transfers(features_df, args.sqlite_path, multipliers)
    
    if args.dry_run:
        print("Dry run - no files written")
        return
    
    # Save output
    if args.output_file:
        output_path = args.output_file
    else:
        input_path = Path(args.features_file)
        output_path = input_path.parent / f"{input_path.stem}_with_transfers{input_path.suffix}"
    
    patched_df.to_csv(output_path, index=False)
    print(f"Saved patched features to {output_path}")

if __name__ == '__main__':
    main()
