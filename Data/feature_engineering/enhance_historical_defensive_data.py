"""
Historical Defensive Contributions Enhancement Script

This script processes historical FBRef data to calculate defensive contributions
that would have been awarded under the new FPL 2025/26 scoring rules.

This is essential for training ML models to predict defensive contribution likelihood.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path
import sys
import argparse
from typing import Dict, List, Optional

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))
from defensive_contributions import DefensiveContributionCalculator
from player_features import PlayerFeatureEngine


def load_fbref_data(db_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load FBRef data from SQLite database
    
    Args:
        db_path: Path to the FBRef SQLite database
        limit: Optional limit on number of records to process
        
    Returns:
        DataFrame with combined FBRef defensive and match data
    """
    print(f"📂 Loading FBRef data from {db_path}")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"❌ FBRef database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Query to combine defensive actions, miscellaneous stats, and player info
        query = """
        SELECT 
            pi.match_id,
            pi.player_id,
            pi.name,
            pi.position,
            pi.minutes,
            pi.home_away,
            pi.start,
            
            -- Defensive actions (CBIT components)
            COALESCE(da.clearances, 0) as clearances,
            COALESCE(da.blocks, 0) as blocks,
            COALESCE(da.interceptions, 0) as interceptions,
            COALESCE(da.tackles_won, 0) as tackles_won,
            
            -- Miscellaneous stats (for recoveries)
            COALESCE(m.recoveries, 0) as recoveries,
            COALESCE(m.yellow_cards, 0) as yellow_cards,
            COALESCE(m.red_cards, 0) as red_cards,
            
            -- Summary stats for context
            COALESCE(s.goals, 0) as goals,
            COALESCE(s.assists, 0) as assists,
            COALESCE(s.xG, 0) as xG,
            COALESCE(s.xA, 0) as xA,
            
            -- Match info
            ma.date,
            ma.home_team,
            ma.away_team,
            ma.competition,
            ma.season
            
        FROM Player_Info pi
        LEFT JOIN Defensive_Actions da ON pi.match_id = da.match_id AND pi.player_id = da.player_id
        LEFT JOIN Miscellaneous m ON pi.match_id = m.match_id AND pi.player_id = m.player_id  
        LEFT JOIN Summary s ON pi.match_id = s.match_id AND pi.player_id = s.player_id
        LEFT JOIN Match ma ON pi.match_id = ma.match_id
        
        WHERE pi.minutes > 0  -- Only include players who actually played
        ORDER BY ma.date, pi.match_id, pi.player_id
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        df = pd.read_sql_query(query, conn)
        
        print(f"✅ Loaded {len(df)} player-match records")
        print(f"📊 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🏆 Competitions: {df['competition'].nunique()} ({', '.join(df['competition'].unique())})")
        print(f"📅 Seasons: {df['season'].nunique()} ({', '.join(df['season'].unique())})")
        
        return df
        
    finally:
        conn.close()


def enhance_historical_data(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Enhance historical data with defensive contributions
    
    Args:
        df: DataFrame with FBRef data
        output_path: Path to save enhanced data
        
    Returns:
        Enhanced DataFrame with defensive contribution features
    """
    print("\n🛡️ Calculating defensive contributions for historical data...")
    
    # Initialize calculator
    calc = DefensiveContributionCalculator()
    
    # Calculate defensive contributions
    enhanced_df = calc.calculate_historical_defensive_contributions(df)
    
    # Add rolling features using PlayerFeatureEngine
    print("\n📊 Adding rolling window features...")
    feature_engine = PlayerFeatureEngine()
    
    # We need to create a 'GW' column for rolling features - use a proxy based on date
    enhanced_df['date'] = pd.to_datetime(enhanced_df['date'])
    enhanced_df = enhanced_df.sort_values(['name', 'date'])
    
    # Create a synthetic gameweek number within each season
    enhanced_df['GW'] = enhanced_df.groupby(['name', 'season']).cumcount() + 1
    
    # Add defensive contribution features to rolling features
    enhanced_df = feature_engine.add_defensive_contributions(enhanced_df)
    
    # Calculate rolling features
    enhanced_df = feature_engine.calculate_rolling_features(
        enhanced_df, 
        group_col='name',
        sort_cols=['name', 'season', 'date']
    )
    
    # Save enhanced data
    print(f"\n💾 Saving enhanced data to {output_path}")
    enhanced_df.to_csv(output_path, index=False)
    # Also save as Parquet for compatibility
    if output_path.endswith('.csv'):
        parquet_path = output_path.replace('.csv', '.parquet')
    else:
        parquet_path = str(output_path) + '.parquet'
    enhanced_df.to_parquet(parquet_path, index=False)
    print(f"💾 Saved enhanced data to {parquet_path}")
    
    return enhanced_df


def analyze_defensive_contributions(df: pd.DataFrame) -> None:
    """
    Analyze defensive contribution patterns in the data
    
    Args:
        df: Enhanced DataFrame with defensive contributions
    """
    print("\n📈 Defensive Contribution Analysis")
    print("=" * 50)
    
    # Overall statistics
    total_matches = len(df)
    total_contributions = df['defensive_contribution_points'].sum()
    contribution_rate = (df['defensive_contribution_points'] > 0).mean()
    
    print(f"Total player-match records: {total_matches:,}")
    print(f"Total defensive contribution points awarded: {total_contributions:,}")
    print(f"Defensive contribution rate: {contribution_rate:.1%}")
    
    # By position analysis
    print("\n📊 Defensive Contributions by Position:")
    position_analysis = df.groupby('position').agg({
        'defensive_contribution_points': ['count', 'sum', 'mean'],
        'cbit_score': 'mean',
        'cbirt_score': 'mean'
    }).round(3)
    
    position_analysis.columns = ['Matches', 'Total_Points', 'Avg_Points', 'Avg_CBIT', 'Avg_CBIRT']
    position_analysis['Contribution_Rate'] = (
        df.groupby('position')['defensive_contribution_points'].apply(lambda x: (x > 0).mean()).round(3)
    )
    
    print(position_analysis)
    
    # Top performers
    print("\n🏆 Top 10 Defensive Contribution Performers:")
    top_performers = df.groupby('name').agg({
        'defensive_contribution_points': 'sum',
        'position': 'first',
        'clearances': 'mean',
        'blocks': 'mean', 
        'interceptions': 'mean',
        'tackles_won': 'mean',
        'recoveries': 'mean'
    }).sort_values('defensive_contribution_points', ascending=False).head(10)
    
    print(top_performers.round(2))
    
    # Minutes played correlation
    print(f"\n⏱️ Correlation with minutes played: {df['minutes'].corr(df['defensive_contribution_points']):.3f}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Calculate defensive contributions for historical FBRef data')
    script_dir = Path(__file__).parent.resolve()
    default_db_path = script_dir.parent.parent / 'FBRef_DB' / 'master.db'
    default_output_path = script_dir.parent / 'outputs' / 'enhanced_fbref_with_defensive_contributions.csv'

    parser.add_argument('--db_path', type=str,
                       default=str(default_db_path),
                       help='Path to FBRef SQLite database')
    parser.add_argument('--output_path', type=str,
                       default=str(default_output_path),
                       help='Output path for enhanced CSV file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records to process (for testing)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only run analysis on existing enhanced data')

    args = parser.parse_args()

    # Resolve paths relative to script location if not absolute
    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = (script_dir / args.db_path).resolve()
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = (script_dir / args.output_path).resolve()

    if args.analyze_only:
        if not output_path.exists():
            print(f"❌ Enhanced data file not found: {output_path}")
            return

        print(f"📊 Loading existing enhanced data from {output_path}")
        df = pd.read_csv(output_path)
        analyze_defensive_contributions(df)
        return

    # Load and enhance data
    df = load_fbref_data(str(db_path), limit=args.limit)
    enhanced_df = enhance_historical_data(df, str(output_path))
    analyze_defensive_contributions(enhanced_df)

    print(f"\n✅ Enhancement complete! Enhanced data saved to: {output_path}")


if __name__ == "__main__":
    # Force working directory to script's directory for robust path handling
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
