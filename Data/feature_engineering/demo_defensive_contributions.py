"""
Defensive Contributions Implementation Demo

This script demonstrates the full defensive contributions implementation
including data processing, feature engineering, and model training workflow.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add feature engineering path
sys.path.append(str(Path(__file__).parent))
from defensive_contributions import DefensiveContributionCalculator, test_defensive_contribution_calculator
from player_features import PlayerFeatureEngine


def create_sample_fbref_data(n_matches: int = 1000) -> pd.DataFrame:
    """
    Create sample FBRef-style data for testing
    
    Args:
        n_matches: Number of player-match records to generate
        
    Returns:
        DataFrame with sample defensive stats
    """
    print(f"🎲 Generating {n_matches} sample player-match records...")
    
    np.random.seed(42)
    
    # Player pools by position
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    position_weights = [0.1, 0.3, 0.4, 0.2]  # Realistic distribution
    
    players = {
        'GKP': [f'Goalkeeper_{i}' for i in range(1, 21)],
        'DEF': [f'Defender_{i}' for i in range(1, 61)], 
        'MID': [f'Midfielder_{i}' for i in range(1, 81)],
        'FWD': [f'Forward_{i}' for i in range(1, 41)]
    }
    
    data = []
    
    for i in range(n_matches):
        # Sample position and player
        position = np.random.choice(positions, p=position_weights)
        player = np.random.choice(players[position])
        
        # Generate realistic defensive stats based on position
        if position == 'GKP':
            # Goalkeepers: minimal defensive actions
            clearances = np.random.poisson(1)
            blocks = np.random.poisson(0.5)
            interceptions = np.random.poisson(0.5)
            tackles_won = np.random.poisson(0.2)
            recoveries = np.random.poisson(2)
            
        elif position == 'DEF':
            # Defenders: high defensive actions
            clearances = np.random.poisson(5)
            blocks = np.random.poisson(2)
            interceptions = np.random.poisson(3)
            tackles_won = np.random.poisson(2.5)
            recoveries = np.random.poisson(4)
            
        elif position == 'MID':
            # Midfielders: moderate defensive actions, higher recoveries
            clearances = np.random.poisson(1.5)
            blocks = np.random.poisson(1)
            interceptions = np.random.poisson(2.5)
            tackles_won = np.random.poisson(2)
            recoveries = np.random.poisson(6)
            
        else:  # FWD
            # Forwards: minimal defensive actions
            clearances = np.random.poisson(0.5)
            blocks = np.random.poisson(0.3)
            interceptions = np.random.poisson(0.8)
            tackles_won = np.random.poisson(1)
            recoveries = np.random.poisson(3)
        
        # Other match context
        minutes = np.random.choice([0, 30, 45, 60, 75, 90], p=[0.1, 0.05, 0.05, 0.1, 0.2, 0.5])
        
        data.append({
            'match_id': f'M{i//22 + 1:04d}',  # ~22 players per match
            'player_id': f'P{hash(player) % 10000:04d}',
            'name': player,
            'position': position,
            'minutes': minutes,
            'clearances': clearances,
            'blocks': blocks,
            'interceptions': interceptions,
            'tackles_won': tackles_won,
            'recoveries': recoveries,
            'goals': np.random.poisson(0.1) if position in ['MID', 'FWD'] else np.random.poisson(0.05),
            'assists': np.random.poisson(0.15) if position in ['MID', 'FWD'] else np.random.poisson(0.05),
            'xG': np.random.exponential(0.2) if position in ['MID', 'FWD'] else np.random.exponential(0.05),
            'xA': np.random.exponential(0.15) if position in ['MID', 'FWD'] else np.random.exponential(0.05),
            'yellow_cards': np.random.choice([0, 1], p=[0.9, 0.1]),
            'home_away': np.random.choice(['H', 'A']),
            'start': 1 if minutes >= 60 else np.random.choice([0, 1], p=[0.7, 0.3]),
            'date': f'2024-{np.random.randint(8, 13):02d}-{np.random.randint(1, 29):02d}',
            'season': '2024-25',
            'competition': 'Premier League'
        })
    
    df = pd.DataFrame(data)
    
    print(f"✅ Generated sample data with {len(df)} records")
    print(f"📊 Position distribution: {df['position'].value_counts().to_dict()}")
    
    return df


def demo_defensive_contributions_pipeline():
    """Run the complete defensive contributions demo"""
    print("🛡️ DEFENSIVE CONTRIBUTIONS IMPLEMENTATION DEMO")
    print("=" * 60)
    
    # Step 1: Test the defensive contribution calculator
    print("\n1️⃣ Testing Defensive Contribution Calculator")
    print("-" * 45)
    test_defensive_contribution_calculator()
    
    # Step 2: Create sample data
    print("\n2️⃣ Creating Sample FBRef Data")
    print("-" * 35)
    sample_data = create_sample_fbref_data(n_matches=2000)
    
    # Step 3: Calculate defensive contributions
    print("\n3️⃣ Calculating Defensive Contributions")
    print("-" * 40)
    calc = DefensiveContributionCalculator()
    enhanced_data = calc.calculate_historical_defensive_contributions(sample_data)
    
    # Step 4: Feature engineering with rolling windows
    print("\n4️⃣ Feature Engineering with Rolling Windows")
    print("-" * 45)
    feature_engine = PlayerFeatureEngine()
    
    # Create synthetic gameweek progression
    enhanced_data['date'] = pd.to_datetime(enhanced_data['date'])
    enhanced_data = enhanced_data.sort_values(['name', 'date'])
    enhanced_data['GW'] = enhanced_data.groupby(['name', 'season']).cumcount() + 1
    
    # Calculate rolling features
    final_data = feature_engine.calculate_rolling_features(
        enhanced_data,
        group_col='name',
        sort_cols=['name', 'season', 'date']
    )
    
    # Step 5: Analysis and insights
    print("\n5️⃣ Analysis and Insights")
    print("-" * 25)
    
    # Overall defensive contribution statistics
    total_records = len(final_data)
    total_contributions = final_data['defensive_contribution_points'].sum()
    contribution_rate = (final_data['defensive_contribution_points'] > 0).mean()
    
    print(f"📈 Total player-match records: {total_records:,}")
    print(f"🏆 Total defensive contribution points: {total_contributions:,}")
    print(f"📊 Defensive contribution rate: {contribution_rate:.1%}")
    
    # Position breakdown
    print("\n📊 Defensive Contributions by Position:")
    position_stats = final_data.groupby('position').agg({
        'defensive_contribution_points': ['count', 'sum', 'mean'],
        'cbit_score': 'mean',
        'cbirt_score': 'mean'
    }).round(3)
    
    position_stats.columns = ['Games', 'Total_Points', 'Avg_Points', 'Avg_CBIT', 'Avg_CBIRT']
    print(position_stats)
    
    # Rolling averages effectiveness
    print("\n📈 Rolling Averages Preview (Top 5 Defenders by CBIT):")
    defenders = final_data[final_data['position'] == 'DEF'].copy()
    if len(defenders) > 0:
        top_defenders = defenders.nlargest(5, 'cbit_score_avg_5gw')[
            ['name', 'cbit_score', 'cbit_score_avg_3gw', 'cbit_score_avg_5gw', 'defensive_contribution_points']
        ]
        print(top_defenders.round(2))
    
    # Step 6: Feature contract validation
    print("\n6️⃣ Feature Contract Validation")
    print("-" * 32)
    
    expected_features = [
        'clearances_avg_3gw', 'blocks_avg_3gw', 'interceptions_avg_3gw',
        'tackles_won_avg_3gw', 'recoveries_avg_3gw', 'cbit_score_avg_3gw',
        'cbirt_score_avg_3gw', 'defensive_contribution_points_avg_3gw'
    ]
    
    missing_features = [f for f in expected_features if f not in final_data.columns]
    present_features = [f for f in expected_features if f in final_data.columns]
    
    print(f"✅ Present defensive features: {len(present_features)}/{len(expected_features)}")
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    else:
        print("🎉 All expected defensive features are present!")
    
    # Step 7: Save sample data for further testing
    print("\n7️⃣ Saving Sample Data")
    print("-" * 22)
    
    output_path = Path(__file__).parent / "sample_enhanced_defensive_data.csv"
    final_data.to_csv(output_path, index=False)
    print(f"💾 Saved enhanced sample data to: {output_path}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📁 Enhanced data available at: {output_path}")
    print(f"🚀 Ready for model training and integration!")
    
    return final_data


def analyze_defensive_contribution_patterns(df: pd.DataFrame):
    """Additional analysis of defensive contribution patterns"""
    print("\n🔍 DETAILED DEFENSIVE CONTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Minutes vs defensive contributions correlation
    print("⏱️ Minutes vs Defensive Contributions Analysis:")
    minutes_corr = df['minutes'].corr(df['defensive_contribution_points'])
    print(f"   Correlation: {minutes_corr:.3f}")
    
    # Get players who frequently earn defensive contributions
    frequent_contributors = df.groupby(['name', 'position']).agg({
        'defensive_contribution_points': ['sum', 'count', 'mean'],
        'cbit_score': 'mean',
        'cbirt_score': 'mean'
    }).round(3)
    
    frequent_contributors.columns = ['Total_DC_Points', 'Games', 'DC_Rate', 'Avg_CBIT', 'Avg_CBIRT']
    frequent_contributors = frequent_contributors[frequent_contributors['Games'] >= 5]  # Minimum games
    top_contributors = frequent_contributors.nlargest(10, 'Total_DC_Points')
    
    print(f"\n🏆 Top 10 Defensive Contributors (min 5 games):")
    print(top_contributors)
    
    # Position-specific thresholds analysis
    print(f"\n📊 Position-Specific Threshold Analysis:")
    for position in ['DEF', 'MID', 'FWD']:
        pos_data = df[df['position'] == position]
        if len(pos_data) == 0:
            continue
            
        if position == 'DEF':
            score_col = 'cbit_score'
            threshold = 10
        else:
            score_col = 'cbirt_score'
            threshold = 12
            
        above_threshold = pos_data[pos_data[score_col] >= threshold]
        success_rate = len(above_threshold) / len(pos_data) if len(pos_data) > 0 else 0
        
        print(f"   {position}: {success_rate:.1%} of games meet threshold ({threshold}+ {score_col.replace('_', ' ').title()})")


if __name__ == "__main__":
    # Run the complete demo
    enhanced_data = demo_defensive_contributions_pipeline()
    
    # Run additional analysis
    analyze_defensive_contribution_patterns(enhanced_data)
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run the historical data enhancement script with real FBRef data")
    print(f"   2. Train the defensive contributions model")
    print(f"   3. Integrate into the FBRef assembly pipeline")
    print(f"   4. Backtest predictions against actual FPL data")
