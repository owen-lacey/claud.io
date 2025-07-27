#!/usr/bin/env python3
"""
Team Goals Conceded Model
Predicts goals conceded by each team for upcoming fixtures using Poisson distribution.
Foundation for clean sheet and defensive returns predictions.
"""

import pandas as pd
import numpy as np
import json
import joblib
import sys
import os

# Add parent directory to Python path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from feature_engineering.player_features import PlayerFeatureEngine, load_historical_data, load_teams_data

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def filter_training_data(df, minutes_threshold=450):
    """
    Filter training data to players with substantial playing time
    
    Args:
        df: Training dataframe
        minutes_threshold: Minimum minutes played (default: 450 = ~5 games)
    
    Returns:
        Filtered dataframe with only players who have substantial sample size
    """
    
    print(f"📊 Filtering training data...")
    print(f"   Original samples: {len(df):,}")
    
    # Filter based on average minutes per gameweek
    if 'minutes_avg_5gw' in df.columns:
        # Use rolling average if available (better indicator)
        filtered_df = df[df['minutes_avg_5gw'] >= (minutes_threshold / 5)]
        print(f"   Filtered by avg_minutes >= {minutes_threshold/5:.0f} per game")
    elif 'minutes' in df.columns:
        # Fall back to total minutes
        filtered_df = df[df['minutes'] >= minutes_threshold]
        print(f"   Filtered by total_minutes >= {minutes_threshold}")
    else:
        print(f"   ⚠️ No minutes column found, using all data (RISKY)")
        filtered_df = df
    
    print(f"   Filtered samples: {len(filtered_df):,}")
    print(f"   Removed: {len(df) - len(filtered_df):,} low-minutes players")
    print(f"   Retention rate: {len(filtered_df)/len(df)*100:.1f}%")
    
    return filtered_df


def load_data():
    """Load all required data files"""
    print("📊 Loading training data...")
    
    # Load the parsed GW data with features
    historical_df = pd.read_csv('/Users/owen/src/Personal/fpl-team-picker/Data/raw/parsed_gw.csv', low_memory=False)
    
    # Convert data types safely
    # Handle was_home column which has mixed string/numeric values
    def convert_was_home(val):
        if val == 'True' or val == True or val == 1:
            return 1
        elif val == 'False' or val == False or val == 0:
            return 0
        else:
            # For other values, assume it's a fixture ID and we can't determine home/away
            return 0  # Default to away
    
    historical_df['was_home'] = historical_df['was_home'].apply(convert_was_home)
    historical_df['GW'] = pd.to_numeric(historical_df['GW'], errors='coerce').fillna(1).astype(int)
    
    print(f"✅ Historical data: {len(historical_df):,} records with opponent strength features")
    
    # Verify opponent strength columns exist
    required_columns = ['opponent_attack_strength', 'opponent_defence_strength', 
                       'opponent_overall_strength', 'fixture_attractiveness']
    
    missing_columns = [col for col in required_columns if col not in historical_df.columns]
    if missing_columns:
        print(f"⚠️ Missing opponent strength columns: {missing_columns}")
    else:
        print(f"✅ Opponent strength features available")
    
    # Load current players
    with open('/Users/owen/src/Personal/fpl-team-picker/Data/database/players.json', 'r') as f:
        players_data = json.load(f)
    print(f"✅ Current players: {len(players_data)}")
    
    # Load teams
    with open('/Users/owen/src/Personal/fpl-team-picker/Data/database/teams.json', 'r') as f:
        teams_data = json.load(f)
    print(f"✅ Teams: {len(teams_data)}")
    
    # Load fixtures
    with open('/Users/owen/src/Personal/fpl-team-picker/Data/database/fixtures.json', 'r') as f:
        fixtures_data = json.load(f)
    print(f"✅ Fixtures: {len(fixtures_data)}")
    
    return historical_df, players_data, teams_data, fixtures_data

def create_team_defensive_stats(historical_df, teams_data):
    """
    Create comprehensive team defensive statistics from historical data
    """
    print("\n🛡️  Calculating team defensive statistics...")
    
    # Create team mapping
    team_mapping = {team['id']: team['name'] for team in teams_data}
    
    # Group by team and gameweek to get team-level stats
    team_stats = []
    
    for team_id, team_name in team_mapping.items():
        print(f"Processing {team_name}...")
        
        # Get all players from this team
        team_players = historical_df[historical_df['team'] == team_name].copy()
        
        if len(team_players) == 0:
            continue
        
        # Ensure numeric columns
        numeric_cols = ['goals_conceded', 'clean_sheets', 'total_points', 'minutes', 'goals_scored', 'assists', 'saves']
        for col in numeric_cols:
            team_players[col] = pd.to_numeric(team_players[col], errors='coerce').fillna(0)
            
        # Group by gameweek to get team performance per gameweek
        gw_stats = team_players.groupby(['GW']).agg({
            'goals_conceded': 'first',  # Same for all players in team
            'clean_sheets': 'first',    # Same for all players in team
            'was_home': 'first',        # Same for all players in team
            'total_points': 'sum',      # Team total points
            'minutes': 'sum',           # Total team minutes
            'goals_scored': 'sum',      # Team goals scored
            'assists': 'sum',           # Team assists
            'saves': 'sum',            # Team saves
        }).reset_index()
        
        gw_stats['team_id'] = team_id
        gw_stats['team_name'] = team_name
        team_stats.append(gw_stats)
    
    # Combine all team stats
    team_df = pd.concat(team_stats, ignore_index=True)
    
    print(f"✅ Team defensive stats created: {len(team_df)} team-gameweek records")
    
    return team_df

def engineer_defensive_features(team_df, feature_engine):
    """
    Engineer features for goals conceded prediction using shared feature engine
    
    Args:
        team_df: Team defensive statistics DataFrame
        feature_engine: Shared PlayerFeatureEngine instance
    
    Returns:
        DataFrame with engineered features
    """
    return feature_engine.calculate_team_rolling_features(team_df)

def fit_poisson_models(team_df, feature_cols):
    """
    Fit Poisson regression models for goals conceded prediction
    
    Args:
        team_df: Team data with engineered features
        feature_cols: List of feature column names from shared engine
    """
    print("\n📈 Fitting Poisson models for goals conceded...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Remove rows with NaN values
    model_df = team_df.dropna().copy()
    
    # Add opponent strength features to the dataframe
    if 'opponent_attack_strength' in model_df.columns:
        model_df['opponent_attack_strength_normalized'] = model_df['opponent_attack_strength'] / 1400.0
        print("✅ Added normalized opponent attack strength")
    else:
        model_df['opponent_attack_strength_normalized'] = 0.82  # Default ~1150/1400
        print("⚠️ Using default opponent attack strength")
    
    if 'fixture_attractiveness' in model_df.columns:
        # For goals conceded, we use fixture difficulty - harder fixtures = more goals conceded
        model_df['fixture_difficulty'] = 1.0 - model_df['fixture_attractiveness']
        print("✅ Added fixture difficulty (harder fixtures = more goals conceded)")
    else:
        model_df['fixture_difficulty'] = 0.5  # Default neutral
        print("⚠️ Using default fixture difficulty")
    
    # Add was_home to features if not already included
    model_features = feature_cols.copy()
    if 'was_home' not in model_features:
        model_features.append('was_home')
    
    # Add opponent strength features to the feature list
    enhanced_feature_cols = model_features + [
        'opponent_attack_strength_normalized',
        'fixture_difficulty'
    ]
    
    # Prepare features and target
    X = model_df[enhanced_feature_cols].copy()
    y = model_df['goals_conceded'].copy()
    
    print(f"✅ Training samples: {len(X):,}")
    print(f"✅ Features: {len(enhanced_feature_cols)} (including {len(enhanced_feature_cols) - len(model_features)} opponent strength features)")
    print(f"✅ Goals conceded distribution:")
    print(y.describe())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"✅ Training set: {len(X_train):,}")
    print(f"✅ Test set: {len(X_test):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit Poisson regression
    poisson_model = PoissonRegressor(
        alpha=1.0,  # L2 regularization
        max_iter=1000,
        fit_intercept=True
    )
    
    poisson_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = poisson_model.predict(X_train_scaled)
    y_test_pred = poisson_model.predict(X_test_scaled)
    
    # Evaluate model
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n📊 Poisson Model Performance:")
    print(f"  Training MSE: {train_mse:.3f}")
    print(f"  Test MSE: {test_mse:.3f}")
    print(f"  Training MAE: {train_mae:.3f}")
    print(f"  Test MAE: {test_mae:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': enhanced_feature_cols,
        'coefficient': poisson_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\n🎯 Feature Coefficients:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.3f}")
    
    # Highlight opponent strength features
    print(f"\n🏆 Opponent Strength Feature Impact:")
    opponent_attack_coef = None
    fixture_difficulty_coef = None
    
    for _, row in feature_importance.iterrows():
        if row['feature'] == 'opponent_attack_strength_normalized':
            opponent_attack_coef = row['coefficient']
            print(f"   opponent_attack_strength_normalized = {opponent_attack_coef:.4f}")
        elif row['feature'] == 'fixture_difficulty':
            fixture_difficulty_coef = row['coefficient']
            print(f"   fixture_difficulty                   = {fixture_difficulty_coef:.4f}")
    
    if opponent_attack_coef is not None:
        if opponent_attack_coef > 0:
            print(f"   ✅ Stronger attacks increase goals conceded (expected behavior)")
        else:
            print(f"   ⚠️ Unexpected: stronger attacks reduce goals conceded")
            
    if fixture_difficulty_coef is not None:
        if fixture_difficulty_coef > 0:
            print(f"   ✅ Harder fixtures increase goals conceded (expected behavior)")
        else:
            print(f"   ⚠️ Unexpected: harder fixtures reduce goals conceded")
    
    return poisson_model, scaler, enhanced_feature_cols

def create_prediction_functions(poisson_model, scaler, feature_cols, teams_data):
    """
    Create functions to predict goals conceded for any team
    """
    print("\n🎯 Creating prediction functions...")
    
    def predict_goals_conceded_distribution(team_id, is_home, recent_stats, max_goals=5):
        """
        Predict probability distribution of goals conceded for a team
        
        Args:
            team_id: Team ID
            is_home: 1 if home, 0 if away
            recent_stats: Dict with recent team stats
            max_goals: Maximum goals to calculate probabilities for
            
        Returns:
            Dict with probabilities for 0, 1, 2, ... max_goals
        """
        
        # Prepare features in the same order as training
        feature_values = []
        for col in feature_cols:
            if col == 'was_home':
                feature_values.append(is_home)
            elif col == 'opponent_attack_strength_normalized':
                feature_values.append(recent_stats.get('opponent_attack_strength_normalized', 0.82))  # Default ~1150/1400
            elif col == 'fixture_difficulty':
                feature_values.append(recent_stats.get('fixture_difficulty', 0.5))  # Default neutral
            else:
                # Use appropriate defaults for each feature type
                if 'goals_conceded' in col:
                    default_val = 1.0
                elif 'clean_sheets' in col:
                    default_val = 0.3
                elif 'goals_scored' in col:
                    default_val = 1.0
                elif 'total_points' in col:
                    default_val = 20.0
                elif col == 'defensive_strength':
                    default_val = 1.0
                elif col == 'recent_form':
                    default_val = 20.0
                elif col == 'season_progress':
                    default_val = 15
                else:
                    default_val = 0.0
                    
                feature_values.append(recent_stats.get(col, default_val))
        
        features = np.array([feature_values])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict expected goals conceded
        expected_goals = poisson_model.predict(features_scaled)[0]
        
        # Calculate Poisson probabilities
        probabilities = {}
        for goals in range(max_goals + 1):
            prob = stats.poisson.pmf(goals, expected_goals)
            probabilities[goals] = prob
        
        return {
            'expected_goals_conceded': expected_goals,
            'probabilities': probabilities,
            'clean_sheet_probability': probabilities[0]
        }
    
    def get_team_name(team_id):
        """Get team name from team ID"""
        for team in teams_data:
            if team['id'] == team_id:
                return team['name']
        return f"Team_{team_id}"
    
    return predict_goals_conceded_distribution, get_team_name

def save_model(poisson_model, scaler, feature_cols, feature_engine):
    """Save the trained model and associated objects"""
    print("\n💾 Saving team goals conceded model...")
    
    model_data = {
        'model': poisson_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_engine': feature_engine,
        'model_type': 'team_goals_conceded_poisson',
        'trained_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, 'team_goals_conceded_model.pkl')
    print("✅ Model saved to team_goals_conceded_model.pkl")
    print("✅ Shared feature engine included in model data")

def demonstrate_predictions(predict_fn, get_team_name_fn, teams_data):
    """Demonstrate model predictions"""
    print("\n🎯 Example Team Goals Conceded Predictions:\n")
    
    # Example predictions for different teams
    example_teams = teams_data[:3]  # First 3 teams
    
    for team in example_teams:
        team_id = team['id']
        team_name = get_team_name_fn(team_id)
        
        # Example recent stats (would come from actual data in production)
        example_stats = {
            'goals_conceded_avg_3gw': 1.2,
            'goals_conceded_avg_5gw': 1.1,
            'goals_conceded_avg_10gw': 1.3,
            'clean_sheets_avg_3gw': 0.33,
            'clean_sheets_avg_5gw': 0.4,
            'goals_scored_avg_3gw': 1.8,
            'goals_scored_avg_5gw': 1.6,
            'defensive_strength': 0.8,
            'recent_form': 25.0,
            'season_progress': 15
        }
        
        # Home prediction
        home_pred = predict_fn(team_id, 1, example_stats)
        print(f"{team_name} (Home):")
        print(f"  Expected goals conceded: {home_pred['expected_goals_conceded']:.2f}")
        print(f"  Clean sheet probability: {home_pred['clean_sheet_probability']:.3f}")
        
        # Away prediction
        away_pred = predict_fn(team_id, 0, example_stats)
        print(f"{team_name} (Away):")
        print(f"  Expected goals conceded: {away_pred['expected_goals_conceded']:.2f}")
        print(f"  Clean sheet probability: {away_pred['clean_sheet_probability']:.3f}")
        print()

def main(feature_engine=None):
    """Main training pipeline with shared feature engineering"""
    print("🚀 Starting Team Goals Conceded Model Training Pipeline\n")
    
    # Load data
    historical_df, players_data, teams_data, fixtures_data = load_data()
    
    # Initialize shared feature engine if not provided
    if feature_engine is None:
        feature_engine = PlayerFeatureEngine(teams_data)
    
    # Create team defensive statistics
    team_df = create_team_defensive_stats(historical_df, teams_data)
    
    # Engineer features using shared engine
    team_df = engineer_defensive_features(team_df, feature_engine)
    
    # Get feature columns from shared engine
    feature_cols = feature_engine.get_team_feature_columns()
    
    # Fit Poisson models
    poisson_model, scaler, enhanced_feature_cols = fit_poisson_models(team_df, feature_cols)
    
    # Create prediction functions
    predict_fn, get_team_name_fn = create_prediction_functions(
        poisson_model, scaler, enhanced_feature_cols, teams_data
    )
    
    # Save model with shared feature engine
    save_model(poisson_model, scaler, enhanced_feature_cols, feature_engine)
    
    # Demonstrate predictions
    demonstrate_predictions(predict_fn, get_team_name_fn, teams_data)
    
    print("🎯 Team Goals Conceded Model training completed!")
    print("Ready to predict goals conceded probabilities for any team!")
    print("✅ Shared feature engineering ensures consistent features between training and prediction")
    print("\n📋 Model provides:")
    print("  • Expected goals conceded")
    print("  • Clean sheet probability")
    print("  • Full probability distribution (0, 1, 2+ goals)")
    print("  • Foundation for all defensive player clean sheet bonuses")
    print("\n🎉 ALL 6 FPL MODELS ENHANCED WITH OPPONENT STRENGTH!")
    print("🚀 Complete fixture-aware prediction system ready!")
    print("✅ Goals, Assists, Minutes, Saves, Yellow Cards, Goals Conceded")
    print("🎯 Ready for Stage 3: Integration and FPL scoring system!")

if __name__ == "__main__":
    main()
