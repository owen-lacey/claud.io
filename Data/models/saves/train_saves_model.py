#!/usr/bin/env python3
"""
Goalkeeper Saves Model
Predicts saves by goalkeepers using Poisson regression.
Key component for goalkeeper scoring predictions and clean sheet probabilities.
"""

import pandas as pd
import numpy as np
import json
import sys
import os

# Add parent directory to Python path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

# Add the parent directory to the path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering.player_features import PlayerFeatureEngine, load_historical_data, load_teams_data

# Set style for plots
plt.style.use('default')


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
    def convert_was_home(val):
        if val == 'True' or val == True or val == 1:
            return 1
        elif val == 'False' or val == False or val == 0:
            return 0
        else:
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
    
    return historical_df, players_data, teams_data

def filter_goalkeepers(historical_df):
    """Filter to goalkeepers only"""
    print("\n🥅 Filtering to goalkeepers...")
    
    # Focus on goalkeepers only
    gk_df = historical_df[
        historical_df['position'] == 'GK'
    ].copy()
    
    # Ensure numeric columns
    numeric_cols = ['saves', 'goals_conceded', 'clean_sheets', 'minutes', 
                   'total_points', 'bonus', 'starts', 'expected_goals_conceded']
    
    for col in numeric_cols:
        gk_df[col] = pd.to_numeric(gk_df[col], errors='coerce').fillna(0)
    
    print(f"✅ Goalkeeper data: {len(gk_df):,} records")
    print(f"✅ Unique goalkeepers: {gk_df['name'].nunique()}")
    
    return gk_df

def engineer_saves_features(gk_df, feature_engine):
    """Engineer features for saves prediction using shared feature engineering"""
    print("\n🔧 Engineering saves features with shared PlayerFeatureEngine...")
    
    # Define features relevant for saves prediction
    saves_features = [
        'saves', 'goals_conceded', 'clean_sheets', 'expected_goals_conceded',
        'minutes', 'starts', 'total_points', 'bonus'
    ]
    
    # Use shared feature engineering for rolling averages with specific features
    gk_df_with_features = feature_engine.calculate_rolling_features(
        gk_df,
        group_col='name',
        sort_cols=['name', 'GW'],
        rolling_features=saves_features
    )
    
    # Team defensive context (aggregate all team GK stats per gameweek)
    team_defense = gk_df_with_features.groupby(['team', 'GW']).agg({
        'goals_conceded': 'first',  # Same for all GKs in team
        'saves': 'sum',  # Total team saves
        'clean_sheets': 'first'  # Same for all GKs in team
    }).reset_index()
    
    team_defense.columns = ['team', 'GW', 'team_goals_conceded', 'team_saves', 'team_clean_sheet']
    gk_df_with_features = gk_df_with_features.merge(team_defense, on=['team', 'GW'], how='left')
    
    # Venue effect for saves
    gk_df_with_features['home_saves_avg_5gw'] = gk_df_with_features[gk_df_with_features['was_home'] == 1].groupby('name')['saves'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    
    gk_df_with_features['away_saves_avg_5gw'] = gk_df_with_features[gk_df_with_features['was_home'] == 0].groupby('name')['saves'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    
    # Fill NaN values with overall averages
    gk_df_with_features['home_saves_avg_5gw'] = gk_df_with_features['home_saves_avg_5gw'].fillna(gk_df_with_features['saves_avg_5gw'])
    gk_df_with_features['away_saves_avg_5gw'] = gk_df_with_features['away_saves_avg_5gw'].fillna(gk_df_with_features['saves_avg_5gw'])
    
    # Save efficiency (saves per goal conceded)
    gk_df_with_features['save_efficiency'] = gk_df_with_features['saves_avg_5gw'] / (gk_df_with_features['goals_conceded_avg_5gw'] + 0.1)
    
    # Defensive workload (expected goals conceded - actual goals conceded)
    gk_df_with_features['defensive_workload'] = gk_df_with_features['expected_goals_conceded_avg_5gw'] - gk_df_with_features['goals_conceded_avg_5gw']
    
    # Team defensive strength (inverse of goals conceded)
    gk_df_with_features['team_defensive_strength'] = 1 / (gk_df_with_features['goals_conceded_avg_5gw'] + 0.1)
    
    # Recent form
    gk_df_with_features['recent_form'] = gk_df_with_features['total_points_avg_3gw']
    
    print("✅ Saves features engineered using shared engine")
    
    return gk_df_with_features

def fit_saves_model(gk_df):
    """Fit Poisson regression model for saves prediction"""
    print("\n📈 Fitting Poisson model for saves...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Remove rows with NaN values and filter to goalkeepers with playing time
    model_df = gk_df.dropna().copy()
    model_df = model_df[model_df['minutes_avg_5gw'] > 30]  # Filter to regular playing GKs
    
    print(f"✅ Filtered to goalkeepers with regular playing time: {len(model_df):,} records")
    
    # Define features for the model (using shared feature engineering column names)
    feature_cols = [
        'saves_avg_3gw', 'saves_avg_5gw', 'saves_avg_10gw',
        'goals_conceded_avg_3gw', 'goals_conceded_avg_5gw', 'goals_conceded_avg_10gw',
        'expected_goals_conceded_avg_3gw', 'expected_goals_conceded_avg_5gw', 'expected_goals_conceded_avg_10gw',
        'clean_sheets_avg_3gw', 'clean_sheets_avg_5gw',
        'minutes_avg_3gw', 'minutes_avg_5gw',
        'starts_avg_3gw', 'starts_avg_5gw',
        'save_efficiency', 'defensive_workload', 'team_defensive_strength',
        'recent_form', 'was_home',
        # NEW: Opponent strength features for fixture difficulty
        'opponent_attack_strength_normalized', 'fixture_difficulty_inverted'
    ]
    
    # Add opponent strength features to the dataframe
    if 'opponent_attack_strength' in model_df.columns:
        model_df['opponent_attack_strength_normalized'] = model_df['opponent_attack_strength'] / 1400.0
        print("✅ Added normalized opponent attack strength")
    else:
        model_df['opponent_attack_strength_normalized'] = 0.82  # Default ~1150/1400
        print("⚠️ Using default opponent attack strength")
    
    if 'fixture_attractiveness' in model_df.columns:
        # For saves, we invert fixture attractiveness - harder fixtures = more saves needed
        model_df['fixture_difficulty_inverted'] = 1.0 - model_df['fixture_attractiveness']
        print("✅ Added inverted fixture difficulty (harder fixtures = more saves)")
    else:
        model_df['fixture_difficulty_inverted'] = 0.5  # Default neutral
        print("⚠️ Using default fixture difficulty")
    
    # Prepare features and target
    X = model_df[feature_cols].copy()
    y = model_df['saves'].copy()
    
    print(f"✅ Training samples: {len(X):,}")
    print(f"✅ Features: {len(feature_cols)}")
    print(f"✅ Saves distribution:")
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
    
    print(f"\n📊 Saves Model Performance:")
    print(f"  Training MSE: {train_mse:.3f}")
    print(f"  Test MSE: {test_mse:.3f}")
    print(f"  Training MAE: {train_mae:.3f}")
    print(f"  Test MAE: {test_mae:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
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
        elif row['feature'] == 'fixture_difficulty_inverted':
            fixture_difficulty_coef = row['coefficient']
            print(f"   fixture_difficulty_inverted        = {fixture_difficulty_coef:.4f}")
    
    if opponent_attack_coef is not None:
        if opponent_attack_coef > 0:
            print(f"   ✅ Stronger attack increases saves (expected behavior)")
        else:
            print(f"   ⚠️ Unexpected: stronger attack reduces saves")
            
    if fixture_difficulty_coef is not None:
        if fixture_difficulty_coef > 0:
            print(f"   ✅ Harder fixtures increase saves (expected behavior)")
        else:
            print(f"   ⚠️ Unexpected: harder fixtures reduce saves")
    
    return poisson_model, scaler, feature_cols

def create_prediction_functions(poisson_model, scaler, feature_cols):
    """Create functions to predict saves for any goalkeeper"""
    print("\n🎯 Creating prediction functions...")
    
    def predict_goalkeeper_saves_distribution(gk_stats, max_saves=10):
        """
        Predict probability distribution of saves for a goalkeeper
        
        Args:
            gk_stats: Dict with goalkeeper recent stats
            max_saves: Maximum saves to calculate probabilities for
            
        Returns:
            Dict with expected saves and probabilities
        """
        
        # Prepare features
        features = np.array([[
            gk_stats.get('saves_avg_3gw', 2.0),
            gk_stats.get('saves_avg_5gw', 2.0),
            gk_stats.get('saves_avg_10gw', 2.0),
            gk_stats.get('goals_conceded_avg_3gw', 1.2),
            gk_stats.get('goals_conceded_avg_5gw', 1.2),
            gk_stats.get('goals_conceded_avg_10gw', 1.2),
            gk_stats.get('expected_goals_conceded_avg_3gw', 1.3),
            gk_stats.get('expected_goals_conceded_avg_5gw', 1.3),
            gk_stats.get('expected_goals_conceded_avg_10gw', 1.3),
            gk_stats.get('clean_sheets_avg_3gw', 0.3),
            gk_stats.get('clean_sheets_avg_5gw', 0.3),
            gk_stats.get('minutes_avg_3gw', 85),
            gk_stats.get('minutes_avg_5gw', 85),
            gk_stats.get('starts_avg_3gw', 1.0),
            gk_stats.get('starts_avg_5gw', 1.0),
            gk_stats.get('save_efficiency', 2.0),
            gk_stats.get('defensive_workload', 0.1),
            gk_stats.get('team_defensive_strength', 0.8),
            gk_stats.get('recent_form', 4.0),
            gk_stats.get('was_home', 1),
            # NEW: Opponent strength features
            gk_stats.get('opponent_attack_strength_normalized', 0.82),  # Default ~1150/1400
            gk_stats.get('fixture_difficulty_inverted', 0.5)  # Default neutral
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict expected saves
        expected_saves = poisson_model.predict(features_scaled)[0]
        
        # Calculate Poisson probabilities
        probabilities = {}
        for saves in range(max_saves + 1):
            prob = stats.poisson.pmf(saves, expected_saves)
            probabilities[saves] = prob
        
        # High saves probability (4+ saves)
        prob_high_saves = sum(probabilities[i] for i in range(4, max_saves + 1))
        
        return {
            'expected_saves': expected_saves,
            'probabilities': probabilities,
            'prob_high_saves': prob_high_saves
        }
    
    return predict_goalkeeper_saves_distribution

def save_model(poisson_model, scaler, feature_cols, feature_engine):
    """Save the trained model and associated objects"""
    print("\n💾 Saving saves model...")
    
    model_data = {
        'model': poisson_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_engine': feature_engine,
        'model_type': 'saves_poisson',
        'trained_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, 'saves_model.pkl')
    print("✅ Model saved to saves_model.pkl")
    print("✅ Shared feature engine included in model data")

def demonstrate_predictions(predict_fn):
    """Demonstrate model predictions"""
    print("\n🎯 Example Goalkeeper Saves Predictions:\n")
    
    # Example predictions for different goalkeeper types
    examples = [
        {
            'name': 'Premium Goalkeeper (Strong Defense)',
            'stats': {
                'saves_avg_3gw': 3.5, 'saves_avg_5gw': 3.2, 'saves_avg_10gw': 3.0,
                'goals_conceded_avg_3gw': 0.8, 'goals_conceded_avg_5gw': 0.9, 'goals_conceded_avg_10gw': 1.0,
                'expected_goals_conceded_avg_3gw': 1.0, 'expected_goals_conceded_avg_5gw': 1.1, 'expected_goals_conceded_avg_10gw': 1.2,
                'clean_sheets_avg_3gw': 0.5, 'clean_sheets_avg_5gw': 0.4,
                'minutes_avg_3gw': 90, 'minutes_avg_5gw': 90,
                'starts_avg_3gw': 1.0, 'starts_avg_5gw': 1.0,
                'save_efficiency': 4.0, 'defensive_workload': 0.2, 'team_defensive_strength': 1.2,
                'recent_form': 6.0, 'was_home': 1
            }
        },
        {
            'name': 'Budget Goalkeeper (Weak Defense)',
            'stats': {
                'saves_avg_3gw': 4.2, 'saves_avg_5gw': 4.5, 'saves_avg_10gw': 4.3,
                'goals_conceded_avg_3gw': 1.8, 'goals_conceded_avg_5gw': 1.9, 'goals_conceded_avg_10gw': 1.8,
                'xgc_avg_3gw': 2.0, 'xgc_avg_5gw': 2.1, 'xgc_avg_10gw': 2.0,
                'clean_sheets_avg_3gw': 0.1, 'clean_sheets_avg_5gw': 0.15,
                'minutes_avg_3gw': 90, 'minutes_avg_5gw': 90,
                'starts_avg_3gw': 1.0, 'starts_avg_5gw': 1.0,
                'save_efficiency': 2.4, 'defensive_workload': 0.2, 'team_defensive_strength': 0.5,
                'recent_form': 3.0, 'was_home': 0
            }
        }
    ]
    
    for example in examples:
        pred = predict_fn(example['stats'])
        print(f"{example['name']}:")
        print(f"  Expected saves: {pred['expected_saves']:.2f}")
        print(f"  Probability of 4+ saves: {pred['prob_high_saves']:.3f}")
        print(f"  Save probabilities:")
        for saves in range(8):
            if saves in pred['probabilities']:
                print(f"    {saves} saves: {pred['probabilities'][saves]:.3f}")
        print()

def main():
    """Main training pipeline"""
    print("🚀 Starting Saves Model Training Pipeline\n")
    
    # Load data
    historical_df, players_data, teams_data = load_data()
    
    # Initialize shared feature engine with teams data
    feature_engine = PlayerFeatureEngine(teams_data)
    
    # Filter to goalkeepers
    gk_df = filter_goalkeepers(historical_df)
    
    # Engineer features using shared feature engineering
    gk_df = engineer_saves_features(gk_df, feature_engine)
    
    # Fit Poisson models
    poisson_model, scaler, feature_cols = fit_saves_model(gk_df)
    
    # Create prediction functions
    predict_fn = create_prediction_functions(poisson_model, scaler, feature_cols)
    
    # Save model with shared feature engine
    save_model(poisson_model, scaler, feature_cols, feature_engine)
    
    # Demonstrate predictions
    demonstrate_predictions(predict_fn)
    
    print("🎯 Saves Model training completed!")
    print("Ready to predict save probabilities for goalkeepers!")
    print("\n📋 Model provides:")
    print("  • Expected saves per match")
    print("  • Probability of high save counts (4+ saves)")
    print("  • Full probability distribution for saves")
    print("  • Integration with team defensive strength")

if __name__ == "__main__":
    main()
