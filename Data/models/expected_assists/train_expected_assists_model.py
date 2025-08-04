#!/usr/bin/env python3
"""
Expected Assists Model
Predicts assists by individual players using Poisson regression.
Foundation for creative player returns and assist-based bonus points.
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
"""
Expected Assists Model
Predicts assists by individual players using Poisson regression.
Foundation for creative player returns and assist-based bonus points.
"""

import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats

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
    historical_df = pd.read_csv('/Users/owen/src/Personal/fpl-team-picker/Data/raw/parsed_gw_2425.csv', low_memory=False)
    
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
    
    # Load current players from MongoDB
    from database.mongo.mongo_data_loader import load_players_data, load_teams_data
    players_data = load_players_data()
    print(f"✅ Current players: {len(players_data)}")

    # Load teams from MongoDB
    teams_data = load_teams_data()
    print(f"✅ Teams: {len(teams_data)}")
    
    return historical_df, players_data, teams_data

def filter_creative_players(historical_df):
    """Filter to creative players (FWD, MID, DEF) who have assist potential"""
    print("\n🎨 Filtering to creative players...")
    
    # Focus on players who can get assists (all positions except GK)
    creative_df = historical_df[
        historical_df['position'].isin(['FWD', 'MID', 'DEF'])
    ].copy()
    
    # Ensure numeric columns
    numeric_cols = ['assists', 'expected_assists', 'goals_scored', 'expected_goals',
                   'minutes', 'total_points', 'ict_index', 'creativity', 'starts']
    
    for col in numeric_cols:
        creative_df[col] = pd.to_numeric(creative_df[col], errors='coerce').fillna(0)
    
    print(f"✅ Creative players data: {len(creative_df):,} records")
    print(f"✅ Unique players: {creative_df['name'].nunique()}")
    print(f"✅ Position distribution:")
    print(creative_df['position'].value_counts())
    
    return creative_df

def engineer_assist_features(creative_df, feature_engine):
    """Engineer features for assist prediction using shared feature engineering"""
    print("\n🔧 Engineering assist features with shared PlayerFeatureEngine...")
    
    # Define features relevant for assist prediction
    assist_features = [
        'assists', 'expected_assists', 'goals_scored', 'minutes', 'starts',
        'creativity', 'threat', 'expected_goal_involvements', 'bps', 'total_points'
    ]
    
    # Use shared feature engineering for rolling averages with specific features
    creative_df_with_features = feature_engine.calculate_rolling_features(
        creative_df,
        group_col='name',
        sort_cols=['name', 'GW'],
        rolling_features=assist_features
    )
    
    # Position-specific features
    creative_df_with_features['is_midfielder'] = (creative_df_with_features['position'] == 'MID').astype(int)
    creative_df_with_features['is_defender'] = (creative_df_with_features['position'] == 'DEF').astype(int)
    creative_df_with_features['is_forward'] = (creative_df_with_features['position'] == 'FWD').astype(int)
    
    # Team attacking strength (total team assists)
    team_assists = creative_df_with_features.groupby(['team', 'GW'])['assists'].sum().reset_index()
    team_assists.columns = ['team', 'GW', 'team_assists_this_gw']
    creative_df_with_features = creative_df_with_features.merge(team_assists, on=['team', 'GW'], how='left')
    
    # Venue effect for assists
    creative_df_with_features['home_assists_avg_5gw'] = creative_df_with_features[creative_df_with_features['was_home'] == 1].groupby('name')['assists'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    
    creative_df_with_features['away_assists_avg_5gw'] = creative_df_with_features[creative_df_with_features['was_home'] == 0].groupby('name')['assists'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    
    # Fill NaN values with overall averages
    creative_df_with_features['home_assists_avg_5gw'] = creative_df_with_features['home_assists_avg_5gw'].fillna(creative_df_with_features['assists_avg_5gw'])
    creative_df_with_features['away_assists_avg_5gw'] = creative_df_with_features['away_assists_avg_5gw'].fillna(creative_df_with_features['assists_avg_5gw'])
    
    # Assist efficiency (assists vs expected assists)
    creative_df_with_features['assist_efficiency'] = creative_df_with_features['assists_avg_5gw'] / (creative_df_with_features['expected_assists_avg_5gw'] + 0.01)
    
    # Creative output (assists + goals combined)
    creative_df_with_features['creative_output'] = creative_df_with_features['assists_avg_5gw'] + creative_df_with_features['goals_scored_avg_5gw']
    
    # Recent form
    creative_df_with_features['recent_form'] = creative_df_with_features['total_points_avg_3gw']
    
    print("✅ Assist features engineered using shared engine")
    
    return creative_df_with_features

def fit_assists_model(creative_df, feature_engine):
    """Fit Poisson regression model for assists prediction"""
    print("\n📈 Fitting Poisson model for assists...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Remove rows with NaN values and filter to players with some playing time
    model_df = creative_df.dropna().copy()
    
    print(f"✅ Filtered to players with regular playing time: {len(model_df):,} records")
    
    # Get feature columns from shared feature engine
    feature_cols = feature_engine.get_assists_model_feature_columns()
    print(f"✅ Using {len(feature_cols)} features from shared feature engine")
    
    # Add opponent strength features to the dataframe
    if 'opponent_defence_strength' in model_df.columns:
        model_df['opponent_defence_strength_normalized'] = model_df['opponent_defence_strength'] / 1400.0
        print("✅ Added normalized opponent defence strength")
    else:
        model_df['opponent_defence_strength_normalized'] = 0.82  # Default ~1150/1400
        print("⚠️ Using default opponent defence strength")
    
    if 'fixture_attractiveness' not in model_df.columns:
        model_df['fixture_attractiveness'] = 0.5  # Default neutral
        print("⚠️ Using default fixture attractiveness")
    
    # Prepare features and target
    X = model_df[feature_cols].copy()
    y = model_df['assists'].copy()
    
    print(f"✅ Training samples: {len(X):,}")
    print(f"✅ Features: {len(feature_cols)}")
    print(f"✅ Assists distribution:")
    print(y.value_counts().sort_index())
    
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
    
    print(f"\n📊 Expected Assists Model Performance:")
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
    for _, row in feature_importance.head(8).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.3f}")
    
    return poisson_model, scaler, feature_cols

def create_prediction_functions(poisson_model, scaler, feature_cols, feature_engine):
    """Create functions to predict assists for any player"""
    print("\n🎯 Creating prediction functions...")
    
    def predict_player_assists_distribution(player_data, historical_df=None, was_home=True, max_assists=3,
                                           opponent_defence_strength=1100.0, fixture_attractiveness=0.5):
        """
        Predict probability distribution of assists for a player
        
        Args:
            player_data: Dict with current player data
            historical_df: Optional historical data for better rolling averages
            was_home: Whether playing at home
            max_assists: Maximum assists to calculate probabilities for
            opponent_defence_strength: Opponent's defensive strength (raw value)
            fixture_attractiveness: Fixture difficulty score (0-1, higher = easier)
            
        Returns:
            Dict with expected assists and probabilities
        """
        
        # Get historical context if available
        historical_context = None
        if historical_df is not None and 'code' in player_data:
            historical_context = feature_engine.get_historical_context(
                str(player_data['code']), 
                historical_df
            )
        
        # Prepare features using shared logic
        features = feature_engine.prepare_assists_model_features(
            player_data,
            historical_context=historical_context,
            was_home=was_home,
            opponent_defence_strength=opponent_defence_strength,
            fixture_attractiveness=fixture_attractiveness
        )
        
        # Convert to numpy array and scale
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Predict expected assists
        expected_assists = poisson_model.predict(features_scaled)[0]
        
        # Calculate Poisson probabilities
        probabilities = {}
        for assists in range(max_assists + 1):
            prob = stats.poisson.pmf(assists, expected_assists)
            probabilities[assists] = prob
        
        # Probability of getting at least 1 assist
        prob_assist = 1 - probabilities[0]
        
        return {
            'expected_assists': expected_assists,
            'probabilities': probabilities,
            'prob_assist': prob_assist
        }
    
    return predict_player_assists_distribution

def save_model(poisson_model, scaler, feature_cols, feature_engine):
    """Save the trained model and associated objects"""
    print("\n💾 Saving expected assists model...")
    
    model_data = {
        'model': poisson_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_engine': feature_engine,
        'model_type': 'expected_assists_poisson',
        'trained_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, 'expected_assists_model.pkl')
    print("✅ Model saved to expected_assists_model.pkl")
    print("✅ Shared feature engine included in model data")

def demonstrate_predictions(predict_fn):
    """Demonstrate model predictions"""
    print("\n🎯 Example Player Assists Predictions:\n")
    
    # Example predictions for different player types
    examples = [
        {
            'name': 'Creative Midfielder',
            'stats': {
                'assists_avg_3gw': 0.5, 'assists_avg_5gw': 0.4, 'assists_avg_10gw': 0.3,
                'expected_assists_avg_3gw': 0.6, 'expected_assists_avg_5gw': 0.5, 'expected_assists_avg_10gw': 0.4,
                'goals_scored_avg_3gw': 0.2, 'goals_scored_avg_5gw': 0.2,
                'minutes_avg_3gw': 80, 'minutes_avg_5gw': 75,
                'starts_avg_3gw': 0.9, 'starts_avg_5gw': 0.8,
                'creativity_avg_3gw': 60, 'creativity_avg_5gw': 55,
                'assist_efficiency': 1.1, 'creative_output': 0.6, 'recent_form': 7.0,
                'is_midfielder': 1, 'is_defender': 0, 'is_forward': 0, 'was_home': 1
            }
        },
        {
            'name': 'Wing-Back Defender',
            'stats': {
                'assists_avg_3gw': 0.2, 'assists_avg_5gw': 0.15, 'assists_avg_10gw': 0.1,
                'xa_avg_3gw': 0.25, 'xa_avg_5gw': 0.2, 'xa_avg_10gw': 0.15,
                'goals_avg_3gw': 0.05, 'goals_avg_5gw': 0.05,
                'minutes_avg_3gw': 85, 'minutes_avg_5gw': 80,
                'starts_avg_3gw': 1.0, 'starts_avg_5gw': 0.9,
                'creativity_avg_3gw': 25, 'creativity_avg_5gw': 20,
                'assist_efficiency': 0.8, 'creative_output': 0.2, 'recent_form': 5.0,
                'is_midfielder': 0, 'is_defender': 1, 'is_forward': 0, 'was_home': 0
            }
        },
        {
            'name': 'Supporting Forward',
            'stats': {
                'assists_avg_3gw': 0.3, 'assists_avg_5gw': 0.25, 'assists_avg_10gw': 0.2,
                'xa_avg_3gw': 0.35, 'xa_avg_5gw': 0.3, 'xa_avg_10gw': 0.25,
                'goals_avg_3gw': 0.6, 'goals_avg_5gw': 0.5,
                'minutes_avg_3gw': 75, 'minutes_avg_5gw': 70,
                'starts_avg_3gw': 0.8, 'starts_avg_5gw': 0.7,
                'creativity_avg_3gw': 35, 'creativity_avg_5gw': 30,
                'assist_efficiency': 0.9, 'creative_output': 0.75, 'recent_form': 6.5,
                'is_midfielder': 0, 'is_defender': 0, 'is_forward': 1, 'was_home': 1
            }
        }
    ]
    
    for example in examples:
        pred = predict_fn(example['stats'])
        print(f"{example['name']}:")
        print(f"  Expected assists: {pred['expected_assists']:.3f}")
        print(f"  Probability to assist: {pred['prob_assist']:.3f}")
        print(f"  Assist probabilities:")
        for assists, prob in pred['probabilities'].items():
            print(f"    {assists} assists: {prob:.3f}")
        print()

def main():
    """Main training pipeline"""
    print("🚀 Starting Expected Assists Model Training Pipeline\n")
    
    # Load data
    historical_df, players_data, teams_data = load_data()
    
    # Initialize shared feature engine with teams data
    feature_engine = PlayerFeatureEngine(teams_data)
    
    # Filter to creative players
    creative_df = filter_creative_players(historical_df)
    
    # Engineer features using shared feature engineering
    creative_df = engineer_assist_features(creative_df, feature_engine)
    
    # Fit Poisson models
    poisson_model, scaler, feature_cols = fit_assists_model(creative_df, feature_engine)
    
    # Create prediction functions
    predict_fn = create_prediction_functions(poisson_model, scaler, feature_cols, feature_engine)
    
    # Save model with shared feature engine
    save_model(poisson_model, scaler, feature_cols, feature_engine)
    
    # Demonstrate predictions
    demonstrate_predictions(predict_fn)
    
    print("🎯 Expected Assists Model training completed!")
    print("Ready to predict assist probabilities for creative players!")
    print("\n📋 Model provides:")
    print("  • Expected assists per match")
    print("  • Probability to get at least 1 assist")
    print("  • Full probability distribution (0, 1, 2+ assists)")

if __name__ == "__main__":
    main()
