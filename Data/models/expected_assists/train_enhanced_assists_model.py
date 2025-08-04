#!/usr/bin/env python3
"""
Enhanced Expected Assists Model with Opponent Strength
Predicts assists by individual players using Poisson regression with fixture difficulty.
Enhanced with historical opponent strength features for realistic fixture-based predictions.
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
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import shared modules
from feature_engineering.player_features import PlayerFeatureEngine, load_historical_data, load_teams_data

# Set style for plots
plt.style.use('default')


def load_enhanced_training_data():
    """Load training data with opponent strength features"""
    print("📊 Loading enhanced training data with opponent strength features...")
    
    # Load the enhanced dataset
    df = pd.read_csv('../raw/parsed_gw.csv', low_memory=False)
    
    # Verify opponent strength columns exist
    required_columns = ['opponent_attack_strength', 'opponent_defence_strength', 
                       'opponent_overall_strength', 'fixture_attractiveness']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing opponent strength columns: {missing_columns}")
    
    # Convert data types safely
    def convert_was_home(val):
        if val == 'True' or val == True or val == 1:
            return 1
        elif val == 'False' or val == False or val == 0:
            return 0
        else:
            return 0  # Default to away
    
    df['was_home'] = df['was_home'].apply(convert_was_home)
    
    print(f"✅ Loaded {len(df):,} records with opponent strength features")
    print(f"📊 Fixture attractiveness range: {df['fixture_attractiveness'].min():.3f} - {df['fixture_attractiveness'].max():.3f}")
    
    return df


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


def prepare_enhanced_assists_features(player_data: dict, feature_engine: PlayerFeatureEngine, 
                                    position_encoder, team_encoder) -> list:
    """
    Prepare features for assists model with opponent strength integration
    
    Enhanced feature set includes:
    - Original 21 features (assists averages, creativity, etc.)
    - NEW: opponent_defence_strength (affects creative opportunities)  
    - NEW: fixture_attractiveness (easier fixtures boost assists)
    """
    
    # Get original assists features (21 features)
    original_features = feature_engine.prepare_assists_model_features(
        player_data, None, position_encoder, team_encoder
    )
    
    # Add opponent strength features (2 new features)
    opponent_defence_strength = player_data.get('opponent_defence_strength', 1150) / 1400.0  # Normalize 0-1
    fixture_attractiveness = player_data.get('fixture_attractiveness', 0.5)
    
    # Enhanced feature set (23 features total)
    enhanced_features = original_features + [
        opponent_defence_strength,  # Stronger defense = fewer assist opportunities
        fixture_attractiveness      # Easier fixture = more assists expected
    ]
    
    return enhanced_features


def train_enhanced_assists_model():
    """Train the enhanced assists model with opponent strength features"""
    
    print("🚀 Training Enhanced Expected Assists Model with Opponent Strength")
    print("=" * 70)
    
    # Load data
    df = load_enhanced_training_data()
    
    # Filter for substantial playing time
    df = filter_training_data(df, minutes_threshold=50)
    
    # Initialize feature engine
    historical_data = load_historical_data('../raw/parsed_gw.csv')
    teams_data = load_teams_data()
    feature_engine = PlayerFeatureEngine(teams_data)    # Prepare encoders
    position_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    
    # Fit encoders on all data
    position_encoder.fit(df['position'].fillna('Unknown'))
    team_encoder.fit(df['team'].fillna('Unknown'))
    
    print(f"📊 Preparing enhanced features for {len(df):,} training samples...")
    
    # Prepare enhanced features and targets
    X_list = []
    y_list = []
    
    for idx, row in df.iterrows():
        try:
            # Convert row to dict
            player_data = row.to_dict()
            
            # Prepare enhanced features (23 features)
            features = prepare_enhanced_assists_features(
                player_data, feature_engine, position_encoder, team_encoder
            )
            
            X_list.append(features)
            y_list.append(row['assists'])
            
        except Exception as e:
            print(f"⚠️ Error processing row {idx}: {e}")
            continue
    
    # Convert to arrays
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✅ Feature preparation complete!")
    print(f"   Training samples: {len(X):,}")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Assists distribution: mean={y.mean():.3f}, std={y.std():.3f}")
    print(f"   Opponent defence strength range: {X[:, -2].min():.3f} - {X[:, -2].max():.3f}")
    print(f"   Fixture attractiveness range: {X[:, -1].min():.3f} - {X[:, -1].max():.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Train enhanced Poisson regression model
    print("🎯 Training enhanced Poisson regression model...")
    
    model = PoissonRegressor(
        alpha=0.1,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n📈 Enhanced Model Performance:")
    print(f"   Training MSE: {train_mse:.4f}")
    print(f"   Testing MSE:  {test_mse:.4f}")
    print(f"   Training MAE: {train_mae:.4f}")
    print(f"   Testing MAE:  {test_mae:.4f}")
    
    # Analyze opponent strength feature impact
    feature_names = [
        'goals_avg_5gw', 'goals_avg_season', 'assists_avg_5gw', 'assists_avg_season',
        'xA_avg_5gw', 'xA_avg_season', 'creativity_avg_5gw', 'creativity_avg_season',
        'minutes_avg_5gw', 'minutes_avg_season', 'selected_avg_5gw', 'selected_avg_season',
        'transfers_in_avg_5gw', 'transfers_out_avg_5gw', 'value_avg_5gw',
        'ict_index_avg_5gw', 'influence_avg_5gw', 'threat_avg_5gw', 'creativity_points_avg_5gw',
        'was_home', 'position_encoded',
        'opponent_defence_strength_normalized',  # Feature 21 (index -2)
        'fixture_attractiveness'                 # Feature 22 (index -1)
    ]
    
    print(f"\n🎯 Enhanced Feature Coefficients (Top 10):")
    coefficients = model.coef_
    feature_importance = list(zip(feature_names, coefficients))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for i, (feature, coef) in enumerate(feature_importance[:10]):
        print(f"   {i+1:2d}. {feature:35} = {coef:8.4f}")
    
    # Highlight opponent strength features
    print(f"\n🏆 Opponent Strength Feature Impact:")
    opponent_defence_coef = coefficients[-2]  # Second to last feature
    fixture_attractiveness_coef = coefficients[-1]  # Last feature
    
    print(f"   opponent_defence_strength_normalized = {opponent_defence_coef:.4f}")
    print(f"   fixture_attractiveness              = {fixture_attractiveness_coef:.4f}")
    
    if opponent_defence_coef < 0:
        print(f"   ✅ Stronger defense reduces assists (expected behavior)")
    else:
        print(f"   ⚠️ Unexpected: stronger defense increases assists")
        
    if fixture_attractiveness_coef > 0:
        print(f"   ✅ Easier fixtures increase assists (expected behavior)")
    else:
        print(f"   ⚠️ Unexpected: easier fixtures reduce assists")
    
    # Save enhanced model
    model_data = {
        'model': model,
        'position_encoder': position_encoder,
        'team_encoder': team_encoder,
        'feature_names': feature_names,
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_mse': test_mse,
        'test_mae': test_mae,
        'enhanced_features': True,
        'opponent_strength_features': ['opponent_defence_strength_normalized', 'fixture_attractiveness']
    }
    
    joblib.dump(model_data, 'enhanced_expected_assists_model.pkl')
    print(f"\n💾 Enhanced model saved as 'enhanced_expected_assists_model.pkl'")
    
    return model_data


if __name__ == "__main__":
    try:
        model_data = train_enhanced_assists_model()
        print(f"\n🎉 Enhanced Expected Assists Model training completed successfully!")
        print(f"   Features: {len(model_data['feature_names'])}")
        print(f"   Test MSE: {model_data['test_mse']:.4f}")
        print(f"   Test MAE: {model_data['test_mae']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
