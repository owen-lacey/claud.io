#!/usr/bin/env python3
"""
Expected Goals Model
Predicts goals scored by individual players using Poisson regression.
Foundation for attacking returns and bonus points predictions.
"""

# Add parent directory to Python path for shared modules
import sys
import os
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)

import pandas as pd
import numpy as np
import json
import joblib

from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add shared feature engineering module
from feature_engineering.player_features import PlayerFeatureEngine, load_historical_data, load_teams_data

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load all required data files using shared utilities"""
    print("📊 Loading data...")
    
    # Load ENHANCED data with opponent strength features
    historical_df = load_historical_data('/Users/owen/src/Personal/fpl-team-picker/Data/raw/parsed_gw_2425.csv')
    teams_data = load_teams_data('/Users/owen/src/Personal/fpl-team-picker/Data/database/teams.json')
    
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
    
    # Load current players
    with open('/Users/owen/src/Personal/fpl-team-picker/Data/database/players.json', 'r') as f:
        players_data = json.load(f)
    print(f"✅ Current players: {len(players_data)}")
    
    return historical_df, players_data, teams_data

def filter_attacking_players(historical_df):
    """Filter to outfield players (FWD, MID, DEF) - includes defenders for position-aware training"""
    print("\n⚽ Filtering to ALL outfield players (including defenders for position-aware model)...")
    
    # Include forwards, midfielders AND defenders
    attacking_df = historical_df[
        historical_df['position'].isin(['FWD', 'MID', 'DEF'])
    ].copy()
    
    # Ensure numeric columns
    numeric_cols = ['goals_scored', 'expected_goals', 'assists', 'expected_assists', 
                   'minutes', 'total_points', 'ict_index', 'threat', 'starts']
    
    for col in numeric_cols:
        attacking_df[col] = pd.to_numeric(attacking_df[col], errors='coerce').fillna(0)
    
    print(f"✅ Outfield players data: {len(attacking_df):,} records")
    print(f"✅ Unique players: {attacking_df['name'].nunique()}")
    print(f"✅ Position distribution:")
    position_counts = attacking_df['position'].value_counts()
    for pos, count in position_counts.items():
        print(f"    {pos}: {count:,} records ({count/len(attacking_df)*100:.1f}%)")
    
    return attacking_df

def engineer_goal_scoring_features(attacking_df, feature_engine):
    """Engineer features for goal prediction using shared feature engineering with position awareness"""
    print("\n🔧 Engineering goal-scoring features with position awareness...")
    
    # Define features relevant for goal scoring prediction
    goal_scoring_features = [
        'minutes', 'total_points', 'goals_scored', 'assists', 'starts',
        'threat', 'expected_goals', 'expected_assists', 'expected_goal_involvements',
        'creativity', 'bps'
    ]
    
    # Use shared feature engineering for rolling averages with specific features
    attacking_df_with_features = feature_engine.calculate_rolling_features(
        attacking_df,
        group_col='name',
        sort_cols=['name', 'GW'],
        rolling_features=goal_scoring_features
    )
    
    # Add position-specific features (one-hot encoding)
    attacking_df_with_features['is_forward'] = (attacking_df_with_features['position'] == 'FWD').astype(int)
    attacking_df_with_features['is_midfielder'] = (attacking_df_with_features['position'] == 'MID').astype(int)
    attacking_df_with_features['is_defender'] = (attacking_df_with_features['position'] == 'DEF').astype(int)
    
    # Add derived features
    # Goal efficiency (goals per expected goal)
    attacking_df_with_features['goal_efficiency'] = (
        attacking_df_with_features['goals_scored_avg_3gw'] / 
        attacking_df_with_features.get('expected_goals_avg_3gw', 0.1).clip(lower=0.1)
    )
    
    # Recent form (last 3 games total points) - already calculated by shared engine as 'form'
    attacking_df_with_features['recent_form'] = attacking_df_with_features['form']
    
    print(f"✅ Feature engineering completed with position features")
    print(f"✅ Position feature distribution:")
    print(f"    Forwards: {attacking_df_with_features['is_forward'].sum():,}")
    print(f"    Midfielders: {attacking_df_with_features['is_midfielder'].sum():,}")
    print(f"    Defenders: {attacking_df_with_features['is_defender'].sum():,}")
    
    return attacking_df_with_features

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

def fit_goals_model(attacking_df, feature_engine):
    """Fit Poisson regression model for goals prediction using shared feature engineering"""
    print("\n📈 Fitting Poisson model for goals...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Remove rows with NaN values and filter to players with substantial playing time
    model_df = attacking_df.dropna().copy()
    model_df = filter_training_data(model_df, minutes_threshold=450)  # NEW: Improved filtering
    
    print(f"✅ Filtered to players with substantial playing time: {len(model_df):,} records")
    
    # Prepare opponent strength features (normalize opponent defense strength)
    print("🎯 Adding opponent strength features...")
    model_df['opponent_defence_strength_normalized'] = model_df['opponent_defence_strength'] / 1400.0  # Normalize to 0-1
    
    # Verify opponent strength columns are available
    required_opponent_cols = ['opponent_defence_strength', 'fixture_attractiveness']
    missing_cols = [col for col in required_opponent_cols if col not in model_df.columns]
    if missing_cols:
        raise ValueError(f"Missing opponent strength columns: {missing_cols}")
    
    print(f"✅ Opponent strength features ready:")
    print(f"   Defence strength range: {model_df['opponent_defence_strength'].min():.0f} - {model_df['opponent_defence_strength'].max():.0f}")
    print(f"   Fixture attractiveness range: {model_df['fixture_attractiveness'].min():.3f} - {model_df['fixture_attractiveness'].max():.3f}")
    
    # Get feature columns from shared method (single source of truth) - use position-aware version
    feature_cols = feature_engine.get_goals_model_feature_columns()
    
    print(f"🎯 Using position-aware feature definition: {len(feature_cols)} features")
    print(f"   Features: {', '.join(feature_cols[:5])}... (showing first 5)")
    
    # Prepare features and target
    X = model_df[feature_cols].copy()
    y = model_df['goals_scored'].copy()
    
    print(f"✅ Training samples: {len(X):,}")
    print(f"✅ Features: {len(feature_cols)}")
    
    # Show goal distribution by position
    print(f"\n📊 Goals by position in training data:")
    for position in ['DEF', 'MID', 'FWD']:
        pos_goals = model_df[model_df['position'] == position]['goals_scored']
        if len(pos_goals) > 0:
            print(f"   {position}: {pos_goals.mean():.3f} avg goals ({len(pos_goals):,} samples)")
    print(f"✅ Goals distribution:")
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
    
    print(f"\n📊 Position-Aware Expected Goals Model Performance:")
    print(f"  Training MSE: {train_mse:.3f}")
    print(f"  Test MSE: {test_mse:.3f}")
    print(f"  Training MAE: {train_mae:.3f}")
    print(f"  Test MAE: {test_mae:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': poisson_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\n🎯 Feature Coefficients (Top 10):")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:+.3f}")
    
    # Position-specific analysis
    print(f"\n🏷️ Position Coefficients:")
    position_features = ['is_forward', 'is_midfielder', 'is_defender']
    for feature in position_features:
        if feature in feature_importance['feature'].values:
            coef = feature_importance[feature_importance['feature'] == feature]['coefficient'].iloc[0]
            print(f"  {feature}: {coef:+.3f}")
    
    # Test predictions by position
    print(f"\n🎲 Test Set Predictions by Position:")
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    
    for pos_feature, pos_name in [('is_defender', 'DEF'), ('is_midfielder', 'MID'), ('is_forward', 'FWD')]:
        if pos_feature in feature_cols:
            pos_mask = X_test_df[pos_feature] == 1
            if pos_mask.sum() > 0:
                pos_pred = y_test_pred[pos_mask]
                pos_actual = y_test.values[pos_mask]  # Use .values to avoid indexing issue
                print(f"  {pos_name}: {pos_pred.mean():.3f} predicted vs {pos_actual.mean():.3f} actual (n={pos_mask.sum()})")
    
    return poisson_model, scaler, feature_cols

def create_prediction_functions(poisson_model, scaler, feature_cols, feature_engine):
    """Create functions to predict goals for any player using shared feature engineering"""
    print("\n🎯 Creating prediction functions...")
    
    def predict_player_goals_distribution(player_data, historical_df=None, was_home=True, max_goals=4,
                                         opponent_defence_strength=1100.0, fixture_attractiveness=0.5):
        """
        Predict probability distribution of goals for a player using shared feature engineering
        
        Args:
            player_data: Dict with current player data
            historical_df: Optional historical data for better rolling averages 
            was_home: Whether playing at home
            max_goals: Maximum goals to calculate probabilities for
            
        Returns:
            Dict with expected goals and probabilities
        """
        
        # Get historical context if available
        historical_context = None
        if historical_df is not None and 'web_name' in player_data:
            historical_context = feature_engine.get_historical_context(
                player_data['web_name'], 
                historical_df
            )
        
        # Prepare features using shared logic - now includes position features
        features = feature_engine.prepare_goals_model_features(
            player_data,
            historical_context=historical_context,
            was_home=was_home,
            opponent_defence_strength=opponent_defence_strength,
            fixture_attractiveness=fixture_attractiveness
        )
        
        # Convert to numpy array and scale
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Predict expected goals
        expected_goals = poisson_model.predict(features_scaled)[0]
        
        # Calculate Poisson probabilities
        probabilities = {}
        for goals in range(max_goals + 1):
            prob = stats.poisson.pmf(goals, expected_goals)
            probabilities[goals] = prob
        
        # Probability of scoring at least 1 goal
        prob_score = 1 - probabilities[0]
        
        return {
            'expected_goals': expected_goals,
            'probabilities': probabilities,
            'prob_score': prob_score
        }
    
    return predict_player_goals_distribution

def save_model(poisson_model, scaler, feature_cols, feature_engine):
    """Save the trained model and associated objects"""
    print("\n💾 Saving expected goals model...")
    
    model_data = {
        'model': poisson_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_engine': feature_engine,  # Include shared feature engine
        'model_type': 'expected_goals_poisson',
        'trained_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, 'expected_goals_model.pkl')
    print("✅ Model saved to expected_goals_model.pkl")
    print("✅ Shared feature engine included in model data")

def demonstrate_predictions(predict_fn):
    """Demonstrate model predictions"""
    print("\n🎯 Example Player Goals Predictions:\n")
    
    # Example predictions for different player types
    examples = [
        {
            'name': 'Premium Forward',
            'stats': {
                'goals_avg_3gw': 0.8, 'goals_avg_5gw': 0.7, 'goals_avg_10gw': 0.6,
                'xg_avg_3gw': 0.9, 'xg_avg_5gw': 0.8, 'xg_avg_10gw': 0.7,
                'minutes_avg_3gw': 85, 'minutes_avg_5gw': 80, 
                'starts_avg_3gw': 1.0, 'starts_avg_5gw': 0.9,
                'threat_avg_3gw': 80, 'threat_avg_5gw': 75,
                'goal_efficiency': 1.2, 'recent_form': 8.0,
                'is_forward': 1, 'was_home': 1
            }
        },
        {
            'name': 'Attacking Midfielder',
            'stats': {
                'goals_avg_3gw': 0.3, 'goals_avg_5gw': 0.4, 'goals_avg_10gw': 0.3,
                'xg_avg_3gw': 0.4, 'xg_avg_5gw': 0.5, 'xg_avg_10gw': 0.4,
                'minutes_avg_3gw': 75, 'minutes_avg_5gw': 70,
                'starts_avg_3gw': 0.8, 'starts_avg_5gw': 0.7,
                'threat_avg_3gw': 45, 'threat_avg_5gw': 50,
                'goal_efficiency': 0.9, 'recent_form': 6.0,
                'is_forward': 0, 'was_home': 0
            }
        }
    ]
    
    for example in examples:
        pred = predict_fn(example['stats'])
        print(f"{example['name']}:")
        print(f"  Expected goals: {pred['expected_goals']:.3f}")
        print(f"  Probability to score: {pred['prob_score']:.3f}")
        print(f"  Goal probabilities:")
        for goals, prob in pred['probabilities'].items():
            print(f"    {goals} goals: {prob:.3f}")
        print()

def main():
    """Main training pipeline"""
    print("🚀 Starting Expected Goals Model Training Pipeline\n")
    
    # Load data
    historical_df, players_data, teams_data = load_data()
    
    # Initialize feature engine
    feature_engine = PlayerFeatureEngine(teams_data)
    
    # Filter to attacking players
    attacking_df = filter_attacking_players(historical_df)
    
    # Engineer features using shared engine
    attacking_df = engineer_goal_scoring_features(attacking_df, feature_engine)
    
    # Fit Poisson models using shared feature definition
    poisson_model, scaler, feature_cols = fit_goals_model(attacking_df, feature_engine)
    
    # Create prediction functions
    predict_fn = create_prediction_functions(poisson_model, scaler, feature_cols, feature_engine)
    
    # Save model with feature engine
    save_model(poisson_model, scaler, feature_cols, feature_engine)
    
    # Demonstrate predictions
    demonstrate_predictions(predict_fn)
    
    print("🎯 Expected Goals Model training completed!")
    print("Ready to predict goal probabilities for attacking players!")
    print("✅ Shared feature engineering ensures consistent features between training and prediction")
    print("\n📋 Model provides:")
    print("  • Expected goals per match")
    print("  • Probability to score at least 1 goal")
    print("  • Full probability distribution (0, 1, 2+ goals)")

if __name__ == "__main__":
    main()
