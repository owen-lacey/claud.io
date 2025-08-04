# FPL Minutes Model - Complete Training Script
import pandas as pd
import numpy as np
import json
import joblib
import warnings
import sys
import os

# Add parent directory to Python path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add the feature engineering module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'feature_engineering'))
from feature_engineering.player_features import PlayerFeatureEngine, load_historical_data, load_teams_data

warnings.filterwarnings('ignore')

print("🚀 Starting Minutes Model Training Pipeline\n")

# Load data using shared utilities
print("📊 Loading data...")
historical_data = load_historical_data('/Users/owen/src/Personal/fpl-team-picker/Data/raw/parsed_gw_2425.csv')
from database.mongo.mongo_data_loader import load_teams_data, load_players_data
teams_data = load_teams_data()

players_data = load_players_data()

print(f"✅ Current players: {len(players_data):,}")
print(f"✅ Historical data: {len(historical_data):,} records with opponent strength features")

# Verify opponent strength columns exist
required_columns = ['opponent_attack_strength', 'opponent_defence_strength', 
                   'opponent_overall_strength', 'fixture_attractiveness']

missing_columns = [col for col in required_columns if col not in historical_data.columns]
if missing_columns:
    print(f"⚠️ Missing opponent strength columns: {missing_columns}")
else:
    print(f"✅ Opponent strength features available")

# Initialize feature engine
feature_engine = PlayerFeatureEngine(teams_data)

# Create minutes categories
print("\n🏷️  Creating minutes categories...")

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


def categorize_minutes(minutes):
    """Convert minutes to categories for classification"""
    if minutes == 0:
        return 'no_minutes'
    elif minutes < 30:
        return 'few_minutes'
    elif minutes < 70:
        return 'substantial_minutes'
    else:
        return 'full_match'

historical_data['minutes_category'] = historical_data['minutes'].apply(categorize_minutes)

# Display category distribution
category_counts = historical_data['minutes_category'].value_counts()
print("\n📈 Minutes Category Distribution:")
for category, count in category_counts.items():
    percentage = (count / len(historical_data)) * 100
    print(f"  {category}: {count:,} ({percentage:.1f}%)")

# Feature Engineering using shared logic
print("\n🔧 Engineering features with shared PlayerFeatureEngine...")

# Calculate rolling features using the shared engine
historical_data_with_features = feature_engine.calculate_rolling_features(
    historical_data, 
    group_col='name', 
    sort_cols=['name', 'GW']
)

# Encode categorical variables
print("\n🎯 Encoding categorical variables...")
label_encoders = {}
categorical_features = ['position', 'team']

for feature in categorical_features:
    le = LabelEncoder()
    historical_data_with_features[f'{feature}_encoded'] = le.fit_transform(historical_data_with_features[feature])
    label_encoders[feature] = le

# Get feature columns from the shared engine
feature_columns = feature_engine.get_minutes_model_feature_columns()

# Add opponent strength features to the dataframe (preprocessing step)
if 'opponent_overall_strength' in historical_data_with_features.columns:
    historical_data_with_features['opponent_overall_strength_normalized'] = historical_data_with_features['opponent_overall_strength'] / 1400.0
    print("✅ Added normalized opponent overall strength")
else:
    historical_data_with_features['opponent_overall_strength_normalized'] = 0.82  # Default ~1150/1400
    print("⚠️ Using default opponent overall strength")

if 'fixture_attractiveness' not in historical_data_with_features.columns:
    historical_data_with_features['fixture_attractiveness'] = 0.5  # Default neutral
    print("⚠️ Using default fixture attractiveness")

# Prepare training data
print("\n🎯 Preparing training data...")
training_data = historical_data_with_features[feature_columns + ['minutes_category']].dropna()

X = training_data[feature_columns]
y = training_data['minutes_category']

print(f"✅ Training samples: {len(X):,}")
print(f"✅ Features: {len(feature_columns)} (from shared feature engine)")
print(f"✅ All feature columns: {feature_columns}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training set: {len(X_train):,}")
print(f"✅ Test set: {len(X_test):,}")

# Train Random Forest model
print("\n🌲 Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
print("\n📊 Model Performance:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🎯 Feature Importance:")
for _, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# Highlight opponent strength features specifically
print(f"\n🏆 Opponent Strength Feature Impact:")
opponent_strength_features = ['opponent_overall_strength_normalized', 'fixture_attractiveness']
opponent_features_importance = feature_importance[feature_importance['feature'].isin(opponent_strength_features)]
for _, row in opponent_features_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

if len(opponent_features_importance) > 0:
    print(f"✅ Opponent strength features successfully integrated into minutes prediction")

# Save model and encoders
print("\n💾 Saving model and encoders...")
model_data = {
    'model': rf_model,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns,  # All features from shared engine
    'feature_importance': feature_importance.to_dict('records'),
    'feature_engine': feature_engine,  # Include the shared feature engine
    'enhanced_features': True,
    'trained_at': pd.Timestamp.now().isoformat()
}

joblib.dump(model_data, 'minutes_model.pkl')

print("✅ Model saved to minutes_model.pkl")
print("✅ Shared feature engine included in model data")

# Create prediction function using shared feature engine
def predict_minutes_probability(player_data, model_data, historical_df=None):
    """
    Predict minutes probability for a player using shared feature engineering
    
    Args:
        player_data: Dictionary with current player data
        model_data: Loaded model data from pickle file
        historical_df: Optional historical data for better rolling averages
    
    Returns:
        Dictionary with probability for each minutes category
    """
    model = model_data['model']
    feature_engine = model_data.get('feature_engine')
    label_encoders = model_data['label_encoders']
    
    if not feature_engine:
        raise ValueError("Feature engine not found in model data. Please retrain the model.")
    
    # Get historical context if available
    historical_context = None
    if historical_df is not None and 'web_name' in player_data:
        historical_context = feature_engine.get_historical_context(
            player_data['web_name'], 
            historical_df
        )
    
    # Prepare features using shared logic
    features = feature_engine.prepare_minutes_model_features(
        player_data,
        historical_context=historical_context,
        position_encoder=label_encoders.get('position'),
        team_encoder=label_encoders.get('team')
    )
    
    # Convert to numpy array for prediction
    features_array = np.array([features])
    
    # Get probabilities
    probabilities = model.predict_proba(features_array)[0]
    classes = model.classes_
    
    return dict(zip(classes, probabilities))

print("\n🎯 Minutes Model training completed!")
print("Ready to predict minutes probabilities for any player!")
print("✅ Shared feature engineering ensures consistent features between training and prediction")

# Example prediction (you would load model_data from pickle in practice)
print("\n📋 Example prediction ready...")
print("Use predict_minutes_probability(player_data, model_data, historical_df) to get predictions")
print("The historical_df parameter enables proper rolling averages for better accuracy")
