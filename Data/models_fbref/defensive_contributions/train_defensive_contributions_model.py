"""
Defensive Contributions Model Training Script

Trains a machine learning model to predict defensive contribution points
based on historical FBRef data and player characteristics.

Model predicts the likelihood of earning 2 defensive contribution points
under the new FPL 2025/26 scoring rules.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional


class DefensiveContributionModelTrainer:
    """Train and evaluate defensive contribution prediction models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.position_encoder = LabelEncoder()
        self.models = {}
        self.feature_importance = {}
        
        # Define feature sets
        self.base_features = [
            'minutes', 'position', 'start',
            # Historical defensive stats (rolling averages)
            'clearances_avg_3gw', 'clearances_avg_5gw', 'clearances_avg_10gw',
            'blocks_avg_3gw', 'blocks_avg_5gw', 'blocks_avg_10gw',
            'interceptions_avg_3gw', 'interceptions_avg_5gw', 'interceptions_avg_10gw',
            'tackles_won_avg_3gw', 'tackles_won_avg_5gw', 'tackles_won_avg_10gw',
            'recoveries_avg_3gw', 'recoveries_avg_5gw', 'recoveries_avg_10gw',
            # Historical performance
            'cbit_score_avg_3gw', 'cbit_score_avg_5gw', 'cbit_score_avg_10gw',
            'cbirt_score_avg_3gw', 'cbirt_score_avg_5gw', 'cbirt_score_avg_10gw',
            'cbit_threshold_ratio_avg_3gw', 'cbit_threshold_ratio_avg_5gw',
            'cbirt_threshold_ratio_avg_3gw', 'cbirt_threshold_ratio_avg_5gw',
            # General performance context
            'goals_avg_3gw', 'assists_avg_3gw', 'xG_avg_3gw', 'xA_avg_3gw',
            # Fixture difficulty features (NEW)
            'opponent_attack_strength', 'opponent_defense_strength', 'opponent_overall_strength',
            'home_away', 'fixture_difficulty_rating', 'team_strength_differential'
        ]
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load enhanced FBRef data with defensive contributions"""
        print(f"📂 Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Filter for players with sufficient minutes (model relevance)
        df_filtered = df[df['minutes'] >= 15].copy()  # At least 15 minutes played
        
        print(f"✅ Loaded {len(df_filtered)} records (filtered from {len(df)} total)")
        print(f"📊 Defensive contributions: {df_filtered['defensive_contribution_points'].sum():,} points")
        print(f"📈 Contribution rate: {(df_filtered['defensive_contribution_points'] > 0).mean():.1%}")
        
        return df_filtered
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training"""
        print("🔧 Preparing features for model training...")
        
        # Create feature matrix
        available_features = [f for f in self.base_features if f in df.columns]
        missing_features = [f for f in self.base_features if f not in df.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        print(f"✅ Using {len(available_features)} features: {available_features}")
        
        X = df[available_features].copy()
        
        # Handle categorical variables
        if 'position' in X.columns:
            X['position_encoded'] = self.position_encoder.fit_transform(X['position'].fillna('Unknown'))
            X = X.drop('position', axis=1)
        
        # Handle home_away encoding (H=1, A=0)
        if 'home_away' in X.columns:
            X['home_away'] = X['home_away'].map({'H': 1, 'A': 0}).fillna(0)
        
        # Handle start encoding (ensure it's numeric)
        if 'start' in X.columns:
            X['start'] = pd.to_numeric(X['start'], errors='coerce').fillna(0)
        
        # Fill missing values with 0 (makes sense for defensive stats)
        X = X.fillna(0)
        
        # Target: binary classification (0 or 2 points -> 0 or 1)
        y = (df['defensive_contribution_points'] > 0).astype(int)
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train multiple models and select the best one"""
        print("🤖 Training defensive contribution prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model candidates
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\n🔄 Training {name}...")
            
            # Train model
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate model
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✅ {name}: AUC = {auc_score:.3f}, CV = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.Series(
                    model.feature_importances_, 
                    index=X.columns
                ).sort_values(ascending=False)
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        best_model = results[best_model_name]
        
        print(f"\n🏆 Best model: {best_model_name} (CV AUC: {best_model['cv_mean']:.3f})")
        
        # Detailed evaluation of best model
        print(f"\n📊 Detailed evaluation of {best_model_name}:")
        print(classification_report(y_test, best_model['predictions']))
        
        self.models['best'] = best_model['model']
        self.models['best_name'] = best_model_name
        
        return results
    
    def save_model(self, output_dir: str) -> None:
        """Save the trained model and metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / 'defensive_contributions_model.joblib'
        joblib.dump(self.models['best'], model_path)
        print(f"💾 Saved model to {model_path}")
        
        # Save scaler
        scaler_path = output_path / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Saved scaler to {scaler_path}")
        
        # Save encoders
        encoder_path = output_path / 'position_encoder.joblib'
        joblib.dump(self.position_encoder, encoder_path)
        print(f"💾 Saved position encoder to {encoder_path}")
        
        # Save metadata
        metadata = {
            'model_type': self.models['best_name'],
            'features': self.base_features,
            'random_state': self.random_state,
            'feature_importance': self.feature_importance.get(self.models['best_name'], {}).to_dict() if self.models['best_name'] in self.feature_importance else {}
        }
        
        metadata_path = output_path / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Saved metadata to {metadata_path}")
        
        # Print feature importance
        if self.models['best_name'] in self.feature_importance:
            print("\n📈 Top 10 Most Important Features:")
            top_features = self.feature_importance[self.models['best_name']].head(10)
            for feature, importance in top_features.items():
                print(f"  {feature}: {importance:.3f}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train defensive contributions prediction model')
    parser.add_argument('--data_path', type=str,
                       default='../outputs/enhanced_fbref_with_defensive_contributions.csv',
                       help='Path to enhanced FBRef data CSV')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/owen/src/Personal/fpl-team-picker/Data/models_fbref/defensive_contributions',
                       help='Output directory for trained model')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DefensiveContributionModelTrainer(random_state=args.random_state)
    
    # Load data
    df = trainer.load_data(args.data_path)
    
    # Prepare features
    X, y = trainer.prepare_features(df)
    
    # Train models
    results = trainer.train_models(X, y)
    
    # Save best model
    trainer.save_model(args.output_dir)
    
    print(f"\n✅ Training complete! Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
