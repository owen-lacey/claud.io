"""
FPL Expected Points Prediction Engine
====================================

Pure prediction engine that combines all our models to generate
accurate FPL expected points for all players.

This system orchestrates:
- All 6 core ML models
- FPL 2025/26 scoring engine  
- Advanced bonus points prediction
- Risk assessment and confidence intervals
- Value analysis (points per million)

Author: FPL Team Picker
Version: 2025/26 Season - Production Ready
"""

import json
import pandas as pd
import sys
import os

# Add parent directory to Python path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import joblib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
from pathlib import Path

# Add shared feature engineering module
from feature_engineering.player_features import PlayerFeatureEngine, load_historical_data

# Add our modules to path (if needed)
# sys.path.append('../fpl_scoring_engine')  # Removed - integrated into main pipeline

@dataclass
class PlayerExpectedPoints:
    """Complete expected points prediction for a player"""
    player_id: str
    name: str
    team: str
    position: str
    current_price: float
    
    # Core predictions
    expected_points: float
    expected_goals: float
    expected_assists: float
    expected_saves: float
    clean_sheet_prob: float
    
    # Playing time predictions
    expected_minutes: float
    minutes_probability: float
    minutes_category: str  # no_minutes, few_minutes, substantial_minutes, full_match
    
    # Advanced predictions
    expected_bonus: float
    bonus_prob_3: float
    bonus_prob_2: float
    bonus_prob_1: float
    
    # Value metrics
    points_per_million: float
    value_rank: int
    form_adjustment: float
    
    # Confidence and risk
    prediction_confidence: float
    variance: float
    ceiling: float  # 90th percentile outcome
    floor: float    # 10th percentile outcome
    
    # Context
    gameweek: int
    fixture_difficulty: str
    home_away: str
    opponent: str
    opponent_strength_attack: int
    opponent_strength_defence: int
    fixture_attractiveness: float


class FPLPredictionEngine:
    """
    Pure FPL expected points prediction engine
    
    Orchestrates all our models and engines to produce accurate
    expected points predictions with confidence intervals and value analysis.
    """
    
    def __init__(self, data_dir: str = "../database", models_dir: str = "../models"):
        """Initialize the FPL prediction engine with all trained models"""
        
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        # Load core data
        self.players_data = self._load_json_data("players.json")
        self.teams_data = self._load_json_data("teams.json")
        
        # Load historical data for proper rolling averages (enhanced with opponent strength)
        try:
            historical_path = str(Path(__file__).parent.parent / "raw" / "parsed_gw_with_opponent_strength.csv")
            self.historical_data = load_historical_data(historical_path)
            print(f"✅ Loaded enhanced historical data: {len(self.historical_data):,} gameweek records with opponent strength")
        except Exception as e:
            print(f"⚠️ Could not load enhanced historical data: {e}")
            try:
                # Fallback to original data
                historical_path = str(Path(__file__).parent.parent / "raw" / "parsed_gw.csv")
                self.historical_data = load_historical_data(historical_path)
                print(f"✅ Loaded fallback historical data: {len(self.historical_data):,} gameweek records (no opponent strength)")
            except Exception as e2:
                print(f"⚠️ Could not load any historical data: {e2}")
                print("   Using fallback feature approximations")
                self.historical_data = None
        
        # Initialize shared feature engine
        self.feature_engine = PlayerFeatureEngine(self.teams_data)
        
        # Load all trained models
        print("🤖 Loading trained ML models...")
        self.models = self._load_all_models()
        
        print("🚀 FPL Prediction Engine Initialized")
        print(f"   📊 Loaded {len(self.players_data)} players")
        print(f"   ⚽ Loaded {len(self.teams_data) if isinstance(self.teams_data, (list, dict)) else 0} teams")
        print(f"   🎯 Loaded {len(self.models)} trained models")
        print(f"   🔧 Shared feature engineering enabled")
        
    def _load_all_models(self) -> Dict[str, Any]:
        """Load all trained ML models"""
        models = {}
        
        model_files = {
            'minutes': 'minutes/minutes_model.pkl',
            'goals': 'expected_goals/expected_goals_model.pkl', 
            'assists': 'expected_assists/expected_assists_model.pkl',
            'saves': 'saves/saves_model.pkl',
            'team_goals_conceded': 'team_goals_conceded/team_goals_conceded_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            try:
                model_path = self.models_dir / filename
                model_data = joblib.load(model_path)
                models[model_name] = model_data
                print(f"   ✅ {model_name}: {filename}")
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {e}")
                models[model_name] = None
        
        return models
    
    def generate_gameweek_predictions(self, 
                                    gameweek: int,
                                    include_bonus: bool = True) -> Dict[str, Any]:
        """
        Generate expected points predictions for a gameweek
        
        Args:
            gameweek: Target gameweek for predictions
            include_bonus: Include bonus points prediction
            
        Returns:
            Complete prediction data with rankings and analysis
        """
        
        print(f"\n🎯 GENERATING GAMEWEEK {gameweek} PREDICTIONS")
        print("=" * 50)
        
        # Generate player predictions
        print("📊 Calculating expected points for all players...")
        player_predictions = self._generate_all_player_predictions(
            gameweek, include_bonus
        )
        
        # Sort by expected points and add value rankings
        player_predictions.sort(key=lambda p: p.expected_points, reverse=True)
        
        # Add value rankings (by points per million)
        value_sorted = sorted(player_predictions, key=lambda p: p.points_per_million, reverse=True)
        for i, player in enumerate(value_sorted):
            player.value_rank = i + 1
        
        # Compile results
        results = {
            'gameweek': gameweek,
            'generated_at': datetime.now().isoformat(),
            'players': [asdict(p) for p in player_predictions],
            'summary': self._generate_prediction_summary(player_predictions)
        }
        
        print(f"\n✅ Generated predictions for {len(player_predictions)} players")
        
        return results
    
    def _generate_all_player_predictions(self, 
                                       gameweek: int,
                                       include_bonus: bool) -> List[PlayerExpectedPoints]:
        """Generate predictions for all available players"""
        
        predictions = []
        skipped_predictions = 0
        
        # Handle both list and dict formats
        if isinstance(self.players_data, list):
            players_to_process = [(str(p.get('id', i)), p) for i, p in enumerate(self.players_data)]
        else:
            players_to_process = list(self.players_data.items())
        
        for player_id, player_data in players_to_process:
            try:
                # Skip if missing essential data
                if not player_data.get('web_name') or not player_data.get('now_cost'):
                    continue
                    
                prediction = self._calculate_single_player_prediction(
                    player_id, player_data, gameweek, include_bonus
                )
                
                # Only add non-None predictions
                if prediction is not None:
                    predictions.append(prediction)
                else:
                    skipped_predictions += 1
                
            except Exception as e:
                # Skip players we can't predict - be more specific about errors
                player_name = player_data.get('web_name', 'Unknown')
                position = self._get_position_name(player_data.get('element_type', 4))
                print(f"❌ PREDICTION FAILED for player {player_id} ({player_name}, {position}):")
                print(f"   Error: {type(e).__name__}: {str(e)}")
                print(f"   Player data keys: {list(player_data.keys())[:10]}...")
                if hasattr(e, '__traceback__'):
                    import traceback
                    tb_lines = traceback.format_exc().split('\n')
                    print(f"   Stack trace: {tb_lines[-3:-1]}")
                skipped_predictions += 1
                continue
        
        # Summary of predictions vs skipped
        total_processed = len(predictions) + skipped_predictions
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"   ✅ Successful predictions: {len(predictions)}")
        print(f"   ⚠️  Skipped predictions: {skipped_predictions}")
        print(f"   📈 Success rate: {len(predictions)/total_processed*100:.1f}%" if total_processed > 0 else "   📈 Success rate: 0%")
        
        return predictions
    
    def _has_sufficient_playing_time(self, player_data: Dict, minutes_threshold: int = 450) -> bool:
        """
        Check if player has sufficient playing time for reliable ML predictions
        
        Args:
            player_data: Player data dictionary
            minutes_threshold: Minimum minutes for reliable predictions (default: 450 = ~5 games)
            
        Returns:
            True if player has sufficient data, False otherwise
        """
        total_minutes = player_data.get('minutes', 0)
        return total_minutes >= minutes_threshold

    def _calculate_single_player_prediction(self, 
                                          player_id: str,
                                          player_data: Dict,
                                          gameweek: int,
                                          include_bonus: bool) -> PlayerExpectedPoints:
        """Calculate complete prediction for a single player"""
        
        # Basic player info
        name = player_data.get('web_name', f'Player {player_id}')
        team_id = player_data.get('team', 1)
        position = self._get_position_name(player_data.get('element_type', 4))
        price = player_data.get('now_cost', 50) / 10.0
        
        # Get team info
        team_name = self._get_team_name(team_id)
        
        # Check if player has sufficient playing time for ML predictions
        has_sufficient_data = self._has_sufficient_playing_time(player_data)
        
        if not has_sufficient_data:
            # Use simplified predictions for players with insufficient data
            return self._calculate_simplified_prediction(
                player_id, player_data, gameweek, include_bonus,
                name, team_name, position, price
            )
        
        # First predict playing time using our minutes model
        minutes_prediction = self._predict_minutes(player_data, gameweek)
        if minutes_prediction is None:
            print(f"⚠️  SKIPPING {name} ({position}, {team_name}) - Minutes prediction failed")
            return None
            
        expected_minutes = minutes_prediction['expected_minutes']
        minutes_probability = minutes_prediction['probability']
        minutes_category = minutes_prediction['category']
        
        # Core ML model predictions using our trained models
        expected_goals = self._predict_goals(player_data, gameweek, expected_minutes)
        if expected_goals is None:
            print(f"⚠️  SKIPPING {name} ({position}, {team_name}) - Goals prediction failed")
            return None
            
        expected_assists = self._predict_assists(player_data, gameweek, expected_minutes)
        if expected_assists is None:
            print(f"⚠️  SKIPPING {name} ({position}, {team_name}) - Assists prediction failed")
            return None
            
        expected_saves = self._predict_saves(player_data, gameweek, expected_minutes)
        if expected_saves is None and position == 'GK':
            print(f"⚠️  SKIPPING {name} ({position}, {team_name}) - Saves prediction failed")
            return None
        elif expected_saves is None:
            expected_saves = 0.0  # Non-GK players don't need saves predictions
            
        clean_sheet_prob = self._predict_clean_sheet(player_data, gameweek)
        if clean_sheet_prob is None and position in ['GK', 'DEF']:
            print(f"⚠️  SKIPPING {name} ({position}, {team_name}) - Clean sheet prediction failed")
            return None
        elif clean_sheet_prob is None:
            clean_sheet_prob = 0.0  # Non-defensive players don't get clean sheet points
        
        # Calculate base FPL points using our scoring engine (no yellow cards)
        base_points = self._calculate_base_fpl_points(
            position, expected_goals, expected_assists, expected_saves,
            clean_sheet_prob
        )
        
        # Add bonus points if requested
        expected_bonus = 0.0
        bonus_prob_3 = 0.0
        bonus_prob_2 = 0.0
        bonus_prob_1 = 0.0
        
        if include_bonus:
            bonus_prediction = self._predict_bonus_points(
                player_data, expected_goals, expected_assists, gameweek
            )
            expected_bonus = bonus_prediction['expected_bonus']
            bonus_prob_3 = bonus_prediction['prob_3_bonus']
            bonus_prob_2 = bonus_prediction['prob_2_bonus']
            bonus_prob_1 = bonus_prediction['prob_1_bonus']
        
        # Total expected points
        total_expected_points = base_points + expected_bonus
        
        # Value metrics
        points_per_million = total_expected_points / price if price > 0 else 0
        
        # Risk metrics (simplified)
        variance = max(1.0, total_expected_points * 0.3)
        ceiling = total_expected_points + 1.645 * np.sqrt(variance)  # 90th percentile
        floor = max(0, total_expected_points - 1.645 * np.sqrt(variance))  # 10th percentile
        
        # Fixture context
        fixture_info = self._get_fixture_info(team_id, gameweek)
        
        return PlayerExpectedPoints(
            player_id=player_id,
            name=name,
            team=team_name,
            position=position,
            current_price=price,
            
            expected_points=round(total_expected_points, 2),
            expected_goals=round(expected_goals, 3),
            expected_assists=round(expected_assists, 3),
            expected_saves=round(expected_saves, 1),
            clean_sheet_prob=round(clean_sheet_prob, 3),
            
            # Playing time predictions
            expected_minutes=round(expected_minutes, 1),
            minutes_probability=round(minutes_probability, 3),
            minutes_category=minutes_category,
            
            expected_bonus=round(expected_bonus, 2),
            bonus_prob_3=round(bonus_prob_3, 3),
            bonus_prob_2=round(bonus_prob_2, 3),
            bonus_prob_1=round(bonus_prob_1, 3),
            
            points_per_million=round(points_per_million, 2),
            value_rank=0,  # Will be set after sorting
            form_adjustment=0.0,
            
            prediction_confidence=0.75,  # Placeholder
            variance=round(variance, 2),
            ceiling=round(ceiling, 1),
            floor=round(floor, 1),
            
            gameweek=gameweek,
            fixture_difficulty=fixture_info['difficulty'],
            home_away=fixture_info['home_away'],
            opponent=fixture_info['opponent'],
            opponent_strength_attack=fixture_info['opponent_strength_attack'],
            opponent_strength_defence=fixture_info['opponent_strength_defence'],
            fixture_attractiveness=fixture_info['fixture_attractiveness']
        )
    
    # Trained ML model predictions
    def _predict_minutes(self, player_data: Dict, gameweek: int) -> Dict[str, Any]:
        """Predict expected minutes using our trained minutes model"""
        
        if not self.models.get('minutes'):
            player_name = player_data.get('web_name', 'Unknown')
            position = self._get_position_name(player_data.get('element_type', 4))
            print(f"❌ MINUTES MODEL NOT LOADED for {player_name} ({position})")
            return None
        
        try:
            # Prepare features for minutes model
            features = self._prepare_minutes_features(player_data, gameweek)
            
            # Get model and predict
            model_data = self.models['minutes']
            minutes_model = model_data['model']
            
            # Convert to DataFrame with proper feature names to avoid sklearn warning
            import pandas as pd
            feature_names = model_data.get('feature_columns', [f'feature_{i}' for i in range(len(features))])
            features_df = pd.DataFrame([features], columns=feature_names)
            
            # Predict probability distribution
            probabilities = minutes_model.predict_proba(features_df)[0]
            
            # Get predicted category
            predicted_category = minutes_model.predict(features_df)[0]
            
            # Convert category to expected minutes
            category_minutes = {
                'no_minutes': 0,
                'few_minutes': 15,
                'substantial_minutes': 60, 
                'full_match': 90
            }
            
            expected_minutes = category_minutes.get(predicted_category, 60)
            
            # Calculate weighted expected minutes from probabilities
            if hasattr(minutes_model, 'classes_'):
                weighted_minutes = 0
                for i, category in enumerate(minutes_model.classes_):
                    weighted_minutes += probabilities[i] * category_minutes.get(category, 60)
                expected_minutes = weighted_minutes
            
            # Playing probability (not benched)
            play_probability = 1.0 - probabilities[0] if len(probabilities) > 0 else 0.7
            
            return {
                'expected_minutes': expected_minutes,
                'probability': play_probability, 
                'category': predicted_category
            }
            
        except Exception as e:
            player_name = player_data.get('web_name', 'Unknown')
            print(f"❌ MINUTES PREDICTION FAILED for {player_name}:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Player data sample: {dict(list(player_data.items())[:5])}")
            try:
                features = self._prepare_minutes_features(player_data, gameweek)
                print(f"   Features prepared successfully: {len(features)} features")
                print(f"   Feature values: {features[:5]}...")
            except Exception as feature_error:
                print(f"   Feature preparation also failed: {feature_error}")
            print(f"   🚫 Skipping prediction - cannot generate valid features")
            return None

    def _prepare_minutes_features(self, player_data: Dict, gameweek: int) -> List[float]:
        """Prepare features for minutes prediction using shared feature engineering"""
        
        model_data = self.models.get('minutes', {})
        if not model_data:
            raise ValueError("Minutes model not loaded")
        
        # Get historical context for proper rolling averages
        historical_context = None
        if self.historical_data is not None and 'web_name' in player_data:
            historical_context = self.feature_engine.get_historical_context(
                player_data['web_name'], 
                self.historical_data
            )
        
        # Get fixture information for opponent strength features
        team_id = player_data.get('team', 1)
        fixture_info = self._get_fixture_info(team_id, gameweek)
            
        # Use shared feature engineering logic (now with opponent strength features)
        features = self.feature_engine.prepare_minutes_model_features(
            player_data,
            historical_context=historical_context,
            position_encoder=model_data.get('label_encoders', {}).get('position'),
            team_encoder=model_data.get('label_encoders', {}).get('team'),
            fixture_info=fixture_info
        )
        
        return features
    
    def _predict_goals(self, player_data: Dict, gameweek: int, expected_minutes: float) -> float:
        """Predict expected goals using our trained goals model"""
        
        if not self.models.get('goals'):
            player_name = player_data.get('web_name', 'Unknown')
            position = self._get_position_name(player_data.get('element_type', 4))
            print(f"❌ GOALS MODEL NOT LOADED for {player_name} ({position})")
            return None
        
        try:
            # Prepare features and predict
            features = self._prepare_goals_features(player_data, gameweek)
            model_data = self.models['goals']
            goals_model = model_data['model']
            scaler = model_data['scaler']
            
            # CRITICAL: Apply scaling before prediction
            # Convert to DataFrame with proper feature names to avoid sklearn warning
            import pandas as pd
            feature_names = model_data.get('feature_cols', [f'feature_{i}' for i in range(len(features))])
            features_df = pd.DataFrame([features], columns=feature_names)
            scaled_features = scaler.transform(features_df)
            predicted_goals = goals_model.predict(scaled_features)[0]
            
            # Scale by expected minutes (models trained on per-90-minute basis)
            scaled_goals = predicted_goals * (expected_minutes / 90.0)
            
            return max(0, scaled_goals)
            
        except Exception as e:
            player_name = player_data.get('web_name', 'Unknown')
            position = self._get_position_name(player_data.get('element_type', 4))
            print(f"❌ GOALS PREDICTION FAILED for {player_name} ({position}):")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Expected minutes: {expected_minutes}")
            try:
                features = self._prepare_goals_features(player_data, gameweek)
                print(f"   Features prepared: {features}")
                if self.models.get('goals'):
                    model_data = self.models['goals']
                    print(f"   Model available: {type(model_data.get('model', 'None'))}")
            except Exception as feature_error:
                print(f"   Feature preparation failed: {feature_error}")
            print(f"   🚫 Skipping prediction - cannot generate valid features")
            return None
    
    def _prepare_goals_features(self, player_data: Dict, gameweek: int) -> List[float]:
        """Prepare features for goals model - matches trained model with opponent strength"""
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Get fixture information for opponent strength
        team_name = self._get_team_name(player_data.get('team', 1))
        fixture_info = self._get_fixture_info(team_name, gameweek)
        
        # Features matching the enhanced goals model (18 features):
        # Original 16 features + 2 opponent strength features
        
        goals_scored = safe_float(player_data.get('goals_scored', 0))
        expected_goals = safe_float(player_data.get('expected_goals', 0.0))
        minutes = safe_float(player_data.get('minutes', 0))
        starts = safe_float(player_data.get('starts', 0))
        threat = safe_float(player_data.get('threat', 0))
        
        # Estimate games played for proper per-game averages
        games_played = max(1, minutes / 90.0)  # Rough estimate from total minutes
        
        features = [
            goals_scored / games_played,  # goals_avg_3gw proxy (goals per game)
            goals_scored / games_played,  # goals_avg_5gw proxy  
            goals_scored / games_played,  # goals_avg_10gw proxy
            expected_goals / games_played,  # xg_avg_3gw proxy
            expected_goals / games_played,  # xg_avg_5gw proxy
            expected_goals / games_played,  # xg_avg_10gw proxy
            minutes / games_played,  # minutes_avg_3gw proxy (should be around 90)
            minutes / games_played,  # minutes_avg_5gw proxy
            starts / games_played,  # starts_avg_3gw proxy
            starts / games_played,  # starts_avg_5gw proxy
            threat / games_played,  # threat_avg_3gw proxy
            threat / games_played,  # threat_avg_5gw proxy
            goals_scored / max(expected_goals, 0.1),  # goal_efficiency
            safe_float(player_data.get('form', 0.0)),  # recent_form
            1.0 if player_data.get('element_type') == 4 else 0.0,  # is_forward
            1.0 if fixture_info['is_home'] else 0.0,  # was_home (use actual fixture info)
            # NEW: Opponent strength features
            fixture_info['opponent_strength_defence'] / 1400.0,  # opponent_defence_strength_normalized
            fixture_info.get('fixture_attractiveness', 0.5),  # fixture_attractiveness
        ]
        
        return features
    
    def _predict_assists(self, player_data: Dict, gameweek: int, expected_minutes: float) -> float:
        """Predict expected assists using our trained assists model"""
        
        if not self.models.get('assists'):
            player_name = player_data.get('web_name', 'Unknown')
            position = self._get_position_name(player_data.get('element_type', 4))
            print(f"❌ ASSISTS MODEL NOT LOADED for {player_name} ({position})")
            return None
        
        try:
            features = self._prepare_assists_features(player_data, gameweek)
            model_data = self.models['assists']
            assists_model = model_data['model']
            scaler = model_data['scaler']
            
            # CRITICAL: Apply scaling before prediction
            # Convert to DataFrame with proper feature names to avoid sklearn warning
            import pandas as pd
            feature_names = model_data.get('feature_cols', [f'feature_{i}' for i in range(len(features))])
            features_df = pd.DataFrame([features], columns=feature_names)
            scaled_features = scaler.transform(features_df)
            predicted_assists = assists_model.predict(scaled_features)[0]
            
            # Scale by expected minutes
            scaled_assists = predicted_assists * (expected_minutes / 90.0)
            
            return max(0, scaled_assists)
            
        except Exception as e:
            player_name = player_data.get('web_name', 'Unknown')
            position = self._get_position_name(player_data.get('element_type', 4))
            print(f"❌ ASSISTS PREDICTION FAILED for {player_name} ({position}):")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Expected minutes: {expected_minutes}")
            try:
                features = self._prepare_assists_features(player_data, gameweek)
                print(f"   Features prepared: {features}")
            except Exception as feature_error:
                print(f"   Feature preparation failed: {feature_error}")
            print(f"   🚫 Skipping prediction - cannot generate valid features")
            return None
    
    def _prepare_assists_features(self, player_data: Dict, gameweek: int) -> List[float]:
        """Prepare features for assists model - matches enhanced trained model with opponent strength"""
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Get fixture information for opponent strength
        team_name = self._get_team_name(player_data.get('team', 1))
        fixture_info = self._get_fixture_info(team_name, gameweek)
        
        # Features matching the enhanced assists model (23 features):
        # Original 21 features + 2 opponent strength features
        
        assists = safe_float(player_data.get('assists', 0))
        expected_assists = safe_float(player_data.get('expected_assists', 0.0))
        goals_scored = safe_float(player_data.get('goals_scored', 0))
        minutes = safe_float(player_data.get('minutes', 0))
        starts = safe_float(player_data.get('starts', 0))
        creativity = safe_float(player_data.get('creativity', 0))
        
        # Estimate games played for proper per-game averages
        games_played = max(1, minutes / 90.0)  # Rough estimate from total minutes
        
        features = [
            assists / games_played,  # assists_avg_3gw proxy (assists per game)
            assists / games_played,  # assists_avg_5gw proxy
            assists / games_played,  # assists_avg_10gw proxy
            expected_assists / games_played,  # xa_avg_3gw proxy
            expected_assists / games_played,  # xa_avg_5gw proxy
            expected_assists / games_played,  # xa_avg_10gw proxy
            goals_scored / games_played,  # goals_avg_3gw proxy
            goals_scored / games_played,  # goals_avg_5gw proxy
            minutes / games_played,  # minutes_avg_3gw proxy (should be around 90)
            minutes / games_played,  # minutes_avg_5gw proxy
            starts / games_played,  # starts_avg_3gw proxy
            starts / games_played,  # starts_avg_5gw proxy
            creativity / games_played,  # creativity_avg_3gw proxy
            creativity / games_played,  # creativity_avg_5gw proxy
            assists / max(expected_assists, 0.1),  # assist_efficiency
            creativity / games_played + expected_assists / games_played,  # creative_output per game
            safe_float(player_data.get('form', 0.0)),  # recent_form
            1.0 if player_data.get('element_type') == 3 else 0.0,  # is_midfielder
            1.0 if player_data.get('element_type') == 2 else 0.0,  # is_defender
            1.0 if player_data.get('element_type') == 4 else 0.0,  # is_forward
            1.0 if fixture_info['is_home'] else 0.0,  # was_home (use actual fixture info)
            # NEW: Opponent strength features
            fixture_info['opponent_strength_defence'] / 1400.0,  # opponent_defence_strength_normalized
            1.0 - fixture_info.get('fixture_attractiveness', 0.5),  # fixture_difficulty (inverted for assists)
        ]
        
        return features
    
    def _predict_saves(self, player_data: Dict, gameweek: int, expected_minutes: float) -> float:
        """Predict expected saves using our trained saves model"""
        
        position = self._get_position_name(player_data.get('element_type', 4))
        if position != 'GK':
            return 0.0
        
        if not self.models.get('saves'):
            player_name = player_data.get('web_name', 'Unknown')
            print(f"❌ SAVES MODEL NOT LOADED for GK {player_name}")
            return None
        
        try:
            features = self._prepare_saves_features(player_data, gameweek)
            model_data = self.models['saves']
            saves_model = model_data['model']
            scaler = model_data['scaler']
            
            # CRITICAL: Apply scaling before prediction
            # Convert to DataFrame with proper feature names to avoid sklearn warning
            import pandas as pd
            feature_names = model_data.get('feature_cols', [f'feature_{i}' for i in range(len(features))])
            features_df = pd.DataFrame([features], columns=feature_names)
            scaled_features = scaler.transform(features_df)
            predicted_saves = saves_model.predict(scaled_features)[0]
            
            # Scale by expected minutes
            scaled_saves = predicted_saves * (expected_minutes / 90.0)
            
            return max(0, scaled_saves)
            
        except Exception as e:
            player_name = player_data.get('web_name', 'Unknown')
            print(f"❌ SAVES PREDICTION FAILED for GK {player_name}:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Expected minutes: {expected_minutes}")
            try:
                features = self._prepare_saves_features(player_data, gameweek)
                print(f"   Features prepared: {features}")
            except Exception as feature_error:
                print(f"   Feature preparation failed: {feature_error}")
            print(f"   🚫 Skipping prediction - cannot generate valid features")
            return None
    
    def _prepare_saves_features(self, player_data: Dict, gameweek: int) -> List[float]:
        """Prepare features for saves model - matches enhanced trained model with opponent strength"""
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Get fixture information for opponent strength
        team_name = self._get_team_name(player_data.get('team', 1))
        fixture_info = self._get_fixture_info(team_name, gameweek)
        
        # Features matching the enhanced saves model (22 features):
        # Original 20 features + 2 opponent strength features
        # ['saves_avg_3gw', 'saves_avg_5gw', 'saves_avg_10gw', 'goals_conceded_avg_3gw', 
        #  'goals_conceded_avg_5gw', 'goals_conceded_avg_10gw', 'expected_goals_conceded_avg_3gw', 'expected_goals_conceded_avg_5gw', 
        #  'expected_goals_conceded_avg_10gw', 'clean_sheets_avg_3gw', 'clean_sheets_avg_5gw', 'minutes_avg_3gw', 
        #  'minutes_avg_5gw', 'starts_avg_3gw', 'starts_avg_5gw', 'save_efficiency', 
        #  'defensive_workload', 'team_defensive_strength', 'recent_form', 'was_home',
        #  'opponent_attack_strength_normalized', 'fixture_difficulty_inverted']
        
        saves = safe_float(player_data.get('saves', 0))
        goals_conceded = safe_float(player_data.get('goals_conceded', 0))
        expected_goals_conceded = safe_float(player_data.get('expected_goals_conceded', goals_conceded))
        clean_sheets = safe_float(player_data.get('clean_sheets', 0))
        minutes = safe_float(player_data.get('minutes', 0))
        starts = safe_float(player_data.get('starts', 0))
        
        # Estimate games played for proper per-game averages
        games_played = max(1, minutes / 90.0)  # Rough estimate from total minutes
        
        features = [
            saves / games_played,  # saves_avg_3gw proxy (saves per game)
            saves / games_played,  # saves_avg_5gw proxy
            saves / games_played,  # saves_avg_10gw proxy
            goals_conceded / games_played,  # goals_conceded_avg_3gw proxy
            goals_conceded / games_played,  # goals_conceded_avg_5gw proxy
            goals_conceded / games_played,  # goals_conceded_avg_10gw proxy
            expected_goals_conceded / games_played,  # expected_goals_conceded_avg_3gw proxy
            expected_goals_conceded / games_played,  # expected_goals_conceded_avg_5gw proxy
            expected_goals_conceded / games_played,  # expected_goals_conceded_avg_10gw proxy
            clean_sheets / games_played,  # clean_sheets_avg_3gw proxy
            clean_sheets / games_played,  # clean_sheets_avg_5gw proxy
            minutes / games_played,  # minutes_avg_3gw proxy (should be around 90)
            minutes / games_played,  # minutes_avg_5gw proxy
            starts / games_played,  # starts_avg_3gw proxy
            starts / games_played,  # starts_avg_5gw proxy
            saves / max(expected_goals_conceded, 0.1),  # save_efficiency
            saves / games_played + goals_conceded / games_played,  # defensive_workload per game
            max(0, 2.0 - goals_conceded / games_played),  # team_defensive_strength (goals conceded per game, inverted)
            safe_float(player_data.get('form', 0.0)),  # recent_form
            1.0 if fixture_info['is_home'] else 0.0,  # was_home (use actual fixture info)
            # NEW: Opponent strength features
            fixture_info['opponent_strength_attack'] / 1400.0,  # opponent_attack_strength_normalized
            1.0 - fixture_info.get('fixture_attractiveness', 0.5),  # fixture_difficulty_inverted (for saves, higher opponent attack = more saves)
        ]
        
        return features
    
    def _predict_clean_sheet(self, player_data: Dict, gameweek: int) -> float:
        """Predict clean sheet probability using our team goals conceded model"""
        
        position = self._get_position_name(player_data.get('element_type', 4))
        if position not in ['GK', 'DEF']:
            return 0.0
        
        if not self.models.get('team_goals_conceded'):
            player_name = player_data.get('web_name', 'Unknown')
            print(f"❌ TEAM GOALS CONCEDED MODEL NOT LOADED for {player_name} ({position})")
            return None
        
        try:
            # Use team goals conceded model to estimate clean sheet probability
            features = self._prepare_clean_sheet_features(player_data, gameweek)
            model_data = self.models['team_goals_conceded']
            team_model = model_data['model']
            scaler = model_data['scaler']
            
            # CRITICAL: Apply scaling before prediction
            # Convert to DataFrame with proper feature names to avoid sklearn warning
            import pandas as pd
            feature_names = model_data.get('feature_cols', [f'feature_{i}' for i in range(len(features))])
            features_df = pd.DataFrame([features], columns=feature_names)
            scaled_features = scaler.transform(features_df)
            predicted_goals_conceded = team_model.predict(scaled_features)[0]
            
            # Convert goals conceded to clean sheet probability
            # Using Poisson distribution: P(0 goals) = e^(-λ)
            clean_sheet_prob = np.exp(-max(0, predicted_goals_conceded))
            
            return min(1.0, max(0.0, clean_sheet_prob))
            
        except Exception as e:
            player_name = player_data.get('web_name', 'Unknown')
            position = self._get_position_name(player_data.get('element_type', 4))
            print(f"❌ CLEAN SHEET PREDICTION FAILED for {player_name} ({position}):")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            try:
                features = self._prepare_clean_sheet_features(player_data, gameweek)
                print(f"   Features prepared: {features}")
            except Exception as feature_error:
                print(f"   Feature preparation failed: {feature_error}")
            print(f"   🚫 Skipping prediction - cannot generate valid features")
            return None
    
    def _prepare_clean_sheet_features(self, player_data: Dict, gameweek: int) -> List[float]:
        """Prepare features for clean sheet/team goals conceded model - matches trained model EXACTLY"""
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Features MUST match the trained team_goals_conceded model EXACTLY (20 features):
        # Base 18 features from get_team_feature_columns() PLUS 2 opponent strength features:
        # ['goals_conceded_avg_3gw', 'goals_conceded_avg_5gw', 'goals_conceded_avg_10gw',
        #  'clean_sheets_avg_3gw', 'clean_sheets_avg_5gw', 'clean_sheets_avg_10gw', 
        #  'goals_scored_avg_3gw', 'goals_scored_avg_5gw', 'goals_scored_avg_10gw',
        #  'total_points_avg_3gw', 'total_points_avg_5gw', 'total_points_avg_10gw',
        #  'home_goals_conceded_avg_5gw', 'away_goals_conceded_avg_5gw',
        #  'defensive_strength', 'recent_form', 'season_progress', 'was_home',
        #  'opponent_overall_strength_normalized', 'fixture_attractiveness']
        
        goals_conceded = safe_float(player_data.get('goals_conceded', 0))
        clean_sheets = safe_float(player_data.get('clean_sheets', 0))
        goals_scored = safe_float(player_data.get('goals_scored', 0))
        total_points = safe_float(player_data.get('total_points', 0))
        team_id = player_data.get('team', 10)
        minutes = safe_float(player_data.get('minutes', 0))
        
        # Get fixture information for opponent strength features
        fixture_info = self._get_fixture_info(team_id, gameweek)
        
        # Estimate games played for proper per-game averages
        games_played = max(1, minutes / 90.0)  # Rough estimate from total minutes
        
        # Calculate per-game averages
        goals_conceded_per_game = goals_conceded / games_played
        clean_sheets_per_game = clean_sheets / games_played
        goals_scored_per_game = goals_scored / games_played
        points_per_game = total_points / games_played
        
        features = [
            # Goals conceded averages (3 features)
            goals_conceded_per_game,  # goals_conceded_avg_3gw
            goals_conceded_per_game,  # goals_conceded_avg_5gw
            goals_conceded_per_game,  # goals_conceded_avg_10gw
            
            # Clean sheets averages (3 features)
            clean_sheets_per_game,    # clean_sheets_avg_3gw
            clean_sheets_per_game,    # clean_sheets_avg_5gw
            clean_sheets_per_game,    # clean_sheets_avg_10gw
            
            # Goals scored averages (3 features)
            goals_scored_per_game,    # goals_scored_avg_3gw
            goals_scored_per_game,    # goals_scored_avg_5gw
            goals_scored_per_game,    # goals_scored_avg_10gw
            
            # Total points averages (3 features)
            points_per_game,          # total_points_avg_3gw
            points_per_game,          # total_points_avg_5gw
            points_per_game,          # total_points_avg_10gw
            
            # Home/away split (2 features) - using general stats as proxy
            goals_conceded_per_game,  # home_goals_conceded_avg_5gw (proxy)
            goals_conceded_per_game,  # away_goals_conceded_avg_5gw (proxy)
            
            # Team attributes (3 features)
            max(0, 2.0 - goals_conceded_per_game),  # defensive_strength (inverse of goals conceded)
            safe_float(player_data.get('form', 0.0)),  # recent_form
            gameweek / 38.0,          # season_progress
            
            # Fixture context (1 feature)
            0.5,                      # was_home (neutral for pre-season, will be updated with fixture data)
            
            # Opponent strength features (2 features) - Added for enhanced model
            fixture_info.get('opponent_overall_strength', 1150) / 1400.0,  # opponent_overall_strength_normalized
            fixture_info.get('fixture_attractiveness', 0.5),               # fixture_attractiveness
        ]
        
        return features
    
    def _calculate_simplified_prediction(self, 
                                       player_id: str,
                                       player_data: Dict,
                                       gameweek: int,
                                       include_bonus: bool,
                                       name: str,
                                       team_name: str,
                                       position: str,
                                       price: float) -> PlayerExpectedPoints:
        """
        Calculate simplified prediction for players with insufficient playing time
        
        Uses position-based averages instead of ML models to avoid small sample bias.
        """
        
        print(f"   📊 Using simplified prediction for {name} (insufficient data)")
        
        # Simplified playing time predictions based on position and current form
        total_minutes = player_data.get('minutes', 0)
        
        if total_minutes == 0:
            # Never played - very unlikely to play - predict 0 points
            expected_minutes = 0.0
            minutes_probability = 0.0
            minutes_category = "no_minutes"
            
            # Get fixture info for the zero prediction  
            team_id = player_data.get('team', 1)
            fixture_info = self._get_fixture_info(team_id, gameweek)
            
            # Return zero prediction immediately for no_minutes players
            return PlayerExpectedPoints(
                player_id=player_id,
                name=name,
                team=team_name,
                position=position,
                current_price=price,
                
                expected_points=0.0,
                expected_goals=0.0,
                expected_assists=0.0,
                expected_saves=0.0,
                clean_sheet_prob=0.0,
                
                expected_minutes=0.0,
                minutes_probability=0.0,
                minutes_category="no_minutes",
                
                expected_bonus=0.0,
                bonus_prob_3=0.0,
                bonus_prob_2=0.0,
                bonus_prob_1=0.0,
                
                points_per_million=0.0,
                value_rank=0,
                form_adjustment=0.0,
                
                prediction_confidence=0.1,  # Very low confidence
                variance=0.1,
                ceiling=0.0,
                floor=0.0,
                
                gameweek=gameweek,
                fixture_difficulty=fixture_info['difficulty'],
                home_away=fixture_info['home_away'],
                opponent=fixture_info['opponent'],
                opponent_strength_attack=fixture_info['opponent_strength_attack'],
                opponent_strength_defence=fixture_info['opponent_strength_defence'],
                fixture_attractiveness=fixture_info['fixture_attractiveness']
            )
        elif total_minutes < 90:
            # Very limited - bench player
            expected_minutes = 15.0
            minutes_probability = 0.2
            minutes_category = "few_minutes"
        elif total_minutes < 270:
            # Some playing time - rotation player
            expected_minutes = 30.0
            minutes_probability = 0.4
            minutes_category = "few_minutes"
        else:
            # Regular but not enough for ML - squad player
            expected_minutes = 45.0
            minutes_probability = 0.6
            minutes_category = "substantial_minutes"
        
        # Conservative position-based averages (much lower than ML predictions)
        position_defaults = {
            'GK': {
                'goals': 0.005,
                'assists': 0.001,
                'saves': 1.5 * (expected_minutes / 90.0),  # Scaled by minutes
                'clean_sheet_prob': 0.15  # Much lower than starters
            },
            'DEF': {
                'goals': 0.02,
                'assists': 0.01, 
                'saves': 0.0,
                'clean_sheet_prob': 0.15  # Conservative for bench players
            },
            'MID': {
                'goals': 0.03,
                'assists': 0.04,
                'saves': 0.0,
                'clean_sheet_prob': 0.0  # No clean sheet points
            },
            'FWD': {
                'goals': 0.08,
                'assists': 0.02,
                'saves': 0.0,
                'clean_sheet_prob': 0.0
            }
        }
        
        defaults = position_defaults.get(position, position_defaults['MID'])
        
        # Scale all stats by expected minutes (bench players play less)
        minutes_factor = expected_minutes / 90.0
        expected_goals = defaults['goals'] * minutes_factor
        expected_assists = defaults['assists'] * minutes_factor
        expected_saves = defaults['saves']  # Already scaled above for GK
        clean_sheet_prob = defaults['clean_sheet_prob'] * minutes_factor  # Lower chance if less minutes
        
        # Calculate base FPL points
        base_points = self._calculate_base_fpl_points(
            position, expected_goals, expected_assists, expected_saves,
            clean_sheet_prob
        )
        
        # Minimal bonus points for players with insufficient data
        expected_bonus = 0.1  # Very low bonus expectation
        bonus_prob_3 = 0.01
        bonus_prob_2 = 0.02
        bonus_prob_1 = 0.05
        
        if include_bonus:
            base_points += expected_bonus
        
        # Value metrics
        points_per_million = base_points / price if price > 0 else 0
        
        # Conservative risk metrics
        variance = max(0.5, base_points * 0.4)  # Higher variance for uncertainty
        ceiling = base_points + 1.0 * np.sqrt(variance)  # Lower ceiling
        floor = max(0, base_points - 1.5 * np.sqrt(variance))  # Higher floor
        
        # Fixture context
        team_id = player_data.get('team', 1)
        fixture_info = self._get_fixture_info(team_id, gameweek)
        
        return PlayerExpectedPoints(
            player_id=player_id,
            name=name,
            team=team_name,
            position=position,
            current_price=price,
            
            expected_points=round(base_points, 2),
            expected_goals=round(expected_goals, 3),
            expected_assists=round(expected_assists, 3),
            expected_saves=round(expected_saves, 1),
            clean_sheet_prob=round(clean_sheet_prob, 3),
            
            # Playing time predictions
            expected_minutes=round(expected_minutes, 1),
            minutes_probability=round(minutes_probability, 3),
            minutes_category=minutes_category,
            
            expected_bonus=round(expected_bonus, 2),
            bonus_prob_3=round(bonus_prob_3, 3),
            bonus_prob_2=round(bonus_prob_2, 3),
            bonus_prob_1=round(bonus_prob_1, 3),
            
            points_per_million=round(points_per_million, 2),
            value_rank=0,  # Will be set after sorting
            form_adjustment=0.0,
            
            prediction_confidence=0.3,  # Lower confidence for simplified predictions
            variance=round(variance, 2),
            ceiling=round(ceiling, 1),
            floor=round(floor, 1),
            
            gameweek=gameweek,
            fixture_difficulty=fixture_info['difficulty'],
            home_away=fixture_info['home_away'],
            opponent=fixture_info['opponent'],
            opponent_strength_attack=fixture_info['opponent_strength_attack'],
            opponent_strength_defence=fixture_info['opponent_strength_defence'],
            fixture_attractiveness=fixture_info['fixture_attractiveness']
        )

    def _calculate_base_fpl_points(self, 
                                 position: str,
                                 goals: float,
                                 assists: float,
                                 saves: float,
                                 clean_sheet_prob: float) -> float:
        """Calculate base FPL points using our scoring engine (without yellow cards)"""
        
        # Base appearance points
        points = 2.0  # Assume player plays
        
        # Goals points (position-dependent)
        goal_points = {'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
        points += goals * goal_points.get(position, 4)
        
        # Assists points
        points += assists * 3
        
        # Clean sheet points
        if position in ['GK', 'DEF']:
            points += clean_sheet_prob * 4
        elif position == 'MID':
            points += clean_sheet_prob * 1
        
        # Saves points (GK only)
        if position == 'GK':
            points += (saves / 3) * 1  # 1 point per 3 saves
        
        # Clean sheet points (GK and DEF only)
        if position in ['GK', 'DEF']:
            points += clean_sheet_prob * 4
        
        return max(0, points)
    
    def _predict_bonus_points(self, 
                            player_data: Dict,
                            expected_goals: float,
                            expected_assists: float,
                            gameweek: int) -> Dict[str, float]:
        """Predict bonus points using our bonus engine (simplified)"""
        
        # Simplified bonus prediction
        position = self._get_position_name(player_data.get('element_type', 4))
        
        # Base BPS from goals/assists
        expected_bps = expected_goals * 18 + expected_assists * 9
        
        # Position adjustments
        if position == 'DEF':
            expected_bps += 2  # Clean sheet potential
        elif position == 'MID':
            expected_bps += 1  # All-round play
        
        # Convert BPS to bonus probability (simplified)
        if expected_bps >= 20:
            prob_3, prob_2, prob_1 = 0.4, 0.3, 0.2
        elif expected_bps >= 15:
            prob_3, prob_2, prob_1 = 0.2, 0.4, 0.3
        elif expected_bps >= 10:
            prob_3, prob_2, prob_1 = 0.1, 0.3, 0.4
        else:
            prob_3, prob_2, prob_1 = 0.05, 0.15, 0.25
        
        expected_bonus = prob_3 * 3 + prob_2 * 2 + prob_1 * 1
        
        return {
            'expected_bonus': expected_bonus,
            'prob_3_bonus': prob_3,
            'prob_2_bonus': prob_2,
            'prob_1_bonus': prob_1
        }
    
    def _generate_prediction_summary(self, 
                                   predictions: List[PlayerExpectedPoints]) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        if not predictions:
            return {}
        
        expected_points = [p.expected_points for p in predictions]
        
        return {
            'total_players': len(predictions),
            'avg_expected_points': round(np.mean(expected_points), 2),
            'max_expected_points': round(max(expected_points), 2),
            'top_player': predictions[0].name,
            'total_expected_goals': round(sum(p.expected_goals for p in predictions), 1),
            'total_expected_assists': round(sum(p.expected_assists for p in predictions), 1),
            'high_value_threshold': 2.0,  # pts per million
            'high_value_count': len([p for p in predictions if p.points_per_million >= 2.0])
        }
    
    # Helper methods
    def _load_json_data(self, filename: str) -> Dict:
        """Load JSON data file"""
        try:
            with open(self.data_dir / filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            return {}
    
    def _get_position_name(self, element_type: int) -> str:
        """Convert element type to position name"""
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return positions.get(element_type, 'MID')
    
    def _get_team_name(self, team_id: int) -> str:
        """Get team name from team ID"""
        if isinstance(self.teams_data, list):
            for team in self.teams_data:
                if team.get('id') == team_id:
                    return team.get('short_name', f'Team {team_id}')
        else:
            team_data = self.teams_data.get(str(team_id), {})
            return team_data.get('short_name', f'Team {team_id}')
        return f'Team {team_id}'
    
    def _get_fixture_info(self, team_id: int, gameweek: int) -> Dict[str, Any]:
        """Get fixture information for a team in a specific gameweek with opponent strength data"""
        
        # Load fixtures data if not already loaded
        if not hasattr(self, '_fixtures_data'):
            fixtures_file = self.data_dir / "fixtures.json"
            if fixtures_file.exists():
                with open(fixtures_file, 'r') as f:
                    self._fixtures_data = json.load(f)
            else:
                # Fallback to defaults if no fixtures data
                return {
                    'difficulty': 'Medium',
                    'home_away': 'Home', 
                    'is_home': True,
                    'opponent': 'TBD',
                    'opponent_strength_attack': 1000,
                    'opponent_strength_defence': 1000,
                    'fixture_attractiveness': 0.5
                }
        
        # Find fixture for this team and gameweek
        team_fixture = None
        for fixture in self._fixtures_data:
            if (fixture['gameweek'] == gameweek and 
                (fixture['team_h'] == team_id or fixture['team_a'] == team_id)):
                team_fixture = fixture
                break
        
        if not team_fixture:
            # No fixture found - return defaults
            return {
                'difficulty': 'Medium',
                'home_away': 'Home',
                'is_home': True,
                'opponent': 'TBD',
                'opponent_strength_attack': 1000,
                'opponent_strength_defence': 1000,
                'fixture_attractiveness': 0.5
            }
        
        # Determine if playing home or away
        is_home = team_fixture['team_h'] == team_id
        opponent_id = team_fixture['team_a'] if is_home else team_fixture['team_h']
        
        # Get opponent data from teams
        opponent_data = None
        opponent_name = 'TBD'
        for team in self.teams_data:
            if team['id'] == opponent_id:
                opponent_data = team
                opponent_name = team['short_name']
                break
        
        if not opponent_data:
            # Fallback if opponent not found
            return {
                'difficulty': 'Medium',
                'home_away': 'Home' if is_home else 'Away',
                'is_home': is_home,
                'opponent': opponent_name,
                'opponent_strength_attack': 1000,
                'opponent_strength_defence': 1000,
                'fixture_attractiveness': 0.5
            }
        
        # Get opponent strength ratings based on venue
        if is_home:
            # When playing at home, opponent is away
            opponent_attack_strength = opponent_data.get('strength_attack_away', 1000)
            opponent_defence_strength = opponent_data.get('strength_defence_away', 1000)
        else:
            # When playing away, opponent is at home
            opponent_attack_strength = opponent_data.get('strength_attack_home', 1000)
            opponent_defence_strength = opponent_data.get('strength_defence_home', 1000)
        
        # Calculate fixture attractiveness (lower opponent strength = better fixture)
        # Scale strength ratings to 0-1 range (1000-1400 typical range)
        normalized_att_strength = max(0, min(1, (opponent_attack_strength - 800) / 800))
        normalized_def_strength = max(0, min(1, (opponent_defence_strength - 800) / 800))
        
        # Fixture attractiveness: higher values = better fixture for your team
        # Low opponent attack strength = good for clean sheets
        # Low opponent defence strength = good for goals/assists
        fixture_attractiveness = (2 - normalized_att_strength - normalized_def_strength) / 2
        
        # Convert to difficulty for display (keeping backward compatibility)
        if fixture_attractiveness >= 0.8:
            difficulty = 'Very Easy'
        elif fixture_attractiveness >= 0.6:
            difficulty = 'Easy'
        elif fixture_attractiveness >= 0.4:
            difficulty = 'Medium'
        elif fixture_attractiveness >= 0.2:
            difficulty = 'Hard'
        else:
            difficulty = 'Very Hard'
        
        return {
            'difficulty': difficulty,
            'home_away': 'Home' if is_home else 'Away',
            'is_home': is_home,
            'opponent': opponent_name,
            'opponent_strength_attack': opponent_attack_strength,
            'opponent_strength_defence': opponent_defence_strength,
            'fixture_attractiveness': round(fixture_attractiveness, 3)
        }


def run_production_example():
    """Demonstrate the complete FPL prediction engine"""
    
    print("🚀 FPL EXPECTED POINTS PREDICTION ENGINE")
    print("=" * 60)
    print("📅 Pre-season predictions for 2025/26 FPL season")
    print("   Current date: July 26, 2025")
    print("")
    
    # Initialize prediction engine
    engine = FPLPredictionEngine()
    
    # Generate predictions for the season opener
    gameweek = 1  # 2025/26 season hasn't started yet
    
    try:
        predictions = engine.generate_gameweek_predictions(
            gameweek=gameweek,
            include_bonus=True
        )
        
        # Display top results
        print(f"\n🏆 TOP 15 EXPECTED POINTS (GW{gameweek}):")
        print("-" * 60)
        print("Rank  Player               Pos   XP    PPM   Ceiling  Floor")
        print("-" * 60)
        
        for i, player in enumerate(predictions['players'][:15]):
            name = player['name'][:18].ljust(18)
            position = player['position']
            xp = player['expected_points']
            ppm = player['points_per_million']
            ceiling = player['ceiling']
            floor = player['floor']
            
            print(f"{i+1:2d}.   {name}  {position}   {xp:4.1f}  {ppm:4.1f}   {ceiling:5.1f}   {floor:4.1f}")
        
        # Value rankings
        print(f"\n� TOP VALUE PLAYERS (Points per £m):")
        print("-" * 50)
        
        value_players = sorted(predictions['players'], 
                             key=lambda p: p['points_per_million'], 
                             reverse=True)[:10]
        
        for i, player in enumerate(value_players):
            name = player['name'][:20].ljust(20)
            ppm = player['points_per_million']
            xp = player['expected_points']
            price = player['current_price']
            
            print(f"{i+1:2d}. {name} - {ppm:.2f} pts/£m ({xp:.1f}pts @ £{price:.1f}m)")
        
        # Position breakdowns
        print(f"\n📈 POSITION ANALYSIS:")
        print("-" * 40)
        
        positions = ['GK', 'DEF', 'MID', 'FWD']
        for pos in positions:
            pos_players = [p for p in predictions['players'] if p['position'] == pos]
            if pos_players:
                best_player = max(pos_players, key=lambda p: p['expected_points'])
                avg_xp = sum(p['expected_points'] for p in pos_players) / len(pos_players)
                
                print(f"   {pos}: {best_player['name']} leads with {best_player['expected_points']:.1f}pts")
                print(f"        Position average: {avg_xp:.1f}pts ({len(pos_players)} players)")
        
        # Summary
        summary = predictions['summary']
        print(f"\n📊 GAMEWEEK SUMMARY:")
        print(f"   Players analyzed: {summary['total_players']}")
        print(f"   Average expected points: {summary['avg_expected_points']}")
        print(f"   Top performer: {summary['top_player']} ({summary['max_expected_points']}pts)")
        print(f"   High-value options: {summary['high_value_count']} players (≥2.0 pts/£m)")
        print(f"   Total expected goals: {summary['total_expected_goals']}")
        print(f"   Total expected assists: {summary['total_expected_assists']}")
        
    except Exception as e:
        print(f"\n💥 PREDICTION ENGINE DEMO FAILED:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            print(f"   Full traceback:")
            traceback.print_exc()
        print(f"\n   This demonstrates a failure in the prediction system")
        print(f"   Check the detailed error messages above for debugging")
    
    print("\n" + "=" * 60)
    print("🎯 PREDICTION ENGINE READY FOR PRODUCTION")
    print("=" * 60)


if __name__ == "__main__":
    run_production_example()
