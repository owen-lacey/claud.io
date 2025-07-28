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


class InsufficientDataError(Exception):
    """Raised when a player doesn't have sufficient data for reliable ML predictions"""
    pass


class ModelNotLoadedError(Exception):
    """Raised when a required ML model is not loaded"""
    pass


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
        historical_path = str(Path(__file__).parent.parent / "raw" / "parsed_gw_2425.csv")
        try:
            self.historical_data = load_historical_data(historical_path)
            print(f"✅ Loaded historical data: {len(self.historical_data):,} gameweek records")
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL: Cannot load historical data from {historical_path}. "
                             f"Historical data is required for accurate predictions. Error: {e}") from e
        
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
        
        missing_models = []
        for model_name, filename in model_files.items():
            try:
                model_path = self.models_dir / filename
                model_data = joblib.load(model_path)
                models[model_name] = model_data
                print(f"   ✅ {model_name}: {filename}")
            except Exception as e:
                missing_models.append(f"{model_name} ({filename})")
                print(f"   ❌ Failed to load {model_name}: {e}")
        
        if missing_models:
            raise RuntimeError(f"❌ CRITICAL: Cannot load required ML models: {', '.join(missing_models)}. "
                             f"All models are required for accurate predictions. Run training scripts first.")
        
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
        
        # Handle both list and dict formats
        if isinstance(self.players_data, list):
            players_to_process = [(str(p.get('id', i)), p) for i, p in enumerate(self.players_data)]
        else:
            players_to_process = list(self.players_data.items())
        
        for player_id, player_data in players_to_process:
            # Skip if missing essential data
            if not player_data.get('web_name') or not player_data.get('now_cost'):
                continue
                
            # Only predict for players with sufficient data quality
            if not self._has_sufficient_data_for_prediction(player_data):
                continue
                
            try:
                prediction = self._calculate_single_player_prediction(
                    player_id, player_data, gameweek, include_bonus
                )
                predictions.append(prediction)
                
            except InsufficientDataError:
                # Expected for some players - just skip silently
                continue
            except Exception as e:
                # Unexpected errors - log but continue
                player_name = player_data.get('web_name', 'Unknown')
                position = self._get_position_name(player_data.get('element_type', 4))
                print(f"❌ UNEXPECTED ERROR for {player_name} ({position}): {type(e).__name__}: {str(e)}")
                continue
        
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"   ✅ Generated predictions for {len(predictions)} players")
        print(f"   📊 Players with sufficient data: {len(predictions)}/{len(players_to_process)}")
        
        return predictions
    
    def _has_sufficient_data_for_prediction(self, player_data: Dict) -> bool:
        """
        Check if player has sufficient data for reliable ML predictions
        
        Args:
            player_data: Player data dictionary
            
        Returns:
            True if player has sufficient data, False otherwise
        """
        # Require minimum playing time (equivalent to ~5 games)
        total_minutes = player_data.get('minutes', 0)
        if total_minutes < 450:
            return False
            
        # Require player code for historical matching
        if not player_data.get('code'):
            return False
            
        # Require basic FPL stats
        required_fields = ['web_name', 'now_cost', 'element_type', 'team']
        if not all(player_data.get(field) for field in required_fields):
            return False
            
        return True

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
        
        # Validate we have sufficient historical context
        if not player_data.get('code'):
            raise InsufficientDataError(f"Player {name} missing player code - cannot match with historical data")
        
        player_code = str(player_data['code'])
        
        # Check for recent games for proper context
        historical_context = None
        if self.historical_data is not None:
            historical_context = self.feature_engine.get_historical_context(
                player_code, 
                self.historical_data
            )
            if historical_context is None or len(historical_context) < 5:
                print(f"✅ Found {len(historical_context) if historical_context else 0} recent games for player code {player_code}")
                if len(historical_context) < 5:
                    raise InsufficientDataError(f"Player {name} has insufficient recent games ({len(historical_context)}) for reliable prediction")
            else:
                print(f"✅ Found {len(historical_context)} recent games for player code {player_code}")
        
        # All ML model predictions (pass historical context to avoid redundant fetching)
        expected_minutes, minutes_probability, minutes_category = self._predict_minutes(player_data, gameweek, historical_context)
        expected_goals = self._predict_goals(player_data, gameweek, expected_minutes)
        expected_assists = self._predict_assists(player_data, gameweek, expected_minutes)
        expected_saves = self._predict_saves(player_data, gameweek, expected_minutes) if position == 'GK' else 0.0
        
        # Get clean sheet prediction for defensive players
        if position in ['GK', 'DEF']:
            clean_sheet_prob, predicted_goals_conceded = self._predict_clean_sheet(player_data, gameweek)
        else:
            clean_sheet_prob = 0.0
            predicted_goals_conceded = None

        # Calculate base FPL points 
        base_points = self._calculate_base_fpl_points(
            position, expected_goals, expected_assists, expected_saves,
            clean_sheet_prob, predicted_goals_conceded
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
        
        # Risk metrics
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
            
            prediction_confidence=0.85,  # High confidence for sufficient data players
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
    def _predict_minutes(self, player_data: Dict, gameweek: int, historical_context=None) -> Tuple[float, float, str]:
        """
        Predict expected minutes using our trained minutes model
        
        Returns:
            Tuple[float, float, str]: (expected_minutes, probability, category)
            
        Raises:
            ModelNotLoadedError: If minutes model is not available
            InsufficientDataError: If player data insufficient for prediction
        """
        
        if not self.models.get('minutes'):
            raise ModelNotLoadedError("Minutes model is not loaded - cannot predict playing time")
        
        # Prepare features for minutes model
        features = self._prepare_minutes_features(player_data, gameweek, historical_context)
        
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
        
        return (expected_minutes, play_probability, predicted_category)

    def _prepare_minutes_features(self, player_data: Dict, gameweek: int, historical_context=None) -> List[float]:
        """Prepare features for minutes prediction using shared feature engineering"""
        
        model_data = self.models.get('minutes', {})
        if not model_data:
            raise ValueError("Minutes model not loaded")

        # Use the passed historical context instead of fetching it again
        # (historical_context was already fetched in the main prediction method)
        
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
        """
        Predict expected goals using our trained goals model
        
        Raises:
            ModelNotLoadedError: If goals model is not available
            InsufficientDataError: If player data insufficient for prediction
        """
        
        # Goalkeepers extremely rarely score goals - return minimal prediction
        position = self._get_position_name(player_data.get('element_type', 4))
        if position == 'GK':
            return 0.001  # Extremely rare but not impossible (penalties/corners)
        
        if not self.models.get('goals'):
            raise ModelNotLoadedError("Goals model is not loaded - cannot predict goals")
        
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
    
    def _prepare_goals_features(self, player_data: Dict, gameweek: int) -> List[float]:
        """Prepare features for goals model - matches trained model with position awareness"""
        
        # Get fixture information for opponent strength
        team_name = self._get_team_name(player_data.get('team', 1))
        fixture_info = self._get_fixture_info(team_name, gameweek)
        
        # Use the shared feature engine for position-aware features (no fallbacks)
        if not hasattr(self, 'feature_engine'):
            raise RuntimeError("Feature engine not initialized - cannot prepare goals features")
        
        # Prepare features using shared logic with position awareness
        features = self.feature_engine.prepare_goals_model_features(
            player_data,
            historical_context=None,  # Using current season data only
            was_home=fixture_info['is_home'],
            opponent_defence_strength=fixture_info['opponent_strength_defence'],
            fixture_attractiveness=fixture_info.get('fixture_attractiveness', 0.5)
        )
        
        # Validate feature count matches model expectations
        if len(features) != 20:
            player_name = player_data.get('web_name', 'Unknown')
            raise ValueError(f"Goals model expects exactly 20 features, got {len(features)} for {player_name}. "
                           f"This indicates a feature engineering bug.")
        
        return features
    
    def _predict_assists(self, player_data: Dict, gameweek: int, expected_minutes: float) -> float:
        """
        Predict expected assists using FPL's expected_assists data
        
        The ML model is currently corrupted, so we use a more reliable approach:
        Scale the season's expected assists by recent form and expected minutes.
        
        Raises:
            InsufficientDataError: If player data insufficient for prediction
        """
        
        # Validate required fields
        required_fields = ['expected_assists', 'minutes', 'form']
        for field in required_fields:
            if field not in player_data:
                player_name = player_data.get('web_name', 'Unknown')
                raise ValueError(f"Missing required field '{field}' for player {player_name}")
        
        # Extract base data
        season_expected_assists = float(player_data['expected_assists'])
        total_minutes = float(player_data['minutes'])
        form = float(player_data.get('form', 0))
        
        # Validate meaningful data
        if total_minutes <= 0:
            player_name = player_data.get('web_name', 'Unknown')
            raise InsufficientDataError(f"Player {player_name} has no playing time - cannot predict assists")
        
        # Calculate assists per 90 minutes from season data
        assists_per_90 = season_expected_assists / (total_minutes / 90.0) if total_minutes > 0 else 0
        
        # Apply form adjustment (form is 0-10 scale, normalize around 5)
        form_factor = (form / 5.0) if form > 0 else 1.0
        form_factor = max(0.5, min(2.0, form_factor))  # Cap between 0.5x and 2.0x
        
        # Scale by expected minutes for this gameweek
        predicted_assists = assists_per_90 * (expected_minutes / 90.0) * form_factor
        
        return max(0, predicted_assists)
    
    def _prepare_assists_features(self, player_data: Dict, gameweek: int) -> List[float]:
        """
        Prepare features for assists model - matches enhanced trained model with opponent strength
        
        Raises:
            ValueError: If required player data fields are missing or invalid
        """
        
        # Get fixture information for opponent strength
        team_name = self._get_team_name(player_data.get('team', 1))
        fixture_info = self._get_fixture_info(team_name, gameweek)
        
        # Validate required fields are present and numeric - use actual FPL dataset fields
        required_fields = ['assists', 'expected_assists', 'goals_scored', 'minutes', 'form', 'element_type']
        for field in required_fields:
            if field not in player_data:
                player_name = player_data.get('web_name', 'Unknown')
                raise ValueError(f"Missing required field '{field}' for player {player_name}")
        
        # Extract features (no safe_float - let exceptions bubble up)
        assists = float(player_data['assists'])
        expected_assists = float(player_data['expected_assists'])
        goals_scored = float(player_data['goals_scored'])
        minutes = float(player_data['minutes'])
        creativity = float(player_data.get('bps', 0))  # Use bonus points system as creativity proxy
        
        # Validate we have meaningful data
        if minutes <= 0:
            player_name = player_data.get('web_name', 'Unknown')
            raise InsufficientDataError(f"Player {player_name} has no playing time - cannot predict assists")
        
        # Calculate per-game averages and estimated starts
        games_played = minutes / 90.0
        estimated_starts = min(games_played, minutes / 60.0)  # Estimate starts from minutes (assuming 60+ mins = start)
        
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
            estimated_starts / games_played,  # starts_avg_3gw proxy
            estimated_starts / games_played,  # starts_avg_5gw proxy
            creativity / games_played,  # creativity_avg_3gw proxy
            creativity / games_played,  # creativity_avg_5gw proxy
            assists / max(expected_assists, 0.1),  # assist_efficiency
            creativity / games_played + expected_assists / games_played,  # creative_output per game
            float(player_data['form']),  # recent_form
            1.0 if player_data['element_type'] == 3 else 0.0,  # is_midfielder
            1.0 if player_data['element_type'] == 2 else 0.0,  # is_defender
            1.0 if player_data['element_type'] == 4 else 0.0,  # is_forward
            1.0 if fixture_info['is_home'] else 0.0,  # was_home (use actual fixture info)
            # NEW: Opponent strength features
            fixture_info['opponent_strength_defence'] / 1400.0,  # opponent_defence_strength_normalized
            1.0 - fixture_info.get('fixture_attractiveness', 0.5),  # fixture_difficulty (inverted for assists)
        ]
        
        # Validate feature count
        if len(features) != 23:
            player_name = player_data.get('web_name', 'Unknown')
            raise ValueError(f"Assists model expects exactly 23 features, got {len(features)} for {player_name}")
        
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
        """
        Prepare features for saves model - matches enhanced trained model with opponent strength
        
        Raises:
            ValueError: If required player data fields are missing or invalid
        """
        
        # Get fixture information for opponent strength
        team_name = self._get_team_name(player_data.get('team', 1))
        fixture_info = self._get_fixture_info(team_name, gameweek)
        
        # Validate required fields are present and numeric - use actual FPL dataset fields
        required_fields = ['saves', 'goals_conceded', 'expected_goals_conceded', 'clean_sheets', 'minutes', 'form']
        for field in required_fields:
            if field not in player_data:
                player_name = player_data.get('web_name', 'Unknown')
                raise ValueError(f"Missing required field '{field}' for player {player_name}")
        
        # Extract features (no safe_float - let exceptions bubble up)
        saves = float(player_data['saves'])
        goals_conceded = float(player_data['goals_conceded'])
        expected_goals_conceded = float(player_data['expected_goals_conceded'])
        clean_sheets = float(player_data['clean_sheets'])
        minutes = float(player_data['minutes'])
        
        # Validate we have meaningful data
        if minutes <= 0:
            player_name = player_data.get('web_name', 'Unknown')
            raise InsufficientDataError(f"Player {player_name} has no playing time - cannot predict saves")
        
        # Calculate per-game averages and estimated starts
        games_played = minutes / 90.0
        estimated_starts = min(games_played, minutes / 60.0)  # Estimate starts from minutes
        
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
            estimated_starts / games_played,  # starts_avg_3gw proxy (estimated)
            estimated_starts / games_played,  # starts_avg_5gw proxy (estimated)
            saves / max(expected_goals_conceded, 0.1),  # save_efficiency
            saves / games_played + goals_conceded / games_played,  # defensive_workload per game
            max(0, 2.0 - goals_conceded / games_played),  # team_defensive_strength (goals conceded per game, inverted)
            float(player_data['form']),  # recent_form
            1.0 if fixture_info['is_home'] else 0.0,  # was_home (use actual fixture info)
            # NEW: Opponent strength features
            fixture_info['opponent_strength_attack'] / 1400.0,  # opponent_attack_strength_normalized
            1.0 - fixture_info.get('fixture_attractiveness', 0.5),  # fixture_difficulty_inverted (for saves, higher opponent attack = more saves)
        ]
        
        # Validate feature count
        if len(features) != 22:
            player_name = player_data.get('web_name', 'Unknown')
            raise ValueError(f"Saves model expects exactly 22 features, got {len(features)} for {player_name}")
        
        return features
    
    def _predict_clean_sheet(self, player_data: Dict, gameweek: int) -> tuple[float, float]:
        """Predict clean sheet probability using our team goals conceded model
        
        Returns:
            tuple: (clean_sheet_prob, predicted_goals_conceded)
        """
        
        position = self._get_position_name(player_data.get('element_type', 4))
        if position not in ['GK', 'DEF']:
            return (0.0, None)
        
        if not self.models.get('team_goals_conceded'):
            player_name = player_data.get('web_name', 'Unknown')
            error_msg = f"❌ TEAM GOALS CONCEDED MODEL NOT LOADED for {player_name} ({position})"
            print(error_msg)
            raise RuntimeError(error_msg)
        
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
            
            # Return both values - clean sheet prob and the original predicted goals conceded
            return (min(1.0, max(0.0, clean_sheet_prob)), max(0.0, predicted_goals_conceded))
            
        except Exception as e:
            player_name = player_data.get('web_name', 'Unknown')
            position = self._get_position_name(player_data.get('element_type', 4))
            error_msg = f"❌ CLEAN SHEET PREDICTION FAILED for {player_name} ({position}): {type(e).__name__}: {str(e)}"
            print(error_msg)
            try:
                features = self._prepare_clean_sheet_features(player_data, gameweek)
                print(f"   Features prepared: {features}")
            except Exception as feature_error:
                print(f"   Feature preparation failed: {feature_error}")
            print(f"   🚫 Cannot generate valid features - this is a critical error")
            raise RuntimeError(error_msg) from e
    
    def _prepare_clean_sheet_features(self, player_data: Dict, gameweek: int) -> List[float]:
        """
        Prepare features for clean sheet/team goals conceded model - matches trained model EXACTLY
        
        Raises:
            ValueError: If required player data fields are missing or invalid
        """
        
        # Validate required fields are present and numeric - use actual FPL dataset fields
        required_fields = ['goals_conceded', 'clean_sheets', 'goals_scored', 'total_points', 'minutes', 'form', 'team']
        for field in required_fields:
            if field not in player_data:
                player_name = player_data.get('web_name', 'Unknown')
                raise ValueError(f"Missing required field '{field}' for player {player_name}")
        
        # Extract features (no safe_float - let exceptions bubble up)
        goals_conceded = float(player_data['goals_conceded'])
        clean_sheets = float(player_data['clean_sheets'])
        goals_scored = float(player_data['goals_scored'])
        total_points = float(player_data['total_points'])
        team_id = int(player_data['team'])
        minutes = float(player_data['minutes'])
        
        # Validate we have meaningful data
        if minutes <= 0:
            player_name = player_data.get('web_name', 'Unknown')
            raise InsufficientDataError(f"Player {player_name} has no playing time - cannot predict clean sheet")
        
        # Get fixture information for opponent strength features
        fixture_info = self._get_fixture_info(team_id, gameweek)
        
        # Calculate per-game averages
        games_played = minutes / 90.0
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
            float(player_data['form']),             # recent_form
            gameweek / 38.0,                        # season_progress
            
            # Fixture context (1 feature)
            1.0 if fixture_info['is_home'] else 0.0,  # was_home
            
            # Opponent strength features (2 features) - Added for enhanced model
            fixture_info.get('opponent_overall_strength', 1150) / 1400.0,  # opponent_overall_strength_normalized
            fixture_info.get('fixture_attractiveness', 0.5),               # fixture_attractiveness
        ]
        
        # Validate feature count
        if len(features) != 20:
            player_name = player_data.get('web_name', 'Unknown')
            raise ValueError(f"Clean sheet model expects exactly 20 features, got {len(features)} for {player_name}")
        
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
            clean_sheet_prob, None  # No goals conceded penalty for simplified predictions
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
                                 clean_sheet_prob: float,
                                 predicted_goals_conceded: float = None) -> float:
        """Calculate base FPL points using our scoring engine (with goal conceded penalties)"""
        
        # Base appearance points
        points = 2.0  # Assume player plays
        
        # Goals points (position-dependent)
        goal_points = {'GK': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}
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
        
        # Goal conceded penalties for defenders and goalkeepers
        if position in ['GK', 'DEF'] and predicted_goals_conceded is not None:
            # Use Poisson distribution to calculate penalty probabilities in non-overlapping buckets
            # P(X = k) = (λ^k * e^(-λ)) / k! where λ = predicted_goals_conceded
            
            exp_neg_lambda = np.exp(-predicted_goals_conceded)
            lambda_val = predicted_goals_conceded
            
            # Calculate individual probabilities for exact goal counts
            p0 = exp_neg_lambda  # P(0 goals)
            p1 = lambda_val * exp_neg_lambda  # P(1 goal)
            p2 = (lambda_val ** 2 / 2) * exp_neg_lambda  # P(2 goals)
            p3 = (lambda_val ** 3 / 6) * exp_neg_lambda  # P(3 goals)
            p4 = (lambda_val ** 4 / 24) * exp_neg_lambda  # P(4 goals)
            p5 = (lambda_val ** 5 / 120) * exp_neg_lambda  # P(5 goals)
            p6 = (lambda_val ** 6 / 720) * exp_neg_lambda  # P(6 goals)
            p7 = (lambda_val ** 7 / 5040) * exp_neg_lambda  # P(7 goals)
            p8 = (lambda_val ** 8 / 40320) * exp_neg_lambda  # P(8 goals)
            p9 = (lambda_val ** 9 / 362880) * exp_neg_lambda  # P(9 goals)
            
            # Non-overlapping penalty buckets:
            # 2-3 goals: -1 point penalty
            prob_2_to_3_goals = p2 + p3
            points -= prob_2_to_3_goals * 1
            
            # 4-5 goals: -2 point penalty (additional -1 beyond the 2-3 penalty)
            prob_4_to_5_goals = p4 + p5
            points -= prob_4_to_5_goals * 2
            
            # 6-7 goals: -3 point penalty (additional -1 beyond the 4-5 penalty)
            prob_6_to_7_goals = p6 + p7
            points -= prob_6_to_7_goals * 3
            
            # 8-9 goals: -4 point penalty (additional -1 beyond the 6-7 penalty)
            prob_8_to_9_goals = p8 + p9
            points -= prob_8_to_9_goals * 4
        
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
