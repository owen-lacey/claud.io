# Player Feature Engineering - Shared Logic for Training and Prediction
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json

class PlayerFeatureEngine:
    """
    Shared feature engineering logic for both training and prediction.
    
    This ensures that features are calculated consistently between:
    1. Training phase (using historical gameweek data)  
    2. Prediction phase (using current season data + historical context)
    """
    
    def __init__(self, teams_data: Optional[List[Dict]] = None):
        """
        Initialize the feature engine
        
        Args:
            teams_data: List of team dictionaries with id and name
        """
        self.teams_dict = {}
        if teams_data:
            self.teams_dict = {team['id']: team['name'] for team in teams_data}
    
    def calculate_rolling_features(self, df: pd.DataFrame, 
                                 group_col: str = 'name',
                                 sort_cols: List[str] = None,
                                 rolling_features: List[str] = None) -> pd.DataFrame:
        """
        Calculate rolling window features for training data
        
        Args:
            df: DataFrame with historical gameweek data
            group_col: Column to group by (usually player name)
            sort_cols: Columns to sort by (usually ['name', 'GW'])
            rolling_features: List of features to calculate rolling averages for (optional)
            
        Returns:
            DataFrame with rolling features added
        """
        if sort_cols is None:
            sort_cols = [group_col, 'GW'] if 'GW' in df.columns else [group_col]
            
        # Sort data chronologically
        df_sorted = df.sort_values(sort_cols).copy()
        
        # Define default features to calculate rolling averages for
        if rolling_features is None:
            rolling_features = [
                'minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets',
                'saves', 'goals_conceded', 'yellow_cards', 'starts', 'threat',
                'expected_goals', 'expected_assists', 'expected_goal_involvements',
                'expected_goals_conceded', 'creativity', 'bps'
            ]
        window_sizes = [3, 5, 10]
        
        print(f"📊 Calculating rolling features for {len(rolling_features)} metrics...")
        
        for feature in rolling_features:
            if feature not in df_sorted.columns:
                print(f"⚠️ Feature '{feature}' not found in data, skipping...")
                continue
            
            # Ensure feature is numeric
            try:
                df_sorted[feature] = pd.to_numeric(df_sorted[feature], errors='coerce').fillna(0)
            except Exception as e:
                print(f"⚠️ Could not convert feature '{feature}' to numeric, skipping... Error: {e}")
                continue
                
            for window in window_sizes:
                col_name = f'{feature}_avg_{window}gw'
                
                # Calculate rolling average, grouped by player
                df_sorted[col_name] = (
                    df_sorted.groupby(group_col)[feature]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
        
        # Calculate form (last 3 games total points sum)
        if 'total_points' in df_sorted.columns:
            df_sorted['form'] = (
                df_sorted.groupby(group_col)['total_points']
                .rolling(window=3, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
        
        return df_sorted
    
    def get_historical_context(self, player_identifier: str, 
                             historical_df: pd.DataFrame,
                             num_recent_games: int = 5) -> Optional[Dict]:
        """
        Get recent historical context for a player to calculate rolling averages
        
        Args:
            player_identifier: Player's cross-season consistent code OR name (for backward compatibility)
            historical_df: DataFrame with historical gameweek data (should include 'player_code' column if available)
            num_recent_games: Number of recent games to consider
            
        Returns:
            Dictionary with recent game statistics or None if not found
        """
        # Require player_code column for robust cross-season consistency
        if 'player_code' not in historical_df.columns:
            raise ValueError(f"❌ CRITICAL: 'player_code' column not found in historical data. This is required for cross-season player matching. Please ensure historical data includes player codes.")
        
        # Require player_code identifier (numeric and >= 5 digits)
        if not (player_identifier.isdigit() and len(player_identifier) >= 5):
            raise ValueError(f"❌ CRITICAL: Player identifier '{player_identifier}' is not a valid player code (must be numeric with ≥5 digits). Please use player codes instead of names for reliable matching.")
        
        # Look up player by code (ensure both sides are integers for reliable comparison)
        try:
            player_code_int = int(player_identifier)
        except ValueError:
            raise ValueError(f"❌ CRITICAL: Player identifier '{player_identifier}' cannot be converted to integer.")
        
        player_games = historical_df[
            historical_df['player_code'] == player_code_int
        ].sort_values('GW').tail(num_recent_games)
        
        if len(player_games) > 0:
            print(f"✅ Found {len(player_games)} recent games for player code {player_identifier}")
        else:
            print(f"⚠️ No historical games found for player code {player_identifier}")
            return None
            
        # Calculate averages from recent games
        recent_stats = {}
        for window in [3, 5, 10]:
            window_games = player_games.tail(window)
            
            if len(window_games) > 0:
                recent_stats.update({
                    f'minutes_avg_{window}gw': window_games['minutes'].mean(),
                    f'total_points_avg_{window}gw': window_games['total_points'].mean(),
                    f'goals_scored_avg_{window}gw': window_games['goals_scored'].mean(),
                    f'assists_avg_{window}gw': window_games['assists'].mean(),
                    f'clean_sheets_avg_{window}gw': window_games['clean_sheets'].mean(),
                    f'saves_avg_{window}gw': window_games.get('saves', pd.Series([0]*len(window_games))).mean(),
                    f'goals_conceded_avg_{window}gw': window_games.get('goals_conceded', pd.Series([0]*len(window_games))).mean(),
                    f'yellow_cards_avg_{window}gw': window_games.get('yellow_cards', pd.Series([0]*len(window_games))).mean(),
                    f'starts_avg_{window}gw': window_games.get('starts', pd.Series([0]*len(window_games))).mean(),
                    f'threat_avg_{window}gw': window_games.get('threat', pd.Series([0]*len(window_games))).mean(),
                    f'expected_goals_avg_{window}gw': window_games.get('expected_goals', pd.Series([0]*len(window_games))).mean(),
                    f'expected_assists_avg_{window}gw': window_games.get('expected_assists', pd.Series([0]*len(window_games))).mean(),
                    f'creativity_avg_{window}gw': window_games.get('creativity', pd.Series([0]*len(window_games))).mean(),
                    f'bps_avg_{window}gw': window_games.get('bps', pd.Series([0]*len(window_games))).mean(),
                })
        
        # Form = sum of last 3 games total points
        form_games = player_games.tail(3)
        recent_stats['form'] = form_games['total_points'].sum() if len(form_games) > 0 else 0.0
        
        return recent_stats
    
    def prepare_minutes_model_features(self, 
                                     player_data: Dict,
                                     historical_context: Optional[Dict] = None,
                                     position_encoder = None,
                                     team_encoder = None,
                                     fixture_info: Optional[Dict] = None) -> List[float]:
        """
        Prepare features for minutes model prediction
        
        Args:
            player_data: Current player data dictionary
            historical_context: Recent game statistics (from get_historical_context)
            position_encoder: Trained LabelEncoder for positions
            team_encoder: Trained LabelEncoder for teams
            fixture_info: Fixture information with opponent strength
            
        Returns:
            List of feature values in the correct order (12 features total)
        """
        
        # Get position and team encodings
        position = self._get_position_name(player_data.get('element_type', 4))
        team_id = player_data.get('team', 1)
        
        if position_encoder:
            try:
                position_encoded = position_encoder.transform([position])[0]
            except ValueError:
                position_encoded = 0  # Unknown position
        else:
            position_encoded = 0
            
        if team_encoder:
            try:
                # Ensure team_id is properly formatted for encoder
                team_name = self.teams_dict.get(team_id, str(team_id))
                team_encoded = team_encoder.transform([team_name])[0]
            except (ValueError, TypeError, KeyError):
                team_encoded = 0  # Unknown team
        else:
            team_encoded = 0
        
        # Get form from current data or historical context
        form = float(player_data.get('form', 0.0))
        
        # Get rolling averages from historical context
        if historical_context:
            features = [
                position_encoded,
                team_encoded,
                form,
                historical_context.get('minutes_avg_3gw', 0.0),
                historical_context.get('minutes_avg_5gw', 0.0),
                historical_context.get('total_points_avg_3gw', 0.0),
                historical_context.get('total_points_avg_5gw', 0.0),
                historical_context.get('goals_scored_avg_3gw', 0.0),
                historical_context.get('assists_avg_3gw', 0.0),
                historical_context.get('clean_sheets_avg_3gw', 0.0),
            ]
        else:
            # Fallback to season averages if no historical context
            print(f"⚠️ No historical context for player, using season averages as fallback")
            
            # Estimate games played from minutes
            total_minutes = float(player_data.get('minutes', 0))
            games_played = max(1, total_minutes / 90.0)
            
            total_points = float(player_data.get('total_points', 0))
            goals_scored = float(player_data.get('goals_scored', 0))
            assists = float(player_data.get('assists', 0))
            clean_sheets = float(player_data.get('clean_sheets', 0))
            
            features = [
                position_encoded,
                team_encoded,
                form,
                total_minutes / games_played,  # minutes_avg_3gw approximation
                total_minutes / games_played,  # minutes_avg_5gw approximation  
                total_points / games_played,   # total_points_avg_3gw approximation
                total_points / games_played,   # total_points_avg_5gw approximation
                goals_scored / games_played,   # goals_scored_avg_3gw approximation
                assists / games_played,        # assists_avg_3gw approximation
                clean_sheets / games_played,   # clean_sheets_avg_3gw approximation
            ]
        
        # Add opponent strength features (required for enhanced minutes model)
        if fixture_info:
            opponent_overall_strength_normalized = fixture_info.get('opponent_overall_strength', 1150) / 1400.0
            fixture_attractiveness = fixture_info.get('fixture_attractiveness', 0.5)
        else:
            # Default values when no fixture info available
            opponent_overall_strength_normalized = 0.82  # Default ~1150/1400
            fixture_attractiveness = 0.5  # Neutral fixture
        
        # Append opponent strength features (features 10 and 11)
        features.extend([
            opponent_overall_strength_normalized,
            fixture_attractiveness
        ])
        
        return features
    
    def prepare_goals_model_features(self, 
                                   player_data: Dict,
                                   historical_context: Optional[Dict] = None,
                                   position_encoder = None,
                                   team_encoder = None,
                                   was_home: bool = True,
                                   opponent_defence_strength: float = 1100.0,
                                   fixture_attractiveness: float = 0.5) -> List[float]:
        """
        Prepare features for expected goals model prediction
        
        This method ALWAYS uses the position-aware 20-feature format that includes
        all positions (GK, DEF, MID, FWD) with position one-hot encoding.
        
        Args:
            player_data: Current player data dictionary
            historical_context: Recent game statistics
            position_encoder: Trained LabelEncoder for positions (optional)
            team_encoder: Trained LabelEncoder for teams (optional)
            was_home: Whether playing at home
            opponent_defence_strength: Opponent's defensive strength (raw value)
            fixture_attractiveness: Fixture difficulty score (0-1, higher = easier)
            
        Returns:
            List of 20 feature values for position-aware goals model
        """
        
        # Get basic info
        position = self._get_position_name(player_data.get('element_type', 4))
        is_forward = 1.0 if position == 'FWD' else 0.0
        is_midfielder = 1.0 if position == 'MID' else 0.0
        is_defender = 1.0 if position == 'DEF' else 0.0
        was_home_encoded = 1.0 if was_home else 0.0
        
        # Normalize opponent defence strength (same as training)
        opponent_defence_strength_normalized = opponent_defence_strength / 1400.0
        
        if historical_context:
            # Calculate goal efficiency
            goals_3gw = historical_context.get('goals_scored_avg_3gw', 0.0)
            xg_3gw = historical_context.get('expected_goals_avg_3gw', 0.1)
            goal_efficiency = goals_3gw / max(xg_3gw, 0.1)
            
            # Recent form
            recent_form = historical_context.get('form', 0.0)
            
            # Position-aware feature set (20 features total)
            features = [
                historical_context.get('goals_scored_avg_3gw', 0.0),     # goals_avg_3gw
                historical_context.get('goals_scored_avg_5gw', 0.0),     # goals_avg_5gw  
                historical_context.get('goals_scored_avg_10gw', 0.0),    # goals_avg_10gw
                historical_context.get('expected_goals_avg_3gw', 0.0),   # xg_avg_3gw
                historical_context.get('expected_goals_avg_5gw', 0.0),   # xg_avg_5gw
                historical_context.get('expected_goals_avg_10gw', 0.0),  # xg_avg_10gw
                historical_context.get('minutes_avg_3gw', 0.0),          # minutes_avg_3gw
                historical_context.get('minutes_avg_5gw', 0.0),          # minutes_avg_5gw
                historical_context.get('starts_avg_3gw', 0.0),           # starts_avg_3gw
                historical_context.get('starts_avg_5gw', 0.0),           # starts_avg_5gw
                historical_context.get('threat_avg_3gw', 0.0),           # threat_avg_3gw
                historical_context.get('threat_avg_5gw', 0.0),           # threat_avg_5gw
                goal_efficiency,                                         # goal_efficiency
                recent_form,                                             # recent_form
                is_forward,                                              # is_forward
                is_midfielder,                                           # is_midfielder
                is_defender,                                             # is_defender
                was_home_encoded,                                        # was_home
                opponent_defence_strength_normalized,                    # opponent_defence_strength_normalized
                fixture_attractiveness,                                  # fixture_attractiveness
            ]
        else:
            # Fallback to season averages
            games_played = max(1, float(player_data.get('minutes', 0)) / 90.0)
            goals_scored = float(player_data.get('goals_scored', 0))
            expected_goals = float(player_data.get('expected_goals', 0))
            total_points = float(player_data.get('total_points', 0))
            minutes = float(player_data.get('minutes', 0))
            threat = float(player_data.get('threat', 0))
            
            goal_efficiency = goals_scored / max(expected_goals, 0.1)
            
            # Position-aware fallback features (20 features total)
            features = [
                goals_scored / games_played,      # goals_avg_3gw proxy
                goals_scored / games_played,      # goals_avg_5gw proxy
                goals_scored / games_played,      # goals_avg_10gw proxy
                expected_goals / games_played,    # xg_avg_3gw proxy
                expected_goals / games_played,    # xg_avg_5gw proxy
                expected_goals / games_played,    # xg_avg_10gw proxy
                minutes / games_played,           # minutes_avg_3gw proxy
                minutes / games_played,           # minutes_avg_5gw proxy
                1.0,                              # starts_avg_3gw proxy (assume starting if playing)
                1.0,                              # starts_avg_5gw proxy
                threat / games_played,            # threat_avg_3gw proxy
                threat / games_played,            # threat_avg_5gw proxy
                goal_efficiency,                  # goal_efficiency
                player_data.get('form', 0.0),    # recent_form (proper FPL form field)
                is_forward,                       # is_forward
                is_midfielder,                    # is_midfielder
                is_defender,                      # is_defender
                was_home_encoded,                 # was_home
                opponent_defence_strength_normalized,  # opponent_defence_strength_normalized
                fixture_attractiveness,           # fixture_attractiveness
            ]
        
        # Defensive check: ensure we always return exactly 20 features
        if len(features) != 20:
            raise ValueError(f"❌ CRITICAL: Goals model expects exactly 20 features, got {len(features)}. "
                           f"This indicates a bug in feature preparation.")
        
        return features
    
    def get_goals_model_feature_columns(self) -> List[str]:
        """
        Get feature column names for goals model in the correct order.
        This must match the order in prepare_goals_model_features.
        
        ALWAYS returns the position-aware 20-feature format.
        
        Returns:
            List of 20 feature column names
        """
        return [
            # Goal scoring rolling averages (3, 5, 10 gameweeks) - 6 features
            'goals_scored_avg_3gw', 'goals_scored_avg_5gw', 'goals_scored_avg_10gw',
            'expected_goals_avg_3gw', 'expected_goals_avg_5gw', 'expected_goals_avg_10gw',
            # Playing time rolling averages - 4 features  
            'minutes_avg_3gw', 'minutes_avg_5gw', 'starts_avg_3gw', 'starts_avg_5gw',
            # Threat rolling averages - 2 features
            'threat_avg_3gw', 'threat_avg_5gw', 
            # Derived features - 2 features
            'goal_efficiency', 'recent_form',
            # Position features (one-hot encoded) - 3 features
            'is_forward', 'is_midfielder', 'is_defender',
            # Venue - 1 feature
            'was_home',
            # Opponent strength features - 2 features
            'opponent_defence_strength_normalized', 'fixture_attractiveness'
        ]
    
    def prepare_assists_model_features(self, 
                                     player_data: Dict,
                                     historical_context: Optional[Dict] = None,
                                     position_encoder = None,
                                     team_encoder = None,
                                     was_home: bool = True,
                                     opponent_defence_strength: float = 1100.0,
                                     fixture_attractiveness: float = 0.5) -> List[float]:
        """
        Prepare features for expected assists model prediction
        
        Args:
            player_data: Current player data dictionary
            historical_context: Recent game statistics  
            position_encoder: Trained LabelEncoder for positions (optional)
            team_encoder: Trained LabelEncoder for teams (optional)
            was_home: Whether playing at home
            opponent_defence_strength: Opponent's defensive strength (raw value)
            fixture_attractiveness: Fixture difficulty score (0-1, higher = easier)
            
        Returns:
            List of feature values for assists model
        """
        
        position = self._get_position_name(player_data.get('element_type', 4))
        is_midfielder = 1.0 if position == 'MID' else 0.0
        is_defender = 1.0 if position == 'DEF' else 0.0
        is_forward = 1.0 if position == 'FWD' else 0.0
        was_home_encoded = 1.0 if was_home else 0.0
        
        # Normalize opponent defence strength (same as training)
        opponent_defence_strength_normalized = opponent_defence_strength / 1400.0
        
        if historical_context:
            # Calculate creative efficiency
            assists_3gw = historical_context.get('assists_avg_3gw', 0.0)
            xa_3gw = historical_context.get('expected_assists_avg_3gw', 0.1)
            assist_efficiency = assists_3gw / max(xa_3gw, 0.1)
            
            # Creative output
            creativity_3gw = historical_context.get('creativity_avg_3gw', 0.0)
            creativity_5gw = historical_context.get('creativity_avg_5gw', 0.0)
            creative_output = (creativity_3gw + creativity_5gw) / 2.0
            
            features = [
                historical_context.get('assists_avg_3gw', 0.0),           # assists_avg_3gw
                historical_context.get('assists_avg_5gw', 0.0),           # assists_avg_5gw
                historical_context.get('assists_avg_10gw', 0.0),          # assists_avg_10gw
                historical_context.get('expected_assists_avg_3gw', 0.0),  # expected_assists_avg_3gw
                historical_context.get('expected_assists_avg_5gw', 0.0),  # expected_assists_avg_5gw
                historical_context.get('expected_assists_avg_10gw', 0.0), # expected_assists_avg_10gw
                historical_context.get('goals_scored_avg_3gw', 0.0),      # goals_scored_avg_3gw
                historical_context.get('goals_scored_avg_5gw', 0.0),      # goals_scored_avg_5gw
                historical_context.get('minutes_avg_3gw', 0.0),           # minutes_avg_3gw
                historical_context.get('minutes_avg_5gw', 0.0),           # minutes_avg_5gw
                historical_context.get('starts_avg_3gw', 0.0),            # starts_avg_3gw
                historical_context.get('starts_avg_5gw', 0.0),            # starts_avg_5gw
                historical_context.get('creativity_avg_3gw', 0.0),        # creativity_avg_3gw
                historical_context.get('creativity_avg_5gw', 0.0),        # creativity_avg_5gw
                assist_efficiency,                                        # assist_efficiency
                creative_output,                                          # creative_output
                historical_context.get('form', 0.0),                     # recent_form
                is_midfielder,                                            # is_midfielder
                is_defender,                                              # is_defender
                is_forward,                                               # is_forward
                was_home_encoded,                                         # was_home
                # NEW: Opponent strength features (matches training)
                opponent_defence_strength_normalized,                     # opponent_defence_strength_normalized
                fixture_attractiveness,                                   # fixture_attractiveness
            ]
        else:
            # Fallback to season averages
            games_played = max(1, float(player_data.get('minutes', 0)) / 90.0)
            assists = float(player_data.get('assists', 0))
            expected_assists = float(player_data.get('expected_assists', 0))
            creativity = float(player_data.get('creativity', 0))
            minutes = float(player_data.get('minutes', 0))
            total_points = float(player_data.get('total_points', 0))
            goals_scored = float(player_data.get('goals_scored', 0))
            
            assist_efficiency = assists / max(expected_assists, 0.1)
            creative_output = creativity / games_played
            
            features = [
                assists / games_played,           # assists_avg_3gw proxy
                assists / games_played,           # assists_avg_5gw proxy
                assists / games_played,           # assists_avg_10gw proxy
                expected_assists / games_played,  # expected_assists_avg_3gw proxy
                expected_assists / games_played,  # expected_assists_avg_5gw proxy
                expected_assists / games_played,  # expected_assists_avg_10gw proxy
                goals_scored / games_played,      # goals_scored_avg_3gw proxy
                goals_scored / games_played,      # goals_scored_avg_5gw proxy
                minutes / games_played,           # minutes_avg_3gw proxy
                minutes / games_played,           # minutes_avg_5gw proxy
                1.0,                              # starts_avg_3gw proxy
                1.0,                              # starts_avg_5gw proxy
                creativity / games_played,        # creativity_avg_3gw proxy
                creativity / games_played,        # creativity_avg_5gw proxy
                assist_efficiency,                # assist_efficiency
                creative_output,                  # creative_output
                total_points / games_played,      # recent_form proxy
                is_midfielder,                    # is_midfielder
                is_defender,                      # is_defender
                is_forward,                       # is_forward
                was_home_encoded,                 # was_home
                # NEW: Opponent strength features (matches training)
                opponent_defence_strength_normalized,                     # opponent_defence_strength_normalized
                fixture_attractiveness,                                   # fixture_attractiveness
            ]
        
        return features
    
    def get_assists_model_feature_columns(self) -> List[str]:
        """
        Get feature column names for assists model in the correct order.
        This must match the order in prepare_assists_model_features.
        
        Returns:
            List of feature column names (23 features total)
        """
        return [
            # Assist rolling averages (3, 5, 10 gameweeks) - 6 features
            'assists_avg_3gw', 'assists_avg_5gw', 'assists_avg_10gw',
            'expected_assists_avg_3gw', 'expected_assists_avg_5gw', 'expected_assists_avg_10gw',
            # Goal scoring rolling averages - 2 features
            'goals_scored_avg_3gw', 'goals_scored_avg_5gw',
            # Playing time rolling averages - 4 features
            'minutes_avg_3gw', 'minutes_avg_5gw', 'starts_avg_3gw', 'starts_avg_5gw',
            # Creativity rolling averages - 2 features
            'creativity_avg_3gw', 'creativity_avg_5gw',
            # Derived features - 3 features
            'assist_efficiency', 'creative_output', 'recent_form',
            # Position indicators - 3 features
            'is_midfielder', 'is_defender', 'is_forward',
            # Home/away indicator - 1 feature
            'was_home',
            # Opponent strength features - 2 features
            'opponent_defence_strength_normalized', 'fixture_attractiveness'
        ]
    
    def prepare_saves_model_features(self, 
                                   player_data: Dict,
                                   historical_context: Optional[Dict] = None,
                                   position_encoder = None,
                                   team_encoder = None,
                                   was_home: bool = True,
                                   opponent_attack_strength: float = 1100.0,
                                   fixture_attractiveness: float = 0.5) -> List[float]:
        """
        Prepare features for saves model prediction
        
        Args:
            player_data: Current player data dictionary
            historical_context: Recent game statistics  
            position_encoder: Trained LabelEncoder for positions (optional)
            team_encoder: Trained LabelEncoder for teams (optional)
            was_home: Whether playing at home
            opponent_attack_strength: Opponent's attacking strength (raw value)
            fixture_attractiveness: Fixture difficulty score (0-1, higher = easier)
            
        Returns:
            List of feature values for saves model (22 features total)
        """
        
        was_home_encoded = 1.0 if was_home else 0.0
        
        # Normalize opponent attack strength (same as training)
        opponent_attack_strength_normalized = opponent_attack_strength / 1400.0
        
        # For saves, we invert fixture attractiveness - harder fixtures = more saves needed
        fixture_difficulty_inverted = 1.0 - fixture_attractiveness
        
        if historical_context:
            # Calculate save efficiency
            saves_3gw = historical_context.get('saves_avg_3gw', 0.0)
            gc_3gw = historical_context.get('goals_conceded_avg_3gw', 0.1)
            save_efficiency = saves_3gw / max(gc_3gw, 0.1)
            
            # Defensive workload (expected vs actual goals conceded)
            xgc_5gw = historical_context.get('expected_goals_conceded_avg_5gw', 0.0)
            gc_5gw = historical_context.get('goals_conceded_avg_5gw', 0.0)
            defensive_workload = xgc_5gw - gc_5gw
            
            # Team defensive strength (inverse of goals conceded)
            team_defensive_strength = 1.0 / (gc_5gw + 0.1)
            
            # Recent form
            recent_form = historical_context.get('form', 0.0)
            
            features = [
                historical_context.get('saves_avg_3gw', 0.0),              # saves_avg_3gw
                historical_context.get('saves_avg_5gw', 0.0),              # saves_avg_5gw
                historical_context.get('saves_avg_10gw', 0.0),             # saves_avg_10gw
                historical_context.get('goals_conceded_avg_3gw', 0.0),     # goals_conceded_avg_3gw
                historical_context.get('goals_conceded_avg_5gw', 0.0),     # goals_conceded_avg_5gw
                historical_context.get('goals_conceded_avg_10gw', 0.0),    # goals_conceded_avg_10gw
                historical_context.get('expected_goals_conceded_avg_3gw', 0.0),  # expected_goals_conceded_avg_3gw
                historical_context.get('expected_goals_conceded_avg_5gw', 0.0),  # expected_goals_conceded_avg_5gw
                historical_context.get('expected_goals_conceded_avg_10gw', 0.0), # expected_goals_conceded_avg_10gw
                historical_context.get('clean_sheets_avg_3gw', 0.0),       # clean_sheets_avg_3gw
                historical_context.get('clean_sheets_avg_5gw', 0.0),       # clean_sheets_avg_5gw
                historical_context.get('minutes_avg_3gw', 0.0),            # minutes_avg_3gw
                historical_context.get('minutes_avg_5gw', 0.0),            # minutes_avg_5gw
                historical_context.get('starts_avg_3gw', 0.0),             # starts_avg_3gw
                historical_context.get('starts_avg_5gw', 0.0),             # starts_avg_5gw
                save_efficiency,                                           # save_efficiency
                defensive_workload,                                        # defensive_workload
                team_defensive_strength,                                   # team_defensive_strength
                recent_form,                                               # recent_form
                was_home_encoded,                                          # was_home
                # NEW: Opponent strength features (matches training)
                opponent_attack_strength_normalized,                       # opponent_attack_strength_normalized
                fixture_difficulty_inverted,                               # fixture_difficulty_inverted
            ]
        else:
            # Fallback to season averages
            games_played = max(1, float(player_data.get('minutes', 0)) / 90.0)
            saves = float(player_data.get('saves', 0))
            goals_conceded = float(player_data.get('goals_conceded', 0))
            expected_goals_conceded = float(player_data.get('expected_goals_conceded', 0))
            clean_sheets = float(player_data.get('clean_sheets', 0))
            minutes = float(player_data.get('minutes', 0))
            total_points = float(player_data.get('total_points', 0))
            
            save_efficiency = saves / max(goals_conceded, 0.1)
            defensive_workload = expected_goals_conceded / games_played - goals_conceded / games_played
            team_defensive_strength = 1.0 / (goals_conceded / games_played + 0.1)
            
            features = [
                saves / games_played,                 # saves_avg_3gw proxy
                saves / games_played,                 # saves_avg_5gw proxy
                saves / games_played,                 # saves_avg_10gw proxy
                goals_conceded / games_played,        # goals_conceded_avg_3gw proxy
                goals_conceded / games_played,        # goals_conceded_avg_5gw proxy
                goals_conceded / games_played,        # goals_conceded_avg_10gw proxy
                expected_goals_conceded / games_played,  # expected_goals_conceded_avg_3gw proxy
                expected_goals_conceded / games_played,  # expected_goals_conceded_avg_5gw proxy
                expected_goals_conceded / games_played,  # expected_goals_conceded_avg_10gw proxy
                clean_sheets / games_played,          # clean_sheets_avg_3gw proxy
                clean_sheets / games_played,          # clean_sheets_avg_5gw proxy
                minutes / games_played,               # minutes_avg_3gw proxy
                minutes / games_played,               # minutes_avg_5gw proxy
                1.0,                                  # starts_avg_3gw proxy
                1.0,                                  # starts_avg_5gw proxy
                save_efficiency,                      # save_efficiency
                defensive_workload,                   # defensive_workload
                team_defensive_strength,              # team_defensive_strength
                total_points / games_played,          # recent_form proxy
                was_home_encoded,                     # was_home
                # NEW: Opponent strength features (matches training)
                opponent_attack_strength_normalized,  # opponent_attack_strength_normalized
                fixture_difficulty_inverted,          # fixture_difficulty_inverted
            ]
        
        return features
    
    def get_saves_model_feature_columns(self) -> List[str]:
        """
        Get feature column names for saves model in the correct order.
        This must match the order in prepare_saves_model_features.
        
        Returns:
            List of feature column names (22 features total)
        """
        return [
            # Save rolling averages (3, 5, 10 gameweeks) - 3 features  
            'saves_avg_3gw', 'saves_avg_5gw', 'saves_avg_10gw',
            # Goals conceded rolling averages - 3 features
            'goals_conceded_avg_3gw', 'goals_conceded_avg_5gw', 'goals_conceded_avg_10gw',
            # Expected goals conceded rolling averages - 3 features
            'expected_goals_conceded_avg_3gw', 'expected_goals_conceded_avg_5gw', 'expected_goals_conceded_avg_10gw',
            # Clean sheets rolling averages - 2 features
            'clean_sheets_avg_3gw', 'clean_sheets_avg_5gw',
            # Minutes and starts rolling averages - 4 features
            'minutes_avg_3gw', 'minutes_avg_5gw', 'starts_avg_3gw', 'starts_avg_5gw',
            # Derived features - 4 features
            'save_efficiency', 'defensive_workload', 'team_defensive_strength', 'recent_form',
            # Home/away indicator - 1 feature
            'was_home',
            # Opponent strength features - 2 features
            'opponent_attack_strength_normalized', 'fixture_difficulty_inverted'
        ]
    
    def prepare_team_goals_conceded_features(self, 
                                           team_stats: Dict, 
                                           was_home: bool = True,
                                           opponent_attack_strength: float = 1100.0,
                                           fixture_attractiveness: float = 0.5) -> List[float]:
        """
        Prepare features for team goals conceded model prediction
        
        Args:
            team_stats: Dictionary with team recent statistics
            was_home: Whether team is playing at home
            opponent_attack_strength: Opponent's attacking strength (raw value)
            fixture_attractiveness: Fixture difficulty score (0-1, higher = easier)
            
        Returns:
            List of feature values for team goals conceded model (20 features total)
        """
        
        was_home_encoded = 1.0 if was_home else 0.0
        
        # Normalize opponent attack strength (same as training)
        opponent_attack_strength_normalized = opponent_attack_strength / 1400.0
        
        # Fixture difficulty (same as fixture_attractiveness for team goals conceded)
        fixture_difficulty = fixture_attractiveness
        
        # Prepare features in the order expected by the model
        features = [
            # Team defensive rolling averages (3, 5, 10 gameweeks)
            team_stats.get('goals_conceded_avg_3gw', 1.0),          # goals_conceded_avg_3gw
            team_stats.get('goals_conceded_avg_5gw', 1.0),          # goals_conceded_avg_5gw
            team_stats.get('goals_conceded_avg_10gw', 1.0),         # goals_conceded_avg_10gw
            team_stats.get('clean_sheets_avg_3gw', 0.3),            # clean_sheets_avg_3gw
            team_stats.get('clean_sheets_avg_5gw', 0.3),            # clean_sheets_avg_5gw
            team_stats.get('clean_sheets_avg_10gw', 0.3),           # clean_sheets_avg_10gw
            team_stats.get('goals_scored_avg_3gw', 1.0),            # goals_scored_avg_3gw
            team_stats.get('goals_scored_avg_5gw', 1.0),            # goals_scored_avg_5gw
            team_stats.get('goals_scored_avg_10gw', 1.0),           # goals_scored_avg_10gw
            team_stats.get('total_points_avg_3gw', 20.0),           # total_points_avg_3gw
            team_stats.get('total_points_avg_5gw', 20.0),           # total_points_avg_5gw
            team_stats.get('total_points_avg_10gw', 20.0),          # total_points_avg_10gw
            team_stats.get('home_goals_conceded_avg_5gw', 1.0),     # home_goals_conceded_avg_5gw
            team_stats.get('away_goals_conceded_avg_5gw', 1.0),     # away_goals_conceded_avg_5gw
            team_stats.get('defensive_strength', 1.0),              # defensive_strength
            team_stats.get('recent_form', 20.0),                    # recent_form
            team_stats.get('season_progress', 15),                  # season_progress
            was_home_encoded,                                       # was_home
            # NEW: Opponent strength features (matches training)
            opponent_attack_strength_normalized,                    # opponent_attack_strength_normalized
            fixture_difficulty,                                     # fixture_difficulty
        ]
        
        return features
    
    def get_team_goals_conceded_feature_columns(self) -> List[str]:
        """
        Get feature column names for team goals conceded model in the correct order.
        This must match the order in prepare_team_goals_conceded_features.
        
        Returns:
            List of feature column names (20 features total)
        """
        return [
            # Team defensive rolling averages (3, 5, 10 gameweeks) - 12 features
            'goals_conceded_avg_3gw', 'goals_conceded_avg_5gw', 'goals_conceded_avg_10gw',
            'clean_sheets_avg_3gw', 'clean_sheets_avg_5gw', 'clean_sheets_avg_10gw',
            'goals_scored_avg_3gw', 'goals_scored_avg_5gw', 'goals_scored_avg_10gw',
            'total_points_avg_3gw', 'total_points_avg_5gw', 'total_points_avg_10gw',
            # Home/away specific features - 2 features
            'home_goals_conceded_avg_5gw', 'away_goals_conceded_avg_5gw',
            # Derived team features - 3 features
            'defensive_strength', 'recent_form', 'season_progress',
            # Home/away indicator - 1 feature
            'was_home',
            # Opponent strength features - 2 features
            'opponent_attack_strength_normalized', 'fixture_difficulty'
        ]
    
    def add_opponent_strength_features_to_team_data(self, team_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add opponent strength features to team data for training.
        
        Args:
            team_df: Team data with basic stats (from create_team_defensive_stats)
            historical_df: Historical data with opponent strength features
            
        Returns:
            team_df with opponent strength features added
        """
        print("  🎯 Adding opponent strength features for team model training...")
        
        # Create a mapping of team+GW to opponent strength features
        opponent_features = historical_df.groupby(['team', 'GW']).first()[
            ['opponent_attack_strength', 'fixture_attractiveness']
        ].reset_index()
        
        # Rename team column to match team_df
        opponent_features = opponent_features.rename(columns={'team': 'team_name'})
        
        # Normalize opponent attack strength (consistent with other models)
        opponent_features['opponent_attack_strength_normalized'] = (
            opponent_features['opponent_attack_strength'] / 1400.0
        )
        
        # Rename fixture_attractiveness to fixture_difficulty for consistency
        opponent_features['fixture_difficulty'] = opponent_features['fixture_attractiveness']
        
        # Merge with team_df
        team_df_enhanced = team_df.merge(
            opponent_features[['team_name', 'GW', 'opponent_attack_strength_normalized', 'fixture_difficulty']],
            on=['team_name', 'GW'],
            how='left'
        )
        
        # Fill any missing values with defaults
        team_df_enhanced['opponent_attack_strength_normalized'] = team_df_enhanced['opponent_attack_strength_normalized'].fillna(0.82)  # ~1150/1400
        team_df_enhanced['fixture_difficulty'] = team_df_enhanced['fixture_difficulty'].fillna(0.5)  # Neutral
        
        print(f"    ✅ Added opponent strength features to {len(team_df_enhanced)} team records")
        print(f"    ✅ Features: opponent_attack_strength_normalized, fixture_difficulty")
        
        return team_df_enhanced
    
    def _get_position_name(self, element_type: int) -> str:
        """Convert element_type to position name"""
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return position_map.get(element_type, 'MID')
    
    def get_minutes_model_feature_columns(self) -> List[str]:
        """
        Get feature column names for minutes model in the correct order.
        This must match the order in prepare_minutes_model_features.
        
        Returns:
            List of feature column names (12 features total)
        """
        return [
            'position_encoded', 'team_encoded', 'form',
            'minutes_avg_3gw', 'minutes_avg_5gw',
            'total_points_avg_3gw', 'total_points_avg_5gw',
            'goals_scored_avg_3gw', 'assists_avg_3gw', 'clean_sheets_avg_3gw',
            # Opponent strength features - 2 features
            'opponent_overall_strength_normalized', 'fixture_attractiveness'
        ]

    def calculate_team_rolling_features(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling features for team-level data (goals conceded model)
        
        Args:
            team_df: DataFrame with team performance data
            
        Returns:
            DataFrame with team rolling features
        """
        print("🔧 Engineering team defensive features with shared engine...")
        
        # Sort by team and gameweek for chronological rolling averages
        team_df_sorted = team_df.sort_values(['team_id', 'GW']).reset_index(drop=True)
        
        # Define rolling windows
        rolling_windows = [3, 5, 10]
        
        for window in rolling_windows:
            print(f"  📊 Adding {window}-game rolling averages...")
            
            # Goals conceded rolling average
            team_df_sorted[f'goals_conceded_avg_{window}gw'] = team_df_sorted.groupby('team_id')['goals_conceded'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Clean sheets rolling average  
            team_df_sorted[f'clean_sheets_avg_{window}gw'] = team_df_sorted.groupby('team_id')['clean_sheets'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Goals scored rolling average
            team_df_sorted[f'goals_scored_avg_{window}gw'] = team_df_sorted.groupby('team_id')['goals_scored'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Total points rolling average
            team_df_sorted[f'total_points_avg_{window}gw'] = team_df_sorted.groupby('team_id')['total_points'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # Home/Away specific averages
        print("  🏠 Adding home/away specific averages...")
        
        # Home goals conceded average
        home_mask = team_df_sorted['was_home'] == 1
        team_df_sorted.loc[home_mask, 'home_goals_conceded_avg_5gw'] = team_df_sorted[home_mask].groupby('team_id')['goals_conceded'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        # Away goals conceded average  
        away_mask = team_df_sorted['was_home'] == 0
        team_df_sorted.loc[away_mask, 'away_goals_conceded_avg_5gw'] = team_df_sorted[away_mask].groupby('team_id')['goals_conceded'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        # Fill NaN values with overall averages
        team_df_sorted['home_goals_conceded_avg_5gw'] = team_df_sorted['home_goals_conceded_avg_5gw'].fillna(team_df_sorted['goals_conceded_avg_5gw'])
        team_df_sorted['away_goals_conceded_avg_5gw'] = team_df_sorted['away_goals_conceded_avg_5gw'].fillna(team_df_sorted['goals_conceded_avg_5gw'])
        
        # Additional derived features
        print("  ⚡ Adding derived defensive features...")
        
        # Defensive strength (inverse of goals conceded)
        team_df_sorted['defensive_strength'] = 1 / (team_df_sorted['goals_conceded_avg_5gw'] + 0.1)
        
        # Recent form (last 3 games total points)
        team_df_sorted['recent_form'] = team_df_sorted.groupby('team_id')['total_points'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        
        # Season progress
        team_df_sorted['season_progress'] = team_df_sorted['GW']
        
        print("✅ Team defensive features engineered using shared engine")
        
        return team_df_sorted

    def get_team_feature_columns(self) -> List[str]:
        """
        Get the standard feature column names for team defensive model
        
        Returns:
            List of feature column names
        """
        return [
            'goals_conceded_avg_3gw', 'goals_conceded_avg_5gw', 'goals_conceded_avg_10gw',
            'clean_sheets_avg_3gw', 'clean_sheets_avg_5gw', 'clean_sheets_avg_10gw', 
            'goals_scored_avg_3gw', 'goals_scored_avg_5gw', 'goals_scored_avg_10gw',
            'total_points_avg_3gw', 'total_points_avg_5gw', 'total_points_avg_10gw',
            'home_goals_conceded_avg_5gw', 'away_goals_conceded_avg_5gw',
            'defensive_strength', 'recent_form', 'season_progress', 'was_home'
        ]

    def calculate_yellow_cards_features(self, outfield_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling features for yellow cards model
        
        Args:
            outfield_df: DataFrame with outfield player data
            
        Returns:
            DataFrame with yellow cards rolling features
        """
        print("🔧 Engineering yellow cards features with shared engine...")
        
        # Define yellow cards specific features
        yellow_cards_features = [
            'yellow_cards', 'red_cards', 'minutes', 'starts',
            'goals_scored', 'assists', 'clean_sheets', 'total_points'
        ]
        
        # Calculate rolling features using the base method
        outfield_df = self.calculate_rolling_features(
            outfield_df, 
            group_col='name',
            sort_cols=['name', 'GW'],
            rolling_features=yellow_cards_features
        )
        
        # Additional yellow cards specific features
        print("  ⚡ Adding yellow cards specific derived features...")
        
        # Disciplinary rate (cards per minute)
        outfield_df['disciplinary_rate'] = (
            outfield_df['yellow_cards_avg_5gw'] / (outfield_df['minutes_avg_5gw'] + 1)
        ) * 90  # Per 90 minutes
        
        # Aggression indicator (cards when playing regularly)
        outfield_df['aggression_indicator'] = np.where(
            outfield_df['starts_avg_5gw'] > 0.5,
            outfield_df['yellow_cards_avg_5gw'],
            0
        )
        
        # Position risk factor (will be enhanced with position encoding)
        outfield_df['position_risk'] = outfield_df.get('position_encoded', 0)
        
        print("✅ Yellow cards features engineered using shared engine")
        
        return outfield_df

    def get_yellow_cards_feature_columns(self) -> List[str]:
        """
        Get the standard feature column names for yellow cards model
        
        Returns:
            List of feature column names
        """
        return [
            'yellow_cards_avg_3gw', 'yellow_cards_avg_5gw', 'yellow_cards_avg_10gw',
            'red_cards_avg_3gw', 'red_cards_avg_5gw', 'red_cards_avg_10gw',
            'minutes_avg_3gw', 'minutes_avg_5gw', 'minutes_avg_10gw',
            'starts_avg_3gw', 'starts_avg_5gw', 'starts_avg_10gw',
            'goals_scored_avg_3gw', 'goals_scored_avg_5gw', 'goals_scored_avg_10gw',
            'assists_avg_3gw', 'assists_avg_5gw', 'assists_avg_10gw',
            'clean_sheets_avg_3gw', 'clean_sheets_avg_5gw', 'clean_sheets_avg_10gw',
            'total_points_avg_3gw', 'total_points_avg_5gw', 'total_points_avg_10gw',
            'disciplinary_rate', 'aggression_indicator', 'position_risk'
        ]


def load_historical_data(filepath: str) -> pd.DataFrame:
    """Load and prepare historical gameweek data"""
    print(f"📊 Loading historical data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} gameweek records")
    return df


def load_teams_data(filepath: str) -> List[Dict]:
    """Load teams data"""
    with open(filepath, 'r') as f:
        teams_data = json.load(f)
    print(f"✅ Loaded {len(teams_data)} teams")
    return teams_data


# Example usage:
if __name__ == "__main__":
    # Load data
    historical_df = load_historical_data('/Users/owen/src/Personal/fpl-team-picker/Data/raw/parsed_gw_2425.csv')
    teams_data = load_teams_data('/Users/owen/src/Personal/fpl-team-picker/Data/database/teams.json')
    
    # Initialize feature engine
    feature_engine = PlayerFeatureEngine(teams_data)
    
    # For training: calculate rolling features
    training_df = feature_engine.calculate_rolling_features(historical_df)
    print(f"✅ Training data with rolling features: {training_df.shape}")
    
    # For prediction: get historical context for a player
    player_context = feature_engine.get_historical_context('Haaland', historical_df)
    if player_context:
        print(f"✅ Historical context loaded for Haaland")
        print(f"   Recent form: {player_context.get('form', 0):.1f}")
        print(f"   Avg minutes (5gw): {player_context.get('minutes_avg_5gw', 0):.1f}")
