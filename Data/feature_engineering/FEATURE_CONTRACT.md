# FPL Pipeline Feature Contract Documentation

This document describes the feature contract for each model in the FPL prediction pipeline, extracted from `Data/feature_engineering/player_features.py` and the model training scripts.

## Data Sources and Assumptions

### FPL API Dependencies
- **Primary Data**: FPL API bootstrap-static endpoint (`https://fantasy.premierleague.com/api/bootstrap-static/`)
- **Historical Data**: `Data/raw/parsed_gw_2425.csv` (parsed gameweek data from previous seasons)
- **Player Codes**: Stable player codes for cross-season matching
- **Team Data**: MongoDB/JSON fallback for teams, strength ratings, fixture difficulty

### Core Pipeline Components
1. **Ingestion**: `Data/notebooks/extract.ipynb` - FPL API extraction
2. **Feature Engineering**: `Data/feature_engineering/player_features.py` - shared rolling features
3. **Assembly**: `Data/final_assembly/fpl_assembly_pipeline.py` - model orchestration
4. **Models**: `Data/models/*/` - trained ML models per target
5. **Predictions**: `Data/predictions_2025_26/generate_predictions.py` - prediction CLI

---

## Rolling Feature Contract

All models use rolling window features calculated by `PlayerFeatureEngine.calculate_rolling_features()`:

### Base Input Columns (Historical Data)
```python
required_columns = [
    'name',           # Player name (string)
    'GW',             # Gameweek (int, 1-38)
    'minutes',        # Minutes played (int, 0-90+)
    'total_points',   # FPL points (int)
    'goals_scored',   # Goals (int)
    'assists',        # Assists (int)
    'clean_sheets',   # Clean sheets (int, 0-1)
    'saves',          # Saves (int, GK only)
    'goals_conceded', # Goals conceded (int, GK/DEF)
    'yellow_cards',   # Yellow cards (int)
    'starts',         # Started match (int, 0-1)
    'threat',         # FPL threat index (float)
    'expected_goals', # xG (float)
    'expected_assists', # xA (float)
    'expected_goal_involvements', # xGI (float)
    'expected_goals_conceded', # xGC (float, GK/DEF)
    'creativity',     # FPL creativity index (float)
    'bps'            # Bonus points system score (int)
]
```

### Rolling Windows
- **Window Sizes**: 3, 5, 10 gameweeks
- **Naming Convention**: `{metric}_avg_{window}gw`
- **Min Periods**: 1 (no nulls)
- **Grouping**: By player name (requires stable player identification)

---

## Model-Specific Feature Contracts

### 1. Minutes Model
**Target**: Expected minutes (float, 0-90+)
**Features**: 12 features
**Method**: `prepare_minutes_model_features()`

```python
feature_columns = [
    'position_encoded',      # Position (int, 0-3: GK/DEF/MID/FWD)
    'team_encoded',          # Team (int, encoded)
    'form',                  # FPL form (float)
    'minutes_avg_3gw',       # Rolling minutes (float)
    'minutes_avg_5gw',       # Rolling minutes (float)
    'total_points_avg_3gw',  # Rolling points (float)
    'total_points_avg_5gw',  # Rolling points (float)
    'goals_scored_avg_3gw',  # Rolling goals (float)
    'assists_avg_3gw',       # Rolling assists (float)
    'clean_sheets_avg_3gw',  # Rolling clean sheets (float)
    'opponent_overall_strength_normalized', # Opponent strength (float, 0-1)
    'fixture_attractiveness' # Fixture difficulty (float, 0-1)
]
```

**Null Policy**: Historical context required; season averages as fallback

### 2. Expected Goals Model
**Target**: Expected goals (float, 0+)
**Features**: 20 features (position-aware)
**Method**: `prepare_goals_model_features()`

```python
feature_columns = [
    # Goal scoring rolling averages - 6 features
    'goals_scored_avg_3gw', 'goals_scored_avg_5gw', 'goals_scored_avg_10gw',
    'expected_goals_avg_3gw', 'expected_goals_avg_5gw', 'expected_goals_avg_10gw',
    # Playing time rolling averages - 4 features
    'minutes_avg_3gw', 'minutes_avg_5gw', 'starts_avg_3gw', 'starts_avg_5gw',
    # Threat rolling averages - 2 features
    'threat_avg_3gw', 'threat_avg_5gw',
    # Derived features - 2 features
    'goal_efficiency',       # goals / max(xG, 0.1)
    'recent_form',          # FPL form
    # Position features (one-hot) - 3 features
    'is_forward', 'is_midfielder', 'is_defender',
    # Venue - 1 feature
    'was_home',             # Home/away (0/1)
    # Opponent strength - 2 features
    'opponent_defence_strength_normalized', # Opponent defense (float, 0-1)
    'fixture_attractiveness' # Fixture difficulty (float, 0-1)
]
```

### 3. Expected Assists Model
**Target**: Expected assists (float, 0+)
**Features**: 16 features
**Method**: `prepare_assists_model_features()`

```python
feature_columns = [
    # Assist rolling averages - 6 features
    'assists_avg_3gw', 'assists_avg_5gw', 'assists_avg_10gw',
    'expected_assists_avg_3gw', 'expected_assists_avg_5gw', 'expected_assists_avg_10gw',
    # Playing time - 4 features
    'minutes_avg_3gw', 'minutes_avg_5gw', 'starts_avg_3gw', 'starts_avg_5gw',
    # Creativity - 2 features
    'creativity_avg_3gw', 'creativity_avg_5gw',
    # Derived features - 2 features
    'assist_efficiency',     # assists / max(xA, 0.1)
    'recent_form',
    # Venue - 1 feature
    'was_home',
    # Opponent strength - 1 feature
    'fixture_attractiveness'
]
```

### 4. Saves Model (Goalkeepers)
**Target**: Expected saves (float, 0+)
**Features**: 16 features
**Method**: `prepare_saves_model_features()`

```python
feature_columns = [
    # Save rolling averages - 6 features
    'saves_avg_3gw', 'saves_avg_5gw', 'saves_avg_10gw',
    'goals_conceded_avg_3gw', 'goals_conceded_avg_5gw', 'goals_conceded_avg_10gw',
    # Playing time - 4 features
    'minutes_avg_3gw', 'minutes_avg_5gw', 'starts_avg_3gw', 'starts_avg_5gw',
    # Clean sheet probability - 2 features
    'clean_sheets_avg_3gw', 'clean_sheets_avg_5gw',
    # Team defensive metrics - 2 features
    'team_goals_conceded_avg_5gw', 'team_clean_sheets_avg_5gw',
    # Venue - 1 feature
    'was_home',
    # Opponent strength - 1 feature
    'opponent_attack_strength_normalized'
]
```

### 5. Team Goals Conceded Model
**Target**: Team goals conceded (float, 0+)
**Features**: 18 features
**Method**: `prepare_team_goals_conceded_features()`

```python
feature_columns = [
    # Team defensive rolling averages - 12 features
    'goals_conceded_avg_3gw', 'goals_conceded_avg_5gw', 'goals_conceded_avg_10gw',
    'clean_sheets_avg_3gw', 'clean_sheets_avg_5gw', 'clean_sheets_avg_10gw',
    'goals_scored_avg_3gw', 'goals_scored_avg_5gw', 'goals_scored_avg_10gw',
    'total_points_avg_3gw', 'total_points_avg_5gw', 'total_points_avg_10gw',
    # Home/away specific - 2 features
    'home_goals_conceded_avg_5gw', 'away_goals_conceded_avg_5gw',
    # Derived team features - 3 features
    'defensive_strength', 'recent_form', 'season_progress',
    # Venue - 1 feature
    'was_home',
    # Opponent strength - 2 features
    'opponent_attack_strength_normalized', 'fixture_difficulty'
]
```

---

## FPL-Specific Derived Fields

### FPL-Only Metrics (No Direct FBRef Equivalent)
- **threat**: FPL's attacking threat index
- **creativity**: FPL's creative influence index
- **bps**: Bonus points system score
- **form**: FPL's rolling form metric
- **total_points**: FPL scoring system points

### Opponent Strength Features
- **opponent_overall_strength**: Team strength rating (raw, ~800-1400)
- **opponent_attack_strength**: Attacking strength rating
- **opponent_defence_strength**: Defensive strength rating  
- **fixture_attractiveness**: Derived difficulty score (0-1, higher = easier)

### Encoding Requirements
- **Position Encoding**: GK=0, DEF=1, MID=2, FWD=3
- **Team Encoding**: LabelEncoder fitted on team names
- **Home/Away**: 1 for home, 0 for away

---

## Data Quality Requirements

### Player Identification
- **Cross-Season Matching**: Player codes required for historical context
- **Name Standardization**: Consistent player names across seasons
- **Position Stability**: Position changes tracked

### Missing Data Policy
- **New Players**: Season averages as fallback features
- **Insufficient History**: Minimum 1 gameweek for rolling features
- **Injured Players**: Zero minutes, preserve other rolling averages

### Temporal Consistency
- **Gameweek Alignment**: All data aligned to FPL gameweek calendar
- **Rolling Window Ordering**: Chronological sorting by [name, GW]
- **Future Leakage**: No future information in historical features

---

## Notes for FBRef Migration

### Migration Strategy: Feature Redesign (Not 1:1 Replication)
Since the FPL models were proof-of-concept, we can redesign the feature set using FBRef's richer data rather than maintaining exact feature parity. The goal is **better predictive performance**, not feature compatibility.

### Direct Mappings (Core Stats)
- `minutes`, `goals`, `assists`, `yellow_cards` → direct equivalents
- `expected_goals` (xG), `expected_assists` (xA) → FBRef xG/xA (potentially more accurate)
- `starts` → derived from minutes or start flag
- `clean_sheets`, `saves`, `goals_conceded` → direct equivalents

### FPL-Specific Features to Replace (Not Approximate)
Instead of trying to replicate FPL's proprietary metrics, use FBRef's native features:
- **Replace `threat`** → `shots`, `shots_on_target`, `shot_distance`, `shot_angle`
- **Replace `creativity`** → `key_passes`, `passes_final_third`, `crosses`, `through_balls`
- **Replace `bps`** → Ignore (FPL-specific scoring artifact)
- **Replace `total_points` & `form`** → Use underlying performance metrics directly
- **Replace opponent strength** → Calculate from FBRef team performance data

### New FBRef-Native Opportunities
- **Defensive Actions**: `tackles`, `interceptions`, `blocks`, `clearances`
- **Passing Quality**: `pass_completion_rate`, `progressive_passes`, `long_balls`
- **Physical Metrics**: `distance_covered`, `sprints`, `duels_won`
- **Positional Data**: More granular position/role information
- **Set Pieces**: `corners`, `free_kicks`, `penalties`
- **Cross-League Context**: Prior league performance, transfer adaptation patterns

### Simplified Feature Philosophy
- **Fewer, higher-quality features** rather than trying to match FPL's 12-20 feature models
- **Domain-specific features** per position (GK vs outfield) rather than one-size-fits-all
- **Native rolling windows** on meaningful stats rather than derived indices
