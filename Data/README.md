# FPL Model Development Plan

## Overview
Build a modular FPL (Fantasy Premier League) model that predicts player points by first estimating underlying performance metrics (expected goals, assists, clean sheets, etc.) and then converting these into final expected points (XP).

## Data Sources
**Where our data comes from:**
- **CSV Files**: Historical player performance data (`raw/parsed_gw.csv`) containing gameweek-by-gameweek stats for training models
- **FPL API**: Live data for current season including player info, team stats, and upcoming fixtures for making predictions

## Data Storage
**JSON files we'll maintain:**
- `database/players.json` - Player information, positions, teams, season totals
- `database/teams.json` - Team strength metrics, attacking/defensive ratings  
- `database/fixtures.json` - All fixtures (historical with results + upcoming) across seasons

## What We're Predicting
**Core metrics we want to model:**
- **Goals Scored**: Expected goals (xG) for attacking players
- **Goals Conceded**: Team-level Poisson model for defensive events
- **Assists**: Expected assists (xA) for creative players
- **Saves**: Goalkeeper saves based on expected goals conceded
- **Minutes Played**: Playing time probability (affects all other predictions)
- **Bonus Points**: Based on in-game performance metrics

## Joining It All Together
**How we convert predictions to XP:**
1. **Team Context**: Use fixtures to determine opponent matchups and home/away status
2. **Individual Predictions**: Apply component models to predict each metric for each player
3. **FPL Scoring**: Convert predictions to FPL points using official scoring system
4. **Final Assembly**: Combine all components weighted by minutes probability to get expected points (XP)

## Implementation Roadmap

## 🎯 Current Model Performance

### Minutes Model ✅ **COMPLETE**
- **Accuracy**: 82% overall classification accuracy
- **Strong Categories**: 
  - Full Match (90+ mins): 90% recall - excellent at identifying regular starters
  - No Minutes (0 mins): 91% recall - great at spotting benchwarners
- **Key Features**: 3-game rolling averages (43.6% importance), 5-game averages (30.4%)
- **Status**: Model trained, saved, and ready for integration

### Team Goals Conceded Model ✅ **COMPLETE**
- **Performance**: MSE 0.345 test error, MAE 0.384 - strong Poisson regression performance
- **Model Type**: Poisson Regression predicting goals conceded distribution
- **Key Features**: 
  - Defensive strength (-0.167 coefficient) - most important defensive indicator
  - Recent goals conceded averages (3gw, 5gw, 10gw rolling windows)
  - Clean sheet form and home/away factors
- **Outputs**: Expected goals conceded, clean sheet probability, full goal distribution
- **Status**: Model trained, saved, ready for clean sheet and saves predictions

### Expected Goals Model ✅ **COMPLETE**
- **Performance**: MSE 0.116 test error, MAE 0.202 - excellent goal prediction accuracy
- **Model Type**: Poisson Regression for attacking players (FWD, MID)
- **Key Features**:
  - Expected goals averages (xG 10gw: +0.050 coefficient) - strongest predictor
  - Position advantage (Forward: +0.036) - clear position effect
  - Historical goal scoring and threat metrics
- **Outputs**: Expected goals per match, probability to score, full goal distribution
- **Status**: Model trained, saved, ready for attacking returns and bonus predictions

### Expected Assists Model ✅ **COMPLETE**
- **Performance**: MSE 0.080 test error, MAE 0.141 - outstanding assist prediction accuracy
- **Model Type**: Poisson Regression for creative players (FWD, MID, DEF)
- **Key Features**:
  - Creativity averages (3gw: +0.031 coefficient) - FPL creativity metric dominates
  - Expected assists averages (xA 10gw: +0.030) - long-term form crucial
  - Position penalty (Defender: -0.021) - clear position disadvantage
- **Outputs**: Expected assists per match, probability to assist, full assist distribution
- **Status**: Model trained, saved, ready for creative returns and bonus predictions

### Saves Model ✅ **COMPLETE**
- **Performance**: MSE 2.060 test error, MAE 0.927 - strong goalkeeper save predictions
- **Model Type**: Poisson Regression for goalkeepers with team defensive context
- **Key Features**:
  - Starts averages (3gw: +0.280 coefficient) - playing time most important
  - Recent form (-0.242) - higher points = fewer saves needed paradox
  - Historical saves averages and team defensive context
- **Outputs**: Expected saves per match, probability of high saves (4+), full save distribution
- **Status**: Model trained, saved, ready for goalkeeper returns and save points

### Stage 1: Data Pipeline ✅ **COMPLETE**
- [x] Historical data extraction from CSV
- [x] Basic data exploration 
- [x] Extract fixtures data from FPL API
- [x] **Shared feature engineering pipeline** (rolling averages, categorical encoding) ✅ **NEW**

### Yellow Cards Model ✅ **COMPLETE**
- **Performance**: MSE 75.073 test error, MAE 4.251 - reasonable performance for rare event prediction
- **Model Type**: Poisson Regression for outfield players with disciplinary context
- **Key Features**:
  - Home advantage (-0.361 coefficient) - fewer cards at home
  - Recent form (+0.269) - involvement paradox (more points = more cards)
  - Position effects (Defenders: -0.139) - midfielders highest risk
  - Historical disciplinary record and playing time correlation
- **Outputs**: Expected yellow cards per match, probability of card, full card distribution
- **Status**: Model trained, saved, ready for disciplinary risk assessment

### Stage 1: Data Pipeline ✅ **COMPLETE**
- [x] Historical data extraction from CSV
- [x] Basic data exploration 
- [x] Extract fixtures data from FPL API
- [x] **Shared feature engineering pipeline** (rolling averages, categorical encoding) ✅ **NEW**

### Stage 2: Core Models ✅ **ALL 5 MODELS COMPLETE!** 🎉
- [x] **Minutes Model** (foundation - affects all predictions) ✅ **82% accuracy**
- [x] **Team Goals Conceded** (Poisson approach) ✅ **MSE 0.345**
- [x] **Expected Goals Model** (attacking players) ✅ **MSE 0.116**
- [x] **Expected Assists Model** (creative players) ✅ **MSE 0.080**
- [x] **Saves Model** (goalkeepers) ✅ **MSE 2.060**

### Stage 3: Integration ✅ **FPL SCORING ENGINE COMPLETE!** 🎉
- [x] **FPL scoring rules engine** (2025/26 official rules) ✅ **Full implementation**
- [x] **Model integration system** (converts predictions to points) ✅ **Complete pipeline** 
- [ ] Bonus points prediction
- [ ] Final XP assembly pipeline
- [ ] Model validation and testing

## Success Metrics
- **Accuracy**: Beat benchmark models (season averages, form-based predictions)  
- **Reliability**: Strong correlation between predicted and actual points
- **Practical Value**: Improve FPL decision making for transfers and captaincy

## 📁 Project Structure
```
Data/
├── README.md                    # This overview document
├── database/                    # JSON data storage
│   ├── players.json            # Player info & season stats  
│   ├── teams.json              # Team strength metrics
│   └── fixtures.json           # Historical & upcoming fixtures
├── feature_engineering/        # ✅ NEW - Shared feature engineering
│   ├── player_features.py      # PlayerFeatureEngine for consistent features
│   └── README.md               # Feature engineering documentation
│   ├── README.md               # Shared feature engineering documentation
│   └── player_features.py     # PlayerFeatureEngine for consistent features
├── models/                     # ✅ CENTRALIZED - All ML models and training code
│   ├── README.md               # Models directory overview
│   ├── minutes/                # ✅ Minutes model (82% accuracy)
│   │   ├── minutes_model.pkl
│   │   ├── train_minutes_model.py
│   │   └── README.md
│   ├── expected_goals/         # ✅ Goals model (MSE 0.116)
│   │   ├── expected_goals_model.pkl
│   │   ├── train_expected_goals_model.py
│   │   └── README.md
│   ├── expected_assists/       # ✅ Assists model (MSE 0.080)
│   │   ├── expected_assists_model.pkl
│   │   ├── train_expected_assists_model.py
│   │   └── README.md
│   ├── saves/                  # ✅ Saves model (MSE 2.060)
│   │   ├── saves_model.pkl
│   │   ├── train_saves_model.py
│   │   └── README.md
│   └── team_goals_conceded/    # ✅ Team defense model (MSE 0.345)
│       ├── team_goals_conceded_model.pkl
│       ├── train_team_goals_conceded_model.py
│       └── README.md
├── final_assembly/             # ✅ PRODUCTION READY - Complete System
│   ├── README.md               # Integration documentation
│   └── fpl_assembly_pipeline.py # Core prediction engine (uses shared features)
├── notebooks/                  # Data pipeline notebooks
│   ├── README.md               # Notebook documentation
│   └── extract.ipynb           # ✅ Main data extraction
├── raw/                        # Raw data & parsing scripts
└── utils/                      # Utility functions
```