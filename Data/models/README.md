# FPL Models Directory 🤖

**Centralized location for all ML models, training scripts, and model documentation.**

## 📁 Directory Structure

```
models/
├── README.md                           # This overview
├── minutes/                           # Minutes/playing time prediction
│   ├── minutes_model.pkl             # Trained model (12 features, 82% accuracy)
│   ├── train_minutes_model.py        # Training script  
│   └── README.md                     # Model documentation
├── expected_goals/                   # Goal prediction for attacking players
│   ├── expected_goals_model.pkl      # Trained model (16 features, MSE 0.116)
│   ├── train_expected_goals_model.py # Training script
│   └── README.md                     # Model documentation  
├── expected_assists/                 # Assist prediction for creative players
│   ├── expected_assists_model.pkl    # Trained model (15 features, MSE 0.080)
│   ├── train_expected_assists_model.py # Training script
│   └── README.md                     # Model documentation
├── saves/                           # Goalkeeper save predictions
│   ├── saves_model.pkl              # Trained model (16 features, MSE 2.060)
│   ├── train_saves_model.py         # Training script
│   └── README.md                    # Model documentation
└── team_goals_conceded/             # Team defensive modeling
    ├── team_goals_conceded_model.pkl # Trained model (18 features, MSE 0.345)
    ├── train_team_goals_conceded_model.py # Training script
    └── README.md                    # Model documentation
```

## 🎯 Model Overview

All 5 models are **production-ready** and integrated into the prediction pipeline:

| Model | Purpose | Features | Performance | Status |
|-------|---------|----------|-------------|--------|
| **Minutes** | Playing time probability | 12 | 82% accuracy | ✅ Complete |
| **Goals** | Expected goals (FWD/MID) | 16 | MSE 0.116 | ✅ Complete |
| **Assists** | Expected assists (all positions) | 15 | MSE 0.080 | ✅ Complete |
| **Saves** | Goalkeeper saves | 16 | MSE 2.060 | ✅ Complete |
| **Team Goals** | Goals conceded (team-level) | 18 | MSE 0.345 | ✅ Complete |

## 🔧 Usage

### Training a Model
```bash
# Navigate to specific model directory
cd models/minutes/

# Run training script
python3 train_minutes_model.py
```

### Loading in Prediction Pipeline
```python
# Models are automatically loaded by FPLPredictionEngine
from fpl_assembly_pipeline import FPLPredictionEngine

engine = FPLPredictionEngine()  # Loads all 6 models
predictions = engine.generate_gameweek_predictions(19)
```

## 🏗️ Model Architecture

- **Model Type**: Scikit-learn ML models (RandomForest, Poisson Regression)
- **Feature Engineering**: Shared `PlayerFeatureEngine` for consistency
- **Opponent Awareness**: Enhanced with fixture difficulty and team strength
- **Integration**: Seamless integration with FPL scoring rules

## 📊 Data Dependencies

- **Historical Data**: `/Data/raw/parsed_gw.csv`
- **Player Data**: `/Data/database/players.json`
- **Team Data**: `/Data/database/teams.json` 
- **Feature Engineering**: `/Data/feature_engineering/player_features.py`

## 🎉 Benefits of This Structure

1. **Organization**: All model-related code in one place
2. **Consistency**: Standardized structure across all models
3. **Maintainability**: Easy to update, retrain, or add new models
4. **Documentation**: Each model has dedicated README with performance metrics
5. **Integration**: Clean separation between models and prediction pipeline
