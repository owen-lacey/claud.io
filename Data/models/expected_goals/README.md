# Expected Goals Model ⚽

**Status**: ✅ **COMPLETE** - MSE 0.116, MAE 0.202

## Overview
Predicts goals scored by individual attacking players (forwards and midfielders) using Poisson regression. This is the foundation for attacking returns and goal-based bonus points predictions.

## Model Performance
- **Test MSE**: 0.116 - Excellent prediction accuracy for goal scoring
- **Test MAE**: 0.202 - Average error ~0.2 goals per prediction
- **Model Type**: Poisson Regression with L2 regularization
- **Training Data**: 7,222 records from attacking players with regular playing time

### Goals Distribution in Training Data
| Goals | Count | Percentage |
|-------|-------|------------|
| 0 goals | 6,433 | 89.1% |
| 1 goal | 711 | 9.8% |
| 2 goals | 67 | 0.9% |
| 3+ goals | 11 | 0.2% |

### Key Feature Coefficients
| Feature | Coefficient | Impact |
|---------|-------------|---------|
| Expected Goals Avg (10gw) | +0.050 | Long-term xG strongest predictor |
| Expected Goals Avg (5gw) | +0.038 | Medium-term xG form |
| Is Forward | +0.036 | Position advantage for goal scoring |
| Expected Goals Avg (3gw) | +0.035 | Recent xG form |
| Goals Avg (10gw) | +0.034 | Historical goal scoring |
| Goals Avg (5gw) | +0.030 | Recent goal scoring |
| Threat Avg (5gw) | +0.029 | FPL threat metric importance |
| Minutes Avg (3gw) | +0.028 | Playing time correlation |

## Key Insights
1. **Expected Goals Dominates**: xG metrics are the strongest predictors, confirming quality of underlying data
2. **Position Matters**: Forwards have significant advantage (+0.036 coefficient)
3. **Long-term Form**: 10-game averages slightly outperform short-term form
4. **Playing Time**: Recent minutes strongly correlate with goal scoring opportunities

## Example Predictions
### Premium Forward (Home)
- **Expected Goals**: 0.427 per match
- **Probability to Score**: 34.7%
- **Most Likely**: 0 goals (65.3%), 1 goal (27.9%), 2 goals (5.9%)

### Attacking Midfielder (Away)  
- **Expected Goals**: 0.210 per match
- **Probability to Score**: 18.9%
- **Most Likely**: 0 goals (81.1%), 1 goal (17.0%), 2 goals (1.8%)

## Files
- `train_expected_goals_model.py` - Complete model training pipeline

## Usage
```python
# Load the trained model
import joblib
model_data = joblib.load('expected_goals_model.pkl')

# Make predictions (returns expected goals and probabilities)
prediction = predict_player_goals_distribution(player_stats)
```

## Integration Points
- **FPL Points**: Goals worth 4 points (MID) or 5 points (FWD)
- **Bonus Points**: Goals contribute significantly to BPS
- **Captain Selection**: High goal probability players are captain candidates
- **Minutes Model**: Should be weighted by minutes probability for final XP
