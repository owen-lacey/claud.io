# Expected Assists Model 🎨

**Status**: ✅ **COMPLETE** - MSE 0.080, MAE 0.141

## Overview
Predicts assists by individual creative players (forwards, midfielders, and defenders) using Poisson regression. This is the foundation for creative player returns and assist-based bonus points predictions.

## Model Performance
- **Test MSE**: 0.080 - Outstanding prediction accuracy for assists
- **Test MAE**: 0.141 - Average error ~0.14 assists per prediction
- **Model Type**: Poisson Regression with L2 regularization
- **Training Data**: 11,618 records from players with regular playing time across all outfield positions

### Assists Distribution in Training Data
| Assists | Count | Percentage |
|---------|-------|------------|
| 0 assists | 10,795 | 92.9% |
| 1 assist | 764 | 6.6% |
| 2 assists | 51 | 0.4% |
| 3+ assists | 8 | 0.1% |

### Key Feature Coefficients
| Feature | Coefficient | Impact |
|---------|-------------|---------|
| Creativity Avg (3gw) | +0.031 | FPL creativity metric strongest predictor |
| Expected Assists Avg (10gw) | +0.030 | Long-term xA form crucial |
| Creativity Avg (5gw) | +0.029 | Medium-term creativity consistency |
| Expected Assists Avg (3gw) | +0.027 | Recent xA performance |
| Expected Assists Avg (5gw) | +0.026 | Medium-term xA form |
| Creative Output | +0.025 | Combined assists + goals metric |
| Is Defender | -0.021 | Position penalty for defenders |
| Goals Avg (5gw) | +0.021 | Attacking players often get both |

## Key Insights
1. **Creativity Dominates**: FPL's creativity metric is the strongest predictor, validating its usefulness
2. **Expected Assists Matter**: xA metrics are consistently important across all timeframes
3. **Position Effects**: Defenders have clear disadvantage (-0.021 coefficient)
4. **Attacking Correlation**: Players who score goals also tend to get assists
5. **Timeframe Balance**: Long-term form (10gw) slightly edges short-term (3gw)

## Example Predictions
### Creative Midfielder (Home)
- **Expected Assists**: 0.214 per match
- **Probability to Assist**: 19.2%
- **Most Likely**: 0 assists (80.8%), 1 assist (17.2%), 2 assists (1.8%)

### Wing-Back Defender (Away)
- **Expected Assists**: 0.097 per match  
- **Probability to Assist**: 9.2%
- **Most Likely**: 0 assists (90.8%), 1 assist (8.8%), 2 assists (0.4%)

### Supporting Forward (Home)
- **Expected Assists**: 0.144 per match
- **Probability to Assist**: 13.4%
- **Most Likely**: 0 assists (86.6%), 1 assist (12.5%), 2 assists (0.9%)

## Files
- `train_expected_assists_model.py` - Complete model training pipeline

## Usage
```python
# Load the trained model
import joblib
model_data = joblib.load('expected_assists_model.pkl')

# Make predictions (returns expected assists and probabilities)
prediction = predict_player_assists_distribution(player_stats)
```

## Integration Points
- **FPL Points**: Assists worth 3 points for all outfield positions
- **Bonus Points**: Assists contribute significantly to BPS calculations
- **Creative Player Selection**: High assist probability players valuable for team selection
- **Minutes Model**: Should be weighted by minutes probability for final XP calculation
