# Saves Model 🥅

**Status**: ✅ **COMPLETE** - MSE 2.060, MAE 0.927

## Overview
Predicts saves by goalkeepers using Poisson regression, integrating team defensive context with individual goalkeeper performance. This is the foundation for goalkeeper returns and save-based bonus points.

## Model Performance
- **Test MSE**: 2.060 - Strong prediction accuracy for goalkeeper saves
- **Test MAE**: 0.927 - Average error ~0.9 saves per prediction
- **Model Type**: Poisson Regression with L2 regularization
- **Training Data**: 820 records from goalkeepers with regular playing time

### Saves Distribution in Training Data
- **Mean Saves**: 1.32 per match
- **Range**: 0-13 saves per match
- **Distribution**: Heavy right skew (many low-save games, few high-save games)

### Key Feature Coefficients
| Feature | Coefficient | Impact |
|---------|-------------|---------|
| Starts Avg (3gw) | +0.280 | Playing time most important |
| Recent Form | -0.242 | Higher points = fewer saves needed |
| Starts Avg (5gw) | +0.228 | Consistent starting crucial |
| Saves Avg (3gw) | +0.110 | Recent save history predictive |
| Saves Avg (10gw) | +0.096 | Long-term save consistency |
| Minutes Avg (3gw) | +0.080 | Playing time correlation |
| Saves Avg (5gw) | +0.076 | Medium-term save form |
| Clean Sheets Avg (3gw) | +0.046 | Defensive context matters |

## Key Insights
1. **Playing Time Dominates**: Starting consistently is the strongest predictor
2. **Form Paradox**: Higher recent FPL points correlates with fewer saves (strong defenses need fewer saves)
3. **Historical Saves**: Past save performance is predictive across all timeframes
4. **Team Context**: Clean sheet history provides defensive context
5. **Save Opportunity**: Weaker defenses create more save opportunities

## Example Predictions
### Premium Goalkeeper (Strong Defense, Home)
- **Expected Saves**: 2.52 per match
- **Probability of 4+ Saves**: 24.6%
- **Most Likely**: 2 saves (25.6%), 3 saves (21.5%), 1 save (20.3%)

### Budget Goalkeeper (Weak Defense, Away)
- **Expected Saves**: 3.11 per match
- **Probability of 4+ Saves**: 37.7%
- **Most Likely**: 3 saves (22.4%), 2 saves (21.6%), 4 saves (17.4%)

## Strategic Implications
- **Budget GKs**: Weak defenses create more save opportunities
- **Premium GKs**: Strong defenses mean fewer saves but more clean sheets
- **Save Points**: 1 point per 3 saves in FPL - high save GKs can be valuable
- **Bonus Potential**: High save counts contribute to bonus point system

## Files
- `train_saves_model.py` - Complete model training pipeline

## Usage
```python
# Load the trained model
import joblib
model_data = joblib.load('saves_model.pkl')

# Make predictions (returns expected saves and probabilities)
prediction = predict_goalkeeper_saves_distribution(gk_stats)
```

## Integration Points
- **FPL Points**: 1 point per 3 saves (rounded down)
- **Clean Sheets**: Should combine with Team Goals Conceded model
- **Bonus Points**: High save counts contribute to BPS
- **Minutes Model**: Must be weighted by goalkeeper minutes probability
