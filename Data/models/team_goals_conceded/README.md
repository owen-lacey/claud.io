# Team Goals Conceded Model 🛡️

**Status**: ✅ **COMPLETE** - MSE 0.345, MAE 0.384

## Overview
Models the expected number of goals each team will concede in upcoming fixtures using Poisson regression. This feeds into clean sheet probabilities and goalkeeper saves predictions.

## Model Performance
- **Test MSE**: 0.345 - Strong prediction accuracy for goals conceded
- **Test MAE**: 0.384 - Average error less than half a goal
- **Model Type**: Poisson Regression with L2 regularization
- **Training Data**: 619 team-gameweek records across multiple seasons

### Key Feature Coefficients
| Feature | Coefficient | Impact |
|---------|-------------|---------|
| Defensive Strength | -0.167 | Lower values = fewer goals conceded |
| Goals Conceded Avg (3gw) | +0.143 | Recent form strongly predictive |
| Goals Conceded Avg (10gw) | +0.139 | Season-long form matters |
| Goals Conceded Avg (5gw) | +0.131 | Medium-term form factor |
| Clean Sheets Avg (5gw) | +0.103 | Defensive consistency indicator |

## Overview
Models the expected number of goals each team will concede in upcoming fixtures using Poisson regression. This feeds into clean sheet probabilities and goalkeeper saves predictions.

## Model Approach
- **Method**: Poisson Regression
- **Target**: Goals conceded per team per match
- **Features**: Team defensive strength, opponent attacking strength, home/away factors, recent form

## Key Components
1. **Team Defensive Ratings** - Baseline defensive strength per team
2. **Opponent Attack Ratings** - Attacking quality of opposing teams  
3. **Home/Away Effects** - Venue impact on defensive performance
4. **Form Adjustments** - Recent defensive form (goals conceded trends)
5. **Fixture Difficulty** - Historical matchup patterns

## Expected Outputs
- **Goals Conceded Probability Distribution** - P(0), P(1), P(2), P(3+) goals conceded
- **Clean Sheet Probability** - P(goals conceded = 0)
- **Expected Goals Conceded** - Mean expectation for saves model

## Files
- `train_team_goals_conceded_model.py` - Complete Poisson regression implementation

## Usage (Once Complete)
```python
# Predict goals conceded for a team fixture
expected_goals_conceded = predict_team_goals_conceded(
    team_id=team_id,
    opponent_id=opponent_id, 
    is_home=True,
    model_data=model_data
)

# Get clean sheet probability
clean_sheet_prob = poisson.pmf(0, expected_goals_conceded)
```

## Integration Points
- **Clean Sheets**: Defender/goalkeeper clean sheet points
- **Saves Model**: Expected saves = f(expected goals conceded)
- **Bonus Points**: Clean sheets contribute to bonus calculations
