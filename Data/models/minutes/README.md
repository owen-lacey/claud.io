# Minutes Model 🎯

**Status**: ✅ **COMPLETE** - 82% accuracy

## Overview
Foundation model that predicts the probability of a player receiving different amounts of playing time in a match. This is crucial as it affects all other predictions (goals, assists, etc.).

## Model Performance
- **Overall Accuracy**: 82%
- **Model Type**: Random Forest Classifier
- **Categories**: 4 minutes buckets (no_minutes, few_minutes, substantial_minutes, full_match)

### Category Performance
| Category | Precision | Recall | F1-Score | Description |
|----------|-----------|--------|----------|-------------|
| No Minutes (0) | 92% | 91% | 91% | Benched players |
| Full Match (90+) | 76% | 90% | 83% | Regular starters |
| Few Minutes (1-30) | 51% | 51% | 51% | Late substitutes |
| Substantial (31-89) | 36% | 13% | 19% | Early hooks/rotation |

## Key Features
1. **minutes_avg_3gw** (43.6%) - 3-game rolling average of minutes
2. **minutes_avg_5gw** (30.4%) - 5-game rolling average of minutes  
3. **clean_sheets_avg_3gw** (6.8%) - Recent defensive performance
4. **total_points_avg_3gw** (6.0%) - Recent fantasy points form
5. **form** (4.9%) - Official FPL form rating

## Files
- `train_minutes_model.py` - Production training script with shared feature engineering
- Uses `../feature_engineering/player_features.py` for consistent feature calculation

## Usage
```python
# Load the trained model
import pickle
with open('minutes_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Make predictions (returns probabilities for each category)
probabilities = predict_minutes_probability(player_data, model_data)
```

## Next Steps
- Integrate into main FPL prediction pipeline
- Use minutes probabilities to weight other model predictions
- Consider ensemble methods for improved accuracy on middle categories
