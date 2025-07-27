#!/bin/bash
# FPL Predictions Quick Generator
# Simple wrapper script for generating FPL predictions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/generate_predictions.py"

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: generate_predictions.py not found"
    exit 1
fi

# Default to GW1 if no argument provided
GAMEWEEK=${1:-1}

echo "🚀 FPL Quick Predictions"
echo "========================"
echo "📊 Generating for: GW$GAMEWEEK"
echo ""

# Run the Python script
python "$PYTHON_SCRIPT" "$GAMEWEEK"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Predictions complete!"
    echo "📁 Files saved in: $SCRIPT_DIR"
    echo ""
    echo "💡 Quick commands:"
    echo "   View CSV: open top_predictions_gw${GAMEWEEK}_2025_26.csv"
    echo "   View JSON: cat summary_gw${GAMEWEEK}_2025_26.json | head -50"
else
    echo "❌ Prediction generation failed"
    exit 1
fi
