#!/usr/bin/env python3
"""
FPL Prediction vs Reality Analysis
==================================

Compare our 2025/26 GW1 predictions against actual 2024/25 season performance
to understand model calibration and bias patterns.

Author: Owen Lacey
Date: July 28, 2025
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add parent directory to Python path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our prediction engine
from final_assembly.fpl_assembly_pipeline import FPLPredictionEngine


class PredictionRealityAnalyzer:
    """Analyze how our predictions compare to actual player performance"""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "database"
        self.historical_data_path = Path(__file__).parent.parent / "raw" / "parsed_gw_2425.csv"
        self.results = {}
        
    def load_data(self):
        """Load all necessary data"""
        print("📊 Loading data...")
        
        # Load current season players (2025/26)
        with open(self.data_dir / "players.json", 'r') as f:
            self.current_players = json.load(f)
        print(f"   ✅ Loaded {len(self.current_players)} current players")
        
        # Load historical gameweek data (2024/25 season)
        self.historical_data = pd.read_csv(self.historical_data_path)
        print(f"   ✅ Loaded {len(self.historical_data)} historical gameweek records")
        
        # Calculate 2024/25 season averages per player
        self.season_averages = self.calculate_season_averages()
        print(f"   ✅ Calculated averages for {len(self.season_averages)} players from 2024/25")
        
    def calculate_season_averages(self) -> Dict[str, Dict]:
        """Calculate average points per game for each player in 2024/25"""
        print("🧮 Calculating 2024/25 season averages...")
        
        # Group by player (using player_code as unique identifier)
        player_stats = defaultdict(list)
        total_rows = 0
        valid_rows = 0
        
        for _, row in self.historical_data.iterrows():
            total_rows += 1
            player_code = str(row.get('player_code', ''))
            if player_code and player_code != 'nan' and player_code != 'None':
                # Skip assistant managers
                position = self._normalize_position(row.get('position', 'MID'))
                if position is None:  # Skip AMs
                    continue
                    
                # Only include gameweeks where player actually played (minutes > 0)
                minutes = row.get('minutes', 0)
                if minutes > 0:
                    valid_rows += 1
                    # Safely convert numeric fields
                    def safe_float(val, default=0.0):
                        try:
                            return float(val) if val not in [None, '', 'nan', 'None'] else default
                        except (ValueError, TypeError):
                            return default
                    
                    player_stats[player_code].append({
                        'total_points': safe_float(row.get('total_points', 0)),
                        'minutes': safe_float(minutes),
                        'goals_scored': safe_float(row.get('goals_scored', 0)),
                        'assists': safe_float(row.get('assists', 0)),
                        'clean_sheets': safe_float(row.get('clean_sheets', 0)),
                        'saves': safe_float(row.get('saves', 0)),
                        'web_name': row.get('name', 'Unknown'),
                        'position': position,  # Use normalized position
                        'team': row.get('team', 'Unknown')
                    })
        
        print(f"   📊 Processed {total_rows} total rows, {valid_rows} with minutes > 0")
        print(f"   📊 Found {len(player_stats)} unique players with game time")
        
        # Calculate averages for each player
        averages = {}
        for player_code, games in player_stats.items():
            if len(games) >= 3:  # Only include players with at least 3 games
                avg_points = np.mean([g['total_points'] for g in games])
                total_minutes = sum([g['minutes'] for g in games])
                total_games = len(games)
                
                averages[player_code] = {
                    'avg_points_per_game': avg_points,
                    'total_games_played': total_games,
                    'total_minutes': total_minutes,
                    'avg_minutes_per_game': total_minutes / total_games,
                    'avg_goals_per_game': np.mean([g['goals_scored'] for g in games]),
                    'avg_assists_per_game': np.mean([g['assists'] for g in games]),
                    'avg_clean_sheets_per_game': np.mean([g['clean_sheets'] for g in games]),
                    'avg_saves_per_game': np.mean([g['saves'] for g in games]),
                    'web_name': games[0]['web_name'],
                    'position': games[0]['position'],
                    'team_2425': games[0]['team']
                }
        
        return averages
    
    def generate_2526_predictions(self) -> List[Dict]:
        """Generate predictions for 2025/26 GW1"""
        print("🔮 Generating 2025/26 GW1 predictions...")
        
        engine = FPLPredictionEngine()
        predictions_data = engine.generate_gameweek_predictions(gameweek=1, include_bonus=True)
        
        return predictions_data['players']
    
    def match_players_and_analyze(self) -> Dict[str, Any]:
        """Match current players with historical performance and analyze differences"""
        print("🔍 Matching players and analyzing prediction vs reality...")
        
        # Generate current predictions
        current_predictions = self.generate_2526_predictions()
        
        # Create lookup for current players by name (as backup)
        current_player_lookup = {}
        for player in self.current_players:
            name = player.get('web_name', '').strip()
            if name:
                current_player_lookup[name] = player
        
        matched_data = []
        unmatched_predictions = []
        unmatched_historical = []
        
        # Try to match each prediction with historical data
        for pred in current_predictions:
            player_name = pred['name']
            matched = False
            
            # First, try to find by player code in current players data
            current_player = None
            for player in self.current_players:
                if player.get('web_name', '').strip() == player_name:
                    current_player = player
                    break
            
            if current_player:
                player_code = str(current_player.get('code', ''))
                
                # Look for this player code in historical averages
                if player_code in self.season_averages:
                    historical = self.season_averages[player_code]
                    
                    # Calculate prediction vs reality difference
                    predicted_points = pred['expected_points']
                    actual_avg_points = historical['avg_points_per_game']
                    difference = predicted_points - actual_avg_points
                    percentage_diff = (difference / actual_avg_points * 100) if actual_avg_points > 0 else 0
                    
                    matched_data.append({
                        'player_name': player_name,
                        'player_code': player_code,
                        'position': pred['position'],
                        'current_price': pred['current_price'],
                        'predicted_points_2526': predicted_points,
                        'actual_avg_points_2425': actual_avg_points,
                        'difference': difference,
                        'percentage_difference': percentage_diff,
                        'games_played_2425': historical['total_games_played'],
                        'minutes_per_game_2425': historical['avg_minutes_per_game'],
                        'team_2425': historical['team_2425'],
                        'prediction_higher': difference > 0,
                        # Additional stats for deeper analysis
                        'predicted_goals': pred.get('expected_goals', 0),
                        'actual_avg_goals': historical['avg_goals_per_game'],
                        'predicted_assists': pred.get('expected_assists', 0),
                        'actual_avg_assists': historical['avg_assists_per_game'],
                        'predicted_saves': pred.get('expected_saves', 0),
                        'actual_avg_saves': historical['avg_saves_per_game'],
                    })
                    matched = True
            
            if not matched:
                unmatched_predictions.append(pred)
        
        # Find historical players without current predictions
        for player_code, historical in self.season_averages.items():
            found_in_predictions = any(
                str(cp.get('code', '')) == player_code 
                for cp in self.current_players
            )
            if not found_in_predictions:
                unmatched_historical.append(historical)
        
        print(f"   ✅ Matched {len(matched_data)} players")
        print(f"   ⚠️  {len(unmatched_predictions)} predictions without historical data")
        print(f"   ⚠️  {len(unmatched_historical)} historical players not in current season")
        
        return {
            'matched_data': matched_data,
            'unmatched_predictions': unmatched_predictions,
            'unmatched_historical': unmatched_historical
        }
    
    def analyze_by_position(self, matched_data: List[Dict]) -> Dict[str, Dict]:
        """Analyze prediction accuracy by position"""
        print("📈 Analyzing by position...")
        
        position_analysis = {}
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_data = [p for p in matched_data if p['position'] == position]
            
            if pos_data:
                predicted_points = [p['predicted_points_2526'] for p in pos_data]
                actual_points = [p['actual_avg_points_2425'] for p in pos_data]
                differences = [p['difference'] for p in pos_data]
                percentage_diffs = [p['percentage_difference'] for p in pos_data]
                
                position_analysis[position] = {
                    'count': len(pos_data),
                    'avg_predicted': np.mean(predicted_points),
                    'avg_actual': np.mean(actual_points),
                    'avg_difference': np.mean(differences),
                    'avg_percentage_difference': np.mean(percentage_diffs),
                    'median_difference': np.median(differences),
                    'std_difference': np.std(differences),
                    'predictions_higher_count': sum(1 for p in pos_data if p['prediction_higher']),
                    'predictions_higher_pct': sum(1 for p in pos_data if p['prediction_higher']) / len(pos_data) * 100,
                    'max_over_prediction': max(differences),
                    'max_under_prediction': min(differences),
                    'best_predicted_player': max(pos_data, key=lambda x: -abs(x['difference']))['player_name'],
                    'worst_predicted_player': max(pos_data, key=lambda x: abs(x['difference']))['player_name'],
                }
        
        return position_analysis
    
    def generate_summary_report(self, analysis_results: Dict) -> str:
        """Generate a comprehensive summary report"""
        matched_data = analysis_results['matched_data']
        position_analysis = self.analyze_by_position(matched_data)
        
        # Overall statistics
        total_matched = len(matched_data)
        overall_pred_avg = np.mean([p['predicted_points_2526'] for p in matched_data])
        overall_actual_avg = np.mean([p['actual_avg_points_2425'] for p in matched_data])
        overall_diff = overall_pred_avg - overall_actual_avg
        overall_pct_diff = (overall_diff / overall_actual_avg * 100) if overall_actual_avg > 0 else 0
        
        predictions_higher = sum(1 for p in matched_data if p['prediction_higher'])
        predictions_higher_pct = predictions_higher / total_matched * 100
        
        # Find extreme cases
        max_over = max(matched_data, key=lambda x: x['difference'])
        max_under = min(matched_data, key=lambda x: x['difference'])
        best_prediction = min(matched_data, key=lambda x: abs(x['difference']))
        worst_prediction = max(matched_data, key=lambda x: abs(x['difference']))
        
        report = f"""
🔍 FPL PREDICTION vs REALITY ANALYSIS
=====================================
Comparing 2025/26 GW1 predictions against 2024/25 season averages

📊 OVERALL SUMMARY
-----------------
• Players analyzed: {total_matched}
• Average predicted points (2025/26): {overall_pred_avg:.2f}
• Average actual points (2024/25): {overall_actual_avg:.2f}
• Overall difference: {overall_diff:+.2f} ({overall_pct_diff:+.1f}%)
• Predictions higher than reality: {predictions_higher}/{total_matched} ({predictions_higher_pct:.1f}%)

📈 POSITION BREAKDOWN
--------------------
"""
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            if position in position_analysis:
                pos = position_analysis[position]
                report += f"""
{position} ({pos['count']} players):
• Avg predicted: {pos['avg_predicted']:.2f} pts
• Avg actual (24/25): {pos['avg_actual']:.2f} pts  
• Difference: {pos['avg_difference']:+.2f} pts ({pos['avg_percentage_difference']:+.1f}%)
• Predictions higher: {pos['predictions_higher_count']}/{pos['count']} ({pos['predictions_higher_pct']:.1f}%)
• Best prediction: {pos['best_predicted_player']}
• Worst prediction: {pos['worst_predicted_player']}
"""
        
        report += f"""

🎯 EXTREME CASES
---------------
• Biggest over-prediction: {max_over['player_name']} ({max_over['position']})
  Predicted: {max_over['predicted_points_2526']:.2f}, Actual: {max_over['actual_avg_points_2425']:.2f}
  Difference: {max_over['difference']:+.2f} ({max_over['percentage_difference']:+.1f}%)

• Biggest under-prediction: {max_under['player_name']} ({max_under['position']})
  Predicted: {max_under['predicted_points_2526']:.2f}, Actual: {max_under['actual_avg_points_2425']:.2f}
  Difference: {max_under['difference']:+.2f} ({max_under['percentage_difference']:+.1f}%)

• Most accurate prediction: {best_prediction['player_name']} ({best_prediction['position']})
  Predicted: {best_prediction['predicted_points_2526']:.2f}, Actual: {best_prediction['actual_avg_points_2425']:.2f}
  Difference: {best_prediction['difference']:+.2f} ({best_prediction['percentage_difference']:+.1f}%)

• Least accurate prediction: {worst_prediction['player_name']} ({worst_prediction['position']})
  Predicted: {worst_prediction['predicted_points_2526']:.2f}, Actual: {worst_prediction['actual_avg_points_2425']:.2f}
  Difference: {worst_prediction['difference']:+.2f} ({worst_prediction['percentage_difference']:+.1f}%)

📋 TOP 10 OVER-PREDICTIONS (Model too optimistic)
"""
        
        over_predictions = sorted(matched_data, key=lambda x: x['difference'], reverse=True)[:10]
        for i, player in enumerate(over_predictions, 1):
            report += f"\n{i:2d}. {player['player_name']:20s} ({player['position']}) "
            report += f"Pred: {player['predicted_points_2526']:4.1f} vs Actual: {player['actual_avg_points_2425']:4.1f} "
            report += f"({player['difference']:+.1f})"
        
        report += f"""

📋 TOP 10 UNDER-PREDICTIONS (Model too pessimistic)
"""
        
        under_predictions = sorted(matched_data, key=lambda x: x['difference'])[:10]
        for i, player in enumerate(under_predictions, 1):
            report += f"\n{i:2d}. {player['player_name']:20s} ({player['position']}) "
            report += f"Pred: {player['predicted_points_2526']:4.1f} vs Actual: {player['actual_avg_points_2425']:4.1f} "
            report += f"({player['difference']:+.1f})"
        
        report += f"""

🎓 MODEL INSIGHTS
----------------
"""
        
        if overall_diff > 0:
            report += f"• Model is generally OPTIMISTIC (over-predicting by {overall_diff:.2f} pts on average)\n"
        else:
            report += f"• Model is generally PESSIMISTIC (under-predicting by {abs(overall_diff):.2f} pts on average)\n"
        
        # Position-specific insights
        most_optimistic_pos = max(position_analysis.keys(), key=lambda x: position_analysis[x]['avg_difference'])
        most_pessimistic_pos = min(position_analysis.keys(), key=lambda x: position_analysis[x]['avg_difference'])
        
        report += f"• Most optimistic for: {most_optimistic_pos} (+{position_analysis[most_optimistic_pos]['avg_difference']:.2f} pts)\n"
        report += f"• Most pessimistic for: {most_pessimistic_pos} ({position_analysis[most_pessimistic_pos]['avg_difference']:+.2f} pts)\n"
        
        # Data quality notes
        report += f"""
📝 DATA NOTES
-------------
• Analysis based on players who played ≥3 games in 2024/25
• Predictions are for single gameweek (GW1), actuals are season averages
• {len(analysis_results['unmatched_predictions'])} predictions couldn't be matched (likely new signings)
• {len(analysis_results['unmatched_historical'])} historical players not in current season
"""
        
        return report
    
    def export_detailed_data(self, analysis_results: Dict, filename: str = "prediction_vs_reality_detailed.csv"):
        """Export detailed comparison data to CSV"""
        matched_data = analysis_results['matched_data']
        
        df = pd.DataFrame(matched_data)
        df = df.sort_values('difference', ascending=False)
        
        export_path = Path(__file__).parent / filename
        df.to_csv(export_path, index=False)
        
        print(f"📁 Detailed data exported to: {export_path}")
        return export_path
    
    def _get_position_name(self, element_type: int) -> str:
        """Convert element type to position name"""
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return positions.get(element_type, 'MID')
    
    def _normalize_position(self, position: str) -> str:
        """Normalize position names to match our prediction format"""
        position = str(position).upper()
        if position == 'AM':
            return None  # Filter out Assistant Managers
        elif position == 'MID':
            return 'MID'
        elif position == 'GK':
            return 'GK'
        elif position == 'DEF':
            return 'DEF'
        elif position == 'FWD':
            return 'FWD'
        else:
            return 'MID'  # Default fallback
    
    def run_full_analysis(self):
        """Run the complete prediction vs reality analysis"""
        print("🚀 STARTING PREDICTION vs REALITY ANALYSIS")
        print("=" * 60)
        
        # Load all data
        self.load_data()
        
        # Match players and analyze
        analysis_results = self.match_players_and_analyze()
        
        # Generate and display report
        report = self.generate_summary_report(analysis_results)
        print(report)
        
        # Export detailed data
        csv_path = self.export_detailed_data(analysis_results)
        
        # Save report to file
        report_path = Path(__file__).parent / "prediction_vs_reality_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📁 Full report saved to: {report_path}")
        print(f"📁 Detailed data saved to: {csv_path}")
        
        print("\n" + "=" * 60)
        print("🎯 ANALYSIS COMPLETE")
        print("=" * 60)
        
        return analysis_results


def main():
    """Run the prediction vs reality analysis"""
    analyzer = PredictionRealityAnalyzer()
    results = analyzer.run_full_analysis()
    return results


if __name__ == "__main__":
    main()
