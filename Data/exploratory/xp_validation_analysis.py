#!/usr/bin/env python3
"""
FPL Expected Points Validation Analysis
======================================

Compare our ML-based expected points predictions against FPL's official xP values
from the 2024/25 season to evaluate model performance.

This script:
1. Loads our 2025/26 GW1-5 predictions
2. Loads FPL's 2024/25 xP data from parsed CSV files
3. Matches players between seasons using player codes
4. Analyzes prediction accuracy by position, gameweek, and value ranges
5. Generates detailed performance metrics and visualizations
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any

class XPValidationAnalyzer:
    """Analyze our xP predictions vs FPL's official xP values"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.predictions_dir = self.project_root / "predictions_2025_26"
        self.raw_data_path = self.project_root / "raw" / "parsed_gw_2425.csv"
        
        # Data containers
        self.our_predictions = {}  # {gw: [predictions]}
        self.fpl_xp_data = None   # DataFrame with FPL's xP values
        self.matched_data = []    # List of matched predictions vs reality
        
    def load_our_predictions(self) -> Dict[int, List[Dict]]:
        """Load our 2025/26 predictions for GW1-5"""
        print("🔮 Loading our 2025/26 GW1-5 predictions...")
        
        predictions = {}
        for gw in range(1, 6):
            pred_file = self.predictions_dir / f"predictions_gw{gw}_2025_26.json"
            
            if pred_file.exists():
                with open(pred_file, 'r') as f:
                    data = json.load(f)
                    predictions[gw] = data['players']
                    print(f"   ✅ GW{gw}: {len(data['players'])} predictions")
            else:
                print(f"   ❌ Missing GW{gw}: {pred_file}")
                
        self.our_predictions = predictions
        return predictions
    
    def load_fpl_xp_data(self) -> pd.DataFrame:
        """Load FPL's official xP data from 2024/25 season"""
        print(f"📊 Loading FPL xP data from {self.raw_data_path}...")
        
        try:
            # Load the parsed CSV with xP data
            df = pd.read_csv(self.raw_data_path)
            
            # Basic data cleaning
            df = df.dropna(subset=['xP', 'total_points', 'player_code'])
            df = df[df['player_code'] != '']  # Remove empty player codes
            
            # Convert data types
            df['xP'] = pd.to_numeric(df['xP'], errors='coerce')
            df['total_points'] = pd.to_numeric(df['total_points'], errors='coerce')
            df['GW'] = pd.to_numeric(df['GW'], errors='coerce')
            
            # Filter out invalid data
            df = df.dropna(subset=['xP', 'total_points', 'GW'])
            
            print(f"   ✅ Loaded {len(df):,} gameweek records")
            print(f"   📈 Gameweeks: {df['GW'].min():.0f} to {df['GW'].max():.0f}")
            print(f"   🧑‍💼 Unique players: {df['player_code'].nunique():,}")
            
            self.fpl_xp_data = df
            return df
            
        except Exception as e:
            print(f"   ❌ Error loading xP data: {e}")
            raise
    
    def calculate_player_xp_averages(self) -> Dict[str, Dict]:
        """Calculate average xP per game for each player from 2024/25 data"""
        print("📊 Calculating player xP averages from 2024/25 season...")
        
        # Ensure player_code is string type
        self.fpl_xp_data['player_code'] = self.fpl_xp_data['player_code'].astype(str)
        
        player_stats = {}
        
        # Group by player and calculate averages
        player_groups = self.fpl_xp_data.groupby('player_code')
        
        for player_code, group in player_groups:
            # Filter for players with meaningful playing time
            meaningful_data = group[group['minutes'] > 0]
            
            if len(meaningful_data) >= 5:  # At least 5 games with playing time
                stats = {
                    'name': group['name'].iloc[0],
                    'position': group['position'].iloc[0],
                    'team': group['team'].iloc[0],
                    'games_played': len(meaningful_data),
                    'total_games': len(group),
                    'avg_xp': meaningful_data['xP'].mean(),
                    'avg_actual_points': meaningful_data['total_points'].mean(),
                    'avg_minutes': meaningful_data['minutes'].mean(),
                    'xp_vs_actual_diff': meaningful_data['xP'].mean() - meaningful_data['total_points'].mean(),
                    'xp_std': meaningful_data['xP'].std(),
                    'consistency_score': 1 - (meaningful_data['xP'].std() / meaningful_data['xP'].mean()) if meaningful_data['xP'].mean() > 0 else 0
                }
                player_stats[str(player_code)] = stats  # Ensure key is string
        
        print(f"   ✅ Calculated averages for {len(player_stats)} players with sufficient data")
        return player_stats
    
    def match_predictions_with_xp(self) -> List[Dict]:
        """Match our predictions with FPL's historical xP averages"""
        print("🔗 Matching our predictions with FPL xP averages...")
        
        player_averages = self.calculate_player_xp_averages()
        matched_data = []
        
        total_predictions = 0
        matched_count = 0
        
        print(f"   📊 Historical averages calculated for {len(player_averages)} players")
        
        for gw, predictions in self.our_predictions.items():
            total_predictions += len(predictions)
            
            for pred in predictions:
                player_code = str(pred.get('code', ''))
                
                # Debug first few matches
                if matched_count < 5:
                    print(f"   🔍 Checking player {pred['name']} with code '{player_code}'")
                    if player_code in player_averages:
                        print(f"      ✅ Found match!")
                    else:
                        print(f"      ❌ No match found")
                        # Show sample of available codes
                        sample_codes = list(player_averages.keys())[:5]
                        print(f"      Sample available codes: {sample_codes}")
                
                if player_code in player_averages:
                    historical_stats = player_averages[player_code]
                    
                    # Calculate prediction vs historical average difference
                    our_prediction = pred['expected_points']
                    fpl_avg_xp = historical_stats['avg_xp']
                    difference = our_prediction - fpl_avg_xp
                    percentage_diff = (difference / fpl_avg_xp * 100) if fpl_avg_xp > 0 else 0
                    
                    matched_record = {
                        'gameweek': gw,
                        'player_name': pred['name'],
                        'player_code': player_code,
                        'position': pred['position'],
                        'team': pred['team'],
                        'current_price': pred['current_price'],
                        
                        # Our predictions
                        'our_expected_points': our_prediction,
                        'our_expected_goals': pred['expected_goals'],
                        'our_expected_assists': pred['expected_assists'],
                        'our_expected_minutes': pred['expected_minutes'],
                        'our_clean_sheet_prob': pred['clean_sheet_prob'],
                        
                        # FPL historical averages
                        'fpl_avg_xp': fpl_avg_xp,
                        'fpl_avg_actual_points': historical_stats['avg_actual_points'],
                        'fpl_avg_minutes': historical_stats['avg_minutes'],
                        'fpl_games_played': historical_stats['games_played'],
                        'fpl_xp_std': historical_stats['xp_std'],
                        'fpl_consistency_score': historical_stats['consistency_score'],
                        
                        # Comparison metrics
                        'xp_difference': difference,
                        'xp_percentage_diff': percentage_diff,
                        'our_prediction_higher': our_prediction > fpl_avg_xp,
                        'abs_difference': abs(difference),
                        'squared_difference': difference ** 2,
                        
                        # Value analysis
                        'value_tier': self._get_value_tier(pred['current_price']),
                        'minutes_tier': self._get_minutes_tier(pred['expected_minutes']),
                    }
                    
                    matched_data.append(matched_record)
                    matched_count += 1
        
        print(f"   ✅ Matched {matched_count:,} predictions out of {total_predictions:,} total ({matched_count/total_predictions*100:.1f}%)")
        
        self.matched_data = matched_data
        return matched_data
    
    def _get_value_tier(self, price: float) -> str:
        """Categorize players by price tier"""
        if price >= 10.0:
            return "Premium (10.0+)"
        elif price >= 7.5:
            return "Mid-tier (7.5-9.9)"
        elif price >= 5.5:
            return "Budget (5.5-7.4)"
        else:
            return "Cheap (< 5.5)"
    
    def _get_minutes_tier(self, minutes: float) -> str:
        """Categorize players by expected minutes"""
        if minutes >= 75:
            return "Nailed (75+ min)"
        elif minutes >= 45:
            return "Regular (45-74 min)"
        elif minutes >= 15:
            return "Rotation (15-44 min)"
        else:
            return "Bench (< 15 min)"
    
    def analyze_overall_performance(self) -> Dict[str, Any]:
        """Analyze overall prediction performance"""
        print("📈 Analyzing overall prediction performance...")
        
        if not self.matched_data:
            print("   ❌ No matched data available")
            return {}
        
        df = pd.DataFrame(self.matched_data)
        
        # Core metrics
        mae = df['abs_difference'].mean()
        rmse = np.sqrt(df['squared_difference'].mean())
        mse = df['squared_difference'].mean()
        
        # Bias analysis
        mean_diff = df['xp_difference'].mean()
        our_higher_pct = (df['our_prediction_higher'].sum() / len(df)) * 100
        
        # Correlation analysis
        correlation = df['our_expected_points'].corr(df['fpl_avg_xp'])
        
        # Distribution analysis
        our_mean = df['our_expected_points'].mean()
        our_std = df['our_expected_points'].std()
        fpl_mean = df['fpl_avg_xp'].mean()
        fpl_std = df['fpl_avg_xp'].std()
        
        # Accuracy tiers
        excellent = len(df[df['abs_difference'] <= 0.5]) / len(df) * 100
        good = len(df[(df['abs_difference'] > 0.5) & (df['abs_difference'] <= 1.0)]) / len(df) * 100
        poor = len(df[df['abs_difference'] > 2.0]) / len(df) * 100
        
        results = {
            'total_predictions': len(df),
            'mae': mae,
            'rmse': rmse,
            'mse': mse,
            'correlation': correlation,
            'mean_bias': mean_diff,
            'our_higher_percentage': our_higher_pct,
            'our_mean': our_mean,
            'our_std': our_std,
            'fpl_mean': fpl_mean,
            'fpl_std': fpl_std,
            'excellent_predictions_pct': excellent,
            'good_predictions_pct': good,
            'poor_predictions_pct': poor,
        }
        
        return results
    
    def analyze_by_position(self) -> Dict[str, Dict]:
        """Analyze prediction performance by position"""
        print("🎯 Analyzing performance by position...")
        
        df = pd.DataFrame(self.matched_data)
        position_analysis = {}
        
        for position in df['position'].unique():
            pos_data = df[df['position'] == position]
            
            if len(pos_data) > 0:
                analysis = {
                    'count': len(pos_data),
                    'mae': pos_data['abs_difference'].mean(),
                    'rmse': np.sqrt(pos_data['squared_difference'].mean()),
                    'correlation': pos_data['our_expected_points'].corr(pos_data['fpl_avg_xp']),
                    'mean_bias': pos_data['xp_difference'].mean(),
                    'our_higher_pct': (pos_data['our_prediction_higher'].sum() / len(pos_data)) * 100,
                    'avg_our_prediction': pos_data['our_expected_points'].mean(),
                    'avg_fpl_xp': pos_data['fpl_avg_xp'].mean(),
                    'excellent_pct': len(pos_data[pos_data['abs_difference'] <= 0.5]) / len(pos_data) * 100,
                }
                position_analysis[position] = analysis
        
        return position_analysis
    
    def analyze_by_value_tier(self) -> Dict[str, Dict]:
        """Analyze prediction performance by price tier"""
        print("💰 Analyzing performance by value tier...")
        
        df = pd.DataFrame(self.matched_data)
        value_analysis = {}
        
        for tier in df['value_tier'].unique():
            tier_data = df[df['value_tier'] == tier]
            
            if len(tier_data) > 0:
                analysis = {
                    'count': len(tier_data),
                    'mae': tier_data['abs_difference'].mean(),
                    'rmse': np.sqrt(tier_data['squared_difference'].mean()),
                    'correlation': tier_data['our_expected_points'].corr(tier_data['fpl_avg_xp']),
                    'mean_bias': tier_data['xp_difference'].mean(),
                    'our_higher_pct': (tier_data['our_prediction_higher'].sum() / len(tier_data)) * 100,
                    'avg_our_prediction': tier_data['our_expected_points'].mean(),
                    'avg_fpl_xp': tier_data['fpl_avg_xp'].mean(),
                }
                value_analysis[tier] = analysis
        
        return value_analysis
    
    def analyze_by_gameweek(self) -> Dict[int, Dict]:
        """Analyze prediction performance by gameweek"""
        print("📅 Analyzing performance by gameweek...")
        
        df = pd.DataFrame(self.matched_data)
        gw_analysis = {}
        
        for gw in sorted(df['gameweek'].unique()):
            gw_data = df[df['gameweek'] == gw]
            
            analysis = {
                'count': len(gw_data),
                'mae': gw_data['abs_difference'].mean(),
                'rmse': np.sqrt(gw_data['squared_difference'].mean()),
                'correlation': gw_data['our_expected_points'].corr(gw_data['fpl_avg_xp']),
                'mean_bias': gw_data['xp_difference'].mean(),
                'our_higher_pct': (gw_data['our_prediction_higher'].sum() / len(gw_data)) * 100,
            }
            gw_analysis[gw] = analysis
        
        return gw_analysis
    
    def find_biggest_differences(self, top_n: int = 20) -> Dict[str, List[Dict]]:
        """Find players with biggest prediction differences"""
        print(f"🔍 Finding top {top_n} biggest differences...")
        
        df = pd.DataFrame(self.matched_data)
        
        # Sort by absolute difference
        df_sorted = df.sort_values('abs_difference', ascending=False)
        
        biggest_diffs = df_sorted.head(top_n)[['player_name', 'position', 'team', 'current_price', 
                                             'our_expected_points', 'fpl_avg_xp', 'xp_difference', 
                                             'xp_percentage_diff']].to_dict('records')
        
        # Find biggest overestimations and underestimations
        overestimations = df_sorted[df_sorted['our_prediction_higher']].head(10)[
            ['player_name', 'position', 'our_expected_points', 'fpl_avg_xp', 'xp_difference']
        ].to_dict('records')
        
        underestimations = df_sorted[~df_sorted['our_prediction_higher']].head(10)[
            ['player_name', 'position', 'our_expected_points', 'fpl_avg_xp', 'xp_difference']
        ].to_dict('records')
        
        return {
            'biggest_differences': biggest_diffs,
            'biggest_overestimations': overestimations,
            'biggest_underestimations': underestimations
        }
    
    def create_visualizations(self):
        """Create visualization plots for the analysis"""
        print("📊 Creating visualizations...")
        
        df = pd.DataFrame(self.matched_data)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Scatter plot: Our predictions vs FPL xP
        ax1 = plt.subplot(3, 3, 1)
        plt.scatter(df['fpl_avg_xp'], df['our_expected_points'], alpha=0.6, s=30)
        
        # Add diagonal line (perfect predictions)
        max_val = max(df['fpl_avg_xp'].max(), df['our_expected_points'].max())
        min_val = min(df['fpl_avg_xp'].min(), df['our_expected_points'].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect prediction')
        
        plt.xlabel('FPL Historical Avg xP')
        plt.ylabel('Our Predicted xP')
        plt.title('Our Predictions vs FPL Historical xP')
        plt.legend()
        
        # Add correlation text
        correlation = df['our_expected_points'].corr(df['fpl_avg_xp'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Histogram of differences
        ax2 = plt.subplot(3, 3, 2)
        plt.hist(df['xp_difference'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Perfect prediction')
        plt.xlabel('Difference (Our Prediction - FPL xP)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Differences')
        plt.legend()
        
        # 3. Box plot by position
        ax3 = plt.subplot(3, 3, 3)
        positions = df['position'].unique()
        position_diffs = [df[df['position'] == pos]['xp_difference'].values for pos in positions]
        plt.boxplot(position_diffs, labels=positions)
        plt.axhline(0, color='red', linestyle='--', alpha=0.7)
        plt.ylabel('Prediction Difference')
        plt.title('Prediction Differences by Position')
        
        # 4. MAE by position
        ax4 = plt.subplot(3, 3, 4)
        position_mae = df.groupby('position')['abs_difference'].mean()
        plt.bar(position_mae.index, position_mae.values, alpha=0.7)
        plt.ylabel('Mean Absolute Error')
        plt.title('MAE by Position')
        plt.xticks(rotation=45)
        
        # 5. Performance by price tier
        ax5 = plt.subplot(3, 3, 5)
        value_mae = df.groupby('value_tier')['abs_difference'].mean()
        plt.bar(range(len(value_mae)), value_mae.values, alpha=0.7)
        plt.xticks(range(len(value_mae)), value_mae.index, rotation=45)
        plt.ylabel('Mean Absolute Error')
        plt.title('MAE by Price Tier')
        
        # 6. Performance by gameweek
        ax6 = plt.subplot(3, 3, 6)
        gw_mae = df.groupby('gameweek')['abs_difference'].mean()
        plt.plot(gw_mae.index, gw_mae.values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Gameweek')
        plt.ylabel('Mean Absolute Error')
        plt.title('MAE by Gameweek')
        plt.grid(True, alpha=0.3)
        
        # 7. Correlation by position
        ax7 = plt.subplot(3, 3, 7)
        position_corr = df.groupby('position').apply(
            lambda x: x['our_expected_points'].corr(x['fpl_avg_xp']) if len(x) > 1 else 0
        )
        plt.bar(position_corr.index, position_corr.values, alpha=0.7)
        plt.ylabel('Correlation')
        plt.title('Correlation by Position')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # 8. Bias analysis (mean difference by position)
        ax8 = plt.subplot(3, 3, 8)
        position_bias = df.groupby('position')['xp_difference'].mean()
        colors = ['red' if x < 0 else 'green' for x in position_bias.values]
        plt.bar(position_bias.index, position_bias.values, alpha=0.7, color=colors)
        plt.axhline(0, color='black', linestyle='-', alpha=0.5)
        plt.ylabel('Mean Bias (+ = Overestimate)')
        plt.title('Prediction Bias by Position')
        plt.xticks(rotation=45)
        
        # 9. Accuracy distribution
        ax9 = plt.subplot(3, 3, 9)
        accuracy_ranges = ['≤ 0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '> 2.0']
        accuracy_counts = [
            len(df[df['abs_difference'] <= 0.5]),
            len(df[(df['abs_difference'] > 0.5) & (df['abs_difference'] <= 1.0)]),
            len(df[(df['abs_difference'] > 1.0) & (df['abs_difference'] <= 1.5)]),
            len(df[(df['abs_difference'] > 1.5) & (df['abs_difference'] <= 2.0)]),
            len(df[df['abs_difference'] > 2.0])
        ]
        
        plt.pie(accuracy_counts, labels=accuracy_ranges, autopct='%1.1f%%', startangle=90)
        plt.title('Prediction Accuracy Distribution\n(Absolute Error Ranges)')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(__file__).parent / "xp_validation_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualizations saved to: {output_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        print("📝 Generating comprehensive analysis report...")
        
        overall = self.analyze_overall_performance()
        by_position = self.analyze_by_position()
        by_value = self.analyze_by_value_tier()
        by_gameweek = self.analyze_by_gameweek()
        differences = self.find_biggest_differences()
        
        report = f"""
🔍 FPL EXPECTED POINTS VALIDATION ANALYSIS
=========================================
Comparing our ML predictions against FPL's official xP values from 2024/25 season

📊 OVERALL PERFORMANCE SUMMARY
-----------------------------
• Total predictions analyzed: {overall['total_predictions']:,}
• Mean Absolute Error (MAE): {overall['mae']:.3f} points
• Root Mean Square Error (RMSE): {overall['rmse']:.3f} points
• Correlation with FPL xP: {overall['correlation']:.3f}
• Mean bias: {overall['mean_bias']:+.3f} points ({'Overestimate' if overall['mean_bias'] > 0 else 'Underestimate'})

📈 PREDICTION DISTRIBUTION
• Our average prediction: {overall['our_mean']:.2f} points (σ = {overall['our_std']:.2f})
• FPL average xP: {overall['fpl_mean']:.2f} points (σ = {overall['fpl_std']:.2f})
• Our predictions higher: {overall['our_higher_percentage']:.1f}% of the time

🎯 ACCURACY BREAKDOWN
• Excellent (≤ 0.5 error): {overall['excellent_predictions_pct']:.1f}%
• Good (0.5-1.0 error): {overall['good_predictions_pct']:.1f}%
• Poor (> 2.0 error): {overall['poor_predictions_pct']:.1f}%

🏃‍♂️ PERFORMANCE BY POSITION
---------------------------"""

        for position, stats in by_position.items():
            report += f"""
{position}:
  • Count: {stats['count']} predictions
  • MAE: {stats['mae']:.3f} | RMSE: {stats['rmse']:.3f} | Correlation: {stats['correlation']:.3f}
  • Bias: {stats['mean_bias']:+.3f} ({'Over' if stats['mean_bias'] > 0 else 'Under'}estimate)
  • Our avg: {stats['avg_our_prediction']:.2f} vs FPL avg: {stats['avg_fpl_xp']:.2f}
  • Excellent predictions: {stats['excellent_pct']:.1f}%"""

        report += f"""

💰 PERFORMANCE BY PRICE TIER
----------------------------"""

        for tier, stats in by_value.items():
            report += f"""
{tier}:
  • Count: {stats['count']} | MAE: {stats['mae']:.3f} | Correlation: {stats['correlation']:.3f}
  • Bias: {stats['mean_bias']:+.3f} | Our higher: {stats['our_higher_pct']:.1f}%"""

        report += f"""

📅 PERFORMANCE BY GAMEWEEK
--------------------------"""

        for gw, stats in by_gameweek.items():
            report += f"""
GW{gw}: MAE {stats['mae']:.3f} | RMSE {stats['rmse']:.3f} | Correlation {stats['correlation']:.3f} | Bias {stats['mean_bias']:+.3f}"""

        report += f"""

🔍 BIGGEST PREDICTION DIFFERENCES
---------------------------------
Top 10 Overestimations:"""

        for i, player in enumerate(differences['biggest_overestimations'][:10], 1):
            report += f"""
{i:2d}. {player['player_name']} ({player['position']}): Our {player['our_expected_points']:.2f} vs FPL {player['fpl_avg_xp']:.2f} (+{player['xp_difference']:.2f})"""

        report += f"""

Top 10 Underestimations:"""

        for i, player in enumerate(differences['biggest_underestimations'][:10], 1):
            report += f"""
{i:2d}. {player['player_name']} ({player['position']}): Our {player['our_expected_points']:.2f} vs FPL {player['fpl_avg_xp']:.2f} ({player['xp_difference']:.2f})"""

        report += f"""

🎯 KEY INSIGHTS
--------------
"""

        # Generate insights based on the data
        if overall['correlation'] > 0.7:
            report += "✅ Strong correlation with FPL xP indicates our model captures similar patterns\n"
        elif overall['correlation'] > 0.5:
            report += "📊 Moderate correlation with FPL xP - room for improvement\n"
        else:
            report += "⚠️ Low correlation with FPL xP - significant model discrepancies\n"

        if abs(overall['mean_bias']) < 0.3:
            report += "✅ Low overall bias - predictions are well calibrated\n"
        else:
            report += f"⚠️ Significant bias detected - we consistently {'overestimate' if overall['mean_bias'] > 0 else 'underestimate'}\n"

        if overall['excellent_predictions_pct'] > 40:
            report += "✅ High percentage of excellent predictions (≤ 0.5 error)\n"
        else:
            report += "📈 Opportunity to improve prediction accuracy\n"

        # Position-specific insights
        best_position = max(by_position.items(), key=lambda x: x[1]['correlation'])
        worst_position = min(by_position.items(), key=lambda x: x[1]['correlation'])
        
        report += f"📍 Best position accuracy: {best_position[0]} (correlation: {best_position[1]['correlation']:.3f})\n"
        report += f"📍 Worst position accuracy: {worst_position[0]} (correlation: {worst_position[1]['correlation']:.3f})\n"

        report += f"""

💡 RECOMMENDATIONS
-----------------
"""

        if overall['mae'] > 1.0:
            report += "• Consider model recalibration - MAE above 1.0 point\n"
        
        if overall['correlation'] < 0.6:
            report += "• Investigate feature engineering - correlation below 0.6\n"
        
        if abs(overall['mean_bias']) > 0.5:
            report += "• Address systematic bias in predictions\n"
        
        worst_mae_position = max(by_position.items(), key=lambda x: x[1]['mae'])
        if worst_mae_position[1]['mae'] > 1.2:
            report += f"• Focus on improving {worst_mae_position[0]} predictions (highest MAE: {worst_mae_position[1]['mae']:.3f})\n"

        return report
    
    def export_detailed_data(self, filename: str = "xp_validation_detailed.csv"):
        """Export detailed validation data to CSV"""
        df = pd.DataFrame(self.matched_data)
        
        output_path = Path(__file__).parent / filename
        df.to_csv(output_path, index=False)
        
        print(f"📁 Detailed validation data exported to: {output_path}")
        return output_path
    
    def run_complete_analysis(self):
        """Run the complete validation analysis"""
        print("🚀 Starting Complete xP Validation Analysis")
        print("=" * 60)
        
        try:
            # Load data
            self.load_our_predictions()
            self.load_fpl_xp_data()
            
            # Match and analyze
            self.match_predictions_with_xp()
            
            if not self.matched_data:
                print("❌ No matched data found - cannot proceed with analysis")
                return
            
            # Generate visualizations
            self.create_visualizations()
            
            # Generate and display report
            report = self.generate_report()
            print(report)
            
            # Export detailed data
            self.export_detailed_data()
            
            print("\n✅ Complete validation analysis finished!")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

def main():
    """Main execution function"""
    analyzer = XPValidationAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
