#!/usr/bin/env python3
"""
FPL Gameweek Predictions Generator
=================================

Generates predicted points for all FPL players for any gameweek
and saves results to JSON files for the 2025/26 season.

Usage:
    python generate_predictions.py [gameweek]
    
Examples:
    python generate_predictions.py         # Generates for GW1
    python generate_predictions.py 5       # Generates for GW5
    python generate_predictions.py 1-10    # Generates for GW1 through GW10

Output:
    - predictions_gw{X}_2025_26.json: Complete prediction data
    - summary_gw{X}_2025_26.json: Summary statistics and top performers
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the final_assembly directory to path
sys.path.append('../final_assembly')

from fpl_assembly_pipeline import FPLPredictionEngine


class PredictionGenerator:
    """Generates and saves FPL predictions for the 2025/26 season"""
    
    def __init__(self, output_dir: str = "./"):
        """Initialize the prediction generator"""
        self.output_dir = Path(output_dir)
        self.engine = FPLPredictionEngine()
        self.season = "2025_26"
        
    def generate_gameweek_predictions(self, gameweek: int) -> Dict[str, Any]:
        """Generate predictions for a specific gameweek"""
        
        print(f"\n🎯 Generating predictions for Gameweek {gameweek}")
        print("=" * 50)
        
        # Generate predictions using our engine
        predictions = self.engine.generate_gameweek_predictions(
            gameweek=gameweek,
            include_bonus=True
        )
        
        # Add metadata
        predictions['metadata'] = {
            'season': self.season,
            'gameweek': gameweek,
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0.0',
            'total_players': len(predictions['players'])
        }
        
        return predictions
    
    def save_predictions(self, predictions: Dict[str, Any], gameweek: int) -> Dict[str, str]:
        """Save predictions to JSON files"""
        
        # Full predictions file
        predictions_filename = f"predictions_gw{gameweek}_{self.season}.json"
        predictions_path = self.output_dir / predictions_filename
        
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Summary file
        summary_data = self._create_summary(predictions)
        summary_filename = f"summary_gw{gameweek}_{self.season}.json"
        summary_path = self.output_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Create CSV for easy viewing
        csv_filename = f"top_predictions_gw{gameweek}_{self.season}.csv"
        csv_path = self.output_dir / csv_filename
        self._save_csv_summary(predictions, csv_path)
        
        return {
            'predictions_file': str(predictions_path),
            'summary_file': str(summary_path),
            'csv_file': str(csv_path)
        }
    
    def _create_summary(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the predictions"""
        
        players = predictions['players']
        
        # Top performers by position
        positions = ['GK', 'DEF', 'MID', 'FWD']
        top_by_position = {}
        
        for pos in positions:
            pos_players = [p for p in players if p['position'] == pos]
            if pos_players:
                # Top 5 by expected points
                top_5 = sorted(pos_players, key=lambda p: p['expected_points'], reverse=True)[:5]
                # Top 5 by value (points per million)
                value_5 = sorted(pos_players, key=lambda p: p['points_per_million'], reverse=True)[:5]
                
                top_by_position[pos] = {
                    'top_expected_points': [
                        {
                            'name': p['name'],
                            'team': p['team'],
                            'expected_points': p['expected_points'],
                            'price': p['current_price']
                        } for p in top_5
                    ],
                    'top_value': [
                        {
                            'name': p['name'],
                            'team': p['team'],
                            'points_per_million': p['points_per_million'],
                            'expected_points': p['expected_points'],
                            'price': p['current_price']
                        } for p in value_5
                    ]
                }
        
        # Overall top performers
        top_20_overall = sorted(players, key=lambda p: p['expected_points'], reverse=True)[:20]
        top_20_value = sorted(players, key=lambda p: p['points_per_million'], reverse=True)[:20]
        
        # Price brackets analysis
        price_brackets = {
            'budget': [p for p in players if p['current_price'] <= 5.0],
            'mid_range': [p for p in players if 5.0 < p['current_price'] <= 8.0],
            'premium': [p for p in players if p['current_price'] > 8.0]
        }
        
        bracket_analysis = {}
        for bracket, bracket_players in price_brackets.items():
            if bracket_players:
                best_player = max(bracket_players, key=lambda p: p['expected_points'])
                avg_xp = sum(p['expected_points'] for p in bracket_players) / len(bracket_players)
                avg_ppm = sum(p['points_per_million'] for p in bracket_players) / len(bracket_players)
                
                bracket_analysis[bracket] = {
                    'count': len(bracket_players),
                    'best_player': best_player['name'],
                    'best_expected_points': best_player['expected_points'],
                    'average_expected_points': round(avg_xp, 2),
                    'average_points_per_million': round(avg_ppm, 2)
                }
        
        return {
            'metadata': predictions['metadata'],
            'gameweek': predictions['gameweek'],
            'summary_stats': predictions['summary'],
            'top_performers': {
                'overall_top_20': [
                    {
                        'rank': i + 1,
                        'name': p['name'],
                        'team': p['team'],
                        'position': p['position'],
                        'expected_points': p['expected_points'],
                        'price': p['current_price'],
                        'points_per_million': p['points_per_million']
                    } for i, p in enumerate(top_20_overall)
                ],
                'value_top_20': [
                    {
                        'rank': i + 1,
                        'name': p['name'],
                        'team': p['team'],
                        'position': p['position'],
                        'points_per_million': p['points_per_million'],
                        'expected_points': p['expected_points'],
                        'price': p['current_price']
                    } for i, p in enumerate(top_20_value)
                ]
            },
            'by_position': top_by_position,
            'price_analysis': bracket_analysis
        }
    
    def _save_csv_summary(self, predictions: Dict[str, Any], csv_path: Path):
        """Save top 50 players to CSV for easy viewing"""
        
        import csv
        
        players = predictions['players'][:50]  # Top 50 only
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'rank', 'name', 'team', 'position', 'expected_points',
                'expected_goals', 'expected_assists', 'expected_bonus',
                'current_price', 'points_per_million', 'ceiling', 'floor'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, player in enumerate(players):
                writer.writerow({
                    'rank': i + 1,
                    'name': player['name'],
                    'team': player['team'],
                    'position': player['position'],
                    'expected_points': player['expected_points'],
                    'expected_goals': player['expected_goals'],
                    'expected_assists': player['expected_assists'],
                    'expected_bonus': player['expected_bonus'],
                    'current_price': player['current_price'],
                    'points_per_million': player['points_per_million'],
                    'ceiling': player['ceiling'],
                    'floor': player['floor']
                })
    
    def generate_multiple_gameweeks(self, gameweeks: List[int]) -> Dict[int, Dict[str, str]]:
        """Generate predictions for multiple gameweeks"""
        
        results = {}
        
        for gw in gameweeks:
            print(f"\n📊 Processing Gameweek {gw}...")
            predictions = self.generate_gameweek_predictions(gw)
            file_paths = self.save_predictions(predictions, gw)
            results[gw] = file_paths
            
            print(f"✅ Saved predictions for GW{gw}")
            print(f"   📄 Full data: {Path(file_paths['predictions_file']).name}")
            print(f"   📋 Summary: {Path(file_paths['summary_file']).name}")
            print(f"   📊 CSV: {Path(file_paths['csv_file']).name}")
        
        return results


def parse_gameweek_range(gw_input: str) -> List[int]:
    """Parse gameweek input (e.g., '1', '5', '1-10')"""
    
    if '-' in gw_input:
        start, end = map(int, gw_input.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(gw_input)]


def main():
    """Main function to handle command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Generate FPL predictions for the 2025/26 season',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_predictions.py           # Generate GW1 predictions
  python generate_predictions.py 5         # Generate GW5 predictions
  python generate_predictions.py 1-5       # Generate GW1 through GW5
        """
    )
    
    parser.add_argument(
        'gameweeks',
        nargs='?',
        default='1',
        help='Gameweek(s) to generate predictions for (e.g., 1, 5, 1-10)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./',
        help='Output directory for prediction files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    try:
        # Parse gameweek input
        gameweeks = parse_gameweek_range(args.gameweeks)
        
        # Validate gameweek range
        for gw in gameweeks:
            if not (1 <= gw <= 38):
                raise ValueError(f"Gameweek {gw} is out of range (1-38)")
        
        print("🚀 FPL PREDICTIONS GENERATOR - 2025/26 SEASON")
        print("=" * 55)
        print(f"📅 Generating predictions for: GW{gameweeks[0]}" + 
              (f" to GW{gameweeks[-1]}" if len(gameweeks) > 1 else ""))
        print(f"📁 Output directory: {Path(args.output_dir).resolve()}")
        
        # Initialize generator
        generator = PredictionGenerator(args.output_dir)
        
        # Generate predictions
        if len(gameweeks) == 1:
            predictions = generator.generate_gameweek_predictions(gameweeks[0])
            file_paths = generator.save_predictions(predictions, gameweeks[0])
            
            print(f"\n✅ PREDICTIONS GENERATED FOR GW{gameweeks[0]}")
            print("=" * 40)
            print(f"📄 Full predictions: {Path(file_paths['predictions_file']).name}")
            print(f"📋 Summary: {Path(file_paths['summary_file']).name}")
            print(f"📊 CSV export: {Path(file_paths['csv_file']).name}")
            
            # Show top 5 performers
            top_5 = predictions['players'][:5]
            print(f"\n🏆 TOP 5 EXPECTED POINTS:")
            for i, player in enumerate(top_5):
                print(f"  {i+1}. {player['name']} ({player['position']}) - {player['expected_points']:.1f}pts")
        
        else:
            results = generator.generate_multiple_gameweeks(gameweeks)
            
            print(f"\n✅ PREDICTIONS GENERATED FOR {len(gameweeks)} GAMEWEEKS")
            print("=" * 50)
            print(f"📁 Generated {len(results) * 3} files total")
            print(f"🎯 Gameweeks: {gameweeks[0]} to {gameweeks[-1]}")
    
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
