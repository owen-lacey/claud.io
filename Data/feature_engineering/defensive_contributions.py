"""
Defensive Contributions Calculator for FPL 2025/26 Season

New FPL scoring rules:
- Defenders: 2 points for 10+ CBIT (Clearances, Blocks, Interceptions, Tackles_won)
- Midfielders/Forwards: 2 points for 12+ CBIRT (Clearances, Blocks, Interceptions, Tackles_won, Recoveries)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class DefensiveContributionCalculator:
    """Calculate defensive contribution points based on FPL 2025/26 rules"""
    
    # Thresholds for defensive contribution points
    DEFENDER_THRESHOLD = 10  # CBIT for defenders
    MIDFIELDER_FORWARD_THRESHOLD = 12  # CBIRT for midfielders/forwards
    POINTS_AWARDED = 2
    
    # FBRef to FPL position mapping
    POSITION_MAPPING = {
        # Defenders
        'CB': 'DEF', 'RB': 'DEF', 'LB': 'DEF', 'WB': 'DEF', 
        'RWB': 'DEF', 'LWB': 'DEF', 'DF': 'DEF',
        # Goalkeepers  
        'GK': 'GKP', 'GKP': 'GKP',
        # Midfielders
        'CM': 'MID', 'AM': 'MID', 'DM': 'MID', 'RM': 'MID', 'LM': 'MID',
        'CAM': 'MID', 'CDM': 'MID', 'MF': 'MID',
        # Forwards/Wingers
        'FW': 'FWD', 'CF': 'FWD', 'ST': 'FWD', 'RW': 'FWD', 'LW': 'FWD',
        'RF': 'FWD', 'LF': 'FWD', 'SS': 'FWD'
    }
    
    def __init__(self):
        """Initialize the defensive contribution calculator"""
        pass
    
    def normalize_position(self, position: str) -> str:
        """
        Convert FBRef position codes to FPL position codes
        
        Args:
            position: FBRef position string (e.g., 'CB', 'AM,CM', 'WB,LM')
            
        Returns:
            FPL position code ('DEF', 'MID', 'FWD', 'GKP')
        """
        if not position or pd.isna(position):
            return 'Unknown'
        
        position_str = str(position).upper().strip()
        
        # Handle compound positions (e.g., 'AM,CM', 'WB,LM')
        if ',' in position_str:
            # Split by comma and map each position
            positions = [pos.strip() for pos in position_str.split(',')]
            mapped_positions = []
            
            for pos in positions:
                if pos in self.POSITION_MAPPING:
                    mapped_positions.append(self.POSITION_MAPPING[pos])
                    
            if mapped_positions:
                # Return the most common position, prioritizing DEF > MID > FWD > GKP
                if 'DEF' in mapped_positions:
                    return 'DEF'
                elif 'MID' in mapped_positions:
                    return 'MID'  
                elif 'FWD' in mapped_positions:
                    return 'FWD'
                else:
                    return mapped_positions[0]
                    
        # Single position
        if position_str in self.POSITION_MAPPING:
            return self.POSITION_MAPPING[position_str]
            
        # Fallback - try to infer from common patterns
        if any(x in position_str for x in ['CB', 'RB', 'LB', 'WB', 'DEF']):
            return 'DEF'
        elif any(x in position_str for x in ['CM', 'AM', 'DM', 'MID']):
            return 'MID'
        elif any(x in position_str for x in ['FW', 'ST', 'CF', 'RW', 'LW', 'FOR']):
            return 'FWD'
        elif any(x in position_str for x in ['GK', 'GOA']):
            return 'GKP'
        
        return 'Unknown'
    
    def calculate_cbit_score(self, clearances: float, blocks: float, 
                           interceptions: float, tackles_won: float) -> float:
        """
        Calculate CBIT score (for defenders)
        
        Args:
            clearances: Number of clearances
            blocks: Number of blocks  
            interceptions: Number of interceptions
            tackles_won: Number of successful tackles
            
        Returns:
            Total CBIT score
        """
        return sum([
            clearances or 0,
            blocks or 0, 
            interceptions or 0,
            tackles_won or 0
        ])
    
    def calculate_cbirt_score(self, clearances: float, blocks: float,
                            interceptions: float, tackles_won: float, 
                            recoveries: float) -> float:
        """
        Calculate CBIRT score (for midfielders/forwards)
        
        Args:
            clearances: Number of clearances
            blocks: Number of blocks
            interceptions: Number of interceptions  
            tackles_won: Number of successful tackles
            recoveries: Number of ball recoveries
            
        Returns:
            Total CBIRT score
        """
        return sum([
            clearances or 0,
            blocks or 0,
            interceptions or 0, 
            tackles_won or 0,
            recoveries or 0
        ])
    
    def calculate_defensive_contribution_points(self, position: str, 
                                              clearances: float = 0,
                                              blocks: float = 0,
                                              interceptions: float = 0,
                                              tackles_won: float = 0,
                                              recoveries: float = 0) -> int:
        """
        Calculate defensive contribution points based on position and stats
        
        Args:
            position: Player position (FBRef or FPL format)
            clearances: Number of clearances
            blocks: Number of blocks
            interceptions: Number of interceptions
            tackles_won: Number of successful tackles
            recoveries: Number of ball recoveries
            
        Returns:
            Defensive contribution points (0 or 2)
        """
        # Normalize position to FPL format
        normalized_position = self.normalize_position(position)
        
        if normalized_position == 'DEF':
            # Defenders use CBIT (no recoveries)
            score = self.calculate_cbit_score(clearances, blocks, interceptions, tackles_won)
            return self.POINTS_AWARDED if score >= self.DEFENDER_THRESHOLD else 0
            
        elif normalized_position in ['MID', 'FWD']:
            # Midfielders and forwards use CBIRT (includes recoveries)
            score = self.calculate_cbirt_score(clearances, blocks, interceptions, tackles_won, recoveries)
            return self.POINTS_AWARDED if score >= self.MIDFIELDER_FORWARD_THRESHOLD else 0
            
        else:
            # Goalkeepers and unknown positions don't get defensive contribution points
            return 0
    
    def add_defensive_contribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add defensive contribution features to a DataFrame
        
        Args:
            df: DataFrame with defensive stats and position columns
            
        Returns:
            DataFrame with added defensive contribution features
        """
        df = df.copy()
        
        # Calculate raw defensive scores
        df['cbit_score'] = df.apply(lambda row: self.calculate_cbit_score(
            row.get('clearances', 0),
            row.get('blocks', 0), 
            row.get('interceptions', 0),
            row.get('tackles_won', 0)
        ), axis=1)
        
        df['cbirt_score'] = df.apply(lambda row: self.calculate_cbirt_score(
            row.get('clearances', 0),
            row.get('blocks', 0),
            row.get('interceptions', 0), 
            row.get('tackles_won', 0),
            row.get('recoveries', 0)
        ), axis=1)
        
        # Calculate defensive contribution points
        df['defensive_contribution_points'] = df.apply(lambda row: 
            self.calculate_defensive_contribution_points(
                row.get('position', ''),
                row.get('clearances', 0),
                row.get('blocks', 0),
                row.get('interceptions', 0),
                row.get('tackles_won', 0),
                row.get('recoveries', 0)
            ), axis=1
        )
        
        # Add probability features (useful for ML)
        df['cbit_threshold_ratio'] = df['cbit_score'] / self.DEFENDER_THRESHOLD
        df['cbirt_threshold_ratio'] = df['cbirt_score'] / self.MIDFIELDER_FORWARD_THRESHOLD
        
        return df
    
    def calculate_historical_defensive_contributions(self, fbref_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defensive contributions for historical FBRef data
        
        Expected columns in fbref_df:
        - match_id, player_id, position
        - clearances, blocks, interceptions, tackles_won, recoveries
        
        Args:
            fbref_df: DataFrame with FBRef defensive stats
            
        Returns:
            DataFrame with defensive contribution points added
        """
        # Add defensive contribution features
        enhanced_df = self.add_defensive_contribution_features(fbref_df)
        
        print(f"📊 Processed {len(enhanced_df)} player-match records")
        print(f"✅ Defensive contributions awarded: {enhanced_df['defensive_contribution_points'].sum()} points")
        
        # Summary statistics
        contribution_summary = enhanced_df.groupby('position')['defensive_contribution_points'].agg([
            'count', 'sum', 'mean'
        ]).round(3)
        
        print("\n📈 Defensive Contribution Summary by Position:")
        print(contribution_summary)
        
        return enhanced_df


def test_defensive_contribution_calculator():
    """Test the defensive contribution calculator with sample data"""
    calc = DefensiveContributionCalculator()
    
    # Test cases
    test_cases = [
        # Defender scenarios
        {'position': 'DEF', 'clearances': 8, 'blocks': 2, 'interceptions': 1, 'tackles_won': 0, 'recoveries': 5, 'expected': 2},  # 11 CBIT >= 10
        {'position': 'DEF', 'clearances': 5, 'blocks': 2, 'interceptions': 1, 'tackles_won': 1, 'recoveries': 10, 'expected': 0},  # 9 CBIT < 10
        
        # Midfielder scenarios  
        {'position': 'MID', 'clearances': 3, 'blocks': 1, 'interceptions': 4, 'tackles_won': 2, 'recoveries': 3, 'expected': 2},  # 13 CBIRT >= 12
        {'position': 'MID', 'clearances': 2, 'blocks': 1, 'interceptions': 3, 'tackles_won': 1, 'recoveries': 4, 'expected': 0},  # 11 CBIRT < 12
        
        # Forward scenarios
        {'position': 'FWD', 'clearances': 1, 'blocks': 0, 'interceptions': 2, 'tackles_won': 1, 'recoveries': 8, 'expected': 2},  # 12 CBIRT >= 12
        
        # Goalkeeper scenarios
        {'position': 'GKP', 'clearances': 15, 'blocks': 5, 'interceptions': 5, 'tackles_won': 5, 'recoveries': 10, 'expected': 0},  # GKP gets 0
    ]
    
    print("🧪 Testing Defensive Contribution Calculator\n")
    
    for i, case in enumerate(test_cases, 1):
        result = calc.calculate_defensive_contribution_points(
            case['position'], case['clearances'], case['blocks'], 
            case['interceptions'], case['tackles_won'], case['recoveries']
        )
        
        status = "✅" if result == case['expected'] else "❌"
        print(f"{status} Test {i}: {case['position']} - Expected: {case['expected']}, Got: {result}")
        
        if case['position'] == 'DEF':
            cbit = calc.calculate_cbit_score(case['clearances'], case['blocks'], case['interceptions'], case['tackles_won'])
            print(f"   CBIT Score: {cbit} (threshold: {calc.DEFENDER_THRESHOLD})")
        else:
            cbirt = calc.calculate_cbirt_score(case['clearances'], case['blocks'], case['interceptions'], case['tackles_won'], case['recoveries'])
            print(f"   CBIRT Score: {cbirt} (threshold: {calc.MIDFIELDER_FORWARD_THRESHOLD})")
        print()


if __name__ == "__main__":
    test_defensive_contribution_calculator()
