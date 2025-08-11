#!/usr/bin/env python3
"""
FBRef-FPL Player ID Crosswalk Builder
====================================

Creates a comprehensive crosswalk between FBRef and FPL player IDs with validation.

Features:
- Downloads latest FPL-ID-Map from GitHub  
- Builds canonical player dataset from FBRef database
- Creates validated crosswalk with fuzzy matching for unmapped players
- Outputs durable CSV and SQLite tables for pipeline use

Usage:
    python build_player_crosswalk.py [--output-dir Data/crosswalk/]
"""
import sqlite3
import pandas as pd
import requests
from io import StringIO
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from difflib import SequenceMatcher


class PlayerCrosswalkBuilder:
    
    def __init__(self, fbref_db_path: str, output_dir: str = "Data/crosswalk/"):
        self.fbref_db_path = fbref_db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ID mapping sources
        self.fpl_id_map_url = "https://raw.githubusercontent.com/ChrisMusson/FPL-ID-Map/refs/heads/main/Master.csv"
        
    def download_fpl_id_map(self) -> pd.DataFrame:
        """Download and parse the FPL-ID-Map Master.csv"""
        print("📥 Downloading FPL-ID-Map from GitHub...")
        response = requests.get(self.fpl_id_map_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download FPL-ID-Map: {response.status_code}")
        
        id_map_df = pd.read_csv(StringIO(response.text))
        print(f"✅ Downloaded {len(id_map_df)} FPL player records")
        return id_map_df
    
    def extract_fbref_players(self) -> pd.DataFrame:
        """Extract canonical player dataset from FBRef database"""
        print("📊 Extracting player data from FBRef database...")
        
        query = """
        SELECT 
            pi.player_id as fbref_player_id,
            pi.name as fbref_name,
            pi.nation,
            pi.age,
            pi.position,
            m.competition,
            m.season,
            COUNT(*) as match_count,
            SUM(pi.minutes) as total_minutes,
            AVG(CAST(pi.minutes AS FLOAT)) as avg_minutes
        FROM Player_Info pi
        JOIN Match m ON pi.match_id = m.match_id  
        WHERE m.competition = 'Premier_League'
        GROUP BY pi.player_id, pi.name, pi.nation, pi.age, pi.position
        ORDER BY total_minutes DESC
        """
        
        with sqlite3.connect(self.fbref_db_path) as conn:
            fbref_players = pd.read_sql_query(query, conn)
        
        print(f"✅ Extracted {len(fbref_players)} FBRef player-position records")
        print(f"   {fbref_players['fbref_player_id'].nunique()} unique players")
        return fbref_players
    
    def consolidate_player_positions(self, fbref_players: pd.DataFrame) -> pd.DataFrame:
        """Consolidate multiple position entries per player"""
        print("🔄 Consolidating player positions...")
        
        # Aggregate by player to get canonical record
        player_summary = fbref_players.groupby('fbref_player_id').agg({
            'fbref_name': 'first',  # Should be consistent
            'nation': 'first',
            'age': 'first', 
            'position': lambda x: '|'.join(sorted(set(x))),  # Combine positions
            'match_count': 'sum',
            'total_minutes': 'sum',
            'avg_minutes': 'mean'
        }).reset_index()
        
        # Determine primary position (most common or most minutes)
        position_priorities = {
            'GK': 1, 'CB': 2, 'LB': 3, 'RB': 4, 'WB': 5,
            'DM': 6, 'AM': 7, 'LW': 8, 'RW': 9, 'FW': 10
        }
        
        def get_primary_position(position_str: str) -> str:
            positions = position_str.split('|')
            # Remove compound positions, prefer simple ones
            simple_positions = [p for p in positions if ',' not in p and len(p) <= 3]
            if not simple_positions:
                simple_positions = [positions[0].split(',')[0]]  # Take first part of compound
            
            # Sort by priority (GK first, then outfield)
            simple_positions.sort(key=lambda x: position_priorities.get(x, 99))
            return simple_positions[0]
        
        player_summary['canonical_position'] = player_summary['position'].apply(get_primary_position)
        
        print(f"✅ Consolidated to {len(player_summary)} unique players")
        return player_summary
    
    def normalize_name(self, name: str) -> str:
        """Normalize player names for matching"""
        if pd.isna(name):
            return ""
        
        # Remove diacritics and special characters
        name = str(name).lower()
        # Simple diacritic removal (could be enhanced)
        replacements = {
            'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ā': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e', 'ē': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i', 'ī': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'ō': 'o', 'ø': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'ū': 'u',
            'ñ': 'n', 'ç': 'c', 'ş': 's', 'ğ': 'g', 'ı': 'i',
            'ć': 'c', 'č': 'c', 'ž': 'z', 'š': 's', 'đ': 'd'
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        # Remove extra spaces and hyphens
        name = re.sub(r'[^\w\s]', ' ', name)
        name = ' '.join(name.split())
        return name
    
    def build_crosswalk(self, fpl_map: pd.DataFrame, fbref_players: pd.DataFrame) -> pd.DataFrame:
        """Build validated crosswalk between FPL and FBRef IDs"""
        print("🔗 Building player ID crosswalk...")
        
        # Prepare FPL data
        fpl_map = fpl_map.copy()
        fpl_map['fpl_name'] = (fpl_map['first_name'].fillna('') + ' ' + fpl_map['second_name'].fillna('')).str.strip()
        fpl_map['fpl_name_normalized'] = fpl_map['fpl_name'].apply(self.normalize_name)
        
        # Prepare FBRef data  
        fbref_players = fbref_players.copy()
        fbref_players['fbref_name_normalized'] = fbref_players['fbref_name'].apply(self.normalize_name)
        
        # Direct mapping from existing ID map
        direct_maps = []
        fpl_to_fbref = dict(zip(fpl_map['code'], fpl_map['fbref']))
        
        for _, fbref_row in fbref_players.iterrows():
            fbref_id = fbref_row['fbref_player_id']
            
            # Find FPL matches for this FBRef ID
            fpl_matches = fpl_map[fpl_map['fbref'] == fbref_id]
            
            for _, fpl_row in fpl_matches.iterrows():
                direct_maps.append({
                    'fbref_player_id': fbref_id,
                    'fpl_player_code': fpl_row['code'],
                    'fbref_name': fbref_row['fbref_name'],
                    'fpl_name': fpl_row['fpl_name'],
                    'canonical_position': fbref_row['canonical_position'],
                    'nation': fbref_row['nation'],
                    'age': fbref_row['age'],
                    'total_minutes': fbref_row['total_minutes'],
                    'match_method': 'direct_id_map',
                    'confidence': 1.0
                })
        
        print(f"✅ Found {len(direct_maps)} direct ID matches")
        
        # Fuzzy matching for unmapped players
        mapped_fbref_ids = set(row['fbref_player_id'] for row in direct_maps)
        unmapped_fbref = fbref_players[~fbref_players['fbref_player_id'].isin(mapped_fbref_ids)]
        
        fuzzy_maps = []
        fuzzy_threshold = 0.8
        
        print(f"🔍 Attempting fuzzy matching for {len(unmapped_fbref)} unmapped players...")
        
        for _, fbref_row in unmapped_fbref.iterrows():
            fbref_name_norm = fbref_row['fbref_name_normalized']
            
            best_match = None
            best_score = 0
            
            for _, fpl_row in fpl_map.iterrows():
                if pd.notna(fpl_row['fbref']):  # Skip already mapped FPL players
                    continue
                    
                fpl_name_norm = fpl_row['fpl_name_normalized']
                score = SequenceMatcher(None, fbref_name_norm, fpl_name_norm).ratio()
                
                if score > best_score and score >= fuzzy_threshold:
                    best_score = score
                    best_match = fpl_row
            
            if best_match is not None:
                fuzzy_maps.append({
                    'fbref_player_id': fbref_row['fbref_player_id'],
                    'fpl_player_code': best_match['code'],
                    'fbref_name': fbref_row['fbref_name'],
                    'fpl_name': best_match['fpl_name'],
                    'canonical_position': fbref_row['canonical_position'],
                    'nation': fbref_row['nation'],
                    'age': fbref_row['age'],
                    'total_minutes': fbref_row['total_minutes'],
                    'match_method': 'fuzzy_name',
                    'confidence': round(best_score, 3)
                })
        
        print(f"✅ Found {len(fuzzy_maps)} fuzzy name matches")
        
        # Combine all matches
        all_matches = direct_maps + fuzzy_maps
        crosswalk_df = pd.DataFrame(all_matches)
        
        print(f"📊 Total crosswalk entries: {len(crosswalk_df)}")
        print(f"   Direct matches: {len(direct_maps)}")
        print(f"   Fuzzy matches: {len(fuzzy_maps)}")
        print(f"   Unmapped FBRef players: {len(unmapped_fbref) - len(fuzzy_maps)}")
        
        return crosswalk_df
    
    def save_crosswalk(self, crosswalk_df: pd.DataFrame):
        """Save crosswalk to CSV and SQLite"""
        print("💾 Saving crosswalk files...")
        
        # Save CSV
        csv_path = self.output_dir / "fbref_fpl_crosswalk.csv"
        crosswalk_df.to_csv(csv_path, index=False)
        print(f"✅ Saved CSV: {csv_path}")
        
        # Save SQLite  
        sqlite_path = self.output_dir / "player_crosswalk.db"
        with sqlite3.connect(sqlite_path) as conn:
            crosswalk_df.to_sql('player_crosswalk', conn, if_exists='replace', index=False)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fbref_id ON player_crosswalk(fbref_player_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fpl_code ON player_crosswalk(fpl_player_code)")
            
        print(f"✅ Saved SQLite: {sqlite_path}")
        
        # Print summary stats
        print("\n📊 Crosswalk Summary:")
        print(f"   Total mapped players: {len(crosswalk_df)}")
        print(f"   High confidence (≥0.9): {len(crosswalk_df[crosswalk_df['confidence'] >= 0.9])}")
        print(f"   Medium confidence (0.8-0.9): {len(crosswalk_df[(crosswalk_df['confidence'] >= 0.8) & (crosswalk_df['confidence'] < 0.9)])}")
        print(f"   Positions: {crosswalk_df['canonical_position'].value_counts().to_dict()}")
    
    def run(self):
        """Execute complete crosswalk building process"""
        print("🚀 Building FBRef-FPL Player Crosswalk...")
        
        # Download FPL ID mapping
        fpl_map = self.download_fpl_id_map()
        
        # Extract FBRef players  
        fbref_players = self.extract_fbref_players()
        
        # Consolidate positions
        fbref_consolidated = self.consolidate_player_positions(fbref_players)
        
        # Build crosswalk
        crosswalk = self.build_crosswalk(fpl_map, fbref_consolidated)
        
        # Save results
        self.save_crosswalk(crosswalk)
        
        print("✅ Crosswalk building complete!")


def main():
    parser = argparse.ArgumentParser(description="Build FBRef-FPL player ID crosswalk")
    parser.add_argument("--fbref-db", default="/Users/owen/src/Personal/FBRef_DB/master.db",
                       help="Path to FBRef SQLite database")
    parser.add_argument("--output-dir", default="Data/crosswalk/",
                       help="Output directory for crosswalk files")
    
    args = parser.parse_args()
    
    builder = PlayerCrosswalkBuilder(args.fbref_db, args.output_dir)
    builder.run()


if __name__ == "__main__":
    main()
