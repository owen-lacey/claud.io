#!/usr/bin/env python3
"""
Add FBRef IDs to MongoDB FPL Players
====================================

Downloads the FPL-ID-Map from GitHub and updates each player in your MongoDB
with the corresponding fbref_id, if available.

Usage:
    python add_fbref_ids_to_mongo.py
"""
import pandas as pd
import requests
from io import StringIO
import sys, os
from pathlib import Path

def main():
    # Ensure Data directory is in sys.path for import
    data_dir = Path(__file__).resolve().parent
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    from database.mongo.fpl_mongo_client import FPLMongoClient


    # Download the FPL-ID-Map Master.csv from GitHub
    csv_url = "https://raw.githubusercontent.com/ChrisMusson/FPL-ID-Map/refs/heads/main/Master.csv"
    response = requests.get(csv_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download FPL-ID-Map Master.csv: {response.status_code}")

    id_map_df = pd.read_csv(StringIO(response.text))

    # Print column names to identify correct mapping
    # Build mapping: FPL ID ('code') -> FBRef ID ('fbref')
    # Ensure mapping keys are integers
    fpl_to_fbref = {int(k): v for k, v in zip(id_map_df['code'], id_map_df['fbref']) if pd.notnull(k) and pd.notnull(v)}

    # Connect to MongoDB and fetch all players
    mongo_client = FPLMongoClient()
    players = mongo_client.get_all_players()



    # Print sample FPL IDs from MongoDB
    print("Sample FPL IDs from MongoDB (player['id']):", [p.get('id') for p in players[:10]])
    print("Sample FPL codes from MongoDB (player['code']):", [p.get('code') for p in players[:10]])

    # Print sample FPL IDs from mapping
    print("Sample FPL IDs from mapping (code):", list(fpl_to_fbref.keys())[:10])

    # Check for intersection between player['code'] and mapping keys
    player_codes = set(p.get('code') for p in players if p.get('code') is not None)
    mapping_codes = set(fpl_to_fbref.keys())
    intersection = player_codes & mapping_codes
    print(f"Number of matching codes between MongoDB and mapping: {len(intersection)}")
    if intersection:
        print("Sample matching codes:", list(intersection)[:10])

    updated_count = 0
    for player in players:
        code = player.get('code')
        if code is not None:
            try:
                code_int = int(code)
            except Exception:
                continue
            fbref_id = fpl_to_fbref.get(code_int)
            if fbref_id:
                player['fbref_id'] = fbref_id
                updated_count += 1

    if updated_count > 0:
        mongo_client.bulk_upsert_players(players)
        print(f"✅ Updated {updated_count} players with fbref_id in MongoDB.")
    else:
        print("No matching FPL IDs found in the mapping. No updates made.")

    mongo_client.disconnect()

if __name__ == "__main__":
    main()
