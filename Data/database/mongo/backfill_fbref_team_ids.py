#!/usr/bin/env python3
"""
Backfill fbref_id on MongoDB teams collection using FBRef SQLite database.

- Reads distinct team names from SQLite Match table (Premier League)
- Maps FPL teams to FBRef team names via alias map + exact matching
- Writes `fbref_id` (string FBRef team name) into each team document

Usage:
  python backfill_fbref_team_ids.py \
    --sqlite-path "/Users/owen/src/Personal/FBRef_DB/master.db" \
    --season 2024-2025

Options:
  --dry-run         Only print mappings, do not update Mongo
  --force           Overwrite existing fbref_id values

Env vars:
  FBREF_SQLITE_PATH   Path to SQLite DB (fallback if --sqlite-path omitted)
"""
import os
import sys
import argparse
import sqlite3
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Ensure Data path is importable
DATA_DIR = Path(__file__).resolve().parents[2]
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from database.mongo.fpl_mongo_client import FPLMongoClient

# Conservative alias map: FPL name -> FBRef team name
TEAM_ALIAS: Dict[str, str] = {
    "Manchester United": "Manchester Utd",
    "Man Utd": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Nottingham Forest": "Nott'ham Forest",
    "Nott'm Forest": "Nott'ham Forest",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Sheffield United": "Sheffield Utd",
    "Manchester City": "Manchester City",
    "Man City": "Manchester City",
    "Burnley": "Burnley",  # Note: may not be in current season
    "Leeds": "Leeds United",  # Note: may not be in current season
    "Sunderland": "Sunderland",  # Note: may not be in current season
    # Occasionally FPL names slightly differ in punctuation/case
}

PREMIER_LEAGUE_KEYS = {"Premier_League"}


def load_fbref_teams(sqlite_path: str, season: Optional[str]) -> List[str]:
    con = sqlite3.connect(sqlite_path)
    try:
        cur = con.cursor()
        params: Tuple = ()
        base_sql = "SELECT DISTINCT home_team FROM Match WHERE competition IN ({})".format(
            ",".join(["?" for _ in PREMIER_LEAGUE_KEYS])
        )
        comp_params = tuple(PREMIER_LEAGUE_KEYS)
        # Season filtering optional; if provided, include
        if season:
            base_sql += " AND season = ?"
            params = comp_params + (season,)
        else:
            params = comp_params
        cur.execute(base_sql, params)
        home = [r[0] for r in cur.fetchall()]
        cur.execute(base_sql.replace("home_team", "away_team"), params)
        away = [r[0] for r in cur.fetchall()]
        teams = sorted({t for t in home + away if isinstance(t, str) and t})
        return teams
    finally:
        con.close()


def pick_best_match(fpl_name: str, fbref_names: List[str]) -> Optional[str]:
    # Exact match first
    if fpl_name in fbref_names:
        return fpl_name
    # Alias mapping
    if fpl_name in TEAM_ALIAS and TEAM_ALIAS[fpl_name] in fbref_names:
        return TEAM_ALIAS[fpl_name]
    # Liberal fallback: drop punctuation and compare lowercased tokens
    import re
    def norm(s: str) -> str:
        return re.sub(r"[^a-z]", "", s.lower())
    f_norm = norm(fpl_name)
    for fb in fbref_names:
        if norm(fb) == f_norm:
            return fb
    # Token-based contains (e.g., "West Ham" vs "West Ham United")
    f_tokens = set(fpl_name.lower().split())
    best = None
    for fb in fbref_names:
        fb_tokens = set(fb.lower().split())
        if f_tokens.issubset(fb_tokens) or fb_tokens.issubset(f_tokens):
            best = fb
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite-path", default=os.getenv("FBREF_SQLITE_PATH", "/Users/owen/src/Personal/FBRef_DB/master.db"))
    ap.add_argument("--season", default=None, help="Season string like 2024-2025; defaults to latest if omitted")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sqlite_path = args.sqlite_path
    if not Path(sqlite_path).exists():
        print(f"❌ SQLite DB not found: {sqlite_path}")
        sys.exit(1)

    fbref_names = load_fbref_teams(sqlite_path, args.season)
    if not fbref_names:
        print("❌ No FBRef team names found")
        sys.exit(1)

    client = FPLMongoClient()
    try:
        teams = client.get_all_teams()
        updates = []
        missing = []
        for t in teams:
            fpl_name = t.get("name") or t.get("short_name")
            if not fpl_name:
                missing.append(t.get("id"))
                continue
            current = t.get("fbref_id")
            if current and not args.force:
                continue
            fb = pick_best_match(fpl_name, fbref_names)
            if fb:
                t["fbref_id"] = fb
                updates.append(t)
            else:
                missing.append(fpl_name)
        if args.dry_run:
            print("--- Proposed updates ---")
            for t in updates:
                print(f"{t.get('id')}: {t.get('name')} -> {t.get('fbref_id')}")
            if missing:
                print("--- Unmatched ---")
                for m in missing:
                    print(m)
            print(f"Would update {len(updates)} team docs")
        else:
            if updates:
                client.bulk_upsert_teams(updates)
                print(f"✅ Updated fbref_id for {len(updates)} teams")
            if missing:
                print(f"⚠️ Unmatched teams: {missing}")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
