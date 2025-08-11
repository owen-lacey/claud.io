# FBRef Schema Runbook

Purpose: guide to inspect the SQLite schema, align canonical queries, and produce inputs for the source adapter and feature engine.

SQLite DB: /Users/owen/src/Personal/FBRef_DB/master.db

## 1) Introspect the database

List all tables:

```
python -m fbref_ingest.schema_introspection --db "/Users/owen/src/Personal/FBRef_DB/master.db"
```

Describe a specific table (repeat for candidates like players, teams, matches, player_stats, team_stats):

```
python -m fbref_ingest.schema_introspection --db "/Users/owen/src/Personal/FBRef_DB/master.db" --table player_stats
```

Recommended to capture outputs into a file for review:

```
python -m fbref_ingest.schema_introspection --db "/Users/owen/src/Personal/FBRef_DB/master.db" | tee Data/fbref_ingest/_schema_tables.txt
python -m fbref_ingest.schema_introspection --db "/Users/owen/src/Personal/FBRef_DB/master.db" --table player_stats | tee Data/fbref_ingest/_schema_player_stats.txt
```

## 2) Align canonical queries

Edit `Data/fbref_ingest/queries.py` to match your actual table/column names found above, then generate raw CSVs:

```
python -m fbref_ingest.queries --db "/Users/owen/src/Personal/FBRef_DB/master.db" --out Data/fbref_ingest/outputs
```

Key outputs expected:
- players.csv
- teams.csv
- matches.csv
- player_game_stats.csv
- team_game_stats.csv

## 3) Transform to canonical parquet

Normalize, derive per-90s, and write canonical parquet files:

```
python -m fbref_ingest.transform --inputs Data/fbref_ingest/outputs --out Data/fbref_ingest/canonical
```

Outputs:
- Data/fbref_ingest/canonical/*.parquet

## 4) Create gameweek mapping (date → GW)

Until transform.py emits this automatically, create a CSV `Data/fbref_ingest/gw_mapping.csv` with columns:

```
match_date, GW
2025-08-11,1
2025-08-12,1
...
```

Tips:
- Use your fixtures calendar to map dates to the official FPL gameweeks for the 2025/26 season.
- Include every unique match_date from `matches.parquet`.

## 5) Preview the source adapter

Use the adapter to validate the canonical outputs form a valid feature contract:

```
python -m feature_engineering.source_adapter --fbref-canonical Data/fbref_ingest/canonical --gw-map Data/fbref_ingest/gw_mapping.csv
```

Expected: a preview of the adapted DataFrame with columns:
- player_id, name, team, position, season, GW
- minutes, total_points, goals_scored, assists, clean_sheets, saves, goals_conceded, yellow_cards, starts
- threat, creativity, bps
- expected_goals, expected_assists, expected_goal_involvements, expected_goals_conceded

Notes:
- Some FPL-specific fields are placeholders in the first iteration; extend queries and transform as needed to populate.
- The adapter drops rows with missing GW; ensure the GW mapping covers all matches.
