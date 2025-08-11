# FBRef Ingest Module

Purpose: Provide reproducible, deterministic ingestion of FBRef data from the SQLite database into canonical tables used for feature engineering and model training.

SQLite DB: /Users/owen/src/Personal/FBRef_DB/master.db

## Components
- schema_introspection.py: Utilities to inspect tables, columns, and foreign keys.
- queries.py: Parameterized SQL queries to generate canonical tables (players, teams, matches, player_game_stats, team_game_stats).
- transform.py: Normalization utilities (units, per-90, rolling windows, gameweek mapping) and output writers.

## Conventions
- Use UTC timestamps where applicable
- Explicit units in column names when ambiguous (e.g., minutes, per90)
- Deterministic GW mapping from match_date → FPL gameweek (cached table)
- All outputs have a schema_version and source metadata

## Quickstart
```
python -m fbref_ingest.schema_introspection --db "/Users/owen/src/Personal/FBRef_DB/master.db"
python -m fbref_ingest.queries --db "/Users/owen/src/Personal/FBRef_DB/master.db" --out Data/fbref_ingest/outputs
python -m fbref_ingest.transform --inputs Data/fbref_ingest/outputs --out Data/fbref_ingest/canonical
```

## Outputs
- Data/fbref_ingest/outputs/*.csv (raw query results)
- Data/fbref_ingest/canonical/*.parquet (cleaned, canonical tables)
- Data/fbref_ingest/gw_mapping.csv (date → FPL GW)
