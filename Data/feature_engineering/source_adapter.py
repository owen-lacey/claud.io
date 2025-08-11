#!/usr/bin/env python3
"""
Source Adapter

Bridges different raw data sources (FPL API, FBRef canonical) into a
unified feature contract required by training and prediction engines.

This module focuses on producing a per-player-per-gameweek DataFrame
with a consistent set of base columns used by the feature engine.

Design goals:
- Be explicit about every column: source, units, dtype, null policy
- Never mutate inputs in-place
- Keep all derivations in one place for easy auditing

Status:
- FBRef path supports a minimal contract using canonical outputs from
  Data/fbref_ingest/transform.py. Some fields are placeholders (0.0)
  until we extend canonical queries to include them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Columns the feature engine commonly expects for rolling stats
BASE_CONTRACT_COLUMNS: List[str] = [
    'player_id', 'name', 'team', 'position', 'season', 'GW',
    'minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets',
    'saves', 'goals_conceded', 'yellow_cards', 'starts',
    'threat', 'creativity', 'bps',
    'expected_goals', 'expected_assists', 'expected_goal_involvements',
    'expected_goals_conceded',
]


@dataclass
class AdapterConfig:
    fbref_canonical_dir: Optional[Path] = None
    gw_mapping_csv: Optional[Path] = None
    team_map: Optional[Dict] = None
    position_map: Optional[Dict] = None
    season_label: str = '2025_26'


class SourceAdapter:
    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()

    def from_fbref(self) -> pd.DataFrame:
        if not self.config.fbref_canonical_dir:
            raise ValueError("fbref_canonical_dir must be provided in AdapterConfig")

        cdir = Path(self.config.fbref_canonical_dir)
        p_players = cdir / 'players.parquet'
        p_teams = cdir / 'teams.parquet'
        p_matches = cdir / 'matches.parquet'
        p_pgs = cdir / 'player_game_stats.parquet'

        for p in [p_players, p_teams, p_matches, p_pgs]:
            if not p.exists():
                raise FileNotFoundError(f"Missing canonical file: {p}")

        players = pd.read_parquet(p_players)
        teams = pd.read_parquet(p_teams).rename(columns={'name': 'team_name'})
        matches = pd.read_parquet(p_matches)
        pgs = pd.read_parquet(p_pgs)

        # Merge lookup info
        df = (
            pgs.merge(players, how='left', left_on='fbref_player_id', right_on='fbref_player_id', suffixes=('', '_player'))
               .merge(teams, how='left', left_on='fbref_team_id', right_on='fbref_team_id')
               .merge(matches[['match_id', 'match_date']], how='left', on='match_id')
        )

        # Coalesce match_date columns if duplicated by merge
        if 'match_date' not in df.columns:
            if 'match_date_x' in df.columns or 'match_date_y' in df.columns:
                left = df['match_date_x'] if 'match_date_x' in df.columns else pd.NaT
                right = df['match_date_y'] if 'match_date_y' in df.columns else pd.NaT
                df['match_date'] = pd.to_datetime(left).fillna(pd.to_datetime(right))
                # Clean up
                drop_cols = [c for c in ['match_date_x', 'match_date_y'] if c in df.columns]
                if drop_cols:
                    df.drop(columns=drop_cols, inplace=True)
            else:
                raise KeyError('match_date not found after merges')

        # Apply GW mapping
        df['match_date'] = pd.to_datetime(df['match_date'])
        if self.config.gw_mapping_csv and Path(self.config.gw_mapping_csv).exists():
            gw_map = pd.read_csv(self.config.gw_mapping_csv)
            gw_map['match_date'] = pd.to_datetime(gw_map['match_date'])
            df = df.merge(gw_map[['match_date', 'GW']], how='left', on='match_date')
        else:
            df['GW'] = pd.NA

        # Position mapping
        pos_map = (self.config.position_map or {})
        df['position'] = df.get('position', 'MID')
        df['position'] = df['position'].map(pos_map).fillna(df['position']).replace({
            'Goalkeeper': 'GK', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD'
        })

        # Team mapping (id or short name)
        team_map = (self.config.team_map or {})
        df['team'] = df['team_name'].map(team_map).fillna(df['team_name'])

        # Build base contract columns
        out = pd.DataFrame()
        out['player_id'] = df['fbref_player_id']
        out['name'] = df['name'] if 'name' in df.columns else df.get('player_name', df['fbref_player_id'].astype(str))
        out['team'] = df['team']
        out['position'] = df['position']
        out['season'] = self.config.season_label
        out['GW'] = df['GW']

        out['minutes'] = pd.to_numeric(df.get('minutes', 0), errors='coerce').fillna(0.0)
        out['expected_goals'] = pd.to_numeric(df.get('xg', 0), errors='coerce').fillna(0.0)
        out['expected_assists'] = pd.to_numeric(df.get('xa', 0), errors='coerce').fillna(0.0)
        out['expected_goal_involvements'] = out['expected_goals'] + out['expected_assists']

        # Placeholders / approximations
        defaults = {
            'total_points': 0.0,
            'goals_scored': pd.to_numeric(df.get('goals', 0), errors='coerce').fillna(0.0),
            'assists': pd.to_numeric(df.get('assists', 0), errors='coerce').fillna(0.0),
            'clean_sheets': 0.0,
            'saves': 0.0,
            'goals_conceded': 0.0,
            'yellow_cards': pd.to_numeric(df.get('yellow_cards', 0), errors='coerce').fillna(0.0),
            'starts': pd.to_numeric(df.get('starts', 0), errors='coerce').fillna(0.0),
            'expected_goals_conceded': 0.0,
            'bps': 0.0,
        }
        for k, v in defaults.items():
            out[k] = v

        key_passes = pd.to_numeric(df.get('key_passes', 0), errors='coerce').fillna(0.0)
        shots = pd.to_numeric(df.get('shots', 0), errors='coerce').fillna(0.0)
        out['creativity'] = key_passes.round(4)
        out['threat'] = shots.round(4)

        # Add rolling window features if present
        for window in [3, 5]:
            col_min = f'minutes_last_{window}'
            col_starts = f'starts_last_{window}'
            col_std = f'minutes_std_last_{window}'
            if col_min in df.columns:
                out[col_min] = df[col_min]
            if col_starts in df.columns:
                out[col_starts] = df[col_starts]
            if col_std in df.columns:
                out[col_std] = df[col_std]

        # Filter rows with missing GW
        out = out[~out['GW'].isna()].copy()
        if len(out) == 0:
            print('No rows after GW mapping; fill Data/fbref_ingest/gw_mapping.csv and rerun.')
        else:
            out['GW'] = out['GW'].astype(int)

        return self._ensure_contract(out)

    def from_fpl_like(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in BASE_CONTRACT_COLUMNS:
            if c not in out.columns:
                out[c] = 0
        num_cols = [c for c in BASE_CONTRACT_COLUMNS if c not in {'player_id', 'name', 'team', 'position', 'season'}]
        out[num_cols] = out[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        out['GW'] = out['GW'].astype(int)
        out['season'] = out['season'].astype(str)
        return self._ensure_contract(out)

    def _ensure_contract(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in BASE_CONTRACT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required cols for contract: {missing}")
        out = df[BASE_CONTRACT_COLUMNS].copy()
        out['player_id'] = out['player_id'].astype(str)
        out['name'] = out['name'].astype(str)
        out['team'] = out['team'].astype(str)
        out['position'] = out['position'].astype(str)
        out['season'] = out['season'].astype(str)
        if out['GW'].dtype.name != 'int64' and out['GW'].dtype.name != 'int32':
            out['GW'] = out['GW'].astype('int64')
        numeric_cols = [c for c in BASE_CONTRACT_COLUMNS if c not in {'player_id', 'name', 'team', 'position', 'season', 'GW'}]
        out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preview source adaptation outputs')
    parser.add_argument('--fbref-canonical', help='Path to Data/fbref_ingest/canonical directory')
    parser.add_argument('--gw-map', help='Optional CSV path with columns [match_date, GW]')
    args = parser.parse_args()

    if args.fbref_canonical:
        adapter = SourceAdapter(AdapterConfig(fbref_canonical_dir=Path(args.fbref_canonical), gw_mapping_csv=Path(args.gw_map) if args.gw_map else None))
        df = adapter.from_fbref()
        print(df.head(10))
        print(f"Rows: {len(df):,}, Columns: {list(df.columns)}")
    else:
        print("Provide --fbref-canonical to preview FBRef adaptation.")
