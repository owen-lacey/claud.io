#!/usr/bin/env python3
"""
FBRef Transform Utilities

Normalize and aggregate raw FBRef extracts into canonical tables
used by the feature engineering pipeline.
"""
import argparse
from pathlib import Path
import pandas as pd


def per90(df: pd.DataFrame, cols, minutes_col='minutes') -> pd.DataFrame:
    factor = 90 / df[minutes_col].clip(lower=1)
    for c in cols:
        out = f"{c}_per90"
        df[out] = (df[c] * factor).round(4)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True, help='Directory with raw CSVs from queries.py')
    parser.add_argument('--out', required=True, help='Directory to write canonical outputs')
    args = parser.parse_args()

    in_dir = Path(args.inputs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and transform player_game_stats
    pgs_csv = in_dir / 'player_game_stats.csv'
    pgs = pd.read_csv(pgs_csv)

    # Ensure numeric
    for c in ['minutes','shots','key_passes','xg','xa','goals','assists','yellow_cards','starts']:
        if c in pgs.columns:
            pgs[c] = pd.to_numeric(pgs[c], errors='coerce').fillna(0)

    pgs = per90(pgs, cols=['shots', 'key_passes', 'xg', 'xa'])

    # Sort for rolling windows
    pgs = pgs.sort_values(['fbref_player_id', 'season', 'match_date'])

    # Rolling window features (last 3 and 5 matches)
    for window in [3, 5]:
        pgs[f'minutes_last_{window}'] = (
            pgs.groupby(['fbref_player_id', 'season'])['minutes']
            .rolling(window, min_periods=1).sum().reset_index(level=[0,1], drop=True)
        )
        pgs[f'starts_last_{window}'] = (
            pgs.groupby(['fbref_player_id', 'season'])['starts']
            .rolling(window, min_periods=1).sum().reset_index(level=[0,1], drop=True)
        )
        pgs[f'minutes_std_last_{window}'] = (
            pgs.groupby(['fbref_player_id', 'season'])['minutes']
            .rolling(window, min_periods=1).std().reset_index(level=[0,1], drop=True).fillna(0)
        )

    # Write parquet
    pgs.to_parquet(out_dir / 'player_game_stats.parquet', index=False)

    # Pass-through players, teams, matches
    for name in ['players', 'teams', 'matches', 'team_game_stats']:
        src = in_dir / f'{name}.csv'
        if src.exists():
            df = pd.read_csv(src)
            df.to_parquet(out_dir / f'{name}.parquet', index=False)

    # Create a stub GW mapping CSV template from matches unique dates if not present
    gw_map_path = out_dir.parent / 'gw_mapping.csv'
    matches_path = out_dir / 'matches.parquet'
    if matches_path.exists() and not gw_map_path.exists():
        m = pd.read_parquet(matches_path)
        uniq_dates = (
            pd.to_datetime(m['match_date'])
            .dropna()
            .drop_duplicates()
            .sort_values()
            .dt.strftime('%Y-%m-%d')
            .to_frame(name='match_date')
        )
        uniq_dates['GW'] = ''
        uniq_dates.to_csv(gw_map_path, index=False)
        print(f"Generated GW mapping template at {gw_map_path} — fill GW values and re-run adapter.")

    print(f"Canonical outputs written to {out_dir}")


if __name__ == '__main__':
    main()
