"""
FBRef Prediction Engine

Loads FBRef-trained models from Data/models_fbref, loads canonical features for the requested gameweek,
and generates predictions for all targets (minutes, xG, xA, saves, goals_conceded).
"""
from pathlib import Path
import pandas as pd
import joblib
import os
import math
from typing import Optional, Set
from typing import Dict

class FBRefPredictionEngine:
    def __init__(self, models_dir: str = "Data/models_fbref"):
        self.models_dir = Path(models_dir)
        self.target_names = [
            "minutes",
            "xg",
            "xa",
            "saves",
            "goals_conceded"
        ]
        self.models = self._load_models()
        # Per-target feature columns (enhanced with team strength features)
        self.FEATURES = {
            "xg": [
                'shots', 'shots_per90', 'key_passes', 'key_passes_per90',
                'minutes_last_3', 'minutes_last_5', 'starts_last_3', 'starts_last_5',
                'minutes_std_last_3', 'minutes_std_last_5',
                'goals', 'assists', 'position', 'fbref_team_id', 'minutes',
                'team_attack_index', 'team_defense_index'
            ],
            "xa": [
                'key_passes', 'key_passes_per90', 'assists',
                'minutes_last_3', 'minutes_last_5', 'starts_last_3', 'starts_last_5',
                'minutes_std_last_3', 'minutes_std_last_5',
                'shots', 'shots_per90', 'goals', 'position', 'fbref_team_id', 'minutes',
                'team_attack_index', 'team_defense_index'
            ],
            "minutes": [
                'minutes_last_3', 'minutes_last_5', 'starts_last_3', 'starts_last_5',
                'minutes_std_last_3', 'minutes_std_last_5',
                'shots_per90', 'key_passes_per90', 'xg_per90', 'xa_per90',
                'goals', 'assists', 'yellow_cards', 'position', 'fbref_team_id',
                'team_attack_index', 'team_defense_index'
            ],
            "saves": [
                'minutes_last_3', 'minutes_last_5', 'starts_last_3', 'starts_last_5',
                'minutes_std_last_3', 'minutes_std_last_5',
                'shots', 'shots_per90', 'key_passes', 'key_passes_per90',
                'goals', 'assists', 'yellow_cards', 'fbref_team_id', 'minutes',
                # Enhanced with team strength features
                'team_defense_index', 'opponent_attack_index'
            ],
            # Team-level model; loaded from team_game_stats
            "goals_conceded": [
                'shots_allowed', 'xg_allowed'
            ]
        }
        # --- Fixture-aware GC settings (configurable via env) ---
        self.gc_use_fixtures = os.getenv('FBREF_GC_USE_FIXTURES', 'true').lower() in ('1', 'true', 'yes')
        try:
            self.gc_rolling_n = int(os.getenv('FBREF_GC_ROLLING_N', '6'))
        except Exception:
            self.gc_rolling_n = 6
        try:
            self.gc_home_factor = float(os.getenv('FBREF_GC_HOME_FACTOR', '0.95'))
            self.gc_away_factor = float(os.getenv('FBREF_GC_AWAY_FACTOR', '1.05'))
        except Exception:
            self.gc_home_factor = 0.95
            self.gc_away_factor = 1.05

    def _load_models(self):
        models = {}
        # Model filenames (enhanced models are now primary)
        model_filenames = {
            "minutes": "minutes_model.joblib",
            "xg": "xg_model.joblib", 
            "xa": "xa_model.joblib",
            "saves": "saves_model.joblib",
            "goals_conceded": "goals_conceded_model.joblib"
        }
        for target in self.target_names:
            joblib_path = self.models_dir / target / model_filenames[target]
            pkl_path = joblib_path.with_suffix('.pkl')
            if joblib_path.exists():
                models[target] = joblib.load(joblib_path)
            elif pkl_path.exists():
                models[target] = joblib.load(pkl_path)
            else:
                raise FileNotFoundError(f"Model file not found: {joblib_path} or {pkl_path}")
        return models

    def _load_features(self, gameweek: int):
        # Always resolve from project root for robustness
        base_dir = Path(__file__).resolve().parent.parent.parent / "Data" / "fbref_ingest" / "canonical"
        parquet_path = base_dir / f"features_gw{gameweek}.parquet"
        csv_path = base_dir / f"features_gw{gameweek}.csv"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Features file not found for GW{gameweek}: {parquet_path} or {csv_path}")

    def _load_team_features(self):
        base_dir = Path(__file__).resolve().parent.parent.parent / "Data" / "fbref_ingest" / "canonical"
        # Prefer parquet
        tpq = base_dir / "team_game_stats.parquet"
        tcsv = base_dir / "team_game_stats.csv"
        if tpq.exists():
            return pd.read_parquet(tpq)
        elif tcsv.exists():
            return pd.read_csv(tcsv)
        else:
            raise FileNotFoundError(f"Team canonical not found: {tpq} or {tcsv}")

    def _load_matches(self):
        """Load canonical matches table (home/away xG)."""
        base_dir = Path(__file__).resolve().parent.parent.parent / "Data" / "fbref_ingest" / "canonical"
        mpq = base_dir / "matches.parquet"
        mcsv = base_dir / "matches.csv"
        if mpq.exists():
            return pd.read_parquet(mpq)
        elif mcsv.exists():
            return pd.read_csv(mcsv)
        else:
            raise FileNotFoundError(f"Matches canonical not found: {mpq} or {mcsv}")

    def _compute_team_indices(self, rolling_n: int):
        """Compute simple attack and defense indices from canonical data.
        attack_index: mean xG for over last N matches (home_xg/away_xg)
        defense_index: mean xG allowed (team_game_stats.xg_allowed) over last N matches
        Returns: (attack_idx: dict[str,float], defense_idx: dict[str,float], league_avg_attack: float)
        """
        try:
            matches = self._load_matches()
            team_stats = self._load_team_features()
        except Exception:
            return {}, {}, 1.0

        # Restrict to latest season in data to avoid mixing eras
        season_col = 'season' if 'season' in matches.columns else None
        if season_col is not None and not matches.empty:
            latest_season = sorted(matches['season'].unique())[-1]
            matches = matches[matches['season'] == latest_season].copy()
        if 'season' in team_stats.columns and not team_stats.empty:
            latest_season_ts = sorted(team_stats['season'].unique())[-1]
            team_stats = team_stats[team_stats['season'] == latest_season_ts].copy()

        # Build per-team xG for list using both home and away rows
        atk_vals: Dict[str, list] = {}
        for _, r in matches.iterrows():
            ht, at = r.get('home_team_id'), r.get('away_team_id')
            hxg, axg = float(r.get('home_xg', 0) or 0), float(r.get('away_xg', 0) or 0)
            if isinstance(ht, str):
                atk_vals.setdefault(ht, []).append(hxg)
            if isinstance(at, str):
                atk_vals.setdefault(at, []).append(axg)
        # Rolling last N mean
        attack_idx: Dict[str, float] = {t: (sum(v[-rolling_n:]) / max(1, min(len(v), rolling_n))) for t, v in atk_vals.items() if v}
        # League avg attack
        all_last = [x for v in atk_vals.values() for x in v[-rolling_n:]]
        league_avg_attack = (sum(all_last) / max(1, len(all_last))) if all_last else 1.0

        # Defense index from xg_allowed in team_game_stats
        def_vals: Dict[str, list] = {}
        for _, r in team_stats.iterrows():
            t = r.get('fbref_team_id')
            if isinstance(t, str):
                def_vals.setdefault(t, []).append(float(r.get('xg_allowed', 0) or 0))
        defense_idx: Dict[str, float] = {t: (sum(v[-rolling_n:]) / max(1, min(len(v), rolling_n))) for t, v in def_vals.items() if v}

        return attack_idx, defense_idx, (league_avg_attack if league_avg_attack > 0 else 1.0)

    def _build_fixture_map_fbref(self, gameweek: int) -> Dict[str, Dict[str, str]]:
        """Build a mapping from FBRef team id -> {opponent: fbref_id, venue: 'H'|'A'} for the GW.
        Requires Mongo teams to have 'fbref_id' field. Falls back to empty mapping if Mongo not available.
        """
        try:
            # Late import to avoid hard dependency if Mongo not present
            import sys as _sys
            from pathlib import Path as _Path
            data_dir = _Path(__file__).resolve().parent.parent.parent / 'Data'
            if str(data_dir) not in _sys.path:
                _sys.path.insert(0, str(data_dir))
            from database.mongo.mongo_data_loader import load_teams_data, get_fixtures_by_gameweek
        except Exception:
            return {}
        try:
            teams = load_teams_data()
            fixtures = get_fixtures_by_gameweek(gameweek)
        except Exception:
            return {}
        # Build FPL team id -> fbref_id map
        fpl_to_fbref: Dict[int, str] = {}
        for t in teams or []:
            tid = t.get('id')
            fb = t.get('fbref_id') or t.get('fbref_team_id')
            if tid is not None and isinstance(fb, str) and fb:
                fpl_to_fbref[int(tid)] = fb
        fmap: Dict[str, Dict[str, str]] = {}
        for fx in fixtures or []:
            th, ta = fx.get('team_h'), fx.get('team_a')
            h_fb = fpl_to_fbref.get(int(th)) if th is not None else None
            a_fb = fpl_to_fbref.get(int(ta)) if ta is not None else None
            if isinstance(h_fb, str) and isinstance(a_fb, str):
                fmap[h_fb] = {'opponent': a_fb, 'venue': 'H'}
                fmap[a_fb] = {'opponent': h_fb, 'venue': 'A'}
        return fmap

    def _encode_categoricals(self, df: pd.DataFrame, cols):
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype('category').cat.codes
        return df

    def _normalize_position(self, pos_val):
        # Normalize a variety of possible position labels into FPL-like ones
        if pos_val is None:
            return 'UNK'
        p = str(pos_val).upper()
        if p in ('GK', 'G', 'GOALKEEPER'):
            return 'GK'
        if p in ('DEF', 'D', 'DF', 'DEFENDER'):
            return 'DEF'
        if p in ('MID', 'M', 'MF', 'MIDFIELDER'):
            return 'MID'
        if p in ('FWD', 'FW', 'ST', 'ATT', 'FORWARD', 'STRIKER'):
            return 'FWD'
        return str(pos_val)

    def generate_gameweek_predictions(self, gameweek: int, include_bonus: bool = True, allowed_fbref_ids: Optional[Set[str]] = None, element_type_map: Optional[Dict[str, int]] = None):
        # Load base player-level features (latest matches already materialized in features_gw{n}.*)
        base_df = self._load_features(gameweek)

        # Ensure identifiers available
        pid_col = 'fbref_player_id' if 'fbref_player_id' in base_df.columns else base_df.columns[0]
        name_col = 'player_name' if 'player_name' in base_df.columns else ('name' if 'name' in base_df.columns else None)
        team_col = 'team_name' if 'team_name' in base_df.columns else ('fbref_team_id' if 'fbref_team_id' in base_df.columns else None)
        pos_col = 'position' if 'position' in base_df.columns else None

        # Copy to avoid SettingWithCopy and reset index to align with model.predict outputs
        df = base_df.copy().reset_index(drop=True)

        # Optional filtering to a provided set of FBRef player IDs
        if allowed_fbref_ids is not None and 'fbref_player_id' in df.columns:
            allowed_set = {str(x) for x in allowed_fbref_ids}
            df = df[df['fbref_player_id'].astype(str).isin(allowed_set)].reset_index(drop=True)

        # If Mongo-provided element_type map is available, override/define position using it
        if element_type_map is not None and 'fbref_player_id' in df.columns:
            etype_series = df['fbref_player_id'].astype(str).map(element_type_map)
            # Map 1,2,3,4 -> GK, DEF, MID, FWD
            def map_et(e):
                if e == 1: return 'GK'
                if e == 2: return 'DEF'
                if e == 3: return 'MID'
                if e == 4: return 'FWD'
                return None
            df['position'] = etype_series.apply(map_et).fillna(df.get('position'))
            pos_col = 'position'

        # If no rows after filtering, return empty structure
        if df.empty:
            return { 'players': [], 'summary': {'total_players': 0, 'average_expected_points': 0.0, 'max_expected_points': 0.0}, 'gameweek': gameweek }

        # Prepare predictions containers
        preds = {t: None for t in self.target_names}

        # XG
        xg_feats = self.FEATURES['xg']
        xg_X = df.reindex(columns=xg_feats, fill_value=0).copy()
        xg_X = self._encode_categoricals(xg_X, ['position', 'fbref_team_id'])
        preds['xg'] = self.models['xg'].predict(xg_X)

        # XA
        xa_feats = self.FEATURES['xa']
        xa_X = df.reindex(columns=xa_feats, fill_value=0).copy()
        xa_X = self._encode_categoricals(xa_X, ['position', 'fbref_team_id'])
        preds['xa'] = self.models['xa'].predict(xa_X)

        # Minutes
        min_feats = self.FEATURES['minutes']
        min_X = df.reindex(columns=min_feats, fill_value=0).copy()
        min_X = self._encode_categoricals(min_X, ['position', 'fbref_team_id'])
        preds['minutes'] = self.models['minutes'].predict(min_X)

        # Saves (GK only) — predict for GKs and set others to 0
        saves_feats = self.FEATURES['saves']
        saves_X = df.reindex(columns=saves_feats, fill_value=0).copy()
        saves_X = self._encode_categoricals(saves_X, ['fbref_team_id'])
        if pos_col and df[pos_col].dtype.name == 'object':
            is_gk = df[pos_col] == 'GK'
        else:
            # If already encoded or missing, approximate no-filter
            is_gk = pd.Series([False] * len(df), index=df.index)
        saves_pred = pd.Series(0.0, index=df.index)
        if is_gk.any():
            saves_pred.loc[is_gk] = self.models['saves'].predict(saves_X.loc[is_gk])
        preds['saves'] = saves_pred.values

        # Goals conceded (team-level)
        team_df = self._load_team_features()
        gc_feats = self.FEATURES['goals_conceded']
        # Use latest date in team_df
        if 'match_date' in team_df.columns:
            latest_team_date = team_df['match_date'].max()
            team_latest = team_df[team_df['match_date'] == latest_team_date].copy()
        else:
            team_latest = team_df.copy()
        team_X = team_latest.reindex(columns=gc_feats, fill_value=0)
        team_pred = self.models['goals_conceded'].predict(team_X)
        # Map by team id
        team_id_col = 'fbref_team_id' if 'fbref_team_id' in team_latest.columns else None
        team_map = {}
        if team_id_col:
            team_map = dict(zip(team_latest[team_id_col], team_pred))
        gc_player = df['fbref_team_id'].map(team_map) if 'fbref_team_id' in df.columns else pd.Series(0.0, index=df.index)

        # --- Fixture-aware override for goals_conceded ---
        if self.gc_use_fixtures and 'fbref_team_id' in df.columns:
            try:
                fixture_map = self._build_fixture_map_fbref(gameweek)
                atk_idx, def_idx, league_avg = self._compute_team_indices(self.gc_rolling_n)
                def calc_fixture_gc(team_fb: str) -> Optional[float]:
                    info = fixture_map.get(team_fb)
                    if not info:
                        return None
                    opp = info['opponent']
                    venue = info['venue']
                    d = def_idx.get(team_fb)
                    a = atk_idx.get(opp)
                    if d is None or a is None or league_avg <= 0:
                        return None
                    factor = self.gc_home_factor if venue == 'H' else self.gc_away_factor
                    return max(0.0, float(d) * float(a) / float(league_avg) * float(factor))
                if fixture_map and atk_idx and def_idx:
                    fa_series = df['fbref_team_id'].map(calc_fixture_gc)
                    # Where fixture-aware value exists, use it; otherwise fallback to team model
                    gc_player = fa_series.fillna(gc_player)
            except Exception:
                # Fallback silently to team model if anything goes wrong
                pass

        preds['goals_conceded'] = gc_player.fillna(0).values

        # Build players output
        DEFAULT_PRICE_BY_POS = {'GK': 4.5, 'DEF': 4.5, 'MID': 6.5, 'FWD': 7.0}
        GOAL_POINTS = {'GK': 6.0, 'DEF': 6.0, 'MID': 5.0, 'FWD': 4.0}
        CS_POINTS = {'GK': 4.0, 'DEF': 4.0, 'MID': 1.0, 'FWD': 0.0}
        ASSIST_POINTS = 3.0

        def p_zero_goals(lambda_gc: float) -> float:
            try:
                if lambda_gc is None or lambda_gc < 0:
                    return 0.0
                return math.exp(-float(lambda_gc))
            except Exception:
                return 0.0

        players = []
        for i, row in df.iterrows():
            exp_minutes = float(preds['minutes'][i]) if preds['minutes'] is not None else 90.0
            exp_xg = float(preds['xg'][i]) if preds['xg'] is not None else 0.0
            exp_xa = float(preds['xa'][i]) if preds['xa'] is not None else 0.0
            exp_saves = float(preds['saves'][i]) if preds['saves'] is not None else 0.0
            exp_gc = float(preds['goals_conceded'][i]) if preds['goals_conceded'] is not None else 0.0

            pos_val = row.get(pos_col, None)
            norm_pos = self._normalize_position(pos_val)

            # Minutes points (continuous approximation: up to 2 points at 60+ mins)
            minutes_pts = max(0.0, min(2.0, exp_minutes / 60.0))

            # Attacking points (position-aware goal points)
            g_pts = GOAL_POINTS.get(norm_pos, 4.0)
            att_pts = exp_xg * g_pts + exp_xa * ASSIST_POINTS

            # Clean sheet points via Poisson(lambda = exp_gc)
            cs_w = CS_POINTS.get(norm_pos, 0.0)
            cs_pts = p_zero_goals(exp_gc) * cs_w

            # Saves points for GK
            saves_pts = (exp_saves / 3.0) if norm_pos == 'GK' else 0.0

            # Goals conceded penalty for GK/DEF: -1 per 2 conceded (approx -0.5 * E[GC])
            gc_penalty = (-0.5 * exp_gc) if norm_pos in {'GK', 'DEF'} else 0.0

            expected_points = minutes_pts + att_pts + cs_pts + saves_pts + gc_penalty

            price = DEFAULT_PRICE_BY_POS.get(norm_pos, 6.0)
            variance = max(1.0, expected_points * 0.3)
            ceiling = expected_points + 1.645 * math.sqrt(variance)
            floor = max(0.0, expected_points - 1.645 * math.sqrt(variance))
            code_val = row.get(pid_col, None)

            players.append({
                'player_id': code_val,
                'code': code_val,  # Use FBRef id as a stand-in code for compatibility
                'name': row.get(name_col, str(row.get(pid_col, 'unknown'))),
                'team': row.get(team_col, None),
                'position': norm_pos,
                'expected_minutes': exp_minutes,
                'expected_goals': exp_xg,
                'expected_assists': exp_xa,
                'expected_saves': exp_saves,
                'expected_goals_conceded': exp_gc,
                'expected_bonus': 0.0,
                'expected_points': expected_points,
                'current_price': price,
                'points_per_million': expected_points / price if price else 0.0,
                'ceiling': round(ceiling, 2),
                'floor': round(floor, 2),
            })

        # Sort players by expected points desc for downstream CSV/top-N selection
        players.sort(key=lambda p: p['expected_points'], reverse=True)

        # Minimal summary compatible with downstream usage
        if players:
            avg_xp = sum(p['expected_points'] for p in players) / len(players)
            summary = {
                'total_players': len(players),
                'average_expected_points': round(avg_xp, 2),
                'max_expected_points': round(players[0]['expected_points'], 2),
            }
        else:
            summary = {'total_players': 0, 'average_expected_points': 0.0, 'max_expected_points': 0.0}

        return { 'players': players, 'summary': summary, 'gameweek': gameweek }
