#!/usr/bin/env python3
"""
FBRef Canonical Queries

Provide parameterized SQL to extract canonical tables from the FBRef SQLite DB.
Aligned to schema in Data/fbref_ingest/schema/master.db.sql
"""
import argparse
import sqlite3
from pathlib import Path


def run_query(conn: sqlite3.Connection, sql: str, out_path: Path):
    import csv
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    out_dir = Path(args.out)

    # Players derived from Player_Info
    players_sql = r'''
        WITH latest AS (
            SELECT
                player_id,
                MAX(minutes) AS max_minutes
            FROM Player_Info
            GROUP BY player_id
        )
        SELECT
            pi.player_id AS fbref_player_id,
            pi.name AS name,
            pi.nation AS nationality,
            pi.position AS position
        FROM Player_Info pi
        JOIN latest l ON l.player_id = pi.player_id AND l.max_minutes = pi.minutes
        GROUP BY pi.player_id
    '''

    # Teams derived from Match (use team name as id for now)
    teams_sql = r'''
        SELECT DISTINCT home_team AS fbref_team_id, home_team AS name FROM "Match"
        UNION
        SELECT DISTINCT away_team AS fbref_team_id, away_team AS name FROM "Match"
    '''

    # Matches from Match table (team ids are team names here)
    matches_sql = r'''
        SELECT
            match_id,
            season,
            date AS match_date,
            home_team AS home_team_id,
            away_team AS away_team_id,
            home_goals,
            away_goals,
            home_xG AS home_xg,
            away_xG AS away_xg
        FROM "Match"
    '''

    # Player game stats from Player_Info + Summary + Passing + Goalkeeper + Match
    player_game_stats_sql = r'''
        SELECT
            pi.match_id,
            m.season,
            m.date AS match_date,
            pi.player_id AS fbref_player_id,
            CASE WHEN pi.home_away = 'H' THEN m.home_team ELSE m.away_team END AS fbref_team_id,
            pi.position AS position,
            COALESCE(pi.minutes, 0) AS minutes,
            COALESCE(s.shots, 0) AS shots,
            COALESCE(pa.key_passes, 0) AS key_passes,
            COALESCE(s.xG, 0.0) AS xg,
            COALESCE(s.xA, COALESCE(pa.xA, 0.0)) AS xa,
            COALESCE(s.goals, 0) AS goals,
            COALESCE(s.assists, COALESCE(pa.assists, 0)) AS assists,
            COALESCE(s.yellow_cards, 0) AS yellow_cards,
            COALESCE(pi.start, 0) AS starts,
            COALESCE(gk.saves, 0) AS saves
        FROM Player_Info pi
        JOIN "Match" m ON m.match_id = pi.match_id
        LEFT JOIN Summary s ON s.match_id = pi.match_id AND s.player_id = pi.player_id
        LEFT JOIN Passing pa ON pa.match_id = pi.match_id AND pa.player_id = pi.player_id
        LEFT JOIN Goalkeeper gk ON gk.match_id = pi.match_id AND gk.player_id = pi.player_id
    '''

    # Team game stats: shots_allowed from opponent Summary; xg_allowed and goals_conceded from Match
    team_game_stats_sql = r'''
        WITH home_shots AS (
            SELECT s.match_id, SUM(COALESCE(s.shots,0)) AS shots
            FROM Summary s
            JOIN Player_Info pi ON pi.match_id = s.match_id AND pi.player_id = s.player_id
            WHERE pi.home_away = 'H'
            GROUP BY s.match_id
        ),
        away_shots AS (
            SELECT s.match_id, SUM(COALESCE(s.shots,0)) AS shots
            FROM Summary s
            JOIN Player_Info pi ON pi.match_id = s.match_id AND pi.player_id = s.player_id
            WHERE pi.home_away = 'A'
            GROUP BY s.match_id
        )
        SELECT
            m.match_id,
            m.season,
            m.date AS match_date,
            m.home_team AS fbref_team_id,
            COALESCE(away_shots.shots, 0) AS shots_allowed,
            COALESCE(m.away_xG, 0.0) AS xg_allowed,
            m.away_goals AS goals_conceded
        FROM "Match" m
        LEFT JOIN away_shots ON away_shots.match_id = m.match_id
        UNION ALL
        SELECT
            m.match_id,
            m.season,
            m.date AS match_date,
            m.away_team AS fbref_team_id,
            COALESCE(home_shots.shots, 0) AS shots_allowed,
            COALESCE(m.home_xG, 0.0) AS xg_allowed,
            m.home_goals AS goals_conceded
        FROM "Match" m
        LEFT JOIN home_shots ON home_shots.match_id = m.match_id
    '''

    queries = {
        'players': players_sql,
        'teams': teams_sql,
        'matches': matches_sql,
        'player_game_stats': player_game_stats_sql,
        'team_game_stats': team_game_stats_sql,
    }

    for name, sql in queries.items():
        run_query(conn, sql, out_dir / f"{name}.csv")

    conn.close()


if __name__ == '__main__':
    main()
