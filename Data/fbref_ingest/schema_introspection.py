#!/usr/bin/env python3
"""
FBRef Schema Introspection

Utilities to inspect the SQLite schema to assist building canonical queries.
"""
import argparse
import sqlite3
from pathlib import Path
from typing import List


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [r[0] for r in cur.fetchall()]


def describe_table(conn: sqlite3.Connection, table: str) -> None:
    print(f"\n== {table} ==")
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = cur.fetchall()
    for cid, name, ctype, notnull, dflt, pk in cols:
        print(f"- {name} ({ctype}) notnull={notnull} pk={pk} default={dflt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True, help='Path to FBRef SQLite db')
    parser.add_argument('--table', help='Optional table to describe')
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        if args.table:
            describe_table(conn, args.table)
        else:
            tables = list_tables(conn)
            print("Tables:")
            for t in tables:
                print(f"- {t}")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
