"""
Game loaders. Each returns a normalized DataFrame with exactly:

    game_date | season | home_team | away_team | home_pts | away_pts | home_win

one row per completed game, sorted chronologically.
"""

import sqlite3
import sys

import pandas as pd

from config import REQUIRED_COLS


def from_sqlite(db_path, table="game"):
    """
    Kaggle-style NBA SQLite dump.

    Column names differ between dumps. Check yours first:
        sqlite3 nba.sqlite ".schema game"
    and adjust the aliases below. This targets the wyattowalsh schema.
    """
    con = sqlite3.connect(db_path)
    try:
        query = f"""
            SELECT game_date      AS game_date,
                   season_id      AS season,
                   team_name_home AS home_team,
                   team_name_away AS away_team,
                   pts_home       AS home_pts,
                   pts_away       AS away_pts
            FROM {table}
            WHERE pts_home IS NOT NULL AND pts_away IS NOT NULL
        """
        df = pd.read_sql(query, con)
    finally:
        con.close()
    return normalize(df)


def from_balldontlie(start_season=2023, end_season=2027):
    """
    Completed games from balldontlie, seasons [start, end).
    Cached per season in bdl_cache/; see bdl.py.
    """
    import bdl

    raw = bdl.games(range(start_season, end_season))
    return normalize(raw[raw.status_state == "final"])


def from_sqlite_plus_bdl(db_path, start_season=2023, end_season=2027):
    """
    Kaggle history + balldontlie for everything after it.

    The Kaggle dump ends June 2023, so the default pulls 2023-24 onward.
    balldontlie team names are mapped onto the Kaggle spellings so each
    franchise keeps one continuous Elo rating.
    """
    from schedule import align_team_names

    old = from_sqlite(db_path)
    new = from_balldontlie(start_season, end_season)
    new = new[new.game_date > old.game_date.max()]

    # match against the most recent season's names only, so nickname
    # fallback can't land on a defunct franchise (e.g. New Orleans Hornets)
    recent = old[old.season == old.season.max()]
    known = set(recent.home_team) | set(recent.away_team)
    new, unresolved = align_team_names(new, known)
    if unresolved:
        print(f"warning: balldontlie teams not in DB, dropped: {unresolved}", file=sys.stderr)

    print(f"sqlite through {old.game_date.max().date()}, "
          f"+{len(new):,} games from balldontlie", file=sys.stderr)
    return normalize(pd.concat([old, new], ignore_index=True))


def from_nba_api(start_season=2015, end_season=2027):
    """
    Live pull from stats.nba.com. Requires: pip install nba_api

    The endpoint returns one row per team per game, so this pivots to
    one row per game by splitting on the '@' in the MATCHUP field.
    """
    from nba_api.stats.endpoints import leaguegamefinder

    frames = []
    for year in range(start_season, end_season):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        )
        raw = finder.get_data_frames()[0]
        raw["season"] = year
        frames.append(raw)

    logs = pd.concat(frames, ignore_index=True)
    logs["is_home"] = ~logs["MATCHUP"].str.contains("@")

    home = logs[logs.is_home][["GAME_ID", "GAME_DATE", "season", "TEAM_NAME", "PTS"]]
    away = logs[~logs.is_home][["GAME_ID", "TEAM_NAME", "PTS"]]

    df = home.merge(away, on="GAME_ID", suffixes=("_home", "_away")).rename(
        columns={
            "GAME_DATE": "game_date",
            "TEAM_NAME_home": "home_team",
            "TEAM_NAME_away": "away_team",
            "PTS_home": "home_pts",
            "PTS_away": "away_pts",
        }
    )
    return normalize(df)


def from_csv(path):
    return normalize(pd.read_csv(path))


def normalize(df):
    """Validate, coerce types, sort, and derive the target column."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"loader produced no {missing}; got {list(df.columns)}")

    df = df[REQUIRED_COLS].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Kaggle season_id looks like '22015' — the trailing 4 digits are the year.
    # balldontlie gives a plain 2015; str[-4:] handles both.
    df["season"] = df["season"].astype(str).str[-4:].astype(int)

    df = df.dropna()
    df = df[df.home_pts != df.away_pts]          # no ties in basketball; drop bad rows

    # Kaggle dumps are known to contain duplicate games. Left in, they
    # double-count Elo updates for whoever is duplicated.
    df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"])

    df = df.sort_values("game_date").reset_index(drop=True)
    df["home_win"] = (df.home_pts > df.away_pts).astype(int)
    return df


def load(source, db=None, csv=None, seasons=(2023, 2027)):
    """Dispatch used by main.py."""
    if source == "sqlite":
        return from_sqlite(db)
    if source == "sqlite+bdl":
        return from_sqlite_plus_bdl(db, *seasons)
    if source == "bdl":
        return from_balldontlie(*seasons)
    if source == "nba_api":
        return from_nba_api(*seasons)
    if source == "csv":
        return from_csv(csv)
    raise ValueError(f"unknown source: {source}")