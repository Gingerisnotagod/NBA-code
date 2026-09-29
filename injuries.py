"""
Manual injury / rest adjustments.

Edit injuries.csv before running predictions. One row per player who is OUT:

    team,player,penalty,date
    76ers,Joel Embiid,top15,
    76ers,Tyrese Maxey,top30,2026-10-22
    Celtics,Jaylen Brown,80,

team     full name or just the nickname ("76ers", "Celtics")
player   for your own reference; shown in the output
penalty  a rank tier, or any number of Elo points (~28 Elo = 1 point of spread):
             top5 150 | top15 125 | top30 95 | top60 60 | top120 30 | bench 10
date     blank = every game until you delete the row (long-term injury)
         YYYY-MM-DD = that date only (rest night, one-game absence)

Lines starting with # are ignored. The penalty applies to that prediction
only; it never changes the team's stored Elo.
"""

import sys
from pathlib import Path

import pandas as pd

TIERS = {
    # league-wide rank tiers
    "top5": 150,     # MVP candidates
    "top15": 125,    # All-NBA level
    "top30": 95,     # All-Star level
    "top60": 60,     # very good starter
    "top120": 30,    # average starter / key rotation
    "bench": 10,     # 121+
    # older shorthand, still accepted
    "mvp": 150, "allstar": 80, "all-star": 80, "starter": 35,
}

TEMPLATE = """\
# team,player,penalty,date   -- see injuries.py for the format
# penalty: top5=150 top15=125 top30=95 top60=60 top120=30 bench=10, or any number
# date: blank = every game until removed; YYYY-MM-DD = that game only
team,player,penalty,date
"""


def load(path="injuries.csv", known_teams=None):
    """
    Read the injury file. Creates an empty template if it doesn't exist.
    known_teams: team names the model knows; used to resolve nicknames
    and catch typos. Returns a DataFrame (possibly empty).
    """
    path = Path(path)
    if not path.exists():
        path.write_text(TEMPLATE)
        print(f"created empty {path}; add injured players there", file=sys.stderr)

    df = pd.read_csv(path, comment="#", dtype=str, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"team", "player", "penalty"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if "date" not in df.columns:
        df["date"] = None

    df = df.dropna(subset=["team"]).copy()
    if df.empty:
        return df

    for col in ("team", "player", "penalty"):
        df[col] = df[col].fillna("").str.strip()

    df["penalty"] = df.penalty.map(_parse_penalty)
    df["date"] = pd.to_datetime(df["date"].where(df["date"].notna() & (df["date"].str.strip() != "")),
                                errors="coerce").dt.normalize()

    if known_teams is not None:
        df["team"], bad = _resolve_teams(df.team, known_teams)
        if bad:
            print(f"warning: injuries.csv teams not recognized, ignored: {bad}", file=sys.stderr)
        df = df.dropna(subset=["team"])

    return df.reset_index(drop=True)


def penalty(injuries, team, game_date):
    """Total Elo penalty for a team on a date, plus a label of who's out."""
    if injuries is None or injuries.empty:
        return 0.0, ""
    game_date = pd.to_datetime(game_date).normalize()
    rows = injuries[(injuries.team == team)
                    & (injuries.date.isna() | (injuries.date == game_date))]
    if rows.empty:
        return 0.0, ""
    label = ", ".join(f"{p or '?'} (-{v:g})" for p, v in zip(rows.player, rows.penalty))
    return float(rows.penalty.sum()), label


def _parse_penalty(value):
    v = str(value).strip().lower()
    if v in TIERS:
        return float(TIERS[v])
    try:
        return float(v)
    except ValueError:
        raise ValueError(
            f"injuries.csv: bad penalty '{value}'. Use a number or one of {sorted(TIERS)}"
        ) from None


def _resolve_teams(names, known_teams):
    """Map 'Celtics' / 'celtics' / 'Boston Celtics' onto the model's team names."""
    known = list(known_teams)
    by_full = {k.lower(): k for k in known}
    by_nick = {}
    for k in known:
        by_nick.setdefault(k.split()[-1].lower(), k)

    def resolve(name):
        n = name.strip().lower()
        return by_full.get(n) or by_nick.get(n.split()[-1] if n else "")

    resolved = names.map(resolve)
    bad = sorted(set(names[resolved.isna()]))
    return resolved, bad