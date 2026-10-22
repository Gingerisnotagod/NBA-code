"""
Upcoming schedule fetchers.

Two sources, same output shape (game_date | game_id | home_team | away_team | status):

  bdl  — balldontlie /games, which includes unplayed games. Default, because
         it works on the school network and shares bdl.py's cache.
  cdn  — NBA's static CDN JSON. No key needed, but untested on your network.

status: 1 = not started, 2 = live, 3 = final
"""

import datetime as dt
import json
import urllib.request

import pandas as pd

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"

# The CDN feed rejects a bare urllib user-agent.
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
}


def current_season(today=None):
    """Season start year. Aug onward counts as the upcoming season."""
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def fetch_season_schedule_bdl(season=None):
    """Full season from balldontlie, played and unplayed."""
    import bdl

    season = current_season() if season is None else season
    games = bdl.season_games(season)
    if games.empty:
        raise RuntimeError(f"balldontlie has no games yet for the {season} season")

    out = games[["game_date", "game_id", "home_team", "away_team"]].copy()
    out["status"] = games.status_state.map(bdl.STATUS_CODE).fillna(0).astype(int)

    # tip-off in US Eastern; older cache files have no tip_utc column -> NaT
    tip = games["tip_utc"] if "tip_utc" in games else pd.Series(pd.NA, index=games.index)
    out["tip"] = (pd.to_datetime(tip, utc=True, errors="coerce")
                    .dt.tz_convert("America/New_York").dt.tz_localize(None))
    return out.sort_values(["game_date", "tip"]).reset_index(drop=True)


def fetch_season_schedule(url=SCHEDULE_URL, timeout=30):
    """
    NBA CDN feed. Full league schedule, played and unplayed.

    returns DataFrame: game_date | game_id | home_team | away_team | status
    """
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    rows = []
    for game_date in payload["leagueSchedule"]["gameDates"]:
        for game in game_date["games"]:
            home = game["homeTeam"]
            away = game["awayTeam"]
            rows.append(
                {
                    "game_date": pd.to_datetime(game["gameDateEst"]).tz_localize(None),
                    "game_id": game["gameId"],
                    "home_team": f"{home['teamCity']} {home['teamName']}".strip(),
                    "away_team": f"{away['teamCity']} {away['teamName']}".strip(),
                    "status": game["gameStatus"],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("schedule feed returned no games")
    return df.sort_values("game_date").reset_index(drop=True)


# 1 = not started, 2 = live. Live games are included because the model only
# uses pregame information, so its pick is identical to a pregame pick, and
# balldontlie sometimes marks a game "in progress" well before tip-off.
OPEN_STATUSES = (1, 2)


def upcoming(schedule, start=None, days=1):
    """
    Unplayed (or in-progress) games in a date window.

    start: date to begin from (default: today)
    days:  window length; 1 = just that day, 7 = the week ahead
    """
    start = pd.to_datetime(start or dt.date.today()).normalize()
    end = start + pd.Timedelta(days=days)

    window = schedule[
        (schedule.game_date >= start)
        & (schedule.game_date < end)
        & (schedule.status.isin(OPEN_STATUSES))
    ]
    return window.reset_index(drop=True)


def next_slate(schedule, after=None):
    """The next date that has any unplayed games. Useful in the preseason."""
    after = pd.to_datetime(after or dt.date.today()).normalize()
    future = schedule[(schedule.game_date >= after) & (schedule.status.isin(OPEN_STATUSES))]
    if future.empty:
        return future
    first_date = future.game_date.min()
    return future[future.game_date == first_date].reset_index(drop=True)


def align_team_names(schedule, known_names):
    """
    The schedule feed and your historical DB may spell teams differently
    ('LA Clippers' vs 'Los Angeles Clippers'). Map schedule names onto the
    names your model was trained on, matching on the last word (the nickname,
    which is stable) when the full string doesn't match.

    Pass only CURRENT team names as known_names, or the nickname fallback can
    pick a defunct franchise (Charlotte vs New Orleans Hornets).

    Returns (aligned_schedule, unresolved_names).
    """
    known = set(known_names)
    by_nickname = {}
    for name in known:
        by_nickname.setdefault(name.split()[-1].lower(), name)

    def resolve(name):
        if name in known:
            return name
        return by_nickname.get(name.split()[-1].lower())

    out = schedule.copy()
    out["home_team"] = out.home_team.map(resolve)
    out["away_team"] = out.away_team.map(resolve)

    bad = out[out.home_team.isna() | out.away_team.isna()]
    unresolved = sorted(
        (set(schedule.loc[bad.index, "home_team"]) | set(schedule.loc[bad.index, "away_team"]))
        - known
    )
    return out.dropna(subset=["home_team", "away_team"]).reset_index(drop=True), unresolved