"""
balldontlie client. Free tier gets /games only, at 5 requests/min.

Set your key once (PowerShell), then open a NEW terminal:
    setx BDL_API_KEY "your-key-here"

Every season is cached to bdl_cache/games_<season>.csv. Finished seasons are
never re-downloaded; the in-progress season only re-pulls from its earliest
unfinished game onward, so a nightly run is usually 1-2 requests.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://api.balldontlie.io/v1"
MIN_INTERVAL = 12.5            # seconds between calls -> under 5/min on free tier
CACHE_DIR = Path("bdl_cache")

# status_state values that will never change again
DONE_STATES = {"final", "postponed", "canceled", "abandoned"}

# map onto schedule.py's codes: 1 = not started, 2 = live, 3 = final
STATUS_CODE = {"scheduled": 1, "delayed": 1, "in_progress": 2, "suspended": 2, "final": 3}

COLUMNS = ["game_id", "game_date", "season", "home_team", "away_team",
           "home_pts", "away_pts", "status_state", "postseason"]

_last_call = 0.0
_session = {}                  # season -> DataFrame, so one run never fetches a season twice


def _key():
    key = os.environ.get("BDL_API_KEY")
    if not key:
        raise RuntimeError(
            'BDL_API_KEY not set. In PowerShell: setx BDL_API_KEY "your-key", '
            "then open a new terminal."
        )
    return key


def _get(path, params, retries=4):
    global _last_call
    url = f"{BASE_URL}/{path}?{urllib.parse.urlencode(params, doseq=True)}"
    req = urllib.request.Request(url, headers={
        "Authorization": _key(),
        "Accept": "application/json",
        "User-Agent": "nba-predictor/1.0",
    })

    for attempt in range(retries):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError("balldontlie 401: bad API key, or endpoint not on your tier") from exc
            if exc.code == 429 or exc.code >= 500:
                backoff = 60 * (attempt + 1) if exc.code == 429 else 10 * (attempt + 1)
                print(f"  balldontlie {exc.code}, retrying in {backoff}s", file=sys.stderr)
                time.sleep(backoff)
                continue
            raise
    raise RuntimeError(f"balldontlie: gave up after {retries} tries: {url}")


def _paginate(path, params):
    params = {**params, "per_page": 100}
    rows = []
    while True:
        payload = _get(path, params)
        rows.extend(payload["data"])
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            return rows
        params["cursor"] = cursor


def _flatten(raw):
    rows = [
        {
            "game_id": g["id"],
            "game_date": g["date"][:10],
            "season": g["season"],
            "home_team": g["home_team"]["full_name"],
            "away_team": g["visitor_team"]["full_name"],
            "home_pts": g.get("home_team_score"),
            "away_pts": g.get("visitor_team_score"),
            "status_state": g.get("status_state")
                            or ("final" if g.get("status") == "Final" else "scheduled"),
            "postseason": bool(g.get("postseason", False)),
        }
        for g in raw
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["game_date"] = pd.to_datetime(df.game_date)
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def season_games(season, refresh=False):
    """
    Every game in a season (played and unplayed), preseason excluded.
    season = starting year: 2025 means 2025-26 (same convention as the Kaggle DB).
    """
    if season in _session and not refresh:
        return _session[season]

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"games_{season}.csv"
    label = f"{season}-{str(season + 1)[-2:]}"

    if path.exists() and not refresh:
        cached = pd.read_csv(path, parse_dates=["game_date"])
        pending = cached[~cached.status_state.isin(DONE_STATES)]
        if len(cached) and pending.empty:
            _session[season] = cached
            return cached
        since = pending.game_date.min() if len(pending) else None
    else:
        cached, since = None, None

    params = {"seasons[]": [season]}
    if since is not None:
        params["start_date"] = since.date().isoformat()
        print(f"  balldontlie: updating {label} from {params['start_date']}", file=sys.stderr)
    else:
        print(f"  balldontlie: pulling {label} (~3 min on free tier)", file=sys.stderr)

    fresh = _flatten(_paginate("games", params))
    if since is not None:
        df = pd.concat([cached[cached.game_date < since], fresh], ignore_index=True)
        df = df.drop_duplicates("game_id", keep="last")
    else:
        df = fresh

    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    _session[season] = df
    return df


def games(seasons, refresh=False):
    """Concatenate season_games over an iterable of season start years."""
    frames = [season_games(s, refresh=refresh) for s in seasons]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)