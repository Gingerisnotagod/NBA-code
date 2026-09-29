"""
Pick tracking.

Every --upcoming run appends its predictions to picks_log.csv. --score
compares those picks with actual results and reports how the model did.

If a game was predicted more than once (e.g. you ran it on Monday and again
on game day), only the LATEST prediction before tip-off counts, since that's
the one with the most up-to-date results and injury info.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

LOG_PATH = "picks_log.csv"
LOG_COLS = ["predicted_at", "game_date", "home_team", "away_team",
            "home_win_prob", "pick", "confidence", "out"]


def save_picks(preds, path=LOG_PATH):
    """Append a run's predictions to the log. Returns number of rows saved."""
    if preds.empty:
        return 0
    rows = preds.copy()
    rows["predicted_at"] = dt.datetime.now().isoformat(timespec="seconds")
    rows["game_date"] = pd.to_datetime(rows.game_date).dt.date.astype(str)
    if "out" not in rows:
        rows["out"] = ""
    rows = rows[LOG_COLS]

    path = Path(path)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(rows)


def score(games, path=LOG_PATH):
    """
    games: normalized history DataFrame (has game_date, home_team, away_team, home_win)
    Prints a report. Returns the graded picks DataFrame.
    """
    path = Path(path)
    if not path.exists():
        print(f"no {path} yet; run --upcoming first to start logging picks")
        return pd.DataFrame()

    log = pd.read_csv(path, dtype={"out": str})
    log["game_date"] = pd.to_datetime(log.game_date)
    log["predicted_at"] = pd.to_datetime(log.predicted_at)

    # a prediction only counts if it was made before the game's date ended
    log = log[log.predicted_at < log.game_date + pd.Timedelta(days=1)]
    latest = (log.sort_values("predicted_at")
                 .drop_duplicates(["game_date", "home_team", "away_team"], keep="last"))

    results = games[["game_date", "home_team", "away_team", "home_win"]].copy()
    results["game_date"] = pd.to_datetime(results.game_date).dt.normalize()
    graded = latest.merge(results, on=["game_date", "home_team", "away_team"], how="inner")

    pending = len(latest) - len(graded)
    if graded.empty:
        print(f"no finished games to grade yet ({pending} picks waiting on results)")
        return graded

    graded["winner"] = np.where(graded.home_win == 1, graded.home_team, graded.away_team)
    graded["correct"] = (graded.pick == graded.winner).astype(int)
    p = graded.home_win_prob.clip(1e-6, 1 - 1e-6)
    graded["brier"] = (p - graded.home_win) ** 2

    n, acc = len(graded), graded.correct.mean()
    print(f"\ngraded {n} picks ({pending} still pending)")
    print(f"  accuracy  {acc:.1%}   ({graded.correct.sum()}/{n})")
    print(f"  brier     {graded.brier.mean():.4f}   (0.25 = coin flip; lower is better)")
    print(f"  home-team baseline  {graded.home_win.mean():.1%}")

    # by week
    graded["week"] = graded.game_date.dt.to_period("W-SUN").dt.start_time.dt.date
    weekly = graded.groupby("week").agg(games=("correct", "size"), accuracy=("correct", "mean"))
    print("\nby week")
    print(weekly.to_string(formatters={"accuracy": "{:.1%}".format}))

    # calibration: do 70% picks win ~70% of the time?
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0001]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-80%", "80%+"]
    graded["bucket"] = pd.cut(graded.confidence, bins=bins, labels=labels, right=False)
    cal = graded.groupby("bucket", observed=True).agg(
        games=("correct", "size"), predicted=("confidence", "mean"), actual=("correct", "mean"))
    print("\nby confidence (predicted vs actual win rate for the pick)")
    print(cal.to_string(formatters={"predicted": "{:.1%}".format, "actual": "{:.1%}".format}))

    # picks where an injury adjustment was applied
    inj = graded[graded.out.fillna("").str.len() > 0]
    if len(inj):
        print(f"\ngames with injury adjustments: {inj.correct.mean():.1%} correct ({len(inj)} games)")

    misses = graded[graded.correct == 0].sort_values("confidence", ascending=False).head(5)
    if len(misses):
        print("\nbiggest misses")
        print(misses[["game_date", "home_team", "away_team", "pick", "confidence", "winner"]]
              .assign(game_date=lambda d: d.game_date.dt.date)
              .to_string(index=False, formatters={"confidence": "{:.1%}".format}))
    return graded