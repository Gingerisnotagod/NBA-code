"""
Fit on all history, predict unplayed games.

Uses features.row_for() — the same function that built the training rows —
so inference cannot drift from training.
"""

import numpy as np
import pandas as pd

import features
import models
from config import FEATURES


def fit_final(feats, kind="logistic"):
    """Fit on every game available. Use only after backtesting, never to score."""
    model = models.all_models()[kind]
    model.fit(feats[FEATURES], feats.home_win)
    return model


def roll_into_new_season(state):
    """
    Advance league state across the offseason.

    features.build() only applies the season rollover when it SEES a game
    from a new season in the data. If your history ends in June and you're
    predicting an October opener, the state is still sitting at end-of-last-
    season: Elo un-regressed, and form / net-rating / rest carrying values
    from games five months old. Call this once before predicting a new
    season, or every early-season prediction is quietly wrong.

    Mutates and returns state.
    """
    state.new_season()
    return state


def predict(model, state, matchups, game_date):
    """
    model     — from fit_final()
    state     — LeagueState from features.build()
    matchups  — list of (home_team, away_team)
    game_date — date of the games, for rest-day computation

    Team names must match the strings in your training data exactly.
    Unknown names raise rather than silently defaulting to a 1500 rating.
    """
    game_date = pd.to_datetime(game_date)
    known = set(state.ratings)

    unknown = {t for pair in matchups for t in pair if t not in known}
    if unknown:
        raise ValueError(f"unknown team names: {sorted(unknown)}")

    rows = []
    for home, away in matchups:
        row = {"home_team": home, "away_team": away}
        row.update(features.row_for(state, home, away, game_date))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["home_win_prob"] = model.predict_proba(out[FEATURES])[:, 1]
    out["pick"] = np.where(out.home_win_prob >= 0.5, out.home_team, out.away_team)
    out["confidence"] = np.maximum(out.home_win_prob, 1 - out.home_win_prob)

    return out[["home_team", "away_team", "home_win_prob", "pick", "confidence"]]


def predict_slate(model, state, slate):
    """
    slate — DataFrame with game_date, home_team, away_team (from schedule.py)

    Games are grouped by date so rest days are computed correctly for each.
    """
    if slate.empty:
        return pd.DataFrame(columns=["game_date", "home_team", "away_team",
                                     "home_win_prob", "pick", "confidence"])

    out = []
    for date, day_games in slate.groupby("game_date"):
        matchups = list(zip(day_games.home_team, day_games.away_team))
        preds = predict(model, state, matchups, date)
        preds.insert(0, "game_date", date)
        out.append(preds)

    return pd.concat(out, ignore_index=True).sort_values(
        ["game_date", "confidence"], ascending=[True, False]
    ).reset_index(drop=True)


def advance_state(state, results):
    """
    Fold completed games back in so tomorrow's predictions use today's results.

    results — DataFrame with game_date, home_team, away_team, home_pts, away_pts

    Run this nightly during the season. Without it, your Elo freezes on the
    day you last pulled data and drifts further from reality all year.
    """
    for game in results.sort_values("game_date").itertuples(index=False):
        state.record_game(
            game.home_team, game.away_team,
            game.home_pts, game.away_pts,
            pd.to_datetime(game.game_date),
        )
    return state


def power_rankings(state, top=10):
    """Current Elo ratings, best first. Sanity check on the pipeline."""
    ranked = sorted(state.ratings.items(), key=lambda kv: -kv[1])[:top]
    return pd.DataFrame(ranked, columns=["team", "elo"])