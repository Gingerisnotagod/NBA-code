"""
Feature construction.

The whole no-leakage guarantee lives in this file. build() walks games in
chronological order and, for each one:

    1. READS the current league state into a feature row
    2. WRITES that game's result into the league state

Never the other way round. If you add a feature, add it in step 1 and update
it in step 2 — do not compute it with a groupby over the full DataFrame, which
is how future information gets into training data.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import elo
from config import (
    DEFAULT_REST_DAYS,
    ELO_HOME_ADV,
    ELO_START,
    MAX_REST_DAYS,
    ROLL_WINDOW,
)


@dataclass
class LeagueState:
    """
    Mutable league state as of a point in time.

    After build() returns, this holds the state as of the last game in the
    data — which is exactly what predict.py needs for unplayed games.
    """

    ratings: dict = field(default_factory=lambda: defaultdict(lambda: ELO_START))
    margins: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=ROLL_WINDOW)))
    opp_ratings: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=ROLL_WINDOW)))
    season_pts: dict = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    last_played: dict = field(default_factory=dict)

    def new_season(self):
        """Regress ratings toward the mean; drop everything season-scoped."""
        for team in list(self.ratings):
            self.ratings[team] = elo.regress_to_mean(self.ratings[team])
        self.margins.clear()
        self.opp_ratings.clear()
        self.season_pts.clear()
        self.last_played.clear()

    def form(self, team):
        """Mean point margin over the rolling window."""
        return float(np.mean(self.margins[team])) if self.margins[team] else 0.0

    def net_rating(self, team):
        """Season-to-date points for minus against, per game."""
        pf, pa, n = self.season_pts[team]
        return (pf - pa) / n if n else 0.0

    def strength_of_schedule(self, team):
        """Mean Elo of recent opponents."""
        return float(np.mean(self.opp_ratings[team])) if self.opp_ratings[team] else ELO_START

    def rest_days(self, team, game_date):
        if team not in self.last_played:
            return DEFAULT_REST_DAYS
        return min((game_date - self.last_played[team]).days, MAX_REST_DAYS)

    def record_game(self, home, away, home_pts, away_pts, game_date):
        margin = home_pts - away_pts
        home_elo, away_elo = self.ratings[home], self.ratings[away]

        # opponent strength must be logged BEFORE ratings move
        self.opp_ratings[home].append(away_elo)
        self.opp_ratings[away].append(home_elo)

        self.ratings[home], self.ratings[away] = elo.update(
            home_elo, away_elo, margin, ELO_HOME_ADV
        )

        self.margins[home].append(margin)
        self.margins[away].append(-margin)

        self.season_pts[home][0] += home_pts
        self.season_pts[home][1] += away_pts
        self.season_pts[home][2] += 1
        self.season_pts[away][0] += away_pts
        self.season_pts[away][1] += home_pts
        self.season_pts[away][2] += 1

        self.last_played[home] = game_date
        self.last_played[away] = game_date


def row_for(state, home, away, game_date):
    """
    Build one feature row from current state. Shared by build() and
    predict.py so that training and inference cannot drift apart.
    """
    home_elo = state.ratings[home]
    away_elo = state.ratings[away]
    rest_home = state.rest_days(home, game_date)
    rest_away = state.rest_days(away, game_date)

    return {
        "elo_diff": (home_elo + ELO_HOME_ADV) - away_elo,
        "elo_home_prob": elo.expected(home_elo + ELO_HOME_ADV, away_elo),
        "form_diff": state.form(home) - state.form(away),
        "netrtg_diff": state.net_rating(home) - state.net_rating(away),
        "rest_diff": rest_home - rest_away,
        "b2b_home": int(rest_home <= 1),
        "b2b_away": int(rest_away <= 1),
        "sos_diff": state.strength_of_schedule(home) - state.strength_of_schedule(away),
    }


def build(games):
    """
    games: normalized DataFrame from data.py
    returns: (features DataFrame, LeagueState as of the final game)
    """
    state = LeagueState()
    current_season = None
    rows = []

    for game in games.itertuples(index=False):
        if game.season != current_season:
            current_season = game.season
            state.new_season()

        home, away = game.home_team, game.away_team

        # 1. READ
        row = {
            "game_date": game.game_date,
            "season": game.season,
            "home_team": home,
            "away_team": away,
            **row_for(state, home, away, game.game_date),
            "home_win": game.home_win,
        }
        rows.append(row)

        # 2. WRITE
        state.record_game(home, away, game.home_pts, game.away_pts, game.game_date)

    return pd.DataFrame(rows), state