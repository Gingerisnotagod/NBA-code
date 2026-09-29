"""
Walk-forward validation.

Train on every season before S, test on S, advance. This is the only honest
way to score a time series model. Do not swap in k-fold cross-validation:
shuffled folds put future games in the training set and will hand you an
accuracy number several points too high.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import models
from config import FEATURES, MIN_TEST_GAMES, MIN_TRAIN_SEASONS


def walk_forward(feats, min_train_seasons=MIN_TRAIN_SEASONS):
    """
    returns: (per_season DataFrame, pooled metrics DataFrame)

    Two baselines are scored alongside the models:
      home_always — pick the home team every time
      elo_only    — pick whoever Elo favors, no ML at all

    If the models don't clear elo_only by a real margin, the ML layer is
    just adding variance and you should ship the Elo number.
    """
    seasons = sorted(feats.season.unique())
    collected = defaultdict(list)
    per_season = []

    for i, season in enumerate(seasons):
        if i < min_train_seasons:
            continue

        train = feats[feats.season < season]
        test = feats[feats.season == season]
        if len(test) < MIN_TEST_GAMES:
            continue

        X_train, y_train = train[FEATURES], train.home_win
        X_test, y_test = test[FEATURES], test.home_win

        row = {"season": season, "n": len(test)}
        row["home_always"] = y_test.mean()

        elo_prob = test.elo_home_prob.values
        row["elo_only"] = accuracy_score(y_test, (elo_prob >= 0.5).astype(int))
        collected["elo_only"].append((y_test.values, elo_prob))

        for name, model in models.all_models().items():
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            row[name] = accuracy_score(y_test, (prob >= 0.5).astype(int))
            collected[name].append((y_test.values, prob))

        per_season.append(row)

    if not per_season:
        raise ValueError(
            f"no testable seasons: need > {min_train_seasons} seasons of data"
        )

    pooled = {}
    for name, chunks in collected.items():
        y = np.concatenate([c[0] for c in chunks])
        p = np.concatenate([c[1] for c in chunks])
        pooled[name] = models.evaluate(y, p)

    return pd.DataFrame(per_season), pd.DataFrame(pooled).T