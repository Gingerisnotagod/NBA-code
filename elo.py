"""Elo rating math. Pure functions — no state, no I/O."""

from config import ELO_K, ELO_SEASON_CARRY, ELO_SEASON_MEAN


def expected(rating_a, rating_b):
    """Probability that A beats B. Feed home advantage in via rating_a."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(winner_rating, loser_rating, margin):
    """
    538-style margin-of-victory multiplier.

    Two things going on:
      (|MOV| + 3) ** 0.8   -> blowouts move ratings more, with diminishing returns
      / (7.5 + 0.006 * d)  -> damps autocorrelation, so good teams don't run away
                              by repeatedly blowing out bad ones
    """
    diff = winner_rating - loser_rating
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * diff)


def update(home_elo, away_elo, margin, home_adv):
    """
    Return (new_home_elo, new_away_elo) after a game.

    `margin` is home_pts - away_pts. Home advantage is applied for the
    purposes of the update but is not baked into the stored ratings.
    """
    home_eff = home_elo + home_adv
    home_won = margin > 0

    if home_won:
        mult = mov_multiplier(home_eff, away_elo, margin)
        shift = ELO_K * mult * (1.0 - expected(home_eff, away_elo))
        return home_elo + shift, away_elo - shift

    mult = mov_multiplier(away_elo, home_eff, margin)
    shift = ELO_K * mult * (1.0 - expected(away_elo, home_eff))
    return home_elo - shift, away_elo + shift


def regress_to_mean(rating):
    """Applied to every team at the start of a new season."""
    return ELO_SEASON_CARRY * rating + (1.0 - ELO_SEASON_CARRY) * ELO_SEASON_MEAN