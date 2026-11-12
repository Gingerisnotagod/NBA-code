"""
See whether your picks were right.

    py results.py                     games from your most recent day of finished picks
    py results.py --date 2026-11-10   games on one date
    py results.py --days 3            the last 3 days that have finished picks

Output, one line per game (home team first):

    Game       Pick  Prob  Winner  Result
    LALvsDET   LAL   .536  DET     Incorrect

Pick is who the model thought would win, Prob is the chance it gave them.

Reads picks_log.csv, and gets final scores from balldontlie (usually one quick
request, since bdl_cache already has older games). Changes nothing in your files.
"""

import argparse
import sys

import pandas as pd

import bdl

LOG_PATH = "picks_log.csv"

# nickname (last word of the team name) -> abbreviation
ABBR = {
    "hawks": "ATL", "celtics": "BOS", "nets": "BKN", "hornets": "CHA", "bulls": "CHI",
    "cavaliers": "CLE", "mavericks": "DAL", "nuggets": "DEN", "pistons": "DET",
    "warriors": "GSW", "rockets": "HOU", "pacers": "IND", "clippers": "LAC",
    "lakers": "LAL", "grizzlies": "MEM", "heat": "MIA", "bucks": "MIL",
    "timberwolves": "MIN", "pelicans": "NOP", "knicks": "NYK", "thunder": "OKC",
    "magic": "ORL", "76ers": "PHI", "suns": "PHX", "blazers": "POR", "kings": "SAC",
    "spurs": "SAS", "raptors": "TOR", "jazz": "UTA", "wizards": "WAS",
}


def nick(name):
    return str(name).split()[-1].lower()


def abbr(name):
    return ABBR.get(nick(name), str(name)[:3].upper())


def current_season(today):
    return today.year if today.month >= 8 else today.year - 1


def main():
    p = argparse.ArgumentParser(description="Were my picks right?")
    p.add_argument("--date", help="YYYY-MM-DD: show games on this date")
    p.add_argument("--days", type=int, default=1,
                   help="how many recent days with finished picks to show (default 1)")
    args = p.parse_args()

    try:
        log = pd.read_csv(LOG_PATH)
    except FileNotFoundError:
        sys.exit(f"no {LOG_PATH} yet; run py main.py --upcoming first")

    log["game_date"] = pd.to_datetime(log.game_date).dt.normalize()
    log["predicted_at"] = pd.to_datetime(log.predicted_at)
    # only predictions made before the game's day ended; newest one per game
    log = log[log.predicted_at < log.game_date + pd.Timedelta(days=1)]
    log = (log.sort_values("predicted_at")
              .drop_duplicates(["game_date", "home_team", "away_team"], keep="last"))
    if log.empty:
        sys.exit("no picks in the log yet")

    # final scores for the season(s) the picks cover
    seasons = sorted({current_season(d) for d in log.game_date})
    games = pd.concat([bdl.season_games(s) for s in seasons], ignore_index=True)
    games = games[games.status_state == "final"].copy()
    games["game_date"] = pd.to_datetime(games.game_date).dt.normalize()
    games["home_nick"] = games.home_team.map(nick)
    games["away_nick"] = games.away_team.map(nick)
    games["winner_nick"] = games.home_nick.where(games.home_pts > games.away_pts, games.away_nick)

    log["home_nick"] = log.home_team.map(nick)
    log["away_nick"] = log.away_team.map(nick)
    done = log.merge(games[["game_date", "home_nick", "away_nick", "winner_nick"]],
                     on=["game_date", "home_nick", "away_nick"], how="inner")
    if done.empty:
        sys.exit("none of your picks have final scores yet")

    if args.date:
        day = pd.to_datetime(args.date).normalize()
        done = done[done.game_date == day]
        if done.empty:
            sys.exit(f"no finished picks on {day.date()}")
    else:
        last_days = sorted(done.game_date.unique())[-max(args.days, 1):]
        done = done[done.game_date.isin(last_days)]

    done = done.sort_values(["game_date", "home_team"])
    right = 0
    for date, day_games in done.groupby("game_date"):
        print(f"\n{date.date()}")
        print(f"{'Game':<10} {'Pick':<5} {'Prob':<5} {'Winner':<7} Result")
        for g in day_games.itertuples():
            pick_prob = g.home_win_prob if nick(g.pick) == g.home_nick else 1 - g.home_win_prob
            correct = nick(g.pick) == g.winner_nick
            right += correct
            game = f"{abbr(g.home_team)}vs{abbr(g.away_team)}"
            prob = f"{pick_prob:.3f}".lstrip("0")
            print(f"{game:<10} {abbr(g.pick):<5} {prob:<5} {ABBR[g.winner_nick]:<7} "
                  f"{'Correct' if correct else 'Incorrect'}")

    print(f"\n{right}/{len(done)} correct ({right / len(done):.0%})")


if __name__ == "__main__":
    main()