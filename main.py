"""
CLI entrypoint.

Predict upcoming games (the main use):
    py main.py --upcoming
    py main.py --upcoming --days 7

Validate the model before trusting it:
    py main.py --backtest --min-season 1985

One-off matchup:
    py main.py --game "Boston Celtics" "Miami Heat"

Default data source is the Kaggle DB (nba.sqlite) topped up with balldontlie
from 2023-24 on, and the schedule comes from balldontlie. Needs BDL_API_KEY set.
"""

import argparse
import sys

import backtest as backtest_mod
import data
import features
import injuries as injuries_mod
import models
import predict
import schedule as schedule_mod
import tracker
from config import FEATURES


def parse_args():
    p = argparse.ArgumentParser(description="NBA game winner predictor")

    src = p.add_argument_group("data source")
    src.add_argument("--source", choices=["sqlite+bdl", "sqlite", "bdl", "nba_api", "csv"],
                     default="sqlite+bdl")
    src.add_argument("--db", default="nba.sqlite")
    src.add_argument("--csv", default="games.csv")
    src.add_argument("--seasons", nargs=2, type=int, default=[2023, 2027],
                     metavar=("START", "END"),
                     help="API season range [START, END); 2025 = 2025-26")
    src.add_argument("--min-season", type=int, default=None,
                     help="drop seasons before this year")
    src.add_argument("--schedule-source", choices=["bdl", "cdn"], default="bdl")
    src.add_argument("--schedule-season", type=int, default=None,
                     help="season to pull the schedule for (default: current)")

    act = p.add_argument_group("what to do")
    act.add_argument("--upcoming", action="store_true",
                     help="predict scheduled games that have not been played")
    act.add_argument("--days", type=int, default=1,
                     help="window for --upcoming (default 1 = next slate only)")
    act.add_argument("--start", default=None, help="start date, YYYY-MM-DD")
    act.add_argument("--new-season", action="store_true",
                     help="force offseason rollover before predicting (auto-applied "
                          "for --upcoming when history ends in an earlier season)")
    act.add_argument("--backtest", action="store_true", help="walk-forward validation")
    act.add_argument("--rankings", action="store_true", help="current Elo top 10")
    act.add_argument("--game", nargs=2, metavar=("HOME", "AWAY"), action="append",
                     help="single matchup; repeatable")
    act.add_argument("--date", default=None, help="date for --game")
    act.add_argument("--score", action="store_true",
                     help="grade logged picks (picks_log.csv) against actual results")
    act.add_argument("--no-save", action="store_true",
                     help="don't log this run's picks to picks_log.csv")
    act.add_argument("--injuries", default="injuries.csv",
                     help="manual injury file (see injuries.py); 'none' to ignore")

    args = p.parse_args()
    if not any([args.upcoming, args.backtest, args.rankings, args.game, args.score]):
        args.upcoming = True          # sensible default
    return args


def main():
    args = parse_args()

    games = data.load(args.source, db=args.db, csv=args.csv, seasons=args.seasons)
    if args.min_season:
        games = games[games.season >= args.min_season].reset_index(drop=True)
    print(f"history: {len(games):,} games through {games.game_date.max().date()}")

    feats, state = features.build(games)
    model = predict.fit_final(feats, kind="logistic")

    target_season = args.schedule_season or schedule_mod.current_season()
    last_season = int(games.season.max())
    if args.new_season or (args.upcoming and last_season < target_season):
        predict.roll_into_new_season(state)
        print(f"applied offseason rollover ({last_season} -> {target_season}): "
              f"Elo regressed, form/rest cleared")

    # current-franchise names only, so nickname matching can't hit a defunct team
    recent = games[games.season == last_season]
    current_names = set(recent.home_team) | set(recent.away_team)

    inj = None
    if args.injuries.lower() != "none":
        inj = injuries_mod.load(args.injuries, known_teams=current_names)
        if len(inj):
            print(f"injury adjustments loaded: {len(inj)} player(s)")

    if args.score:
        tracker.score(games)

    if args.backtest:
        per_season, pooled = backtest_mod.walk_forward(feats)
        print("\nper-season accuracy (walk-forward)")
        print(per_season.round(4).to_string(index=False))
        print("\npooled out-of-sample")
        print(pooled.round(4).to_string())
        print("\nlogistic coefficients (standardized)")
        for name, coef in models.coefficients(model, FEATURES):
            print(f"  {name:<16} {coef:+.4f}")

    if args.rankings:
        print("\ncurrent elo")
        print(predict.power_rankings(state).round(1).to_string(index=False))

    if args.game:
        date = args.date or games.game_date.max()
        matchups = [tuple(g) for g in args.game]
        print(f"\nprediction for {str(date)[:10]}")
        print(predict.predict(model, state, matchups, date, inj).round(4).to_string(index=False))

    if args.upcoming:
        try:
            if args.schedule_source == "bdl":
                sched = schedule_mod.fetch_season_schedule_bdl(target_season)
            else:
                sched = schedule_mod.fetch_season_schedule()
        except Exception as exc:
            print(f"\ncould not fetch schedule: {exc}", file=sys.stderr)
            return

        sched, unresolved = schedule_mod.align_team_names(sched, current_names)
        if unresolved:
            print(f"warning: unmatched team names, skipped: {unresolved}", file=sys.stderr)

        slate = schedule_mod.upcoming(sched, start=args.start, days=args.days)
        if slate.empty:
            slate = schedule_mod.next_slate(sched, after=args.start)
            if slate.empty:
                print("\nno unplayed games found in the schedule.")
                return
            print(f"\nno games in window; showing next slate "
                  f"({slate.game_date.iloc[0].date()})")

        preds = predict.predict_slate(model, state, slate, inj)
        print(f"\n{len(preds)} upcoming games, most confident first\n")
        print(preds.round({"home_win_prob": 3, "confidence": 3}).to_string(index=False))

        if not args.no_save:
            saved = tracker.save_picks(preds)
            print(f"\nlogged {saved} picks to {tracker.LOG_PATH} (grade later with --score)")


if __name__ == "__main__":
    main()