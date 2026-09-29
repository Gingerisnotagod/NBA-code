"""Tunable constants. Everything else imports from here."""

# --- Elo ---
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 65.0         # elo points; ~= 59% home win rate for equal teams
ELO_SEASON_CARRY = 0.75     # keep 75% of rating across seasons, regress 25% to mean
ELO_SEASON_MEAN = 1505.0

# --- Rolling features ---
ROLL_WINDOW = 10            # games in the form / strength-of-schedule window
DEFAULT_REST_DAYS = 3       # assumed rest for a team's first game of a season
MAX_REST_DAYS = 7           # cap: 8 days off is not meaningfully better than 7

# --- Backtest ---
MIN_TRAIN_SEASONS = 3       # seasons of history before the first test season
MIN_TEST_GAMES = 100        # skip seasons with too few games (partial data)

# --- Model input ---
# Order matters only for readability; the models key off column names.
FEATURES = [
    "elo_diff",
    "elo_home_prob",
    "form_diff",
    "netrtg_diff",
    "rest_diff",
    "b2b_home",
    "b2b_away",
    "sos_diff",
]

# Columns every loader must produce before feature building.
REQUIRED_COLS = ["game_date", "season", "home_team", "away_team", "home_pts", "away_pts"]