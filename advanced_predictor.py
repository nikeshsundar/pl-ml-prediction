"""
PREMIER LEAGUE ADVANCED PREDICTOR
Uses 15+ years of data, ELO ratings, H2H, rolling form,
shot stats, discipline, and XGBoost.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ========== 1. LOAD ALL DATA ==========
print("Loading 15 seasons of Premier League data...")
seasons = [
    (y1, y2)
    for y1, y2 in [
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (25, 26),
    ]
]

all_data = []
for y1, y2 in seasons:
    try:
        df = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{y1}{y2}/E0.csv")
        df["Season"] = f"{y1}{y2}"
        all_data.append(df)
    except:
        pass

data = pd.concat(all_data, ignore_index=True)
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
print(
    f"Loaded {len(data)} matches ({data['Date'].min().year}-{data['Date'].max().year})"
)


# ========== 2. ELO RATINGS ==========
def compute_elo(data, k=32, home_adv=100):
    elo = {}
    ratings = []
    for _, match in data.iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        elo_h = elo.get(home, 1500)
        elo_a = elo.get(away, 1500)
        eh = elo_h + home_adv
        wh = 1 / (10 ** ((elo_a - eh) / 400) + 1)
        wa = 1 / (10 ** ((eh - elo_a) / 400) + 1)
        wd = 1 - wh - wa
        if match["FTR"] == "H":
            s_h, s_a = 1, 0
        elif match["FTR"] == "A":
            s_h, s_a = 0, 1
        else:
            s_h = s_a = 0.5
        elo[home] = elo_h + k * (s_h - wh)
        elo[away] = elo_a + k * (s_a - wa)
        ratings.append(
            {
                "HomeElo": elo[home],
                "AwayElo": elo[away],
                "EloDiff": elo[home] - elo[away],
            }
        )
    return pd.DataFrame(ratings)


print("Computing ELO ratings...")
elo_df = compute_elo(data)


# ========== 3. FEATURE ENGINEERING ==========
print("Engineering features...")

# Column name normalization - some seasons use different names
cols = data.columns.tolist()


def get_feature_stats(df, team, match_date, n_games, home_only=False, away_only=False):
    """Get rolling averages for a team over last N games."""
    mask = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    if home_only:
        mask = df["HomeTeam"] == team
    elif away_only:
        mask = df["AwayTeam"] == team
    past = df[mask & (df["Date"] < match_date)].tail(n_games)

    if len(past) == 0:
        return {}

    stats = {}
    is_home = past["HomeTeam"] == team

    # Basic results
    pts = np.where(
        is_home,
        np.where(past["FTR"] == "H", 3, np.where(past["FTR"] == "D", 1, 0)),
        np.where(past["FTR"] == "A", 3, np.where(past["FTR"] == "D", 1, 0)),
    )
    stats[f"Pts_last{n_games}"] = pts.sum()
    stats[f"PtsPerc_last{n_games}"] = pts.sum() / (len(past) * 3)

    # Goals
    gf = np.where(is_home, past["FTHG"], past["FTAG"])
    ga = np.where(is_home, past["FTAG"], past["FTHG"])
    stats[f"GF_last{n_games}"] = gf.sum()
    stats[f"GA_last{n_games}"] = ga.sum()
    stats[f"GD_last{n_games}"] = gf.sum() - ga.sum()
    stats[f"GFpg_last{n_games}"] = gf.mean()
    stats[f"GApg_last{n_games}"] = ga.mean()

    # Both teams scored
    stats[f"BTTS_last{n_games}"] = ((gf > 0) & (ga > 0)).mean()

    # Clean sheets
    stats[f"CS_last{n_games}"] = (ga == 0).mean()

    # Shots (available in most seasons)
    if "HS" in past.columns and "AS" in past.columns:
        shots_f = np.where(is_home, past["HS"].fillna(0), past["AS"].fillna(0))
        shots_a = np.where(is_home, past["AS"].fillna(0), past["HS"].fillna(0))
        stats[f"ShotsF_last{n_games}"] = shots_f.mean()
        stats[f"ShotsA_last{n_games}"] = shots_a.mean()

    # Shots on target
    if "HST" in past.columns and "AST" in past.columns:
        sot_f = np.where(is_home, past["HST"].fillna(0), past["AST"].fillna(0))
        sot_a = np.where(is_home, past["AST"].fillna(0), past["HST"].fillna(0))
        stats[f"SoTF_last{n_games}"] = sot_f.mean()
        stats[f"SoTA_last{n_games}"] = sot_a.mean()
        if shots_f.sum() > 0:
            stats[f"SoTPercF_last{n_games}"] = sot_f.sum() / shots_f.sum()

    # Corners
    if "HC" in past.columns and "AC" in past.columns:
        corn_f = np.where(is_home, past["HC"].fillna(0), past["AC"].fillna(0))
        corn_a = np.where(is_home, past["AC"].fillna(0), past["HC"].fillna(0))
        stats[f"CornF_last{n_games}"] = corn_f.mean()
        stats[f"CornA_last{n_games}"] = corn_a.mean()

    # Fouls
    if "HF" in past.columns and "AF" in past.columns:
        f_f = np.where(is_home, past["HF"].fillna(0), past["AF"].fillna(0))
        f_a = np.where(is_home, past["AF"].fillna(0), past["HF"].fillna(0))
        stats[f"FoulF_last{n_games}"] = f_f.mean()
        stats[f"FoulA_last{n_games}"] = f_a.mean()

    # Cards
    if "HY" in past.columns and "AY" in past.columns:
        y_f = np.where(is_home, past["HY"].fillna(0), past["AY"].fillna(0))
        y_a = np.where(is_home, past["AY"].fillna(0), past["HY"].fillna(0))
        stats[f"YC_last{n_games}"] = y_f.mean()
    if "HR" in past.columns and "AR" in past.columns:
        r_f = np.where(is_home, past["HR"].fillna(0), past["AR"].fillna(0))
        stats[f"RC_last{n_games}"] = r_f.mean()

    return stats


def get_hth_stats(df, home, away, match_date, n=5):
    """Head-to-head stats between two teams."""
    hth = df[
        (
            ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
            | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
        )
        & (df["Date"] < match_date)
    ].tail(n)
    if len(hth) == 0:
        return {}

    home_wins = ((hth["HomeTeam"] == home) & (hth["FTR"] == "H")).sum() + (
        (hth["AwayTeam"] == home) & (hth["FTR"] == "A")
    ).sum()
    away_wins = ((hth["HomeTeam"] == away) & (hth["FTR"] == "H")).sum() + (
        (hth["AwayTeam"] == away) & (hth["FTR"] == "A")
    ).sum()
    draws = len(hth) - home_wins - away_wins

    return {
        "HTH_GP": len(hth),
        "HTH_HomeWins": home_wins,
        "HTH_AwayWins": away_wins,
        "HTH_Draws": draws,
    }


def get_season_stats(df, team, match_date):
    """Get stats for the current season so far (before this match)."""
    season = df[
        (df["Date"] < match_date)
        & (df["Date"] >= match_date - timedelta(days=365))  # approx 1 season
        & ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
    ]
    if len(season) == 0:
        return {}

    is_home = season["HomeTeam"] == team
    pts = np.where(
        is_home,
        np.where(season["FTR"] == "H", 3, np.where(season["FTR"] == "D", 1, 0)),
        np.where(season["FTR"] == "A", 3, np.where(season["FTR"] == "D", 1, 0)),
    )
    gf = np.where(is_home, season["FTHG"], season["FTAG"])
    ga = np.where(is_home, season["FTAG"], season["FTHG"])

    return {
        "SeasonPts": pts.sum(),
        "SeasonGP": len(season),
        "SeasonPPG": pts.sum() / len(season),
        "SeasonGD": gf.sum() - ga.sum(),
        "SeasonGFpg": gf.mean(),
        "SeasonGApg": ga.mean(),
    }


# Build feature matrix
print("Building feature matrix...")
features_list = []
targets_list = []
eloi = 0
errors = 0

for idx, match in data.iterrows():
    try:
        row = {}
        home, away = match["HomeTeam"], match["AwayTeam"]
        match_date = match["Date"]

        # ELO (already computed)
        row["HomeElo"] = elo_df.loc[idx, "HomeElo"]
        row["AwayElo"] = elo_df.loc[idx, "AwayElo"]
        row["EloDiff"] = elo_df.loc[idx, "EloDiff"]

        # Form for different windows
        for n in [3, 5, 10]:
            for side, team in [("H", home), ("A", away)]:
                stats = get_feature_stats(data, team, match_date, n)
                for k, v in stats.items():
                    row[f"{side}_{k}"] = v
                # Home/away specific form (5 games)
                if n == 5:
                    home_stats = get_feature_stats(
                        data, team, match_date, 5, home_only=True
                    )
                    away_stats = get_feature_stats(
                        data, team, match_date, 5, away_only=True
                    )
                    for k, v in home_stats.items():
                        row[f"{side}_Home_{k}"] = v
                    for k, v in away_stats.items():
                        row[f"{side}_Away_{k}"] = v

        # Head-to-head
        hth = get_hth_stats(data, home, away, match_date)
        for k, v in hth.items():
            row[k] = v

        # Season stats
        for side, team in [("H", home), ("A", away)]:
            ss = get_season_stats(data, team, match_date)
            for k, v in ss.items():
                row[f"{side}_{k}"] = v

        features_list.append(row)
        targets_list.append(match["FTR"])
    except Exception as e:
        errors += 1
        continue

print(f"Built features for {len(features_list)} matches ({errors} errors)")
print(f"Total features: {len(features_list[0]) if features_list else 0}")


# ========== 4. TRAIN MODEL ==========
print("\nTraining model...")
X = pd.DataFrame(features_list).fillna(0)
y = pd.Series(targets_list).map({"H": 2, "D": 1, "A": 0})

# Drop rows with no feature variance
X = X.loc[:, X.std() > 0]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X.shape[1]}")

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Try XGBoost first, fall back to Random Forest
try:
    from xgboost import XGBClassifier

    print("Using XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train_s, y_train)
except ImportError:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    print("XGBoost not installed, using Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
    )
    model.fit(X_train_s, y_train)


# ========== 5. EVALUATE ==========
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)
acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_proba)

print("\n" + "=" * 60)
print(f"{'MODEL PERFORMANCE':^60}")
print("=" * 60)
print(f"Accuracy:          {acc:.1%}")
print(f"Log Loss:          {ll:.3f}  (lower = better)")
print(f"Baseline (always H): 45.0%")
print(f"Improvement:       {acc - 0.45:+.1%}")
print("-" * 60)
print("\nClassification Report:")
print(
    classification_report(y_test, y_pred, target_names=["Away Win", "Draw", "Home Win"])
)


# ========== 6. FEATURE IMPORTANCE ==========
print("\nTop 20 Most Important Features:")
if hasattr(model, "feature_importances_"):
    imp = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])
    for name, val in imp[:20]:
        print(f"  {name:35s} {val:.4f}")


# ========== 7. PREDICT WEEK 1 ==========
print("\n" + "=" * 60)
print(f"{'WEEK 1 - 2026/27 PREDICTIONS':^60}")
print("=" * 60)

# Get latest team stats from the last season's end
last_date = data["Date"].max()


def get_current_team_stats(df, team, as_of_date):
    """Get all features for a team as if predicting a match today."""
    stats = get_feature_stats(df, team, as_of_date, 5)
    home_stats = get_feature_stats(df, team, as_of_date, 5, home_only=True)
    away_stats = get_feature_stats(df, team, as_of_date, 5, away_only=True)
    ss = get_season_stats(df, team, as_of_date)

    # Get ELO (use last known)
    last_match = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].iloc[-1]
    last_idx = last_match.name
    elo_val = elo_df.loc[
        last_idx, "HomeElo" if last_match["HomeTeam"] == team else "AwayElo"
    ]

    side_stats = {}
    for prefix, s in [("", stats), ("Home_", home_stats), ("Away_", away_stats)]:
        for k, v in s.items():
            side_stats[f"{prefix}{k}"] = v
    for k, v in ss.items():
        side_stats[k] = v
    side_stats["Elo"] = elo_val

    return side_stats


week1_fixtures = [
    ("Arsenal", "Coventry"),
    ("Hull", "Man United"),
    ("Everton", "Crystal Palace"),
    ("Ipswich", "Sunderland"),
    ("Nott'm Forest", "Leeds"),
    ("Brentford", "Tottenham"),
    ("Brighton", "Aston Villa"),
    ("Man City", "Bournemouth"),
    ("Newcastle", "Liverpool"),
    ("Fulham", "Chelsea"),
]

# Team name mapping for the data
name_map = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Coventry": None,
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull": None,
    "Ipswich": None,
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Man City": "Man City",
    "Man United": "Man United",
    "Newcastle": "Newcastle",
    "Nott'm Forest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
}

# Pre-compute team features
team_features = {}
for team in pd.unique(data[["HomeTeam", "AwayTeam"]].values.ravel()):
    team_features[team] = get_current_team_stats(
        data, team, last_date + timedelta(days=1)
    )

# Promoted team estimate (last 3 promoted teams averaged)
promoted_feats = {}
if team_features:
    sample_keys = list(next(iter(team_features.values())).keys())
    prom_vals = {}
    for k in sample_keys:
        try:
            vals = [v[k] for v in team_features.values() if k in v]
            prom_vals[k] = float(np.percentile(vals, 15)) if vals else 0.0
        except:
            prom_vals[k] = 0.0
    promoted_feats = prom_vals

# Template for feature vector (all columns must match training)
feature_template = {col: 0.0 for col in X.columns}

for home, away in week1_fixtures:
    h_name = name_map.get(home)
    a_name = name_map.get(away)
    home_feats = team_features.get(h_name, promoted_feats) if h_name else promoted_feats
    away_feats = team_features.get(a_name, promoted_feats) if a_name else promoted_feats

    feat_row = feature_template.copy()
    for key, val in home_feats.items():
        col = f"H_{key}"
        if col in feat_row:
            feat_row[col] = val
    for key, val in away_feats.items():
        col = f"A_{key}"
        if col in feat_row:
            feat_row[col] = val
    # ELO
    if "HomeElo" in feat_row and "HomeElo" in home_feats:
        feat_row["HomeElo"] = home_feats["HomeElo"]
    elif "HomeElo" in feat_row and "Elo" in home_feats:
        feat_row["HomeElo"] = home_feats["Elo"]
    if "AwayElo" in feat_row and "AwayElo" in away_feats:
        feat_row["AwayElo"] = away_feats["AwayElo"]
    elif "AwayElo" in feat_row and "Elo" in away_feats:
        feat_row["AwayElo"] = away_feats["Elo"]
    if "EloDiff" in feat_row:
        feat_row["EloDiff"] = feat_row["HomeElo"] - feat_row["AwayElo"]

    full_row = pd.DataFrame([feat_row])
    full_row_scaled = scaler.transform(full_row)

    probs = model.predict_proba(full_row_scaled)[0]
    pred = model.predict(full_row_scaled)[0]
    label = {2: "HOME WIN", 1: "DRAW", 0: "AWAY WIN"}[pred]
    prob_h = probs[2] if len(probs) > 2 else 0
    prob_d = probs[1] if len(probs) > 1 else 0
    prob_a = probs[0] if len(probs) > 0 else 0
    conf = "STRONG" if max(probs) > 0.5 else "LEAN" if max(probs) > 0.4 else "TOSS-UP"
    print(
        f"{home:20s} vs {away:20s}  {label:10s}  {prob_h:.0%}/{prob_d:.0%}/{prob_a:.0%}  [{conf}]"
    )

print("\n" + "=" * 60)
print(f"Model accuracy on historical test data: {acc:.1%}")
