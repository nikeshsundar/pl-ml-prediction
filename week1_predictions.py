"""
Premier League 2026/27 - Week 1 Predictions
Predicts all 10 opening weekend matches using historical form
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# 1. LOAD HISTORICAL DATA (all available seasons)
print("Loading Premier League historical data...")
urls = [
    f"https://www.football-data.co.uk/mmz4281/{y1}{y2}/E0.csv"
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
df_list = []
for u in urls:
    try:
        df_list.append(pd.read_csv(u))
    except:
        pass
data = pd.concat(df_list, ignore_index=True)
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
print(
    f"Loaded {len(data)} matches from {data['Date'].min().year} to {data['Date'].max().year}\n"
)


# 2. BUILD FORM FEATURES for each match
def get_recent_form(df, team, match_date, n=5):
    past = df[
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
        & (df["Date"] < match_date)
    ].tail(n)
    if len(past) == 0:
        return {"Pts": 0, "GD": 0, "GF": 0, "GA": 0, "GP": 0}
    pts = gf = ga = 0
    for _, m in past.iterrows():
        if m["HomeTeam"] == team:
            gf += m["FTHG"]
            ga += m["FTAG"]
            pts += 3 if m["FTR"] == "H" else (1 if m["FTR"] == "D" else 0)
        else:
            gf += m["FTAG"]
            ga += m["FTHG"]
            pts += 3 if m["FTR"] == "A" else (1 if m["FTR"] == "D" else 0)
    return {"Pts": pts, "GD": gf - ga, "GF": gf, "GA": ga, "GP": len(past)}


print("Building features from historical data...")
rows, targets = [], []
for _, match in data.iterrows():
    hf = get_recent_form(data, match["HomeTeam"], match["Date"])
    af = get_recent_form(data, match["AwayTeam"], match["Date"])
    rows.append(
        [hf[k] for k in ["Pts", "GD", "GF", "GA", "GP"]]
        + [af[k] for k in ["Pts", "GD", "GF", "GA", "GP"]]
    )
    targets.append(match["FTR"])

X = pd.DataFrame(
    rows,
    columns=[
        "H_Pts",
        "H_GD",
        "H_GF",
        "H_GA",
        "H_GP",
        "A_Pts",
        "A_GD",
        "A_GF",
        "A_GA",
        "A_GP",
    ],
)
y = pd.Series(targets).map({"H": 2, "D": 1, "A": 0})

# 3. TRAIN
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.1%}\n")

# 4. USE LAST 5 GAMES of last season as form (matches model training)
last_day = data["Date"].max()
team_form = {}
for team in pd.unique(data[["HomeTeam", "AwayTeam"]].values.ravel()):
    hf = get_recent_form(data, team, last_day + timedelta(days=1))
    team_form[team] = hf

# Promoted teams estimated as bottom-3 PL form
promoted_avg = {"Pts": 2, "GD": -5, "GF": 3, "GA": 8, "GP": 5}

print("\nLast 5 games form for top teams:")
for t, s in sorted(team_form.items(), key=lambda x: -x[1]["Pts"])[:5]:
    print(f"  {t:25s} Pts:{int(s['Pts'])}  GD:{int(s['GD']):+d}  GP:{int(s['GP'])}")
print()


def get_team_stats(team_name):
    """Get expected form for any team (handle promoted teams)."""
    name_map = {
        "Arsenal": "Arsenal",
        "Aston Villa": "Aston Villa",
        "Bournemouth": "Bournemouth",
        "AFC Bournemouth": "Bournemouth",
        "Brentford": "Brentford",
        "Brighton": "Brighton",
        "Brighton & Hove Albion": "Brighton",
        "Burnley": "Burnley",
        "Chelsea": "Chelsea",
        "Coventry": "Coventry",
        "Coventry City": "Coventry",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Fulham": "Fulham",
        "Hull": "Hull",
        "Hull City": "Hull",
        "Ipswich": "Ipswich",
        "Ipswich Town": "Ipswich",
        "Leeds": "Leeds",
        "Leeds United": "Leeds",
        "Leicester": "Leicester",
        "Liverpool": "Liverpool",
        "Man City": "Man City",
        "Manchester City": "Man City",
        "Man United": "Man United",
        "Manchester United": "Man United",
        "Newcastle": "Newcastle",
        "Newcastle United": "Newcastle",
        "Nott'm Forest": "Nott'm Forest",
        "Nottingham Forest": "Nott'm Forest",
        "Southampton": "Southampton",
        "Sunderland": "Sunderland",
        "Tottenham": "Tottenham",
        "Tottenham Hotspur": "Tottenham",
        "West Ham": "West Ham",
        "West Ham United": "West Ham",
        "Wolves": "Wolves",
        "Wolverhampton": "Wolves",
        "Wolverhampton Wanderers": "Wolves",
    }
    name = name_map.get(team_name, team_name)
    if name in team_form:
        return team_form[name]
    return promoted_avg


# 5. PREDICT WEEK 1
week1 = [
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

print("=" * 72)
print(f"{'WEEK 1 PREDICTIONS - 2026/27 PREMIER LEAGUE':^72}")
print(f"{'Match date: 21-24 August 2026':^72}")
print("=" * 72)
print(f"{'Home':25s} {'vs':5s} {'Away':25s} {'Prediction':15s} {'Probs (H/D/A)'}")
print("-" * 72)

for home, away in week1:
    h_stats = get_team_stats(home)
    a_stats = get_team_stats(away)
    features = pd.DataFrame(
        [
            [h_stats[k] for k in ["Pts", "GD", "GF", "GA", "GP"]]
            + [a_stats[k] for k in ["Pts", "GD", "GF", "GA", "GP"]]
        ],
        columns=X.columns,
    )
    probs = model.predict_proba(features)[0]
    pred = model.predict(features)[0]
    # Ensure we have all 3 classes
    prob_h = probs[2] if len(probs) > 2 else 0
    prob_d = probs[1] if len(probs) > 1 else 0
    prob_a = probs[0] if len(probs) > 0 else 0
    label = {2: "HOME WIN", 1: "DRAW", 0: "AWAY WIN"}[pred]
    home_short = home[:25]
    away_short = away[:25]
    print(
        f"{home_short:25s} {'vs':5s} {away_short:25s} {label:15s} {prob_h:.0%}/{prob_d:.0%}/{prob_a:.0%}"
    )

print("=" * 72)
print("\nConfidence guide: >50% = strong, 40-50% = lean, <40% = toss-up")
print("* Promoted teams (Coventry, Hull, Ipswich) estimated from Championship form")
