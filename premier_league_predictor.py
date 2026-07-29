"""
Premier League Match Predictor
--------------------------------
Predicts: Home Win / Draw / Away Win (classification)
Using historical match data from football-data.co.uk
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from urllib.request import urlopen
import warnings

warnings.filterwarnings("ignore")

# 1. LOAD HISTORICAL DATA (2010-2025 Premier League seasons)
print("Downloading Premier League historical data...")
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
    ]
]
df_list = []
for u in urls:
    try:
        df_list.append(pd.read_csv(u))
    except:
        pass
data = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(data)} matches\n")

# 2. PREPARE DATA - select columns we need
cols = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HST",
    "AST",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]
data = data[[c for c in cols if c in data.columns]]


# 3. CREATE FEATURES from each team's recent form
def eng_features(df):
    """Calculate rolling form features for each team."""
    rows = []
    for _, match in df.iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        past_h = df[
            ((df["HomeTeam"] == home) | (df["AwayTeam"] == home))
            & (df["Date"] < match["Date"])
        ].tail(5)
        past_a = df[
            ((df["HomeTeam"] == away) | (df["AwayTeam"] == away))
            & (df["Date"] < match["Date"])
        ].tail(5)

        def team_stats(past, team):
            if len(past) == 0:
                return [0, 0, 0, 0, 0]
            pts = 0
            gf = ga = 0
            for _, m in past.iterrows():
                if m["HomeTeam"] == team:
                    gf += m["FTHG"]
                    ga += m["FTAG"]
                    pts += 3 if m["FTR"] == "H" else (1 if m["FTR"] == "D" else 0)
                else:
                    gf += m["FTAG"]
                    ga += m["FTHG"]
                    pts += 3 if m["FTR"] == "A" else (1 if m["FTR"] == "D" else 0)
            return [pts, gf - ga, gf, ga, len(past)]

        rows.append(team_stats(past_h, home) + team_stats(past_a, away))
    cols = [
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
    ]
    return pd.DataFrame(rows, columns=cols, index=df.index)


# Sort by date
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Build features
feat = eng_features(data)
X = feat.fillna(0)

# Target: H=2, D=1, A=0
y = data["FTR"].map({"H": 2, "D": 1, "A": 0})

# 4. TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 5. EVALUATE
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("========== MODEL PERFORMANCE ==========")
print(f"Accuracy: {acc:.1%}")
print("\nClassification Report:")
print(
    classification_report(y_test, y_pred, target_names=["Away Win", "Draw", "Home Win"])
)
print("Feature importance (top 5):")
imp = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])[:5]
for name, val in imp:
    print(f"  {name:8s}  {val:.3f}")
print("=======================================\n")

# 6. PREDICT A SAMPLE UPCOMING MATCH
print("========== SAMPLE PREDICTIONS ==========")
# Create sample features for some hypothetical matches
sample_matches = [
    ("Manchester City", "Arsenal"),
    ("Liverpool", "Chelsea"),
    ("Everton", "Manchester United"),
]
# Use recent average team stats as a stand-in
avg_stats = X.mean().to_dict()
for home, away in sample_matches:
    # Use average stats for demo (in real use, compute from actual recent form)
    sample = avg_stats.copy()
    # Slightly adjust for team strength (rough heuristic)
    sample["H_Pts"] *= (
        1.3
        if home in ["Manchester City", "Liverpool"]
        else (0.9 if home in ["Everton"] else 1.0)
    )
    sample["A_Pts"] *= (
        1.3
        if away in ["Manchester City", "Liverpool"]
        else (0.9 if away in ["Everton"] else 1.0)
    )
    probs = model.predict_proba(pd.DataFrame([sample]))[0]
    result = model.predict(pd.DataFrame([sample]))[0]
    label = {2: "Home Win", 1: "Draw", 0: "Away Win"}[result]
    print(
        f"{home:25s} vs {away:20s} => {label:12s}  "
        f"(H:{probs[2]:.0%} D:{probs[1]:.0%} A:{probs[0]:.0%})"
    )

print("\nRun: python predict_future.py to predict your own matches!")
