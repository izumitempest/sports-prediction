from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime

# Import helper functions if needed, or redefine them for a self-contained app
# For this task, we'll redefine get_latest_stats to maintain independence

app = FastAPI(title="Sports Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join("data", "raw", "all_seasons.csv")
MODEL_PATH = os.path.join("models", "rf_model.pkl")

# Load model and data at startup
if not os.path.exists(MODEL_PATH):
    print("Warning: Model not found. Please train the model first.")
    rf = None
else:
    rf = joblib.load(MODEL_PATH)

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    all_teams_names = sorted(pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique())
    # Recreate team_map exactly as in training
    # Note: training used unique() which might depend on appearance order.
    # To be safe, we should use the exact same logic as in train_model.py/predict.py
    # predict.py does: all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    train_teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    team_map = {team: i for i, team in enumerate(train_teams)}
else:
    df = None
    all_teams_names = []
    team_map = {}


class TeamRequest(BaseModel):
    home_team: str
    away_team: str


def get_latest_stats(team_name):
    if df is None:
        return 0, 0, 0

    team_games = (
        df[(df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)]
        .sort_values("Date")
        .tail(5)
    )

    points = []
    goals_scored = 0
    goals_conceded = 0

    for _, row in team_games.iterrows():
        if row["HomeTeam"] == team_name:
            goals_scored += row["FTHG"]
            goals_conceded += row["FTAG"]
            if row["FTR"] == "H":
                points.append(3)
            elif row["FTR"] == "D":
                points.append(1)
            else:
                points.append(0)
        else:  # AwayTeam
            goals_scored += row["FTAG"]
            goals_conceded += row["FTHG"]
            if row["FTR"] == "A":
                points.append(3)
            elif row["FTR"] == "D":
                points.append(1)
            else:
                points.append(0)

    form_points = sum(points)
    avg_goals = goals_scored / len(team_games) if len(team_games) > 0 else 0
    avg_conceded = goals_conceded / len(team_games) if len(team_games) > 0 else 0

    return form_points, avg_goals, avg_conceded


@app.get("/teams")
async def get_teams():
    return [
        {
            "name": team,
            "logoUrl": f"https://picsum.photos/seed/{team.replace(' ', '')}/40/40",
        }
        for team in all_teams_names
    ]


@app.post("/predict")
async def predict_match(request: TeamRequest):
    if rf is None or df is None:
        raise HTTPException(status_code=500, detail="Model or data not loaded.")

    if request.home_team not in team_map or request.away_team not in team_map:
        raise HTTPException(
            status_code=400, detail="One or both teams not found in history."
        )

    h_form, h_att, h_def = get_latest_stats(request.home_team)
    a_form, a_att, a_def = get_latest_stats(request.away_team)

    X = pd.DataFrame(
        [
            {
                "HomeTeamCode": team_map[request.home_team],
                "AwayTeamCode": team_map[request.away_team],
                "HomeFormPoints": h_form,
                "AwayFormPoints": a_form,
                "HomeAvgGoals": h_att,
                "AwayAvgGoals": a_att,
                "HomeAvgConceded": h_def,
                "AwayAvgConceded": a_def,
            }
        ]
    )

    probs = rf.predict_proba(X)[0]

    # Matching PredictionOutcome enum in Gopredict/types.ts:
    # HOME = 'Home Win', DRAW = 'Draw', AWAY = 'Away Win'
    winner_idx = probs.argmax()
    prediction_label = (
        "Home Win" if winner_idx == 0 else "Away Win" if winner_idx == 2 else "Draw"
    )

    # Matching types.ts in frontend:
    # PredictionOutcome = 'HOME' | 'AWAY' | 'DRAW'

    return {
        "id": f"{request.home_team}-{request.away_team}-{int(datetime.now().timestamp())}",
        "homeTeam": {
            "name": request.home_team,
            "logoUrl": f"https://picsum.photos/seed/{request.home_team.replace(' ', '')}/40/40",
        },
        "awayTeam": {
            "name": request.away_team,
            "logoUrl": f"https://picsum.photos/seed/{request.away_team.replace(' ', '')}/40/40",
        },
        "league": "Premier League",
        "country": "England",
        "countryCode": "gb-eng",
        "matchDate": datetime.now().isoformat(),  # Just for display
        "prediction": prediction_label,
        "probabilities": {
            "home": float(probs[0]),
            "draw": float(probs[1]),
            "away": float(probs[2]),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
