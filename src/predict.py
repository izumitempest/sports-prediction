import pandas as pd
import joblib
import os
import sys
from feature_engineering import calculate_recent_form

DATA_PATH = os.path.join("data", "raw", "all_seasons.csv")
MODEL_PATH = os.path.join("models", "rf_model.pkl")

def get_latest_stats(df, team):
    """
    Get the latest form stats for a team.
    """
    # We need to sort by date and find the last row where Team played
    # The df should already have 'HomeFormPoints', etc. computed if we ran preprocess
    # But wait, preprocess computes it based on the *previous* games.
    # So if we take the last row where the team played, the 'FormPoints' in that row
    # represents the form *entering* that match.
    # We want the form *after* that match (i.e. including that match).
    
    # Actually, simpler:
    # 1. Filter all games involving the team.
    # 2. Calculate the points/goals for those games.
    # 3. Take the last 5.
    
    team_games = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(5)
    
    if len(team_games) < 5:
        print(f"Warning: Less than 5 games found for {team}")
    
    # Calculate points obtained in these games
    points = []
    goals_scored = 0
    goals_conceded = 0
    
    for _, row in team_games.iterrows():
        if row['HomeTeam'] == team:
            goals_scored += row['FTHG']
            goals_conceded += row['FTAG']
            if row['FTR'] == 'H': points.append(3)
            elif row['FTR'] == 'D': points.append(1)
            else: points.append(0)
        else: # AwayTeam
            goals_scored += row['FTAG']
            goals_conceded += row['FTHG']
            if row['FTR'] == 'A': points.append(3)
            elif row['FTR'] == 'D': points.append(1)
            else: points.append(0)
            
    form_points = sum(points)
    avg_goals = goals_scored / len(team_games) if len(team_games) > 0 else 0
    avg_conceded = goals_conceded / len(team_games) if len(team_games) > 0 else 0
    
    return form_points, avg_goals, avg_conceded

def predict(home_team, away_team):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train it first.")
        return

    print("Loading model and data...")
    rf = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # We need the team encoding map used during training
    # In `feature_engineering.py`, we did:
    # all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    # team_map = {team: i for i, team in enumerate(all_teams)}
    # Since we didn't save the map, we must recreate it exactly as before.
    # This works ONLY if we use the EXACT SAME dataset (all_seasons.csv).
    # Ideally, we should have saved the encoder.
    
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_map = {team: i for i, team in enumerate(all_teams)}
    
    if home_team not in team_map or away_team not in team_map:
        print("Error: One of the teams not found in history.")
        return

    # features = ['HomeTeamCode', 'AwayTeamCode', 'HomeFormPoints', 'AwayFormPoints', 
    #             'HomeAvgGoals', 'AwayAvgGoals', 'HomeAvgConceded', 'AwayAvgConceded']
    
    h_form, h_att, h_def = get_latest_stats(df, home_team)
    a_form, a_att, a_def = get_latest_stats(df, away_team)
    
    print(f"\nStats for {home_team}: Form={h_form}, GF={h_att:.1f}, GA={h_def:.1f}")
    print(f"Stats for {away_team}: Form={a_form}, GF={a_att:.1f}, GA={a_def:.1f}")
    
    X = pd.DataFrame([{
        'HomeTeamCode': team_map[home_team],
        'AwayTeamCode': team_map[away_team],
        'HomeFormPoints': h_form,
        'AwayFormPoints': a_form,
        'HomeAvgGoals': h_att,
        'AwayAvgGoals': a_att,
        'HomeAvgConceded': h_def,
        'AwayAvgConceded': a_def
    }])
    
    # Predict
    probs = rf.predict_proba(X)[0]
    classes = rf.classes_ # [0, 1, 2] -> [Home, Draw, Away] usually check mappings
    
    # My mapping was H=0, D=1, A=2
    print("\nPrediction Probabilities:")
    print(f"{home_team} Win: {probs[0]:.1%}")
    print(f"Draw:       {probs[1]:.1%}")
    print(f"{away_team} Win: {probs[2]:.1%}")
    
    winner_idx = probs.argmax()
    idx_to_label = {0: home_team, 1: "Draw", 2: away_team}
    print(f"\nPredicted Result: {idx_to_label[winner_idx]}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/predict.py 'Home Team' 'Away Team'")
    else:
        predict(sys.argv[1], sys.argv[2])
