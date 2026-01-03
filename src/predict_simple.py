import pickle
import os
import sys

MODEL_FILE = os.path.join("models", "simple_model.pkl")

def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)

def predict(home, away):
    model = load_model()
    if not model:
        print("Model not found. Run src/train_simple_model.py first.")
        return

    if home not in model:
        print(f"Team {home} not found in database.")
        return
    if away not in model:
        print(f"Team {away} not found in database.")
        return
    
    h_stats = model[home]
    a_stats = model[away]
    
    # Simple Expected Goals Model (Very naive)
    # We take the average of (Home Team Scoring Ability) and (Away Team Conceding Ability)
    exp_home_goals = (h_stats['HomeAttack'] + a_stats['AwayDefense']) / 2.0
    exp_away_goals = (a_stats['AwayAttack'] + h_stats['HomeDefense']) / 2.0
    
    print(f"\n--- Prediction: {home} vs {away} ---")
    print(f"{home} Home stats: Scored {h_stats['HomeAttack']:.2f}, Conceded {h_stats['HomeDefense']:.2f}")
    print(f"{away} Away stats: Scored {a_stats['AwayAttack']:.2f}, Conceded {a_stats['AwayDefense']:.2f}")
    
    print(f"\nExpected Score:")
    print(f"{home}: {exp_home_goals:.2f}")
    print(f"{away}: {exp_away_goals:.2f}")
    
    if exp_home_goals > exp_away_goals + 0.5:
        print(f"Prediction: {home} to WIN")
    elif exp_away_goals > exp_home_goals + 0.5:
        print(f"Prediction: {away} to WIN")
    else:
        print(f"Prediction: DRAW or Close Match")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/predict_simple.py 'HomeTeam' 'AwayTeam'")
    else:
        predict(sys.argv[1], sys.argv[2])
