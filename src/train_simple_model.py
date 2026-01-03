import csv
import os
import pickle
from collections import defaultdict

DATA_DIR = os.path.join("data", "raw")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "simple_model.pkl")

def load_all_data():
    all_rows = []
    # List files in date order usually helps, but filename sorting works for 2021, 2122 etc.
    # Actually need to sort by season properly.
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "E0" in f and "all_seasons" not in f]
    files.sort() # 'E0_2021.csv', 'E0_2122.csv', ...
    
    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"Reading {path}...")
        with open(path, 'r', encoding='unicode_escape') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Basic validation
                if row['FTR'] and row['HomeTeam'] and row['AwayTeam']:
                     all_rows.append(row)
    return all_rows

def train_simple_model():
    data = load_all_data()
    print(f"Loaded {len(data)} matches.")
    
    # Model: Calculate "Home Strength" and "Away Strength"
    # Logic: 
    # HomeStrength = (Goals Scored at Home) / (Games Played at Home)
    # AwayStrength = (Goals Scored Away) / (Games Played Away)
    # Also track Defense.
    
    team_stats = defaultdict(lambda: {'HG': 0, 'AG': 0, 'HC': 0, 'AC': 0, 'H_Games': 0, 'A_Games': 0})
    
    for row in data:
        home = row['HomeTeam']
        away = row['AwayTeam']
        try:
            fthg = int(row['FTHG'])
            ftag = int(row['FTAG'])
        except:
            continue
            
        # Update Home Team Stats
        stats = team_stats[home]
        stats['HG'] += fthg
        stats['HC'] += ftag
        stats['H_Games'] += 1
        
        # Update Away Team Stats
        stats = team_stats[away]
        stats['AG'] += ftag
        stats['AC'] += fthg
        stats['A_Games'] += 1
        
    # Calculate Averages
    model = {}
    for team, stats in team_stats.items():
        h_games = max(1, stats['H_Games'])
        a_games = max(1, stats['A_Games'])
        
        model[team] = {
            'HomeAttack': stats['HG'] / h_games,
            'HomeDefense': stats['HC'] / h_games,
            'AwayAttack': stats['AG'] / a_games,
            'AwayDefense': stats['AC'] / a_games,
            'OverallGames': h_games + a_games
        }
        
    # Save Model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model trained on {len(model)} teams and saved to {MODEL_FILE}")
    
    # Print top 5 teams by Home Attack
    sorted_teams = sorted(model.items(), key=lambda x: x[1]['HomeAttack'], reverse=True)
    print("\nTop 5 Attacking Home Teams:")
    for t, s in sorted_teams[:5]:
        print(f"{t}: {s['HomeAttack']:.2f} goals/game")

if __name__ == "__main__":
    train_simple_model()
