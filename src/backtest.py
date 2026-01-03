import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from feature_engineering import preprocess_data

DATA_PATH = os.path.join("data", "raw", "all_seasons.csv")
MODEL_PATH = os.path.join("models", "rf_model.pkl")

def run_backtest():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Model or Data not found.")
        return

    print("Loading data and model...")
    df = pd.read_csv(DATA_PATH)
    rf = joblib.load(MODEL_PATH)
    
    # Preprocess
    df = preprocess_data(df)
    
    # Use the same Test Set split as training (Last Season)
    # Note: In a real scenario, we should carefully handle the split to ensure no leakage.
    # We assume 'Season' column exists.
    seasons = sorted(df['Season'].unique())
    test_season = seasons[-1]
    
    print(f"Backtesting on Season: {test_season}")
    test_df = df[df['Season'] == test_season].copy()
    
    features = [
        'HomeTeamCode', 'AwayTeamCode', 
        'HomeFormPoints', 'AwayFormPoints', 
        'HomeAvgGoals', 'AwayAvgGoals', 
        'HomeAvgConceded', 'AwayAvgConceded'
    ]
    
    # Drop NAs
    test_df = test_df.dropna(subset=features + ['B365H', 'B365D', 'B365A'])
    
    X_test = test_df[features]
    y_test = test_df['Target']
    
    # Get Probabilities
    probs = rf.predict_proba(X_test)
    
    # Betting Logic
    initial_bankroll = 1000
    current_bankroll = initial_bankroll
    bet_amount = 25 # Smaller bet size
    policy_threshold = 0.10 # Need >10% expected value to bet
    
    # Filter out "Long Shots" (odds > 5.0) as models often overpredict them
    max_odds_threshold = 5.0
    
    bets_placed = 0
    wins = 0
    
    history = []
    
    print("\n--- Starting Simulation ---")
    
    for i, (index, row) in enumerate(test_df.iterrows()):
        # Odds
        odds_h = row['B365H']
        odds_d = row['B365D']
        odds_a = row['B365A']
        
        # Model Probs
        prob_h = probs[i][0]
        prob_d = probs[i][1]
        prob_a = probs[i][2]
        
        # Calculate EV
        # EV = (Prob * Odds) - 1
        ev_h = (prob_h * odds_h) - 1
        ev_d = (prob_d * odds_d) - 1
        ev_a = (prob_a * odds_a) - 1
        
        # Determine best bet
        candidates = []
        if ev_h > policy_threshold and odds_h < max_odds_threshold: candidates.append((ev_h, 'H', odds_h))
        if ev_d > policy_threshold and odds_d < max_odds_threshold: candidates.append((ev_d, 'D', odds_d))
        if ev_a > policy_threshold and odds_a < max_odds_threshold: candidates.append((ev_a, 'A', odds_a))
        
        if not candidates:
            history.append(current_bankroll)
            continue
            
        # Pick the one with highest EV
        best_bet = max(candidates, key=lambda x: x[0])
        pick = best_bet[1]
        odd_val = best_bet[2]
        
        outcome = row['FTR'] # H, D, A
        
        bets_placed += 1
        
        # Place Bet
        if pick == outcome:
            profit = (bet_amount * odd_val) - bet_amount
            wins += 1
            result_str = "WIN"
        else:
            profit = -bet_amount
            result_str = "LOSS"
            
        current_bankroll += profit
        
        # log occasional bets
        if i % 50 == 0:
            print(f"Match: {row['HomeTeam']} vs {row['AwayTeam']}. Pick: {pick} @ {odd_val}. Result: {outcome} ({result_str}). Bank: {current_bankroll:.2f}")

        history.append(current_bankroll)

    # Summary
    roi = ((current_bankroll - initial_bankroll) / initial_bankroll) * 100
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
    
    print("\n--- Backtest Report ---")
    print(f"Season: {test_season}")
    print(f"Total Bets: {bets_placed} / {len(test_df)} matches")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Final Bankroll: ${current_bankroll:.2f} (Start: ${initial_bankroll})")
    print(f"ROI: {roi:.2f}%")
    
    if roi > 0:
        print("RESULT: PROFITABLE ✅")
    else:
        print("RESULT: NOT PROFITABLE ❌ (Try tuning threshold or model)")

if __name__ == "__main__":
    run_backtest()
