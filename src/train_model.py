import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import preprocess_data

DATA_PATH = os.path.join("data", "raw", "all_seasons.csv")

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Data file {DATA_PATH} not found. Run data_loader.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    print("Preprocessing and Feature Engineering...")
    df = preprocess_data(df)
    
    # Features to use
    features = [
        'HomeTeamCode', 'AwayTeamCode', 
        'HomeFormPoints', 'AwayFormPoints', 
        'HomeAvgGoals', 'AwayAvgGoals', 
        'HomeAvgConceded', 'AwayAvgConceded'
    ]
    target = 'Target'
    
    # Drop rows with NaNs (first few games of season has no history)
    df_model = df.dropna(subset=features + [target]).copy()
    
    X = df_model[features]
    y = df_model[target]
    
    print(f"Training on {len(df_model)} matches.")
    
    # Time Series Split (train on past, predict future)
    seasons = sorted(df_model['Season'].unique())
    test_season = seasons[-1] # Last available season
    train_seasons = seasons[:-1]
    
    print(f"Training on seasons: {train_seasons}")
    print(f"Testing on season: {test_season}")
    
    train_mask = df_model['Season'].isin(train_seasons)
    test_mask = df_model['Season'] == test_season
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # --- Optimization Start ---
    from sklearn.model_selection import RandomizedSearchCV
    
    print("Optimizing Hyperparameters...")
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    # Use TimeSeriesSplit for CV
    tscv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=tscv, scoring='accuracy', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    
    print(f"Best Params: {search.best_params_}")
    best_rf = search.best_estimator_
    # --- Optimization End ---
    
    # Predict
    preds = best_rf.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy on {test_season} season: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['Home', 'Draw', 'Away']))
    
    # Feature Importance
    importances = pd.Series(best_rf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances)
    
    # Baseline (Always predict Home Win)
    # 0 = Home Win
    baseline_preds = np.zeros(len(y_test)) 
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"\nBaseline (Always Home Win) Accuracy: {baseline_acc:.4f}")
    
    # Save Model
    import joblib
    if not os.path.exists("models"):
        os.makedirs("models")
        
    print("\nSaving model to models/rf_model.pkl")
    joblib.dump(best_rf, "models/rf_model.pkl")
    
    # We should also save the team map if we want to use it later, 
    # but for now we recreated it. Ideally, save the preprocessing artifacts.
    # For this simple example, we'll just save the model.

if __name__ == "__main__":
    train()
