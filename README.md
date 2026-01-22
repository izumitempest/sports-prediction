# ⚽ Sports Prediction Algorithm - EPL Match Predictor

A machine learning project that predicts English Premier League (EPL) soccer match outcomes and evaluates betting profitability. This isn't just a "who will win" predictor—it's a complete pipeline that downloads data, engineers features, trains models, and simulates betting strategies.

---

## 🎯 What Does This Project Do?

This project answers three questions:

1. **Who will win?** (Home, Draw, or Away)
2. **How confident are we?** (Probability percentages)
3. **Is it profitable to bet on this?** (Expected Value analysis)

---

## 🧠 How It Works: The Complete Pipeline

### 1. Data Collection (`src/data_loader.py`)

**What it does**: Downloads historical match data from [football-data.co.uk](https://www.football-data.co.uk).

**How it works**:

- Fetches CSV files for the last 5 EPL seasons (2019/20 - 2023/24)
- Each CSV contains ~380 matches (one full season)
- Combines all seasons into a single file: `data/raw/all_seasons.csv`

**Key columns we use**:

- `HomeTeam`, `AwayTeam`: Team names
- `FTHG`, `FTAG`: Full Time Home Goals, Away Goals
- `FTR`: Full Time Result (H/D/A)
- `B365H`, `B365D`, `B365A`: Bet365 betting odds for Home/Draw/Away

**Run it**:

```bash
python src/data_loader.py
```

---

### 2. Feature Engineering (`src/feature_engineering.py`)

**What it does**: Transforms raw match data into meaningful statistics that capture team performance.

**Why we need this**: Raw team names aren't useful for ML. We need numbers that represent "form", "strength", "momentum".

**Features we create**:

| Feature           | What It Means           | How It's Calculated                                          |
| ----------------- | ----------------------- | ------------------------------------------------------------ |
| `HomeFormPoints`  | Recent home form        | Sum of points from last 5 home games (Win=3, Draw=1, Loss=0) |
| `AwayFormPoints`  | Recent away form        | Sum of points from last 5 away games                         |
| `HomeAvgGoals`    | Home attacking power    | Average goals scored at home (last 5 games)                  |
| `AwayAvgGoals`    | Away attacking power    | Average goals scored away (last 5 games)                     |
| `HomeAvgConceded` | Home defensive weakness | Average goals conceded at home (last 5 games)                |
| `AwayAvgConceded` | Away defensive weakness | Average goals conceded away (last 5 games)                   |
| `HomeTeamCode`    | Team identifier         | Unique integer for each team                                 |
| `AwayTeamCode`    | Opponent identifier     | Unique integer for each team                                 |

**The Magic**: We use a **rolling window** approach. For each match, we only look at the previous 5 games—this prevents "future leakage" (using information that wouldn't be available at prediction time).

**Example**:

```text
Arsenal's last 5 home games: W, W, D, W, L
HomeFormPoints = 3 + 3 + 1 + 3 + 0 = 10 points
```

---

### 3. Model Training (`src/train_model.py`)

**What it does**: Trains a Random Forest Classifier to predict match outcomes.

**Why Random Forest?**

- Handles non-linear relationships (e.g., "form" matters more for top teams)
- Doesn't require feature scaling
- Provides feature importance rankings
- Resistant to overfitting with proper tuning

**The Training Process**:

1. **Load & Preprocess Data**

   ```python
   df = pd.read_csv("data/raw/all_seasons.csv")
   df = preprocess_data(df)  # Adds all features
   ```

2. **Time-Based Split** (Critical!)

   - Train on: 2019/20, 2020/21, 2021/22, 2022/23
   - Test on: 2023/24
   - **Why?** In sports, we can't shuffle data randomly—we must predict the future based on the past.

3. **Hyperparameter Optimization**

   ```python
   RandomizedSearchCV(
       n_estimators=[50, 100, 200],      # Number of trees
       max_depth=[None, 10, 20],          # Tree depth
       min_samples_split=[2, 5, 10]       # Min samples to split
   )
   ```

   - Tries different combinations
   - Uses Time Series Cross-Validation
   - Picks the best performing setup

4. **Train Final Model**
   - Best parameters: 200 trees, unlimited depth, min_samples_split=10
   - Trains on all training data
   - Saves to `models/rf_model.pkl`

**Output**:

```text
Model Accuracy: 54.5%
Baseline (Always Home): 46.1%
Improvement: +8.4%

Feature Importances:
HomeTeamCode       15.6%  ← Team identity matters most
AwayTeamCode       13.8%
AwayAvgGoals       12.8%  ← Recent scoring form
HomeAvgConceded    11.8%
...
```

**Run it**:

```bash
python src/train_model.py
```

---

### 4. Making Predictions (`src/predict.py`)

**What it does**: Predicts the outcome of a specific matchup by calculating real-time form from the latest available data.

**Line-by-Line Logic**:

1.  **Data Loading & Encoding**:
    - The script loads the trained model and the historical dataset.
    - Since team identifiers must match training, it recreates the `team_map` (e.g., `{'Arsenal': 5, ...}`) from the full historical record.
    ```python
    rf = joblib.load("models/rf_model.pkl")
    df = pd.read_csv("data/raw/all_seasons.csv")
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_map = {team: i for i, team in enumerate(all_teams)}
    ```

2.  **Dynamic Stat Retrieval (`get_latest_stats`)**:
    - For the input teams, the script scans the CSV for the **last 5 occurrences** of each team (whether they played at Home or Away).
    - It manually calculates points (3 for W, 1 for D, 0 for L) and goal averages for these 5 games to get the absolute latest "momentum" and "efficiency" metrics.
    ```python
    team_games = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(5)
    ```

3.  **The Inference Vector**:
    - It organizes these stats into a single-row DataFrame that matches the model's training features:
    ```python
    X = pd.DataFrame([{
        'HomeTeamCode': team_map[home_team],
        'AwayTeamCode': team_map[away_team],
        'HomeFormPoints': h_form,
        'HomeAvgGoals': h_att,
        'HomeAvgConceded': h_def,
        # ... and so on
    }])
    ```

4.  **Probability Calculation**:
    - Instead of a simple "Win/Loss", the model outputs a probability array:
    ```python
    probs = rf.predict_proba(X)[0] # e.g. [0.52, 0.23, 0.25]
    ```
    - The highest probability becomes the final "Predicted Result".

**Usage**:

```bash
python src/predict.py "Arsenal" "Chelsea"
```

**Output Example**:
```
Stats for Arsenal: Form=12, GF=2.4, GA=0.8
Stats for Chelsea: Form=7, GF=1.2, GA=1.6

Arsenal Win: 52.0%
Draw:       23.0%
Chelsea Win: 25.0%

Predicted Result: Arsenal
```

**Usage**:

```bash
python src/predict.py "Arsenal" "Chelsea"
python src/predict.py "Man City" "Liverpool"
```

**Note**: Team names must match exactly as they appear in the dataset (e.g., "Man City" not "Manchester City").

---

### 5. Backtesting Strategy (`src/backtest.py`)

**What it does**: Simulates a betting season to see if the model is profitable.

**The Betting Strategy**:

We use **Expected Value (EV)** betting:

```text
EV = (Model_Probability × Bookmaker_Odds) - 1
```

**Example**:

- Model says: Arsenal has 60% chance to win
- Bookmaker odds: 2.0 (implies 50% chance)
- EV = (0.60 × 2.0) - 1 = 0.20 = **+20% expected value**
- **Decision**: BET! (We have an edge)

**Betting Rules** (to avoid bankruptcy):

1. ✅ Only bet if EV > 10% (conservative threshold)
2. ✅ Ignore "long shots" (odds > 5.0) - models overpredict these
3. ✅ Flat bet: $25 per match (no crazy stakes)
4. ✅ Pick the outcome with highest EV

**The Simulation**:

```python
Starting Bankroll: $1,000
For each match in 2023/24 season:
    1. Calculate EV for Home, Draw, Away
    2. If best EV > 10% AND odds < 5.0:
         - Place $25 bet
         - Update bankroll based on result
    3. Track: wins, losses, bankroll
```

**Results**:

```text
Season: 2023/24
Total Bets: 201 / 380 matches (52.9%)
Win Rate: 29.9%
Final Bankroll: $810.50
ROI: -18.95%
RESULT: NOT PROFITABLE ❌
```

**Why not profitable?**

- **Market Efficiency**: Bookmaker odds are very accurate
- **Low Win Rate**: 29.9% isn't enough with average odds ~2.5
- **Draw Problem**: Model struggles with draws (only 10% recall)

**Run it**:

```bash
python src/backtest.py
```

---

## 📊 Performance Metrics

### Model Accuracy

| Metric              | Phase 1 (Baseline) | Phase 2 (Optimized) |
| ------------------- | ------------------ | ------------------- |
| Accuracy            | 52.1%              | **54.5%**           |
| Baseline (Home Win) | 46.1%              | 46.1%               |
| Improvement         | +6.0%              | **+8.4%**           |

### Betting Performance

| Strategy                            | Bets Placed   | Win Rate | ROI          |
| ----------------------------------- | ------------- | -------- | ------------ |
| Naive (5% EV threshold)             | 344/380 (90%) | 22.7%    | **-324%** 💀 |
| Conservative (10% EV, max odds 5.0) | 201/380 (53%) | 29.9%    | **-18.95%**  |

**Verdict**: The model beats random guessing and identifies value bets, but isn't profitable against the bookmakers yet. The betting market is extremely efficient.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone or download this project**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   This installs:

   - `pandas` - Data manipulation
   - `scikit-learn` - Machine learning
   - `numpy` - Numerical operations
   - `matplotlib` - Plotting (for future visualizations)
   - `seaborn` - Statistical visualizations
   - `requests` - Downloading data

### Quick Start

**Option 1: Run the full pipeline**

```bash
python run_pipeline.py
```

This will:

1. Download 5 seasons of EPL data (~2 minutes)
2. Train the optimized model (~30 seconds)
3. Save model to `models/rf_model.pkl`

**Option 2: Run components individually**

```bash
# 1. Download data
python src/data_loader.py

# 2. Train model
python src/train_model.py

# 3. Make predictions
python src/predict.py "Arsenal" "Chelsea"

# 4. Run backtest
python src/backtest.py
```

---

## 📁 Project Structure

```text
sports-pred-alg/
├── data/
│   └── raw/
│       ├── E0_1920.csv          # Season 2019/20
│       ├── E0_2021.csv          # Season 2020/21
│       ├── ...
│       └── all_seasons.csv      # Combined dataset
├── models/
│   └── rf_model.pkl             # Trained Random Forest
├── src/
│   ├── data_loader.py           # Downloads data
│   ├── feature_engineering.py   # Creates features
│   ├── train_model.py           # Trains & optimizes model
│   ├── predict.py               # Makes predictions
│   └── backtest.py              # Simulates betting
├── run_pipeline.py              # Runs full pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔬 The Math Behind It

### Random Forest Classifier

A Random Forest is an **ensemble** of decision trees. Here's how it works:

1. **Create 200 decision trees**
2. Each tree is trained on a random subset of data
3. Each tree votes on the outcome
4. Final prediction = majority vote

**Example**:

```text
Tree 1: Home Win
Tree 2: Home Win
Tree 3: Draw
Tree 4: Away Win
Tree 5: Home Win
...
Tree 200: Home Win

Votes: Home=120, Draw=40, Away=40
Probabilities: Home=60%, Draw=20%, Away=20%
```

### Expected Value (EV)

EV tells us if a bet is profitable in the long run:

```
EV = (Probability_of_Win × Profit_if_Win) - (Probability_of_Loss × Loss_if_Loss)
```

Simplified for betting:

```
EV = (Model_Prob × Odds) - 1
```

**Example**:

- Model: Arsenal 60% to win
- Odds: 2.5
- EV = (0.60 × 2.5) - 1 = 0.50 = **+50% EV** ✅ BET!

**If EV > 0**: Bet is profitable long-term  
**If EV < 0**: Bet loses money long-term

---

## 🎓 What You Can Learn From This

1. **Time Series ML**: How to properly split temporal data
2. **Feature Engineering**: Converting domain knowledge into features
3. **Hyperparameter Tuning**: Using RandomizedSearchCV
4. **Betting Strategy**: Expected Value and bankroll management
5. **Market Efficiency**: Why beating bookmakers is hard

---

## 🔮 Future Improvements

To make this profitable, you'd need:

### 1. Better Data

- **xG (Expected Goals)**: More accurate than raw goals
- **Player Data**: Injuries, suspensions, lineups
- **Head-to-Head**: Historical matchups
- **Weather**: Rain affects play style
- **Referee**: Some refs give more cards/penalties

### 2. Advanced Models

- **Gradient Boosting**: XGBoost, LightGBM (often better than Random Forest)
- **Neural Networks**: Can capture complex patterns
- **Ensemble**: Combine multiple models

### 3. Smarter Betting

- **Kelly Criterion**: Optimal bet sizing based on edge
- **Arbitrage**: Bet on multiple outcomes across bookmakers
- **In-Play Betting**: Adjust bets during the match
- **Specialize**: Focus on specific bet types (e.g., Over/Under goals)

### 4. More Leagues

- Train on multiple leagues (La Liga, Serie A, Bundesliga)
- More data = better patterns

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.

- Sports betting involves risk
- Past performance doesn't guarantee future results
- The model is currently **not profitable** (-18.95% ROI)
- Never bet money you can't afford to lose
- Gambling can be addictive - seek help if needed

---

## 🤝 Contributing

Ideas for improvement:

- Add more features (xG, player stats)
- Try different models (XGBoost, Neural Networks)
- Implement Kelly Criterion betting
- Add visualization dashboards
- Support more leagues

---

## 📄 License

This project is open source and available for educational use.

---

**Built with Python, Pandas, and Scikit-Learn**  
_Making sports prediction accessible and transparent_ ⚽📊
