# Behind the Scenes: EPL Prediction Engine

This document provides a technical breakdown of the English Premier League (EPL) prediction system, focusing on the data pipeline, model architecture, and inference logic.

---

## Technical Architecture

The system is built on a modular pipeline where data flows through three core scripts:

1. **feature_engineering.py**: Data transformation and signal creation.
2. **train_model.py**: Model optimization and historical learning.
3. **predict.py**: Real-time stat retrieval and outcome inference.

---

## 1. Feature Engineering (feature_engineering.py)

The predictive power of the model comes from its features. Instead of raw team names, we use metrics that represent team performance and momentum.

### Rolling Window Approach
We calculate statistics over a "rolling window" of the last 5 matches.

*   **Logic:** For any given match, we look at the 5 matches immediately preceding it.
*   **Data Leakage Prevention:** We use the `.shift(1)` function in Pandas. This ensures that the statistics for a match on Saturday only include data from games that happened *before* Saturday.
*   **Metrics Created:**
    *   **FormPoints:** Cumulative points (Win=3, Draw=1, Loss=0) over 5 games.
    *   **AvgGoals:** Average goals scored (Attacking Strength).
    *   **AvgConceded:** Average goals allowed (Defensive Weakness).

### Encoding
Teams are mapped to unique integers (Team Codes) to allow the Random Forest algorithm to treat them as distinct categorical entities.

---

## 2. Model Training (train_model.py)

The engine uses a **Random Forest Classifier**, an ensemble method that combines multiple decision trees to improve accuracy and control over-fitting.

### Time-Based Splitting
In sports prediction, standard random shuffling is invalid. We must predict the future based on the past.
*   The script splits data by season.
*   It trains on several past seasons and validates performance on the most recent completed season.

### Hyperparameter Optimization
We use `RandomizedSearchCV` to fine-tune the model.
*   **n_estimators:** Number of trees in the forest (usually 100-200).
*   **max_depth:** Controls how complex each individual tree can become.
*   **min_samples_split:** Prevents the model from creating rules that only apply to a tiny number of matches, which helps generalization.

---

## 3. Prediction Logic (predict.py)

The prediction script bridges the gap between historical training and current matchups.

### Real-Time Stat Retrieval
When a user requests a prediction for "Arsenal vs Chelsea", the script:
1.  Filters the entire historical dataset for every instance where Arsenal and Chelsea appeared.
2.  Selects the **5 most recent** matches for each team.
3.  Calculates their current form and goal averages manually to create a "Live Feature Vector."

### Probability Inference
The model doesn't just return a winner; it returns a probability distribution:
*   Instead of `predict()`, we use `predict_proba()`.
*   This allows the user to see the internal confidence of the model (e.g., a 55% Home Win vs a 20% Draw).

---

## How to Explain the System

Use these frameworks to describe the project to different audiences:

### Technical Audience
"The system utilizes a Random Forest Classifier trained on engineered features from a 5-game rolling window. To prevent data leakage, we implement a shifted temporal window and utilize a strictly chronological season-based split for validation. The model outputs class probabilities rather than hard labels to allow for expected value analysis."

### Non-Technical Audience
"The project acts as a statistical historical analyst. It looks at how two teams have played in their last few games—specifically their scoring ability and defensive strength. It then compares those performance patterns against thousands of matches from previous years to see which outcome (Home Win, Draw, or Away Win) is most likely based on history."

---

## Summary of Mechanics
- **Data Source:** Historical match results (scores and betting odds).
- **Core Signal:** Momentum and efficiency calculated over recent performance.
- **Model Engine:** 200 Decision Trees voting on the most probable result.
- **Goal:** To identify outcomes where statistical probability deviates from public perception.
