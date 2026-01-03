import pandas as pd
import numpy as np

def calculate_recent_form(df, last_n_games=5):
    """
    Calculates points from the last N games for both Home and Away teams.
    3 points for win, 1 for draw, 0 for loss.
    """
    # Create a long-form dataframe to calculate stats per team
    # This is often easier: Date, Team, Opponent, GF, GA, Result, Points
    
    # We need to map FTR to points for Home and Away
    df['HomePoints'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['AwayPoints'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    
    # Create a separate df for all matches from the perspective of the home team and away team
    # treating them as just "Team" and "Stats"
    
    home_df = df[['Date', 'HomeTeam', 'HomePoints', 'FTHG', 'FTAG']].rename(
        columns={'HomeTeam': 'Team', 'HomePoints': 'Points', 'FTHG': 'GoalsScored', 'FTAG': 'GoalsConceded'}
    )
    away_df = df[['Date', 'AwayTeam', 'AwayPoints', 'FTAG', 'FTHG']].rename(
        columns={'AwayTeam': 'Team', 'AwayPoints': 'Points', 'FTAG': 'GoalsScored', 'FTHG': 'GoalsConceded'}
    )
    
    team_stats = pd.concat([home_df, away_df]).sort_values(['Team', 'Date'])
    
    # Calculate rolling sums
    team_stats['FormPoints'] = team_stats.groupby('Team')['Points'].transform(
        lambda x: x.shift(1).rolling(window=last_n_games, min_periods=1).sum()
    ).fillna(0)
    
    team_stats['FormGoalsScored'] = team_stats.groupby('Team')['GoalsScored'].transform(
        lambda x: x.shift(1).rolling(window=last_n_games, min_periods=1).mean()
    ).fillna(0)
    
    team_stats['FormGoalsConceded'] = team_stats.groupby('Team')['GoalsConceded'].transform(
        lambda x: x.shift(1).rolling(window=last_n_games, min_periods=1).mean()
    ).fillna(0)
    
    # Merge back to original df
    # We need to merge for HomeTeam and AwayTeam separately
    
    # Merge for Home Team
    df = df.merge(
        team_stats[['Date', 'Team', 'FormPoints', 'FormGoalsScored', 'FormGoalsConceded']],
        left_on=['Date', 'HomeTeam'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={
        'FormPoints': 'HomeFormPoints',
        'FormGoalsScored': 'HomeAvgGoals',
        'FormGoalsConceded': 'HomeAvgConceded'
    }).drop(columns=['Team'])
    
    # Merge for Away Team
    df = df.merge(
        team_stats[['Date', 'Team', 'FormPoints', 'FormGoalsScored', 'FormGoalsConceded']],
        left_on=['Date', 'AwayTeam'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={
        'FormPoints': 'AwayFormPoints',
        'FormGoalsScored': 'AwayAvgGoals',
        'FormGoalsConceded': 'AwayAvgConceded'
    }).drop(columns=['Team'])
    
    return df

def preprocess_data(df):
    """
    Main function to clean and add features.
    """
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.dropna(subset=['FTR'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Add features
    df = calculate_recent_form(df)
    
    # Encode Target
    # H -> 0, D -> 1, A -> 2
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df['Target'] = df['FTR'].map(target_map)
    
    # Encode Teams
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_map = {team: i for i, team in enumerate(all_teams)}
    df['HomeTeamCode'] = df['HomeTeam'].map(team_map)
    df['AwayTeamCode'] = df['AwayTeam'].map(team_map)
    
    # Ensure Odds columns exist (fill with 1.0 if missing to avoid errors, though they should be there)
    for col in ['B365H', 'B365D', 'B365A']:
        if col not in df.columns:
            df[col] = 1.0
            
    return df
