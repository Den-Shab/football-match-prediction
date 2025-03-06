from collections import defaultdict, deque
import numpy as np


def calculate_team_form(df, window=5):

    team_form = {team: deque(maxlen=window) for team in set(df['HomeTeam']).union(set(df['AwayTeam']))}
    
    home_strength = []
    away_strength = []
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        result = row['FT_Result']
        
        home_strength.append(np.mean([r*elo for r, elo in team_form[home]]) if team_form[home] else 0)
        team_form[home].append((result, row['AwayTeam_Elo']))
        
        away_strength.append(np.mean([r*elo for r, elo in team_form[away]]) if team_form[away] else 0)
        team_form[away].append((-result, row['HomeTeam_Elo']))
    
    df['Home_Strength'] = home_strength
    df['Away_Strength'] = away_strength
    return df


def generate_h2h_features(df):
    h2h_stats = defaultdict(lambda: {'H_wins': 0, 'A_wins': 0, 'Draws': 0})
    home_meetings = defaultdict(lambda: {'H_wins': 0, 'Draws': 0, 'H_goals': 0, 'Matches': 0})
    
    h2h_home_wins = []
    h2h_away_wins = []
    h2h_draws = []
    home_vs_away_wins = []
    home_vs_away_draws = []
    home_vs_away_avg_goals = []
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        key = tuple(sorted([home, away]))
        
        h2h_home_wins.append(h2h_stats[key]['H_wins'])
        h2h_away_wins.append(h2h_stats[key]['A_wins'])
        h2h_draws.append(h2h_stats[key]['Draws'])

        home_vs_away_wins.append(home_meetings[(home, away)]['H_wins'])
        home_vs_away_draws.append(home_meetings[(home, away)]['Draws'])
        matches_home = home_meetings[(home, away)]['Matches']
        home_vs_away_avg_goals.append(home_meetings[(home, away)]['H_goals'] / matches_home if matches_home > 0 else 0)
        
        result = row['FT_Result']
        if result == 1:
            h2h_stats[key]['H_wins'] += 1
        elif result == -1:
            h2h_stats[key]['A_wins'] += 1
        else:
            h2h_stats[key]['Draws'] += 1
    
    df['H2H_Home_Wins'] = h2h_home_wins
    df['H2H_Away_Wins'] = h2h_away_wins
    df['H2H_Draws'] = h2h_draws

    df['Home_vs_Away_Wins'] = home_vs_away_wins
    df['Home_vs_Away_Draws'] = home_vs_away_draws
    df['Home_vs_Away_Avg_Goals'] = home_vs_away_avg_goals
    
    return df