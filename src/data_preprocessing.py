import pandas as pd
import numpy as np
from collections import defaultdict, deque
from src.utils.elo import update_elo, initialize_elo
from src.utils.feature_engineering import calculate_team_form, generate_h2h_features


def preprocess_data(input_path):

    df = pd.read_csv(input_path)
    
    df['HomeTeam'] = df['HomeTeam'].replace({
        'Brighton & Hove Albion': 'Brighton',
        'Ipswich Town': 'Ipswich'
    })
    df['AwayTeam'] = df['AwayTeam'].replace({
        'Brighton & Hove Albion': 'Brighton',
        'Ipswich Town': 'Ipswich'
    })

    df = df[['Date', 'Season', 'HomeTeam', 'AwayTeam', 
            'FTH Goals', 'FTA Goals', 'FT Result']]
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')
    
    result_mapping = {'H': 1, 'D': 0, 'A': -1}
    df['FT_Result'] = df['FT Result'].map(result_mapping)
    
    return df


def generate_all_features(df):
    
    df = initialize_elo(df)
    
    df = calculate_team_form(df)
    
    df = generate_h2h_features(df)
    
    df = calculate_h2h_goals(df)
    
    df = add_team_ranks(df)

    df['Season'] = df['Season'].str[:4].astype(int)
    
    return df


def calculate_h2h_goals(df):
    h2h_goals = defaultdict(lambda: {'home_goals': [], 'away_goals': []})
    
    for idx, row in df.iterrows():
        key = tuple(sorted([row['HomeTeam'], row['AwayTeam']]))
        h2h_goals[key]['home_goals'].append(row['FTH Goals'])
        h2h_goals[key]['away_goals'].append(row['FTA Goals'])
    
    home_avg_goals = []
    away_avg_goals = []
    
    for idx, row in df.iterrows():
        key = tuple(sorted([row['HomeTeam'], row['AwayTeam']]))
        home_avg = np.mean(h2h_goals[key]['home_goals'][:-1]) if len(h2h_goals[key]['home_goals']) > 1 else 0
        away_avg = np.mean(h2h_goals[key]['away_goals'][:-1]) if len(h2h_goals[key]['away_goals']) > 1 else 0
        
        home_avg_goals.append(home_avg)
        away_avg_goals.append(away_avg)
    
    df['H2H_Avg_Home_Goals'] = home_avg_goals
    df['H2H_Avg_Away_Goals'] = away_avg_goals
    
    return df


def add_team_ranks(df):
    season_tables = defaultdict(lambda: defaultdict(lambda: {'points': 0, 'goals': 0}))
    
    for idx, row in df.iterrows():
        season = row['Season']
        
        season_tables[season][row['HomeTeam']]['points'] += 3 if row['FT_Result'] == 1 else (1 if row['FT_Result'] == 0 else 0)
        season_tables[season][row['HomeTeam']]['goals'] += row['FTH Goals']
        
        season_tables[season][row['AwayTeam']]['points'] += 3 if row['FT_Result'] == -1 else (1 if row['FT_Result'] == 0 else 0)
        season_tables[season][row['AwayTeam']]['goals'] += row['FTA Goals']
    
    home_ranks = []
    away_ranks = []
    
    for idx, row in df.iterrows():
        season = row['Season']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        sorted_table = sorted(
            season_tables[season].items(),
            key=lambda x: (-x[1]['points'], -x[1]['goals'])
        )
        
        home_rank = [t[0] for t in sorted_table].index(home_team) + 1 if home_team in season_tables[season] else 0
        away_rank = [t[0] for t in sorted_table].index(away_team) + 1 if away_team in season_tables[season] else 0
        
        home_ranks.append(home_rank)
        away_ranks.append(away_rank)
    
    df['Home_Team_Rank'] = home_ranks
    df['Away_Team_Rank'] = away_ranks
    
    return df


if __name__ == "__main__":
    raw_df = preprocess_data('data/England_CSV.csv')
    
    processed_df = generate_all_features(raw_df)
    
    processed_df.to_csv('data/processed_matches.csv', index=False)
    print("Data preprocessing completed!")