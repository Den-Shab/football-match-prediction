import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from src.utils.helpers import outer_default, inner_default
import os
from src.data_preprocessing import preprocess_data, generate_all_features

class MatchPredictor:

    def __init__(self, data_path='data/processed_matches.csv'):
        self.model = joblib.load('models/football_model.pkl')
        self.df = pd.read_csv(data_path)
        self.label_encoder = joblib.load('models/label_encoder.pkl')
        self.season_tables = joblib.load('models/season_tables.pkl')


    def _calculate_season_tables(self):
        tables = defaultdict(lambda: defaultdict(lambda: {'points': 0, 'goals': 0, 'matches': 0}))
        
        for _, row in self.df.iterrows():
            season = row['Season']
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            tables[season][home]['goals'] += row['FTH Goals']
            tables[season][home]['matches'] += 1
            tables[season][home]['points'] += 3 if row['FT_Result'] == 1 else (1 if row['FT_Result'] == 0 else 0)
            
            tables[season][away]['goals'] += row['FTA Goals']
            tables[season][away]['matches'] += 1
            tables[season][away]['points'] += 3 if row['FT_Result'] == -1 else (1 if row['FT_Result'] == 0 else 0)
            
        return tables


    def add_new_matches(self, new_matches_raw): #TODO
        pass


    def merge_and_reprocess(self): #TODO
        pass


    def _get_team_rank(self, season, team):
        if season not in self.season_tables:
            return 0
        
        sorted_table = sorted(
            self.season_tables[season].items(),
            key=lambda x: (-x[1]['points'], -x[1]['goals'])
        )
        return [t[0] for t in sorted_table].index(team) + 1 if team in self.season_tables[season] else 0


    def get_current_season(self):
        return sorted(self.season_tables.keys())[-1]


    def get_season_table(self, season):
        teams = self.season_tables[season]
        return sorted(teams.items(), key=lambda x: (-x[1]['points'], -x[1]['goals']))


    def get_latest_team_data(self, home_team, away_team):
        latest_matches = self.df.sort_values('Date')
        season = 2024

        home_data = latest_matches[latest_matches['HomeTeam'] == home_team].iloc[-1]
        away_data = latest_matches[latest_matches['AwayTeam'] == away_team].iloc[-1]

        h2h_matches = latest_matches[
            ((latest_matches['HomeTeam'] == home_team) & (latest_matches['AwayTeam'] == away_team)) |
            ((latest_matches['HomeTeam'] == away_team) & (latest_matches['AwayTeam'] == home_team))
        ]

        h2h_home_at_home_matches = latest_matches[
            (latest_matches['HomeTeam'] == home_team) & (latest_matches['AwayTeam'] == away_team)]
        
        teams = self.label_encoder.transform([home_team, away_team])
        return {
            "Season": season,
            "HomeTeam": teams[0],
            "AwayTeam": teams[1],
            'HomeTeam_Elo': home_data['HomeTeam_Elo'],
            'AwayTeam_Elo': away_data['AwayTeam_Elo'],
            'Home_Strength': home_data['Home_Strength'],
            'Away_Strength': away_data['Away_Strength'],
            'H2H_Home_Wins': len(h2h_matches[h2h_matches['FT_Result'] == 1]),
            'H2H_Away_Wins': len(h2h_matches[h2h_matches['FT_Result'] == -1]),
            'H2H_Draws': len(h2h_matches[h2h_matches['FT_Result'] == 0]),
            'H2H_Avg_Home_Goals': h2h_matches['FTH Goals'].mean() if not h2h_matches.empty else 0,
            'H2H_Avg_Away_Goals': h2h_matches['FTA Goals'].mean() if not h2h_matches.empty else 0,
            "Home_vs_Away_Wins": len(h2h_home_at_home_matches[h2h_home_at_home_matches['FT_Result'] == 1]),
            "Home_vs_Away_Draws": len(h2h_home_at_home_matches[h2h_home_at_home_matches['FT_Result'] == 0]),
            "Home_vs_Away_Avg_Goals": h2h_home_at_home_matches["FTH Goals"].mean() if not h2h_home_at_home_matches.empty else 0,
            'Home_Team_Rank': self._get_team_rank('2024/25', home_team),
            'Away_Team_Rank': self._get_team_rank('2024/25', away_team)
        }


    def predict(self, home_team, away_team):
        features = self.get_latest_team_data(home_team, away_team)
        feature_df = pd.DataFrame([features])
        
        prediction = self.model.predict(feature_df)[0]
        result_map = {0: 'Draw', 1: f'{home_team} Win', 2: f'{away_team} Win'}
        
        return {
            'prediction': result_map[prediction],
            'probability': self.model.predict_proba(feature_df)[0].max(),
            'features': features
        }


if __name__ == "__main__":
    predictor = MatchPredictor()
    
    test_matches = [
         ['Chelsea', 'Wolves'],
         ['Ipswich', 'Man City'],
         ['Man United', 'Brighton'],
         ["Nott'm Forest", 'Southampton'],
         ['Everton', 'Tottenham'],
         ['Arsenal', 'Aston Villa'],
         ['Brentford', 'Liverpool'],
         ['Leicester', 'Fulham'],
         ['Newcastle', 'Bournemouth'],
         ['West Ham', 'Crystal Palace']
    ]
    
    for home, away in test_matches:
        result = predictor.predict(home, away)
        print(f"{home} vs {away}: {result['prediction']} (Confidence: {result['probability']:.2f})")