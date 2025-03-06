import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from collections import defaultdict
from src.utils.helpers import outer_default, inner_default
import json
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


class FootballTrainer:

    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.label_encoder = LabelEncoder()
        self.season_tables = defaultdict(outer_default)


    def _encode_teams(self):

        all_teams = pd.concat([self.df['HomeTeam'], self.df['AwayTeam']])
        self.label_encoder.fit(all_teams)
        self.df['HomeTeam'] = self.label_encoder.transform(self.df['HomeTeam'])
        self.df['AwayTeam'] = self.label_encoder.transform(self.df['AwayTeam'])


    def _calculate_season_stats(self):

        for _, row in self.df.iterrows():
            season = row['Season']
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            self.season_tables[season][home]['goals'] += row['FTH Goals']
            self.season_tables[season][home]['points'] += 3 if row['FT_Result'] == 1 else (1 if row['FT_Result'] == 0 else 0)
            
            self.season_tables[season][away]['goals'] += row['FTA Goals']
            self.season_tables[season][away]['points'] += 3 if row['FT_Result'] == -1 else (1 if row['FT_Result'] == 0 else 0)
            

    def _add_rank_features(self):

        self.df['Home_Team_Rank'] = self.df.apply(
            lambda row: self._get_team_rank(row['Season'], row['HomeTeam']), axis=1
        )
        self.df['Away_Team_Rank'] = self.df.apply(
            lambda row: self._get_team_rank(row['Season'], row['AwayTeam']), axis=1
        )
        

    def _get_team_rank(self, season, team):

        if season not in self.season_tables:
            return 0
        
        sorted_table = sorted(
            self.season_tables[season].items(),
            key=lambda x: (-x[1]['points'], -x[1]['goals'])
        )
        return [t[0] for t in sorted_table].index(team) + 1 if team in self.season_tables[season] else 0


    def _generate_report(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_importance(self.model, ax=ax)
        plt.savefig('analyse/feature_importance.png')
        plt.close()


    def prepare_features(self):
 
        self._encode_teams()
        
        self._calculate_season_stats()
        
        self._add_rank_features()
        
        features = [
            'Season', 'HomeTeam', 'AwayTeam',
            'HomeTeam_Elo', 'AwayTeam_Elo',
            'Home_Strength', 'Away_Strength',
            'H2H_Home_Wins', 'H2H_Away_Wins', 'H2H_Draws',
            'H2H_Avg_Home_Goals', 'H2H_Avg_Away_Goals',
            'Home_vs_Away_Wins', 'Home_vs_Away_Draws', 
            'Home_vs_Away_Avg_Goals',
            'Home_Team_Rank', 'Away_Team_Rank'
        ]
        
        return self.df[features], self.df['FT_Result']


    def _save_artifacts(self):
        meta = {
            'features': list(self.df.columns),
            'best_params': self.model.get_params(),
        }
        
        with open('analyse/metadata.json', 'w') as f:
            json.dump(meta, f)

    def train(self, test_size=0.2, random_state=42, tune_hyperparams=False):

        X, y = self.prepare_features()
        y = y.replace(-1, 2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if tune_hyperparams:
            param_grid = {
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7, 9, 15],
                'n_estimators': [100, 200, 300, 500],
                'subsample': [0.6, 0.8, 1.0]
            }
            
            grid_search = GridSearchCV(
                XGBClassifier(),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2
            )
            
            grid_search.fit(X_train, y_train)
            print("Best parameters:", grid_search.best_params_)
            print("Best CV accuracy:", grid_search.best_score_)
            
            self.model = grid_search.best_estimator_
        else:
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
        self.model.fit(X_train, y_train)

        self._save_artifacts()
        self._generate_report(X_test, y_test)
        
        joblib.dump(self.model, 'models/football_model.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.season_tables, 'models/season_tables.pkl')
        
        print(f"Training Accuracy: {self.model.score(X_train, y_train):.2f}")
        print(f"Test Accuracy: {self.model.score(X_test, y_test):.2f}")


if __name__ == "__main__":
    trainer = FootballTrainer('data/processed_matches.csv')
    trainer.train()