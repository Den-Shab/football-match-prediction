def update_elo(elo, home, away, result, k=30):
    R_home = 10 ** (elo[home] / 400)
    R_away = 10 ** (elo[away] / 400)

    E_home = R_home / (R_home + R_away)
    E_away = R_away / (R_home + R_away)

    S_home = 1 if result == 1 else (0.5 if result == 0 else 0)
    S_away = 1 if result == -1 else (0.5 if result == 0 else 0)

    elo[home] += k * (S_home - E_home)
    elo[away] += k * (S_away - E_away)
    return elo


def initialize_elo(df):
    elo_ratings = {team: 1500 for team in set(df['HomeTeam']).union(set(df['AwayTeam']))}
    
    home_elo = []
    away_elo = []
    
    for _, row in df.iterrows():
        home, away, result = row['HomeTeam'], row['AwayTeam'], row['FT_Result']
        elo_ratings = update_elo(elo_ratings, home, away, result)
        home_elo.append(elo_ratings[home])
        away_elo.append(elo_ratings[away])
    
    df['HomeTeam_Elo'] = home_elo
    df['AwayTeam_Elo'] = away_elo
    return df