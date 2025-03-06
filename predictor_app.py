import tkinter as tk
from tkinter import ttk
from src.predict import MatchPredictor

class FootballPredictorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Football Match Predictor")
        self.root.geometry("600x400")
        
        self.predictor = MatchPredictor()
        
        self.current_season = self.get_current_season()
        self.season_table = self.get_season_table()
        self.teams = self.get_sorted_teams()
        
        self.create_widgets()


    def get_current_season(self):
        return sorted(self.predictor.season_tables.keys())[-1]


    def get_season_table(self):
        decoder = self.predictor.label_encoder.inverse_transform
        raw_table = self.predictor.season_tables[self.current_season]
        
        decoded_table = {
            decoder([int(team_id)])[0]: stats 
            for team_id, stats in raw_table.items()
        }
        
        return sorted(
            decoded_table.items(),
            key=lambda x: (-x[1]['points'], -x[1]['goals'])
        )


    def get_sorted_teams(self):
        return [team[0] for team in self.season_table]


    def create_widgets(self):
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 12))
        style.configure('TButton', font=('Arial', 12))
        style.configure('TCombobox', font=('Arial', 11))

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        season_header = f"Current Season: {self.current_season}"
        ttk.Label(main_frame, text=season_header, font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(main_frame, text="Home Team:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.home_team = ttk.Combobox(main_frame, values=self.teams, state="readonly")
        self.home_team.grid(row=1, column=1, padx=10, pady=5, sticky=tk.EW)

        ttk.Label(main_frame, text="Away Team:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.away_team = ttk.Combobox(main_frame, values=self.teams, state="readonly")
        self.away_team.grid(row=2, column=1, padx=10, pady=5, sticky=tk.EW)

        predict_btn = ttk.Button(main_frame, text="Predict Match", command=self.make_prediction)
        predict_btn.grid(row=3, column=0, columnspan=2, pady=15)

        self.result_label = ttk.Label(main_frame, text="", font=('Arial', 14, 'bold'), 
                                    wraplength=500, justify=tk.CENTER)
        self.result_label.grid(row=4, column=0, columnspan=2)

        table_frame = ttk.Frame(main_frame)
        table_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        ttk.Label(table_frame, text="Pos", width=5).grid(row=0, column=0)
        ttk.Label(table_frame, text="Team", width=25).grid(row=0, column=1)
        ttk.Label(table_frame, text="Pts", width=10).grid(row=0, column=2)
        ttk.Label(table_frame, text="Goals", width=10).grid(row=0, column=3)

        for i, (team, stats) in enumerate(self.season_table, start=1):
            ttk.Label(table_frame, text=str(i)).grid(row=i, column=0)
            ttk.Label(table_frame, text=team).grid(row=i, column=1)  
            ttk.Label(table_frame, text=str(stats['points'])).grid(row=i, column=2)
            ttk.Label(table_frame, text=str(stats['goals'])).grid(row=i, column=3)

        main_frame.columnconfigure(1, weight=1)


    def get_current_rank(self, team_name):
        for position, (team, _) in enumerate(self.season_table, start=1):
            if team == team_name:
                return position
        return "N/A" 


    def make_prediction(self):
        home = self.home_team.get()
        away = self.away_team.get()
        
        if not home or not away:
            self.result_label.config(text="Please select both teams!", foreground="red")
            return
            
        if home == away:
            self.result_label.config(text="Teams must be different!", foreground="red")
            return
            
        try:
            result = self.predictor.predict(home, away)
            confidence = result['probability'] * 100
            prediction_text = (
                f"Prediction: {result['prediction']}\n"
                f"Confidence: {confidence:.1f}%\n"
                f"Home Rank: {self.get_current_rank(home)}"
                f"| Away Rank: {self.get_current_rank(away)}"
            )
            self.result_label.config(text=prediction_text, foreground="green")
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", foreground="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = FootballPredictorApp(root)
    root.mainloop()
