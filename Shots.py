import pandas as pd
import numpy as np
from io import StringIO
from shot_helper_functions import shots_cleaner, Strokes_Gained_Category, relative_SG, detailed_category, shot_feature_engineering, missing_shots_handler

with open('C:/Users/Owner/OneDrive/Desktop/SportEdge/data/rshot.txt', 'r', encoding='utf-8', errors='ignore') as file:
    content = file.read()

raw_shot_stats = pd.read_csv(StringIO(content), delimiter=';')

# Clean
shot_stats = shots_cleaner(raw_shot_stats)

# Add Strokes Gained categories
shot_stats['SGCategory'] = shot_stats.apply(Strokes_Gained_Category, axis=1)

# Deal with initial missing data
shot_stats = missing_shots_handler(shot_stats)

# Compute relevant Strokes Gained metrics
relative_SG(shot_stats)
shot_stats['detailed_category'] = shot_stats.apply(detailed_category, axis=1)

# Feature Engineering
shot_stats = shot_feature_engineering(shot_stats)

# Assign Players to teams
US_players = [
    32102, 39977, 51766, 47504, 27644, 35450,
    48081, 50525, 33448, 34046, 46046, 36689
]
Int_players = [
    28089, 29926, 31646, 32839, 39997, 45157,
    48119, 33399, 37455, 39058, 39971
]

US_Team_shots = shot_stats[shot_stats['PlayerID'].isin(US_players)]
Int_Team_shots = shot_stats[shot_stats['PlayerID'].isin(Int_players)]

