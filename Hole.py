import pandas as pd
import numpy as np
from io import StringIO
from hole_helper_functions import hole_cleaner, hole_missing_data_handler, hole_feature_engineering

with open('C:/Users/Owner/OneDrive/Desktop/SportEdge/data/rhole.txt', 'r', encoding='utf-8', errors='ignore') as hole_file:
    content = hole_file.read()
raw_hole_stats = pd.read_csv(StringIO(content), delimiter=';')

# Clean & deal with missing data
hole_stats = hole_cleaner(raw_hole_stats)
hole_stats = hole_missing_data_handler(hole_stats)

# Feature Engineering
hole_stats = hole_feature_engineering(hole_stats)

# Assign Players to teams
US_players = [
    32102, 39977, 51766, 47504, 27644, 35450,
    48081, 50525, 33448, 34046, 46046, 36689
]
Int_players = [
    28089, 29926, 31646, 32839, 39997, 45157,
    48119, 33399, 37455, 39058, 39971
]
US_Team_holes = hole_stats['PlayerID'].isin(US_players)
Int_Team_holes = hole_stats['PlayerID'].isin(Int_players)

print(hole_stats['PlayerID'].isin(US_players).any())
print(hole_stats['PlayerID'].isin(Int_players).any())
