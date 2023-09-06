import numpy as np
import pandas as pd


def shots_cleaner(raw_shot_stats):
    """
    Function for cleaning shots data. Changes column names & Types. Drops irrelevant cols.
    """
    raw_shot_stats = raw_shot_stats.rename(columns={col: col.replace('#', '') for col in raw_shot_stats.columns})
    raw_shot_stats.columns = raw_shot_stats.columns.str.strip()
    raw_shot_stats['EventID'] = raw_shot_stats['Year'].astype(str) + raw_shot_stats['Tourn.'].astype(str)
    raw_shot_stats.rename(columns={
        'of Strokes': 'Strokes',
        'Tourn.': 'TournID',
        'Player': 'PlayerID',
        'Course': 'CourseID',
        'Hole Score': 'HoleScore',
        'Shot': 'ShotNo',
        'Shot Type(S/P/D)': 'ShotType',
        'Distance': 'ShotDistance',
        'Distance to Pin': 'FromDistance',
        'From Location(Scorer)': 'FromLie',
        'Distance to Hole after the Shot': 'ToDistance',
        'Par Value': 'Par',
        'To Location(Scorer)': 'ToLie',
        'Distance from Center': 'DistanceFromCentre',
        'Left/Right': 'LeftRight',
        'Strokes Gained/Baseline': 'SGBaseline',
        'In the Hole Flag': 'InHoleFlag'
    }, inplace=True)
    relevant_features = [
        'Year',
        'TournID',
        'PlayerID',
        'CourseID',
        'EventID',
        'Round',
        'Hole',
        'Par',
        'Yardage',
        'HoleScore',
        'ShotNo',
        'ShotType',
        'Strokes',
        'FromLie',
        'ToLie',
        'ShotDistance',
        'FromDistance',
        'InHoleFlag',
        'ToDistance',
        'DistanceFromCentre',
        'LeftRight',
        'SGBaseline',
    ]
    raw_shot_stats = raw_shot_stats.loc[:, relevant_features]
    pd.to_numeric(raw_shot_stats['HoleScore'], errors='coerce')
    column_types = {
        'Year': 'Int64',
        'TournID': 'str',
        'PlayerID': 'str',
        'CourseID': 'str',
        'EventID': 'str',
        'Round': 'Int64',
        'Hole': 'Int64',
        'Par': 'Int64',
        'Yardage': 'Int64',
        'ShotType': 'str',
        'ShotNo': 'Int64',
        'FromLie': 'str',
        'ShotDistance': 'Int64',
        'InHoleFlag': 'str',
        'ToDistance': 'Int64',
        'DistanceFromCentre': 'Int64',
        'LeftRight': 'str',
        'SGBaseline': 'float64',
        'Strokes': 'Int64',
    }
    raw_shot_stats = raw_shot_stats.astype(column_types)
    shot_stats = raw_shot_stats
    return shot_stats

def filter_sg(x):
    return np.where((x > 0.5) | (x < -0.5), np.nan, x)

def missing_shots_handler(shot_stats):
    shot_stats.loc[shot_stats['ShotDistance'] == 0, 'ShotDistance'] = np.nan
    shot_stats.loc[shot_stats['FromDistance'] == 0, 'FromDistance'] = np.nan
    # Replace 0s in ToDistance with NaN, unless InHoleFlag is 'Y'
    condition = (shot_stats['ToDistance'] == 0) & (shot_stats['InHoleFlag'] != 'Y')
    shot_stats.loc[condition, 'ToDistance'] = np.nan
    shot_stats['SGBaseline'] = shot_stats.groupby(['SGCategory', 'CourseID', 'Round', 'EventID', 'Hole'])['SGBaseline'].transform(
        filter_sg)
    return shot_stats

def Strokes_Gained_Category(row):
    """
    Assigns an SG category to each shot
    """
    if row['Par'] >= 4 and row['ShotNo'] == 1:
        return 'Off the Tee'
    elif row['FromDistance'] > 1800:
        return 'Approach'
    elif row['FromDistance'] <= 1800 and row['FromLie'] != 'Green':
        return 'Around the Green'
    elif row['FromLie'] == 'Green':
        return 'Putt'
    else:
        return 'Other'


def relative_SG(shot_stats):
    """
    Finds Average SG & Adjusted SG for each player, also filters missing data 
    """
    AvgSG = shot_stats.groupby(['SGCategory', 'CourseID', 'Round', 'EventID', 'Hole'])['SGBaseline'].transform('mean')
    shot_stats['AvgSG'] = AvgSG
    shot_stats['AdjSG'] = shot_stats['SGBaseline'] - shot_stats['AvgSG']
    return shot_stats

def detailed_category(row):
    if pd.isna(row['SGCategory']) or pd.isna(row['ToDistance']):
        return 'Other'
    if row['SGCategory'] == 'Off the Tee':
        return 'OTT Long' if row['ToDistance'] >= 10080 else 'OTT Short'
    elif row['SGCategory'] == 'Approach':
        if 1800 <= row['ToDistance'] < 3600:
            return 'App50-100'
        elif 3600 <= row['ToDistance'] < 5400:
            return 'App100-150'
        elif 5400 <= row['ToDistance'] < 7200:
            return 'App150-200'
        elif 7200 <= row['ToDistance'] < 9000:
            return 'App200-250'
        elif row['ToDistance'] >= 9000:
            return 'App250+'
    elif row['SGCategory'] == 'Around the Green':
        if 900 <= row['ToDistance'] < 1800:
            return 'ARG25-50'
        elif 0 <= row['ToDistance'] < 900:
            return 'ARG0-25'
    elif row['SGCategory'] == 'Putt':
        if 0 <= row['ToDistance'] < 6:
            return 'Putt0-6'
        elif 6 <= row['ToDistance'] < 15:
            return 'Putt6-15'
        elif 15 <= row['ToDistance'] < 30:
            return 'Putt15-30'
        elif row['ToDistance'] >= 30:
            return 'Putt30+'
    return 'Other'

def mean_skipna(x):
    """
    Used in conjunction with transform to skip rows with nan
    """
    return x.dropna().mean()

def shot_feature_engineering(shot_stats):
    """

    """
    # Average Scores
    shot_stats['HoleScore'].replace('', np.nan, inplace=True)
    shot_stats['HoleScore'] = pd.to_numeric(shot_stats['HoleScore'], errors='coerce')
    shot_stats['HoleAvg'] = shot_stats.groupby(['EventID', 'Hole'])['HoleScore'].transform('mean').round(0)
    shot_stats['Vs_HoleAvg'] = shot_stats['HoleScore'] - shot_stats['HoleAvg']

    # Round
    shot_stats['RoundScore'] = shot_stats.groupby(['EventID', 'PlayerID', 'Round'])['Strokes'].transform('sum')
    shot_stats['RoundAvg'] = shot_stats.groupby(['EventID', 'Round'])['RoundScore'].transform('mean')
    shot_stats['Vs_RoundAvg'] = shot_stats['RoundScore'] - shot_stats['RoundAvg']

    # Event Average, wouldn't this be the same as course average?
    shot_stats['EventAvg'] = shot_stats.groupby(['EventID'])['RoundScore'].transform('mean')
    shot_stats['Vs_EventAvg'] = shot_stats['RoundScore'] - shot_stats['EventAvg']

    # Compare against the field
    shot_stats['Vs_Field'] = np.nan
    # Driving Distance
    shot_stats['DrivingDistance'] = np.where(shot_stats['SGCategory'] == 'Off the Tee', shot_stats['ShotDistance'],
                                             None)
    shot_stats['DD_Avg'] = shot_stats.groupby(['EventID', 'Hole'])['DrivingDistance'].transform('mean')
    shot_stats.loc[shot_stats['SGCategory'] == 'Off the Tee', 'Vs_Field'] = shot_stats['DrivingDistance'] - shot_stats[
        'DD_Avg']
    # Fairway found flagged by 1's, so we can sum them for average
    shot_stats['Fairway'] = np.where((shot_stats['SGCategory'] == 'Off the Tee') & (shot_stats['ToLie'] == 'Fairway'),
                                     1, 0)
    subset_df = shot_stats[shot_stats['SGCategory'] == 'Off the Tee']
    fairway_percent = subset_df.groupby(['EventID', 'Hole']).apply(
        lambda x: (x['Fairway'].sum() / len(x)) * 100).reset_index(name='FairwayAvg')
    shot_stats = pd.merge(shot_stats, fairway_percent, how='left', on=['EventID', 'Hole'])
    shot_stats.loc[shot_stats['SGCategory'] != 'Off the Tee', 'FairwayAvg'] = None

    # Approach
    app_relevant_categories = ['App50-100', 'App100-150', 'App150-200', 'App200-250', 'App250+']
    subset_df = shot_stats[shot_stats['detailed_category'].isin(app_relevant_categories)]
    Avg_Approach = subset_df.groupby(['EventID', 'Hole', 'detailed_category'])['ToDistance'].mean().reset_index(
        name='Avg_Approach')
    shot_stats = pd.merge(shot_stats, Avg_Approach, how='left', on=['EventID', 'Hole', 'detailed_category'])
    shot_stats.loc[shot_stats['detailed_category'].isin(app_relevant_categories), 'Vs_Field'] = shot_stats[
                                                                                                    'ToDistance'] - \
                                                                                                shot_stats[
                                                                                                    'Avg_Approach']

    # Around the Green
    arg_relevant_categories = ['ARG25-50', 'ARG0-25']
    subset_df = shot_stats[shot_stats['detailed_category'].isin(arg_relevant_categories)]
    Avg_ARG = subset_df.groupby(['EventID', 'Hole', 'detailed_category'])['ToDistance'].mean().reset_index(
        name='Avg_ARG')
    shot_stats = pd.merge(shot_stats, Avg_ARG, how='left', on=['EventID', 'Hole', 'detailed_category'])
    shot_stats.loc[shot_stats['detailed_category'].isin(arg_relevant_categories), 'Vs_Field'] = shot_stats[
                                                                                                    'ToDistance'] - \
                                                                                                shot_stats['Avg_ARG']

    # Putting
    Putt_relevant_categories = ['Putt0-6', 'Putt6-15', 'Putt15-30', 'Putt30+']
    subset_df = shot_stats[shot_stats['detailed_category'].isin(Putt_relevant_categories)]
    Avg_Putt = subset_df.groupby(['EventID', 'Hole', 'detailed_category'])['ToDistance'].mean().reset_index(
        name='Avg_Putt')
    shot_stats = pd.merge(shot_stats, Avg_Putt, how='left', on=['EventID', 'Hole', 'detailed_category'])
    shot_stats.loc[shot_stats['detailed_category'].isin(Putt_relevant_categories), 'Vs_Field'] = shot_stats[
                                                                                                     'ToDistance'] - \
                                                                                                 shot_stats['Avg_Putt']
    # Drop irrelevant columns only used for calculation
    shot_stats.drop(['DrivingDistance', 'DD_Avg', 'Avg_Putt', 'Avg_ARG', 'Avg_Approach'], axis='columns', inplace=True)
    return shot_stats
