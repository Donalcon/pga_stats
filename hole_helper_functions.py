import numpy as np
import pandas as pd

def hole_cleaner(raw_hole_stats):
    """
    Cleans Hole level stats, change col names, drop irellevant cols, formats types/
    """
    raw_hole_stats = raw_hole_stats.rename(columns={col: col.replace('#', '') for col in raw_hole_stats.columns})
    raw_hole_stats.columns = raw_hole_stats.columns.str.strip()
    raw_hole_stats.rename(columns={
        'Player': 'PlayerID',
        'Tournament Schedule': 'TournID',
        'Tournament Year': 'Year',
        'Course': 'CourseID',
        'Actual Yard': 'Yardage',
        'Hit Fwy': 'Fairway',
        'Hit Green': 'GIR',
        'Driving Distance (rounded)': 'DrivingDistance',
        'Tee Shot Landing Loc': 'TeeShotFinishLie',
        'Tee Shot Detail Landing Loc': 'DetailedTeeShotFinishLie',
        'RTP Score': 'ScoreToPar',
        'Score': 'HoleScore',
        'Shot': 'ShotNo',
        'Appr Shot Dist to the Pin': 'AppDistance',
        'Appr Shot Prox to the Hole': 'AppProx',
        'Appr Shot Landing Loc': 'AppShotFinishLie',
        'OTT Strokes Gained': 'SGOTT',
        'APP Strokes Gained': 'SGAPP',
        'ARG Strokes Gained': 'SGARG',
        'Putts Gained': 'SGPutt',
    }, inplace=True)
    raw_hole_stats['EventID'] = raw_hole_stats['Year'].astype(str) + raw_hole_stats['TournID'].astype(str)
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
        'ScoreToPar',
        'Fairway',
        'GIR',
        'DrivingDistance',
        'TeeShotFinishLie',
        'AppDistance',
        'AppProx',
        'AppShotFinishLie',
        'SGOTT',
        'SGAPP',
        'SGARG',
        'SGPutt'
    ]
    raw_hole_stats = raw_hole_stats.loc[:, relevant_features]
    # Create a dictionary to map column names to their desired datatypes
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
        'HoleScore': 'Int64',
        'ScoreToPar': 'Int64',
        'Fairway': 'bool',
        'GIR': 'bool',
        'DrivingDistance': 'Int64',
        'TeeShotFinishLie': 'str',
        'AppDistance': 'Int64',
        'AppProx': 'Int64',
        'AppShotFinishLie': 'str',
        'SGOTT': 'float64',
        'SGAPP': 'float64',
        'SGARG': 'float64',
        'SGPutt': 'float64',
    }
    raw_hole_stats = raw_hole_stats.astype(column_types)
    hole_stats = raw_hole_stats
    return hole_stats


def zero_handler(df, column):
    """
    Function for checking percent of row entries = 0 in a given column.
    """
    # total entries for each group
    total_count = df.groupby(['EventID', 'Round', 'CourseID', 'Hole']).size()
    count_zeros = df[df[column] == 0].groupby(['EventID', 'Round', 'CourseID', 'Hole']).size()
    percentage_zeros = (count_zeros.sum() / total_count.sum()) * 100
    return percentage_zeros

def filter_sg(row):
    """
    filters SG for missing data
    """
    if abs(row) > 0.5:
        return np.nan
    else:
        return row  # Use 'row' instead of 'x'


def hole_missing_data_handler(df):
    """
    Function for handling missing data. Cycles through relevant columns, if any have less than 10% 0's, we change to nan.
    """
    # Set DD to nan for all Par 3's
    df.loc[df['Par'] == 3, 'DrivingDistance'] = np.nan
    # Additional logic for AppProx
    df.loc[(df['AppShotFinishLie'] != 'Hole') & (df['AppProx'] == 0), 'AppProx'] = np.nan
    # Cycle through cols checking for 0's, amend if necessary
    features_to_check = [
        'DrivingDistance',
        'Yardage',
        'SGOTT',
        'SGAPP',
        'SGARG',
        'SGPutt',
        'AppProx'
    ]
    for feature in features_to_check:
        percentage_zeros = zero_handler(df, feature)
        if percentage_zeros > 10:
            print(f"Replacing zeros with NaN in column: {feature}")
            df.loc[df[feature] == 0, feature] = np.nan
    SGcategories = ['SGOTT', 'SGAPP', 'SGARG', 'SGPutt']
    for category in SGcategories:
        df[category] = df[category].apply(filter_sg)
    return df


def calculate_bool_avg(group, column):
    """
    Calculates Average % for boolean cols like Fairways and Greens In Regulation
    """
    total = len(group)
    hit = len(group[group[column] == True])
    return hit / total if total != 0 else 0


def categorize_hole_lengths(df, yardage_col='Yardage'):
    """
    Categorize hole lengths by Yardage. Bins increment by 50 yards.
    """
    # Define the yardage bins and their labels
    bins = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, float('inf')]
    labels = ['100-150', '151-200', '201-250', '251-300', '301-350', '351-400', '401-450', '451-500', '501-550',
              '551-600', '601-650', '650+']

    # Create a new column with the categorized yardages
    df['HoleLengthCategory'] = pd.cut(df[yardage_col], bins=bins, labels=labels, right=False, include_lowest=True)

    return df

def extreme_percentage_cols(df, column):
    """
    Takes in cols where values are percenatages, betwwon 0 and 1. Changes groups where stat is > 10% or < 90%
    """
    df.loc[(df[column] > 0) & (df[column] < 0.1), column] = np.nan
    df.loc[(df[column] > 0.9) & (df[column] < 1), column] = np.nan


def hole_feature_engineering(hole_stats):
    """
    Computes Averages, relative to average and other features.
    """
    # Average & Relatives
    hole_stats['HoleAvg'] = hole_stats.groupby(['EventID', 'Hole'])['HoleScore'].transform('mean')
    hole_stats['Vs_HoleAvg'] = hole_stats['HoleScore'] - hole_stats['HoleAvg']
    hole_stats['DD_Avg'] = hole_stats.groupby(['EventID', 'Hole'])['DrivingDistance'].transform('mean')
    hole_stats['Vs_DDAvg'] = hole_stats['DrivingDistance'] - hole_stats['DD_Avg']

    # Strokes Gained Categories
    hole_stats['SGOTT_avg'] = hole_stats.groupby(['EventID', 'Hole'])['SGOTT'].transform('mean')
    hole_stats['Vs_SGOTT_avg'] = hole_stats['SGOTT'] - hole_stats['SGOTT_avg']
    hole_stats['SGAPP_avg'] = hole_stats.groupby(['EventID', 'Hole'])['SGAPP'].transform('mean')
    hole_stats['Vs_SGAPP_avg'] = hole_stats['SGAPP'] - hole_stats['SGAPP_avg']
    hole_stats['SGARG_avg'] = hole_stats.groupby(['EventID', 'Hole'])['SGARG'].transform('mean')
    hole_stats['Vs_SGARG_avg'] = hole_stats['SGARG'] - hole_stats['SGARG_avg']
    hole_stats['SGPutt_avg'] = hole_stats.groupby(['EventID', 'Hole'])['SGPutt'].transform('mean')
    hole_stats['Vs_SGPutt_avg'] = hole_stats['SGPutt'] - hole_stats['SGPutt_avg']

    # Fairway Accuracy
    FairwayAvg = hole_stats.groupby(['EventID', 'Hole']).apply(calculate_bool_avg, column='Fairway')
    # Align with the original index & merge
    FairwayAvg = FairwayAvg.reset_index(name='FairwayAvg')
    hole_stats = pd.merge(hole_stats, FairwayAvg, on=['EventID', 'Hole'], how='left')
    hole_stats['FairwayAvg'].replace(0, np.nan, inplace=True)
    extreme_percentage_cols(hole_stats, 'FairwayAvg')
    hole_stats['RelativeFairway'] = np.where(hole_stats['Fairway'] == True,
                                             1 - hole_stats['FairwayAvg'],
                                             - hole_stats['FairwayAvg'])
    hole_stats['RelativeFairway'].replace(0, np.nan, inplace=True)

    # Greens In Regulation
    GIRavg = hole_stats.groupby(['EventID', 'Hole']).apply(calculate_bool_avg, column='GIR')
    GIRavg = GIRavg.reset_index(name='GIRavg')
    hole_stats = pd.merge(hole_stats, GIRavg, on=['EventID', 'Hole'], how='left')
    # Accounting for missing data
    hole_stats['GIRavg'].replace(0, np.nan, inplace=True)
    extreme_percentage_cols(hole_stats, 'GIRavg')
    hole_stats['RelativeGIR'] = np.where(hole_stats['GIR'] == True,
                                             1 - hole_stats['GIRavg'],
                                             - hole_stats['GIRavg'])
    hole_stats['RelativeGIR'].replace(0, np.nan, inplace=True)

    # Hole Length Group
    hole_stats = categorize_hole_lengths(hole_stats)
    return hole_stats

