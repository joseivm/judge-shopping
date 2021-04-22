import pandas as pd
import numpy as np
import os
import datetime
import re

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
county_file = PROJECT_DIR + '/data/raw/county_list.csv'
processed_daily_schedule_file = PROJECT_DIR + '/data/processed/daily_schedule_data.csv'
processed_sentencing_data_file = PROJECT_DIR + '/data/raw/sentencing_data.csv'
judge_name_id_mapping_file = PROJECT_DIR + '/data/processed/judge_name_id_mapping.csv'
day_names = {'1':'Monday','2':'Tuesday','3':'Wednesday','4':'Thursday','5':'Friday',
'6':'Saturday','7':'Sunday'}
weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# Output files

sdf = pd.read_csv(processed_sentencing_data_file)
cdf = pd.read_csv(processed_daily_schedule_file)
sdf['Plea'] = (sdf['Trial']+1)%2
sdf_cols = ['Date','County','Circuit','JudgeID','Trial','Week','JudgeName','Plea','EventID']
sdf = sdf[sdf_cols]

def discrepancy_table(sdf,cdf,assignment_type):
    scdf = cdf[cdf.AssignmentType == assignment_type]
    fdf = sdf.merge(scdf,on=['Date','Week','JudgeName'])
    fdf['Circuit_y'] = fdf.Circuit_y.replace({'na':np.nan})
    fdf['Circuit_y'] = fdf.Circuit_y.astype('float').astype('Int64')
    fdf.loc[fdf.County_x != fdf.County_y,'MatchType'] = 'Disagree'
    fdf.loc[fdf.Circuit_x == fdf.Circuit_y,'MatchType'] = 'Circuit'
    fdf.loc[fdf.County_x == fdf.County_y,'MatchType'] = 'Agree'
    fdf.sort_values('MatchType',inplace=True)
    fdf = fdf.drop_duplicates('EventID',keep='first')

    fdf['Weekday'] = fdf.Date.apply(get_weekday)
    counts = fdf.groupby(['JudgeID','County_x','StartDate','Weekday','FullAssignment','MatchType'])[['Plea','Trial']].sum().reset_index()
    counts['County'] = counts.apply(add_sentence_counts,axis=1)
    counts = add_conflict_color(counts)
    counts['Judge'] = counts.JudgeID.str.replace('Judge ','',regex=False).astype(int)
    conflict_weeks = counts.loc[counts.MatchType.isin(['Disagree','Circuit']),['Judge','StartDate']].drop_duplicates()
    df = counts.pivot_table(index=['Judge','StartDate','FullAssignment'],columns='Weekday',values='County',aggfunc= lambda x: ', '.join(x))
    df = pd.DataFrame(df.to_records())
    cols = ['Judge','StartDate','FullAssignment']+weekdays[:5]

    df.sort_values(['Judge','StartDate'],inplace=True)
    df = df.merge(conflict_weeks,on=['Judge','StartDate'])
    return df[cols]

def get_weekday(date_str):
    date = datetime.datetime.strptime(date_str,'%Y-%m-%d')
    day_num = str(date.isoweekday())
    day_name = day_names[day_num]
    return day_name

def add_conflict_color(df):
    df.loc[df.MatchType == 'Disagree','County'] = df.loc[df.MatchType == 'Disagree','County'].apply(add_color,args=('red',))
    df.loc[df.MatchType == 'Circuit','County'] = df.loc[df.MatchType == 'Circuit','County'].apply(add_color,args=('blue',))
    df.loc[df.MatchType == 'Agree','County'] = df.loc[df.MatchType == 'Agree','County'].apply(add_color,args=('green',))
    return df

def add_color(county,color):
    # return  "\textcolor{{{0}}}{{{1}}}".format(color,county)
    return  "\{{{0}}}{{{1}}}".format(color,county)

def add_sentence_counts(row):
    return '{} ({},{})'.format(row['County_x'],row['Plea'],row['Trial'])
