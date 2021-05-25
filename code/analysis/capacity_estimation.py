import pandas as pd
import numpy as np
import os
import datetime
import re
import pickle

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")


# Input files/dirs
processed_daily_schedule_file = PROJECT_DIR + '/data/processed/daily_schedule_data.csv'
processed_sentencing_data_file = PROJECT_DIR + '/data/raw/sentencing_data.csv'
judge_name_id_mapping_file = PROJECT_DIR + '/data/processed/judge_name_id_mapping.csv'

# Output files
processed_sentencing_data_file = PROJECT_DIR + '/data/processed/sentencing_data.csv'

def load_sentencing_data():
    sdf = pd.read_csv(processed_sentencing_data_file)
    sdf = sdf.loc[sdf.Date.notna(),:]
    sdf['Plea'] = (sdf['Trial']+1)%2
    return sdf

def load_calendar_data():
    cdf = pd.read_csv(processed_daily_schedule_file)
    mapping = pd.read_csv(judge_name_id_mapping_file)
    cdf = cdf.merge(mapping,on='JudgeName')
    return cdf

def get_judge_county_event_counts():
    sdf = load_sentencing_data()
    sdf = restrict_sentencing_sample(sdf)
    counts = sdf.groupby(['JudgeName','JudgeID','County'])[['Plea','Trial']].sum().reset_index()
    return counts

def restrict_sentencing_sample(sdf):
    daily_sentences = sdf.groupby(['Date','JudgeName','County']).size().reset_index(name="N")
    outlier_jcs = daily_sentences.loc[daily_sentences['N'] >= 35,['JudgeName','County']].drop_duplicates()
    outlier_jcs['Outlier'] = True
    trial_jcs = sdf.loc[sdf.Trial == 1,['JudgeName','County']].drop_duplicates()

    sample = trial_jcs.merge(outlier_jcs,on=['JudgeName','County'],how='left')
    sample = sample.loc[~(sample.Outlier == True),['JudgeName','County']].drop_duplicates()

    sdf = sdf.merge(sample,on=['JudgeName','County'])
    return sdf

def get_day_assignments():
    cdf = load_calendar_data()
    day_weights = cdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    day_weights['Days'] = 1/day_weights['N']
    cdf = cdf.merge(day_weights,on=['JudgeName','Date'])
    assigned_days = cdf.groupby(['JudgeName','County'])['Days'].sum().reset_index()
    return assigned_days

def make_full_data():
    plea_service_rate = 10.7
    counts = get_judge_county_event_counts()
    assigned_days = get_day_assignments()
    df = counts.merge(assigned_days,on=['JudgeName','County'])
    df['PleaDays'] = df['Plea']/plea_service_rate
    df = df[df.Trial >=2]
    trial_service_rate = df.Trial.sum()/((df.Days.sum())-df.PleaDays.sum())

def discrepancy_analysis():
    plea_service_rate = 10.7
    counts = get_judge_county_event_counts()
    assigned_days = get_day_assignments()
    df = counts.merge(assigned_days,on=['JudgeName','County'])
    df['PleaDays'] = df['Plea']/plea_service_rate
    df = df[df.Trial >=2]

    ndf = pd.read_csv('nasser_judge_county_df.csv')
    ndf['JudgeID'] = ndf.JudgeID.apply(lambda x: 'Judge '+str(x-1))

    tst = df.merge(ndf,on=['JudgeID','County'])
    tst['DayDiff'] = tst['Days_x']-tst['Days_y']
    tst['PleaDiff'] = tst['Plea'] - tst['Pleas']
    tst['TrialDiff'] = tst['Trial'] - tst['Trials']

    tst.DayDiff.mean()
    tst.PleaDiff.mean()
    tst.TrialDiff.mean()

dicts = np.load('Sentencing_Data_Judge_Date_Location_Counts.npy',allow_pickle=True)
plea_dict = dicts[2]
pdf = []
for judge_county_day, count in plea_dict.items():
    id, date, county = judge_county_day
    id -= 1
    id = 'Judge ' + str(id)
    pdf.append({'JudgeID':id,'Date':date,'County':county,'Pleas':count})

pdf = pd.DataFrame(pdf)
pdf = pdf.groupby(['JudgeID','County'])['Pleas'].sum().reset_index()
ptst = df.merge(pdf,on=['JudgeID','County'])

trial_dict = dicts[3]
tdf = []
for judge_county_day, count in trial_dict.items():
    id, date, county = judge_county_day
    id -= 1
    id = 'Judge ' + str(id)
    tdf.append({'JudgeID':id,'Date':date,'County':county,'Trials':count})

tdf = pd.DataFrame(tdf)
tdf = tdf.groupby(['JudgeID','County'])['Trials'].sum().reset_index()
ttst = df.merge(tdf,on=['JudgeID','County'])


js = pickle.load(open('judge_county_dates.pkl','rb'))
ddf = []
for judge_county, datelist in js.items():
    judge, county = judge_county
    count = len(set(datelist))
    ddf.append({'JudgeName':judge,'County':county,'N':count})

ddf = pd.DataFrame(ddf)
ddf.loc[ddf.JudgeName == 'COOPER, TW','JudgeName'] = 'COOPER TW'
ddf.loc[ddf.JudgeName == 'FLOYD, H.','JudgeName'] = 'FLOYD H'
ddf.loc[ddf.JudgeName == 'FLOYD, S.','JudgeName'] = 'FLOYD S'
ctst = df.merge(ddf,on=['JudgeName','County'])
