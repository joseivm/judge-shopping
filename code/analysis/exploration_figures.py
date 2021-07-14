import pandas as pd
import numpy as np
import os
import datetime
import re
import pickle
import math
import matplotlib.pyplot as plt

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
processed_daily_schedule_file = PROJECT_DIR + '/data/processed/daily_schedule_data.csv'
processed_sentencing_data_file = PROJECT_DIR + '/data/raw/sentencing_data.csv'
judge_name_id_mapping_file = PROJECT_DIR + '/data/processed/judge_name_id_mapping.csv'
county_file = PROJECT_DIR + '/data/raw/county_list.csv'

# Output files
optimization_data_folder = PROJECT_DIR + '/data/optimization/'

holidays = ['2000-09-04','2000-10-09','2000-11-10','2000-11-23','2000-12-24',
'2000-12-25','2000-12-26','2001-01-01','2001-01-15','2001-02-19','2001-05-10',
'2001-05-28','2001-07-04','2000-07-04']

def load_sentencing_data():
    sdf = pd.read_csv(processed_sentencing_data_file)
    return sdf

def load_calendar_data():
    cdf = pd.read_csv(processed_daily_schedule_file)
    mapping = pd.read_csv(judge_name_id_mapping_file)
    cdf = cdf.merge(mapping,on='JudgeName')
    cdf = cdf.loc[~cdf.Date.isin(holidays),:]
    return cdf

def fraction_gs_by_judge_hist():
    cdf = load_calendar_data()
    day_weights = cdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    day_weights['Total'] = 1/day_weights['N']
    cdf = cdf.merge(day_weights,on=['JudgeName','Date'])
    cdf['JudgeID'] = cdf.JudgeID.str.slice(start=6)
    cdf['JudgeID'] = cdf.JudgeID.astype(int)
    cdf['GS'] = cdf.WorkType == 'GS'
    # cdf['GS'] = cdf['WorkType'].str.contains('GS',regex=False)
    # cdf['GS'] = cdf.GS.replace({True:1,False:0,np.nan:0})

    days = cdf.groupby('JudgeID')[['GS','Total']].sum().reset_index()
    days['Fraction'] = days['GS']/days['Total']
    days.sort_values('JudgeID',inplace=True)
    plt.figure(figsize=(9,5))
    plt.bar(days.JudgeID,days.Fraction)
    plt.ylabel('Fraction GS')
    plt.grid(axis='y')
    plt.title('Fraction of GS Assignments by Judge')
    filename = PROJECT_DIR + '/output/figures/Exploration/fraction_gs_by_judge.png'
    plt.savefig(filename, dpi=120)

def fraction_missing_by_group_hist(group):
    sdf = pd.read_csv(processed_sentencing_data_file)
    sdf['MissingDate'] = sdf.Date.isna()
    sdf['Total'] = 1
    sdf['JudgeID'] = sdf.JudgeID.str.slice(start=6)
    sdf['JudgeID'] = sdf.JudgeID.astype(int)
    events = sdf.groupby(group)[['MissingDate','Total']].sum().reset_index()
    events['FractionMissing'] = events['MissingDate']/events['Total']
    events.sort_values(group,inplace=True)

    plt.figure(figsize=(9,5))
    plt.bar(events[group],events.FractionMissing)
    if group == 'County': plt.xticks(rotation=90)
    plt.ylabel('Fraction Missing Date')
    plt.grid(axis='y')
    plt.title('Fraction of Sentencing Events Missing Date by {}'.format(group))
    filename = PROJECT_DIR + '/output/figures/Exploration/fraction_missing_date_by_{}.png'.format(group)
    plt.savefig(filename)

def count_missing_by_group_hist(group):
    sdf = pd.read_csv(processed_sentencing_data_file)
    sdf['MissingDate'] = sdf.Date.isna()
    sdf['Total'] = 1
    sdf['JudgeID'] = sdf.JudgeID.str.slice(start=6)
    sdf['JudgeID'] = sdf.JudgeID.astype(int)
    events = sdf.groupby(group)[['MissingDate','Total']].sum().reset_index()
    events.sort_values(group,inplace=True)

    plt.figure(figsize=(9,5))
    plt.bar(events[group],events.MissingDate)
    if group == 'County': plt.xticks(rotation=90)
    plt.ylabel('Events Missing Date')
    plt.grid(axis='y')
    plt.title('Number of Sentencing Events Missing Date by {}'.format(group))
    filename = PROJECT_DIR + '/output/figures/Exploration/count_missing_date_by_{}_hist.png'.format(group)
    plt.savefig(filename)

def recovered_dates_by_month_hist():
    cdf = load_calendar_data()
    sdf = load_sentencing_data()
    mdf = sdf.loc[sdf.Date.isna(),['JudgeID','County','EventID']]
    cdf = cdf.drop_duplicates(['JudgeID','County','Week'])

    mdf = mdf.merge(cdf,on=['JudgeID','County'])
    mdf['Date'] = mdf['Date'].astype('datetime64')
    mdf = mdf.drop_duplicates('EventID',keep=False)

    mdf.groupby(mdf['Date'].dt.month).size().plot(kind='bar')
    plt.ylabel('N')
    plt.xlabel('Month')
    plt.grid(True)
    plt.title('Number of events with missing dates by month')
    filename = PROJECT_DIR + '/output/figures/Exploration/missing_date_month_hist.png'
    plt.savefig(filename)

def average_pleas_by_work_type():
    sdf = load_sentencing_data()
    sdf = sdf.loc[sdf.Date.notna(),:]
    counts = sdf.groupby(['JudgeID','Date'])['Plea'].sum().reset_index()
    counts = counts.reset_index().rename(columns={'index':'DayCountyID'})
    cdf = load_calendar_data()
    day_weights = cdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    day_weights['Days'] = 1/day_weights['N']
    cdf = cdf.merge(day_weights,on=['JudgeName','Date'])

    fdf = pd.merge(counts,cdf,on=['Date','JudgeID'],how='right')
    fdf.loc[fdf.Plea.isna(),'Plea'] = 0
    fdf['WeightedPlea'] = fdf['Plea']*fdf['Days']

    bad_worktypes = ['Orientation School','Medical','Family Death','Sick','Military']
    counts = fdf.groupby('WorkType')[['WeightedPlea','Days']].sum().reset_index()
    counts['PleasPerDay'] = counts['WeightedPlea']/counts['Days']
    counts = counts.loc[~counts.WorkType.isin(bad_worktypes),:]

    plt.figure(figsize=(9,7))
    plt.bar(counts.WorkType,counts.PleasPerDay)
    plt.ylabel('Average Daily Pleas')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.title('Average Daily Pleas by Work Type')
    filename = PROJECT_DIR + '/output/figures/Exploration/avg_pleas_by_worktype.png'
    plt.savefig(filename)

def average_pleas_by_work_type_cond():
    # consider integrating day weights into mean calculation
    sdf = load_sentencing_data()
    sdf = sdf.loc[sdf.Date.notna(),:]
    cdf = load_calendar_data()

    fdf = sdf.merge(cdf,on=['Date','JudgeID'])
    fdf['Circuit_y'] = fdf.Circuit_y.replace({'na':np.nan})
    fdf['Circuit_y'] = fdf.Circuit_y.astype('float').astype('Int64')
    fdf.loc[fdf.County_x != fdf.County_y,'MatchType'] = 'Disagree'
    fdf.loc[fdf.Circuit_x == fdf.Circuit_y,'MatchType'] = 'Circuit'
    fdf.loc[fdf.County_x == fdf.County_y,'MatchType'] = 'Agree'
    fdf.sort_values('MatchType',inplace=True)
    fdf = fdf.drop_duplicates('EventID',keep='first')

    fdf.loc[fdf.MatchType == 'Disagree','WorkType'] = 'Disagree'
    counts = fdf.groupby(['JudgeID','Date','WorkType'])['Plea'].sum().reset_index()
    means = counts.groupby('WorkType')['Plea'].mean().reset_index()

    plt.figure(figsize=(8,7))
    plt.bar(means.WorkType,means.Plea)
    plt.ylabel('Average Daily Pleas')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.title('Average Daily Pleas by Work Type, Cond.')
    filename = PROJECT_DIR + '/output/figures/Exploration/avg_pleas_by_worktype_cond.png'
    plt.savefig(filename)

def daily_plea_hist_by_work_type():
    sdf = load_sentencing_data()
    sdf = sdf.loc[sdf.Date.notna(),:]
    cdf = load_calendar_data()

    fdf = sdf.merge(cdf,on=['Date','JudgeID'])
    fdf['Circuit_y'] = fdf.Circuit_y.replace({'na':np.nan})
    fdf['Circuit_y'] = fdf.Circuit_y.astype('float').astype('Int64')
    fdf.loc[fdf.County_x != fdf.County_y,'MatchType'] = 'Disagree'
    fdf.loc[fdf.Circuit_x == fdf.Circuit_y,'MatchType'] = 'Circuit'
    fdf.loc[fdf.County_x == fdf.County_y,'MatchType'] = 'Agree'
    fdf.sort_values('MatchType',inplace=True)
    fdf = fdf.drop_duplicates('EventID',keep='first')

    fdf.loc[fdf.MatchType == 'Disagree','WorkType'] = 'Disagree'
    counts = fdf.groupby(['JudgeID','Date','WorkType'])[['Plea','Trial']].sum().reset_index()
    work_types = counts.WorkType.unique()

    fig, axes = plt.subplots(4,3,figsize=(10,10))
    for work_type, ax in zip(work_types,axes.ravel()):
        work_type_counts = counts.loc[counts.WorkType == work_type,:]
        ax.hist(work_type_counts.Plea,bins=30)
        ax.set_title(work_type)

    plt.tight_layout()
    filename = PROJECT_DIR + '/output/figures/Exploration/daily_plea_hist_by_work_type.png'
    plt.savefig(filename)

def county_trial_histograms():
    sdf = load_sentencing_data()
    cdf = load_calendar_data()
    home_circuits = sdf.groupby(['JudgeID','HomeCircuit']).size().reset_index(name='N')
    home_circuits['HomeCircuit'] = home_circuits.HomeCircuit.astype('Int64').astype(str)

    cdf = cdf.merge(home_circuits[['JudgeID','HomeCircuit']],on='JudgeID',how='left')
    counties = pd.read_csv(county_file)

    home_counties = cdf.loc[(cdf.HomeCircuit == cdf.Circuit)&
                            (cdf.County.isin(counties.County)),:].groupby(['JudgeID','County']).size(
                            ).reset_index(name='N').sort_values('N',ascending=False).groupby('JudgeID').head(1)
    home_counties.rename(columns={'County':'HomeCounty'},inplace=True)
    home_counties.drop(columns=['N'],inplace=True)
    sdf = sdf.merge(home_counties,on='JudgeID',how='left')
    sdf['HomeJudge'] = 'Non-Resident'
    sdf.loc[sdf.HomeCounty == sdf.County,'HomeJudge'] = 'Resident'
    trial_counts = sdf.loc[sdf.Trial == 1,:].groupby(['County','HomeJudge']).size().reset_index(name='N')

    counties = counties.County.unique()
    county_groups = np.array_split(counties,np.arange(12,len(counties),12))
    i = 0
    for group in county_groups:
        fig, axes = plt.subplots(6,2,figsize=(8,8))
        for county, ax in zip(group,axes.ravel()):
            county_trials = trial_counts.loc[trial_counts.County == county,:]
            ax.bar(county_trials.HomeJudge,county_trials.N,color=['b','r'])
            ax.set_title(county)
            ax.grid(axis='y')

        plt.tight_layout()
        filename = PROJECT_DIR + '/output/figures/Exploration/county_trial_hist_{}.png'.format(i)
        plt.savefig(filename)
        i += 1

def main():
    average_pleas_by_work_type()
    average_pleas_by_work_type_cond()
    fraction_gs_by_judge_hist()
    recovered_dates_by_month_hist()
    county_trial_histograms()
    for group in ['County','JudgeID']:
        fraction_missing_by_group_hist(group)
        count_missing_by_group_hist(group)

main()
