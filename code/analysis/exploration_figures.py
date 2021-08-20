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
    # sdf = add_work_type(sdf)
    return sdf

def add_work_type(sdf):
    cdf = load_calendar_data()
    cdf_cols = ['JudgeID','Date','County','WorkType']
    sdf = sdf.merge(cdf[cdf_cols],on=['JudgeID','Date','County'],how='left')
    sdf.loc[sdf.WorkType.isna(),'WorkType'] = 'Disagree'
    return sdf

def load_calendar_data():
    cdf = pd.read_csv(processed_daily_schedule_file)
    mapping = pd.read_csv(judge_name_id_mapping_file)
    cdf = cdf.merge(mapping,on='JudgeName')
    cdf = cdf.loc[~cdf.Date.isin(holidays),:]
    return cdf

def get_clean_day_counts():
    sdf = load_sentencing_data()
    sdf = remove_conflicting_days(sdf)
    sdf = remove_non_GS_days(sdf)
    sdf = remove_multi_county_days(sdf)
    sdf = remove_multi_assignment_days(sdf)
    sdf = remove_judges_with_few_clean_days(sdf)
    counts = sdf.groupby(['JudgeID','County','Date'])[['Plea']].sum().reset_index()
    return counts

def remove_conflicting_days(sdf):
    cdf = load_calendar_data()
    fdf = sdf.merge(cdf,on=['Date','JudgeName'])
    fdf['Circuit_y'] = fdf.Circuit_y.replace({'na':np.nan})
    fdf['Circuit_y'] = fdf.Circuit_y.astype('float').astype('Int64')
    fdf.loc[fdf.County_x != fdf.County_y,'MatchType'] = 'Disagree'
    fdf.loc[fdf.Circuit_x == fdf.Circuit_y,'MatchType'] = 'Circuit'
    fdf.loc[fdf.County_x == fdf.County_y,'MatchType'] = 'Agree'
    fdf.sort_values('MatchType',inplace=True)
    fdf = fdf.drop_duplicates('EventID',keep='first')
    fdf = fdf.loc[fdf.MatchType == 'Disagree',['Date','JudgeName','MatchType']].drop_duplicates(['Date','JudgeName'])
    sdf = sdf.merge(fdf,on=['Date','JudgeName'],how='left')
    sdf = sdf.loc[sdf.MatchType.isna(),:]
    return sdf

def remove_non_GS_days(sdf):
    cdf = load_calendar_data()
    gs_days = cdf.loc[cdf.WorkType == 'GS',['JudgeName','Date']]
    sdf = sdf.merge(gs_days,on=['JudgeName','Date'])
    return sdf

def remove_multi_county_days(sdf):
    judge_events = sdf.groupby(['JudgeID','Date','County']).size().reset_index(name='N')
    counties_by_day = judge_events.groupby(['JudgeID','Date']).size().reset_index(name='N')
    single_county_days = counties_by_day.loc[counties_by_day.N == 1,['JudgeID','Date']]
    sdf = sdf.merge(single_county_days,on=['JudgeID','Date'])
    return sdf

def remove_multi_assignment_days(sdf):
    cdf = load_calendar_data()
    day_assignments = cdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    single_assignment_days = day_assignments.loc[day_assignments.N == 1,['JudgeName','Date']]
    sdf = sdf.merge(single_assignment_days,on=['JudgeName','Date'])
    return sdf

def remove_judges_with_few_clean_days(sdf,threshold=10):
    judge_clean_days = sdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    good_judges = judge_clean_days.loc[judge_clean_days.N >= threshold,['JudgeName','Date']]
    sdf = sdf.merge(good_judges,on=['JudgeName','Date'])
    return sdf

##### GS Exploration #####
def fraction_gs_by_group_hist(group):
    cdf = load_calendar_data()
    cdf['JudgeID'] = cdf.JudgeID.str.slice(start=6)
    cdf['JudgeID'] = cdf.JudgeID.astype(int)
    cdf['GS'] = cdf.WorkType == 'GS'
    cdf['GS'] = cdf['GS']*cdf['Days']

    days = cdf.groupby(group)[['GS','Days']].sum().reset_index()
    days['Fraction'] = days['GS']/days['Days']
    days.sort_values(group,inplace=True)
    plt.figure(figsize=(9,5))
    if group == 'County':
        plt.xticks(rotation=90)
        days = days.loc[~days.County.str.contains('Cir'),:]
    plt.bar(days[group],days.Fraction)

    plt.ylabel('Fraction GS')
    plt.grid(axis='y')
    plt.title('Fraction of GS Assignments by {}'.format(group))
    filename = PROJECT_DIR + '/output/figures/Exploration/fraction_gs_by_{}.png'.format(group)
    plt.savefig(filename, dpi=120)

def fraction_gs_clean_by_judge_hist():
    cdf = load_calendar_data()
    sdf = get_clean_day_counts()
    cdf = cdf.merge(sdf,on=['JudgeID','County','Date'],how='left')
    cdf['JudgeID'] = cdf.JudgeID.str.slice(start=6)
    cdf['JudgeID'] = cdf.JudgeID.astype(int)
    cdf['GS'] = cdf.WorkType == 'GS'
    cdf['GS'] = cdf['GS']*cdf['Days']
    cdf['Clean'] = cdf.Plea.notna()
    cdf['Clean'] = cdf['Clean']*cdf['Days']

    days = cdf.groupby('JudgeID')[['GS','Clean','Days']].sum().reset_index()
    days['FractionGS'] = days['GS']/days['Days']
    days['FractionClean'] = days['Clean']/days['Days']
    days.sort_values('JudgeID',inplace=True)
    fig, axes = plt.subplots(2,1,figsize=(8,8))
    axes[0].bar(days['JudgeID'],days.FractionGS)
    axes[0].set_title('Fraction GS Assignments by Judge')
    axes[0].grid(axis='y')


    axes[1].bar(days.JudgeID,days.FractionClean)
    axes[1].set_title('Fraction Clean Days by Judge')
    axes[1].grid(axis='y')
    axes[1].tick_params(labelrotation=90)

    plt.tight_layout()
    filename = PROJECT_DIR + '/output/figures/Exploration/fraction_gs_clean.png'
    plt.savefig(filename, dpi=120)
    plt.close('all')

def average_pleas_by_work_type():
    sdf = load_sentencing_data()
    sdf = sdf.loc[sdf.Date.notna(),:]
    counts = sdf.groupby(['JudgeID','Date'])['Plea'].sum().reset_index()
    counts = counts.reset_index().rename(columns={'index':'DayCountyID'})
    cdf = load_calendar_data()

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

def work_type_summary_stats():
    sdf = load_sentencing_data()
    sdf = sdf.loc[sdf.Date.notna(),:]
    counts = sdf.groupby(['JudgeID','Date'])[['Plea','Trial']].any().reset_index()
    counts = counts.reset_index().rename(columns={'index':'DayCountyID'})
    cdf = load_calendar_data()

    fdf = pd.merge(counts,cdf,on=['JudgeID','Date'],how='right')
    fdf.loc[fdf.Plea.isna(),'Plea'] = 0
    fdf.loc[fdf.Trial.isna(),'Trial'] = 0
    fdf['Plea'] = fdf.Plea.astype(int)
    fdf['Trial'] = fdf.Trial.astype(int)
    fdf['WeightedPlea'] = fdf['Plea']*fdf['Days']
    fdf['WeightedTrial'] = fdf['Trial']*fdf['Days']
    fdf['Empty'] = (fdf['Plea'] + fdf['Trial'] == 0).astype(int)
    fdf['WeightedEmpty'] = fdf['Empty']*fdf['Days']

    bad_worktypes = ['Orientation School','Medical','Family Death','Sick','Military']
    counts = fdf.groupby('WorkType')[['WeightedPlea','WeightedTrial','WeightedEmpty','Days']].sum().reset_index()
    counts['SharePlea'] = counts['WeightedPlea']/counts['Days']
    counts['ShareTrial'] = counts['WeightedTrial']/counts['Days']
    counts['ShareEmpty'] = counts['WeightedEmpty']/counts['Days']
    counts['Share'] = counts['Days']/counts.Days.sum()
    counts = counts.loc[~counts.WorkType.isin(bad_worktypes),['WorkType','Days','Share','SharePlea','ShareTrial','ShareEmpty']]

    filename = PROJECT_DIR + '/output/tables/Exploration/work_type_summary_stats.csv'
    counts.to_csv(filename,float_format="{:,.2f}".format,index=False)

def work_type_distribution():
    cdf = load_calendar_data()
    day_counts = cdf.groupby('WorkType')['Days'].sum().reset_index()
    day_counts['Share'] = day_counts['Days']/day_counts.Days.sum()
    day_counts.sort_values('Share',ascending=False,inplace=True)

    # plt.figure()
    # plt.bar(day_counts.WorkType,day_counts.Share)
    # plt.grid(axis='y')
    # plt.xticks(rotation=90)
    # plt.show()
    filename = PROJECT_DIR + '/output/tables/Exploration/work_type_dist.csv'
    counts.to_csv(filename,float_format="{:,.2f}".format,index=False)

def daily_plea_hist_by_broad_work_type(sdf,county):
    # sdf = load_sentencing_data()
    # sdf = sdf.loc[sdf.Date.notna(),:]
    cdf = load_calendar_data()

    fdf = sdf.merge(cdf,on=['Date','JudgeID'])
    fdf['Circuit_y'] = fdf.Circuit_y.replace({'na':np.nan})
    fdf['Circuit_y'] = fdf.Circuit_y.astype('float').astype('Int64')
    fdf.loc[fdf.County_x != fdf.County_y,'MatchType'] = 'Disagree'
    fdf.loc[fdf.Circuit_x == fdf.Circuit_y,'MatchType'] = 'Circuit'
    fdf.loc[fdf.County_x == fdf.County_y,'MatchType'] = 'Agree'
    fdf.sort_values('MatchType',inplace=True)
    fdf = fdf.drop_duplicates('EventID',keep='first')

    fdf.loc[fdf.MatchType != 'Agree','WorkType'] = 'Disagree'
    fig, axes = plt.subplots(3,1,figsize=(8,10))
    gs_counts = fdf.loc[(fdf.WorkType == 'GS') & (fdf.County_x == county),:].groupby(['JudgeID','Date'])['Plea'].sum().reset_index()
    plea_mean = gs_counts.Plea.mean()
    axes[1].hist(gs_counts.Plea,bins=50)
    if not np.isnan(plea_mean):
        axes[1].axvline(x=plea_mean,color='r')
        x_ticks = np.append(axes[1].get_xticks(),plea_mean)
        axes[1].set_xticks(x_ticks)
    axes[1].set_title('GS Days')
    axes[1].tick_params(labelrotation=90)

    clean_counts = get_clean_day_counts()
    clean_counts = clean_counts.loc[clean_counts.County == county,:]
    plea_mean = clean_counts.Plea.mean()
    axes[2].hist(clean_counts.Plea,bins=50)
    if not np.isnan(plea_mean):
        axes[2].axvline(x=plea_mean,color='r')
        x_ticks = np.append(axes[2].get_xticks(),plea_mean)
        axes[2].set_xticks(x_ticks)
    axes[2].set_title('Clean Days')
    axes[2].tick_params(labelrotation=90)

    non_gs_counts = fdf.loc[(fdf.WorkType != 'GS')& (fdf.County_x == county),:].groupby(['JudgeID','Date'])['Plea'].sum().reset_index()
    plea_mean = non_gs_counts.Plea.mean()
    axes[0].hist(non_gs_counts.Plea,bins=50)
    if not np.isnan(plea_mean):
        axes[0].axvline(x=plea_mean,color='r')
        x_ticks = np.append(axes[0].get_xticks(),plea_mean)
        axes[0].set_xticks(x_ticks)
    axes[0].set_title('Non-GS Days')
    axes[0].tick_params(labelrotation=90)

    plt.tight_layout()
    filename = PROJECT_DIR + '/output/figures/Exploration/County/daily_plea_hist_by_broad_work_type_{}.png'.format(county)
    plt.savefig(filename)
    plt.close('all')

##### Other Stuff #####
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

    cdf = cdf.merge(home_counties,on='JudgeID',how='left')
    sdf = sdf.merge(home_counties,on='JudgeID',how='left')
    sdf['Residence'] = 'Non-Resident'
    sdf.loc[sdf.HomeCircuit == sdf.Circuit,'Residence'] = 'Circuit'
    sdf.loc[sdf.HomeCounty == sdf.County,'Residence'] = 'County'
    trial_counts = sdf.loc[sdf.Trial == 1,:].groupby(['County','Residence']).size().reset_index(name='N')

    counties = counties.County.unique()
    county_groups = np.array_split(counties,np.arange(12,len(counties),12))
    sns.set_style('whitegrid')
    i = 0
    for group in county_groups:
        fig, axes = plt.subplots(6,2,figsize=(8,8))
        for county, ax in zip(group,axes.ravel()):
            county_trials = trial_counts.loc[trial_counts.County == county,:]
            if county_trials.empty: continue
            g = sns.barplot(ax=ax,x='Residence',y='N',hue='Residence',
                palette={'Non-Resident':'g','Circuit':'b','County':'r'},
                data=county_trials,dodge=False)
            g.legend_.remove()
            ax.set_title(county)

        plt.tight_layout()
        filename = PROJECT_DIR + '/output/figures/Exploration/county_trial_hist_{}.png'.format(i)
        plt.savefig(filename)
        i += 1

def travel_probability_table():
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

    cdf = cdf.merge(home_counties,on='JudgeID',how='left')
    cdf['Travel'] = 'Non-Circuit'
    cdf.loc[cdf.Circuit == cdf.HomeCircuit, 'Travel'] = 'Circuit'
    cdf.loc[cdf.County == cdf.HomeCounty, 'Travel'] = 'County'
    cdf.loc[cdf.WorkType == 'in chambers', 'Travel'] = 'County'

    counts = cdf.groupby(['JudgeID','Travel']).size().reset_index(name='N')
    counts['Share'] = counts.groupby('JudgeID')['N'].apply(lambda x: x/x.sum())

    counts = counts.pivot(index='JudgeID',columns='Travel',values='Share')
    counts = pd.DataFrame(counts.to_records())
    counts = counts.loc[:,['JudgeID','County','Circuit','Non-Circuit']]

    filename = PROJECT_DIR + '/output/tables/Exploration/travel_probability.csv'
    counts.to_csv(filename,float_format="{:,.2f}".format,index=False)

def lead_up_investigation(nonGS=False,window=2):
    sdf = load_sentencing_data()
    sdf = sdf.loc[sdf.Date.notna(),:]
    counts = sdf.groupby(['JudgeID','Date'])[['Plea','Trial']].sum().reset_index()
    cdf = load_calendar_data()

    fdf = pd.merge(counts,cdf,on=['JudgeID','Date'],how='right')
    fdf.loc[fdf.Plea.isna(),'Plea'] = 0
    fdf.loc[fdf.Trial.isna(),'Trial'] = 0
    fdf['Plea'] = fdf.Plea.astype(int)
    fdf['Trial'] = fdf.Trial.astype(int)
    fdf['Empty'] = (fdf['Plea'] + fdf['Trial'] == 0).astype(int)

    if nonGS:
        trials = fdf.loc[(fdf.Trial == 1) & (fdf.WorkType != 'GS'),['JudgeID','County','Week']]
        kind = 'NonGS'
    else:
        trials = fdf.loc[fdf.Trial == 1,['JudgeID','County','Week']]
        kind = 'All'
    trials = trials.reset_index().rename(columns={'index':'TrialID'})
    preceding_dates = []
    weeks = [str(i) for i in range(1,53)]
    for index, row in trials.iterrows():
        week_num, week_year = row['Week'].split('-')
        week_idx = int(week_num)-1
        preceding_week_idx = [week_idx - i for i in range(1,window+1)]
        year_adjustments = [idx < 0 for idx in preceding_week_idx]
        years = [int(week_year)- 1*adj for adj in year_adjustments]
        week_nums = [weeks[i] for i in preceding_week_idx]
        new_weeks = [week_num+'-'+str(year) for week_num, year in zip(week_nums,years)]
        new_dates = [{'JudgeID':row['JudgeID'],'Week':week,'Preceding':True,
        'TrialID':row['TrialID'],'County':row['County']} for week in new_weeks]
        preceding_dates += new_dates

    pdf = pd.DataFrame(preceding_dates)

    fdf = fdf.merge(pdf,on=['JudgeID','Week','County'],how='left')
    empty_days = fdf.loc[(fdf.Empty == True) & (fdf.Preceding == True),:]
    plt.figure()
    empty_days.groupby('TrialID').size().plot(kind='hist')
    filename = PROJECT_DIR + '/output/figures/Exploration/empty_days_hist_{}_{}.png'.format(kind,window)
    plt.savefig(filename)

    plt.figure()
    empty_days.groupby('WorkType').size().reset_index(name='N').sort_values('N',ascending=False).plot(kind='bar',x='WorkType',y='N')
    plt.xticks(rotation=45)
    filename = PROJECT_DIR + '/output/figures/Exploration/empty_days_work_type_{}_{}.png'.format(kind,window)
    plt.savefig(filename)
    plt.close('all')

def sentencing_events_by_work_type():
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
    counts = fdf.groupby(['WorkType'])[['Plea','Trial']].sum().reset_index()
    counts['Plea Share'] = counts['Plea']/counts.Plea.sum()
    counts['Trial Share'] = counts['Trial']/counts.Trial.sum()

    counts = counts.loc[:,['WorkType','Plea Share','Trial Share']]

    filename = PROJECT_DIR + '/output/tables/Exploration/events_by_work_type.csv'
    counts.to_csv(filename,float_format="{:,.2f}".format,index=False)


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
